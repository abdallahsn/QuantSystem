#!/usr/bin/env python3
"""
Threshold exploration for day-trade feature rows (OHLC + ATR on a fixed bar grid).

Reads a parquet such as day_trading_features_v19_tensorfix.parquet and produces:
  - ATR distribution and percentiles (overall + by rough FX-futures UTC session bucket)
  - Forward return stats for several bar horizons
  - First-touch TP/SL simulation:
      - race_mode=coupled: classic single-path 4-barrier check (SHORT can be mechanically rare when tp_mult>=sl_mult).
      - race_mode=independent: long-leg TP-vs-SL and short-leg TP-vs-SL are scored separately; earlier winning leg labels the row.

PNG histograms are written only when `--plots` is passed (and matplotlib is available); `--no-plots` forces them off.
  - Sensitivity grid + optional weekly stability slices
  - Approximate soft_label / soft_sample_weight-style diagnostics

This is exploratory: it does not run the refinery. Use actual refinery output to
validate a final (tp_mult, sl_mult, horizon) choice.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Literal

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


TieBreak = Literal["tp_first", "sl_first", "ambiguous_neutral"]
RaceMode = Literal["coupled", "independent"]


def _assign_session_utc(dt: pd.Series) -> pd.Series:
    """Coarse UTC session labels for diagnostics (tune windows to your venue)."""

    hour = dt.dt.hour
    sess = pd.Series("Other", index=dt.index, dtype="object")
    sess.loc[(hour >= 0) & (hour < 7)] = "Asian"
    sess.loc[(hour >= 7) & (hour < 12)] = "London"
    sess.loc[(hour >= 12) & (hour < 16)] = "Overlap"
    sess.loc[(hour >= 16) & (hour < 21)] = "New_York"
    return sess


def _percentile_table(s: pd.Series) -> pd.Series:
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    out = s.quantile(qs)
    out.index = [f"p{int(q * 100)}" for q in qs]
    return out


def _resolve_intrabar_outcome(
    hi: float,
    lo: float,
    *,
    tp_long: float,
    sl_long: float,
    tp_short: float,
    sl_short: float,
    tie_break: TieBreak,
) -> str | None:
    """
    Sequential barrier checks on OHLC extremes for one bar.

    tp_first matches the common naive snippet:
      LONG TP via high, LONG SL via low, SHORT TP via low, SHORT SL via high.

    sl_first swaps TP vs SL ordering for both directions (more conservative on directional counts).

    ambiguous_neutral: if more than one distinct outcome flag fires within the same bar, treat as SL/neutral tie.
    """

    long_tp = hi >= tp_long
    long_sl = lo <= sl_long
    short_tp = lo <= tp_short
    short_sl = hi >= sl_short

    if tie_break == "tp_first":
        if long_tp:
            return "LONG"
        if long_sl:
            return "NEUTRAL_SL"
        if short_tp:
            return "SHORT"
        if short_sl:
            return "NEUTRAL_SL"
        return None

    if tie_break == "sl_first":
        if long_sl:
            return "NEUTRAL_SL"
        if long_tp:
            return "LONG"
        if short_sl:
            return "NEUTRAL_SL"
        if short_tp:
            return "SHORT"
        return None

    flags: list[str] = []
    if long_tp:
        flags.append("LONG")
    if long_sl:
        flags.append("NEUTRAL_SL")
    if short_tp:
        flags.append("SHORT")
    if short_sl:
        flags.append("NEUTRAL_SL")
    if not flags:
        return None
    uniq = set(flags)
    if len(uniq) > 1:
        return "NEUTRAL_SL"
    return flags[0]


def _first_long_event_bar(
    high: np.ndarray,
    low: np.ndarray,
    base_i: int,
    *,
    tp_long: float,
    sl_long: float,
    horizon: int,
    tie_break: TieBreak,
) -> tuple[str, int]:
    """Returns (WIN|LOSE|TIMEOUT, bar_index_within_horizon)."""

    for j in range(1, horizon + 1):
        hi = float(high[base_i + j])
        lo = float(low[base_i + j])
        long_tp_hit = hi >= tp_long
        long_sl_hit = lo <= sl_long
        if tie_break == "tp_first":
            if long_tp_hit:
                return "WIN", j
            if long_sl_hit:
                return "LOSE", j
        elif tie_break == "sl_first":
            if long_sl_hit:
                return "LOSE", j
            if long_tp_hit:
                return "WIN", j
        else:
            if long_tp_hit and long_sl_hit:
                return "LOSE", j
            if long_tp_hit:
                return "WIN", j
            if long_sl_hit:
                return "LOSE", j

    return "TIMEOUT", horizon


def _first_short_event_bar(
    high: np.ndarray,
    low: np.ndarray,
    base_i: int,
    *,
    tp_short: float,
    sl_short: float,
    horizon: int,
    tie_break: TieBreak,
) -> tuple[str, int]:
    for j in range(1, horizon + 1):
        hi = float(high[base_i + j])
        lo = float(low[base_i + j])
        short_tp_hit = lo <= tp_short
        short_sl_hit = hi >= sl_short
        if tie_break == "tp_first":
            if short_tp_hit:
                return "WIN", j
            if short_sl_hit:
                return "LOSE", j
        elif tie_break == "sl_first":
            if short_sl_hit:
                return "LOSE", j
            if short_tp_hit:
                return "WIN", j
        else:
            if short_tp_hit and short_sl_hit:
                return "LOSE", j
            if short_tp_hit:
                return "WIN", j
            if short_sl_hit:
                return "LOSE", j

    return "TIMEOUT", horizon


def simulate_tp_sl_first_touch_numpy(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    *,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    atr_floor: float,
    tie_break: TieBreak,
    race_mode: RaceMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      bias_codes: int8 array aligned to rows [0 .. n-horizon-1]
                  0 LONG, 1 SHORT, 2 NEUTRAL (SL/timeout/ambiguous)
      hit_bar: int16 bars until resolution (1..horizon)
      event_score: float32 score (0 neutral)
    """
    n = len(close)
    m = n - horizon
    out_bias = np.full(m, 2, dtype=np.int8)
    out_hit = np.zeros(m, dtype=np.int16)
    out_es = np.zeros(m, dtype=np.float32)

    for i in range(m):
        entry = float(close[i])
        a = max(float(atr[i]), atr_floor)
        tp_long = entry + tp_mult * a
        sl_long = entry - sl_mult * a
        tp_short = entry - tp_mult * a
        sl_short = entry + sl_mult * a

        if race_mode == "coupled":
            hit = None
            hb = horizon
            for j in range(1, horizon + 1):
                hi = float(high[i + j])
                lo = float(low[i + j])
                outcome = _resolve_intrabar_outcome(
                    hi,
                    lo,
                    tp_long=tp_long,
                    sl_long=sl_long,
                    tp_short=tp_short,
                    sl_short=sl_short,
                    tie_break=tie_break,
                )
                if outcome is None:
                    continue
                hit = outcome
                hb = j
                break

            if hit is None:
                out_bias[i] = 2
                out_hit[i] = horizon
                out_es[i] = 0.0
            elif hit == "LONG":
                out_bias[i] = 0
                out_hit[i] = hb
                es = float(np.clip(1.0 - (hb - 1) / max(horizon, 1), 0.3, 1.0))
                out_es[i] = es
            elif hit == "SHORT":
                out_bias[i] = 1
                out_hit[i] = hb
                es = float(np.clip(1.0 - (hb - 1) / max(horizon, 1), 0.3, 1.0))
                out_es[i] = es
            else:
                out_bias[i] = 2
                out_hit[i] = hb
                out_es[i] = 0.0
        else:
            lg, lj = _first_long_event_bar(high, low, i, tp_long=tp_long, sl_long=sl_long, horizon=horizon, tie_break=tie_break)
            sg, sj = _first_short_event_bar(high, low, i, tp_short=tp_short, sl_short=sl_short, horizon=horizon, tie_break=tie_break)

            long_win = lg == "WIN"
            short_win = sg == "WIN"

            label = 2
            hb = horizon
            if long_win and short_win:
                if lj < sj:
                    label = 0
                    hb = lj
                elif sj < lj:
                    label = 1
                    hb = sj
                else:
                    label = 2
                    hb = lj
            elif long_win:
                label = 0
                hb = lj
            elif short_win:
                label = 1
                hb = sj

            out_bias[i] = label
            out_hit[i] = hb
            if label == 2:
                out_es[i] = 0.0
            else:
                out_es[i] = float(np.clip(1.0 - (hb - 1) / max(horizon, 1), 0.3, 1.0))

    return out_bias, out_hit, out_es


def add_approx_soft_labels(
    bias_codes: np.ndarray,
    event_score: np.ndarray,
    *,
    clip_long: tuple[float, float] = (0.55, 1.0),
    clip_short: tuple[float, float] = (0.0, 0.45),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ratio-normalized soft labels:
      ratio = event_score / mean(event_score | directional)
      LONG:  clip(0.5 + 0.5 * ratio)
      SHORT: clip(0.5 - 0.5 * ratio)
      NEUTRAL: 0.5
    """
    soft = np.full(len(bias_codes), 0.5, dtype=np.float64)
    wgt = np.zeros(len(bias_codes), dtype=np.float64)

    directional = bias_codes != 2
    mean_es = float(np.mean(event_score[directional])) if np.any(directional) else 0.0
    eps = 1e-9
    ratio = np.zeros_like(event_score, dtype=np.float64)
    ratio[directional] = event_score[directional] / (mean_es + eps)

    long_mask = bias_codes == 0
    short_mask = bias_codes == 1

    soft[long_mask] = np.clip(0.5 + 0.5 * ratio[long_mask], clip_long[0], clip_long[1])
    soft[short_mask] = np.clip(0.5 - 0.5 * ratio[short_mask], clip_short[0], clip_short[1])

    # Heuristic sample weight: emphasize confident directional rows.
    dev = np.abs(soft - 0.5)
    mean_dev = float(np.mean(dev[directional])) if np.any(directional) else 0.0
    if np.any(directional):
        if mean_dev <= eps:
            wgt[directional] = 1.0
        else:
            wgt[directional] = np.clip(dev[directional] / mean_dev, 0.25, 3.0)
    return soft, wgt


def summarize_simulation(
    bias_codes: np.ndarray,
    hit_bar: np.ndarray,
    event_score: np.ndarray,
    soft: np.ndarray,
    *,
    horizon: int,
) -> dict:
    total = len(bias_codes)
    neutral = bias_codes == 2
    directional = ~neutral
    long_c = int(np.sum(bias_codes == 0))
    short_c = int(np.sum(bias_codes == 1))
    neu_c = int(np.sum(neutral))
    denom_ls = long_c + short_c
    long_share = long_c / denom_ls if denom_ls else np.nan
    event_rate = denom_ls / total if total else np.nan
    neutral_share = neu_c / total if total else np.nan

    mean_hit_time = float(np.mean(hit_bar[directional])) if np.any(directional) else np.nan

    mad_soft = float(np.mean(np.abs(soft - 0.5)))
    directional_soft_dev = float(np.mean(np.abs(soft[directional] - 0.5))) if np.any(directional) else np.nan
    pct_extreme = float(np.mean(np.abs(soft - 0.5) >= 0.10))

    return {
        "horizon": horizon,
        "n_rows": total,
        "long": long_c,
        "short": short_c,
        "neutral": neu_c,
        "long_share": long_share,
        "event_rate": event_rate,
        "neutral_share": neutral_share,
        "mean_hit_time": mean_hit_time,
        "mean_event_score": float(np.mean(event_score[directional])) if np.any(directional) else 0.0,
        "mean_abs_soft_dev_all": mad_soft,
        "mean_abs_soft_dev_directional": directional_soft_dev,
        "pct_soft_extreme_ge_0p10": pct_extreme,
    }


def analyze_atr(df: pd.DataFrame, session: pd.Series, out_dir: str, *, enable_plots: bool) -> pd.DataFrame:
    atr = df["atr_14"].astype(float)
    rows = []
    plot_lib = plt if enable_plots else None
    for label in ["ALL"] + sorted(session.unique().tolist()):
        sub = atr if label == "ALL" else atr.loc[session.index[session == label]]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        stats = _percentile_table(sub)
        stats.name = label
        rows.append(stats)
        if plot_lib is not None and len(sub) > 0:
            clipped = sub.clip(lower=sub.quantile(0.001), upper=sub.quantile(0.999))
            plot_lib.figure(figsize=(8, 4))
            plot_lib.hist(clipped, bins=60, color="#4472c4")
            plot_lib.title(f"ATR14 distribution ({label})")
            plot_lib.xlabel("atr_14")
            plot_lib.ylabel("count")
            plot_lib.tight_layout()
            plot_lib.savefig(os.path.join(out_dir, f"hist_atr_{label}.png"), dpi=140)
            plot_lib.close()

    out = pd.concat(rows, axis=1).T
    out.to_csv(os.path.join(out_dir, "atr_percentiles_by_session.csv"))
    return out


def analyze_returns(df: pd.DataFrame, horizons: list[int], out_dir: str, *, enable_plots: bool) -> pd.DataFrame:
    close = df["close"].astype(float).to_numpy()
    rows = []
    for h in horizons:
        fwd = np.full(len(close), np.nan, dtype=float)
        fwd[:-h] = close[h:] / close[:-h] - 1.0
        s = pd.Series(fwd).replace([np.inf, -np.inf], np.nan).dropna()
        stats = {
            "horizon_bars": h,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "p10": float(s.quantile(0.10)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
        }
        rows.append(stats)

        plot_lib = plt if enable_plots else None
        if plot_lib is not None and len(s) > 0:
            clipped = s.clip(lower=s.quantile(0.001), upper=s.quantile(0.999))
            plot_lib.figure(figsize=(8, 4))
            plot_lib.hist(clipped, bins=80, color="#70ad47")
            plot_lib.title(f"Forward simple returns ({h} bars)")
            plot_lib.xlabel("return")
            plot_lib.ylabel("count")
            plot_lib.tight_layout()
            plot_lib.savefig(os.path.join(out_dir, f"hist_fwd_ret_{h}bars.png"), dpi=140)
            plot_lib.close()

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "forward_return_summary.csv"), index=False)
    return out


def weekly_stability(
    ts: pd.Series,
    *,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    tie_break: TieBreak,
    race_mode: RaceMode,
    atr_floor: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    out_dir: str,
) -> pd.DataFrame:
    """
    Repeat simulation on weekly subsets (UTC week starting Monday).
    Entry timestamps align to rows [0 .. n-horizon-1].
    """
    ts_entry = pd.to_datetime(ts.iloc[: len(ts) - horizon])
    ts_entry = ts_entry.dt.tz_convert("UTC").dt.tz_localize(None).reset_index(drop=True)
    weeks = ts_entry.dt.to_period("W-MON")

    rows = []
    for wk, idx in weeks.groupby(weeks).groups.items():
        ix = np.array(list(idx), dtype=int)
        if len(ix) <= horizon:
            continue
        sub_close = close[ix]
        sub_high = high[ix]
        sub_low = low[ix]
        sub_atr = atr[ix]

        bc, hb, es = simulate_tp_sl_first_touch_numpy(
            sub_close,
            sub_high,
            sub_low,
            sub_atr,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            horizon=horizon,
            atr_floor=atr_floor,
            tie_break=tie_break,
            race_mode=race_mode,
        )
        if bc.size == 0:
            continue
        soft, _ = add_approx_soft_labels(bc, es)
        summ = summarize_simulation(bc, hb, es, soft, horizon=horizon)
        summ["week"] = str(wk)
        rows.append(summ)

    out = pd.DataFrame(rows).sort_values("week")
    out.to_csv(
        os.path.join(out_dir, f"weekly_stability_tp{tp_mult}_sl{sl_mult}_h{horizon}.csv"),
        index=False,
    )
    return out


def session_distribution_from_bias(
    ts: pd.Series,
    bias_codes: np.ndarray,
    horizon: int,
    session: pd.Series,
    out_dir: str,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Bias counts per session bucket on entry timestamps."""
    sess_entry = session.iloc[: len(session) - horizon].reset_index(drop=True)
    bias_series = pd.Series(bias_codes, name="bias_code")

    labels = []
    for code in bias_series:
        if code == 0:
            labels.append("LONG")
        elif code == 1:
            labels.append("SHORT")
        else:
            labels.append("NEUTRAL")

    tmp = pd.DataFrame({"session": sess_entry.values, "bias": labels})
    tbl = pd.crosstab(tmp["session"], tmp["bias"], normalize="index") if len(tmp) else pd.DataFrame()
    tbl.to_csv(os.path.join(out_dir, f"{prefix}_bias_share_by_session.csv"))
    counts = pd.crosstab(tmp["session"], tmp["bias"]) if len(tmp) else pd.DataFrame()
    counts.to_csv(os.path.join(out_dir, f"{prefix}_bias_counts_by_session.csv"))
    return tbl


def maybe_compare_to_parquet(df: pd.DataFrame, bias_codes: np.ndarray, horizon: int, out_dir: str) -> None:
    if "bias_label" not in df.columns:
        return
    # Align parquet bias (assuming ordered rows match simulation indexing)
    raw = df["bias_label"].iloc[: len(df) - horizon].reset_index(drop=True)

    # Map parquet labels -> codes if numeric vs strings
    if pd.api.types.is_numeric_dtype(raw):
        parquet_codes = raw.astype(np.int8).to_numpy()
    else:
        mapper = {"LONG": 0, "SHORT": 1, "NEUTRAL": 2}
        parquet_codes = raw.map(lambda x: mapper.get(str(x).upper(), np.nan)).fillna(2).astype(np.int8).to_numpy()

    agree = np.mean(parquet_codes == bias_codes)
    pd.DataFrame({"agreement_rate": [float(agree)]}).to_csv(os.path.join(out_dir, "refinery_bias_agreement.csv"), index=False)


def soft_label_vs_parquet_metrics(
    df: pd.DataFrame,
    approx_soft: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """MAE / Pearson correlation vs parquet soft_label on aligned entry rows."""

    if "soft_label" not in df.columns:
        raise ValueError("Parquet has no soft_label column (needed for --compare-soft-label).")

    ref = df["soft_label"].iloc[: len(df) - horizon].reset_index(drop=True).astype(float).to_numpy()
    if len(ref) != len(approx_soft):
        raise ValueError(f"soft_label alignment mismatch: parquet slice {len(ref)} vs sim {len(approx_soft)}")

    mask = np.isfinite(ref) & np.isfinite(approx_soft)
    if not np.any(mask):
        return {"mae_soft_vs_parquet": float("nan"), "corr_soft_vs_parquet": float("nan"), "n_finite_pairs": 0.0}

    r = ref[mask]
    a = approx_soft[mask]
    mae = float(np.mean(np.abs(r - a)))
    if len(r) > 1 and float(np.std(r)) > 0 and float(np.std(a)) > 0:
        corr = float(np.corrcoef(r, a)[0, 1])
    else:
        corr = float("nan")

    return {
        "mae_soft_vs_parquet": mae,
        "corr_soft_vs_parquet": corr,
        "n_finite_pairs": float(np.sum(mask)),
    }


def composite_soft_neutral_score(mae_soft: float, neutral_share: float, *, weight_soft: float) -> float:
    """
    Lower is better: emphasize matching parquet soft labels and/or shrinking neutral_share.
    weight_soft * MAE + (1-weight_soft) * neutral_share
    """
    w = float(np.clip(weight_soft, 0.0, 1.0))
    if not np.isfinite(mae_soft):
        return float("nan")
    return w * mae_soft + (1.0 - w) * float(neutral_share)


def parse_grid_vals(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_horizons(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    default_parquet = os.path.join(
        os.path.dirname(__file__),
        "day_trading_features_v19_tensorfix.parquet",
    )

    p = argparse.ArgumentParser(description="Analyze thresholds via OHLC+ATR simulations.")
    p.add_argument("--parquet", default=default_parquet, help="Path to features parquet.")
    p.add_argument(
        "parquet_pos",
        nargs="?",
        default=None,
        help="Optional parquet path as final positional argument (overrides --parquet).",
    )
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "threshold_analysis_out"))
    p.add_argument("--horizons", default="1,3,6,12,24", help="Comma horizons for returns + optional simulations.")
    p.add_argument("--grid-tp", default="1.2,1.5,1.8,2.0", help="Comma-separated tp_mult grid.")
    p.add_argument("--grid-sl", default="0.7,1.0,1.2", help="Comma-separated sl_mult grid.")
    p.add_argument(
        "--grid-horizon",
        default="12",
        help='Comma-separated horizons for sensitivity grid (e.g. "6,8,12").',
    )
    p.add_argument("--atr-floor", type=float, default=1e-4, help="Floor on ATR to avoid divide-by-zero artifacts.")
    p.add_argument("--tie-break", choices=["tp_first", "sl_first", "ambiguous_neutral"], default="tp_first")
    p.add_argument(
        "--race-mode",
        choices=["coupled", "independent"],
        default="independent",
        help=(
            "coupled matches the classic 4-barrier snippet on each candle path (SHORT may be rare when tp_mult>=sl_mult). "
            "independent scores long-leg TP-vs-SL and short-leg TP-vs-SL separately, then picks earlier winner "
            "(better for LONG vs SHORT balance diagnostics)."
        ),
    )
    p.add_argument("--plots", action="store_true", help="Write PNG histograms (requires matplotlib).")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG histograms.")
    p.add_argument("--weekly-stability", action="store_true", help="Emit weekly slices for default tp/sl/h.")
    p.add_argument("--weekly-tp", type=float, default=None)
    p.add_argument("--weekly-sl", type=float, default=None)
    p.add_argument("--weekly-h", type=int, default=None)
    p.add_argument("--compare-refinery-bias", action="store_true", help="Compare simulated bias vs parquet bias_label.")
    p.add_argument(
        "--compare-soft-label",
        action="store_true",
        help="Report MAE/corr vs parquet soft_label for each grid row (+ midpoint CSV).",
    )
    p.add_argument(
        "--soft-label-weight",
        type=float,
        default=0.5,
        help="When --compare-soft-label: composite score = w*MAE(soft)+(1-w)*neutral_share (lower is better).",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Print grid progress.")

    args = p.parse_args()

    parquet_path = args.parquet_pos or args.parquet

    horizons = parse_horizons(args.horizons)
    tp_grid = parse_grid_vals(args.grid_tp)
    sl_grid = parse_grid_vals(args.grid_sl)
    grid_horizons = parse_horizons(args.grid_horizon)
    if not grid_horizons:
        raise ValueError("--grid-horizon produced an empty list")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    required = {"open", "high", "low", "close", "atr_14", "ts_event"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")

    df = df.sort_values("ts_event").reset_index(drop=True)
    ts = pd.to_datetime(df["ts_event"], utc=True)

    session = _assign_session_utc(ts)

    enable_plots = bool(plt is not None and args.plots and not args.no_plots)

    atr_tbl = analyze_atr(df, session, args.out_dir, enable_plots=enable_plots)

    ret_tbl = analyze_returns(df, horizons, args.out_dir, enable_plots=enable_plots)

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    atr = df["atr_14"].to_numpy(dtype=np.float64)

    meta = {
        "parquet": os.path.abspath(parquet_path),
        "rows": int(len(df)),
        "time_start": str(ts.iloc[0]),
        "time_end": str(ts.iloc[-1]),
        "tie_break": args.tie_break,
        "race_mode": args.race_mode,
        "atr_floor": args.atr_floor,
        "compare_soft_label": bool(args.compare_soft_label),
        "soft_label_weight": float(args.soft_label_weight),
        "verbose": bool(args.verbose),
        "plots": bool(enable_plots),
        "grid_horizons": grid_horizons,
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Sensitivity grid
    grid_rows = []
    total_cells = len(tp_grid) * len(sl_grid) * len(grid_horizons)
    cell_i = 0
    for h0 in grid_horizons:
        for tp in tp_grid:
            for sl in sl_grid:
                cell_i += 1
                if args.verbose:
                    print(f"[grid {cell_i}/{total_cells}] tp={tp} sl={sl} horizon={h0}", flush=True)
                bc, hb, es = simulate_tp_sl_first_touch_numpy(
                    close,
                    high,
                    low,
                    atr,
                    tp_mult=float(tp),
                    sl_mult=float(sl),
                    horizon=int(h0),
                    atr_floor=float(args.atr_floor),
                    tie_break=args.tie_break,
                    race_mode=args.race_mode,
                )
                soft, wgt = add_approx_soft_labels(bc, es)
                summ = summarize_simulation(bc, hb, es, soft, horizon=int(h0))
                summ.update(
                    {
                        "tp_mult": float(tp),
                        "sl_mult": float(sl),
                        "mean_soft_sample_weight_directional": float(np.mean(wgt[bc != 2])) if np.any(bc != 2) else np.nan,
                    }
                )
                if args.compare_soft_label:
                    sm = soft_label_vs_parquet_metrics(df, soft, int(h0))
                    summ.update(sm)
                    summ["composite_soft_neutral"] = composite_soft_neutral_score(
                        sm["mae_soft_vs_parquet"],
                        summ["neutral_share"],
                        weight_soft=args.soft_label_weight,
                    )
                grid_rows.append(summ)

    grid_df = pd.DataFrame(grid_rows)
    out_grid_path = os.path.join(args.out_dir, "sensitivity_grid.csv")
    if args.compare_soft_label and "composite_soft_neutral" in grid_df.columns:
        grid_df.sort_values(["composite_soft_neutral", "mae_soft_vs_parquet"], ascending=[True, True]).to_csv(
            out_grid_path, index=False
        )
    else:
        grid_df.sort_values(["neutral_share", "event_rate"], ascending=[True, False]).to_csv(out_grid_path, index=False)

    # Save a "screened" view for quick manual selection
    ok = grid_df.copy()
    if "long_share" in ok.columns:
        ok = ok[(ok["long_share"].between(0.45, 0.55, inclusive="both")) | ok["long_share"].isna()]
    if "event_rate" in ok.columns:
        ok = ok[(ok["event_rate"].between(0.15, 0.30, inclusive="both")) | ok["event_rate"].isna()]
    if "neutral_share" in ok.columns:
        ok = ok[ok["neutral_share"] <= 0.70]
    ok.to_csv(os.path.join(args.out_dir, "sensitivity_grid_screened_ruleofthumb.csv"), index=False)

    # Session distribution for baseline grid midpoint / first combo
    tp0 = float(tp_grid[len(tp_grid) // 2])
    sl0 = float(sl_grid[len(sl_grid) // 2])
    h_mid = int(grid_horizons[len(grid_horizons) // 2])
    bc0, hb0, es0 = simulate_tp_sl_first_touch_numpy(
        close,
        high,
        low,
        atr,
        tp_mult=tp0,
        sl_mult=sl0,
        horizon=h_mid,
        atr_floor=float(args.atr_floor),
        tie_break=args.tie_break,
        race_mode=args.race_mode,
    )
    soft0, _ = add_approx_soft_labels(bc0, es0)
    session_distribution_from_bias(ts, bc0, h_mid, session, args.out_dir, prefix=f"midgrid_tp{tp0}_sl{sl0}_h{h_mid}")

    soft_diag_mid = {
        "mean_abs_soft_dev_all": float(np.mean(np.abs(soft0 - 0.5))),
        "neutral_share": float(np.mean(bc0 == 2)),
        "directional_share": float(np.mean(bc0 != 2)),
    }
    if args.compare_soft_label:
        soft_diag_mid.update(soft_label_vs_parquet_metrics(df, soft0, h_mid))
        soft_diag_mid["composite_soft_neutral"] = composite_soft_neutral_score(
            soft_diag_mid["mae_soft_vs_parquet"],
            soft_diag_mid["neutral_share"],
            weight_soft=args.soft_label_weight,
        )
    pd.DataFrame([soft_diag_mid]).to_csv(os.path.join(args.out_dir, "soft_diag_midgrid.csv"), index=False)

    if args.compare_refinery_bias:
        maybe_compare_to_parquet(df, bc0, h_mid, args.out_dir)

    if args.weekly_stability:
        wtp = args.weekly_tp if args.weekly_tp is not None else tp0
        wsl = args.weekly_sl if args.weekly_sl is not None else sl0
        wh = args.weekly_h if args.weekly_h is not None else h_mid
        weekly_stability(
            ts,
            tp_mult=float(wtp),
            sl_mult=float(wsl),
            horizon=int(wh),
            tie_break=args.tie_break,
            race_mode=args.race_mode,
            atr_floor=float(args.atr_floor),
            close=close,
            high=high,
            low=low,
            atr=atr,
            out_dir=args.out_dir,
        )

    print("Wrote outputs to:", os.path.abspath(args.out_dir))
    print("Parquet:", os.path.abspath(parquet_path))
    if args.verbose:
        print(
            f"Grid shape: {len(tp_grid)} tp × {len(sl_grid)} sl × {len(grid_horizons)} horizons = {total_cells} rows",
            flush=True,
        )
    print("ATR percentiles preview:\n", atr_tbl.head())
    print("\nForward return summary preview:\n", ret_tbl.head())
    if args.compare_soft_label and "composite_soft_neutral" in grid_df.columns:
        print(
            "\nSensitivity grid (composite_soft_neutral ascending — lower is better):\n",
            grid_df.sort_values(["composite_soft_neutral", "mae_soft_vs_parquet"]).head(12),
        )
    else:
        print("\nSensitivity grid (sorted by neutral_share):\n", grid_df.sort_values("neutral_share").head(12))
    print("\nRule-of-thumb screened grid:\n", ok.head(12))
    if ok.empty:
        print(
            "\nNote: screened grid is empty — your dataset likely violates one of the rule-of-thumb bands "
            "(long_share in [45%,55%], event_rate in [15%,30%], neutral_share ≤ 70%). "
            "Inspect sensitivity_grid.csv and adjust economics or widen bands."
        )


if __name__ == "__main__":
    main()
