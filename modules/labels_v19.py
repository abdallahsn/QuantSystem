"""
labels_v19.py - Unified causal labels for QuantSystem
=====================================================
Compatibility wrapper around the unified order-book + price-action labeling pipeline.

External callers still use `build_causal_event_labels(...)`, but internally the
labels now come from:

  order book scan -> feature engineering -> event filter
  -> forward scan -> direction/quality/regime labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.dynamic_labels import (
    DIR_LONG,
    DIR_NEUTRAL,
    DIR_SHORT,
    LABEL_CANCEL,
    QUALITY_NONE,
    QUALITY_STRONG,
    QUALITY_WEAK,
    REGIME_RANGING,
    append_dynamic_orderbook_features,
    build_event_filter,
    engineer_features,
    label_with_forward_scan,
)


BIAS_LONG = DIR_LONG
BIAS_SHORT = DIR_SHORT
BIAS_NEUTRAL = DIR_NEUTRAL

SETUP_ABSORPTION = 0
SETUP_SPOOFING = 1
SETUP_OBI = 2
SETUP_MIXED = 3


def _infer_setup_labels(df: pd.DataFrame) -> np.ndarray:
    absorb = pd.to_numeric(df.get("absorption_intensity", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).abs().values
    spoof = pd.to_numeric(df.get("spoofing_ratio", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).abs().values
    obi = pd.to_numeric(df.get("obi", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).abs().values

    absorb_sig = absorb > 0.30
    spoof_sig = spoof > 0.20
    obi_sig = obi > 0.30
    sig_count = absorb_sig.astype(np.int8) + spoof_sig.astype(np.int8) + obi_sig.astype(np.int8)

    setup = np.full(len(df), SETUP_MIXED, dtype=np.int8)
    setup[(sig_count == 1) & absorb_sig] = SETUP_ABSORPTION
    setup[(sig_count == 1) & spoof_sig] = SETUP_SPOOFING
    setup[(sig_count == 1) & obi_sig] = SETUP_OBI
    return setup


def _liquidity_score(df: pd.DataFrame) -> np.ndarray:
    volume = pd.to_numeric(df.get("volume", df.get("size", pd.Series(np.zeros(len(df))))), errors="coerce").fillna(0.0).clip(lower=0.0).values
    depth = pd.to_numeric(df.get("liquidity_density", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).clip(lower=0.0).values
    gap = pd.to_numeric(df.get("gap_size", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).clip(lower=0.0).values
    wall_bid = pd.to_numeric(df.get("bid_wall_strength", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).clip(lower=0.0).values
    wall_ask = pd.to_numeric(df.get("ask_wall_strength", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).clip(lower=0.0).values
    obi = pd.to_numeric(df.get("obi", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).abs().values

    wall = np.maximum(wall_bid, wall_ask)
    raw = np.log1p(volume) * (1.0 + obi) * (1.0 + wall) + 0.25 * depth + 0.10 * gap
    return np.clip(raw, 0.0, 100.0).astype(np.float32)


def _quality_to_conf(signal_quality: pd.Series | np.ndarray) -> np.ndarray:
    q = pd.Series(signal_quality).fillna(QUALITY_NONE).astype(np.int8)
    out = np.zeros(len(q), dtype=np.float32)
    out[q.values == QUALITY_WEAK] = 0.5
    out[q.values == QUALITY_STRONG] = 1.0
    return out


def _empty_output(out: pd.DataFrame) -> pd.DataFrame:
    out["bias_label"] = np.array([], dtype=np.int8)
    out["setup_label"] = np.array([], dtype=np.int8)
    out["conf_label"] = np.array([], dtype=np.float32)
    out["signal_quality"] = np.array([], dtype=np.int8)
    out["is_expansion"] = np.array([], dtype=np.int8)
    out["liq_score"] = np.array([], dtype=np.float32)
    out["regime_label"] = np.array([], dtype=np.int8)
    out["label_end_ts"] = pd.to_datetime([])
    out["forward_return"] = np.array([], dtype=np.float32)
    out["label_horizon_steps"] = np.array([], dtype=np.int32)
    out["long_label"] = np.array([], dtype=np.int8)
    out["short_label"] = np.array([], dtype=np.int8)
    out["event_flag"] = np.array([], dtype=np.int8)
    return out


def build_causal_event_labels(
    df: pd.DataFrame,
    horizon: int = 50,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    neutral_mult: float = 0.35,
    tick_size: float = 1e-4,
) -> pd.DataFrame:
    """
    Build causal labels using unified order-book features and price-only future direction.

    Signature kept for backwards compatibility with the current refinery.
    """

    out = df.copy()
    n = len(out)
    if n == 0:
        return _empty_output(out)

    price_col = "price" if "price" in out.columns else ("close" if "close" in out.columns else "micro_price")
    out[price_col] = pd.to_numeric(out.get(price_col, pd.Series(np.zeros(n))), errors="coerce").ffill().fillna(0.0).astype(np.float32)
    out["close"] = out[price_col].astype(np.float32)
    out["volume"] = pd.to_numeric(out.get("size", out.get("volume", pd.Series(np.zeros(n)))), errors="coerce").fillna(0.0).astype(np.float32)
    out["cvd"] = pd.to_numeric(out.get("cvd", pd.Series(np.zeros(n))), errors="coerce").fillna(0.0).astype(np.float32)

    dynamic_cols = [
        "micro_price", "bid_wall_strength", "ask_wall_strength",
        "distance_to_wall", "gap_size", "liquidity_density",
    ]
    has_dynamic_context = all(col in out.columns for col in dynamic_cols)
    has_depth = any(col.startswith("bid_px_") for col in out.columns) and any(col.startswith("ask_px_") for col in out.columns)

    if (not has_dynamic_context) and has_depth:
        out = append_dynamic_orderbook_features(
            out,
            tick_size=tick_size,
            price_col=price_col,
            cvd_col="cvd",
            volume_col="volume",
        )
    else:
        for col in dynamic_cols:
            if col not in out.columns:
                out[col] = 0.0
        if "micro_price" in out.columns:
            out["micro_price"] = pd.to_numeric(out["micro_price"], errors="coerce").fillna(out["close"]).astype(np.float32)

    roll_window = max(12, min(64, max(6, int(horizon // 2) if horizon > 0 else 20)))
    out = engineer_features(out, roll_window=roll_window)

    vol_mult = float(np.clip(1.0 + neutral_mult, 1.15, 1.80))
    obi_thr = float(np.clip(neutral_mult, 0.15, 0.40))
    wall_thr = float(np.clip(0.80 + neutral_mult * 0.5, 0.90, 1.20))
    event_mask = build_event_filter(out, vol_mult=vol_mult, obi_thr=obi_thr, wall_str_thr=wall_thr)
    labeled = label_with_forward_scan(out, out, max_bars_forward=horizon, tick_size=tick_size)

    quiet_rows = ~event_mask.fillna(False)
    if quiet_rows.any():
        labeled.loc[quiet_rows, "bias_label"] = BIAS_NEUTRAL
        labeled.loc[quiet_rows, "signal_quality"] = QUALITY_NONE
        labeled.loc[quiet_rows, "long_label"] = LABEL_CANCEL
        labeled.loc[quiet_rows, "short_label"] = LABEL_CANCEL

    ts = pd.to_datetime(
        labeled.get("ts_event", pd.Series(pd.RangeIndex(n))),
        utc=True,
        errors="coerce",
    ).dt.tz_localize(None)
    if ts.isna().all():
        ts = pd.Series(pd.date_range("2026-01-01", periods=n, freq="s"))
    else:
        ts = ts.ffill().bfill()

    end_idx = np.minimum(np.arange(n) + max(int(horizon), 1), n - 1).astype(np.int32)
    prices = labeled["close"].astype(np.float64).values

    labeled["setup_label"] = _infer_setup_labels(labeled)
    labeled["conf_label"] = _quality_to_conf(labeled["signal_quality"])
    labeled["is_expansion"] = (
        (labeled["bias_label"].astype(np.int8) != BIAS_NEUTRAL)
        & (labeled["signal_quality"].astype(np.int8) == QUALITY_STRONG)
    ).astype(np.int8)
    labeled["liq_score"] = _liquidity_score(labeled)
    labeled["regime_label"] = pd.to_numeric(labeled.get("regime", REGIME_RANGING), errors="coerce").fillna(REGIME_RANGING).astype(np.int8)
    labeled["label_end_ts"] = pd.to_datetime(ts.iloc[end_idx].to_numpy())
    labeled["forward_return"] = (prices[end_idx] - prices).astype(np.float32)
    labeled["label_horizon_steps"] = (end_idx - np.arange(n)).astype(np.int32)
    labeled["event_flag"] = event_mask.fillna(False).astype(np.int8)

    total = max(len(labeled), 1)
    bias_counts = labeled["bias_label"].value_counts()
    qual_counts = labeled["signal_quality"].value_counts()
    print(
        f"  Final Bias: LONG={bias_counts.get(BIAS_LONG, 0)}({bias_counts.get(BIAS_LONG, 0) / total:.0%}) "
        f"SHORT={bias_counts.get(BIAS_SHORT, 0)}({bias_counts.get(BIAS_SHORT, 0) / total:.0%}) "
        f"NEUTRAL={bias_counts.get(BIAS_NEUTRAL, 0)}({bias_counts.get(BIAS_NEUTRAL, 0) / total:.0%})"
    )
    print(
        f"  Final Quality: STRONG={qual_counts.get(QUALITY_STRONG, 0)} "
        f"WEAK={qual_counts.get(QUALITY_WEAK, 0)} "
        f"NONE={qual_counts.get(QUALITY_NONE, 0)} "
        f"| Events={int(labeled['event_flag'].sum())}/{total}"
    )

    return labeled
