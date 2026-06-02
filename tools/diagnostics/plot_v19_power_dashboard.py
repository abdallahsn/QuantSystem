#!/usr/bin/env python3
"""
QS V19 — لوحة قرار موحّدة: شموع (قابلة للتجميع) + CatBoost LONG/SHORT + عمق LOB (أسلوب Bookmap خفيف)
+ لوحات CVD / OBI / امتصاص / كايل / نظام + Hawkes.

يعتمد على مجلّد refinery (final/features_*.parquet + lob_tensors.npy + lob_tensor_timestamps.npy).
للتنبّه: نموذج CatBoost المرحلة الأولى في V19 يستخدم prepare_training_data.CATBOOST_ADVISOR_FEATURES؛
يجب وجود ملف catboost_advisor_v19.cbm داخل --models-dir (مثلاً train_full بعد تدريب كامل).

مثال (PowerShell، من جذر المشروع):
  py -3.13 plot_v19_power_dashboard.py --pipeline pipeline_mbo7_refinery_latest ^
    --models-dir pipeline_mbo7_refinery_latest/train_full ^
    --out dashboard.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from sklearn.isotonic import IsotonicRegression

from modules.feature_factory_v19 import apply_scaler_params_to_frame
from modules.oof_stacking import align_probability_columns
from prepare_training_data import CATBOOST_ADVISOR_FEATURES

try:
    from catboost import CatBoostClassifier

    _CB_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None  # type: ignore[misc, assignment]
    _CB_AVAILABLE = False

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore[assignment]

from tools.diagnostics.plot_best_soft_label_lob_heatmap import (  # noqa: E402
    _find_best_soft_label_shard,
    _load_lob,
)


def _parquet_cols(path: str) -> set[str]:
    if pq is not None:
        try:
            return set(pq.read_schema(path).names)
        except Exception:
            pass
    return set(pd.read_parquet(path).columns.astype(str))


def _normalize_freq(freq: str) -> str:
    value = str(freq or "5min").strip()
    if not value:
        return "5min"
    return value.replace("T", "min").replace("H", "h")


def _raw_stat_subset(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    prefix = "raw__"
    data: dict[str, pd.Series] = {}
    for col in cols:
        raw_col = f"{prefix}{col}"
        src = raw_col if raw_col in df.columns else col
        if src in df.columns:
            series = pd.to_numeric(df[src], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            series = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
        data[col] = series
    return pd.DataFrame(data, index=df.index)


def _apply_long_calibrator_local(calibrator: IsotonicRegression | None, probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    if calibrator is None or arr.ndim != 2 or arr.shape[1] < 2:
        return arr.astype(np.float32, copy=True)
    p_long = np.clip(arr[:, 0].astype(np.float64), 1e-6, 1.0 - 1e-6)
    cal_long = np.asarray(calibrator.transform(p_long), dtype=np.float64)
    cal_long = np.clip(cal_long, 1e-6, 1.0 - 1e-6)
    out = np.zeros_like(arr, dtype=np.float32)
    out[:, 0] = cal_long.astype(np.float32)
    out[:, 1] = (1.0 - cal_long).astype(np.float32)
    return out


def _load_scaler(models_dir: str) -> dict[str, Any]:
    path = os.path.join(models_dir, "scaler_params.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing scaler_params.json: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_cb_classes(models_dir: str) -> list[int] | None:
    path = os.path.join(models_dir, "catboost_classes_v19.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    classes = payload.get("classes")
    if not isinstance(classes, list):
        return None
    return [int(x) for x in classes]


def _catboost_probs_for_frame(models_dir: str, df: pd.DataFrame) -> pd.DataFrame | None:
    if not _CB_AVAILABLE:
        print("⚠ CatBoost Python package غير متوفر — تم تخطي احتمالات CB", flush=True)
        return None
    cbm = os.path.join(models_dir, "catboost_advisor_v19.cbm")
    if not os.path.isfile(cbm):
        print(f"⚠ لا يوجد {cbm} — تم تخطي طبقة CatBoost (ضع ملف النموذج أو حدّد مسارًا آخر)", flush=True)
        return None
    scaler_params = _load_scaler(models_dir)
    # استخدام نفس مجموعة المرحلة الأولى في train_v19
    feats = list(CATBOOST_ADVISOR_FEATURES)
    subset_scaler = {k: scaler_params[k] for k in feats if k in scaler_params}
    raw = _raw_stat_subset(df, feats)
    scaled = apply_scaler_params_to_frame(raw, subset_scaler, clip_range=None)
    X = scaled.reindex(columns=feats).fillna(0.0).astype(np.float32).values

    model = CatBoostClassifier()
    model.load_model(cbm)
    classes = _load_cb_classes(models_dir)
    raw_probs = align_probability_columns(
        model.predict_proba(X),
        2,
        classes=classes or getattr(model, "classes_", None),
    )
    cal_path = os.path.join(models_dir, "catboost_calibrator_v19.pkl")
    calibrator: IsotonicRegression | None = None
    if os.path.isfile(cal_path):
        try:
            with open(cal_path, "rb") as fh:
                calibrator = pickle.load(fh)
        except Exception:
            calibrator = None
    probs = _apply_long_calibrator_local(calibrator, raw_probs)
    out = df.copy()
    out["cb_prob_long"] = probs[:, 0].astype(np.float64)
    out["cb_prob_short"] = probs[:, 1].astype(np.float64)
    return out


def _gather_window_frame(final_dir: str, t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(glob.glob(os.path.join(final_dir, "features_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no features_*.parquet under {final_dir}")

    base_cols = {"ts_event", "price", "bid_px_00", "ask_px_00", "cvd", "obi", "absorption_intensity", "kyle_lambda", "hawkes_intensity"}
    base_cols.update(CATBOOST_ADVISOR_FEATURES)
    raw_cols = {f"raw__{c}" for c in CATBOOST_ADVISOR_FEATURES}
    base_cols.update(raw_cols)
    base_cols.update(
        {"soft_label", "regime_label", "regime_cluster", "bias_label", "event_score"},
    )

    want: set[str] = set(base_cols)

    chunks: list[pd.DataFrame] = []

    for shard in paths:
        schema = _parquet_cols(shard)
        avail = sorted(want & schema)
        if "ts_event" not in avail:
            continue
        sub = pd.read_parquet(shard, columns=avail)
        sub["ts_event"] = pd.to_datetime(sub["ts_event"], utc=True, errors="coerce").dt.tz_convert(None)
        sub = sub.dropna(subset=["ts_event"]).loc[(sub["ts_event"] >= t0) & (sub["ts_event"] <= t1)]
        if len(sub):
            chunks.append(sub)

    frame = pd.concat(chunks, ignore_index=True).sort_values("ts_event").drop_duplicates(subset=["ts_event"], keep="last")
    critical = {"ts_event"} | set(CATBOOST_ADVISOR_FEATURES)
    missing_f = sorted(critical - set(frame.columns))

    if "price" in frame.columns:
        mid = pd.to_numeric(frame["price"], errors="coerce")
    else:
        mid = pd.Series(np.nan, index=frame.index)
    if mid.isna().all() and "bid_px_00" in frame.columns and "ask_px_00" in frame.columns:
        b = pd.to_numeric(frame["bid_px_00"], errors="coerce")
        a = pd.to_numeric(frame["ask_px_00"], errors="coerce")
        mid = (b + a) / 2.0
    frame["mid_px"] = mid

    return frame.reset_index(drop=True), missing_f


def _resample_dashboard(events: pd.DataFrame, freq: str) -> pd.DataFrame:
    """شموع ومؤشرات مجمّعة حسب نفس حدود الريسامبل (label=right)."""
    freq_n = _normalize_freq(freq)
    ev = events.copy()
    ev["ts_event"] = pd.to_datetime(ev["ts_event"], errors="coerce")
    ev = ev.dropna(subset=["ts_event"]).sort_values("ts_event")
    ev = ev.set_index("ts_event")
    if ev.empty:
        return pd.DataFrame()

    mid = pd.to_numeric(ev["mid_px"], errors="coerce")
    ohlc = mid.resample(freq_n, label="right", closed="right").ohlc()
    ohlc = ohlc.dropna(how="all")

    def _num(col: str, default: float = 0.0) -> pd.Series:
        if col not in ev.columns:
            return pd.Series(default, index=ev.index, dtype=np.float64)
        return pd.to_numeric(ev[col], errors="coerce").fillna(default)

    cvd_last = _num("cvd").resample(freq_n, label="right", closed="right").last()
    cvd_first = _num("cvd").resample(freq_n, label="right", closed="right").first()
    cvd_delta = (cvd_last - cvd_first).rename("cvd_delta")
    obi = _num("obi").resample(freq_n, label="right", closed="right").mean().rename("obi")
    absorption = _num("absorption_intensity").resample(freq_n, label="right", closed="right").mean().rename("absorption_intensity")
    kyle = _num("kyle_lambda").resample(freq_n, label="right", closed="right").mean().rename("kyle_lambda")
    hawkes = _num("hawkes_intensity").resample(freq_n, label="right", closed="right").mean().rename("hawkes_intensity")
    regime = None
    if "regime_cluster" in ev.columns:
        regime = (
            pd.to_numeric(ev["regime_cluster"], errors="coerce")
            .resample(freq_n, label="right", closed="right")
            .mean()
            .rename("regime_cluster")
        )
    elif "regime_label" in ev.columns:
        regime = (
            pd.to_numeric(ev["regime_label"], errors="coerce")
            .resample(freq_n, label="right", closed="right")
            .mean()
            .rename("regime_label")
        )

    cl, cs = pd.Series(0.5, index=ohlc.index), pd.Series(0.5, index=ohlc.index)
    if "cb_prob_long" in ev.columns and "cb_prob_short" in ev.columns:
        cl = pd.to_numeric(ev["cb_prob_long"], errors="coerce").resample(freq_n, label="right", closed="right").mean()
        cs = pd.to_numeric(ev["cb_prob_short"], errors="coerce").resample(freq_n, label="right", closed="right").mean()

    parts = [ohlc, cvd_delta, obi, absorption, kyle, hawkes]
    labels = pd.concat(parts, axis=1)
    labels["cb_prob_long"] = cl.reindex(labels.index).fillna(0.5)
    labels["cb_prob_short"] = cs.reindex(labels.index).fillna(0.5)

    if regime is not None:
        labels["regime"] = regime.reindex(labels.index)

    bars = labels.dropna(subset=["open", "high", "low", "close"]).copy()
    bars["cb_direction_idx"] = np.argmax(bars[["cb_prob_long", "cb_prob_short"]].values, axis=1)
    return bars


def _lob_depth_behind_price(
    lob: np.ndarray,
    lob_ts: pd.Series,
    bar_index: pd.DatetimeIndex,
) -> np.ndarray:
    """مصفوفة (levels، bars): قناة العمق في آخر خطوة زمنية داخل كل تنسور."""
    lob_df = (
        pd.DataFrame({"tensor_idx": np.arange(len(lob_ts), dtype=np.int32), "ts_event": lob_ts})
        .dropna(subset=["ts_event"])
        .sort_values("ts_event")
    )
    if lob_df.empty or len(bar_index) == 0:
        return np.zeros((1, len(bar_index)), dtype=np.float32)

    depths: list[np.ndarray] = []
    for t_end in bar_index:
        row = pd.DataFrame({"ts_event": [pd.Timestamp(t_end)]})
        m = pd.merge_asof(row.sort_values("ts_event"), lob_df, on="ts_event", direction="backward")
        idx_raw = m["tensor_idx"].iloc[0]
        if pd.isna(idx_raw):
            depths.append(np.zeros(lob.shape[2], dtype=np.float32))
            continue
        idx = int(idx_raw)
        snap = np.asarray(lob[idx], dtype=np.float32)
        if snap.ndim != 3 or snap.shape[-1] < 1:
            depths.append(np.zeros(max(lob.shape[2], 1), dtype=np.float32))
            continue
        depth_prof = snap[-1, :, 0].copy()
        depths.append(depth_prof)
    mat = np.stack(depths, axis=1)
    return mat


def _draw_candles(ax, bars: pd.DataFrame, zs: np.ndarray | None, ylims: tuple[float, float]) -> None:
    xdt = bars.index.to_pydatetime()
    xn = mdates.date2num(xdt)
    w = np.median(np.diff(xn)) * 0.55 if len(xn) > 1 else 0.0004

    if zs is not None and zs.shape[1] == len(bars):
        ylo, yhi = ylims
        ax.imshow(
            zs,
            aspect="auto",
            cmap="gist_heat_r",
            origin="upper",
            alpha=0.35,
            interpolation="nearest",
            extent=(
                mdates.date2num(bars.index[0].to_pydatetime()) - w,
                mdates.date2num(bars.index[-1].to_pydatetime()) + w,
                ylo,
                yhi,
            ),
            zorder=1,
            vmin=float(np.percentile(zs, 10)) if np.isfinite(np.percentile(zs, 10)) else None,
            vmax=float(np.percentile(zs, 90)) if np.isfinite(np.percentile(zs, 90)) else None,
        )

    for xi, (_, row) in zip(xn, bars.iterrows(), strict=False):
        o, hi, lo, cl = row["open"], row["high"], row["low"], row["close"]
        if pd.isna([o, hi, lo, cl]).any():
            continue
        color = "#26e07f" if cl >= o else "#ff5568"
        ax.plot([xi, xi], [lo, hi], color=color, linewidth=1.1, alpha=0.95, zorder=3)
        body_low, body_hi = sorted((o, cl))
        h = body_hi - body_low if body_hi > body_low else 1e-8
        ax.add_patch(
            patches.Rectangle(
                (xi - w / 2, body_low),
                w,
                h,
                facecolor=color,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.9,
                zorder=4,
            )
        )


def _decorate_signals(ax, bars: pd.DataFrame, edge: float) -> tuple[int, int]:
    """ماسات LONG تحت الشمعة ، SHORT فوق الشمعة."""
    xdt = bars.index.to_pydatetime()
    xn = mdates.date2num(xdt)
    w = np.median(np.diff(xn)) * 0.55 if len(xn) > 1 else 0.0004
    n_long = n_short = 0
    for xi, (_, row) in zip(xn, bars.iterrows(), strict=False):
        pl, ps = float(row["cb_prob_long"]), float(row["cb_prob_short"])
        if np.isnan(pl) or np.isnan(ps):
            continue
        if pl >= ps + edge and pl >= 0.52:
            ax.scatter(
                xi,
                row["low"] - w * 1.8,
                marker="d",
                s=54,
                color="#26e07f",
                edgecolors="white",
                linewidths=0.6,
                zorder=8,
                label="" if n_long else "CatBoost LONG",
            )
            n_long += 1
        elif ps >= pl + edge and ps >= 0.52:
            ax.scatter(
                xi,
                row["high"] + w * 1.8,
                marker="d",
                s=54,
                color="#ff5568",
                edgecolors="white",
                linewidths=0.6,
                zorder=8,
                label="" if n_short else "CatBoost SHORT",
            )
            n_short += 1
    handles, labels_txt = ax.get_legend_handles_labels()
    uniq: dict[str, Any] = {}
    for h, lab in zip(handles, labels_txt, strict=False):
        if lab and lab not in uniq:
            uniq[lab] = h
    if uniq:
        ax.legend(list(uniq.values()), list(uniq.keys()), loc="upper left", fontsize=8, framealpha=0.68)
    return n_long, n_short


def main() -> None:
    p = argparse.ArgumentParser(description="QS V19 power dashboard — LOB + CatBoost + مؤشرات")
    p.add_argument("--pipeline", default="pipeline_mbo7_refinery_latest", help="مجلّد refinery")
    p.add_argument(
        "--models-dir",
        default="",
        help="مجلّد فيه scaler_params.json + catboost_advisor_v19.cbm؛ إذا فُرِغ يُجرّب <pipeline>/train_full",
    )
    p.add_argument("--out", required=True, help="PNG الناتجة")
    p.add_argument("--window-minutes", type=float, default=180.0, help="عرض الزمن ± حول مركز الانحناء")
    p.add_argument("--freq", default="5min", help="تجميع الشموع (مثلاً 5min أو 15min)")
    p.add_argument("--anchor", choices=("best-soft-label", "ts"), default="best-soft-label")
    p.add_argument("--ts-event", default="", help="إذا anchor=ts — وقت ISO حدث الميزات مركزًا")
    p.add_argument("--signal-edge", type=float, default=0.06, help="حدّ الفارق بين واحتمالي LONG/SHORT لرسم الماسة")
    p.add_argument("--no-lob", action="store_true", help="عدم رسم هيتماب العمق")
    args = p.parse_args()

    root = os.path.abspath(args.pipeline.rstrip("\\/ "))
    final_dir = os.path.join(root, "final")
    plt.style.use("dark_background")

    if args.anchor == "best-soft-label":
        center_ts, soft_max, _ = _find_best_soft_label_shard(final_dir)
        print(f"anchor=best-soft-label | ts={center_ts} | soft_label={soft_max:g}", flush=True)
    else:
        if not str(args.ts_event).strip():
            print("⚠ تحتاج --ts-event مع --anchor ts", file=sys.stderr)
            sys.exit(2)
        center_ts = pd.to_datetime(args.ts_event, utc=True).tz_convert(None)

    hw = pd.Timedelta(minutes=float(args.window_minutes))
    t0, t1 = center_ts - hw, center_ts + hw
    events, miss = _gather_window_frame(final_dir, t0, t1)
    if miss:
        print(f"⚠ أعمدة ناقصة في النوافذ (قد تُعبأ بصفور): {', '.join(miss[:24])}", flush=True)
    print(f"rows in ±{args.window_minutes:g} min: {len(events):,}", flush=True)

    candidate_models = (
        os.path.abspath(args.models_dir.strip()) if args.models_dir.strip() else os.path.join(root, "train_full")
    )
    models_dir = candidate_models if os.path.isdir(candidate_models) else ""
    if models_dir:
        try:
            cb_frame = _catboost_probs_for_frame(models_dir, events)
            if cb_frame is not None:
                events = cb_frame
        except Exception as exc:
            print(f"⚠ CatBoost inference failed ({exc}) — proceeding without overlays", flush=True)

    freq_n = _normalize_freq(args.freq)
    bars_raw = _resample_dashboard(events, freq_n)
    if bars_raw.empty:
        print("لا توجد شموع بعد التجميع — زِد النوافذ أو غيّر --freq.", file=sys.stderr)
        sys.exit(1)

    lob_aligned: np.ndarray | None = None
    if not args.no_lob:
        lob, lob_ts = _load_lob(root)
        lob_mat = _lob_depth_behind_price(np.asarray(lob), lob_ts, bars_raw.index)
        y_min_b = float(bars_raw["low"].min())
        y_max_b = float(bars_raw["high"].max())
        rng = max(y_max_b - y_min_b, 5e-5)
        pad = max(rng * 0.12, rng * 0.001)
        ylims_plot = (y_min_b - pad, y_max_b + pad)
        lob_aligned = lob_mat.astype(np.float32)
    else:
        y_min_b = float(bars_raw["low"].min())
        y_max_b = float(bars_raw["high"].max())
        pad = max((y_max_b - y_min_b) * 0.06, 1e-8)
        ylims_plot = (y_min_b - pad, y_max_b + pad)

    fig_height = 12.5
    fig = plt.figure(figsize=(17, fig_height), layout="constrained")
    gs = fig.add_gridspec(7, 1, height_ratios=[3.0, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95], hspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    axes_p = [fig.add_subplot(gs[i + 1, 0], sharex=ax0) for i in range(6)]

    ax0.axvline(pd.Timestamp(center_ts), color="#00fff7", linestyle="--", linewidth=1.35, alpha=0.85, label="قرار المركز")

    _draw_candles(ax0, bars_raw, lob_aligned, ylims_plot)
    nl, ns = _decorate_signals(ax0, bars_raw, float(args.signal_edge))

    title = (
        f"QS V19 — {_normalize_freq(args.freq)} | LOB depth (channel 0) + CatBoost + microstructure "
        f"\n{center_ts} ±{args.window_minutes:g}m | LONG markers={nl} SHORT markers={ns}"
    )
    ax0.set_title(title, fontsize=10.5)
    ax0.set_ylabel("price (Bookmap-style depth fade beneath candles)")
    ax0.grid(True, alpha=0.18)

    # لوحات فرعية
    x_idx = bars_raw.index

    def _bars_panel(ax: Any, ys: pd.Series, title_txt: str, kind: str) -> None:
        xn = mdates.date2num(x_idx.to_pydatetime())
        vv = ys.reindex(x_idx).fillna(0.0).values.astype(float)
        if kind == "hist":
            w = np.median(np.diff(xn)) * 0.8 if len(xn) > 1 else 0.00035
            pos = vv >= 0
            ax.bar(xn[pos], vv[pos], width=w * 0.95, color="#26e07f", alpha=0.75)
            ax.bar(xn[~pos], vv[~pos], width=w * 0.95, color="#ff5568", alpha=0.75)
        elif kind == "line":
            ax.plot(x_idx, vv, color="#fbc02d", linewidth=1.0)
        elif kind == "fill":
            ax.fill_between(x_idx, vv, color="#b388ff", alpha=0.45)
            ax.plot(x_idx, vv, color="#e1bee7", linewidth=0.6)
        else:
            ax.plot(x_idx, vv, color="#fbc02d", linewidth=1.0)
        ax.set_ylabel(title_txt, fontsize=7.8)
        ax.grid(True, alpha=0.16)

    cvd_series = bars_raw["cvd_delta"] if "cvd_delta" in bars_raw.columns else pd.Series(0.0, index=bars_raw.index)
    _bars_panel(axes_p[0], cvd_series, "CVD Δ", "hist")
    axes_p[0].set_title("Cumulative volume delta Δ (bucket)", fontsize=8, loc="left", color="#aaa")

    _bars_panel(axes_p[1], bars_raw["obi"], "OBI", "hist")
    abs_s = bars_raw["absorption_intensity"] if "absorption_intensity" in bars_raw.columns else pd.Series(0.0, index=bars_raw.index)
    _bars_panel(axes_p[2], abs_s, "Absorption", "fill")
    ky = bars_raw["kyle_lambda"] if "kyle_lambda" in bars_raw.columns else pd.Series(np.nan, index=bars_raw.index)
    axes_p[3].fill_between(ky.dropna().index, ky.dropna().values.astype(float), color="#fff59d33")
    ky2 = ky.reindex(bars_raw.index).ffill().fillna(0.0)
    axes_p[3].plot(bars_raw.index, ky2.values, color="#fdd835", linewidth=1.05)
    axes_p[3].set_ylabel("Kyle λ")
    axes_p[3].grid(True, alpha=0.14)

    ax_combo = axes_p[4]
    hawkes_vals = (
        pd.to_numeric(bars_raw["hawkes_intensity"], errors="coerce")
        if "hawkes_intensity" in bars_raw.columns
        else pd.Series(np.nan, index=bars_raw.index)
    )

    if "regime" in bars_raw.columns:
        rg = pd.to_numeric(bars_raw["regime"], errors="coerce").fillna(0.0)
        xn = mdates.date2num(x_idx.to_pydatetime())
        mx = float(rg.abs().max() or 1.0)
        w = np.median(np.diff(xn)) * 0.82 if len(xn) > 1 else 0.00035
        ax_combo.bar(
            xn,
            ((rg / mx).fillna(0.0).values.astype(float)),
            width=w * 0.95,
            color="#00e676",
            alpha=0.52,
            label="regime (norm)",
        )
        ax_combo.set_ylabel("Regime")
        ax_combo.legend(loc="upper left", fontsize=7, framealpha=0.65)
    else:
        ax_combo.text(0.5, 0.55, "regime_* missing in parquet", ha="center", va="center", transform=ax_combo.transAxes)

    if hawkes_vals.notna().any():
        ax_h = ax_combo.twinx()
        ax_h.plot(bars_raw.index, hawkes_vals.values.astype(float), color="#ff9800", linewidth=1.05, label="Hawkes I")
        ax_h.set_ylabel("Hawkes I", color="#ffb74d")
        ax_h.tick_params(axis="y", labelcolor="#ffb74d")
        ax_h.legend(loc="upper right", fontsize=7, framealpha=0.65)

    ax_soft = axes_p[5]
    if "soft_label" in events.columns:
        se = pd.to_numeric(events["soft_label"], errors="coerce")
        ev_idx = pd.to_datetime(events["ts_event"], errors="coerce")
        sg = pd.Series(se.values, index=ev_idx).sort_index().dropna()
        if len(sg.index):
            soft_bar = sg.resample(freq_n, label="right", closed="right").mean().reindex(bars_raw.index).ffill().bfill()
            ax_soft.plot(bars_raw.index, soft_bar.values.astype(float), color="#40c4ff", linewidth=1.05, alpha=0.95, label="soft_label μ")
            ax_soft.legend(loc="upper left", fontsize=7)
            ax_soft.grid(True, alpha=0.12)
            ax_soft.set_ylabel("soft_label")
        else:
            ax_soft.text(0.5, 0.5, "(no soft_label values)", ha="center", va="center", transform=ax_soft.transAxes)
    else:
        ax_soft.text(0.5, 0.5, "(no soft_label column)", ha="center", va="center", transform=ax_soft.transAxes)

    ax_soft.set_xlabel("time")

    for ax in axes_p[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.setp(ax0.get_xticklabels(), visible=False)

    outp = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {outp}", flush=True)


if __name__ == "__main__":
    main()
