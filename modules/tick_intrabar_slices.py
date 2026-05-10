"""
Intrabar features from MBO ticks (slice within each bar).
Canonical source for prepare_day_trading.py — DO NOT confuse with intrabar_mbp_microstructure.py (MBP10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_dt64(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)
    return out


def _coef_var(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    m = float(np.mean(x))
    if abs(m) < 1e-12:
        return 0.0
    return float(np.std(x) / (abs(m) + 1e-12))


def enrich_bars_with_intrabar(
    df_bars: pd.DataFrame,
    df_ticks: pd.DataFrame,
    *,
    freq: str = "5min",
    n_slices: int = 6,
) -> pd.DataFrame:
    """
    Split each bar into `n_slices` time slices and compute spike-robust microstructure features.

    This is designed for day-trading (bar-level) pipeline: keep "max spike" signals instead of mean dilution.
    It only relies on columns if present; otherwise fills zeros (safe for smaller datasets / partial inputs).
    """
    bars = df_bars.copy().sort_values("ts_event").reset_index(drop=True)
    if bars.empty:
        return bars

    ticks = df_ticks.copy()
    if "ts_event" not in ticks.columns:
        raise ValueError("df_ticks must contain ts_event")

    ticks["ts_event"] = _to_dt64(ticks["ts_event"])
    ticks = ticks[ticks["ts_event"].notna()].sort_values("ts_event")
    if ticks.empty:
        # add columns as zeros for schema stability
        for col in (
            "spoof_peak_slice",
            "spoof_burst_flag",
            "cancel_burst",
            "absorption_speed",
            "pressure_phase",
            "cvd_velocity_max",
            "cvd_direction_pct",
            "cvd_early_vs_late",
            "ofi_peak_slice",
            "size_dispersion",
            "absorption_bar_slice_max",
        ):
            bars[col] = 0.0
        bars["pressure_phase"] = "uniform"
        return bars

    # Ensure required numeric series exist (fallback to zeros)
    def num(col: str) -> pd.Series:
        if col not in ticks.columns:
            return pd.Series(np.zeros(len(ticks), dtype=np.float32), index=ticks.index)
        return pd.to_numeric(ticks[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cancel_ratio = num("cancel_ratio").astype(np.float32)
    spoof_ratio = num("spoofing_ratio").astype(np.float32)
    absorption = num("absorption_intensity").astype(np.float32)
    cvd = num("cvd").astype(np.float64)
    size = num("size").astype(np.float32)

    # Compute a robust OFI proxy per tick if explicit column missing.
    if "order_flow_imbalance" in ticks.columns:
        ofi_tick = pd.to_numeric(ticks["order_flow_imbalance"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        side = ticks.get("side", pd.Series("", index=ticks.index)).astype(str).str.upper()
        buy = (size.where(side.eq("A"), 0.0)).astype(np.float64)
        sell = (size.where(side.eq("B"), 0.0)).astype(np.float64)
        tot = (buy + sell).replace(0.0, np.nan)
        ofi_tick = ((buy - sell) / tot).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)

    bars["ts_event"] = _to_dt64(bars["ts_event"])
    bar_starts = bars["ts_event"].to_numpy(dtype="datetime64[ns]", copy=False)
    bar_delta = pd.to_timedelta(freq)
    slice_td = bar_delta / max(int(n_slices), 1)

    # Pre-allocate outputs
    spoof_peak_slice = np.zeros(len(bars), dtype=np.float32)
    spoof_burst_flag = np.zeros(len(bars), dtype=np.int8)
    cancel_burst = np.zeros(len(bars), dtype=np.float32)
    absorption_speed = np.zeros(len(bars), dtype=np.float32)
    pressure_phase = np.array(["uniform"] * len(bars), dtype=object)
    cvd_velocity_max = np.zeros(len(bars), dtype=np.float32)
    cvd_direction_pct = np.zeros(len(bars), dtype=np.float32)
    cvd_early_vs_late = np.zeros(len(bars), dtype=np.float32)
    ofi_peak_slice = np.zeros(len(bars), dtype=np.float32)
    size_dispersion = np.zeros(len(bars), dtype=np.float32)
    absorption_bar_slice_max = np.zeros(len(bars), dtype=np.float32)

    tick_ts = ticks["ts_event"].to_numpy(dtype="datetime64[ns]", copy=False)

    for i, t0 in enumerate(bar_starts):
        if pd.isna(t0):
            continue
        t1 = t0 + np.timedelta64(int(bar_delta.total_seconds() * 1e9), "ns")

        # slice ticks via boolean mask (simple + correct; small data target)
        m = (tick_ts >= t0) & (tick_ts < t1)
        if not np.any(m):
            continue

        sub_idx = np.flatnonzero(m)
        sub_ts = tick_ts[sub_idx]

        # compute slice ids
        rel = (sub_ts - t0).astype("timedelta64[ns]").astype(np.int64)
        slice_ns = int(slice_td.total_seconds() * 1e9)
        slice_id = np.clip(rel // max(slice_ns, 1), 0, max(int(n_slices) - 1, 0)).astype(np.int32)

        # aggregate per-slice
        spoof_s = []
        cancel_s = []
        abs_s = []
        ofi_s = []
        cvd_s = []
        size_s = []

        for s in range(int(n_slices)):
            sm = slice_id == s
            if not np.any(sm):
                spoof_s.append(0.0)
                cancel_s.append(0.0)
                abs_s.append(0.0)
                ofi_s.append(0.0)
                cvd_s.append(0.0)
                size_s.append(0.0)
                continue

            ids = sub_idx[sm]
            spoof_s.append(float(np.nanmax(spoof_ratio.iloc[ids].to_numpy(dtype=np.float32, copy=False))))
            cancel_s.append(float(np.nanmax(cancel_ratio.iloc[ids].to_numpy(dtype=np.float32, copy=False))))
            abs_s.append(float(np.nanmean(absorption.iloc[ids].to_numpy(dtype=np.float32, copy=False))))
            ofi_s.append(float(np.nanmax(np.abs(ofi_tick.iloc[ids].to_numpy(dtype=np.float32, copy=False)))))

            c = cvd.iloc[ids].to_numpy(dtype=np.float64, copy=False)
            cvd_s.append(float(c[-1] - c[0]) if c.size else 0.0)

            sz = size.iloc[ids].to_numpy(dtype=np.float64, copy=False)
            size_s.append(float(np.nanmean(sz)) if sz.size else 0.0)

        spoof_s = np.asarray(spoof_s, dtype=np.float32)
        cancel_s = np.asarray(cancel_s, dtype=np.float32)
        abs_s = np.asarray(abs_s, dtype=np.float32)
        ofi_s = np.asarray(ofi_s, dtype=np.float32)
        cvd_s = np.asarray(cvd_s, dtype=np.float32)
        size_s = np.asarray(size_s, dtype=np.float32)

        # 1) max spoof slice + burst flag
        peak = float(np.max(spoof_s))
        spoof_peak_slice[i] = peak
        mean_spoof = float(np.mean(spoof_s)) if spoof_s.size else 0.0
        spoof_burst_flag[i] = 1 if (mean_spoof > 1e-9 and peak >= 2.0 * mean_spoof) else 0

        # 2) cancel burst
        cancel_burst[i] = float(np.max(cancel_s))

        # 3) absorption speed (CV across slices)
        absorption_speed[i] = float(_coef_var(abs_s))
        absorption_bar_slice_max[i] = float(np.max(abs_s))  # helper for downstream if needed

        # 4) pressure phase
        # pick where absorption concentrated
        thirds = np.array_split(np.arange(int(n_slices)), 3)
        third_scores = [float(np.mean(abs_s[idx])) if len(idx) else 0.0 for idx in thirds]
        best = int(np.argmax(third_scores)) if third_scores else 1
        if np.allclose(third_scores, third_scores[0], atol=1e-6):
            pressure_phase[i] = "uniform"
        else:
            pressure_phase[i] = ("early", "mid", "late")[best]

        # 5) cvd velocity + direction consistency
        # velocity max = max abs per-slice delta normalized by slice duration
        sec = max(float(slice_td.total_seconds()), 1e-6)
        cvd_velocity_max[i] = float(np.max(np.abs(cvd_s)) / sec)
        signs = np.sign(cvd_s)
        nonzero = signs != 0
        if np.any(nonzero):
            dominant = np.sign(np.sum(signs[nonzero]))
            if dominant == 0:
                dominant = 1.0
            cvd_direction_pct[i] = float(np.mean(signs[nonzero] == dominant))
        else:
            cvd_direction_pct[i] = 0.0

        # early vs late pressure
        k = max(1, int(n_slices) // 3)
        early = float(np.sum(cvd_s[:k]))
        late = float(np.sum(cvd_s[-k:]))
        denom = abs(early) + abs(late) + 1e-9
        cvd_early_vs_late[i] = float((early - late) / denom)

        # 6) OFI peak slice
        ofi_peak_slice[i] = float(np.max(ofi_s))

        # 7) size dispersion (CV of mean trade size across slices)
        size_dispersion[i] = float(_coef_var(size_s))

    bars["spoof_peak_slice"] = spoof_peak_slice
    bars["spoof_burst_flag"] = spoof_burst_flag.astype(np.int8)
    bars["cancel_burst"] = cancel_burst
    bars["absorption_speed"] = absorption_speed
    bars["pressure_phase"] = pressure_phase
    bars["cvd_velocity_max"] = cvd_velocity_max
    bars["cvd_direction_pct"] = cvd_direction_pct
    bars["cvd_early_vs_late"] = cvd_early_vs_late
    bars["ofi_peak_slice"] = ofi_peak_slice
    bars["size_dispersion"] = size_dispersion
    bars["absorption_bar_slice_max"] = absorption_bar_slice_max

    return bars
