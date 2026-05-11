from __future__ import annotations

import numpy as np
import pandas as pd


MBP_LEVELS_DEFAULT = 10
MBP_INTRABAR_SLICES_DEFAULT = 12


def _to_dt64_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return s.astype(np.float64)


def _slice_ids(ts: np.ndarray, t0: np.datetime64, slice_ns: int, n_slices: int) -> np.ndarray:
    rel = (ts - t0).astype("timedelta64[ns]").astype(np.int64)
    sid = rel // max(int(slice_ns), 1)
    return np.clip(sid, 0, max(int(n_slices) - 1, 0)).astype(np.int32)


def _depth_slope(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64).ravel()
    if v.size < 2:
        return 0.0
    x = np.arange(v.size, dtype=np.float64)
    return float(np.polyfit(x, v, 1)[0])


def enrich_bars_with_intrabar_mbp(
    df_bars: pd.DataFrame,
    df_mbp: pd.DataFrame,
    *,
    freq: str = "5min",
    n_slices: int = MBP_INTRABAR_SLICES_DEFAULT,
    levels: int = MBP_LEVELS_DEFAULT,
) -> pd.DataFrame:
    """
    Powerful MBP10 intrabar aggregation: split each bar into `n_slices` and compute book-based features.

    Requires (at least): ts_event, bid_px_00/ask_px_00, bid_sz_00/ask_sz_00 ...
    """
    bars = df_bars.copy().sort_values("ts_event").reset_index(drop=True)
    if bars.empty:
        return bars

    mbp = df_mbp.copy()
    if "ts_event" not in mbp.columns:
        raise ValueError("df_mbp must contain ts_event")

    mbp["ts_event"] = _to_dt64_ns(mbp["ts_event"])
    mbp = mbp[mbp["ts_event"].notna()].sort_values("ts_event")
    if mbp.empty:
        print("  ⚠️ enrich_bars_with_intrabar_mbp: dataframe MBP فارغ — عمود mbp_* يبقى صفرًا.")
        return bars

    # Basic book arrays
    bid0 = _num(mbp, "bid_px_00")
    ask0 = _num(mbp, "ask_px_00")
    spread = (ask0 - bid0).clip(lower=0.0)
    mid = (ask0 + bid0) / 2.0

    bid_sz_cols = [f"bid_sz_{i:02d}" for i in range(int(levels))]
    ask_sz_cols = [f"ask_sz_{i:02d}" for i in range(int(levels))]
    bid_ct_cols = [f"bid_ct_{i:02d}" for i in range(int(levels)) if f"bid_ct_{i:02d}" in mbp.columns]
    ask_ct_cols = [f"ask_ct_{i:02d}" for i in range(int(levels)) if f"ask_ct_{i:02d}" in mbp.columns]

    bid_sz_mat = np.vstack([_num(mbp, c).to_numpy(dtype=np.float64, copy=False) for c in bid_sz_cols]).T
    ask_sz_mat = np.vstack([_num(mbp, c).to_numpy(dtype=np.float64, copy=False) for c in ask_sz_cols]).T
    bid_depth = bid_sz_mat.sum(axis=1)
    ask_depth = ask_sz_mat.sum(axis=1)
    depth_sum = bid_depth + ask_depth
    imbalance = np.where(depth_sum > 0, (bid_depth - ask_depth) / depth_sum, 0.0)

    # microprice using L1 sizes when possible
    b1 = bid_sz_mat[:, 0] if bid_sz_mat.shape[1] else np.zeros(len(mbp), dtype=np.float64)
    a1 = ask_sz_mat[:, 0] if ask_sz_mat.shape[1] else np.zeros(len(mbp), dtype=np.float64)
    denom = (b1 + a1)
    microprice = np.where(denom > 0, (ask0.to_numpy() * b1 + bid0.to_numpy() * a1) / denom, mid.to_numpy())

    ts = mbp["ts_event"].to_numpy(dtype="datetime64[ns]", copy=False)

    n_slices_eff = int(max(int(n_slices), 2))
    bar_delta = pd.to_timedelta(freq)
    slice_ns = int((bar_delta / n_slices_eff).total_seconds() * 1e9)
    bar_starts = _to_dt64_ns(bars["ts_event"]).to_numpy(dtype="datetime64[ns]", copy=False)

    # outputs (max/volatility-aware)
    out = {
        "mbp_spread_max": np.zeros(len(bars), dtype=np.float32),
        "mbp_spread_mean": np.zeros(len(bars), dtype=np.float32),
        "mbp_spread_std": np.zeros(len(bars), dtype=np.float32),
        "mbp_depth_bid_max": np.zeros(len(bars), dtype=np.float32),
        "mbp_depth_ask_max": np.zeros(len(bars), dtype=np.float32),
        "mbp_depth_sum_max": np.zeros(len(bars), dtype=np.float32),
        "mbp_imbalance_peak": np.zeros(len(bars), dtype=np.float32),
        "mbp_imbalance_direction_pct": np.zeros(len(bars), dtype=np.float32),
        "mbp_microprice_dev_max": np.zeros(len(bars), dtype=np.float32),
        "mbp_wall_bid_peak": np.zeros(len(bars), dtype=np.float32),
        "mbp_wall_ask_peak": np.zeros(len(bars), dtype=np.float32),
        "mbp_depth_shock_flag": np.zeros(len(bars), dtype=np.int8),
        "mbp_bid_slope_intrabar": np.zeros(len(bars), dtype=np.float32),
        "mbp_ask_slope_intrabar": np.zeros(len(bars), dtype=np.float32),
        "mbp_depth_accel": np.zeros(len(bars), dtype=np.float32),
    }

    # "wall" heuristic: level0 share of side depth (higher => concentrated liquidity)
    bid_wall = np.where(bid_depth > 0, (bid_sz_mat[:, 0] / bid_depth), 0.0)
    ask_wall = np.where(ask_depth > 0, (ask_sz_mat[:, 0] / ask_depth), 0.0)

    missing_bar = 0
    for i, t0 in enumerate(bar_starts):
        if pd.isna(t0):
            continue
        t1 = t0 + np.timedelta64(int(bar_delta.total_seconds() * 1e9), "ns")
        m = (ts >= t0) & (ts < t1)
        if not np.any(m):
            missing_bar += 1
            continue
        idx = np.flatnonzero(m)
        sid = _slice_ids(ts[idx], t0, slice_ns, n_slices_eff)

        # slice aggregations
        sp_s = np.zeros(n_slices_eff, dtype=np.float64)
        imb_s = np.zeros(n_slices_eff, dtype=np.float64)
        bd_s = np.zeros(n_slices_eff, dtype=np.float64)
        ad_s = np.zeros(n_slices_eff, dtype=np.float64)
        mpdev_s = np.zeros(n_slices_eff, dtype=np.float64)
        bw_s = np.zeros(n_slices_eff, dtype=np.float64)
        aw_s = np.zeros(n_slices_eff, dtype=np.float64)

        bid_slope_acc = 0.0
        ask_slope_acc = 0.0
        accel_acc = 0.0
        used_slices = 0

        for s in range(n_slices_eff):
            sm = sid == s
            if not np.any(sm):
                continue
            ids = idx[sm]
            sp_s[s] = float(np.max(spread.iloc[ids].to_numpy(dtype=np.float64, copy=False)))
            imb_s[s] = float(np.mean(imbalance[ids]))
            bd_s[s] = float(np.max(bid_depth[ids]))
            ad_s[s] = float(np.max(ask_depth[ids]))
            mpdev_s[s] = float(np.max(np.abs(microprice[ids] - mid.iloc[ids].to_numpy(dtype=np.float64, copy=False))))
            bw_s[s] = float(np.max(bid_wall[ids]))
            aw_s[s] = float(np.max(ask_wall[ids]))
            bd_slice = bid_depth[ids].astype(np.float64, copy=False)
            ad_slice = ask_depth[ids].astype(np.float64, copy=False)
            bid_slope_acc += _depth_slope(bd_slice)
            ask_slope_acc += _depth_slope(ad_slice)
            if bd_slice.size > 1:
                # اتجاهي: موجب = تراكم bid depth مقابل ask depth، سالب = سحب سيولة.
                bid_delta = float(bd_slice[-1] - bd_slice[0])
                ask_delta = float(ad_slice[-1] - ad_slice[0])
                accel_acc += (bid_delta - ask_delta)
            used_slices += 1

        denom_slices = float(max(used_slices, 1))

        out["mbp_spread_max"][i] = float(np.max(sp_s))
        out["mbp_spread_mean"][i] = float(np.mean(sp_s))
        out["mbp_spread_std"][i] = float(np.std(sp_s))
        out["mbp_depth_bid_max"][i] = float(np.max(bd_s))
        out["mbp_depth_ask_max"][i] = float(np.max(ad_s))
        out["mbp_depth_sum_max"][i] = float(np.max(bd_s + ad_s))
        out["mbp_imbalance_peak"][i] = float(np.max(np.abs(imb_s)))

        signs = np.sign(imb_s)
        nz = signs != 0
        if np.any(nz):
            dom = np.sign(np.sum(signs[nz]))
            if dom == 0:
                dom = 1.0
            out["mbp_imbalance_direction_pct"][i] = float(np.mean(signs[nz] == dom))
        else:
            out["mbp_imbalance_direction_pct"][i] = 0.0

        out["mbp_microprice_dev_max"][i] = float(np.max(mpdev_s))
        out["mbp_wall_bid_peak"][i] = float(np.max(bw_s))
        out["mbp_wall_ask_peak"][i] = float(np.max(aw_s))

        # depth shock burst: max depth is >2x mean depth
        mean_depth = float(np.mean(bd_s + ad_s)) if np.any(bd_s + ad_s) else 0.0
        peak_depth = float(np.max(bd_s + ad_s))
        out["mbp_depth_shock_flag"][i] = 1 if (mean_depth > 1e-9 and peak_depth >= 2.0 * mean_depth) else 0

        out["mbp_bid_slope_intrabar"][i] = float(bid_slope_acc / denom_slices)
        out["mbp_ask_slope_intrabar"][i] = float(ask_slope_acc / denom_slices)
        out["mbp_depth_accel"][i] = float(accel_acc / denom_slices)

    for k, v in out.items():
        bars[k] = v

    if missing_bar > 0:
        print(
            f"  ⚠️ enrich_bars_with_intrabar_mbp: {missing_bar}/{len(bars)} bars بدون أي صف MBP داخل الشمعة — "
            "ميزات mbp_* ستظل صفرًا لتلك الشموع."
        )

    return bars

