#!/usr/bin/env python3
"""
Pre-training checks for prepare_day_trading.py output.

Usage:
  py -3.13 verify_day_trading_dataset.py --data pipeline_.../day_trading_features.parquet
  py -3.13 verify_day_trading_dataset.py --data ... --lob .../lob_tensors.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Columns train_v19.load_training_csv hard-requires (minimal gate)
REQUIRED_FOR_TRAIN = (
    "ts_event",
    "label_end_ts",
    "bias_label",
    "soft_label",
    "label_confidence",
    "soft_sample_weight",
    "event_flag",
    "train_event_flag",
    "signal_quality",
    "forward_return",
)

# prepare_day_trading adds raw__* for CATBOOST_ADVISOR_FEATURES_DT (32)
EXPECTED_RAW_PREFIX = "raw__"
EXPECTED_RAW_COUNT = 32
MIN_RELIABLE_TRAIN_ROWS_WARN = 1000
LABEL_DERIVED_FEATURES = (
    "bias_label",
    "conf_label",
    "signal_quality",
    "forward_return",
    "label_horizon_steps",
    "effective_horizon",
    "label_dynamic_threshold",
    "path_outcome",
    "adverse_path_flag",
    "bias_label_detail",
    "neutral_reason",
    "timeout_move_exceeded_band",
    "soft_label",
    "label_confidence",
    "soft_label_long",
    "soft_label_short",
    "soft_sample_weight",
    "mc_sample_weight",
    "label_stability",
)

INTRABAR_MBO = (
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
)

INTRABAR_MBP = (
    "mbp_spread_max",
    "mbp_spread_mean",
    "mbp_spread_std",
    "mbp_depth_bid_max",
    "mbp_depth_ask_max",
    "mbp_depth_sum_max",
    "mbp_imbalance_peak",
    "mbp_imbalance_direction_pct",
    "mbp_microprice_dev_max",
    "mbp_wall_bid_peak",
    "mbp_wall_ask_peak",
    "mbp_depth_shock_flag",
)

INTRABAR_MBP_STRONG = (
    "mbp_imbalance_signed_peak",
    "mbp_imbalance_mean",
)


def _series_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    finite = np.isfinite(x.to_numpy(dtype=np.float64, na_value=np.nan))
    n = len(x)
    n_null = int(x.isna().sum())
    n_fin = int(finite.sum())
    arr = x.to_numpy(dtype=np.float64, copy=False)
    arr = arr[finite]
    if arr.size == 0:
        return {"n": n, "null_pct": n_null / max(n, 1), "zero_pct": 1.0, "min": None, "max": None, "mean": None}
    zero_pct = float(np.mean(np.abs(arr) < 1e-12)) if arr.size else 1.0
    return {
        "n": n,
        "null_pct": n_null / max(n, 1),
        "zero_pct": zero_pct,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Verify day_trading_features.parquet (+ optional LOB) before train_v19.")
    p.add_argument("--data", required=True, help="Path to day_trading_features.parquet")
    p.add_argument("--lob", default=None, help="Path to lob_tensors.npy (optional)")
    p.add_argument("--lob_ts", default=None, help="Path to lob_tensor_timestamps.npy (default: next to --lob)")
    args = p.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        print(f"❌ Missing parquet: {data_path}")
        return 1

    print(f"📥 Loading: {data_path}")
    df = pd.read_parquet(data_path)
    n, ncol = len(df), len(df.columns)
    print(f"   Rows: {n:,} | Columns: {ncol}")
    if n < MIN_RELIABLE_TRAIN_ROWS_WARN:
        print(
            f"⚠️ Rows below {MIN_RELIABLE_TRAIN_ROWS_WARN:,}: OK for smoke/refinery checks, "
            "not enough for reliable full training."
        )

    ok = True

    missing = [c for c in REQUIRED_FOR_TRAIN if c not in df.columns]
    if missing:
        ok = False
        print(f"❌ Missing required columns for train_v19: {missing}")
    else:
        print("✅ Required train_v19 columns present")

    raw_cols = [c for c in df.columns if c.startswith(EXPECTED_RAW_PREFIX)]
    if len(raw_cols) < EXPECTED_RAW_COUNT:
        ok = False
        print(
            f"⚠️ Expected at least {EXPECTED_RAW_COUNT} {EXPECTED_RAW_PREFIX}* columns; "
            f"found {len(raw_cols)} (training may warn about missing raw stats)."
        )
    else:
        print(f"✅ raw__* columns: {len(raw_cols)}")

    leaked_raw = sorted(
        c for c in raw_cols
        if c[len(EXPECTED_RAW_PREFIX):] in set(LABEL_DERIVED_FEATURES)
    )
    if leaked_raw:
        ok = False
        print(f"❌ raw__ feature surface contains label-derived columns: {leaked_raw}")
    else:
        print("✅ raw__ feature surface excludes label-derived columns")

    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        if ts.isna().any():
            ok = False
            print(f"❌ ts_event has {int(ts.isna().sum())} invalid timestamps")
        dup = int(ts.duplicated().sum())
        if dup:
            ok = False
            print(f"❌ Duplicate ts_event rows: {dup}")
        if not ts.is_monotonic_increasing:
            print("⚠️ ts_event not strictly increasing — train_v19 will sort; better to sort parquet once.")
        else:
            print("✅ ts_event: no duplicates, monotonic increasing")

    if "bias_label" in df.columns:
        vc = df["bias_label"].value_counts().sort_index()
        print(f"✅ bias_label distribution:\n{vc.to_string()}")

    def _check_group(name: str, cols: tuple[str, ...]) -> None:
        nonlocal ok
        present = [c for c in cols if c in df.columns]
        absent = [c for c in cols if c not in df.columns]
        if absent:
            print(f"⚠️ {name}: missing {len(absent)}/{len(cols)} → {absent[:6]}{'...' if len(absent) > 6 else ''}")
            if name == "MBP intrabar":
                ok = False
            return
        print(f"✅ {name}: all {len(cols)} columns present")
        # quick signal check: not all zeros / not all NaN
        bad = []
        for c in present:
            st = _series_stats(df[c])
            if st["null_pct"] > 0.99:
                bad.append(f"{c}(all_null)")
            elif st["zero_pct"] > 0.999 and c not in ("mbp_depth_shock_flag", "spoof_burst_flag"):
                bad.append(f"{c}(~all_zero)")
        if bad:
            print(f"⚠️ {name}: weak/degenerate columns: {bad[:8]}{'...' if len(bad) > 8 else ''}")
        else:
            print(f"   (sanity: no column ~all-null; spread/max features not ~all-zero)")

    _check_group("Intrabar MBO", INTRABAR_MBO)
    _check_group("MBP intrabar", INTRABAR_MBP)

    strong_missing = [c for c in INTRABAR_MBP_STRONG if c not in df.columns]
    if strong_missing:
        print(f"⚠️ MBP signed imbalance columns missing: {strong_missing}")
    else:
        print("✅ MBP signed imbalance columns present")

    if {"lob_imbalance", "order_flow_imbalance"}.issubset(df.columns):
        lob = pd.to_numeric(df["lob_imbalance"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ofi = pd.to_numeric(df["order_flow_imbalance"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = lob.notna() & ofi.notna()
        corr = float("nan")
        if int(valid.sum()) >= 3:
            corr = float(pd.DataFrame({"lob": lob[valid], "ofi": ofi[valid]}).corr(method="spearman").iloc[0, 1])
        depth_rate = None
        if "lob_imbalance_is_depth" in df.columns:
            depth_rate = float(
                pd.to_numeric(df["lob_imbalance_is_depth"], errors="coerce").fillna(0.0).astype(float).mean()
            )
        cov_mean = None
        if "mbp_bar_coverage" in df.columns:
            cov_mean = float(pd.to_numeric(df["mbp_bar_coverage"], errors="coerce").fillna(0.0).mean())
        print(
            "✅ LOB imbalance source sanity: "
            f"spearman_vs_order_flow={corr:.4f} | "
            f"depth_rate={depth_rate if depth_rate is not None else 'n/a'} | "
            f"mbp_cov_mean={cov_mean if cov_mean is not None else 'n/a'}"
        )
        if (
            depth_rate is not None
            and cov_mean is not None
            and cov_mean > 0.30
            and depth_rate < 0.50
        ):
            ok = False
            print("❌ MBP coverage exists, but lob_imbalance mostly uses trade-flow fallback.")
        elif (
            depth_rate is None
            and cov_mean is not None
            and cov_mean > 0.30
            and np.isfinite(corr)
            and abs(corr) > 0.999
        ):
            ok = False
            print("❌ lob_imbalance is indistinguishable from order_flow_imbalance despite MBP coverage.")

    lob_path = args.lob
    if lob_path:
        lob_path = os.path.abspath(lob_path)
        ts_path = args.lob_ts
        if ts_path is None:
            ts_path = os.path.join(os.path.dirname(lob_path), "lob_tensor_timestamps.npy")
        ts_path = os.path.abspath(ts_path)

        if not os.path.isfile(lob_path):
            ok = False
            print(f"❌ LOB file missing: {lob_path}")
        else:
            tensors = np.load(lob_path, mmap_mode="r")
            print(f"✅ LOB tensors: shape={tensors.shape} dtype={tensors.dtype}")
            if tensors.shape[0] != n:
                ok = False
                print(f"❌ LOB row count {tensors.shape[0]} != parquet rows {n}")
            if len(tensors.shape) != 4:
                ok = False
                print(f"❌ Expected 4D LOB (N, 50, 20, 3); got {len(tensors.shape)}D")
            arr = np.asarray(tensors[: min(4096, tensors.shape[0])], dtype=np.float64)
            if not np.isfinite(arr).all():
                ok = False
                print("❌ LOB sample contains NaN/Inf")
            else:
                print(f"   LOB sample finite: min={arr.min():.6g} max={arr.max():.6g}")
                if arr.size:
                    ch_std = arr.reshape(-1, arr.shape[-1]).std(axis=0) if arr.ndim == 4 else np.array([])
                    per_tensor_std = arr.reshape(arr.shape[0], -1).std(axis=1) if arr.ndim >= 2 else np.array([])
                    collapsed_ratio = float(np.mean(per_tensor_std < 1e-8)) if per_tensor_std.size else 1.0
                    print(
                        "   LOB sample channel std: "
                        f"{np.array2string(ch_std, precision=6)} | collapsed_tensor_ratio={collapsed_ratio:.2%}"
                    )
                    if ch_std.size and bool(np.all(ch_std < 1e-8)):
                        ok = False
                        print("❌ LOB sample channels are collapsed/all-flat")
                    elif collapsed_ratio > 0.95:
                        ok = False
                        print("❌ Most sampled LOB tensors are collapsed/all-flat")

        if not os.path.isfile(ts_path):
            ok = False
            print(f"❌ LOB timestamps missing: {ts_path}")
        else:
            lob_ts = np.load(ts_path)
            print(f"✅ LOB timestamps: len={lob_ts.shape[0]} dtype={lob_ts.dtype}")
            if lob_ts.shape[0] != n:
                ok = False
                print(f"❌ LOB ts length {lob_ts.shape[0]} != parquet rows {n}")
            elif "ts_event" in df.columns:
                expected_ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
                expected_ns = expected_ts.astype("datetime64[ns]").astype("int64").to_numpy()
                lob_ns = lob_ts.astype("int64", copy=False)
                exact_ratio = float(np.mean(lob_ns == expected_ns)) if n else 1.0
                print(f"   LOB timestamp exact alignment: {exact_ratio:.2%}")
                if exact_ratio < 0.99:
                    ok = False
                    print("❌ LOB timestamps are not aligned to day_trading_features.ts_event")

    if ok:
        print("\n✅ Pre-training verification passed.")
        return 0
    print("\n❌ Pre-training verification failed — fix issues above before train_v19.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
