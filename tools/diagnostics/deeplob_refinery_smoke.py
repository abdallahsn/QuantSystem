#!/usr/bin/env python3
"""Smoke test for the DeepLOB refinery path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return pd.read_csv(path)


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts_event"] = pd.to_datetime(out.get("ts_event"), utc=True, errors="coerce").dt.tz_localize(None)
    out = out[out["ts_event"].notna()].sort_values("ts_event").reset_index(drop=True)
    return out


def _build_targets(mbp_df: pd.DataFrame, tensor_timestamps: np.ndarray) -> np.ndarray:
    mbp = _prepare_frame(mbp_df)
    mbp["mid"] = (
        pd.to_numeric(mbp.get("bid_px_00"), errors="coerce").fillna(0.0)
        + pd.to_numeric(mbp.get("ask_px_00"), errors="coerce").fillna(0.0)
    ) / 2.0
    mbp["mid_fwd_delta"] = mbp["mid"].shift(-1) - mbp["mid"]

    ts_df = pd.DataFrame(
        {"ts_event": pd.to_datetime(pd.Series(tensor_timestamps), unit="ns", utc=True, errors="coerce").dt.tz_localize(None)}
    )
    merged = ts_df.merge(mbp[["ts_event", "mid_fwd_delta"]], on="ts_event", how="left")
    y = pd.to_numeric(merged["mid_fwd_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    if len(y) and float(np.std(y)) > 1e-8:
        y = (y - float(np.mean(y))) / (float(np.std(y)) + 1e-8)
    elif len(y):
        y = np.linspace(-1.0, 1.0, len(y), dtype=np.float32)

    return y.reshape(-1, 1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepLOB refinery smoke test")
    parser.add_argument("--mbo", default="sample_mbo.csv", help="sample or real MBO csv")
    parser.add_argument("--mbp", default="sample_mbp10.csv", help="sample or real MBP10 csv")
    parser.add_argument("--output", default="outputs_deeplob_smoke", help="output directory for artifacts")
    parser.add_argument("--time_steps", type=int, default=50, help="DeepLOB rolling window length")
    parser.add_argument("--max_tensors", type=int, default=256, help="cap the number of emitted tensors")
    parser.add_argument("--max_train_samples", type=int, default=128, help="cap training samples for speed")
    parser.add_argument("--epochs", type=int, default=1, help="auxiliary training epochs")
    parser.add_argument("--batch", type=int, default=16, help="training batch size")
    parser.add_argument("--expect_gpu", action="store_true", help="fail if TensorFlow does not see a GPU")
    parser.add_argument("--keep_output", action="store_true", help="keep prior output directory contents")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"[FAIL] TensorFlow import failed: {type(exc).__name__}: {exc}")
        return 1

    from modules.deeplob_cnn import (
        TF_AVAILABLE,
        VISUAL_EMB_DIM,
        DeepLOBCNN,
        build_lob_tensor_dataset,
    )

    if not TF_AVAILABLE:
        print("[FAIL] DeepLOB runtime is unavailable because TensorFlow could not be imported.")
        return 1

    output_dir = Path(args.output).resolve()
    if output_dir.exists() and not args.keep_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mbo_path = Path(args.mbo).resolve()
    mbp_path = Path(args.mbp).resolve()

    mbo_df = _load_frame(mbo_path)
    mbp_df = _load_frame(mbp_path)

    gpus = tf.config.list_physical_devices("GPU")
    if args.expect_gpu and not gpus:
        print("[FAIL] TensorFlow does not see any GPU, while --expect_gpu was requested.")
        return 2

    print("=" * 72)
    print("DEEPLOB REFINERY SMOKE TEST")
    print("=" * 72)
    print(f"MBO rows           : {len(mbo_df):,}")
    print(f"MBP rows           : {len(mbp_df):,}")
    print(f"TensorFlow         : {tf.__version__}")
    print(f"Built with CUDA    : {tf.test.is_built_with_cuda()}")
    print(f"Visible GPUs       : {gpus}")

    try:
        with tf.device("/GPU:0" if gpus else "/CPU:0"):
            a = tf.random.uniform((256, 256))
            b = tf.random.uniform((256, 256))
            c = tf.matmul(a, b)
            print(f"Smoke device       : {getattr(c, 'device', 'unknown')}")
    except Exception as exc:
        print(f"[FAIL] TensorFlow device smoke failed: {type(exc).__name__}: {exc}")
        return 1

    lob_path = output_dir / "lob_tensors.npy"
    ts_path = output_dir / "lob_tensor_timestamps.npy"
    model_path = output_dir / "deeplob_cnn_smoke.keras"

    print("\n[1/3] Building LOB tensors from the refinery DeepLOB path...")
    meta = build_lob_tensor_dataset(
        _prepare_frame(mbo_df),
        _prepare_frame(mbp_df),
        time_steps=int(args.time_steps),
        output_path=str(lob_path),
        timestamps_path=str(ts_path),
        max_tensors=int(args.max_tensors),
    )
    print(json.dumps(meta, indent=2))

    if int(meta.get("built_tensors", 0)) <= 0:
        print("[FAIL] No LOB tensors were built.")
        return 1

    tensors = np.load(lob_path, mmap_mode="r")
    timestamps = np.load(ts_path)
    built = min(len(tensors), len(timestamps))
    if built <= 0:
        print("[FAIL] Tensor files were written but no usable tensors were found.")
        return 1

    print(f"\n[2/3] Training DeepLOB CNN on a tiny subset ({built:,} tensors available)...")
    train_n = min(built, int(args.max_train_samples))
    if train_n < 8:
        print(f"[FAIL] Need at least 8 tensors for a meaningful smoke fit, got {train_n}.")
        return 1

    X = np.asarray(tensors[:train_n], dtype=np.float32)
    y = _build_targets(mbp_df, timestamps[:train_n])
    cnn = DeepLOBCNN(brain_file=str(model_path))
    if cnn.model is None:
        print("[FAIL] DeepLOB model could not be constructed.")
        return 1

    cnn.fit_auxiliary(
        X,
        y,
        epochs=max(1, int(args.epochs)),
        batch=max(1, min(int(args.batch), train_n)),
        output_dir=str(output_dir),
    )

    print("\n[3/3] Extracting embeddings...")
    probe_n = min(4, train_n)
    emb = np.asarray(cnn.get_embeddings(X[:probe_n]), dtype=np.float32)
    if emb.shape != (probe_n, VISUAL_EMB_DIM):
        print(f"[FAIL] Unexpected embedding shape: {emb.shape}, expected {(probe_n, VISUAL_EMB_DIM)}")
        return 1

    print(f"Embeddings shape   : {emb.shape}")
    print(f"Model artifact     : {model_path}")
    print(f"LOB tensors        : {lob_path}")
    print(f"LOB timestamps     : {ts_path}")
    print("\n[PASS] DeepLOB refinery path is working end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
