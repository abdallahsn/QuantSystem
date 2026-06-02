#!/usr/bin/env python3
"""
يبحث تحت مجلد المشروع عن ملفات train_v19 / المصفاة / Day Trading.

Usage:
  py -3.13 find_training_artifacts.py
  py -3.13 find_training_artifacts.py "E:\\path\\to\\QuantSystem-master"
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

# أسماء نبحث عنها (مسارات نسبية من الجذر)
KEY_NAMES = {
    "lob_tensors.npy",
    "lob_tensor_timestamps.npy",
    "day_trading_features.parquet",
    "training_features_ready.parquet",
    "training_features_ready.csv",
    "artifact_manifest.json",
    "refinery_split.json",
    "day_trading_manifest.json",
    "selected_features.txt",
    "refinery_report.txt",
    "lob_build_meta.json",
    "final_feature_shards.json",
}

# مجلدات مميزة
KEY_DIR_MARKERS = ("final", "mbo_final")


def main() -> int:
    ap = argparse.ArgumentParser(description="Locate V19 training-related files under a root.")
    ap.add_argument(
        "root",
        nargs="?",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root folder to scan (default: directory containing this script)",
    )
    ap.add_argument("--max-files", type=int, default=5000, help="Stop after reporting this many matches per name")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[ERR] Not a directory: {root}")
        return 1

    print(f"[SCAN] {root}\n")

    by_name: dict[str, list[str]] = defaultdict(list)
    final_dirs: list[str] = []
    mbo_final_dirs: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # تقليل الضوضاء: تخطي أشجار ثقيلة إن رغبت لاحقاً
        base = os.path.basename(dirpath)
        if base in ("__pycache__", ".git", ".venv", "venv", "node_modules"):
            dirnames[:] = []
            continue
        if base == "final" and "final" in KEY_DIR_MARKERS:
            final_dirs.append(dirpath)
        if base == "mbo_final":
            mbo_final_dirs.append(dirpath)

        for fn in filenames:
            if fn in KEY_NAMES:
                full = os.path.join(dirpath, fn)
                if len(by_name[fn]) < args.max_files:
                    by_name[fn].append(full)

    def section(title: str) -> None:
        print("-" * 60)
        print(title)
        print("-" * 60)

    section("Key files (full paths)")
    for name in sorted(KEY_NAMES):
        paths = by_name.get(name, [])
        if not paths:
            print(f"  [MISS] {name}")
        else:
            print(f"  [OK] {name}  ({len(paths)} path(s))")
            for p in paths[:15]:
                print(f"      {p}")
            if len(paths) > 15:
                print(f"      ... +{len(paths) - 15} more")

    section("final/ directories (refinery shards)")
    if not final_dirs:
        print("  (no 'final' dirs)")
    else:
        for p in final_dirs[:30]:
            parquets = [f for f in os.listdir(p) if f.endswith(".parquet")]
            print(f"  [DIR final] {p}  ({len(parquets)} parquet)")
        if len(final_dirs) > 30:
            print(f"  ... +{len(final_dirs) - 30} more")

    section("mbo_final/ directories")
    if not mbo_final_dirs:
        print("  (no mbo_final dirs)")
    else:
        for p in mbo_final_dirs[:30]:
            parquets = [f for f in os.listdir(p) if f.endswith(".parquet")]
            print(f"  [DIR mbo_final] {p}  ({len(parquets)} parquet)")
        if len(mbo_final_dirs) > 30:
            print(f"  ... +{len(mbo_final_dirs) - 30} more")

    section("train_v19 expects")
    print("  - --data: parquet path OR folder with final/ OR manifest with shards")
    print("  - full phase: lob_tensors.npy + lob_tensor_timestamps.npy under artifact root OR --lob / --lob_ts")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
