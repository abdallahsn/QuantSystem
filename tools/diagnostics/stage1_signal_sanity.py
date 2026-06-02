#!/usr/bin/env python3
"""
Chronological Stage-1 signal sanity checks for QuantSystem V19.

This is intentionally simpler than train_v19.py:
- no CatBoost/XGBoost
- no visual branch
- no meta learner
- train-only robust scaling per fold

Its job is to answer whether stable feature subsets can beat random before we
spend time on deeper training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from tools.diagnostics import audit_v19_labels_features as audit


DEFAULT_FEATURE_SETS: dict[str, list[str]] = {
    "micro_price": ["micro_price"],
    "micro_price_micro_atr": ["micro_price", "micro_atr"],
    "micro_price_walls": ["micro_price", "bid_wall_strength", "distance_to_wall"],
    "stable_core": [
        "micro_price",
        "micro_atr",
        "bid_wall_strength",
        "distance_to_wall",
        "volume_burst",
        "cvd_momentum",
    ],
    "top_audit_stable": [
        "micro_price",
        "micro_atr",
        "bid_wall_strength",
        "distance_to_wall",
        "volume_burst",
        "liquidity_density",
    ],
    "catboost_present": list(audit.CATBOOST_ADVISOR_FEATURES),
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def parse_feature_sets(spec: str | None) -> dict[str, list[str]]:
    if not spec:
        return dict(DEFAULT_FEATURE_SETS)
    out: dict[str, list[str]] = {}
    for block in str(spec).split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError("feature set spec must use name:feat1,feat2;name2:feat3")
        name, feats = block.split(":", 1)
        features = [f.strip() for f in feats.split(",") if f.strip()]
        if not features:
            raise ValueError(f"feature set {name!r} has no features")
        out[name.strip()] = features
    return out


def robust_scale_train_only(train_frame: pd.DataFrame, test_frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_train_cols: list[np.ndarray] = []
    x_test_cols: list[np.ndarray] = []
    for feature in features:
        tr = audit.numeric_series(train_frame, feature).fillna(0.0).to_numpy(dtype=np.float64)
        te = audit.numeric_series(test_frame, feature).fillna(0.0).to_numpy(dtype=np.float64)
        med = float(np.nanmedian(tr)) if len(tr) else 0.0
        q25, q75 = np.nanpercentile(tr, [25, 75]) if len(tr) else (0.0, 1.0)
        scale = float(q75 - q25)
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            scale = float(np.nanstd(tr))
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            scale = 1.0
        x_train_cols.append(np.clip((tr - med) / scale, -10.0, 10.0))
        x_test_cols.append(np.clip((te - med) / scale, -10.0, 10.0))
    return (
        np.column_stack(x_train_cols).astype(np.float32),
        np.column_stack(x_test_cols).astype(np.float32),
    )


def _threshold_metrics(y_true: np.ndarray, p_long: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for thr in np.linspace(0.40, 0.60, 41):
        pred = (p_long >= float(thr)).astype(np.int8)
        macro_f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
        if best is None or macro_f1 > float(best["macro_f1"]):
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                pred,
                labels=[1, 0],
                zero_division=0,
            )
            best = {
                "threshold": float(thr),
                "macro_f1": macro_f1,
                "accuracy": float(accuracy_score(y_true, pred)),
                "long_precision": float(precision[0]),
                "long_recall": float(recall[0]),
                "long_f1": float(f1[0]),
                "short_precision": float(precision[1]),
                "short_recall": float(recall[1]),
                "short_f1": float(f1[1]),
                "long_support": int(support[0]),
                "short_support": int(support[1]),
            }
    return best or {}


def run_sanity(
    df: pd.DataFrame,
    *,
    views: list[str],
    feature_sets: dict[str, list[str]],
    n_folds: int,
    test_size: float,
    embargo_pct: float,
    min_train_pct: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for view in views:
        mask, event_col = audit.view_mask(df, view)
        view_df = audit.stable_sort_by_ts(df.loc[mask].copy().reset_index(drop=True))
        if len(view_df) < 300:
            continue
        splits, split_info = audit.build_splits(
            view_df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
        )
        y_all = (
            pd.to_numeric(view_df["bias_label"], errors="coerce").fillna(audit.DIR_NEUTRAL).astype(np.int8)
            == audit.DIR_LONG
        ).astype(np.int8).to_numpy()
        for set_name, raw_features in feature_sets.items():
            features = [f for f in raw_features if f in view_df.columns]
            missing = [f for f in raw_features if f not in view_df.columns]
            if not features:
                continue
            for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
                train_idx = np.asarray(train_idx, dtype=np.int64)
                test_idx = np.asarray(test_idx, dtype=np.int64)
                y_train = y_all[train_idx]
                y_test = y_all[test_idx]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                x_train, x_test = robust_scale_train_only(
                    view_df.iloc[train_idx],
                    view_df.iloc[test_idx],
                    features,
                )
                model = LogisticRegression(
                    penalty="l2",
                    C=0.25,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42 + fold_no,
                    max_iter=1000,
                )
                model.fit(x_train, y_train)
                proba = model.predict_proba(x_test)
                class_order = list(model.classes_)
                long_col = class_order.index(1)
                p_long = proba[:, long_col]
                auc_long = audit.binary_auc(y_test, p_long)
                metrics = _threshold_metrics(y_test, p_long)
                fold_rows.append({
                    "view": view,
                    "event_col": event_col,
                    "feature_set": set_name,
                    "fold": int(fold_no),
                    "features": ",".join(features),
                    "missing_features": ",".join(missing),
                    "n_features": int(len(features)),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "split_rows": int(split_info["rows"]),
                    "auc_long": auc_long,
                    **metrics,
                })

    if fold_rows:
        rows_df = pd.DataFrame(fold_rows)
        for (view, feature_set), group in rows_df.groupby(["view", "feature_set"], dropna=False):
            f1 = pd.to_numeric(group["macro_f1"], errors="coerce").dropna()
            auc = pd.to_numeric(group["auc_long"], errors="coerce").dropna()
            acc = pd.to_numeric(group["accuracy"], errors="coerce").dropna()
            long_rec = pd.to_numeric(group["long_recall"], errors="coerce").dropna()
            short_rec = pd.to_numeric(group["short_recall"], errors="coerce").dropna()
            summary_rows.append({
                "view": str(view),
                "feature_set": str(feature_set),
                "folds": int(len(group)),
                "n_features": int(group["n_features"].iloc[0]),
                "features": str(group["features"].iloc[0]),
                "missing_features": str(group["missing_features"].iloc[0]),
                "macro_f1_mean": float(f1.mean()) if len(f1) else None,
                "macro_f1_min": float(f1.min()) if len(f1) else None,
                "macro_f1_max": float(f1.max()) if len(f1) else None,
                "auc_long_mean": float(auc.mean()) if len(auc) else None,
                "auc_edge_mean": float(np.maximum(auc, 1.0 - auc).mean()) if len(auc) else None,
                "accuracy_mean": float(acc.mean()) if len(acc) else None,
                "long_recall_mean": float(long_rec.mean()) if len(long_rec) else None,
                "short_recall_mean": float(short_rec.mean()) if len(short_rec) else None,
            })
        summary_rows.sort(
            key=lambda r: (
                -1 if r["macro_f1_mean"] is None else -float(r["macro_f1_mean"]),
                -1 if r["auc_edge_mean"] is None else -float(r["auc_edge_mean"]),
            )
        )
    return fold_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in keys})


def write_markdown(path: Path, report: dict[str, Any], top_n: int = 30) -> None:
    lines = [
        "# Stage1 Signal Sanity",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Data: `{report['data_path']}`",
        "",
        "## Summary",
        "",
        "| view | feature_set | folds | features | macro_f1_mean | macro_f1_min | auc_edge_mean | long_recall | short_recall |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summary"][:top_n]:
        lines.append(
            f"| {row['view']} | {row['feature_set']} | {row['folds']} | `{row['features']}` | "
            f"{None if row['macro_f1_mean'] is None else round(float(row['macro_f1_mean']), 4)} | "
            f"{None if row['macro_f1_min'] is None else round(float(row['macro_f1_min']), 4)} | "
            f"{None if row['auc_edge_mean'] is None else round(float(row['auc_edge_mean']), 4)} | "
            f"{None if row['long_recall_mean'] is None else round(float(row['long_recall_mean']), 4)} | "
            f"{None if row['short_recall_mean'] is None else round(float(row['short_recall_mean']), 4)} |"
        )
    lines.extend([
        "",
        "## How To Read This",
        "",
        "- This is not a production model; it is a chronological sanity check.",
        "- If a tiny stable feature set beats CatBoost, Stage1 is likely hurt by noisy/unstable features or weights.",
        "- If all rows stay near 0.50 macro F1, fix label/regime construction before deeper models.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Chronological Stage1 signal sanity checks.")
    p.add_argument("--data", required=True, help="V19 final artifact dir, run dir, parquet, csv, or pickle")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--views", default="event_binary,directional_all,raw_event_directional")
    p.add_argument("--feature-sets", default=None, help="Optional spec: name:f1,f2;other:f3,f4")
    p.add_argument("--n-folds", type=int, default=9)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--embargo-pct", type=float, default=0.02)
    p.add_argument("--min-train-pct", type=float, default=0.20)
    p.add_argument("--max-rows", type=int, default=0, help="Optional head-row cap for smoke tests only")
    args = p.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    views = [v.strip() for v in str(args.views).split(",") if v.strip()]
    invalid = sorted(set(views) - set(audit.DEFAULT_VIEWS))
    if invalid:
        raise ValueError(f"Unsupported views: {invalid}; valid={audit.DEFAULT_VIEWS}")
    feature_sets = parse_feature_sets(args.feature_sets)

    df, data_path, _ = audit.load_frame(args.data, max_rows=args.max_rows)
    df = audit.stable_sort_by_ts(df)
    fold_rows, summary_rows = run_sanity(
        df,
        views=views,
        feature_sets=feature_sets,
        n_folds=args.n_folds,
        test_size=args.test_size,
        embargo_pct=args.embargo_pct,
        min_train_pct=args.min_train_pct,
    )
    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_path": str(data_path),
        "views": views,
        "feature_sets": feature_sets,
        "summary": summary_rows,
        "folds": fold_rows,
    }
    (out_dir / "stage1_signal_sanity.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "stage1_signal_sanity_summary.csv", summary_rows)
    write_csv(out_dir / "stage1_signal_sanity_folds.csv", fold_rows)
    write_markdown(out_dir / "stage1_signal_sanity.md", report)

    print("=== QuantSystem V19 Stage1 signal sanity ===")
    for row in summary_rows[:12]:
        print(
            f"- {row['view']} / {row['feature_set']}: "
            f"macro_f1={row['macro_f1_mean']} auc_edge={row['auc_edge_mean']} "
            f"features={row['features']}"
        )
    print(f"saved: {out_dir / 'stage1_signal_sanity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
