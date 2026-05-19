#!/usr/bin/env python3
"""
Leakage-aware label and feature audit for QuantSystem V19 artifacts.

The script is read-only. It does not train models. It answers:
- Are labels coherent by gate/regime/horizon?
- Do current statistical features have out-of-sample directional signal?
- Is the problem small sample size, noisy labels, or weak feature-target link?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.feature_artifact_v19 import iter_table_chunks
from modules.purging_embargo import walk_forward_expanding


DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2

CATBOOST_ADVISOR_FEATURES = [
    "cvd",
    "obi",
    "absorption_intensity",
    "cancel_ratio",
    "spoofing_ratio",
    "spoofing_duration",
    "liquidity_trap",
    "micro_atr",
    "volume_burst",
    "inter_event_time",
    "micro_price",
    "bid_wall_strength",
    "ask_wall_strength",
    "distance_to_wall",
    "gap_size",
    "liquidity_density",
    "fisher_signal",
    "anomaly",
    "cvd_momentum",
    "cvd_price_divergence",
    "trend_strength",
    "correction_depth",
    "liquidity_sweep",
    "pdh",
    "pdl",
    "dist_to_pdh",
    "price_position",
    "kyle_lambda",
    "hawkes_intensity",
    "vnet",
    "vwap_z_score",
    "cvd_prev_session",
    "cvd_session_open_delta",
    "cvd_session_zscore_causal",
    "cvd_velocity_norm_by_volume",
    "cvd_slope_3b",
    "lob_depth_imbalance",
]

LABEL_AND_META_COLS = {
    "bias_label",
    "setup_label",
    "conf_label",
    "signal_quality",
    "is_expansion",
    "event_flag",
    "train_event_flag",
    "event_score",
    "event_trigger_count",
    "raw_event_score",
    "training_event_score",
    "ts_event",
    "label_end_ts",
    "forward_return",
    "label_horizon_steps",
    "effective_horizon",
    "label_dynamic_threshold",
    "path_outcome",
    "adverse_path_flag",
    "bias_label_raw",
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
    "symbol",
    "raw_symbol",
    "instrument_id",
    "contract_symbol",
    "regime_label",
    "regime_cluster",
}

VIEW_EVENT_BINARY = "event_binary"
VIEW_DIRECTIONAL_ALL = "directional_all"
VIEW_RAW_EVENT_DIRECTIONAL = "raw_event_directional"
DEFAULT_VIEWS = [VIEW_EVENT_BINARY, VIEW_DIRECTIONAL_ALL, VIEW_RAW_EVENT_DIRECTIONAL]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"_read_error": str(exc)}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else str(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def resolve_table_path(data: str) -> Path:
    path = Path(data).expanduser().resolve()
    if path.is_dir():
        final = path / "final"
        if final.is_dir():
            return final
    return path


def resolve_root(data_path: Path) -> Path:
    if data_path.is_dir() and data_path.name.lower() == "final":
        return data_path.parent
    return data_path if data_path.is_dir() else data_path.parent


def load_frame(data: str, max_rows: int = 0) -> tuple[pd.DataFrame, Path, Path]:
    table_path = resolve_table_path(data)
    frames: list[pd.DataFrame] = []
    remaining = int(max_rows or 0)
    for chunk in iter_table_chunks(str(table_path), chunk_rows=None):
        if remaining > 0:
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)
        frames.append(chunk)
        if remaining == 0 and max_rows:
            break
    if not frames:
        raise FileNotFoundError(f"No rows loaded from {table_path}")
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    return df, table_path, resolve_root(table_path)


def stable_sort_by_ts(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_event" not in df.columns:
        return df.reset_index(drop=True)
    out = df.copy()
    out["__row_order__"] = np.arange(len(out), dtype=np.int64)
    out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    out = out.sort_values(["ts_event", "__row_order__"], kind="mergesort").drop(columns=["__row_order__"])
    return out.reset_index(drop=True)


def time_series(df: pd.DataFrame, col: str, fallback: str | None = None) -> pd.Series:
    if col in df.columns:
        s = df[col]
    elif fallback and fallback in df.columns:
        s = df[fallback]
    else:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)


def numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def quantiles(values: pd.Series | np.ndarray, qs: tuple[float, ...] = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict[str, float | None]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) == 0:
        return {f"p{int(q * 100):02d}": None for q in qs}
    return {f"p{int(q * 100):02d}": float(arr.quantile(q)) for q in qs}


def counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().to_dict().items()}


def directional_mask(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df.get("bias_label", DIR_NEUTRAL), errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8)
    return y.isin([DIR_LONG, DIR_SHORT])


def view_mask(df: pd.DataFrame, view: str) -> tuple[pd.Series, str]:
    direction = directional_mask(df)
    if view == VIEW_DIRECTIONAL_ALL:
        return direction, "bias_label"
    if view == VIEW_RAW_EVENT_DIRECTIONAL:
        event = pd.to_numeric(df.get("event_flag", 0), errors="coerce").fillna(0).astype(np.int8) == 1
        return direction & event, "event_flag"
    event_col = "train_event_flag" if "train_event_flag" in df.columns else "event_flag"
    event = pd.to_numeric(df.get(event_col, 0), errors="coerce").fillna(0).astype(np.int8) == 1
    return direction & event, event_col


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int8)
    x = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    y = y[mask]
    x = x[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    sum_pos = float(ranks[y == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    if math.isnan(auc) or math.isinf(auc):
        return None
    return float(np.clip(auc, 0.0, 1.0))


def spearman_ic(x: np.ndarray, y: np.ndarray) -> float | None:
    xs = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    ys = pd.Series(y)
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 20:
        return None
    xr = xs.loc[mask].rank(method="average")
    yr = ys.loc[mask].rank(method="average")
    corr = xr.corr(yr, method="pearson")
    if pd.isna(corr) or math.isinf(float(corr)):
        return None
    return float(corr)


def selected_features_from_file(root: Path) -> list[str]:
    path = root / "selected_features.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def choose_features(df: pd.DataFrame, root: Path, scope: str) -> list[str]:
    scope = str(scope).strip().lower()
    if scope == "catboost":
        candidates = CATBOOST_ADVISOR_FEATURES
    elif scope == "selected":
        candidates = selected_features_from_file(root) or CATBOOST_ADVISOR_FEATURES
    elif scope == "catboost_raw":
        candidates = CATBOOST_ADVISOR_FEATURES + [f"raw__{f}" for f in CATBOOST_ADVISOR_FEATURES]
    elif scope == "all_numeric":
        candidates = [
            c for c in df.columns
            if c not in LABEL_AND_META_COLS
            and not str(c).startswith("soft_label")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        raise ValueError("feature_scope must be one of: catboost, selected, catboost_raw, all_numeric")
    return [c for c in candidates if c in df.columns]


def build_splits(
    df: pd.DataFrame,
    *,
    n_folds: int,
    test_size: float,
    embargo_pct: float,
    min_train_pct: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    n = len(df)
    t0 = time_series(df, "ts_event")
    t1 = time_series(df, "label_end_ts", fallback="ts_event")
    h = numeric_series(df.loc[directional_mask(df)], "label_horizon_steps").dropna()
    dynamic_embargo_rows = max(
        int(math.ceil(n * float(embargo_pct))),
        int(math.ceil(float(np.percentile(h.to_numpy(dtype=np.float64), 95)))) if len(h) else 0,
    )
    effective_embargo_pct = float(dynamic_embargo_rows / max(n, 1))
    splits = list(
        walk_forward_expanding(
            n,
            n_folds=int(n_folds),
            test_size=float(test_size),
            embargo_pct=effective_embargo_pct,
            t0=t0,
            t1=t1,
            min_train_pct=float(min_train_pct),
        )
    )
    return splits, {
        "rows": int(n),
        "n_splits": int(len(splits)),
        "requested_n_folds": int(n_folds),
        "test_size": float(test_size),
        "min_train_pct": float(min_train_pct),
        "base_embargo_pct": float(embargo_pct),
        "dynamic_embargo_rows": int(dynamic_embargo_rows),
        "effective_embargo_pct": float(effective_embargo_pct),
    }


def label_slice_stats(df: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, Any]:
    sub = df.loc[mask].copy()
    y = pd.to_numeric(sub.get("bias_label", DIR_NEUTRAL), errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8)
    out: dict[str, Any] = {
        "slice": name,
        "rows": int(len(sub)),
        "row_pct": float(len(sub) / max(len(df), 1)),
        "label_counts": counts(y),
        "long_short_ratio": None,
        "horizon_steps": quantiles(numeric_series(sub, "label_horizon_steps")) if len(sub) else {},
        "effective_horizon": quantiles(numeric_series(sub, "effective_horizon")) if len(sub) else {},
        "dynamic_threshold": quantiles(numeric_series(sub, "label_dynamic_threshold")) if len(sub) else {},
        "event_score": quantiles(numeric_series(sub, "event_score")) if len(sub) else {},
        "forward_return": quantiles(numeric_series(sub, "forward_return")) if len(sub) else {},
    }
    long_n = int((y == DIR_LONG).sum())
    short_n = int((y == DIR_SHORT).sum())
    if long_n > 0 and short_n > 0:
        out["long_short_ratio"] = float(max(long_n, short_n) / max(min(long_n, short_n), 1))
    for col in ("path_outcome", "bias_label_detail", "neutral_reason", "signal_quality"):
        if col in sub.columns:
            out[f"{col}_counts"] = counts(sub[col])
    return out


def build_label_report(df: pd.DataFrame, views: list[str]) -> dict[str, Any]:
    ts = time_series(df, "ts_event")
    label_end = time_series(df, "label_end_ts", fallback="ts_event")
    invalid_label_end = int(((label_end < ts) & ts.notna() & label_end.notna()).sum())
    nonmonotonic_ts = int((ts.dropna().diff().dt.total_seconds() < 0).sum()) if ts.notna().any() else 0
    duplicate_ts = int(ts.duplicated(keep=False).sum()) if ts.notna().any() else 0
    base_masks: dict[str, pd.Series] = {
        "all_rows": pd.Series(True, index=df.index),
        "directional_all": directional_mask(df),
        "neutral": ~directional_mask(df),
    }
    for view in views:
        mask, _ = view_mask(df, view)
        base_masks[view] = mask
    slices = [label_slice_stats(df, name, mask) for name, mask in base_masks.items()]

    regime_cols = [c for c in ("regime_label", "regime_cluster") if c in df.columns]
    regime_rows: list[dict[str, Any]] = []
    for col in regime_cols:
        for regime_value, group in df.groupby(col, dropna=False):
            mask = group.index
            y = pd.to_numeric(group.get("bias_label", DIR_NEUTRAL), errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8)
            direction = y.isin([DIR_LONG, DIR_SHORT])
            long_n = int((y == DIR_LONG).sum())
            short_n = int((y == DIR_SHORT).sum())
            event_binary_mask, _ = view_mask(group, VIEW_EVENT_BINARY)
            regime_rows.append({
                "regime_col": col,
                "regime": str(regime_value),
                "rows": int(len(group)),
                "row_pct": float(len(group) / max(len(df), 1)),
                "directional_rows": int(direction.sum()),
                "directional_pct": float(direction.mean()) if len(group) else 0.0,
                "event_binary_rows": int(event_binary_mask.sum()),
                "label_counts": counts(y),
                "long_short_ratio": (
                    float(max(long_n, short_n) / max(min(long_n, short_n), 1))
                    if long_n > 0 and short_n > 0 else None
                ),
                "horizon_p50": quantiles(numeric_series(group, "label_horizon_steps")).get("p50"),
                "horizon_p95": quantiles(numeric_series(group, "label_horizon_steps")).get("p95"),
            })
    return {
        "rows": int(len(df)),
        "ts_event_min": str(ts.min()) if ts.notna().any() else None,
        "ts_event_max": str(ts.max()) if ts.notna().any() else None,
        "label_end_min": str(label_end.min()) if label_end.notna().any() else None,
        "label_end_max": str(label_end.max()) if label_end.notna().any() else None,
        "invalid_label_end_before_ts": invalid_label_end,
        "nonmonotonic_ts_after_load": nonmonotonic_ts,
        "duplicate_ts_rows": duplicate_ts,
        "slices": slices,
        "regime_slices": regime_rows,
    }


def compute_feature_fold_metrics(
    df: pd.DataFrame,
    features: list[str],
    views: list[str],
    *,
    n_folds: int,
    test_size: float,
    embargo_pct: float,
    min_train_pct: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    for view in views:
        mask, event_col = view_mask(df, view)
        view_df = df.loc[mask].copy().reset_index(drop=True)
        view_df = stable_sort_by_ts(view_df)
        splits, split_info = build_splits(
            view_df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
        )
        split_info["event_col"] = event_col
        split_reports[view] = split_info
        y_label = pd.to_numeric(view_df["bias_label"], errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8)
        y_long = (y_label.to_numpy(dtype=np.int8) == DIR_LONG).astype(np.int8)
        feature_arrays = {
            feature: numeric_series(view_df, feature).to_numpy(dtype=np.float64)
            for feature in features
        }
        for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)
            y_tr = y_long[train_idx]
            y_te = y_long[test_idx]
            pos = int(np.sum(y_te == 1))
            neg = int(np.sum(y_te == 0))
            if pos == 0 or neg == 0:
                continue
            for feature in features:
                x_all = feature_arrays[feature]
                x_tr = x_all[train_idx]
                x_te = x_all[test_idx]
                finite = np.isfinite(x_te)
                unique = int(len(np.unique(x_te[finite]))) if finite.any() else 0
                if unique <= 1:
                    auc = None
                    ic = None
                    train_auc = None
                    train_signed_auc = None
                    train_signal_side = None
                else:
                    train_auc = binary_auc(y_tr, x_tr)
                    train_signal_side = None if train_auc is None else ("long" if train_auc >= 0.5 else "short")
                    signed_x_te = x_te if (train_auc is not None and train_auc >= 0.5) else -x_te
                    train_signed_auc = None if train_auc is None else binary_auc(y_te, signed_x_te)
                    auc = binary_auc(y_te, x_te)
                    signed_y = np.where(y_te == 1, 1.0, -1.0)
                    ic = spearman_ic(x_te, signed_y)
                auc_edge = None if auc is None else float(max(auc, 1.0 - auc))
                fold_rows.append({
                    "view": view,
                    "fold": int(fold_no),
                    "feature": feature,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "long_rows": pos,
                    "short_rows": neg,
                    "unique_values": unique,
                    "ic": ic,
                    "abs_ic": None if ic is None else abs(float(ic)),
                    "train_auc_long": train_auc,
                    "train_signal_side": train_signal_side,
                    "train_signed_auc": train_signed_auc,
                    "auc_long": auc,
                    "auc_edge": auc_edge,
                    "signal_side": None if auc is None else ("long" if auc >= 0.5 else "short"),
                })

    summary_rows: list[dict[str, Any]] = []
    if fold_rows:
        metrics_df = pd.DataFrame(fold_rows)
        for (view, feature), group in metrics_df.groupby(["view", "feature"], dropna=False):
            valid_auc = pd.to_numeric(group["auc_edge"], errors="coerce").dropna()
            valid_ic = pd.to_numeric(group["abs_ic"], errors="coerce").dropna()
            signed_ic = pd.to_numeric(group["ic"], errors="coerce").dropna()
            train_signed_auc = pd.to_numeric(group["train_signed_auc"], errors="coerce").dropna()
            sides = group["signal_side"].dropna()
            top_side = sides.value_counts().idxmax() if len(sides) else None
            side_stability = float((sides == top_side).mean()) if len(sides) and top_side else None
            train_sides = group["train_signal_side"].dropna()
            train_top_side = train_sides.value_counts().idxmax() if len(train_sides) else None
            train_side_stability = (
                float((train_sides == train_top_side).mean())
                if len(train_sides) and train_top_side else None
            )
            summary_rows.append({
                "view": str(view),
                "feature": str(feature),
                "valid_folds": int(max(len(valid_auc), len(valid_ic))),
                "train_signed_auc_mean": float(train_signed_auc.mean()) if len(train_signed_auc) else None,
                "train_signed_auc_min": float(train_signed_auc.min()) if len(train_signed_auc) else None,
                "train_signed_auc_max": float(train_signed_auc.max()) if len(train_signed_auc) else None,
                "train_signed_auc_gt_50_pct": float((train_signed_auc > 0.5).mean()) if len(train_signed_auc) else None,
                "auc_edge_mean": float(valid_auc.mean()) if len(valid_auc) else None,
                "auc_edge_min": float(valid_auc.min()) if len(valid_auc) else None,
                "auc_edge_max": float(valid_auc.max()) if len(valid_auc) else None,
                "abs_ic_mean": float(valid_ic.mean()) if len(valid_ic) else None,
                "abs_ic_max": float(valid_ic.max()) if len(valid_ic) else None,
                "signed_ic_mean": float(signed_ic.mean()) if len(signed_ic) else None,
                "top_side": top_side,
                "side_stability": side_stability,
                "train_top_side": train_top_side,
                "train_side_stability": train_side_stability,
            })
        summary_rows.sort(
            key=lambda r: (
                -1 if r["train_signed_auc_mean"] is None else -float(r["train_signed_auc_mean"]),
                -1 if r["auc_edge_mean"] is None else -float(r["auc_edge_mean"]),
                -1 if r["abs_ic_mean"] is None else -float(r["abs_ic_mean"]),
            )
        )
    return fold_rows, summary_rows, split_reports


def build_risk_flags(label_report: dict[str, Any], feature_summary: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    if int(label_report.get("invalid_label_end_before_ts", 0)) > 0:
        flags.append("label_end_ts appears before ts_event in at least one row")
    directional = next((s for s in label_report.get("slices", []) if s.get("slice") == "directional_all"), {})
    event_binary = next((s for s in label_report.get("slices", []) if s.get("slice") == VIEW_EVENT_BINARY), {})
    if int(event_binary.get("rows", 0)) < 10_000:
        flags.append("event_binary directional training pool is very small")
    if float(directional.get("row_pct", 0.0) or 0.0) > 0.25:
        flags.append("directional label share is high; inspect whether neutral band is too narrow")
    by_view: dict[str, list[dict[str, Any]]] = {}
    for row in feature_summary:
        by_view.setdefault(str(row.get("view")), []).append(row)
    for view, rows in by_view.items():
        best_auc = max((float(r["auc_edge_mean"]) for r in rows if r.get("auc_edge_mean") is not None), default=0.5)
        best_train_signed_auc = max(
            (float(r["train_signed_auc_mean"]) for r in rows if r.get("train_signed_auc_mean") is not None),
            default=0.5,
        )
        if best_auc < 0.53:
            flags.append(f"{view}: no feature has stable fold AUC edge >= 0.53")
        if best_train_signed_auc < 0.53:
            flags.append(f"{view}: no feature has train-signed OOS AUC >= 0.53")
        if best_auc >= 0.56 and best_train_signed_auc < 0.53:
            flags.append(f"{view}: raw separability exists, but train-learned direction does not transfer")
    for row in feature_summary[:10]:
        if row.get("side_stability") is not None and float(row["side_stability"]) < 0.67:
            flags.append(f"{row['view']}/{row['feature']}: signal side flips across folds")
            break
    return flags


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
            writer.writerow({k: json_safe(row.get(k)) for k in keys})


def write_markdown(report: dict[str, Any], path: Path, top_n: int) -> None:
    label_report = report["label_report"]
    lines = [
        "# QuantSystem V19 Label/Feature Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Data: `{report['data_path']}`",
        "",
        "## Executive Summary",
        "",
        f"- Rows: `{label_report['rows']:,}`",
        f"- Timestamp range: `{label_report['ts_event_min']}` -> `{label_report['ts_event_max']}`",
        f"- Feature scope: `{report['feature_scope']}` | Features audited: `{len(report['features'])}`",
        f"- Risk flags: `{len(report['risk_flags'])}`",
        "",
    ]
    if report["risk_flags"]:
        lines.append("## Risk Flags")
        lines.append("")
        for flag in report["risk_flags"]:
            lines.append(f"- {flag}")
        lines.append("")

    lines.extend(["## Label Slices", ""])
    lines.append("| slice | rows | pct | labels | LS ratio | horizon p50 | horizon p95 |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: |")
    for row in label_report["slices"]:
        h = row.get("horizon_steps") or {}
        lines.append(
            f"| {row['slice']} | {int(row['rows']):,} | {float(row['row_pct']):.2%} | "
            f"`{row.get('label_counts', {})}` | {row.get('long_short_ratio')} | "
            f"{h.get('p50')} | {h.get('p95')} |"
        )

    lines.extend(["", "## Top Feature OOS IC/AUC", ""])
    by_view: dict[str, list[dict[str, Any]]] = {}
    for row in report["feature_summary"]:
        by_view.setdefault(str(row["view"]), []).append(row)
    for view, rows in by_view.items():
        lines.append(f"### {view}")
        lines.append("")
        lines.append("| feature | folds | train_signed_auc | train_min | auc_edge_mean | auc_edge_min | abs_ic_mean | train_side | test_edge_side |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for row in rows[:top_n]:
            lines.append(
                f"| `{row['feature']}` | {row['valid_folds']} | "
                f"{None if row.get('train_signed_auc_mean') is None else round(float(row['train_signed_auc_mean']), 4)} | "
                f"{None if row.get('train_signed_auc_min') is None else round(float(row['train_signed_auc_min']), 4)} | "
                f"{None if row.get('auc_edge_mean') is None else round(float(row['auc_edge_mean']), 4)} | "
                f"{None if row.get('auc_edge_min') is None else round(float(row['auc_edge_min']), 4)} | "
                f"{None if row.get('abs_ic_mean') is None else round(float(row['abs_ic_mean']), 4)} | "
                f"{row.get('train_top_side')} ({None if row.get('train_side_stability') is None else round(float(row['train_side_stability']), 3)}) | "
                f"{row.get('top_side')} ({None if row.get('side_stability') is None else round(float(row['side_stability']), 3)}) |"
            )
        lines.append("")

    if label_report.get("regime_slices"):
        lines.extend(["## Regime Label Slices", ""])
        lines.append("| regime_col | regime | rows | directional_pct | event_rows | labels | LS ratio |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: |")
        for row in label_report["regime_slices"]:
            lines.append(
                f"| {row['regime_col']} | {row['regime']} | {int(row['rows']):,} | "
                f"{float(row['directional_pct']):.2%} | {int(row['event_binary_rows']):,} | "
                f"`{row['label_counts']}` | {row.get('long_short_ratio')} |"
            )
        lines.append("")

    lines.extend([
        "## How To Read This",
        "",
        "- `auc_edge_mean` is `max(AUC, 1-AUC)`, so 0.50 means no directional separation and 0.55+ starts to be interesting.",
        "- `train_signed_auc` is stricter: the feature direction is chosen from the train fold only, then evaluated on the future test fold.",
        "- `abs_ic_mean` is fold-wise absolute Spearman IC against LONG(+1)/SHORT(-1).",
        "- A feature with high mean but low `side_stability` is not production-safe; it flips meaning by time.",
        "- If all views stay below ~0.53 AUC edge, inspect label construction before training deeper models.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Audit V19 labels and feature IC/AUC by chronological fold.")
    p.add_argument("--data", required=True, help="V19 final artifact dir, run dir, parquet, csv, or pickle")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated views: event_binary,directional_all,raw_event_directional")
    p.add_argument("--feature-scope", default="catboost", choices=["catboost", "selected", "catboost_raw", "all_numeric"])
    p.add_argument("--n-folds", type=int, default=9)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--embargo-pct", type=float, default=0.02)
    p.add_argument("--min-train-pct", type=float, default=0.20)
    p.add_argument("--max-rows", type=int, default=0, help="Optional head-row cap for smoke tests only")
    p.add_argument("--top-n", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    views = [v.strip() for v in str(args.views).split(",") if v.strip()]
    invalid_views = sorted(set(views) - set(DEFAULT_VIEWS))
    if invalid_views:
        raise ValueError(f"Unsupported views: {invalid_views}; valid={DEFAULT_VIEWS}")

    df, data_path, root = load_frame(args.data, max_rows=args.max_rows)
    df = stable_sort_by_ts(df)
    if "bias_label" not in df.columns:
        raise ValueError("Missing required column: bias_label")
    if "ts_event" not in df.columns:
        raise ValueError("Missing required column: ts_event")

    features = choose_features(df, root, args.feature_scope)
    if not features:
        raise ValueError(f"No features found for scope={args.feature_scope}")

    label_report = build_label_report(df, views)
    fold_metrics, feature_summary, split_reports = compute_feature_fold_metrics(
        df,
        features,
        views,
        n_folds=args.n_folds,
        test_size=args.test_size,
        embargo_pct=args.embargo_pct,
        min_train_pct=args.min_train_pct,
    )
    risk_flags = build_risk_flags(label_report, feature_summary)
    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_path": str(data_path),
        "root": str(root),
        "feature_scope": args.feature_scope,
        "features": features,
        "views": views,
        "label_report": label_report,
        "split_reports": split_reports,
        "feature_summary": feature_summary,
        "risk_flags": risk_flags,
        "source_reports": {
            "artifact_manifest": read_json(root / "artifact_manifest.json"),
            "label_quality_report": read_json(root / "label_quality_report.json"),
            "contract_consistency_report": read_json(root / "contract_consistency_report.json"),
            "data_integrity_report": read_json(root / "data_integrity_report.json"),
        },
    }

    (out_dir / "label_feature_audit.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "feature_fold_metrics.csv", fold_metrics)
    write_csv(out_dir / "feature_summary.csv", feature_summary)
    write_csv(out_dir / "label_slices.csv", label_report["slices"])
    write_csv(out_dir / "regime_slices.csv", label_report.get("regime_slices", []))
    write_markdown(report, out_dir / "label_feature_audit.md", top_n=int(args.top_n))

    print("=== QuantSystem V19 label/feature audit ===")
    print(f"rows={len(df):,} features={len(features)} views={views}")
    for row in label_report["slices"]:
        print(f"- {row['slice']}: rows={int(row['rows']):,} pct={float(row['row_pct']):.1%} labels={row['label_counts']}")
    for view in views:
        view_rows = [r for r in feature_summary if r["view"] == view]
        if view_rows:
            top = view_rows[0]
            print(
                f"- top {view}: {top['feature']} "
                f"train_signed_auc={top.get('train_signed_auc_mean')} "
                f"auc_edge_mean={top.get('auc_edge_mean')} abs_ic_mean={top.get('abs_ic_mean')}"
            )
    print(f"risk_flags={len(risk_flags)}")
    print(f"saved: {out_dir / 'label_feature_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
