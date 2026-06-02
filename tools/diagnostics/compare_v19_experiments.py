#!/usr/bin/env python3
"""
Compare QuantSystem V19 refinery/training experiment folders.

Example:
    python compare_v19_experiments.py \
        outputs_cleaned_v2 outputs_cleaned_v2_geom outputs_cleaned_v2_more_events \
        --out compare_v19_runs --scan-final

The script is intentionally read-only. It summarizes the artifacts that are
already produced by prepare_training_data.py and train_v19.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any


IMPORTANT_PARQUET_COLUMNS = [
    "ts_event",
    "label_end_ts",
    "bias_label",
    "signal_quality",
    "event_flag",
    "train_event_flag",
    "raw_event_score",
    "training_event_score",
    "soft_label",
    "soft_sample_weight",
    "mc_sample_weight",
    "label_stability",
    "label_horizon_steps",
    "path_outcome",
    "neutral_reason",
    "timeout_move_exceeded_band",
    "regime_label",
    "symbol",
    "raw_symbol",
    "instrument_id",
    "contract_symbol",
]


CSV_COLUMNS = [
    "run",
    "root",
    "kind",
    "sample_weight_mode",
    "stage1_target",
    "rows_full",
    "rows_event",
    "event_view_pct",
    "label_long",
    "label_short",
    "label_neutral",
    "directional_pct",
    "long_short_ratio",
    "raw_event_target_rate",
    "training_event_target_rate",
    "direction_threshold_ticks",
    "tp_mult",
    "sl_mult",
    "label_horizon",
    "stage1_coverage_ratio",
    "catboost_f1_mean",
    "catboost_f1_min",
    "catboost_best_iter_median",
    "xgboost_f1_mean",
    "xgboost_f1_min",
    "meta_train_sequences",
    "meta_val_sequences",
    "visual_coverage_ratio",
    "visual_seq_coverage",
    "compact_profile",
    "meta_best_epoch",
    "meta_best_val_loss",
    "meta_last_val_loss",
    "meta_overfit_delta",
    "bias_threshold",
    "threshold_macro_f1",
    "threshold_long_recall",
    "threshold_short_recall",
    "holdout_accuracy",
    "holdout_macro_f1",
    "holdout_long_f1",
    "holdout_short_f1",
    "holdout_long_recall",
    "holdout_short_recall",
    "holdout_rows",
    "contract_passed",
    "integrity_warning_count",
    "dead_features_count",
    "elapsed_seconds",
    "risk_level",
    "top_issue",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"_read_error": str(exc)}


def first_existing_json(paths: list[Path]) -> tuple[dict[str, Any], str | None]:
    for path in paths:
        if path.exists():
            return read_json(path), str(path.parent)
    return {}, None


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def dig(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def pct(numerator: Any, denominator: Any) -> float | None:
    n = as_float(numerator)
    d = as_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def mean(values: list[Any]) -> float | None:
    xs = [as_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def median(values: list[Any]) -> float | None:
    xs = [as_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def min_value(values: list[Any]) -> float | None:
    xs = [as_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    return float(min(xs)) if xs else None


def normalize_run_root(path: str) -> Path:
    root = Path(path).expanduser()
    if root.name.lower() == "final":
        root = root.parent
    return root.resolve()


def resolve_runs(args_runs: list[str], patterns: list[str]) -> list[Path]:
    seen: set[str] = set()
    roots: list[Path] = []
    for item in [*args_runs, *patterns]:
        matches = glob.glob(item)
        candidates = matches if matches else [item]
        for candidate in candidates:
            root = normalize_run_root(candidate)
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            roots.append(root)
    return roots


def artifact_root_from_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    try:
        path = Path(str(path_value)).expanduser()
    except Exception:
        return None
    if path.name.lower() == "final":
        return path.parent.resolve()
    if path.parent.name.lower() == "final":
        return path.parent.parent.resolve()
    if path.suffix.lower() == ".parquet":
        return path.parent.resolve()
    return path.resolve() if path.is_dir() else path.parent.resolve()


def candidate_report_roots(root: Path, train_manifest: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        resolved = candidate.resolve()
        if resolved not in roots:
            roots.append(resolved)

    add(root)
    inputs = train_manifest.get("inputs") or {}
    for key in ["csv", "data", "artifact", "features"]:
        add(artifact_root_from_path(inputs.get(key)))
    for key in ["lob", "lob_ts"]:
        lob_root = artifact_root_from_path(inputs.get(key))
        if lob_root is not None:
            if lob_root.name.lower() == "features":
                add(lob_root)
                add(lob_root.parent)
            else:
                add(lob_root)
    return roots


def read_report_from_roots(roots: list[Path], filename: str) -> tuple[dict[str, Any], str | None]:
    return first_existing_json([root / filename for root in roots])


def collect_values_by_key(obj: Any, key: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out.extend(collect_values_by_key(v, key))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(collect_values_by_key(item, key))
    return out


def summarize_label_quality(report: dict[str, Any]) -> dict[str, Any]:
    months = report.get("months") or []
    result = {
        "rows": as_int(report.get("rows"), 0) or 0,
        "class_counts": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
        "path_outcome_counts": {},
        "directional_rows": 0,
        "directional_pct": None,
        "long_short_ratio": None,
        "adverse_path_share_weighted": None,
        "timeout_move_exceeded_band_share_weighted": None,
        "months": len(months),
    }
    adverse_num = 0.0
    timeout_band_num = 0.0
    rows_for_rates = 0
    for month in months:
        rows = as_int(month.get("rows"), 0) or 0
        counts = month.get("class_counts") or {}
        for label in ["LONG", "SHORT", "NEUTRAL"]:
            result["class_counts"][label] += as_int(counts.get(label), 0) or 0
        for key, value in (month.get("path_outcome_counts") or {}).items():
            result["path_outcome_counts"][key] = (
                result["path_outcome_counts"].get(key, 0) + (as_int(value, 0) or 0)
            )
        adverse = as_float(month.get("adverse_path_share"))
        timeout_band = as_float(month.get("timeout_move_exceeded_band_share"))
        if rows > 0:
            if adverse is not None:
                adverse_num += adverse * rows
            if timeout_band is not None:
                timeout_band_num += timeout_band * rows
            rows_for_rates += rows
    if not result["rows"]:
        result["rows"] = sum(result["class_counts"].values())
    long_rows = result["class_counts"]["LONG"]
    short_rows = result["class_counts"]["SHORT"]
    directional = long_rows + short_rows
    result["directional_rows"] = directional
    result["directional_pct"] = pct(directional, result["rows"])
    result["long_short_ratio"] = (
        max(long_rows, short_rows) / max(min(long_rows, short_rows), 1)
        if directional
        else None
    )
    if rows_for_rates:
        result["adverse_path_share_weighted"] = adverse_num / rows_for_rates
        result["timeout_move_exceeded_band_share_weighted"] = timeout_band_num / rows_for_rates
    return result


CLASS_ROW_RE = re.compile(
    r"^\s*(LONG|SHORT|NEUTRAL|macro avg|weighted avg)\s+"
    r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$"
)
ACC_ROW_RE = re.compile(r"^\s*accuracy\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$")
THRESHOLD_RE = re.compile(r"Bias LONG threshold:\s*([0-9]*\.?[0-9]+)")


def parse_classification_report(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not text:
        return result
    threshold_match = THRESHOLD_RE.search(text)
    if threshold_match:
        result["bias_long_threshold"] = as_float(threshold_match.group(1))
    for line in text.splitlines():
        m = CLASS_ROW_RE.match(line)
        if m:
            label = m.group(1).replace(" ", "_").lower()
            result[label] = {
                "precision": as_float(m.group(2)),
                "recall": as_float(m.group(3)),
                "f1": as_float(m.group(4)),
                "support": as_int(m.group(5)),
            }
            continue
        m = ACC_ROW_RE.match(line)
        if m:
            result["accuracy"] = as_float(m.group(1))
            result["support"] = as_int(m.group(2))
    return result


def summarize_history(history_report: dict[str, Any]) -> dict[str, Any]:
    hist = history_report.get("history") or {}
    val_loss = [as_float(v) for v in hist.get("val_loss", [])]
    val_loss = [v for v in val_loss if v is not None]
    loss = [as_float(v) for v in hist.get("loss", [])]
    loss = [v for v in loss if v is not None]
    out: dict[str, Any] = {
        "epochs_ran": len(val_loss) or len(loss),
        "best_epoch": None,
        "best_val_loss": None,
        "last_val_loss": val_loss[-1] if val_loss else None,
        "last_train_loss": loss[-1] if loss else None,
        "overfit_delta": None,
        "confidence_head_enabled": history_report.get("confidence_head_enabled"),
        "confidence_target_std": as_float(history_report.get("confidence_target_std")),
        "bias_long_threshold": as_float(history_report.get("bias_long_threshold")),
        "threshold_metrics": history_report.get("threshold_metrics") or {},
        "split": history_report.get("split") or {},
        "profile": history_report.get("profile") or {},
    }
    if val_loss:
        best_val = min(val_loss)
        best_idx = val_loss.index(best_val)
        out["best_epoch"] = best_idx + 1
        out["best_val_loss"] = best_val
        out["overfit_delta"] = val_loss[-1] - best_val
    return out


def read_parquet_scan(root: Path) -> dict[str, Any]:
    final_path = root / "final"
    if final_path.is_file() and final_path.suffix.lower() == ".parquet":
        parquet_files = [final_path]
    elif final_path.is_dir():
        parquet_files = sorted(final_path.glob("*.parquet"))
    else:
        parquet_files = sorted(root.glob("*.parquet"))
    if not parquet_files:
        return {}
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        return {"_scan_error": f"pandas unavailable: {exc}"}

    frames = []
    total_files = 0
    for path in parquet_files:
        total_files += 1
        columns = None
        try:
            import pyarrow.parquet as pq  # type: ignore

            names = set(pq.ParquetFile(path).schema.names)
            columns = [c for c in IMPORTANT_PARQUET_COLUMNS if c in names]
        except Exception:
            columns = None
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception:
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:
                return {"_scan_error": f"failed reading {path}: {exc}"}
        frames.append(frame)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    out: dict[str, Any] = {"parquet_files": total_files, "rows": int(len(df))}

    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], errors="coerce")
        out["ts_min"] = str(ts.min()) if ts.notna().any() else None
        out["ts_max"] = str(ts.max()) if ts.notna().any() else None
        out["missing_ts_rows"] = int(ts.isna().sum())
        out["duplicate_ts_rows"] = int(ts.duplicated(keep=False).sum())
        out["non_monotonic_ts_steps"] = int((ts.diff().dt.total_seconds().fillna(0) < 0).sum())

    for col in ["bias_label", "path_outcome", "neutral_reason", "signal_quality", "regime_label"]:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False).sort_index()
            out[f"{col}_counts"] = {str(k): int(v) for k, v in counts.items()}

    for col in ["event_flag", "train_event_flag", "timeout_move_exceeded_band"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            out[f"{col}_rows"] = int((values > 0).sum())
            out[f"{col}_rate"] = float((values > 0).mean()) if len(values) else 0.0

    for col in ["soft_label", "soft_sample_weight", "mc_sample_weight", "label_stability", "label_horizon_steps"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            finite = values.dropna()
            if len(finite):
                out[f"{col}_mean"] = float(finite.mean())
                out[f"{col}_median"] = float(finite.median())
                out[f"{col}_min"] = float(finite.min())
                out[f"{col}_max"] = float(finite.max())
                if col == "soft_label":
                    out["soft_label_near_0_5_rate"] = float((abs(finite - 0.5) <= 0.02).mean())

    symbol_col = next((c for c in ["symbol", "raw_symbol", "contract_symbol", "instrument_id"] if c in df.columns), None)
    if symbol_col:
        counts = df[symbol_col].fillna("UNKNOWN").astype(str).value_counts()
        out["symbol_source_col"] = symbol_col
        out["symbol_counts"] = {str(k): int(v) for k, v in counts.head(10).items()}
    return out


def summarize_stage1(stage1: dict[str, Any]) -> dict[str, Any]:
    cb_folds = stage1.get("catboost_folds") or []
    xgb_folds = stage1.get("xgboost_folds") or []
    return {
        "coverage_ratio": as_float(stage1.get("coverage_ratio")),
        "catboost_coverage_ratio": as_float(stage1.get("catboost_coverage_ratio")),
        "xgboost_coverage_ratio": as_float(stage1.get("xgboost_coverage_ratio")),
        "regime_coverage_ratio": as_float(stage1.get("regime_coverage_ratio")),
        "catboost_f1_mean": mean([f.get("directional_f1") for f in cb_folds if isinstance(f, dict)]),
        "catboost_f1_min": min_value([f.get("directional_f1") for f in cb_folds if isinstance(f, dict)]),
        "catboost_best_iter_median": median([f.get("best_iteration") for f in cb_folds if isinstance(f, dict)]),
        "catboost_best_iter_min": min_value([f.get("best_iteration") for f in cb_folds if isinstance(f, dict)]),
        "xgboost_f1_mean": mean([f.get("directional_f1") for f in xgb_folds if isinstance(f, dict)]),
        "xgboost_f1_min": min_value([f.get("directional_f1") for f in xgb_folds if isinstance(f, dict)]),
        "xgboost_best_iter_median": median([f.get("best_iteration") for f in xgb_folds if isinstance(f, dict)]),
        "fold_count_catboost": len(cb_folds),
        "fold_count_xgboost": len(xgb_folds),
        "coverage_warning": stage1.get("coverage_warning") or {},
        "fold_zero_pct_summary": stage1.get("fold_zero_pct_summary") or {},
    }


def collect_one_run(root: Path, scan_final: bool) -> dict[str, Any]:
    train_manifest = read_json(root / "manifest.json")
    report_roots = candidate_report_roots(root, train_manifest)
    artifact_manifest, artifact_report_root = read_report_from_roots(report_roots, "artifact_manifest.json")
    label_quality, label_report_root = read_report_from_roots(report_roots, "label_quality_report.json")
    data_integrity, integrity_report_root = read_report_from_roots(report_roots, "data_integrity_report.json")
    contract, contract_report_root = read_report_from_roots(report_roots, "contract_consistency_report.json")
    feature_drift, drift_report_root = read_report_from_roots(report_roots, "feature_coverage_drift_report.json")
    stage1 = read_json(root / "stage1_v19_metrics.json")
    calibration = read_json(root / "calibration_report.json")
    time_split = read_json(root / "time_split_report.json")
    meta_history_raw = read_json(root / "meta_learner_v19_history.json")
    feature_schema = read_json(root / "feature_schema_v19.json")
    refinery_split = read_json(root / "refinery_split.json")
    refinery_timing = read_json(root / "refinery_timing.json")
    lob_build_meta = read_json(root / "lob_build_meta.json")
    day_trading = read_json(root / "day_trading_manifest.json")

    label_summary = summarize_label_quality(label_quality)
    stage1_summary = summarize_stage1(stage1)
    history_summary = summarize_history(meta_history_raw)
    holdout_text = read_text(root / "meta_learner_holdout_report.txt") or read_text(root / "meta_learner_report.txt")
    holdout_report = parse_classification_report(holdout_text)
    final_scan = read_parquet_scan(root) if scan_final else {}

    metrics = train_manifest.get("metrics") or {}
    config = train_manifest.get("config") or {}
    artifact_config = artifact_manifest.get("config") or {}
    meta_profile = metrics.get("meta_learner_profile") or dig(train_manifest, "extra", "meta_learner_profile", default={}) or {}
    training_window = metrics.get("training_window") or dig(train_manifest, "extra", "training_window", default={}) or {}
    threshold_metrics = history_summary.get("threshold_metrics") or {}
    split = history_summary.get("split") or {}

    rows_full = as_int(metrics.get("rows_full")) or as_int(final_scan.get("rows")) or label_summary.get("rows")
    rows_event = as_int(metrics.get("rows_event"))
    event_view_pct = pct(rows_event, rows_full)
    class_counts = label_summary["class_counts"]
    long_rows = class_counts["LONG"]
    short_rows = class_counts["SHORT"]
    neutral_rows = class_counts["NEUTRAL"]

    final_macro = dig(holdout_report, "macro_avg", "f1")
    final_accuracy = holdout_report.get("accuracy")
    bias_threshold = (
        holdout_report.get("bias_long_threshold")
        if holdout_report.get("bias_long_threshold") is not None
        else history_summary.get("bias_long_threshold")
    )

    row = {
        "run": root.name,
        "root": str(root),
        "source_report_root": label_report_root or artifact_report_root,
        "kind": train_manifest.get("kind") or artifact_manifest.get("kind"),
        "sample_weight_mode": metrics.get("sample_weight_mode") or config.get("sample_weight_mode"),
        "stage1_target": metrics.get("stage1_target") or config.get("stage1_target"),
        "rows_full": rows_full,
        "rows_event": rows_event,
        "event_view_pct": event_view_pct,
        "label_long": long_rows,
        "label_short": short_rows,
        "label_neutral": neutral_rows,
        "directional_pct": label_summary.get("directional_pct"),
        "long_short_ratio": label_summary.get("long_short_ratio"),
        "raw_event_target_rate": artifact_config.get("raw_event_target_rate"),
        "training_event_target_rate": artifact_config.get("training_event_target_rate"),
        "direction_threshold_ticks": artifact_config.get("direction_threshold_ticks"),
        "tp_mult": artifact_config.get("tp_mult"),
        "sl_mult": artifact_config.get("sl_mult"),
        "label_horizon": artifact_config.get("label_horizon"),
        "stage1_coverage_ratio": stage1_summary.get("coverage_ratio") or metrics.get("meta_coverage_ratio"),
        "catboost_f1_mean": stage1_summary.get("catboost_f1_mean"),
        "catboost_f1_min": stage1_summary.get("catboost_f1_min"),
        "catboost_best_iter_median": stage1_summary.get("catboost_best_iter_median"),
        "xgboost_f1_mean": stage1_summary.get("xgboost_f1_mean"),
        "xgboost_f1_min": stage1_summary.get("xgboost_f1_min"),
        "meta_train_sequences": as_int(split.get("train_sequences")) or as_int(meta_profile.get("train_sequences")),
        "meta_val_sequences": as_int(split.get("val_sequences")),
        "visual_coverage_ratio": metrics.get("visual_coverage_ratio"),
        "visual_seq_coverage": meta_profile.get("visual_seq_coverage"),
        "compact_profile": bool(meta_profile.get("compact_due_to_visual_coverage") or meta_profile.get("compact_due_to_small_data")),
        "meta_best_epoch": history_summary.get("best_epoch"),
        "meta_best_val_loss": history_summary.get("best_val_loss"),
        "meta_last_val_loss": history_summary.get("last_val_loss"),
        "meta_overfit_delta": history_summary.get("overfit_delta"),
        "bias_threshold": bias_threshold,
        "threshold_macro_f1": threshold_metrics.get("macro_f1"),
        "threshold_long_recall": threshold_metrics.get("long_recall"),
        "threshold_short_recall": threshold_metrics.get("short_recall"),
        "holdout_accuracy": final_accuracy,
        "holdout_macro_f1": final_macro,
        "holdout_long_f1": dig(holdout_report, "long", "f1"),
        "holdout_short_f1": dig(holdout_report, "short", "f1"),
        "holdout_long_recall": dig(holdout_report, "long", "recall"),
        "holdout_short_recall": dig(holdout_report, "short", "recall"),
        "holdout_rows": holdout_report.get("support") or threshold_metrics.get("report_rows"),
        "contract_passed": contract.get("passed"),
        "integrity_warning_count": len(data_integrity.get("warnings") or []),
        "dead_features_count": len(feature_drift.get("dead_features_3m") or []),
        "elapsed_seconds": metrics.get("elapsed_seconds") or dig(artifact_manifest, "metrics", "elapsed_seconds"),
    }

    risks = diagnose(row, data_integrity, contract, feature_drift)
    row["risk_level"] = risks["risk_level"]
    row["top_issue"] = risks["issues"][0] if risks["issues"] else ""

    return {
        "root": str(root),
        "summary": row,
        "diagnosis": risks,
        "reports": {
            "train_manifest": train_manifest,
            "artifact_manifest": artifact_manifest,
            "report_roots": [str(path) for path in report_roots],
            "report_sources": {
                "artifact_manifest": artifact_report_root,
                "label_quality_report": label_report_root,
                "data_integrity_report": integrity_report_root,
                "contract_consistency_report": contract_report_root,
                "feature_coverage_drift_report": drift_report_root,
            },
            "label_quality_summary": label_summary,
            "data_integrity": data_integrity,
            "contract_consistency": contract,
            "feature_drift": feature_drift,
            "stage1_summary": stage1_summary,
            "stage1_raw": stage1,
            "calibration_report": calibration,
            "time_split_report": time_split,
            "meta_history_summary": history_summary,
            "holdout_report": holdout_report,
            "feature_schema": feature_schema,
            "refinery_split": refinery_split,
            "refinery_timing": refinery_timing,
            "lob_build_meta": lob_build_meta,
            "day_trading_manifest": day_trading,
            "final_scan": final_scan,
        },
    }


def _integrity_hard_failure(integrity: dict[str, Any]) -> bool:
    final = integrity.get("final") or {}
    if (as_int(final.get("missing_ts_rows"), 0) or 0) > 0:
        return True
    if (as_int(final.get("non_monotonic_ts_steps"), 0) or 0) > 0:
        return True
    if (as_int(final.get("price_nonpositive_rows"), 0) or 0) > 0:
        return True
    for section_name in ["mbo", "mbp"]:
        section = ((integrity.get("inputs") or {}).get(section_name) or {})
        for part_name in ["raw", "normalized"]:
            part = section.get(part_name) or {}
            if (as_int(part.get("missing_ts_rows"), 0) or 0) > 0:
                return True
            if (as_int(part.get("non_monotonic_ts_steps"), 0) or 0) > 0:
                return True
            if (as_int(part.get("price_nonpositive_rows"), 0) or 0) > 0:
                return True
    return False


def diagnose(row: dict[str, Any], integrity: dict[str, Any], contract: dict[str, Any], drift: dict[str, Any]) -> dict[str, Any]:
    findings: list[tuple[str, str]] = []

    def add(level: str, msg: str) -> None:
        findings.append((level, msg))

    if contract and contract.get("passed") is False:
        add("Critical", f"contract consistency failed: {contract.get('failure_reason')}")
    if len(integrity.get("warnings") or []) > 0 and _integrity_hard_failure(integrity):
        add("High", f"data integrity warnings={len(integrity.get('warnings') or [])}")
    elif len(integrity.get("warnings") or []) > 0:
        add("Low", "duplicate timestamps present; verify stable tie-ordering, but no missing/non-monotonic timestamps detected")
    if as_float(row.get("event_view_pct"), 1.0) is not None and (as_float(row.get("event_view_pct"), 1.0) or 0) < 0.05:
        add("Medium", "event training view is very small vs full data")
    if as_float(row.get("stage1_coverage_ratio"), 1.0) is not None and (as_float(row.get("stage1_coverage_ratio"), 1.0) or 0) < 0.90:
        add("Medium", "Stage1 OOF coverage below 90%")
    if as_float(row.get("visual_seq_coverage"), 1.0) is not None and (as_float(row.get("visual_seq_coverage"), 1.0) or 0) < 0.85:
        add("Medium", "visual sequence coverage triggered compact MetaLearner")
    if as_float(row.get("catboost_f1_mean"), 1.0) is not None and (as_float(row.get("catboost_f1_mean"), 1.0) or 0) < 0.53:
        add("High", "CatBoost OOF directional F1 is near random")
    if as_float(row.get("catboost_best_iter_median"), 10.0) is not None and (as_float(row.get("catboost_best_iter_median"), 10.0) or 0) <= 2:
        add("High", "CatBoost median best_iteration is near zero")
    if as_int(row.get("meta_best_epoch"), 99) == 1 and (as_float(row.get("meta_overfit_delta"), 0.0) or 0) > 0.10:
        add("High", "MetaLearner overfits immediately after epoch 1")
    if as_float(row.get("holdout_macro_f1"), 1.0) is not None and (as_float(row.get("holdout_macro_f1"), 1.0) or 0) < 0.55:
        add("High", "MetaLearner final holdout macro F1 below 0.55")
    long_recall = as_float(row.get("holdout_long_recall"))
    short_recall = as_float(row.get("holdout_short_recall"))
    if long_recall is not None and short_recall is not None and abs(long_recall - short_recall) > 0.20:
        add("Medium", "LONG/SHORT recall imbalance on final holdout")
    if len(drift.get("dead_features_3m") or []) > 0:
        add("Medium", f"dead features detected: {len(drift.get('dead_features_3m') or [])}")

    order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    findings.sort(key=lambda item: order.get(item[0], 0), reverse=True)
    risk_level = findings[0][0] if findings else "Low"
    return {"risk_level": risk_level, "issues": [msg for _, msg in findings]}


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if abs(value) < 1 and value != 0:
            return f"{value:.4f}"
        return f"{value:.3f}"
    return str(value)


def fmt_pct(value: Any) -> str:
    x = as_float(value)
    return "" if x is None else f"{x * 100:.1f}%"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        vals = []
        for key, _ in columns:
            value = row.get(key)
            vals.append(fmt(value).replace("|", "\\|"))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def best_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rows if as_float(r.get("holdout_macro_f1")) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: as_float(r.get("holdout_macro_f1"), -1.0) or -1.0)


def write_outputs(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r["summary"] for r in results]

    with (out_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})

    with (out_dir / "comparison_details.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report = build_markdown_report(results)
    (out_dir / "comparison_report.md").write_text(report, encoding="utf-8")


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    rows = [r["summary"] for r in results]
    winner = best_run(rows)
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    lines = [
        "# QuantSystem V19 Experiment Comparison",
        "",
        f"Generated at: `{now}`",
        "",
    ]
    if winner:
        lines.extend(
            [
                "## Executive Summary",
                "",
                f"Best holdout macro F1: `{winner.get('run')}` = `{fmt(winner.get('holdout_macro_f1'))}`.",
                "Use this only as a model-quality comparison. Trading readiness still requires fee/slippage backtests.",
                "",
            ]
        )

    lines.extend(
        [
            "## Core Comparison",
            "",
            markdown_table(
                rows,
                [
                    ("run", "run"),
                    ("rows_full", "rows_full"),
                    ("rows_event", "event_rows"),
                    ("event_view_pct", "event_pct"),
                    ("sample_weight_mode", "sw_mode"),
                    ("catboost_f1_mean", "cb_f1"),
                    ("catboost_best_iter_median", "cb_iter_med"),
                    ("xgboost_f1_mean", "xgb_f1"),
                    ("visual_seq_coverage", "vis_seq_cov"),
                    ("meta_best_epoch", "best_epoch"),
                    ("holdout_macro_f1", "holdout_f1"),
                    ("holdout_accuracy", "accuracy"),
                    ("risk_level", "risk"),
                ],
            ),
            "",
            "## Label And Gate Comparison",
            "",
            markdown_table(
                rows,
                [
                    ("run", "run"),
                    ("label_long", "LONG"),
                    ("label_short", "SHORT"),
                    ("label_neutral", "NEUTRAL"),
                    ("directional_pct", "dir_pct"),
                    ("long_short_ratio", "LS_ratio"),
                    ("raw_event_target_rate", "raw_target"),
                    ("training_event_target_rate", "train_target"),
                    ("direction_threshold_ticks", "thr_ticks"),
                    ("tp_mult", "tp_mult"),
                    ("sl_mult", "sl_mult"),
                ],
            ),
            "",
            "## MetaLearner Holdout",
            "",
            markdown_table(
                rows,
                [
                    ("run", "run"),
                    ("bias_threshold", "long_thr"),
                    ("threshold_macro_f1", "calib_f1"),
                    ("holdout_macro_f1", "holdout_f1"),
                    ("holdout_long_f1", "long_f1"),
                    ("holdout_short_f1", "short_f1"),
                    ("holdout_long_recall", "long_rec"),
                    ("holdout_short_recall", "short_rec"),
                    ("holdout_rows", "rows"),
                    ("meta_overfit_delta", "overfit_delta"),
                ],
            ),
            "",
            "## Risk Flags",
            "",
        ]
    )

    for result in results:
        issues = result["diagnosis"]["issues"]
        lines.append(f"### {result['summary']['run']} - {result['diagnosis']['risk_level']}")
        if issues:
            for issue in issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- No major automatic flags.")
        lines.append("")

    lines.extend(["## Data Integrity Warning Details", ""])
    for result in results:
        run = result["summary"]["run"]
        warnings = (result["reports"].get("data_integrity") or {}).get("warnings") or []
        contract = result["reports"].get("contract_consistency") or {}
        sources = result["reports"].get("report_sources") or {}
        lines.append(f"### {run}")
        lines.append(f"- report source: `{sources.get('data_integrity_report') or 'missing'}`")
        if contract:
            lines.append(
                f"- contract passed: `{contract.get('passed')}` | "
                f"expected: `{contract.get('expected_symbol')}` | "
                f"detected: `{contract.get('detected_symbol')}`"
            )
        if warnings:
            for warning in warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("- No integrity warnings found.")
        lines.append("")

    lines.extend(
        [
            "## How To Read This",
            "",
            "- If `cb_iter_med` is near 0 and `cb_f1` is near 0.50, the statistical Stage1 surface is not learning stable signal.",
            "- If `best_epoch` is 1 and `overfit_delta` is large, the MetaLearner memorizes quickly and needs less capacity, more data, or cleaner labels.",
            "- If `vis_seq_cov` is below 0.85, the compact MetaLearner profile is expected; improve LOB coverage/alignment before blaming model architecture.",
            "- If labels are balanced but holdout recall is one-sided, threshold calibration or regime-specific calibration is likely needed.",
            "- ML metrics are not trading readiness. Confirm with chronological backtest including fees, spread, slippage, latency, and position sizing.",
            "",
        ]
    )
    return "\n".join(lines)


def print_console_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [r["summary"] for r in results]
    print("\n=== QuantSystem V19 comparison ===")
    for row in rows:
        print(
            f"- {row['run']}: "
            f"event_rows={fmt(row.get('rows_event'))} ({fmt_pct(row.get('event_view_pct'))}), "
            f"cb_f1={fmt(row.get('catboost_f1_mean'))}, "
            f"meta_f1={fmt(row.get('holdout_macro_f1'))}, "
            f"best_epoch={fmt(row.get('meta_best_epoch'))}, "
            f"risk={row.get('risk_level')}"
        )
        if row.get("top_issue"):
            print(f"  top_issue: {row.get('top_issue')}")
    print(f"\nWrote: {out_dir / 'comparison_report.md'}")
    print(f"Wrote: {out_dir / 'comparison_summary.csv'}")
    print(f"Wrote: {out_dir / 'comparison_details.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare QuantSystem V19 output folders and diagnose likely bottlenecks."
    )
    parser.add_argument("runs", nargs="*", help="Output directories, e.g. outputs_cleaned_v2 outputs_cleaned_v2_geom")
    parser.add_argument("--glob", action="append", default=[], help="Glob pattern for output dirs, e.g. 'outputs_cleaned_v2*'")
    parser.add_argument("--out", default=None, help="Directory for comparison outputs")
    parser.add_argument(
        "--scan-final",
        action="store_true",
        help="Also scan final parquet shards for exact counts/timestamp checks. Slower on huge runs.",
    )
    args = parser.parse_args()

    roots = resolve_runs(args.runs, args.glob)
    if not roots:
        parser.error("Provide at least one output directory or --glob pattern.")

    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        print("WARNING: missing run directories:")
        for item in missing:
            print(f"  - {item}")

    results = [collect_one_run(root, scan_final=args.scan_final) for root in roots if root.exists()]
    if not results:
        raise SystemExit("No readable run directories found.")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or f"v19_experiment_compare_{stamp}").expanduser().resolve()
    write_outputs(results, out_dir)
    print_console_summary(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
