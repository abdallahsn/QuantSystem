"""
prepare_v19.py - safe preparation wrapper for QuantSystem V19.

This wrapper preserves the existing preparation entry points while adding:
input validation, small-sample/dry-run modes, artifact validation reports, and
schema guards around future-looking metadata. It does not change model logic.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Iterable

import numpy as np
import pandas as pd

from modules.feature_artifact_v19 import (
    ARTIFACT_MANIFEST_NAME,
    iter_table_chunks,
    read_table,
    resolve_final_feature_paths,
    write_table,
)
from modules.validation_v19 import (
    assert_label_timestamps,
    summarize_label_distribution,
    validate_market_data_frame,
    write_validation_report,
)


ARTIFACT_VALIDATION_COLUMNS = [
    "ts_event",
    "label_end_ts",
    "bias_label",
    "train_event_flag",
    "event_flag",
    "price",
    "close",
    "open",
    "high",
    "low",
    "bid_px_00",
    "ask_px_00",
]

FORBIDDEN_RAW_FEATURE_TOKENS = (
    "label",
    "target",
    "future",
    "forward",
    "end_ts",
    "path_outcome",
    "trade_duration",
)


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _abs(path: str | None) -> str | None:
    return os.path.abspath(os.path.expanduser(str(path))) if path else None


def _json_safe(payload: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def _table_columns(path: str) -> list[str]:
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".pq", ".snappy"}:
        try:
            import pyarrow.parquet as pq

            return [str(col) for col in pq.read_schema(path).names]
        except Exception:
            try:
                from fastparquet import ParquetFile

                return [str(col) for col in ParquetFile(path).columns]
            except Exception:
                pass
        try:
            return [str(col) for col in read_table(path).columns]
        except Exception:
            return []
    if ext in {".csv", ".gz", ".zst"}:
        return [str(col) for col in pd.read_csv(path, nrows=0, compression="infer").columns]
    return [str(col) for col in read_table(path).columns]


def _artifact_columns(path_or_artifact: str) -> list[str]:
    columns: set[str] = set()
    for path in resolve_final_feature_paths(path_or_artifact):
        columns.update(_table_columns(path))
    return sorted(columns)


def _iter_limited_chunks(
    path: str,
    *,
    chunk_rows: int,
    max_rows: int | None = None,
    columns: list[str] | None = None,
) -> Iterable[pd.DataFrame]:
    rows_seen = 0
    for chunk in iter_table_chunks(path, chunk_rows, columns=columns):
        if max_rows is not None:
            remaining = int(max_rows) - rows_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        rows_seen += int(len(chunk))
        yield chunk


def _validate_input_path(
    path: str,
    *,
    kind: str,
    output_dir: str,
    chunk_rows: int,
    max_rows: int | None = None,
    tick_size: float | None = None,
) -> dict:
    issues: list[dict] = []
    rows = 0
    chunks = 0
    min_ts = None
    max_ts = None
    columns: set[str] = set()

    for chunk_no, chunk in enumerate(
        _iter_limited_chunks(path, chunk_rows=chunk_rows, max_rows=max_rows),
        start=1,
    ):
        chunks += 1
        rows += int(len(chunk))
        columns.update(str(c) for c in chunk.columns)
        report = validate_market_data_frame(
            chunk,
            context=f"prepare_v19.{kind}.chunk_{chunk_no}",
            tick_size=tick_size,
            strict=False,
        )
        for issue in report.get("issues", []):
            issues.append({"chunk": chunk_no, **issue})
        if "ts_event" in chunk.columns and len(chunk):
            ts = pd.to_datetime(chunk["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
            ts = ts.dropna()
            if len(ts):
                lo = ts.min()
                hi = ts.max()
                min_ts = lo if min_ts is None else min(min_ts, lo)
                max_ts = hi if max_ts is None else max(max_ts, hi)

    report = {
        "generated_at": _utc_now(),
        "kind": kind,
        "path": _abs(path),
        "rows_scanned": int(rows),
        "chunks_scanned": int(chunks),
        "max_rows_limit": None if max_rows is None else int(max_rows),
        "columns": sorted(columns),
        "ts_min": str(min_ts) if min_ts is not None else None,
        "ts_max": str(max_ts) if max_ts is not None else None,
        "rows_dropped_by_prepare_v19": 0,
        "issues": issues[:200],
        "issue_count": int(len(issues)),
        "passed": not any(issue.get("severity") in {"critical", "high"} for issue in issues),
    }
    filename = f"prep_v19_{kind}_input_validation_report.json"
    report["path_report"] = _json_safe(report, os.path.join(output_dir, filename))
    return report


def _sample_table(path: str, *, rows: int, output_path: str, chunk_rows: int) -> dict:
    if rows <= 0:
        raise ValueError("sample rows must be positive")
    frames: list[pd.DataFrame] = []
    rows_written = 0
    for chunk in _iter_limited_chunks(path, chunk_rows=chunk_rows, max_rows=rows):
        frames.append(chunk.copy())
        rows_written += int(len(chunk))
        if rows_written >= rows:
            break
    sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_table(sample, output_path)
    return {
        "source": _abs(path),
        "sample_path": _abs(output_path),
        "requested_rows": int(rows),
        "sampled_rows": int(len(sample)),
        "rows_dropped_by_prepare_v19": 0,
        "note": "Small-sample mode intentionally takes the first rows only; it does not randomize.",
    }


def _maybe_sample_inputs(args: argparse.Namespace) -> tuple[str, str | None, dict]:
    if int(args.sample_rows or 0) <= 0:
        return args.mbo, args.mbp, {"enabled": False}
    sample_dir = os.path.join(args.output, "_sample_inputs")
    os.makedirs(sample_dir, exist_ok=True)
    mbo_path = os.path.join(sample_dir, "mbo_sample.parquet")
    mbo_report = _sample_table(
        args.mbo,
        rows=int(args.sample_rows),
        output_path=mbo_path,
        chunk_rows=int(args.chunk_rows),
    )
    mbp_path = None
    mbp_report = None
    if args.mbp:
        mbp_path = os.path.join(sample_dir, "mbp_sample.parquet")
        mbp_report = _sample_table(
            args.mbp,
            rows=int(args.sample_rows),
            output_path=mbp_path,
            chunk_rows=int(args.chunk_rows),
        )
    report = {
        "enabled": True,
        "mbo": mbo_report,
        "mbp": mbp_report,
    }
    report["path_report"] = _json_safe(report, os.path.join(args.output, "prep_v19_sample_report.json"))
    return mbo_path, mbp_path, report


def _load_artifact_validation_frame(path_or_artifact: str, *, chunk_rows: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shard_path in resolve_final_feature_paths(path_or_artifact):
        for chunk in iter_table_chunks(
            shard_path,
            chunk_rows,
            columns=ARTIFACT_VALIDATION_COLUMNS,
        ):
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _causal_feature_schema_report(path_or_artifact: str) -> dict:
    columns = _artifact_columns(path_or_artifact)
    raw_feature_cols = [col for col in columns if col.startswith("raw__")]
    forbidden_raw_features = [
        col
        for col in raw_feature_cols
        if any(token in col.lower() for token in FORBIDDEN_RAW_FEATURE_TOKENS)
    ]
    future_metadata_cols = [
        col
        for col in columns
        if any(token in col.lower() for token in FORBIDDEN_RAW_FEATURE_TOKENS)
        and not col.startswith("raw__")
    ]
    return {
        "generated_at": _utc_now(),
        "path": _abs(path_or_artifact),
        "column_count": int(len(columns)),
        "raw_feature_count": int(len(raw_feature_cols)),
        "forbidden_raw_features": forbidden_raw_features,
        "future_or_label_metadata_columns": future_metadata_cols,
        "passed": bool(not forbidden_raw_features),
        "note": (
            "Future/label columns may exist as metadata/targets, but they must not appear "
            "inside raw__ model feature columns."
        ),
    }


def write_artifact_validation_reports(
    path_or_artifact: str,
    *,
    output_dir: str,
    chunk_rows: int,
    strict: bool = True,
) -> dict:
    df = _load_artifact_validation_frame(path_or_artifact, chunk_rows=chunk_rows)
    required = ["ts_event", "label_end_ts", "bias_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Prepared artifact is missing required train_v19 columns: {missing}")

    label_timestamp_report = assert_label_timestamps(df, context="prepare_v19.artifact")
    label_distribution_report = summarize_label_distribution(df, context="prepare_v19.artifact")
    market_report = validate_market_data_frame(df, context="prepare_v19.artifact", strict=False)
    schema_report = _causal_feature_schema_report(path_or_artifact)

    paths = {
        "label_timestamp": write_validation_report(
            label_timestamp_report,
            output_dir,
            "label_timestamp_report.json",
        ),
        "label_distribution": write_validation_report(
            label_distribution_report,
            output_dir,
            "label_distribution_full_report.json",
        ),
        "data_validation": write_validation_report(
            market_report,
            output_dir,
            "data_validation_full_report.json",
        ),
        "causal_feature_schema": write_validation_report(
            schema_report,
            output_dir,
            "causal_feature_schema_report.json",
        ),
    }
    summary = {
        "generated_at": _utc_now(),
        "artifact": _abs(path_or_artifact),
        "rows_validated": int(len(df)),
        "required_columns": required,
        "reports": paths,
        "passed": bool(
            label_timestamp_report.get("passed", False)
            and market_report.get("passed", False)
            and schema_report.get("passed", False)
        ),
    }
    if strict and not summary["passed"]:
        raise ValueError(f"Artifact validation failed; see reports: {paths}")
    summary["path_report"] = _json_safe(summary, os.path.join(output_dir, "prep_v19_artifact_validation_report.json"))
    return summary


def _write_day_bar_manifest(output_dir: str, feature_path: str) -> str:
    rows = 0
    try:
        rows = int(len(read_table(feature_path, columns=["ts_event"])))
    except Exception:
        rows = 0
    manifest = {
        "kind": "prepare_v19_day_bar",
        "created_at_utc": _utc_now(),
        "output_dir": _abs(output_dir),
        "metrics": {"rows": rows},
        "extra": {
            "final_feature_shards": [
                {
                    "name": os.path.basename(feature_path),
                    "path": _abs(feature_path),
                    "rows": rows,
                }
            ]
        },
    }
    return _json_safe(manifest, os.path.join(output_dir, ARTIFACT_MANIFEST_NAME))


def _run_raw_validation(args: argparse.Namespace) -> dict:
    os.makedirs(args.output, exist_ok=True)
    max_rows = int(args.sample_rows) if int(args.sample_rows or 0) > 0 else None
    reports = {
        "generated_at": _utc_now(),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "validation_only": bool(args.validation_only),
        "sample_rows": max_rows,
        "mbo": _validate_input_path(
            args.mbo,
            kind="mbo",
            output_dir=args.output,
            chunk_rows=int(args.chunk_rows),
            max_rows=max_rows,
            tick_size=args.tick_size,
        ),
    }
    if args.mbp:
        reports["mbp"] = _validate_input_path(
            args.mbp,
            kind="mbp",
            output_dir=args.output,
            chunk_rows=int(args.chunk_rows),
            max_rows=max_rows,
            tick_size=args.tick_size,
        )
    reports["passed"] = bool(
        reports["mbo"].get("passed", False)
        and (not args.mbp or reports.get("mbp", {}).get("passed", False))
    )
    reports["path_report"] = _json_safe(reports, os.path.join(args.output, "prep_v19_input_validation_report.json"))
    return reports


def run(args: argparse.Namespace) -> str | None:
    os.makedirs(args.output, exist_ok=True)
    if args.validate_artifact:
        summary = write_artifact_validation_reports(
            args.validate_artifact,
            output_dir=args.output,
            chunk_rows=int(args.chunk_rows),
            strict=bool(args.strict_validation),
        )
        print(f"Artifact validation complete: {summary['path_report']}")
        return args.validate_artifact

    if not args.mbo:
        raise ValueError("--mbo is required unless --validate_artifact is used")
    if args.mode == "event_tick" and not args.mbp:
        raise ValueError("--mbp is required for --mode event_tick")

    input_report = _run_raw_validation(args)
    print(f"Input validation report: {input_report['path_report']}")
    if args.dry_run or args.validation_only:
        print("Dry-run/validation-only mode: no feature artifact was generated.")
        return None
    if args.strict_validation and not input_report.get("passed", False):
        raise ValueError(f"Input validation failed; see {input_report['path_report']}")

    run_mbo, run_mbp, sample_report = _maybe_sample_inputs(args)
    if sample_report.get("enabled"):
        print(f"Small-sample inputs written: {sample_report['path_report']}")

    if args.mode == "event_tick":
        from prepare_training_data import run_refinery

        run_refinery(
            mbo_path=run_mbo,
            mbp_path=run_mbp,
            symbol=args.symbol,
            output_dir=args.output,
            chunk_rows=int(args.chunk_rows),
            chunksize=int(args.chunk_rows),
            label_mode="v19",
            n_workers=args.n_workers,
            mbo_workers=args.mbo_workers,
            mbp_workers=args.mbp_workers,
            resume=bool(args.resume),
            shard_warmup_rows=int(args.shard_warmup_rows),
            label_horizon=int(args.label_horizon),
            event_roll_window=int(args.event_roll_window),
            feature_roll_window=int(args.feature_roll_window),
            direction_threshold_ticks=float(args.direction_threshold_ticks),
            causal_threshold_mode=args.causal_threshold_mode,
            raw_event_target_rate=float(args.raw_event_target_rate),
            training_event_target_rate=float(args.training_event_target_rate),
            training_event_score_threshold=args.training_event_score_threshold,
            lob_event_sample=int(args.lob_event_sample),
            tp_mult=float(args.tp_mult),
            sl_mult=float(args.sl_mult),
            adaptive_horizon=bool(args.adaptive_horizon),
            trend_filter=bool(args.trend_filter),
            trend_filter_strict=bool(args.trend_filter_strict),
            merge_tolerance_ms=int(args.merge_tolerance_ms),
            step4_min_parallel_rows=int(args.step4_min_parallel_rows),
            use_soft_labels=bool(args.use_soft_labels),
            soft_label_mode=args.soft_label_mode,
            config_path=args.config,
            enforce_economic_tp_floor=bool(args.enforce_economic_tp_floor),
            continuous_contract_root=args.continuous_contract_root,
            allow_legacy_session_labels=False,
        )
        artifact_path = args.output
    else:
        from prepare_day_trading import run_day_trading_refinery

        artifact_path = run_day_trading_refinery(
            mbo_dir=run_mbo,
            mbp_path=run_mbp,
            output_dir=args.output,
            freq=args.freq,
            horizon_bars=int(args.horizon),
            tp_atr_mult=float(args.tp_mult),
            sl_atr_mult=float(args.sl_mult),
            build_lob_tensors=not bool(args.no_lob),
            event_threshold_scale=float(args.event_threshold_scale),
            event_threshold_shift=float(args.event_threshold_shift),
            event_target_rate=float(args.event_target_rate),
            event_min_score_floor=float(args.event_min_score_floor),
            kalman_event_floor=float(args.kalman_event_floor),
            weak_event_to_directional=bool(args.weak_event_to_directional),
            weak_event_min_move_atr=float(args.weak_event_min_move_atr),
            sl_to_opposite=bool(args.sl_to_opposite),
            include_weak_directional_in_train=bool(args.include_weak_directional_in_train),
            research_label_overrides=bool(args.research_label_overrides),
        )
        manifest_path = _write_day_bar_manifest(args.output, artifact_path)
        print(f"Day-bar train_v19 artifact manifest: {manifest_path}")

    validation_summary = write_artifact_validation_reports(
        artifact_path,
        output_dir=args.output,
        chunk_rows=int(args.chunk_rows),
        strict=bool(args.strict_validation),
    )
    print(f"Artifact validation report: {validation_summary['path_report']}")
    return artifact_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Safe QuantSystem V19 preparation wrapper")
    p.add_argument("--mode", choices=["event_tick", "day_bar"], default="event_tick")
    p.add_argument("--mbo", default=None)
    p.add_argument("--mbp", default=None)
    p.add_argument("--output", default="outputs_prepare_v19")
    p.add_argument("--config", default=None)
    p.add_argument("--symbol", default="")
    p.add_argument("--continuous_contract_root", default="")
    p.add_argument("--chunk_rows", type=int, default=500_000)
    p.add_argument("--sample_rows", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--validation_only", action="store_true")
    p.add_argument("--validate_artifact", default=None)
    p.add_argument("--strict_validation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tick_size", type=float, default=None)

    p.add_argument("--n_workers", type=int, default=None)
    p.add_argument("--mbo_workers", type=int, default=None)
    p.add_argument("--mbp_workers", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard_warmup_rows", type=int, default=5_000)
    p.add_argument("--label_horizon", type=int, default=150)
    p.add_argument("--event_roll_window", type=int, default=50)
    p.add_argument("--feature_roll_window", type=int, default=150)
    p.add_argument("--direction_threshold_ticks", type=float, default=1.5)
    p.add_argument("--causal_threshold_mode", choices=["expanding", "fixed"], default="expanding")
    p.add_argument("--raw_event_target_rate", type=float, default=0.70)
    p.add_argument("--training_event_target_rate", type=float, default=0.25)
    p.add_argument("--training_event_score_threshold", type=float, default=None)
    p.add_argument("--lob_event_sample", type=int, default=100_000)
    p.add_argument("--merge_tolerance_ms", type=int, default=100)
    p.add_argument("--step4_min_parallel_rows", type=int, default=250_000)
    p.add_argument("--tp_mult", type=float, default=1.5)
    p.add_argument("--sl_mult", type=float, default=1.0)
    p.add_argument("--adaptive_horizon", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trend_filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trend_filter_strict", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enforce_economic_tp_floor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_soft_labels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--soft_label_mode", choices=["analytical", "monte_carlo"], default="analytical")

    p.add_argument("--freq", default="5min", choices=["5min", "15min", "30min"])
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--no_lob", action="store_true")
    p.add_argument("--event_threshold_scale", type=float, default=1.0)
    p.add_argument("--event_threshold_shift", type=float, default=0.0)
    p.add_argument("--event_target_rate", type=float, default=0.20)
    p.add_argument("--event_min_score_floor", type=float, default=0.20)
    p.add_argument("--kalman_event_floor", type=float, default=0.70)
    p.add_argument("--weak_event_to_directional", action="store_true")
    p.add_argument("--weak_event_min_move_atr", type=float, default=0.35)
    p.add_argument("--sl_to_opposite", action="store_true")
    p.add_argument("--include_weak_directional_in_train", action="store_true")
    p.add_argument("--research_label_overrides", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
