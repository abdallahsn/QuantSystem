"""QuantSystem v20 artifact-producing data preparation pipeline.

Phase 2 scope: build a leakage-safe, trainable tabular artifact from raw
Databento MBO + MBP-10 data. MBP is the feature clock; MBO trade-flow state is
aligned with backward as-of joins so feature rows never see future order flow.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_cleaning.market_cleaner import CleanConfig, clean_market_data
from data_ingestion.databento_schema import FeedType, validate_required_columns
from data_ingestion.table_reader import iter_market_chunks, scan_table_schema
from datasets.artifact_writer import write_feature_artifact
from feature_engineering.lob_features import compute_lob_features
from feature_engineering.ofi import compute_mlofi
from labeling.triple_barrier import TripleBarrierConfig, build_triple_barrier_labels
from modules.feature_artifact_v19 import load_feature_artifact, write_table
from modules.validation_v19 import (
    assert_label_timestamps,
    summarize_label_distribution,
    validate_market_data_frame,
    validate_time_splits,
    write_validation_report,
)
from training.splits import PurgedWalkForwardConfig, build_purged_walkforward_splits
from validation.artifact_schema import validate_artifact_schema


SCHEMA_VERSION = "v20.0"
LEVELS = 10
ROLLING_WINDOWS = (5, 20, 50, 100)
MBO_STALE_WARNING_P95_MS = 30_000.0
MBO_STALE_CRITICAL_P95_MS = 120_000.0
CONCAT_ROWS_WARNING = 5_000_000
CONCAT_MEMORY_WARNING_GB = 8.0
TRADE_ACTIONS = {"T", "F", "TRADE", "FILL"}
BUY_SIDE_VALUES = {"A", "ASK", "BUY", "BID_LIFT"}
SELL_SIDE_VALUES = {"B", "BID", "SELL", "ASK_HIT"}
SYMBOL_COLUMNS = ("symbol", "contract_symbol", "instrument_id")
METADATA_COLUMNS = (
    "ts_event",
    "ts_recv",
    "feature_ts",
    "label_start_ts",
    "mbo_state_ts",
    "symbol",
    "contract_symbol",
    "instrument_id",
)
LABEL_COLUMNS = (
    "bias_label",
    "direction_label",
    "tradeability_label",
    "event_flag",
    "train_event_flag",
    "label_end_ts",
    "horizon_end_ts",
    "label_horizon_steps",
    "label_outcome",
    "barrier_hit_type",
    "realized_return",
    "forward_return",
    "label_barrier_upper",
    "label_barrier_lower",
    "label_neutral_abs",
    "signal_quality",
    "conf_label",
    "soft_label",
    "soft_sample_weight",
    "soft_label_confidence",
    "mc_sample_weight",
    "label_stability",
)

# V19 imports these names from prepare_training_data.py. Keep a local copy here
# to avoid importing the heavy V19 preparation stack during lightweight data prep.
V19_COMPAT_FEATURES = (
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
)


@dataclass(frozen=True)
class LoadedFeed:
    frame: pd.DataFrame
    report: dict


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, list, tuple, dict)) else False:
        return None
    return value


def _write_json_report(payload: dict, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return write_validation_report(_json_ready(payload), output_dir, filename)


def _parse_ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp argument: {value}")
    return pd.Timestamp(ts).tz_localize(None)


def _coerce_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(np.float64)


def _top_counts(df: pd.DataFrame, column: str, limit: int = 20) -> dict:
    if column not in df.columns:
        return {}
    counts = df[column].astype(str).value_counts(dropna=False).head(int(limit))
    return {str(k): int(v) for k, v in counts.items()}


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd(), text=True, stderr=subprocess.DEVNULL)
        return out.strip() or None
    except Exception:
        return None


def _current_rss_gb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 3)
    except Exception:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.name == "posix" and rss > 10_000_000:
            return rss / (1024.0 ** 3)
        return rss / (1024.0 ** 2)
    except Exception:
        return None


def _enforce_memory_guard(args: argparse.Namespace, *, stage: str, report: dict | None = None) -> None:
    limit = float(getattr(args, "max_memory_gb", 0.0) or 0.0)
    current = _current_rss_gb()
    if report is not None:
        report.setdefault("memory_checkpoints", []).append({"stage": stage, "rss_gb": current, "limit_gb": limit or None})
    if limit > 0 and current is not None and current > limit:
        raise MemoryError(f"prepare_v20 memory guard exceeded at {stage}: rss_gb={current:.3f} > max_memory_gb={limit:.3f}")


def _dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return total
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return int(total)


def _file_size_report(output_dir: str) -> dict:
    files: list[dict] = []
    for root, _, names in os.walk(output_dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            files.append({
                "path": os.path.relpath(path, output_dir),
                "bytes": int(size),
                "mb": float(size / (1024.0 ** 2)),
            })
    total = sum(item["bytes"] for item in files)
    return {"total_bytes": int(total), "total_mb": float(total / (1024.0 ** 2)), "files": files}


def _catboost_dependency_report() -> dict:
    try:
        import catboost

        return {
            "available": True,
            "version": getattr(catboost, "__version__", None),
            "install_command": None,
            "message": "CatBoost is importable in the current Python environment.",
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
            "install_command": "python -m pip install catboost",
            "requirements_command": "python -m pip install -r requirements.txt",
            "message": "CatBoost is required for train_v19.py --phase catboost sanity training.",
        }


def _dataframe_memory_gb(df: pd.DataFrame | None) -> float:
    if df is None:
        return 0.0
    try:
        return float(df.memory_usage(deep=True).sum()) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _mbo_flow_reliability_report(alignment_report: dict) -> dict:
    age = alignment_report.get("mbo_state_age_ms") or {}
    p95 = age.get("p95")
    match_rate = float(alignment_report.get("match_rate", 0.0) or 0.0)
    warnings = list(alignment_report.get("warnings") or [])
    critical = list(alignment_report.get("critical") or [])
    reliable = bool(match_rate >= 0.95 and p95 is not None and float(p95) <= MBO_STALE_WARNING_P95_MS and not critical)
    stale = not reliable
    return {
        "mbo_flow_features_reliable": reliable,
        "mbo_flow_features_stale": stale,
        "reliability_rule": "reliable iff match_rate>=0.95 and mbo_state_age_ms.p95<=30s and no critical freshness warnings",
        "mbo_dependent_features": [
            "cvd",
            "mbo_cvd",
            "mbo_signed_trade_size_since_prev_mbp",
            "mbo_buy_trade_size_since_prev_mbp",
            "mbo_sell_trade_size_since_prev_mbp",
            "mbo_trade_count_since_prev_mbp",
            "vnet",
            "cvd_momentum",
            "cvd_price_divergence",
            "hawkes_intensity",
            "kyle_lambda",
        ],
        "mbp_mlofi_features_reliable": True,
        "mbp_mlofi_note": "MLOFI is computed from MBP snapshots, so it is not made stale by MBO trade-state age.",
        "warnings": warnings,
        "critical": critical,
    }


def _filter_time_and_symbol(df: pd.DataFrame, args: argparse.Namespace, report: dict) -> pd.DataFrame:
    out = df.copy()
    start = _parse_ts(getattr(args, "start", None))
    end = _parse_ts(getattr(args, "end", None))
    if start is not None:
        before = int(len(out))
        out = out.loc[out["ts_event"] >= start].copy()
        report["rows_dropped_start_filter"] += before - int(len(out))
    if end is not None:
        before = int(len(out))
        out = out.loc[out["ts_event"] <= end].copy()
        report["rows_dropped_end_filter"] += before - int(len(out))

    symbol = getattr(args, "symbol", None)
    if symbol:
        symbol = str(symbol)
        cols = [col for col in SYMBOL_COLUMNS if col in out.columns]
        if not cols:
            report["warnings"].append("symbol_filter_requested_but_no_symbol_column")
            return out
        mask = np.zeros(len(out), dtype=bool)
        for col in cols:
            mask |= out[col].astype(str).to_numpy() == symbol
        before = int(len(out))
        out = out.loc[mask].copy()
        report["rows_dropped_symbol_filter"] += before - int(len(out))
    return out


def _load_feed(path: str, *, feed: FeedType, args: argparse.Namespace) -> LoadedFeed:
    schema = scan_table_schema(path, chunk_rows=args.chunk_rows, expected_feed=feed)
    sample_rows = int(getattr(args, "sample_rows", 0) or 0)
    rows_seen = 0
    chunks = 0
    frames: list[pd.DataFrame] = []
    source_columns: list[str] = []
    report: dict = {
        "path": os.path.abspath(path),
        "feed_type": feed.value,
        "schema": schema.to_dict(),
        "rows_seen": 0,
        "rows_after_cleaning": 0,
        "rows_after_filters": 0,
        "chunks_seen": 0,
        "clean_reports": [],
        "validation_reports": [],
        "rows_dropped_start_filter": 0,
        "rows_dropped_end_filter": 0,
        "rows_dropped_symbol_filter": 0,
        "invalid_size_rows": 0,
        "symbol_counts": {},
        "contract_counts": {},
        "pre_filter_value_counts": {},
        "warnings": [],
        "passed": bool(schema.passed),
    }
    for chunk in iter_market_chunks(path, chunk_rows=args.chunk_rows):
        if sample_rows and rows_seen >= sample_rows:
            break
        if sample_rows:
            chunk = chunk.iloc[: max(sample_rows - rows_seen, 0)].copy()
        if chunk.empty:
            continue
        if not source_columns:
            source_columns = [str(col) for col in chunk.columns]
        chunks += 1
        rows_seen += int(len(chunk))

        required = validate_required_columns(chunk, feed)
        if required.get("missing_columns"):
            report["validation_reports"].append(required)
            report["passed"] = False
            if args.strict:
                raise ValueError(f"{feed.value}: missing required columns {required['missing_columns']}")

        cleaned, clean_report = clean_market_data(
            chunk,
            CleanConfig(sort_by_ts=True, drop_invalid_timestamps=True, drop_exact_duplicates=True),
        )
        report["clean_reports"].append(clean_report.to_dict())

        if "size" in cleaned.columns:
            size = pd.to_numeric(cleaned["size"], errors="coerce")
            report["invalid_size_rows"] += int((size < 0).sum())
            cleaned = cleaned.loc[(size.isna()) | (size >= 0)].copy()

        validation = validate_market_data_frame(
            cleaned,
            context=f"prepare_v20.{feed.value}.chunk_{chunks}",
            tick_size=float(args.tick_size),
            strict=False,
        )
        report["validation_reports"].append(validation)
        for col in SYMBOL_COLUMNS:
            if col in cleaned.columns:
                counts = report["pre_filter_value_counts"].setdefault(col, {})
                for key, value in cleaned[col].astype(str).value_counts(dropna=False).items():
                    counts[str(key)] = int(counts.get(str(key), 0) + int(value))
        cleaned = _filter_time_and_symbol(cleaned, args, report)
        if not cleaned.empty:
            frames.append(cleaned)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("ts_event").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=source_columns)
    report["rows_seen"] = int(rows_seen)
    report["rows_after_cleaning"] = int(sum(r.get("output_rows", 0) for r in report["clean_reports"]))
    report["rows_after_filters"] = int(len(df))
    report["chunks_seen"] = int(chunks)
    for col, counts in list(report["pre_filter_value_counts"].items()):
        report["pre_filter_value_counts"][col] = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20])
    for col in SYMBOL_COLUMNS:
        if col in df.columns:
            counts = _top_counts(df, col)
            if col == "contract_symbol":
                report["contract_counts"] = counts
            else:
                report["symbol_counts"][col] = counts
            if len(counts) > 1 and not getattr(args, "symbol", None):
                report["warnings"].append(f"multiple_values_in_{col}_without_symbol_filter")
    if df.empty:
        report["passed"] = False
        report["warnings"].append("empty_after_cleaning_and_filters")
        if args.strict:
            raise ValueError(f"{feed.value}: empty after cleaning/filtering")
    return LoadedFeed(frame=df, report=report)


def _reject_invalid_mbp_rows(mbp: pd.DataFrame, *, tick_size: float) -> tuple[pd.DataFrame, dict]:
    out = mbp.copy()
    if out.empty or "ts_event" not in out.columns:
        return out, {
            "input_rows": int(len(mbp)),
            "missing_or_nonpositive_bbo_rows": 0,
            "crossed_book_rows": 0,
            "spread_lt_tick_rows": 0,
            "negative_size_rows": 0,
            "null_size_values_filled_zero": 0,
            "rows_rejected": int(len(mbp)),
            "output_rows": 0,
            "tick_size": float(tick_size),
            "passed": False,
            "warnings": ["empty_mbp_after_filters" if out.empty else "missing_ts_event_column"],
        }
    bid = _num(out, "bid_px_00", default=np.nan)
    ask = _num(out, "ask_px_00", default=np.nan)
    spread = ask - bid
    tick = max(float(tick_size), 1e-12)
    valid_prices = bid.gt(0) & ask.gt(0)
    crossed = valid_prices & bid.gt(ask)
    too_tight = valid_prices & spread.add(tick * 1e-6).lt(tick)
    size_cols = [col for col in out.columns if str(col).startswith(("bid_sz_", "ask_sz_"))]
    negative_size_mask = np.zeros(len(out), dtype=bool)
    null_size_values = 0
    for col in size_cols:
        size = pd.to_numeric(out[col], errors="coerce")
        null_size_values += int(size.isna().sum())
        negative_size_mask |= (size < 0).fillna(False).to_numpy(dtype=bool)
        out[col] = size.fillna(0.0)
    keep = valid_prices.to_numpy(dtype=bool) & (~crossed.to_numpy(dtype=bool)) & (~too_tight.to_numpy(dtype=bool)) & (~negative_size_mask)
    report = {
        "input_rows": int(len(mbp)),
        "missing_or_nonpositive_bbo_rows": int((~valid_prices).sum()),
        "crossed_book_rows": int(crossed.sum()),
        "spread_lt_tick_rows": int(too_tight.sum()),
        "negative_size_rows": int(negative_size_mask.sum()),
        "null_size_values_filled_zero": int(null_size_values),
        "rows_rejected": int((~keep).sum()),
        "output_rows": int(keep.sum()),
        "tick_size": float(tick_size),
        "passed": bool(keep.sum() > 0),
        "warnings": [],
    }
    out = out.loc[keep].sort_values("ts_event").reset_index(drop=True)
    return out, report


def _build_mbo_flow_state(mbo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {
        "input_rows": int(len(mbo)),
        "trade_action_rows": 0,
        "invalid_trade_price_rows": 0,
        "invalid_trade_size_rows": 0,
        "unknown_trade_side_rows": 0,
        "state_rows": 0,
        "passed": True,
    }
    if mbo.empty:
        report["passed"] = False
        return pd.DataFrame(columns=["mbo_state_ts"]), report

    action = mbo.get("action", "").astype(str).str.upper()
    trade_mask = action.isin(TRADE_ACTIONS)
    trades = mbo.loc[trade_mask].copy()
    report["trade_action_rows"] = int(len(trades))
    if trades.empty:
        report["passed"] = False
        return pd.DataFrame(columns=["mbo_state_ts"]), report

    price = pd.to_numeric(trades.get("price"), errors="coerce")
    size = pd.to_numeric(trades.get("size"), errors="coerce")
    valid = price.gt(0) & size.ge(0) & size.notna()
    report["invalid_trade_price_rows"] = int((~price.gt(0)).sum())
    report["invalid_trade_size_rows"] = int((~size.ge(0) | size.isna()).sum())
    trades = trades.loc[valid].copy()
    if trades.empty:
        report["passed"] = False
        return pd.DataFrame(columns=["mbo_state_ts"]), report

    side = trades.get("side", "").astype(str).str.upper()
    buy_side = side.isin(BUY_SIDE_VALUES)
    sell_side = side.isin(SELL_SIDE_VALUES)
    report["unknown_trade_side_rows"] = int((~(buy_side | sell_side)).sum())
    signed = np.where(buy_side, size.loc[trades.index].to_numpy(dtype=np.float64), 0.0)
    signed = np.where(sell_side, -size.loc[trades.index].to_numpy(dtype=np.float64), signed)
    trades["signed_trade_size"] = signed
    trades["buy_trade_size"] = np.where(signed > 0, signed, 0.0)
    trades["sell_trade_size"] = np.where(signed < 0, -signed, 0.0)
    trades["trade_count"] = 1
    trades["last_trade_price"] = price.loc[trades.index].to_numpy(dtype=np.float64)

    state = (
        trades.groupby("ts_event", sort=True)
        .agg(
            mbo_signed_trade_size=("signed_trade_size", "sum"),
            mbo_buy_trade_size=("buy_trade_size", "sum"),
            mbo_sell_trade_size=("sell_trade_size", "sum"),
            mbo_trade_count=("trade_count", "sum"),
            mbo_last_trade_price=("last_trade_price", "last"),
        )
        .reset_index()
        .rename(columns={"ts_event": "mbo_state_ts"})
    )
    state["mbo_cvd"] = state["mbo_signed_trade_size"].cumsum()
    state["mbo_cum_buy_volume"] = state["mbo_buy_trade_size"].cumsum()
    state["mbo_cum_sell_volume"] = state["mbo_sell_trade_size"].cumsum()
    state["mbo_cum_trade_count"] = state["mbo_trade_count"].cumsum()
    report["state_rows"] = int(len(state))
    return state.sort_values("mbo_state_ts").reset_index(drop=True), report


def _align_mbo_to_mbp(mbp: pd.DataFrame, mbo_state: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    left = mbp.sort_values("ts_event").reset_index(drop=True).copy()
    if left.empty:
        left["mbo_state_ts"] = pd.NaT
        for col in (
            "mbo_signed_trade_size",
            "mbo_buy_trade_size",
            "mbo_sell_trade_size",
            "mbo_trade_count",
            "mbo_last_trade_price",
            "mbo_cvd",
            "mbo_cum_buy_volume",
            "mbo_cum_sell_volume",
            "mbo_cum_trade_count",
        ):
            left[col] = 0.0
        return left, {
            "method": "empty_mbp_feature_clock",
            "rows": 0,
            "matched_rows": 0,
            "match_rate": 0.0,
            "mbo_state_age_ms": {},
            "mbo_state_age_thresholds": {},
            "mbo_flow_features_reliable": False,
            "warnings": ["empty_mbp_feature_clock"],
            "critical": ["empty_mbp_feature_clock"],
        }
    if mbo_state.empty:
        left["mbo_state_ts"] = pd.NaT
        for col in (
            "mbo_signed_trade_size",
            "mbo_buy_trade_size",
            "mbo_sell_trade_size",
            "mbo_trade_count",
            "mbo_last_trade_price",
            "mbo_cvd",
            "mbo_cum_buy_volume",
            "mbo_cum_sell_volume",
            "mbo_cum_trade_count",
        ):
            left[col] = 0.0
        return left, {
            "method": "empty_mbo_state",
            "matched_rows": 0,
            "match_rate": 0.0,
            "rows": int(len(left)),
            "mbo_state_age_ms": {},
            "mbo_state_age_thresholds": {},
            "mbo_flow_features_reliable": False,
            "warnings": ["empty_mbo_state"],
            "critical": ["empty_mbo_state"],
        }

    right = mbo_state.sort_values("mbo_state_ts").reset_index(drop=True)
    aligned = pd.merge_asof(
        left,
        right,
        left_on="ts_event",
        right_on="mbo_state_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    flow_cols = [
        "mbo_signed_trade_size",
        "mbo_buy_trade_size",
        "mbo_sell_trade_size",
        "mbo_trade_count",
        "mbo_cvd",
        "mbo_cum_buy_volume",
        "mbo_cum_sell_volume",
        "mbo_cum_trade_count",
    ]
    for col in flow_cols:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)
    aligned["mbo_last_trade_price"] = pd.to_numeric(aligned.get("mbo_last_trade_price"), errors="coerce")
    matched = int(aligned["mbo_state_ts"].notna().sum())
    warnings: list[str] = []
    critical: list[str] = []
    age_ms_stats = {}
    age_thresholds = {}
    if matched:
        age_ms = (_coerce_ts(aligned.loc[aligned["mbo_state_ts"].notna(), "ts_event"]) - _coerce_ts(aligned.loc[aligned["mbo_state_ts"].notna(), "mbo_state_ts"])).dt.total_seconds() * 1000.0
        age_values = age_ms.to_numpy(dtype=np.float64)
        total_rows = max(len(aligned), 1)
        age_ms_stats = {
            "min": float(age_ms.min()),
            "median": float(age_ms.median()),
            "p90": float(age_ms.quantile(0.90)),
            "p95": float(age_ms.quantile(0.95)),
            "p99": float(age_ms.quantile(0.99)),
            "max": float(age_ms.max()),
        }
        age_thresholds = {
            "rows_unmatched": int(len(aligned) - matched),
            "pct_unmatched": float((len(aligned) - matched) / total_rows),
            "pct_age_gt_1s": float(np.sum(age_values > 1_000.0) / total_rows),
            "pct_age_gt_5s": float(np.sum(age_values > 5_000.0) / total_rows),
            "pct_age_gt_10s": float(np.sum(age_values > 10_000.0) / total_rows),
            "pct_age_gt_30s": float(np.sum(age_values > 30_000.0) / total_rows),
            "pct_age_gt_60s": float(np.sum(age_values > 60_000.0) / total_rows),
            "denominator_rows": int(len(aligned)),
            "matched_rows": matched,
        }
        if age_ms_stats["p95"] > MBO_STALE_WARNING_P95_MS:
            warnings.append("mbo_state_age_p95_gt_30s")
        if age_ms_stats["p95"] > MBO_STALE_CRITICAL_P95_MS:
            critical.append("mbo_state_age_p95_gt_120s")
        if age_ms_stats["max"] > 60_000.0:
            warnings.append("mbo_state_age_max_gt_60s")
    else:
        critical.append("no_mbo_state_matches")
    mbo_reliable = bool(matched > 0 and not critical and age_ms_stats and age_ms_stats["p95"] <= MBO_STALE_WARNING_P95_MS and matched / max(len(aligned), 1) >= 0.95)
    report = {
        "method": "pd.merge_asof_backward",
        "rows": int(len(aligned)),
        "matched_rows": matched,
        "match_rate": float(matched / max(len(aligned), 1)),
        "mbo_state_age_ms": age_ms_stats,
        "mbo_state_age_thresholds": age_thresholds,
        "mbo_flow_features_reliable": mbo_reliable,
        "warnings": warnings,
        "critical": critical,
        "first_mbp_ts": str(aligned["ts_event"].min()) if len(aligned) else None,
        "last_mbp_ts": str(aligned["ts_event"].max()) if len(aligned) else None,
        "first_mbo_state_ts": str(right["mbo_state_ts"].min()) if len(right) else None,
        "last_mbo_state_ts": str(right["mbo_state_ts"].max()) if len(right) else None,
    }
    return aligned, report


def _add_level_features(features: pd.DataFrame, mbp: pd.DataFrame, *, tick_size: float) -> None:
    mid = pd.to_numeric(features["mid_price"], errors="coerce")
    for level in range(LEVELS):
        bpx_col = f"bid_px_{level:02d}"
        apx_col = f"ask_px_{level:02d}"
        bsz_col = f"bid_sz_{level:02d}"
        asz_col = f"ask_sz_{level:02d}"
        bid_px = _num(mbp, bpx_col, default=np.nan)
        ask_px = _num(mbp, apx_col, default=np.nan)
        bid_sz = _num(mbp, bsz_col, default=0.0)
        ask_sz = _num(mbp, asz_col, default=0.0)
        denom = np.maximum(bid_sz + ask_sz, 1e-12)
        features[f"bid_size_{level:02d}"] = bid_sz.astype(np.float32)
        features[f"ask_size_{level:02d}"] = ask_sz.astype(np.float32)
        features[f"level_imbalance_{level:02d}"] = ((bid_sz - ask_sz) / denom).astype(np.float32)
        features[f"bid_distance_ticks_{level:02d}"] = ((mid - bid_px).fillna(0.0) / max(float(tick_size), 1e-12)).astype(np.float32)
        features[f"ask_distance_ticks_{level:02d}"] = ((ask_px - mid).fillna(0.0) / max(float(tick_size), 1e-12)).astype(np.float32)


def _add_rolling_features(features: pd.DataFrame, *, tick_size: float) -> None:
    mid = pd.to_numeric(features["mid_price"], errors="coerce")
    price_change = mid.diff().fillna(0.0)
    returns = mid.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["price_change"] = price_change.astype(np.float32)
    features["return_1"] = returns.astype(np.float32)
    features["tick_return_1"] = (price_change / max(float(tick_size), 1e-12)).astype(np.float32)
    for window in ROLLING_WINDOWS:
        min_periods = min(max(2, window // 4), window)
        features[f"rolling_return_mean_{window}"] = returns.rolling(window, min_periods=min_periods).mean().fillna(0.0).astype(np.float32)
        features[f"realized_vol_{window}"] = price_change.abs().rolling(window, min_periods=min_periods).std(ddof=0).fillna(float(tick_size)).astype(np.float32)
        features[f"rolling_spread_mean_{window}"] = features["spread"].rolling(window, min_periods=1).mean().fillna(0.0).astype(np.float32)
        features[f"rolling_spread_std_{window}"] = features["spread"].rolling(window, min_periods=min_periods).std(ddof=0).fillna(0.0).astype(np.float32)
        features[f"rolling_obi_mean_{window}"] = features["order_book_imbalance"].rolling(window, min_periods=1).mean().fillna(0.0).astype(np.float32)
        features[f"rolling_depth_imbalance_mean_{window}"] = features["depth_imbalance"].rolling(window, min_periods=1).mean().fillna(0.0).astype(np.float32)
        if "mlofi_sum" in features.columns:
            features[f"rolling_mlofi_sum_mean_{window}"] = features["mlofi_sum"].rolling(window, min_periods=1).mean().fillna(0.0).astype(np.float32)
        if "mbo_trade_count_since_prev_mbp" in features.columns:
            features[f"rolling_trade_count_{window}"] = features["mbo_trade_count_since_prev_mbp"].rolling(window, min_periods=1).sum().fillna(0.0).astype(np.float32)
    features["realized_vol"] = features.get("realized_vol_50", features["price_change"].abs()).fillna(float(tick_size)).astype(np.float32)


def _add_session_features(features: pd.DataFrame) -> None:
    ts = _coerce_ts(features["ts_event"])
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    features["hour"] = ts.dt.hour.astype(np.int16)
    features["minute"] = ts.dt.minute.astype(np.int16)
    features["second"] = ts.dt.second.astype(np.int16)
    features["day_of_week"] = ts.dt.dayofweek.astype(np.int16)
    features["minute_of_day"] = minute_of_day.astype(np.int16)
    angle = 2.0 * np.pi * minute_of_day / 1440.0
    features["hour_sin"] = np.sin(angle).astype(np.float32)
    features["hour_cos"] = np.cos(angle).astype(np.float32)
    features["london_active"] = ((minute_of_day >= 8 * 60) & (minute_of_day < 13 * 60)).astype(np.int8)
    features["ny_active"] = ((minute_of_day >= 13 * 60) & (minute_of_day < 21 * 60)).astype(np.int8)
    features["overlap_active"] = ((minute_of_day >= 13 * 60) & (minute_of_day < 16 * 60)).astype(np.int8)


def _add_mbo_interval_features(features: pd.DataFrame) -> None:
    cumulative_pairs = {
        "mbo_cvd": "mbo_signed_trade_size_since_prev_mbp",
        "mbo_cum_buy_volume": "mbo_buy_trade_size_since_prev_mbp",
        "mbo_cum_sell_volume": "mbo_sell_trade_size_since_prev_mbp",
        "mbo_cum_trade_count": "mbo_trade_count_since_prev_mbp",
    }
    for source_col, out_col in cumulative_pairs.items():
        if source_col in features.columns:
            cumulative = pd.to_numeric(features[source_col], errors="coerce").fillna(0.0)
            features[out_col] = cumulative.diff().fillna(cumulative).astype(np.float32)
    if "mbo_cvd" in features.columns:
        features["mbo_cvd_momentum_20"] = features["mbo_cvd"].diff().rolling(20, min_periods=1).sum().fillna(0.0).astype(np.float32)

    ts = _coerce_ts(features["ts_event"])
    if "mbo_state_ts" in features.columns:
        mbo_ts = _coerce_ts(features["mbo_state_ts"])
        age_ms = (ts - mbo_ts).dt.total_seconds() * 1000.0
        features["mbo_state_age_ms"] = age_ms.fillna(-1.0).astype(np.float32)
    else:
        features["mbo_state_age_ms"] = -1.0
    features["inter_event_time"] = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0).astype(np.float32)


def _add_v19_compat_features(features: pd.DataFrame, *, tick_size: float) -> tuple[pd.DataFrame, dict]:
    def as_float_series(value) -> pd.Series:
        if isinstance(value, pd.Series):
            series = value
        else:
            series = pd.Series(value, index=features.index)
        return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32)

    total_depth = pd.to_numeric(features.get("bid_depth", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(features.get("ask_depth", 0.0), errors="coerce").fillna(0.0)
    spread_ticks = pd.to_numeric(features.get("spread", 0.0), errors="coerce").fillna(0.0) / max(float(tick_size), 1e-12)
    vnet = pd.to_numeric(features.get("mbo_signed_trade_size_since_prev_mbp", 0.0), errors="coerce").fillna(0.0)
    price_change = pd.to_numeric(features.get("price_change", 0.0), errors="coerce").fillna(0.0)
    cvd = pd.to_numeric(features.get("mbo_cvd", 0.0), errors="coerce").fillna(0.0)

    mappings = {
        "cvd": cvd,
        "obi": features.get("order_book_imbalance", 0.0),
        "micro_price": features.get("microprice", features.get("mid_price", 0.0)),
        "lob_depth_imbalance": features.get("depth_imbalance", 0.0),
        "micro_atr": features.get("realized_vol", 0.0),
        "volume_burst": pd.to_numeric(features.get("mbo_trade_count_since_prev_mbp", 0.0), errors="coerce").fillna(0.0),
        "inter_event_time": features.get("inter_event_time", 0.0),
        "liquidity_density": (total_depth / np.maximum(spread_ticks, 1.0)).astype(np.float32),
        "vnet": vnet,
        "cvd_momentum": features.get("mbo_cvd_momentum_20", 0.0),
        "cvd_price_divergence": (cvd.diff().fillna(0.0) * np.sign(price_change.fillna(0.0))).astype(np.float32),
        "trend_strength": (
            pd.to_numeric(features.get("rolling_return_mean_20", 0.0), errors="coerce").fillna(0.0)
            / np.maximum(pd.to_numeric(features.get("realized_vol_20", 0.0), errors="coerce").fillna(float(tick_size)), float(tick_size))
        ).astype(np.float32),
        "liquidity_sweep": (
            pd.to_numeric(features.get("mlofi_top3", 0.0), errors="coerce").abs().fillna(0.0)
            / np.maximum(total_depth, 1.0)
        ).astype(np.float32),
        "kyle_lambda": (price_change.abs() / np.maximum(vnet.abs(), 1.0)).astype(np.float32),
        "hawkes_intensity": features.get("rolling_trade_count_20", 0.0),
    }
    derived = []
    placeholders = []
    compat_cols: dict[str, pd.Series] = {}
    for name in V19_COMPAT_FEATURES:
        if name in mappings:
            compat_cols[name] = as_float_series(mappings[name])
            derived.append(name)
        elif name in features.columns:
            compat_cols[name] = as_float_series(features[name])
        else:
            compat_cols[name] = pd.Series(0.0, index=features.index, dtype=np.float32)
            placeholders.append(name)
    for name in V19_COMPAT_FEATURES:
        compat_cols[f"raw__{name}"] = as_float_series(compat_cols[name])
    compat_frame = pd.DataFrame(compat_cols, index=features.index)
    overlap = [col for col in compat_frame.columns if col in features.columns]
    base = features.drop(columns=overlap) if overlap else features
    out = pd.concat([base, compat_frame], axis=1).copy()
    return out, {"derived_v19_features": derived, "placeholder_zero_v19_features": placeholders}


def _build_features(aligned: pd.DataFrame, *, tick_size: float) -> tuple[pd.DataFrame, dict]:
    mbp = aligned.sort_values("ts_event").reset_index(drop=True).copy()
    features = pd.DataFrame(index=mbp.index)
    for col in METADATA_COLUMNS:
        if col in mbp.columns:
            features[col] = mbp[col]
    features["feature_ts"] = mbp["ts_event"]
    features["label_start_ts"] = mbp["ts_event"]

    lob = compute_lob_features(mbp, levels=LEVELS)
    mlofi = compute_mlofi(mbp, levels=LEVELS)
    features = pd.concat([features, lob, mlofi], axis=1)
    features["microprice"] = pd.to_numeric(features["microprice"], errors="coerce").fillna(features["mid_price"]).astype(np.float32)
    _add_level_features(features, mbp, tick_size=tick_size)
    features = features.copy()

    mbo_cols = [col for col in mbp.columns if str(col).startswith("mbo_")]
    for col in mbo_cols:
        features[col] = mbp[col]
    _add_mbo_interval_features(features)
    features = features.copy()
    _add_rolling_features(features, tick_size=tick_size)
    features = features.copy()
    _add_session_features(features)
    features = features.copy()

    features["price"] = features["mid_price"].astype(np.float32)
    features["open"] = features["mid_price"].astype(np.float32)
    features["high"] = features["mid_price"].astype(np.float32)
    features["low"] = features["mid_price"].astype(np.float32)
    features["close"] = features["mid_price"].astype(np.float32)
    features["micro_price"] = features["microprice"].astype(np.float32)
    features["obi"] = features["order_book_imbalance"].astype(np.float32)
    features["liq_score"] = (1.0 / np.maximum(pd.to_numeric(features["spread"], errors="coerce").fillna(float(tick_size)), float(tick_size))).astype(np.float32)
    features, compat_report = _add_v19_compat_features(features.copy(), tick_size=tick_size)

    critical = ("ts_event", "feature_ts", "label_start_ts", "mid_price", "spread", "microprice", "price")
    critical_nulls = {col: int(features[col].isna().sum()) for col in critical if col in features.columns}
    report = {
        "rows": int(len(features)),
        "critical_nulls": critical_nulls,
        "feature_columns": [str(c) for c in features.columns if c not in METADATA_COLUMNS],
        "v19_compatibility_features": compat_report,
        "passed": bool(not any(critical_nulls.values())),
    }
    return features.reset_index(drop=True), report


def _label_frame(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    cfg = TripleBarrierConfig(
        horizon_rows=int(args.horizon),
        tick_size=float(args.tick_size),
        tp_vol_mult=float(args.tp_mult),
        sl_vol_mult=float(args.sl_mult),
        neutral_mult=float(args.neutral_mult),
        round_trip_cost_ticks=float(args.round_trip_cost_ticks),
        spread_cost_mult=float(args.spread_cost_mult),
        volatility_col="realized_vol",
        price_col="mid_price",
    )
    labeled = build_triple_barrier_labels(features, cfg)
    labels = labeled[[col for col in LABEL_COLUMNS if col in labeled.columns]].copy()
    tradeable = pd.to_numeric(labels.get("tradeability_label", 0), errors="coerce").fillna(0).astype(np.int8)
    labels["signal_quality"] = tradeable
    labels["conf_label"] = tradeable
    labels["soft_label"] = np.float32(0.5)
    labels["soft_sample_weight"] = np.float32(1.0)
    labels["soft_label_confidence"] = np.float32(0.0)
    labels["mc_sample_weight"] = np.float32(1.0)
    labels["label_stability"] = np.float32(1.0)
    truncated = int((pd.to_numeric(labels["label_horizon_steps"], errors="coerce").fillna(0) < int(args.horizon)).sum())
    report = {
        "config": cfg.to_dict(),
        "rows": int(len(labels)),
        "truncated_horizon_rows": truncated,
        "barrier_hit_counts": _top_counts(labels, "barrier_hit_type", limit=20),
        "passed": bool("label_end_ts" in labels.columns and labels["label_end_ts"].notna().all()),
    }
    return labels, report


def _compatibility_only_feature_columns(feature_report: dict) -> list[str]:
    placeholders = (
        feature_report.get("v19_compatibility_features", {}).get("placeholder_zero_v19_features", [])
        if isinstance(feature_report, dict)
        else []
    )
    columns: list[str] = []
    for name in placeholders:
        text = str(name)
        columns.append(text)
        columns.append(f"raw__{text}")
    return sorted(set(columns))


def _feature_columns(df: pd.DataFrame, *, exclude: list[str] | tuple[str, ...] | set[str] | None = None) -> list[str]:
    forbidden = set(METADATA_COLUMNS) | set(LABEL_COLUMNS)
    forbidden.update(str(col) for col in (exclude or ()))
    return [str(col) for col in df.columns if str(col) not in forbidden and not str(col).startswith("label_")]


def _write_artifact_parts(
    output_dir: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    combined: pd.DataFrame,
    metadata_cols: list[str],
    manifest_result,
) -> dict:
    features_path = write_table(features, os.path.join(output_dir, "features.parquet"))
    labels_path = write_table(labels, os.path.join(output_dir, "labels.parquet"))
    metadata = combined[[col for col in metadata_cols if col in combined.columns]].copy()
    metadata_path = write_table(metadata, os.path.join(output_dir, "metadata.parquet"))
    metadata_json_path = os.path.join(output_dir, "metadata.json")
    metadata_payload = {
        "rows": int(len(metadata)),
        "columns": list(metadata.columns),
        "ts_min": str(pd.to_datetime(combined["ts_event"], errors="coerce").min()) if len(combined) else None,
        "ts_max": str(pd.to_datetime(combined["ts_event"], errors="coerce").max()) if len(combined) else None,
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(metadata_payload), f, indent=2)

    with open(manifest_result.manifest_path, encoding="utf-8") as f:
        manifest_payload = json.load(f)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(manifest_payload), f, indent=2)
    return {
        "features_parquet": os.path.abspath(features_path),
        "labels_parquet": os.path.abspath(labels_path),
        "metadata_parquet": os.path.abspath(metadata_path),
        "metadata_json": os.path.abspath(metadata_json_path),
        "artifact_manifest_json": os.path.abspath(manifest_result.manifest_path),
        "manifest_json": os.path.abspath(manifest_path),
        "final_feature_shards": manifest_result.to_dict().get("shards", []),
    }


def _write_date_partitions(
    output_dir: str,
    combined: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_columns: list[str],
    metadata_columns: list[str],
) -> list[dict]:
    if "ts_event" not in combined.columns or combined.empty:
        return []
    partitions: list[dict] = []
    root = os.path.join(output_dir, "partitions")
    dates = _coerce_ts(combined["ts_event"]).dt.strftime("%Y-%m-%d")
    for date_value in sorted(dates.dropna().unique()):
        mask = dates == date_value
        part_dir = os.path.join(root, f"date={date_value}")
        os.makedirs(part_dir, exist_ok=True)
        rows = int(mask.sum())
        features_path = write_table(combined.loc[mask, feature_columns].copy(), os.path.join(part_dir, "features.parquet"))
        labels_path = write_table(combined.loc[mask, label_columns].copy(), os.path.join(part_dir, "labels.parquet"))
        metadata_path = write_table(combined.loc[mask, metadata_columns].copy(), os.path.join(part_dir, "metadata.parquet"))
        partitions.append({
            "date": str(date_value),
            "rows": rows,
            "features": os.path.abspath(features_path),
            "labels": os.path.abspath(labels_path),
            "metadata": os.path.abspath(metadata_path),
        })
    return partitions


def _build_scalability_report(
    *,
    args: argparse.Namespace,
    started_perf: float,
    data_report: dict,
    aligned: pd.DataFrame,
    features: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    combined: pd.DataFrame | None = None,
    artifact_paths: dict | None = None,
) -> dict:
    elapsed = float(time.perf_counter() - started_perf)
    aligned_rows = int(len(aligned))
    feature_rows = int(len(features)) if features is not None else 0
    combined_rows = int(len(combined)) if combined is not None else 0
    combined_mem_gb = _dataframe_memory_gb(combined)
    aligned_mem_gb = _dataframe_memory_gb(aligned)
    features_mem_gb = _dataframe_memory_gb(features)
    labels_mem_gb = _dataframe_memory_gb(labels)
    max_memory_gb = float(getattr(args, "max_memory_gb", 0.0) or 0.0)
    warnings: list[str] = []
    critical: list[str] = []

    concat_rows = combined_rows or aligned_rows
    concat_mem = combined_mem_gb or (aligned_mem_gb + features_mem_gb + labels_mem_gb)
    if concat_rows > CONCAT_ROWS_WARNING:
        warnings.append("concat_row_count_gt_5m")
    if concat_mem > CONCAT_MEMORY_WARNING_GB:
        warnings.append("concat_memory_estimate_gt_8gb")
    if max_memory_gb > 0 and concat_mem > 0.70 * max_memory_gb:
        warnings.append("concat_memory_estimate_gt_70pct_of_max_memory")

    rows_per_day = None
    estimates = {}
    date_range = data_report.get("row_counts", {})
    if combined is not None and "ts_event" in combined.columns and len(combined):
        ts = _coerce_ts(combined["ts_event"])
        span_days = max(float((ts.max() - ts.min()).total_seconds()) / 86_400.0, 1.0)
        rows_per_day = float(len(combined) / span_days)
        bytes_per_row = _dir_size_bytes(args.output) / max(len(combined), 1) if os.path.exists(args.output) else 0.0
        for days in (30, 90, 180):
            est_rows = int(rows_per_day * days)
            estimates[f"{days}d"] = {
                "estimated_rows": est_rows,
                "estimated_artifact_gb": float(est_rows * bytes_per_row / (1024.0 ** 3)),
                "estimated_concat_memory_gb": float((combined_mem_gb / max(len(combined), 1)) * est_rows) if combined_mem_gb else None,
            }
    recommendation = "OK for current run. Validate real multi-day MBP coverage before scaling."
    if warnings or concat_rows > CONCAT_ROWS_WARNING:
        recommendation = "Use day/date partitioned streaming before 3-6 month preparation."
    if estimates.get("180d", {}).get("estimated_concat_memory_gb") and estimates["180d"]["estimated_concat_memory_gb"] > max(max_memory_gb, CONCAT_MEMORY_WARNING_GB):
        recommendation = "3-6 month preparation requires partitioned streaming before full run."

    if getattr(args, "max_rows", 0) and aligned_rows > int(args.max_rows):
        critical.append("max_rows_exceeded")

    artifact_size = _file_size_report(args.output) if artifact_paths is not None else {"total_bytes": 0, "total_mb": 0.0, "files": []}
    return {
        "generated_at": _utc_now(),
        "elapsed_seconds": elapsed,
        "process_rss_gb": _current_rss_gb(),
        "max_memory_gb": max_memory_gb or None,
        "row_estimates_after_filter": {
            "mbo_rows_after_filters": int(data_report.get("row_counts", {}).get("mbo_rows_after_filters", 0)),
            "mbp_rows_after_filters": int(data_report.get("row_counts", {}).get("mbp_rows_after_filters", 0)),
            "feature_clock_rows": aligned_rows,
            "feature_rows_written": feature_rows,
            "combined_rows_written": combined_rows,
        },
        "memory_estimates_gb": {
            "aligned_frame": aligned_mem_gb,
            "features_frame": features_mem_gb,
            "labels_frame": labels_mem_gb,
            "combined_frame": combined_mem_gb,
            "concat_estimate": concat_mem,
        },
        "artifact_size": artifact_size,
        "rows_per_day_estimate": rows_per_day,
        "scale_estimates": estimates,
        "concat_step_is_bottleneck": bool(warnings),
        "warnings": warnings,
        "critical": critical,
        "recommendation": recommendation,
        "passed": bool(not critical),
    }


def _build_leakage_report(combined: pd.DataFrame, feature_cols: list[str]) -> dict:
    timestamp_report = assert_label_timestamps(combined, context="prepare_v20.artifact")
    schema_report = validate_artifact_schema(combined, feature_columns=[f"raw__{c}" for c in V19_COMPAT_FEATURES if f"raw__{c}" in combined.columns]).to_dict()
    feature_ts = _coerce_ts(combined["feature_ts"]) if "feature_ts" in combined.columns else _coerce_ts(combined["ts_event"])
    label_start = _coerce_ts(combined["label_start_ts"]) if "label_start_ts" in combined.columns else _coerce_ts(combined["ts_event"])
    feature_after_start = int((feature_ts > label_start).sum())
    split_report = {"skipped": True, "reason": "not_enough_rows"}
    if len(combined) >= 50:
        splits, wf_report = build_purged_walkforward_splits(
            combined["ts_event"],
            combined["label_end_ts"],
            PurgedWalkForwardConfig(n_folds=2, initial_train_frac=0.6, test_frac=0.2, min_train_rows=10, min_test_rows=5),
        )
        split_report = validate_time_splits(splits, combined["ts_event"], combined["label_end_ts"], context="prepare_v20.split_precheck", strict=False)
        split_report["builder_report"] = wf_report
    report = {
        "generated_at": _utc_now(),
        "timestamp_report": timestamp_report,
        "artifact_schema_report": schema_report,
        "feature_ts_after_label_start_rows": feature_after_start,
        "feature_generation_policy": "MBP current snapshot + previous rows only; MBO aligned with merge_asof(direction='backward')",
        "split_safety_precheck": split_report,
        "feature_columns_checked": feature_cols,
        "passed": bool(timestamp_report.get("passed") and schema_report.get("passed") and feature_after_start == 0),
    }
    return report


def _train_v19_compatibility_report(output_dir: str, combined: pd.DataFrame, compat_report: dict) -> dict:
    minimal = ("ts_event", "label_end_ts", "bias_label", "train_event_flag", "price", "close")
    required_present = {col: bool(col in combined.columns) for col in minimal}
    v19_present = {col: bool(col in combined.columns) for col in V19_COMPAT_FEATURES}
    raw_present = {f"raw__{col}": bool(f"raw__{col}" in combined.columns) for col in V19_COMPAT_FEATURES}
    loader_ok = False
    loader_error = None
    loaded_rows = 0
    try:
        loaded = load_feature_artifact(output_dir, columns=list(minimal) + ["cvd", "obi", "raw__cvd", "raw__obi"])
        loader_ok = True
        loaded_rows = int(len(loaded))
    except Exception as exc:
        loader_error = str(exc)
    placeholder = compat_report.get("v19_compatibility_features", {}).get("placeholder_zero_v19_features", [])
    compatibility_only = _compatibility_only_feature_columns(compat_report)
    report = {
        "generated_at": _utc_now(),
        "loader": "modules.feature_artifact_v19.load_feature_artifact",
        "loader_ok": loader_ok,
        "loader_error": loader_error,
        "catboost_dependency": _catboost_dependency_report(),
        "loaded_rows": loaded_rows,
        "minimal_columns_present": required_present,
        "v19_stat_features_present": v19_present,
        "v19_raw_stat_features_present": raw_present,
        "placeholder_zero_v19_features": placeholder,
        "compatibility_only_feature_columns": compatibility_only,
        "placeholder_features_excluded_from_training": bool(compatibility_only),
        "warning": (
            "Artifact is loadable by train_v19. Placeholder compatibility columns are causal zeros where "
            "V20 preparation does not yet implement the exact V19 heuristic feature. These columns are "
            "excluded from canonical v20 feature_columns and kept only for V19 loader compatibility."
            if placeholder
            else None
        ),
        "passed": bool(loader_ok and all(required_present.values()) and all(v19_present.values()) and all(raw_present.values())),
    }
    return report


def _selected_symbol_contract(df: pd.DataFrame, requested_symbol: str | None) -> tuple[str | None, str | None]:
    symbol = requested_symbol
    contract = None
    for col in ("contract_symbol", "symbol", "instrument_id"):
        if col in df.columns and len(df):
            values = df[col].dropna().astype(str).unique()
            if len(values) == 1:
                if col == "contract_symbol":
                    contract = values[0]
                elif symbol is None:
                    symbol = values[0]
    return symbol, contract


def run(args: argparse.Namespace) -> dict:
    started_perf = time.perf_counter()
    if getattr(args, "validation_only", False):
        args.dry_run = True
    os.makedirs(args.output, exist_ok=True)

    mbo_loaded = _load_feed(args.mbo, feed=FeedType.MBO, args=args)
    mbp_loaded = _load_feed(args.mbp, feed=FeedType.MBP, args=args)
    mbp_clean, mbp_quality = _reject_invalid_mbp_rows(mbp_loaded.frame, tick_size=float(args.tick_size))
    mbo_state, mbo_flow_report = _build_mbo_flow_state(mbo_loaded.frame)
    aligned, alignment_report = _align_mbo_to_mbp(mbp_clean, mbo_state)
    mbo_reliability = _mbo_flow_reliability_report(alignment_report)

    data_report = {
        "generated_at": _utc_now(),
        "phase": "v20_phase2_prepare",
        "dry_run": bool(args.dry_run),
        "inputs": {"mbo": os.path.abspath(args.mbo), "mbp": os.path.abspath(args.mbp)},
        "mbo": mbo_loaded.report,
        "mbp": mbp_loaded.report,
        "mbp_quality": mbp_quality,
        "mbo_flow": mbo_flow_report,
        "alignment": alignment_report,
        "feature_reliability": mbo_reliability,
        "row_counts": {
            "mbo_rows_after_filters": int(len(mbo_loaded.frame)),
            "mbp_rows_after_filters": int(len(mbp_loaded.frame)),
            "mbp_rows_after_quality": int(len(mbp_clean)),
            "aligned_rows": int(len(aligned)),
        },
        "row_estimate_after_date_symbol_filter": {
            "mbo": int(len(mbo_loaded.frame)),
            "mbp": int(len(mbp_loaded.frame)),
            "feature_clock": int(len(aligned)),
        },
        "warnings": [],
        "critical": [],
    }
    if int(mbp_quality["rows_rejected"]) > 0:
        data_report["warnings"].append("mbp_rows_rejected_for_invalid_bbo_or_size")
    if float(alignment_report.get("match_rate", 0.0)) < 0.50:
        data_report["warnings"].append("low_mbo_to_mbp_alignment_match_rate")
    data_report["warnings"].extend(alignment_report.get("warnings", []))
    data_report["critical"].extend(alignment_report.get("critical", []))
    max_rows = int(getattr(args, "max_rows", 0) or 0)
    if max_rows > 0 and len(aligned) > max_rows:
        data_report["critical"].append("max_rows_exceeded")
        data_report["warnings"].append("max_rows_guard_triggered")
    try:
        _enforce_memory_guard(args, stage="after_alignment", report=data_report)
    except MemoryError as exc:
        data_report["critical"].append("max_memory_gb_exceeded_after_alignment")
        data_report["warnings"].append(str(exc))
    data_report["passed"] = bool(len(aligned) > 0 and mbp_quality["output_rows"] > 0 and "max_rows_exceeded" not in data_report["critical"] and "max_memory_gb_exceeded_after_alignment" not in data_report["critical"])
    data_report_path = _write_json_report(data_report, args.output, "data_validation_report.json")
    scalability_report = _build_scalability_report(args=args, started_perf=started_perf, data_report=data_report, aligned=aligned)
    scalability_report_path = _write_json_report(scalability_report, args.output, "scalability_report.json")

    if args.dry_run:
        summary = {
            "generated_at": _utc_now(),
            "dry_run": True,
            "data_validation_report": data_report_path,
            "scalability_report": scalability_report_path,
            "artifact_written": False,
            "passed": bool(data_report["passed"]),
        }
        _write_json_report(summary, args.output, "manifest.json")
        return summary

    if not data_report["passed"]:
        raise RuntimeError("prepare_v20 data validation failed before feature generation")

    features, feature_report = _build_features(aligned, tick_size=float(args.tick_size))
    feature_report["feature_reliability"] = mbo_reliability
    try:
        _enforce_memory_guard(args, stage="after_features", report=feature_report)
    except MemoryError as exc:
        feature_report.setdefault("critical", []).append("max_memory_gb_exceeded_after_features")
        feature_report.setdefault("warnings", []).append(str(exc))
        feature_report["passed"] = False
    if not feature_report["passed"]:
        _write_json_report(feature_report, args.output, "feature_validation_report.json")
        raise RuntimeError(f"prepare_v20 feature validation failed: {feature_report['critical_nulls']}")

    labels, label_report = _label_frame(features, args)
    combined = pd.concat([features.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    combined = combined.sort_values("ts_event").reset_index(drop=True)
    try:
        _enforce_memory_guard(args, stage="after_combined", report=feature_report)
    except MemoryError as exc:
        feature_report.setdefault("critical", []).append("max_memory_gb_exceeded_after_combined")
        feature_report.setdefault("warnings", []).append(str(exc))
        feature_report["passed"] = False
        _write_json_report(feature_report, args.output, "feature_validation_report.json")
        raise
    metadata_cols = [col for col in METADATA_COLUMNS if col in combined.columns]
    compatibility_only_cols = _compatibility_only_feature_columns(feature_report)
    feature_cols = _feature_columns(combined, exclude=compatibility_only_cols)
    feature_report["compatibility_only_feature_columns"] = compatibility_only_cols
    feature_report["placeholder_features_excluded_from_training"] = bool(compatibility_only_cols)
    feature_report["canonical_feature_columns"] = feature_cols

    label_dist_report = summarize_label_distribution(combined, context="prepare_v20.artifact")
    leakage_report = _build_leakage_report(combined, feature_cols)
    feature_report["artifact_schema_report"] = leakage_report["artifact_schema_report"]
    feature_report["null_counts_top20"] = {
        str(k): int(v)
        for k, v in combined[feature_cols].isna().sum().sort_values(ascending=False).head(20).items()
    }
    feature_report["passed"] = bool(feature_report["passed"] and leakage_report["passed"])

    reports = {
        "data_validation_report": "data_validation_report.json",
        "feature_validation_report": "feature_validation_report.json",
        "label_distribution_report": "label_distribution_report.json",
        "leakage_precheck_report": "leakage_precheck_report.json",
        "train_v19_compatibility_report": "train_v19_compatibility_report.json",
        "scalability_report": "scalability_report.json",
    }
    symbol, contract = _selected_symbol_contract(combined, getattr(args, "symbol", None))
    date_range = {
        "start": str(pd.to_datetime(combined["ts_event"], errors="coerce").min()) if len(combined) else None,
        "end": str(pd.to_datetime(combined["ts_event"], errors="coerce").max()) if len(combined) else None,
    }
    label_params = {
        "horizon_rows": int(args.horizon),
        "tp_mult": float(args.tp_mult),
        "sl_mult": float(args.sl_mult),
        "neutral_mult": float(args.neutral_mult),
        "round_trip_cost_ticks": float(args.round_trip_cost_ticks),
        "spread_cost_mult": float(args.spread_cost_mult),
    }
    partition_records = []
    if bool(getattr(args, "write_partitions", False)):
        partition_records = _write_date_partitions(
            args.output,
            combined,
            feature_columns=list(features.columns),
            label_columns=list(labels.columns),
            metadata_columns=metadata_cols,
        )
    manifest_result = write_feature_artifact(
        combined,
        args.output,
        rows_per_shard=int(args.rows_per_shard),
        config={
            "schema_version": SCHEMA_VERSION,
            "levels": LEVELS,
            "rolling_windows": list(ROLLING_WINDOWS),
            "strict": bool(args.strict),
            "sample_rows": int(args.sample_rows or 0),
            "chunk_rows": int(args.chunk_rows or 0),
            "max_rows": max_rows or None,
            "max_memory_gb": float(getattr(args, "max_memory_gb", 0.0) or 0.0) or None,
            "write_partitions": bool(getattr(args, "write_partitions", False)),
        },
        reports=reports,
        inputs={"mbo": os.path.abspath(args.mbo), "mbp": os.path.abspath(args.mbp)},
        row_counts=data_report["row_counts"],
        symbol=symbol,
        contract=contract,
        date_range=date_range,
        metadata_columns=metadata_cols,
        tick_size=float(args.tick_size),
        horizon=int(args.horizon),
        label_params=label_params,
        mbo_flow_features_reliable=bool(mbo_reliability["mbo_flow_features_reliable"]),
        compatibility_only_feature_columns=compatibility_only_cols,
        git_commit=_git_commit(),
        extra={
            "schema_version": SCHEMA_VERSION,
            "feature_parquet": "features.parquet",
            "labels_parquet": "labels.parquet",
            "metadata_parquet": "metadata.parquet",
            "label_columns": list(labels.columns),
            "feature_columns": feature_cols,
            "compatibility_only_feature_columns": compatibility_only_cols,
            "placeholder_features_excluded_from_training": bool(compatibility_only_cols),
            "mbo_flow_features_reliable": bool(mbo_reliability["mbo_flow_features_reliable"]),
            "feature_reliability": mbo_reliability,
            "date_partitions": partition_records,
        },
    )
    artifact_paths = _write_artifact_parts(args.output, features, labels, combined, metadata_cols, manifest_result)

    train_compat_report = _train_v19_compatibility_report(args.output, combined, feature_report)
    scalability_report = _build_scalability_report(
        args=args,
        started_perf=started_perf,
        data_report=data_report,
        aligned=aligned,
        features=features,
        labels=labels,
        combined=combined,
        artifact_paths=artifact_paths,
    )
    _write_json_report(feature_report, args.output, "feature_validation_report.json")
    _write_json_report(label_dist_report, args.output, "label_distribution_report.json")
    _write_json_report(leakage_report, args.output, "leakage_precheck_report.json")
    _write_json_report(train_compat_report, args.output, "train_v19_compatibility_report.json")
    _write_json_report(scalability_report, args.output, "scalability_report.json")

    summary = {
        "generated_at": _utc_now(),
        "dry_run": False,
        "rows": int(len(combined)),
        "paths": artifact_paths,
        "reports": {name: os.path.abspath(os.path.join(args.output, filename)) for name, filename in reports.items()},
        "mbo_flow_features_reliable": bool(mbo_reliability["mbo_flow_features_reliable"]),
        "passed": bool(data_report["passed"] and feature_report["passed"] and leakage_report["passed"] and train_compat_report["passed"] and scalability_report["passed"]),
        "train_v19_compatible": bool(train_compat_report["passed"]),
    }
    _write_json_report(summary, args.output, "prepare_v20_summary.json")

    if args.strict and not summary["passed"]:
        raise RuntimeError("prepare_v20 completed with failed validation in --strict mode")
    print(f"prepare_v20 artifact written: {os.path.abspath(args.output)} rows={len(combined):,}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QuantSystem v20 leakage-safe data preparation")
    p.add_argument("--mbo", required=True, help="Raw Databento MBO CSV/parquet path")
    p.add_argument("--mbp", required=True, help="Raw Databento MBP-10 CSV/parquet path")
    p.add_argument("--output", required=True, help="Output artifact directory")
    p.add_argument("--symbol", default=None, help="Optional exact symbol/contract/instrument filter")
    p.add_argument("--start", default=None, help="Optional inclusive start timestamp")
    p.add_argument("--end", default=None, help="Optional inclusive end timestamp")
    p.add_argument("--tick_size", type=float, default=0.0001, help="6B default tick size")
    p.add_argument("--horizon", type=int, default=50, help="Triple-barrier horizon in feature rows")
    p.add_argument("--tp_mult", type=float, default=1.5, help="Take-profit volatility multiplier")
    p.add_argument("--sl_mult", type=float, default=1.0, help="Stop-loss volatility multiplier")
    p.add_argument("--neutral_mult", type=float, default=0.5, help="Neutral-zone volatility/cost multiplier")
    p.add_argument("--round_trip_cost_ticks", type=float, default=1.0, help="Minimum round-trip cost in ticks")
    p.add_argument("--spread_cost_mult", type=float, default=1.0, help="Spread contribution to label barrier floor")
    p.add_argument("--chunk_rows", type=int, default=500_000, help="CSV read chunk size")
    p.add_argument("--sample_rows", type=int, default=0, help="Limit rows per input feed for smoke tests")
    p.add_argument("--max_rows", type=int, default=0, help="Guardrail: refuse artifact generation if filtered MBP feature rows exceed this count")
    p.add_argument("--max_memory_gb", type=float, default=0.0, help="Optional process-memory guard; 0 disables it")
    p.add_argument("--rows_per_shard", type=int, default=250_000, help="Rows per final/features_*.parquet shard")
    p.add_argument("--write_partitions", action="store_true", help="Also write date-partitioned feature/label/metadata parquet files")
    p.add_argument("--dry_run", action="store_true", help="Run validation/alignment checks without writing train artifacts")
    p.add_argument("--validation_only", action="store_true", help="Deprecated alias for --dry_run")
    p.add_argument("--strict", action="store_true", help="Fail on validation warnings that block production training")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
