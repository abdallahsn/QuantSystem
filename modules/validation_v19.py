"""
validation_v19.py - lightweight production guards for QuantSystem V19.

These checks intentionally sit outside model code. They validate chronology,
labels, leakage boundaries, and execution assumptions before expensive training
or backtesting work starts.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_NAMES = {0: "LONG", 1: "SHORT", 2: "NEUTRAL"}


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ts(series: pd.Series, *, context: str) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    if out.isna().any():
        bad = int(out.isna().sum())
        raise ValueError(f"Invalid timestamps in {context}: null_or_unparseable={bad:,}")
    return out


def _num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def write_validation_report(report: dict, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def validate_market_data_frame(
    df: pd.DataFrame,
    *,
    context: str,
    tick_size: float | None = None,
    strict: bool = False,
) -> dict:
    """Cheap in-memory market-data checks for already-loaded frames."""
    issues: list[dict] = []
    report = {
        "generated_at": _utc_now(),
        "context": context,
        "rows": int(len(df)),
        "issues": issues,
        "passed": True,
    }
    if df is None or df.empty:
        issues.append({"severity": "critical", "code": "empty_frame", "message": "input frame is empty"})
        report["passed"] = False
        if strict:
            raise ValueError(f"{context}: empty input frame")
        return report

    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
        null_ts = int(ts.isna().sum())
        if null_ts:
            issues.append({"severity": "critical", "code": "invalid_ts_event", "count": null_ts})
        if not ts.dropna().is_monotonic_increasing:
            issues.append({"severity": "critical", "code": "ts_event_not_monotonic"})
        dup_ts = int(ts.duplicated().sum())
        if dup_ts:
            issues.append({"severity": "medium", "code": "duplicate_ts_event", "count": dup_ts})

    price_cols = [c for c in ("price", "close", "open", "high", "low") if c in df.columns]
    invalid_price: dict[str, int] = {}
    for col in price_cols:
        vals = _num(df[col], default=np.nan)
        bad = int((~np.isfinite(vals.to_numpy(dtype=np.float64)) | (vals <= 0)).sum())
        if bad:
            invalid_price[col] = bad
    if invalid_price:
        issues.append({"severity": "critical", "code": "invalid_prices", "columns": invalid_price})

    if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
        bid = _num(df["bid_px_00"], default=np.nan)
        ask = _num(df["ask_px_00"], default=np.nan)
        valid_bbo = bid.gt(0) & ask.gt(0)
        missing_bbo = int((~valid_bbo).sum())
        crossed = int((valid_bbo & (bid > ask)).sum())
        locked = int((valid_bbo & (bid == ask)).sum())
        if missing_bbo:
            issues.append({"severity": "high", "code": "missing_best_bid_ask", "count": missing_bbo})
        if crossed:
            issues.append({"severity": "critical", "code": "crossed_book", "count": crossed})
        if locked:
            issues.append({"severity": "medium", "code": "locked_book", "count": locked})
        spread = (ask - bid).where(valid_bbo)
        spread_stats = spread.dropna()
        if len(spread_stats):
            report["spread"] = {
                "median": float(spread_stats.median()),
                "p95": float(spread_stats.quantile(0.95)),
                "max": float(spread_stats.max()),
            }
            if tick_size and tick_size > 0:
                wide = int((spread_stats / float(tick_size) > 20.0).sum())
                if wide:
                    issues.append({"severity": "medium", "code": "very_wide_spread_gt_20_ticks", "count": wide})

    report["passed"] = not any(issue["severity"] in {"critical", "high"} for issue in issues)
    if strict and not report["passed"]:
        raise ValueError(f"{context}: market-data validation failed: {issues[:5]}")
    return report


def assert_label_timestamps(df: pd.DataFrame, *, context: str) -> dict:
    if "ts_event" not in df.columns or "label_end_ts" not in df.columns:
        missing = [c for c in ("ts_event", "label_end_ts") if c not in df.columns]
        raise ValueError(f"{context}: missing required timestamp columns: {missing}")
    ts_event = _ts(df["ts_event"], context=f"{context}.ts_event")
    label_end = _ts(df["label_end_ts"], context=f"{context}.label_end_ts")
    bad = (label_end < ts_event).to_numpy(dtype=bool)
    report = {
        "generated_at": _utc_now(),
        "context": context,
        "rows": int(len(df)),
        "min_ts_event": str(ts_event.min()) if len(ts_event) else None,
        "max_ts_event": str(ts_event.max()) if len(ts_event) else None,
        "max_label_end_ts": str(label_end.max()) if len(label_end) else None,
        "invalid_label_horizons": int(np.sum(bad)),
        "passed": bool(not np.any(bad)),
    }
    if np.any(bad):
        sample = np.flatnonzero(bad)[:5].astype(int).tolist()
        raise ValueError(f"{context}: label_end_ts earlier than ts_event rows={int(np.sum(bad)):,} sample={sample}")
    return report


def summarize_label_distribution(
    df: pd.DataFrame,
    *,
    context: str,
    label_col: str = "bias_label",
    event_col: str = "train_event_flag",
    time_col: str = "ts_event",
) -> dict:
    labels = pd.to_numeric(df.get(label_col, 2), errors="coerce").fillna(2).astype(np.int8)
    counts = {name: int((labels == value).sum()) for value, name in LABEL_NAMES.items()}
    directional = int(counts["LONG"] + counts["SHORT"])
    rows = int(len(df))
    warnings: list[str] = []
    if rows == 0:
        warnings.append("empty_label_frame")
    if directional == 0:
        warnings.append("no_directional_labels")
    if counts["LONG"] == 0 or counts["SHORT"] == 0:
        warnings.append("single_sided_or_collapsed_directional_labels")
    if rows and counts["NEUTRAL"] / max(rows, 1) > 0.98:
        warnings.append("neutral_share_above_98pct")
    if min(counts["LONG"], counts["SHORT"]) > 0:
        imbalance = max(counts["LONG"], counts["SHORT"]) / max(min(counts["LONG"], counts["SHORT"]), 1)
        if imbalance > 20.0:
            warnings.append("long_short_imbalance_gt_20x")
    else:
        imbalance = None

    report = {
        "generated_at": _utc_now(),
        "context": context,
        "rows": rows,
        "class_counts": counts,
        "class_share": {k: float(v / max(rows, 1)) for k, v in counts.items()},
        "directional_rows": directional,
        "directional_share": float(directional / max(rows, 1)),
        "long_short_imbalance": imbalance,
        "warnings": warnings,
    }
    if event_col in df.columns:
        ev = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(np.int8)
        report["event_rows"] = int((ev == 1).sum())
        report["event_share"] = float((ev == 1).mean()) if len(ev) else 0.0
    if time_col in df.columns and rows:
        ts = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None)
        if not ts.isna().all():
            month = ts.dt.to_period("M").astype(str)
            monthly: list[dict] = []
            for key, idx in month.groupby(month, sort=True).groups.items():
                m_labels = labels.loc[idx]
                monthly.append({
                    "month": str(key),
                    "rows": int(len(m_labels)),
                    "LONG": int((m_labels == 0).sum()),
                    "SHORT": int((m_labels == 1).sum()),
                    "NEUTRAL": int((m_labels == 2).sum()),
                })
            report["months"] = monthly
    return report


def assert_no_label_leakage(
    df: pd.DataFrame,
    *,
    cutoff_ts,
    context: str,
    allow_equal: bool = False,
) -> dict:
    if "label_end_ts" not in df.columns:
        raise ValueError(f"{context}: label_end_ts is required for leakage checks")
    cutoff = pd.Timestamp(cutoff_ts)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    label_end = _ts(df["label_end_ts"], context=f"{context}.label_end_ts")
    mask = label_end > cutoff if allow_equal else label_end >= cutoff
    n_bad = int(mask.sum())
    report = {
        "generated_at": _utc_now(),
        "context": context,
        "cutoff_ts": str(cutoff),
        "rows": int(len(df)),
        "max_label_end_ts": str(label_end.max()) if len(label_end) else None,
        "leaking_rows": n_bad,
        "passed": bool(n_bad == 0),
    }
    if n_bad:
        sample = np.flatnonzero(mask.to_numpy(dtype=bool))[:5].astype(int).tolist()
        raise ValueError(
            f"{context}: label leakage detected before split cutoff={cutoff}; "
            f"rows_with_label_end_at_or_after_cutoff={n_bad:,}; sample={sample}"
        )
    return report


def validate_time_splits(
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    t0: pd.Series,
    t1: pd.Series,
    *,
    context: str,
    strict: bool = True,
) -> dict:
    ts0 = _ts(pd.Series(t0), context=f"{context}.t0")
    ts1 = _ts(pd.Series(t1), context=f"{context}.t1")
    folds: list[dict] = []
    issues: list[dict] = []
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        fold = {"fold": int(fold_no), "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx))}
        if len(train_idx) == 0 or len(test_idx) == 0:
            issues.append({"fold": fold_no, "code": "empty_train_or_test"})
            folds.append(fold)
            continue
        train_ts_max = ts0.iloc[train_idx].max()
        train_label_max = ts1.iloc[train_idx].max()
        test_ts_min = ts0.iloc[test_idx].min()
        test_ts_max = ts0.iloc[test_idx].max()
        overlap_idx = int(len(np.intersect1d(train_idx, test_idx)))
        fold.update({
            "train_end_ts": str(train_ts_max),
            "train_max_label_end_ts": str(train_label_max),
            "test_start_ts": str(test_ts_min),
            "test_end_ts": str(test_ts_max),
            "index_overlap": overlap_idx,
        })
        if overlap_idx:
            issues.append({"fold": fold_no, "code": "index_overlap", "count": overlap_idx})
        if train_ts_max >= test_ts_min:
            issues.append({"fold": fold_no, "code": "non_chronological_train_test"})
        if train_label_max >= test_ts_min:
            issues.append({
                "fold": fold_no,
                "code": "label_horizon_crosses_test_start",
                "train_max_label_end_ts": str(train_label_max),
                "test_start_ts": str(test_ts_min),
            })
        folds.append(fold)

    report = {
        "generated_at": _utc_now(),
        "context": context,
        "folds": folds,
        "issues": issues,
        "passed": bool(not issues),
    }
    if strict and issues:
        raise ValueError(f"{context}: unsafe time-series splits: {issues[:5]}")
    return report


def validate_backtest_realism_config(config: dict, *, context: str, strict: bool = False) -> dict:
    warnings: list[str] = []
    errors: list[str] = []

    tick_size = float(config.get("tick_size", 0.0) or 0.0)
    tick_value = float(config.get("tick_value", 0.0) or 0.0)
    round_trip_cost = float(config.get("round_trip_cost_pips", 0.0) or 0.0)
    commission = float(config.get("commission_per_side", 0.0) or 0.0)
    min_spread = float(config.get("min_spread_ticks", 0.0) or 0.0)
    min_slippage = float(config.get("min_slippage_ticks", 0.0) or 0.0)
    latency_rows = int(config.get("latency_rows", 0) or 0)
    max_size = int(config.get("max_size", 0) or 0)

    if tick_size <= 0:
        errors.append("tick_size_must_be_positive")
    if tick_value <= 0:
        errors.append("tick_value_must_be_positive")
    if round_trip_cost < 0 or commission < 0 or min_spread < 0 or min_slippage < 0:
        errors.append("negative_cost_parameter")
    if round_trip_cost <= 0:
        warnings.append("round_trip_cost_pips_is_zero")
    if commission <= 0:
        warnings.append("commission_per_side_is_zero")
    if min_spread < 1.0:
        warnings.append("min_spread_ticks_below_one")
    if min_slippage < 0.5:
        warnings.append("min_slippage_ticks_below_half_tick")
    if latency_rows <= 0:
        warnings.append("latency_rows_zero_or_negative")
    if max_size <= 0:
        errors.append("max_size_must_be_positive")

    report = {
        "generated_at": _utc_now(),
        "context": context,
        "config": {k: config.get(k) for k in sorted(config)},
        "warnings": warnings,
        "errors": errors,
        "passed": bool(not errors and (not strict or not warnings)),
    }
    if errors or (strict and warnings):
        raise ValueError(f"{context}: unrealistic backtest configuration: errors={errors} warnings={warnings}")
    return report
