#!/usr/bin/env python3
"""
Audit raw MBO and MBP market-data files before feature generation.

Examples:
  python3 -m tools.diagnostics.audit_mbo_mbp_data --mbo rich_mbo.csv --mbp rich_mbp.csv
  python3 -m tools.diagnostics.audit_mbo_mbp_data --mbo /data/mbo.parquet --mbp /data/mbp.parquet --expected-root 6B
  python3 -m tools.diagnostics.audit_mbo_mbp_data --mbo raw/mbo --mbp raw/mbp --json-out audit_report.json --fail-on high
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FUTURES_CONTRACT_RE = re.compile(
    r"^(?P<root>.+?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,4})$",
    re.IGNORECASE,
)

TIMESTAMP_COLUMNS = ("ts_event", "ts_recv", "timestamp", "time")
SYMBOL_COLUMNS = ("symbol", "raw_symbol", "contract_symbol", "instrument_id")
STRING_COLUMNS = {"action", "side", "symbol", "raw_symbol", "contract_symbol"}
PRICE_COLUMN_RE = re.compile(r"^(bid|ask)_px_\d+$")
SIZE_COLUMN_RE = re.compile(r"^(bid|ask)_sz_\d+$")

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SEVERITY_LABELS = {v: k for k, v in SEVERITY_ORDER.items()}
UNKNOWN_SYMBOLS = {"", "NAN", "NONE", "NULL", "<NA>", "UNKNOWN"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    return value


def _add_issue(
    report: dict[str, Any],
    severity: str,
    code: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    report.setdefault("issues", []).append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "evidence": evidence or {},
        }
    )


def _max_severity(issues: list[dict[str, Any]]) -> str | None:
    if not issues:
        return None
    value = max(SEVERITY_ORDER.get(str(item.get("severity", "low")), 1) for item in issues)
    return SEVERITY_LABELS[value]


def _is_csv_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip", ".csv.xz", ".gz", ".zst"))


def _is_parquet_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".parquet", ".pq", ".snappy"))


def _expand_input(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    patterns = (
        "*.parquet",
        "*.pq",
        "*.snappy",
        "*.csv",
        "*.csv.gz",
        "*.csv.bz2",
        "*.csv.zip",
        "*.csv.xz",
        "*.gz",
        "*.zst",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(str(path / pattern)))
    files = sorted({p.resolve() for p in files})
    if not files:
        raise FileNotFoundError(f"No supported CSV/Parquet files found in: {path}")
    return files


def _read_one(path: Path) -> pd.DataFrame:
    if _is_parquet_path(path):
        return pd.read_parquet(path)
    if _is_csv_path(path):
        return pd.read_csv(path, low_memory=False, compression="infer")
    raise ValueError(f"Unsupported input file type: {path}")


def _read_market_input(path_text: str, *, dataset: str) -> tuple[pd.DataFrame, list[str]]:
    files = _expand_input(Path(path_text).expanduser().resolve())
    frames = []
    for file_path in files:
        frame = _read_one(file_path)
        frame["_source_file"] = str(file_path)
        frames.append(frame)
    if len(frames) == 1:
        df = frames[0]
    else:
        df = pd.concat(frames, ignore_index=True, sort=False)
    df = _coerce_market_schema(df)
    if "_source_file" in df.columns and len(files) == 1:
        df = df.drop(columns=["_source_file"])
    print(f"Loaded {dataset.upper()}: {len(df):,} rows from {len(files)} file(s)")
    return df, [str(p) for p in files]


def _coerce_market_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TIMESTAMP_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    for col in out.columns:
        if (
            col in {"rtype", "publisher_id", "instrument_id", "depth", "price", "size", "flags", "ts_in_delta", "sequence"}
            or PRICE_COLUMN_RE.match(str(col))
            or SIZE_COLUMN_RE.match(str(col))
            or str(col).startswith(("bid_ct_", "ask_ct_"))
        ):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")

    return out


def _timestamp_column(df: pd.DataFrame) -> str | None:
    for col in TIMESTAMP_COLUMNS:
        if col in df.columns:
            return col
    return None


def _symbol_column(df: pd.DataFrame) -> str | None:
    for col in SYMBOL_COLUMNS:
        if col in df.columns:
            return col
    return None


def _normalized_symbols(series: pd.Series) -> pd.Series:
    symbols = series.astype("string").str.strip().str.upper()
    return symbols.mask(symbols.isna() | symbols.isin(UNKNOWN_SYMBOLS), "UNKNOWN")


def _safe_ts_string(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _month_keys(ts: pd.Series) -> pd.Series:
    valid = pd.to_datetime(ts, utc=True, errors="coerce")
    return valid.dt.strftime("%Y-%m")


def _series_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    arr = values.to_numpy(dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p95": float(np.nanpercentile(arr, 95)),
        "p99": float(np.nanpercentile(arr, 99)),
        "max": float(np.nanmax(arr)),
    }


def _parse_contract_symbol(symbol: str) -> dict[str, str] | None:
    token = str(symbol or "").strip().upper()
    if not token or token in UNKNOWN_SYMBOLS:
        return None
    match = FUTURES_CONTRACT_RE.match(token)
    if not match:
        return None
    return {
        "symbol": token,
        "root": match.group("root").upper(),
        "month_code": match.group("month").upper(),
        "year_suffix": match.group("year"),
    }


def _basic_time_checks(
    df: pd.DataFrame,
    *,
    kind: str,
    report: dict[str, Any],
    ts_col: str | None,
) -> None:
    if ts_col is None:
        _add_issue(report, "critical", f"{kind}_missing_timestamp_column", "No timestamp column found.")
        report["time"] = {"timestamp_column": None}
        return

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    valid = ts.dropna()
    time_report = {
        "timestamp_column": ts_col,
        "invalid_timestamps": int(ts.isna().sum()),
        "valid_timestamps": int(valid.shape[0]),
        "first_timestamp": _safe_ts_string(valid.min() if not valid.empty else None),
        "last_timestamp": _safe_ts_string(valid.max() if not valid.empty else None),
        "monotonic_in_input_order": bool(valid.is_monotonic_increasing) if not valid.empty else False,
        "backward_steps_in_input_order": int((valid.diff().dropna() < pd.Timedelta(0)).sum()) if len(valid) > 1 else 0,
    }
    report["time"] = time_report

    if valid.empty:
        _add_issue(report, "critical", f"{kind}_no_valid_timestamps", f"{kind.upper()} has no valid timestamps.")
    if time_report["invalid_timestamps"] > 0:
        _add_issue(
            report,
            "high",
            f"{kind}_invalid_timestamps",
            f"{kind.upper()} has invalid timestamps.",
            {"rows": time_report["invalid_timestamps"]},
        )
    if not time_report["monotonic_in_input_order"] and valid.shape[0] > 1:
        _add_issue(
            report,
            "medium",
            f"{kind}_non_monotonic_timestamps",
            f"{kind.upper()} timestamps are not monotonic in input order.",
            {"backward_steps": time_report["backward_steps_in_input_order"]},
        )


def _rows_per_month(df: pd.DataFrame, *, ts_col: str | None) -> dict[str, int]:
    if ts_col is None:
        return {}
    months = _month_keys(df[ts_col])
    counts = months.dropna().value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _monthly_active_columns(
    df: pd.DataFrame,
    *,
    kind: str,
    report: dict[str, Any],
    ts_col: str | None,
) -> None:
    if ts_col is None or len(df) == 0:
        report["monthly_columns"] = {
            "consistent_active_columns": False,
            "months": {},
            "differences": {},
        }
        return

    months = _month_keys(df[ts_col])
    active_by_month: dict[str, set[str]] = {}
    for month in sorted(m for m in months.dropna().unique().tolist()):
        subset = df.loc[months == month]
        active: set[str] = set()
        for col in df.columns:
            if col == "_source_file":
                continue
            col_data = subset[col]
            if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                non_blank = col_data.astype("string").str.strip()
                if bool((non_blank.notna() & ~non_blank.isin(UNKNOWN_SYMBOLS)).any()):
                    active.add(str(col))
            elif bool(col_data.notna().any()):
                active.add(str(col))
        active_by_month[str(month)] = active

    union = set().union(*active_by_month.values()) if active_by_month else set()
    intersection = set.intersection(*active_by_month.values()) if active_by_month else set()
    differences: dict[str, dict[str, list[str]]] = {}
    for month, active in active_by_month.items():
        missing = sorted(union - active)
        extra_vs_common = sorted(active - intersection)
        if missing or extra_vs_common:
            differences[month] = {
                "missing_active_columns_vs_union": missing,
                "extra_columns_vs_common_intersection": extra_vs_common,
            }

    consistent = len({tuple(sorted(cols)) for cols in active_by_month.values()}) <= 1
    report["monthly_columns"] = {
        "consistent_active_columns": bool(consistent),
        "months": {month: {"active_column_count": len(cols)} for month, cols in active_by_month.items()},
        "common_active_columns_count": len(intersection),
        "union_active_columns_count": len(union),
        "differences": differences,
    }
    if not consistent:
        _add_issue(
            report,
            "medium",
            f"{kind}_monthly_columns_inconsistent",
            f"{kind.upper()} does not have the same active columns in every month.",
            {"months_with_differences": len(differences)},
        )


def _missing_value_checks(df: pd.DataFrame, *, kind: str, report: dict[str, Any]) -> None:
    missing = df.isna().sum()
    blank_counts: dict[str, int] = {}
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            cleaned = df[col].astype("string").str.strip()
            blank_counts[col] = int((cleaned.isna() | cleaned.isin(UNKNOWN_SYMBOLS)).sum())

    missing_nonzero = {str(k): int(v) for k, v in missing.items() if int(v) > 0 and str(k) != "_source_file"}
    blanks_nonzero = {str(k): int(v) for k, v in blank_counts.items() if int(v) > 0 and str(k) != "_source_file"}
    report["missing_values"] = {
        "columns_with_nulls": missing_nonzero,
        "string_columns_with_blank_or_unknown": blanks_nonzero,
    }

    critical_cols = ["ts_event", "symbol"]
    if kind == "mbo":
        critical_cols.extend(["price", "size", "action", "side"])
    if kind == "mbp":
        critical_cols.extend(["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"])

    critical_missing = {
        col: int(missing.get(col, 0))
        for col in critical_cols
        if col in df.columns and int(missing.get(col, 0)) > 0
    }
    if critical_missing:
        severity = "medium"
        if "ts_event" in critical_missing or (kind == "mbp" and {"bid_px_00", "ask_px_00"} & set(critical_missing)):
            severity = "high"
        _add_issue(
            report,
            severity,
            f"{kind}_critical_missing_values",
            f"{kind.upper()} has missing values in critical columns.",
            critical_missing,
        )


def _duplicate_checks(
    df: pd.DataFrame,
    *,
    kind: str,
    report: dict[str, Any],
    ts_col: str | None,
    symbol_col: str | None,
) -> None:
    full_dup_cols = [c for c in df.columns if str(c) != "_source_file"]
    dup_report: dict[str, Any] = {
        "full_duplicate_rows": int(df.duplicated(subset=full_dup_cols).sum()) if full_dup_cols else 0,
        "duplicate_timestamp_extra_rows": 0,
        "rows_in_duplicate_timestamp_groups": 0,
        "duplicate_timestamp_symbol_extra_rows": 0,
        "duplicate_key_extra_rows": 0,
        "key_columns": [],
    }

    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        valid = ts.notna()
        dup_report["duplicate_timestamp_extra_rows"] = int(ts[valid].duplicated().sum())
        dup_report["rows_in_duplicate_timestamp_groups"] = int(ts[valid].duplicated(keep=False).sum())
        if symbol_col is not None:
            key = pd.DataFrame({"ts": ts, "symbol": _normalized_symbols(df[symbol_col])})
            key = key[key["ts"].notna()]
            dup_report["duplicate_timestamp_symbol_extra_rows"] = int(key.duplicated().sum())

    key_cols_by_kind = {
        "mbo": ("ts_event", "action", "side", "price", "size", "order_id"),
        "mbp": ("ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"),
    }
    key_cols = [c for c in key_cols_by_kind.get(kind, ()) if c in df.columns]
    dup_report["key_columns"] = key_cols
    if key_cols:
        dup_report["duplicate_key_extra_rows"] = int(df.duplicated(subset=key_cols).sum())

    report["duplicates"] = dup_report

    if dup_report["full_duplicate_rows"] > 0:
        _add_issue(
            report,
            "high",
            f"{kind}_full_duplicate_rows",
            f"{kind.upper()} has exact duplicate rows.",
            {"rows": dup_report["full_duplicate_rows"]},
        )
    if dup_report["duplicate_key_extra_rows"] > 0:
        _add_issue(
            report,
            "high",
            f"{kind}_duplicate_key_rows",
            f"{kind.upper()} has duplicate rows on the market-data key.",
            {"rows": dup_report["duplicate_key_extra_rows"], "key_columns": key_cols},
        )
    if dup_report["duplicate_timestamp_extra_rows"] > 0:
        severity = "low" if kind == "mbo" else "medium"
        _add_issue(
            report,
            severity,
            f"{kind}_duplicate_timestamps",
            f"{kind.upper()} has duplicate timestamps. This can be normal for tick feeds but must be handled before joins/resampling.",
            {"extra_rows": dup_report["duplicate_timestamp_extra_rows"]},
        )


def _price_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if str(c) == "price" or PRICE_COLUMN_RE.match(str(c))]


def _price_jump_examples(
    df: pd.DataFrame,
    *,
    price_series: pd.Series,
    ts_col: str | None,
    symbol_col: str | None,
    max_step_pct: float,
    top_n: int,
) -> tuple[int, list[dict[str, Any]]]:
    if ts_col is None or max_step_pct <= 0:
        return 0, []
    work = pd.DataFrame(
        {
            "ts": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
            "price": pd.to_numeric(price_series, errors="coerce"),
        }
    )
    work["symbol"] = _normalized_symbols(df[symbol_col]) if symbol_col is not None else "ALL"
    work = work.dropna(subset=["ts", "price"])
    work = work[work["price"] > 0].sort_values(["symbol", "ts"])
    if work.empty:
        return 0, []

    grouped = work.groupby("symbol", sort=False)
    work["prev_ts"] = grouped["ts"].shift(1)
    work["prev_price"] = grouped["price"].shift(1)
    work["abs_pct_change"] = (work["price"] - work["prev_price"]).abs() / work["prev_price"].abs()
    flagged = work[work["abs_pct_change"] > max_step_pct].copy()
    if flagged.empty:
        return 0, []

    examples = []
    for row in flagged.sort_values("abs_pct_change", ascending=False).head(top_n).itertuples(index=False):
        examples.append(
            {
                "symbol": str(row.symbol),
                "prev_ts": _safe_ts_string(row.prev_ts),
                "ts": _safe_ts_string(row.ts),
                "prev_price": float(row.prev_price),
                "price": float(row.price),
                "abs_pct_change": float(row.abs_pct_change),
            }
        )
    return int(flagged.shape[0]), examples


def _price_checks(
    df: pd.DataFrame,
    *,
    kind: str,
    report: dict[str, Any],
    ts_col: str | None,
    symbol_col: str | None,
    max_price_ratio: float,
    max_step_pct: float,
    top_n: int,
) -> None:
    price_cols = _price_columns(df)
    checks: dict[str, Any] = {}
    if not price_cols:
        _add_issue(report, "high", f"{kind}_missing_price_columns", f"{kind.upper()} has no price columns to audit.")
        report["price_checks"] = checks
        return

    for col in price_cols:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = values.dropna()
        non_positive = int((valid <= 0).sum())
        median = float(valid[valid > 0].median()) if bool((valid > 0).any()) else None
        extreme_low = 0
        extreme_high = 0
        if median and max_price_ratio > 1:
            extreme_low = int((valid > 0).lt(median / max_price_ratio).sum())
            extreme_high = int(valid.gt(median * max_price_ratio).sum())
        checks[col] = {
            "missing": int(values.isna().sum()),
            "non_positive": non_positive,
            "median_positive_price": median,
            "extreme_low_vs_median": extreme_low,
            "extreme_high_vs_median": extreme_high,
            "stats": _series_stats(values),
        }
        if non_positive > 0:
            _add_issue(
                report,
                "high",
                f"{kind}_{col}_non_positive_price",
                f"{kind.upper()} column {col} has non-positive prices.",
                {"rows": non_positive},
            )
        if extreme_low or extreme_high:
            _add_issue(
                report,
                "medium",
                f"{kind}_{col}_extreme_price_ratio",
                f"{kind.upper()} column {col} has prices far from the column median.",
                {"extreme_low": extreme_low, "extreme_high": extreme_high, "max_price_ratio": max_price_ratio},
            )

    main_price: pd.Series | None = None
    if "price" in df.columns:
        main_price = pd.to_numeric(df["price"], errors="coerce")
    elif {"bid_px_00", "ask_px_00"}.issubset(df.columns):
        bid = pd.to_numeric(df["bid_px_00"], errors="coerce")
        ask = pd.to_numeric(df["ask_px_00"], errors="coerce")
        main_price = (bid + ask) / 2.0

    if main_price is not None:
        jump_count, examples = _price_jump_examples(
            df,
            price_series=main_price,
            ts_col=ts_col,
            symbol_col=symbol_col,
            max_step_pct=max_step_pct,
            top_n=top_n,
        )
        checks["large_adjacent_price_jumps"] = {
            "threshold_pct": max_step_pct,
            "rows": jump_count,
            "examples": examples,
        }
        if jump_count > 0:
            _add_issue(
                report,
                "medium",
                f"{kind}_large_adjacent_price_jumps",
                f"{kind.upper()} has adjacent price jumps above the configured threshold.",
                {"rows": jump_count, "threshold_pct": max_step_pct},
            )

    report["price_checks"] = checks


def _infer_tick_size_from_spread(spread: pd.Series) -> float | None:
    values = pd.to_numeric(spread, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0]
    if values.empty:
        return None
    arr = np.asarray(values, dtype=float)
    threshold = max(float(np.nanmedian(arr)) * 1e-8, 1e-12)
    arr = arr[arr > threshold]
    if arr.size == 0:
        return None
    return float(np.nanpercentile(arr, 1))


def _spread_checks(
    df: pd.DataFrame,
    *,
    kind: str,
    report: dict[str, Any],
    tick_size: float | None,
    max_spread_ticks: float,
) -> None:
    if not {"bid_px_00", "ask_px_00"}.issubset(df.columns):
        if kind == "mbp":
            _add_issue(report, "high", "mbp_missing_bbo_columns", "MBP is missing bid_px_00/ask_px_00.")
        report["spread_checks"] = {"available": False}
        return

    bid = pd.to_numeric(df["bid_px_00"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ask = pd.to_numeric(df["ask_px_00"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    spread = ask - bid
    missing_bbo = int((bid.isna() | ask.isna()).sum())
    positive_price_mask = bid.gt(0) & ask.gt(0)
    crossed = int((positive_price_mask & ask.lt(bid)).sum())
    locked = int((positive_price_mask & ask.eq(bid)).sum())
    non_positive_bbo = int((bid.le(0) | ask.le(0)).fillna(False).sum())
    inferred_tick = _infer_tick_size_from_spread(spread)
    used_tick = float(tick_size) if tick_size and tick_size > 0 else inferred_tick
    wide_count = 0
    if used_tick and max_spread_ticks > 0:
        wide_count = int((spread > used_tick * max_spread_ticks).sum())

    size_issues: dict[str, int] = {}
    for col in ("bid_sz_00", "ask_sz_00"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            size_issues[f"{col}_negative"] = int((values.dropna() < 0).sum())

    spread_report = {
        "available": True,
        "missing_bbo_rows": missing_bbo,
        "crossed_rows_ask_below_bid": crossed,
        "locked_rows_ask_equals_bid": locked,
        "non_positive_bbo_price_rows": non_positive_bbo,
        "inferred_tick_size_from_spread": inferred_tick,
        "tick_size_used": used_tick,
        "max_spread_ticks": max_spread_ticks,
        "wide_spread_rows": wide_count,
        "spread_stats": _series_stats(spread),
        "size_issues": size_issues,
    }
    report["spread_checks"] = spread_report

    if missing_bbo > 0:
        _add_issue(report, "medium", f"{kind}_missing_bbo", f"{kind.upper()} has rows with missing BBO.", {"rows": missing_bbo})
    if non_positive_bbo > 0:
        _add_issue(
            report,
            "high",
            f"{kind}_non_positive_bbo",
            f"{kind.upper()} has non-positive top-of-book prices.",
            {"rows": non_positive_bbo},
        )
    if crossed > 0:
        _add_issue(
            report,
            "high",
            f"{kind}_crossed_bbo",
            f"{kind.upper()} has crossed BBO rows where ask_px_00 < bid_px_00.",
            {"rows": crossed},
        )
    if locked > 0:
        _add_issue(
            report,
            "medium",
            f"{kind}_locked_bbo",
            f"{kind.upper()} has locked BBO rows where ask_px_00 == bid_px_00.",
            {"rows": locked},
        )
    if wide_count > 0:
        _add_issue(
            report,
            "medium",
            f"{kind}_wide_spread",
            f"{kind.upper()} has spreads wider than the configured tick threshold.",
            {"rows": wide_count, "tick_size_used": used_tick, "max_spread_ticks": max_spread_ticks},
        )
    negative_sizes = {k: v for k, v in size_issues.items() if v > 0}
    if negative_sizes:
        _add_issue(
            report,
            "high",
            f"{kind}_negative_top_sizes",
            f"{kind.upper()} has negative top-of-book sizes.",
            negative_sizes,
        )


def _gap_checks(
    df: pd.DataFrame,
    *,
    kind: str,
    report: dict[str, Any],
    ts_col: str | None,
    symbol_col: str | None,
    gap_threshold: pd.Timedelta,
    top_n: int,
) -> None:
    gap_report = {
        "threshold_seconds": float(gap_threshold.total_seconds()),
        "large_gap_count": 0,
        "max_gap_seconds": None,
        "examples": [],
    }
    if ts_col is None:
        report["time_gaps"] = gap_report
        return

    work = pd.DataFrame({"ts": pd.to_datetime(df[ts_col], utc=True, errors="coerce")})
    work["symbol"] = _normalized_symbols(df[symbol_col]) if symbol_col is not None else "ALL"
    work = work.dropna(subset=["ts"]).sort_values(["symbol", "ts"])
    if work.shape[0] <= 1:
        report["time_gaps"] = gap_report
        return

    grouped = work.groupby("symbol", sort=False)
    work["prev_ts"] = grouped["ts"].shift(1)
    work["gap"] = work["ts"] - work["prev_ts"]
    flagged = work[work["gap"] > gap_threshold].copy()
    if not flagged.empty:
        gap_report["large_gap_count"] = int(flagged.shape[0])
        gap_report["max_gap_seconds"] = float(flagged["gap"].max().total_seconds())
        examples = []
        for row in flagged.sort_values("gap", ascending=False).head(top_n).itertuples(index=False):
            examples.append(
                {
                    "symbol": str(row.symbol),
                    "prev_ts": _safe_ts_string(row.prev_ts),
                    "ts": _safe_ts_string(row.ts),
                    "gap_seconds": float(row.gap.total_seconds()),
                }
            )
        gap_report["examples"] = examples
        _add_issue(
            report,
            "medium",
            f"{kind}_large_time_gaps",
            f"{kind.upper()} has time gaps larger than the configured threshold.",
            {"rows": gap_report["large_gap_count"], "threshold_seconds": gap_report["threshold_seconds"]},
        )

    report["time_gaps"] = gap_report


def _contract_checks(
    df: pd.DataFrame,
    *,
    kind: str,
    report: dict[str, Any],
    ts_col: str | None,
    symbol_col: str | None,
    expected_root: str | None,
    expected_symbol: str | None,
    top_n: int,
) -> None:
    if symbol_col is None:
        _add_issue(report, "medium", f"{kind}_missing_symbol_column", f"{kind.upper()} has no symbol/contract column.")
        report["contracts"] = {"symbol_column": None}
        return

    symbols = _normalized_symbols(df[symbol_col])
    counts = symbols.value_counts(dropna=False)
    counts_map = {str(k): int(v) for k, v in counts.items()}

    parsed = symbols.map(_parse_contract_symbol)
    parsed_valid = parsed.dropna()
    roots = parsed_valid.map(lambda item: item["root"]) if not parsed_valid.empty else pd.Series(dtype="object")
    root_counts = roots.value_counts().to_dict() if not roots.empty else {}
    root_counts_map = {str(k): int(v) for k, v in root_counts.items()}
    invalid_contract_symbols = sorted({str(s) for s, p in zip(symbols.tolist(), parsed.tolist()) if s != "UNKNOWN" and p is None})

    month_symbol_map: dict[str, list[str]] = {}
    multi_symbol_months: dict[str, list[str]] = {}
    if ts_col is not None:
        months = _month_keys(df[ts_col])
        work = pd.DataFrame({"month": months, "symbol": symbols}).dropna(subset=["month"])
        for month, group in work.groupby("month", sort=True):
            month_symbols = sorted(set(str(v) for v in group["symbol"].tolist() if str(v) != "UNKNOWN"))
            month_symbol_map[str(month)] = month_symbols
            if len(month_symbols) > 1:
                multi_symbol_months[str(month)] = month_symbols

    symbol_ranges: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    if ts_col is not None:
        work = pd.DataFrame(
            {
                "ts": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
                "symbol": symbols,
            }
        ).dropna(subset=["ts"])
        for symbol, group in work.groupby("symbol", sort=True):
            if str(symbol) == "UNKNOWN":
                continue
            symbol_ranges.append(
                {
                    "symbol": str(symbol),
                    "rows": int(group.shape[0]),
                    "first_timestamp": _safe_ts_string(group["ts"].min()),
                    "last_timestamp": _safe_ts_string(group["ts"].max()),
                }
            )
        range_by_symbol = {
            row["symbol"]: (
                pd.Timestamp(row["first_timestamp"]) if row["first_timestamp"] else None,
                pd.Timestamp(row["last_timestamp"]) if row["last_timestamp"] else None,
            )
            for row in symbol_ranges
        }
        for left, right in combinations(sorted(range_by_symbol), 2):
            l0, l1 = range_by_symbol[left]
            r0, r1 = range_by_symbol[right]
            if l0 is None or l1 is None or r0 is None or r1 is None:
                continue
            overlap_start = max(l0, r0)
            overlap_end = min(l1, r1)
            if overlap_start <= overlap_end:
                overlaps.append(
                    {
                        "left_symbol": left,
                        "right_symbol": right,
                        "overlap_start": _safe_ts_string(overlap_start),
                        "overlap_end": _safe_ts_string(overlap_end),
                    }
                )

    contract_report = {
        "symbol_column": symbol_col,
        "symbol_count": int(len(counts_map)),
        "symbol_counts": counts_map,
        "parsed_contract_symbol_count": int(parsed_valid.shape[0]),
        "invalid_contract_symbols": invalid_contract_symbols[:top_n],
        "root_counts": root_counts_map,
        "symbols_per_month": month_symbol_map,
        "months_with_multiple_symbols": multi_symbol_months,
        "symbol_time_ranges": symbol_ranges,
        "overlapping_symbol_ranges": overlaps[:top_n],
    }
    report["contracts"] = contract_report

    expected_root_clean = str(expected_root or "").strip().upper()
    expected_symbol_clean = str(expected_symbol or "").strip().upper()

    if expected_symbol_clean:
        unexpected = sorted(s for s in counts_map if s != expected_symbol_clean and s != "UNKNOWN")
        if unexpected:
            _add_issue(
                report,
                "high",
                f"{kind}_unexpected_symbol",
                f"{kind.upper()} contains symbols other than the expected symbol.",
                {"expected_symbol": expected_symbol_clean, "unexpected_symbols": unexpected[:top_n]},
            )
    if expected_root_clean:
        unexpected_roots = sorted(r for r in root_counts_map if r != expected_root_clean)
        if unexpected_roots:
            _add_issue(
                report,
                "high",
                f"{kind}_unexpected_root",
                f"{kind.upper()} contains futures roots other than the expected root.",
                {"expected_root": expected_root_clean, "unexpected_roots": unexpected_roots[:top_n]},
            )
    if len(root_counts_map) > 1:
        _add_issue(
            report,
            "high",
            f"{kind}_mixed_contract_roots",
            f"{kind.upper()} contains more than one futures root.",
            {"root_counts": root_counts_map},
        )
    if multi_symbol_months:
        _add_issue(
            report,
            "medium",
            f"{kind}_multiple_symbols_same_month",
            f"{kind.upper()} has months containing multiple contract symbols.",
            {"months": dict(list(multi_symbol_months.items())[:top_n])},
        )
    if overlaps:
        _add_issue(
            report,
            "high",
            f"{kind}_overlapping_contract_ranges",
            f"{kind.upper()} has overlapping time ranges across contract symbols.",
            {"overlaps": overlaps[:top_n]},
        )


def _audit_dataset(
    df: pd.DataFrame,
    *,
    kind: str,
    files: list[str],
    gap_threshold: pd.Timedelta,
    tick_size: float | None,
    max_spread_ticks: float,
    max_price_ratio: float,
    max_step_pct: float,
    expected_root: str | None,
    expected_symbol: str | None,
    top_n: int,
) -> dict[str, Any]:
    ts_col = _timestamp_column(df)
    symbol_col = _symbol_column(df)
    report: dict[str, Any] = {
        "kind": kind,
        "files": files,
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns if str(c) != "_source_file"],
        "column_count": int(len([c for c in df.columns if str(c) != "_source_file"])),
        "issues": [],
    }

    _basic_time_checks(df, kind=kind, report=report, ts_col=ts_col)
    report["rows_per_month"] = _rows_per_month(df, ts_col=ts_col)
    _monthly_active_columns(df, kind=kind, report=report, ts_col=ts_col)
    _missing_value_checks(df, kind=kind, report=report)
    _duplicate_checks(df, kind=kind, report=report, ts_col=ts_col, symbol_col=symbol_col)
    _price_checks(
        df,
        kind=kind,
        report=report,
        ts_col=ts_col,
        symbol_col=symbol_col,
        max_price_ratio=max_price_ratio,
        max_step_pct=max_step_pct,
        top_n=top_n,
    )
    _spread_checks(df, kind=kind, report=report, tick_size=tick_size, max_spread_ticks=max_spread_ticks)
    _gap_checks(
        df,
        kind=kind,
        report=report,
        ts_col=ts_col,
        symbol_col=symbol_col,
        gap_threshold=gap_threshold,
        top_n=top_n,
    )
    _contract_checks(
        df,
        kind=kind,
        report=report,
        ts_col=ts_col,
        symbol_col=symbol_col,
        expected_root=expected_root,
        expected_symbol=expected_symbol,
        top_n=top_n,
    )
    report["max_issue_severity"] = _max_severity(report["issues"])
    return report


def _compare_mbo_mbp(mbo_report: dict[str, Any] | None, mbp_report: dict[str, Any] | None, *, top_n: int) -> dict[str, Any]:
    report: dict[str, Any] = {"issues": []}
    if not mbo_report or not mbp_report:
        report["max_issue_severity"] = None
        return report

    mbo_contracts = mbo_report.get("contracts", {})
    mbp_contracts = mbp_report.get("contracts", {})
    mbo_symbols = {s for s in (mbo_contracts.get("symbol_counts") or {}) if s != "UNKNOWN"}
    mbp_symbols = {s for s in (mbp_contracts.get("symbol_counts") or {}) if s != "UNKNOWN"}
    mbo_roots = set((mbo_contracts.get("root_counts") or {}).keys())
    mbp_roots = set((mbp_contracts.get("root_counts") or {}).keys())

    report["symbol_set_comparison"] = {
        "mbo_only_symbols": sorted(mbo_symbols - mbp_symbols),
        "mbp_only_symbols": sorted(mbp_symbols - mbo_symbols),
        "common_symbols": sorted(mbo_symbols & mbp_symbols),
    }
    report["root_set_comparison"] = {
        "mbo_only_roots": sorted(mbo_roots - mbp_roots),
        "mbp_only_roots": sorted(mbp_roots - mbo_roots),
        "common_roots": sorted(mbo_roots & mbp_roots),
    }

    if mbo_roots and mbp_roots and mbo_roots != mbp_roots:
        _add_issue(
            report,
            "high",
            "mbo_mbp_root_mismatch",
            "MBO and MBP contain different futures roots.",
            report["root_set_comparison"],
        )
    if mbo_symbols and mbp_symbols and mbo_symbols != mbp_symbols:
        _add_issue(
            report,
            "medium",
            "mbo_mbp_symbol_mismatch",
            "MBO and MBP contain different contract symbols.",
            report["symbol_set_comparison"],
        )

    mbo_months = mbo_contracts.get("symbols_per_month") or {}
    mbp_months = mbp_contracts.get("symbols_per_month") or {}
    monthly_mismatches: dict[str, dict[str, list[str]]] = {}
    for month in sorted(set(mbo_months) | set(mbp_months)):
        left = set(mbo_months.get(month) or [])
        right = set(mbp_months.get(month) or [])
        if left != right:
            monthly_mismatches[month] = {"mbo_symbols": sorted(left), "mbp_symbols": sorted(right)}
    report["monthly_symbol_mismatches"] = dict(list(monthly_mismatches.items())[:top_n])
    if monthly_mismatches:
        _add_issue(
            report,
            "medium",
            "mbo_mbp_monthly_symbol_mismatch",
            "MBO and MBP monthly contract-symbol coverage differs.",
            {"months": dict(list(monthly_mismatches.items())[:top_n])},
        )

    mbo_start = pd.Timestamp(mbo_report.get("time", {}).get("first_timestamp")) if mbo_report.get("time", {}).get("first_timestamp") else None
    mbo_end = pd.Timestamp(mbo_report.get("time", {}).get("last_timestamp")) if mbo_report.get("time", {}).get("last_timestamp") else None
    mbp_start = pd.Timestamp(mbp_report.get("time", {}).get("first_timestamp")) if mbp_report.get("time", {}).get("first_timestamp") else None
    mbp_end = pd.Timestamp(mbp_report.get("time", {}).get("last_timestamp")) if mbp_report.get("time", {}).get("last_timestamp") else None
    time_overlap = None
    if mbo_start is not None and mbo_end is not None and mbp_start is not None and mbp_end is not None:
        overlap_start = max(mbo_start, mbp_start)
        overlap_end = min(mbo_end, mbp_end)
        if overlap_start <= overlap_end:
            time_overlap = {
                "overlap_start": _safe_ts_string(overlap_start),
                "overlap_end": _safe_ts_string(overlap_end),
                "overlap_seconds": float((overlap_end - overlap_start).total_seconds()),
            }
        else:
            _add_issue(
                report,
                "high",
                "mbo_mbp_no_time_overlap",
                "MBO and MBP time ranges do not overlap.",
                {
                    "mbo_first": _safe_ts_string(mbo_start),
                    "mbo_last": _safe_ts_string(mbo_end),
                    "mbp_first": _safe_ts_string(mbp_start),
                    "mbp_last": _safe_ts_string(mbp_end),
                },
            )
    report["time_overlap"] = time_overlap
    report["max_issue_severity"] = _max_severity(report["issues"])
    return report


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _print_mapping(title: str, mapping: dict[str, Any], *, indent: str = "  ", limit: int | None = None) -> None:
    print(title)
    if not mapping:
        print(f"{indent}(none)")
        return
    items = list(mapping.items())
    if limit is not None:
        items = items[:limit]
    for key, value in items:
        print(f"{indent}{key}: {_format_number(value)}")
    if limit is not None and len(mapping) > limit:
        print(f"{indent}... {len(mapping) - limit} more")


def _print_dataset_report(report: dict[str, Any]) -> None:
    kind = str(report.get("kind", "")).upper()
    print("\n" + "=" * 78)
    print(f"{kind} DATA QUALITY")
    print("=" * 78)
    print(f"Rows: {_format_number(report.get('rows'))} | Columns: {_format_number(report.get('column_count'))}")
    time_report = report.get("time", {})
    print(
        "Timestamp: "
        f"{time_report.get('timestamp_column')} | "
        f"{time_report.get('first_timestamp')} -> {time_report.get('last_timestamp')} | "
        f"monotonic={time_report.get('monotonic_in_input_order')}"
    )

    _print_mapping("Rows per month:", report.get("rows_per_month") or {})

    contracts = report.get("contracts") or {}
    print(
        "Symbols: "
        f"{contracts.get('symbol_count', 0)} via {contracts.get('symbol_column')} | "
        f"roots={contracts.get('root_counts') or {}}"
    )
    top_symbols = dict(list((contracts.get("symbol_counts") or {}).items())[:10])
    _print_mapping("Top symbols:", top_symbols)

    monthly_columns = report.get("monthly_columns") or {}
    print(f"Monthly active columns consistent: {monthly_columns.get('consistent_active_columns')}")
    differences = monthly_columns.get("differences") or {}
    if differences:
        print("Monthly column differences (first 5):")
        for month, payload in list(differences.items())[:5]:
            missing = payload.get("missing_active_columns_vs_union") or []
            extra = payload.get("extra_columns_vs_common_intersection") or []
            print(f"  {month}: missing={missing[:8]} extra={extra[:8]}")

    duplicates = report.get("duplicates") or {}
    print(
        "Duplicate timestamps: "
        f"timestamp_extra={_format_number(duplicates.get('duplicate_timestamp_extra_rows'))}, "
        f"timestamp_symbol_extra={_format_number(duplicates.get('duplicate_timestamp_symbol_extra_rows'))}, "
        f"key_extra={_format_number(duplicates.get('duplicate_key_extra_rows'))}, "
        f"full_rows={_format_number(duplicates.get('full_duplicate_rows'))}"
    )

    missing_values = report.get("missing_values") or {}
    nulls = missing_values.get("columns_with_nulls") or {}
    print(f"Columns with missing values: {len(nulls)}")
    for col, count in list(nulls.items())[:15]:
        print(f"  {col}: {_format_number(count)}")
    if len(nulls) > 15:
        print(f"  ... {len(nulls) - 15} more")

    price_checks = report.get("price_checks") or {}
    price_summary = {}
    for col, payload in price_checks.items():
        if isinstance(payload, dict) and "non_positive" in payload:
            price_summary[col] = {
                "missing": payload.get("missing"),
                "non_positive": payload.get("non_positive"),
                "median": payload.get("median_positive_price"),
                "extreme_low": payload.get("extreme_low_vs_median"),
                "extreme_high": payload.get("extreme_high_vs_median"),
            }
    print("Price sanity:")
    for col, payload in list(price_summary.items())[:12]:
        print(
            f"  {col}: missing={_format_number(payload['missing'])}, "
            f"non_positive={_format_number(payload['non_positive'])}, "
            f"median={_format_number(payload['median'])}, "
            f"extreme_low={_format_number(payload['extreme_low'])}, "
            f"extreme_high={_format_number(payload['extreme_high'])}"
        )
    jumps = price_checks.get("large_adjacent_price_jumps") or {}
    if jumps:
        print(f"Large adjacent price jumps: {_format_number(jumps.get('rows'))} > {jumps.get('threshold_pct'):.2%}")

    spread = report.get("spread_checks") or {}
    if spread.get("available"):
        stats = spread.get("spread_stats") or {}
        print(
            "Spread sanity: "
            f"missing_bbo={_format_number(spread.get('missing_bbo_rows'))}, "
            f"crossed={_format_number(spread.get('crossed_rows_ask_below_bid'))}, "
            f"locked={_format_number(spread.get('locked_rows_ask_equals_bid'))}, "
            f"wide={_format_number(spread.get('wide_spread_rows'))}, "
            f"tick_used={_format_number(spread.get('tick_size_used'))}, "
            f"p99={_format_number(stats.get('p99'))}, max={_format_number(stats.get('max'))}"
        )
    else:
        print("Spread sanity: BBO columns not available")

    gaps = report.get("time_gaps") or {}
    print(
        "Large time gaps: "
        f"{_format_number(gaps.get('large_gap_count'))} "
        f"(threshold={_format_number(gaps.get('threshold_seconds'))} seconds, "
        f"max={_format_number(gaps.get('max_gap_seconds'))} seconds)"
    )
    for item in (gaps.get("examples") or [])[:5]:
        print(f"  {item['symbol']}: {item['prev_ts']} -> {item['ts']} ({item['gap_seconds']:.2f}s)")

    multi_months = contracts.get("months_with_multiple_symbols") or {}
    overlaps = contracts.get("overlapping_symbol_ranges") or []
    print(f"Mixed contract months: {len(multi_months)} | overlapping symbol ranges: {len(overlaps)}")
    for month, symbols in list(multi_months.items())[:5]:
        print(f"  {month}: {symbols}")

    issues = report.get("issues") or []
    print(f"Max issue severity: {report.get('max_issue_severity') or 'none'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  [{str(issue['severity']).upper()}] {issue['code']}: {issue['message']}")
    else:
        print("Issues: none")


def _print_cross_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("MBO/MBP CROSS-CHECK")
    print("=" * 78)
    if not report:
        print("No cross-check report.")
        return
    print(f"Symbols: {report.get('symbol_set_comparison')}")
    print(f"Roots: {report.get('root_set_comparison')}")
    print(f"Time overlap: {report.get('time_overlap')}")
    monthly = report.get("monthly_symbol_mismatches") or {}
    print(f"Monthly symbol mismatches: {len(monthly)}")
    for month, payload in list(monthly.items())[:5]:
        print(f"  {month}: MBO={payload.get('mbo_symbols')} MBP={payload.get('mbp_symbols')}")
    issues = report.get("issues") or []
    print(f"Max issue severity: {report.get('max_issue_severity') or 'none'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  [{str(issue['severity']).upper()}] {issue['code']}: {issue['message']}")
    else:
        print("Issues: none")


def _all_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for report in (payload.get("datasets") or {}).values():
        if report:
            issues.extend(report.get("issues") or [])
    cross = payload.get("cross_checks") or {}
    issues.extend(cross.get("issues") or [])
    return issues


def _should_fail(issues: list[dict[str, Any]], fail_on: str) -> bool:
    fail_on = str(fail_on or "none").lower()
    if fail_on == "none":
        return False
    threshold = SEVERITY_ORDER[fail_on]
    return any(SEVERITY_ORDER.get(str(issue.get("severity", "low")).lower(), 1) >= threshold for issue in issues)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MBO/MBP raw market data for rows/month, schema consistency, timestamps, missing values, prices, spreads, gaps, and contract mixing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mbo", required=True, help="MBO CSV/Parquet file or directory of supported files.")
    parser.add_argument("--mbp", default=None, help="Optional MBP CSV/Parquet file or directory of supported files.")
    parser.add_argument("--json-out", default=None, help="Optional JSON report output path.")
    parser.add_argument("--gap-threshold", default="1h", help="Large time-gap threshold, e.g. 30min, 1h, 1D.")
    parser.add_argument("--tick-size", type=float, default=None, help="Known tick size for spread checks. If omitted, inferred from positive top-of-book spread.")
    parser.add_argument("--max-spread-ticks", type=float, default=20.0, help="Flag BBO spread wider than tick_size * this value.")
    parser.add_argument("--max-price-ratio", type=float, default=5.0, help="Flag prices outside median / ratio and median * ratio.")
    parser.add_argument("--max-step-pct", type=float, default=0.05, help="Flag adjacent price jumps above this percentage per symbol.")
    parser.add_argument("--expected-root", default=None, help="Optional expected futures root, e.g. 6B.")
    parser.add_argument("--expected-symbol", default=None, help="Optional expected single contract symbol, e.g. 6BH5.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of examples to keep in the report.")
    parser.add_argument(
        "--fail-on",
        choices=["none", "low", "medium", "high", "critical"],
        default="critical",
        help="Exit with code 1 if an issue at this severity or higher is found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        gap_threshold = pd.Timedelta(args.gap_threshold)
    except Exception as exc:
        raise SystemExit(f"Invalid --gap-threshold value {args.gap_threshold!r}: {exc}") from exc

    mbo_df, mbo_files = _read_market_input(args.mbo, dataset="mbo")
    mbp_df = None
    mbp_files: list[str] = []
    if args.mbp:
        mbp_df, mbp_files = _read_market_input(args.mbp, dataset="mbp")

    mbo_report = _audit_dataset(
        mbo_df,
        kind="mbo",
        files=mbo_files,
        gap_threshold=gap_threshold,
        tick_size=args.tick_size,
        max_spread_ticks=args.max_spread_ticks,
        max_price_ratio=args.max_price_ratio,
        max_step_pct=args.max_step_pct,
        expected_root=args.expected_root,
        expected_symbol=args.expected_symbol,
        top_n=max(1, int(args.top_n)),
    )
    mbp_report = None
    if mbp_df is not None:
        mbp_report = _audit_dataset(
            mbp_df,
            kind="mbp",
            files=mbp_files,
            gap_threshold=gap_threshold,
            tick_size=args.tick_size,
            max_spread_ticks=args.max_spread_ticks,
            max_price_ratio=args.max_price_ratio,
            max_step_pct=args.max_step_pct,
            expected_root=args.expected_root,
            expected_symbol=args.expected_symbol,
            top_n=max(1, int(args.top_n)),
        )

    cross_report = _compare_mbo_mbp(mbo_report, mbp_report, top_n=max(1, int(args.top_n)))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gap_threshold": str(gap_threshold),
            "tick_size": args.tick_size,
            "max_spread_ticks": args.max_spread_ticks,
            "max_price_ratio": args.max_price_ratio,
            "max_step_pct": args.max_step_pct,
            "expected_root": args.expected_root,
            "expected_symbol": args.expected_symbol,
            "fail_on": args.fail_on,
        },
        "datasets": {"mbo": mbo_report, "mbp": mbp_report},
        "cross_checks": cross_report,
    }
    all_issues = _all_issues(payload)
    payload["overall_max_issue_severity"] = _max_severity(all_issues)

    _print_dataset_report(mbo_report)
    if mbp_report is not None:
        _print_dataset_report(mbp_report)
        _print_cross_report(cross_report)
    print("\n" + "=" * 78)
    print(f"OVERALL MAX ISSUE SEVERITY: {payload['overall_max_issue_severity'] or 'none'}")
    print("=" * 78)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
        print(f"JSON report written to: {out_path}")

    return 1 if _should_fail(all_issues, args.fail_on) else 0


if __name__ == "__main__":
    raise SystemExit(main())
