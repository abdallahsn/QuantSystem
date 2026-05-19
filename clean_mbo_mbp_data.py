#!/usr/bin/env python3
"""
Clean raw MBO/MBP market-data before QuantSystem V19 feature generation.

The script intentionally preserves raw inputs and writes a new cleaned Parquet
artifact that can be passed directly to:

  python prepare_training_data.py --mbo clean_market_data/mbo --mbp clean_market_data/mbp
  python prepare_day_trading.py  --mbo clean_market_data/mbo --mbp clean_market_data/mbp

It fixes deterministic data-integrity problems:
  - invalid timestamps and critical missing values
  - exact duplicates and duplicate market-data keys
  - non-monotonic input order via stable chronological sorting
  - invalid executable MBP BBO rows (missing/non-positive/crossed/locked/wide)
  - obvious median-ratio price outliers
  - isolated adjacent price spikes

It does not invent ticks or fill market closures. Large time gaps are reported,
not forward-filled, because filling tick/order-book data would create synthetic
market states and can leak unrealistic execution assumptions into backtests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from audit_mbo_mbp_data import (
        _audit_dataset,
        _coerce_market_schema,
        _compare_mbo_mbp,
        _expand_input,
        _infer_tick_size_from_spread,
        _jsonable,
        _max_severity,
        _parse_contract_symbol,
        _read_one,
    )
except Exception as exc:  # pragma: no cover - this repo ships audit_mbo_mbp_data.py
    raise SystemExit(f"Cannot import audit helpers from audit_mbo_mbp_data.py: {exc}") from exc


HELPER_COLUMNS = {"__source_file", "__source_row", "__ingest_order"}
TRADE_ACTIONS = {"T", "F", "TRADE", "EXECUTE", "E", "0"}
UNKNOWN_STRINGS = {"", "NAN", "NONE", "NULL", "<NA>", "UNKNOWN"}

MBO_KEY_COLUMNS = ("ts_event", "action", "side", "price", "size", "order_id")
MBP_KEY_COLUMNS = ("ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00")
SORT_TIE_COLUMNS = ("ts_event", "symbol", "ts_recv", "sequence", "__ingest_order")


@dataclass
class CleanConfig:
    tick_size: float | None = None
    max_spread_ticks: float = 20.0
    max_price_ratio: float = 5.0
    max_step_pct: float = 0.05
    drop_locked_bbo: bool = True
    drop_wide_spread: bool = True
    drop_isolated_price_jumps: bool = True
    drop_mbo_reset_actions: bool = True
    recompute_mbp_price_from_mid: bool = True
    preserve_raw_mbp_price: bool = True
    expected_root: str | None = None
    expected_symbol: str | None = None
    trim_to_overlap: bool = False
    gap_threshold: str = "1h"
    top_n: int = 10


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def _audit_command(mbo_dir: Path, mbp_dir: Path | None, cfg: CleanConfig) -> str:
    parts = ["python", "audit_mbo_mbp_data.py", "--mbo", str(mbo_dir)]
    if mbp_dir is not None:
        parts.extend(["--mbp", str(mbp_dir)])
    if cfg.tick_size is not None:
        parts.extend(["--tick-size", str(cfg.tick_size)])
    if cfg.expected_root:
        parts.extend(["--expected-root", str(cfg.expected_root)])
    if cfg.expected_symbol:
        parts.extend(["--expected-symbol", str(cfg.expected_symbol)])
    parts.extend(["--fail-on", "high"])
    return " ".join(parts)


def _prepare_training_command(mbo_dir: Path, mbp_dir: Path | None) -> str:
    parts = ["python", "prepare_training_data.py", "--mbo", str(mbo_dir)]
    if mbp_dir is not None:
        parts.extend(["--mbp", str(mbp_dir)])
    else:
        parts.extend(["--mbp", ""])
    parts.extend(["--output", "outputs_cleaned"])
    return " ".join(parts)


def _prepare_day_trading_command(mbo_dir: Path, mbp_dir: Path | None) -> str:
    parts = ["python", "prepare_day_trading.py", "--mbo", str(mbo_dir)]
    if mbp_dir is not None:
        parts.extend(["--mbp", str(mbp_dir)])
    parts.extend(["--output", "pipeline_day_trading_cleaned/features"])
    return " ".join(parts)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _coerce_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts, index=series.index)
    return ts.dt.tz_localize(None)


def _normal_text(series: pd.Series, *, upper: bool = True) -> pd.Series:
    out = series.astype("string").str.strip()
    if upper:
        out = out.str.upper()
    return out


def _output_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if str(c) not in HELPER_COLUMNS]


def _drop_helpers(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in HELPER_COLUMNS if c in df.columns], errors="ignore")


def _stage_stats(
    stats: dict[str, Any],
    stage: str,
    before: int,
    after: int,
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    stats.setdefault("stages", []).append(
        {
            "stage": stage,
            "before_rows": int(before),
            "after_rows": int(after),
            "dropped_rows": int(max(before - after, 0)),
            "detail": detail or {},
        }
    )


def _filter_rows(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    stats: dict[str, Any],
    stage: str,
    detail: dict[str, Any] | None = None,
) -> pd.DataFrame:
    before = len(df)
    if isinstance(mask, pd.Series):
        keep = mask.fillna(False).to_numpy(dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool)
    out = df.loc[keep].copy()
    _stage_stats(stats, stage, before, len(out), detail=detail)
    return out


def _sort_market_rows(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in SORT_TIE_COLUMNS if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return (
        df.sort_values(sort_cols, kind="mergesort", na_position="last")
        .reset_index(drop=True)
    )


def _drop_exact_duplicates(df: pd.DataFrame, *, stats: dict[str, Any]) -> pd.DataFrame:
    subset = _output_columns(df)
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="last").copy() if subset else df.copy()
    _stage_stats(stats, "drop_exact_duplicate_rows", before, len(out), detail={"subset": subset})
    return out


def _drop_duplicate_keys(
    df: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    stats: dict[str, Any],
    stage: str,
) -> pd.DataFrame:
    keys = [c for c in key_columns if c in df.columns]
    if not keys:
        _stage_stats(stats, stage, len(df), len(df), detail={"key_columns": []})
        return df.copy()
    before = len(df)
    out = df.drop_duplicates(subset=keys, keep="last").copy()
    _stage_stats(stats, stage, before, len(out), detail={"key_columns": keys})
    return out


def _positive_median(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vals = vals[(vals > 0) & vals.notna()]
    if vals.empty:
        return None
    return _safe_float(vals.median())


def _drop_price_ratio_outliers(
    df: pd.DataFrame,
    *,
    price_col: str,
    max_price_ratio: float,
    stats: dict[str, Any],
    stage: str,
) -> pd.DataFrame:
    if price_col not in df.columns or max_price_ratio <= 1:
        _stage_stats(stats, stage, len(df), len(df), detail={"price_col": price_col, "enabled": False})
        return df.copy()

    med = _positive_median(df[price_col])
    if med is None or med <= 0:
        _stage_stats(stats, stage, len(df), len(df), detail={"price_col": price_col, "median": med})
        return df.copy()

    price = pd.to_numeric(df[price_col], errors="coerce")
    lower = med / float(max_price_ratio)
    upper = med * float(max_price_ratio)
    mask = price.between(lower, upper, inclusive="both")
    return _filter_rows(
        df,
        mask,
        stats=stats,
        stage=stage,
        detail={"price_col": price_col, "median": med, "lower": lower, "upper": upper},
    )


def _isolated_price_jump_mask(
    df: pd.DataFrame,
    *,
    price_col: str,
    max_step_pct: float,
) -> pd.Series:
    if price_col not in df.columns or "ts_event" not in df.columns or max_step_pct <= 0:
        return pd.Series(False, index=df.index)

    work = pd.DataFrame(
        {
            "price": pd.to_numeric(df[price_col], errors="coerce"),
            "symbol": df["symbol"].astype("string") if "symbol" in df.columns else "ALL",
        },
        index=df.index,
    )
    grouped = work.groupby("symbol", sort=False, dropna=False)["price"]
    prev_price = grouped.shift(1)
    next_price = grouped.shift(-1)
    price = work["price"]

    prev_pct = (price - prev_price).abs() / prev_price.abs()
    next_pct = (price - next_price).abs() / next_price.abs()
    bridge_pct = (next_price - prev_price).abs() / prev_price.abs()
    finite = (
        price.gt(0)
        & prev_price.gt(0)
        & next_price.gt(0)
        & prev_pct.replace([np.inf, -np.inf], np.nan).notna()
        & next_pct.replace([np.inf, -np.inf], np.nan).notna()
        & bridge_pct.replace([np.inf, -np.inf], np.nan).notna()
    )
    return finite & (prev_pct > max_step_pct) & (next_pct > max_step_pct) & (bridge_pct <= max_step_pct / 2.0)


def _drop_isolated_price_jumps(
    df: pd.DataFrame,
    *,
    price_col: str,
    cfg: CleanConfig,
    stats: dict[str, Any],
    stage: str,
) -> pd.DataFrame:
    if not cfg.drop_isolated_price_jumps:
        _stage_stats(stats, stage, len(df), len(df), detail={"enabled": False})
        return df.copy()
    spike = _isolated_price_jump_mask(df, price_col=price_col, max_step_pct=cfg.max_step_pct)
    return _filter_rows(
        df,
        ~spike,
        stats=stats,
        stage=stage,
        detail={"price_col": price_col, "threshold_pct": cfg.max_step_pct, "isolated_spikes": int(spike.sum())},
    )


def _ensure_symbol(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "symbol" not in out.columns:
        if "raw_symbol" in out.columns:
            out["symbol"] = out["raw_symbol"]
        elif "contract_symbol" in out.columns:
            out["symbol"] = out["contract_symbol"]
        elif "instrument_id" in out.columns:
            out["symbol"] = out["instrument_id"].astype("string")
    if "symbol" in out.columns:
        out["symbol"] = _normal_text(out["symbol"], upper=True)
    return out


def _apply_contract_filters(df: pd.DataFrame, *, cfg: CleanConfig, stats: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    expected_symbol = str(cfg.expected_symbol or "").strip().upper()
    expected_root = str(cfg.expected_root or "").strip().upper()

    if "symbol" not in out.columns:
        _stage_stats(stats, "contract_filter", len(out), len(out), detail={"symbol_column": False})
        return out

    before = len(out)
    mask = pd.Series(True, index=out.index)
    detail: dict[str, Any] = {"expected_symbol": expected_symbol or None, "expected_root": expected_root or None}

    if expected_symbol:
        mask &= out["symbol"].astype("string").str.upper().eq(expected_symbol)
    if expected_root:
        parsed = out["symbol"].astype("string").str.upper().map(_parse_contract_symbol)
        roots = parsed.map(lambda item: item["root"] if item else None)
        mask &= roots.eq(expected_root)
        detail["detected_roots_before_filter"] = {
            str(k): int(v)
            for k, v in roots.fillna("UNPARSED").value_counts(dropna=False).head(20).items()
        }

    out = out.loc[mask.fillna(False)].copy()
    _stage_stats(stats, "contract_filter", before, len(out), detail=detail)
    return out


def _load_market_input(path_text: str, *, dataset: str) -> tuple[pd.DataFrame, list[str]]:
    files = _expand_input(Path(path_text).expanduser().resolve())
    frames: list[pd.DataFrame] = []
    offset = 0
    for file_path in files:
        frame = _read_one(file_path)
        frame = _coerce_market_schema(frame)
        frame["__source_file"] = str(file_path)
        frame["__source_row"] = np.arange(len(frame), dtype=np.int64)
        frame["__ingest_order"] = np.arange(offset, offset + len(frame), dtype=np.int64)
        offset += len(frame)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No input files loaded for {dataset}: {path_text}")
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True, sort=False)
    return df, [str(p) for p in files]


def _coerce_common(df: pd.DataFrame, *, kind: str, cfg: CleanConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "kind": kind,
        "initial_rows": int(len(df)),
        "stages": [],
    }
    out = df.copy()

    if "ts_event" not in out.columns and "ts_recv" in out.columns:
        out["ts_event"] = out["ts_recv"]
    if "ts_event" in out.columns:
        out["ts_event"] = _coerce_ts(out["ts_event"])
    if "ts_recv" in out.columns:
        out["ts_recv"] = _coerce_ts(out["ts_recv"])

    out = _ensure_symbol(out)
    for col in ("action", "side"):
        if col in out.columns:
            out[col] = _normal_text(out[col], upper=True)
    for col in out.columns:
        name = str(col)
        if (
            name in {"rtype", "publisher_id", "instrument_id", "depth", "price", "size", "channel_id", "order_id", "flags", "ts_in_delta", "sequence"}
            or name.startswith(("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_"))
        ):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    critical = ["ts_event", "symbol"]
    if kind == "mbo":
        critical.extend(["action", "side", "price", "size", "order_id"])
    elif kind == "mbp":
        critical.extend(["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"])
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    missing_cols = [c for c in critical if c not in out.columns]
    if missing_cols:
        raise ValueError(f"{kind.upper()} input is missing required columns: {missing_cols}")

    out = _apply_contract_filters(out, cfg=cfg, stats=stats)

    symbol_ok = ~out["symbol"].astype("string").str.strip().str.upper().isin(UNKNOWN_STRINGS)
    ts_ok = out["ts_event"].notna()
    out = _filter_rows(
        out,
        ts_ok & symbol_ok,
        stats=stats,
        stage="drop_invalid_timestamp_or_symbol",
        detail={"required": ["ts_event", "symbol"]},
    )
    return out, stats


def clean_mbo_frame(df: pd.DataFrame, *, cfg: CleanConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    out, stats = _coerce_common(df, kind="mbo", cfg=cfg)

    if cfg.drop_mbo_reset_actions and "action" in out.columns:
        reset_mask = out["action"].astype("string").str.upper().eq("R")
        out = _filter_rows(
            out,
            ~reset_mask,
            stats=stats,
            stage="drop_mbo_reset_actions",
            detail={"actions": ["R"], "rows": int(reset_mask.sum())},
        )

    if "side" in out.columns:
        out["side"] = out["side"].replace({"N": "A"})

    action_ok = ~out["action"].astype("string").str.upper().isin(UNKNOWN_STRINGS)
    side_ok = ~out["side"].astype("string").str.upper().isin(UNKNOWN_STRINGS)
    price = pd.to_numeric(out["price"], errors="coerce")
    size = pd.to_numeric(out["size"], errors="coerce")
    order_id_text = out["order_id"].astype("string").str.strip().str.upper()
    order_id_ok = out["order_id"].notna() & ~order_id_text.isin(UNKNOWN_STRINGS)
    critical_ok = action_ok & side_ok & price.gt(0) & size.gt(0) & order_id_ok
    out = _filter_rows(
        out,
        critical_ok,
        stats=stats,
        stage="drop_mbo_critical_missing_or_invalid",
        detail={"required": list(MBO_KEY_COLUMNS)},
    )

    out["price"] = pd.to_numeric(out["price"], errors="coerce").astype(np.float64)
    out["size"] = pd.to_numeric(out["size"], errors="coerce").round().astype(np.int64)

    out = _drop_exact_duplicates(out, stats=stats)
    out = _sort_market_rows(out)
    out = _drop_price_ratio_outliers(
        out,
        price_col="price",
        max_price_ratio=cfg.max_price_ratio,
        stats=stats,
        stage="drop_mbo_price_ratio_outliers",
    )
    out = _drop_duplicate_keys(out, key_columns=MBO_KEY_COLUMNS, stats=stats, stage="resolve_mbo_duplicate_market_keys_keep_last")
    out = _sort_market_rows(out)
    out = _drop_isolated_price_jumps(out, price_col="price", cfg=cfg, stats=stats, stage="drop_mbo_isolated_price_jumps")
    out = _sort_market_rows(out)

    stats["final_rows"] = int(len(out))
    stats["total_dropped_rows"] = int(stats["initial_rows"] - stats["final_rows"])
    return _drop_helpers(out), stats


def _book_level_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([str(c) for c in df.columns if str(c).startswith(prefix)])


def _fix_book_numeric_columns(out: pd.DataFrame) -> pd.DataFrame:
    for col in _book_level_columns(out, "bid_px_") + _book_level_columns(out, "ask_px_"):
        vals = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[col] = vals.where(vals > 0, 0.0).astype(np.float64)
    for prefix in ("bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_"):
        for col in _book_level_columns(out, prefix):
            vals = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[col] = vals.fillna(0).clip(lower=0).round().astype(np.int64)
    return out


def clean_mbp_frame(df: pd.DataFrame, *, cfg: CleanConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    out, stats = _coerce_common(df, kind="mbp", cfg=cfg)
    out = _fix_book_numeric_columns(out)

    bid0 = pd.to_numeric(out["bid_px_00"], errors="coerce")
    ask0 = pd.to_numeric(out["ask_px_00"], errors="coerce")
    bid_sz0 = pd.to_numeric(out["bid_sz_00"], errors="coerce")
    ask_sz0 = pd.to_numeric(out["ask_sz_00"], errors="coerce")
    spread = ask0 - bid0
    inferred_tick = _infer_tick_size_from_spread(spread)
    used_tick = float(cfg.tick_size) if cfg.tick_size and cfg.tick_size > 0 else inferred_tick

    bbo_ok = bid0.gt(0) & ask0.gt(0) & bid_sz0.ge(0) & ask_sz0.ge(0) & ask0.gt(bid0)
    detail: dict[str, Any] = {
        "missing_or_non_positive_bbo": int((bid0.le(0) | ask0.le(0) | bid0.isna() | ask0.isna()).sum()),
        "crossed_bbo": int((bid0.gt(0) & ask0.gt(0) & ask0.lt(bid0)).sum()),
        "locked_bbo": int((bid0.gt(0) & ask0.gt(0) & ask0.eq(bid0)).sum()),
        "tick_size_used": used_tick,
        "max_spread_ticks": cfg.max_spread_ticks,
    }
    if not cfg.drop_locked_bbo:
        bbo_ok = bid0.gt(0) & ask0.gt(0) & bid_sz0.ge(0) & ask_sz0.ge(0) & ask0.ge(bid0)

    if cfg.drop_wide_spread and used_tick and used_tick > 0 and cfg.max_spread_ticks > 0:
        wide = spread > float(used_tick) * float(cfg.max_spread_ticks)
        detail["wide_spread_rows"] = int(wide.sum())
        bbo_ok &= ~wide.fillna(True)
    else:
        detail["wide_spread_rows"] = 0
        detail["wide_filter_enabled"] = False

    out = _filter_rows(out, bbo_ok, stats=stats, stage="drop_mbp_invalid_executable_bbo", detail=detail)

    if "price" in out.columns and cfg.preserve_raw_mbp_price and "price_raw" not in out.columns:
        out["price_raw"] = pd.to_numeric(out["price"], errors="coerce")
    if cfg.recompute_mbp_price_from_mid:
        out["price"] = ((pd.to_numeric(out["bid_px_00"], errors="coerce") + pd.to_numeric(out["ask_px_00"], errors="coerce")) / 2.0).astype(np.float64)
        _stage_stats(
            stats,
            "recompute_mbp_price_from_mid",
            len(out),
            len(out),
            detail={"price": "(bid_px_00 + ask_px_00) / 2"},
        )
    elif "price" in out.columns:
        out = _drop_price_ratio_outliers(
            out,
            price_col="price",
            max_price_ratio=cfg.max_price_ratio,
            stats=stats,
            stage="drop_mbp_price_ratio_outliers",
        )

    out = _drop_exact_duplicates(out, stats=stats)
    out = _sort_market_rows(out)
    out = _drop_duplicate_keys(out, key_columns=MBP_KEY_COLUMNS, stats=stats, stage="resolve_mbp_duplicate_market_keys_keep_last")
    out = _sort_market_rows(out)
    out = _drop_isolated_price_jumps(out, price_col="price", cfg=cfg, stats=stats, stage="drop_mbp_isolated_price_jumps")
    out = _sort_market_rows(out)

    stats["final_rows"] = int(len(out))
    stats["total_dropped_rows"] = int(stats["initial_rows"] - stats["final_rows"])
    stats["tick_size_used"] = used_tick
    return _drop_helpers(out), stats


def _time_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if "ts_event" not in df.columns or len(df) == 0:
        return None, None
    ts = pd.to_datetime(df["ts_event"], errors="coerce").dropna()
    if ts.empty:
        return None, None
    return pd.Timestamp(ts.min()), pd.Timestamp(ts.max())


def _trim_to_overlap(
    mbo: pd.DataFrame,
    mbp: pd.DataFrame,
    *,
    report: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mbo_start, mbo_end = _time_bounds(mbo)
    mbp_start, mbp_end = _time_bounds(mbp)
    if mbo_start is None or mbo_end is None or mbp_start is None or mbp_end is None:
        report["overlap_trim"] = {"enabled": True, "status": "skipped_missing_bounds"}
        return mbo, mbp

    start = max(mbo_start, mbp_start)
    end = min(mbo_end, mbp_end)
    if start > end:
        raise ValueError(f"MBO/MBP have no time overlap after cleaning: MBO={mbo_start}->{mbo_end}, MBP={mbp_start}->{mbp_end}")

    before_mbo, before_mbp = len(mbo), len(mbp)
    mbo_out = mbo[(mbo["ts_event"] >= start) & (mbo["ts_event"] <= end)].copy()
    mbp_out = mbp[(mbp["ts_event"] >= start) & (mbp["ts_event"] <= end)].copy()
    report["overlap_trim"] = {
        "enabled": True,
        "overlap_start": str(start),
        "overlap_end": str(end),
        "mbo_before_rows": int(before_mbo),
        "mbo_after_rows": int(len(mbo_out)),
        "mbp_before_rows": int(before_mbp),
        "mbp_after_rows": int(len(mbp_out)),
    }
    return mbo_out, mbp_out


def _write_parquet_shards(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    rows_per_shard: int,
    compression: str,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    rows_per_shard = max(int(rows_per_shard), 1)
    for shard_idx, start in enumerate(range(0, len(df), rows_per_shard)):
        shard = df.iloc[start : start + rows_per_shard].copy()
        path = out_dir / f"{stem}_{shard_idx:05d}.parquet"
        shard.to_parquet(path, index=False, compression=compression)
        ts_min, ts_max = _time_bounds(shard)
        records.append(
            {
                "shard_idx": shard_idx,
                "path": str(path.resolve()),
                "rows": int(len(shard)),
                "ts_min": None if ts_min is None else str(ts_min),
                "ts_max": None if ts_max is None else str(ts_max),
            }
        )
    return records


def _build_post_audit(
    *,
    mbo_path: Path,
    mbp_path: Path | None,
    cfg: CleanConfig,
) -> dict[str, Any]:
    gap_threshold = pd.Timedelta(cfg.gap_threshold)
    mbo_df, mbo_files = _load_market_input(str(mbo_path), dataset="mbo_clean")
    mbo_df = _drop_helpers(mbo_df)
    mbo_report = _audit_dataset(
        mbo_df,
        kind="mbo",
        files=mbo_files,
        gap_threshold=gap_threshold,
        tick_size=cfg.tick_size,
        max_spread_ticks=cfg.max_spread_ticks,
        max_price_ratio=cfg.max_price_ratio,
        max_step_pct=cfg.max_step_pct,
        expected_root=cfg.expected_root,
        expected_symbol=cfg.expected_symbol,
        top_n=max(1, cfg.top_n),
    )

    mbp_report = None
    if mbp_path is not None:
        mbp_df, mbp_files = _load_market_input(str(mbp_path), dataset="mbp_clean")
        mbp_df = _drop_helpers(mbp_df)
        mbp_report = _audit_dataset(
            mbp_df,
            kind="mbp",
            files=mbp_files,
            gap_threshold=gap_threshold,
            tick_size=cfg.tick_size,
            max_spread_ticks=cfg.max_spread_ticks,
            max_price_ratio=cfg.max_price_ratio,
            max_step_pct=cfg.max_step_pct,
            expected_root=cfg.expected_root,
            expected_symbol=cfg.expected_symbol,
            top_n=max(1, cfg.top_n),
        )

    cross = _compare_mbo_mbp(mbo_report, mbp_report, top_n=max(1, cfg.top_n))
    issues = []
    issues.extend(mbo_report.get("issues") or [])
    if mbp_report:
        issues.extend(mbp_report.get("issues") or [])
    issues.extend(cross.get("issues") or [])
    return {
        "datasets": {"mbo": mbo_report, "mbp": mbp_report},
        "cross_checks": cross,
        "overall_max_issue_severity": _max_severity(issues),
    }


def clean_market_data(
    *,
    mbo_path: str,
    mbp_path: str | None,
    output_dir: str,
    cfg: CleanConfig,
    rows_per_shard: int,
    compression: str,
    overwrite: bool,
    audit_after: bool,
) -> dict[str, Any]:
    out_root = Path(output_dir).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is not empty: {out_root}. Use --overwrite to replace it.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at_utc": _now_utc(),
        "config": asdict(cfg),
        "inputs": {
            "mbo": str(Path(mbo_path).expanduser().resolve()),
            "mbp": str(Path(mbp_path).expanduser().resolve()) if mbp_path else None,
        },
        "outputs": {},
        "cleaning": {},
    }

    print(f"Loading MBO: {mbo_path}", flush=True)
    mbo_raw, mbo_files = _load_market_input(mbo_path, dataset="mbo")
    report["inputs"]["mbo_files"] = mbo_files
    print(f"Cleaning MBO rows={len(mbo_raw):,}", flush=True)
    mbo_clean, mbo_stats = clean_mbo_frame(mbo_raw, cfg=cfg)
    del mbo_raw

    mbp_clean = None
    mbp_stats = None
    if mbp_path:
        print(f"Loading MBP: {mbp_path}", flush=True)
        mbp_raw, mbp_files = _load_market_input(mbp_path, dataset="mbp")
        report["inputs"]["mbp_files"] = mbp_files
        print(f"Cleaning MBP rows={len(mbp_raw):,}", flush=True)
        mbp_clean, mbp_stats = clean_mbp_frame(mbp_raw, cfg=cfg)
        del mbp_raw

    if cfg.trim_to_overlap and mbp_clean is not None:
        mbo_clean, mbp_clean = _trim_to_overlap(mbo_clean, mbp_clean, report=report)

    mbo_out = out_root / "mbo"
    mbp_out = out_root / "mbp"
    print(f"Writing MBO shards: {mbo_out}", flush=True)
    mbo_records = _write_parquet_shards(
        mbo_clean,
        mbo_out,
        stem="mbo_clean",
        rows_per_shard=rows_per_shard,
        compression=compression,
    )
    del mbo_clean

    mbp_records: list[dict[str, Any]] = []
    if mbp_clean is not None:
        print(f"Writing MBP shards: {mbp_out}", flush=True)
        mbp_records = _write_parquet_shards(
            mbp_clean,
            mbp_out,
            stem="mbp_clean",
            rows_per_shard=rows_per_shard,
            compression=compression,
        )
        del mbp_clean

    report["outputs"] = {
        "root": str(out_root),
        "mbo_dir": str(mbo_out),
        "mbp_dir": str(mbp_out) if mbp_records else None,
        "mbo_shards": mbo_records,
        "mbp_shards": mbp_records,
    }
    report["cleaning"]["mbo"] = mbo_stats
    if mbp_stats is not None:
        report["cleaning"]["mbp"] = mbp_stats

    manifest_path = out_root / "cleaning_manifest.json"
    mbp_cmd_dir = mbp_out if mbp_records else None
    report["verification_commands"] = [
        _audit_command(mbo_out, mbp_cmd_dir, cfg),
        _prepare_training_command(mbo_out, mbp_cmd_dir),
        _prepare_day_trading_command(mbo_out, mbp_cmd_dir),
    ]

    _json_dump(manifest_path, report)

    if audit_after:
        print("Running post-clean audit", flush=True)
        post_audit = _build_post_audit(
            mbo_path=mbo_out,
            mbp_path=mbp_out if mbp_records else None,
            cfg=cfg,
        )
        report["post_clean_audit"] = post_audit
        _json_dump(out_root / "post_clean_audit.json", post_audit)
        _json_dump(manifest_path, report)

    print(f"Cleaning manifest: {manifest_path}", flush=True)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean MBO/MBP raw data into chronological Parquet shards for QuantSystem V19.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mbo", required=True, help="MBO CSV/Parquet file or directory.")
    parser.add_argument("--mbp", default=None, help="Optional MBP CSV/Parquet file or directory.")
    parser.add_argument("--output", default="clean_market_data", help="Output root directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output directory.")
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000, help="Rows per output Parquet shard.")
    parser.add_argument("--compression", default="snappy", help="Parquet compression codec.")
    parser.add_argument("--tick-size", type=float, default=None, help="Known tick size. If omitted, MBP spread checks infer it.")
    parser.add_argument("--max-spread-ticks", type=float, default=20.0, help="Drop MBP rows with spread wider than tick_size times this value.")
    parser.add_argument("--max-price-ratio", type=float, default=5.0, help="Drop main prices outside median/ratio and median*ratio.")
    parser.add_argument("--max-step-pct", type=float, default=0.05, help="Isolated price-spike threshold.")
    parser.add_argument("--keep-locked-bbo", action="store_true", help="Keep MBP rows where ask_px_00 == bid_px_00.")
    parser.add_argument("--keep-wide-spread", action="store_true", help="Keep MBP rows wider than max spread threshold.")
    parser.add_argument("--no-drop-isolated-price-jumps", action="store_true", help="Do not drop isolated adjacent price spikes.")
    parser.add_argument("--keep-mbo-reset-actions", action="store_true", help="Keep MBO reset rows instead of dropping action R.")
    parser.add_argument("--keep-raw-mbp-price", action="store_true", help="Do not recompute MBP price from top-of-book mid.")
    parser.add_argument("--no-preserve-raw-mbp-price", action="store_true", help="Do not copy original MBP price to price_raw before recomputing.")
    parser.add_argument("--expected-root", default=None, help="Optional expected futures root, e.g. 6B.")
    parser.add_argument("--expected-symbol", default=None, help="Optional expected single contract symbol, e.g. 6BM5.")
    parser.add_argument("--trim-to-overlap", action="store_true", help="Trim MBO/MBP to their shared post-clean time overlap.")
    parser.add_argument("--gap-threshold", default="1h", help="Gap threshold used by optional post audit.")
    parser.add_argument("--top-n", type=int, default=10, help="Examples to retain in audit reports.")
    parser.add_argument("--audit-after", action="store_true", help="Run audit_mbo_mbp_data.py logic on cleaned output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cfg = CleanConfig(
        tick_size=args.tick_size,
        max_spread_ticks=float(args.max_spread_ticks),
        max_price_ratio=float(args.max_price_ratio),
        max_step_pct=float(args.max_step_pct),
        drop_locked_bbo=not bool(args.keep_locked_bbo),
        drop_wide_spread=not bool(args.keep_wide_spread),
        drop_isolated_price_jumps=not bool(args.no_drop_isolated_price_jumps),
        drop_mbo_reset_actions=not bool(args.keep_mbo_reset_actions),
        recompute_mbp_price_from_mid=not bool(args.keep_raw_mbp_price),
        preserve_raw_mbp_price=not bool(args.no_preserve_raw_mbp_price),
        expected_root=args.expected_root,
        expected_symbol=args.expected_symbol,
        trim_to_overlap=bool(args.trim_to_overlap),
        gap_threshold=args.gap_threshold,
        top_n=max(1, int(args.top_n)),
    )
    clean_market_data(
        mbo_path=args.mbo,
        mbp_path=args.mbp,
        output_dir=args.output,
        cfg=cfg,
        rows_per_shard=max(1, int(args.rows_per_shard)),
        compression=str(args.compression),
        overwrite=bool(args.overwrite),
        audit_after=bool(args.audit_after),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
