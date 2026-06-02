#!/usr/bin/env python3
"""
Convert rich MBO/MBP CSV files to Parquet and build a threshold-analysis
feature parquet accepted by tools.diagnostics.analyze_thresholds.

Default outputs:
  - rich_mbo.parquet
  - rich_mbp.parquet
  - rich_threshold_features.parquet
  - rich_conversion_report.json

Example:
  python3 convert_rich_csv_to_parquet.py
  python3 -m tools.diagnostics.analyze_thresholds --parquet rich_threshold_features.parquet --no-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRADE_ACTIONS = {"T", "F", "TRADE", "EXECUTE", "E", "0"}
BUY_SIDES = {"A", "ASK", "BUY", "BOT"}
SELL_SIDES = {"B", "BID", "S", "SELL"}
TIMESTAMP_COLUMNS = ("ts_event", "ts_recv")
STRING_COLUMNS = {"action", "side", "symbol"}
NUMERIC_COLUMNS = {
    "rtype",
    "publisher_id",
    "instrument_id",
    "depth",
    "price",
    "size",
    "channel_id",
    "order_id",
    "flags",
    "ts_in_delta",
    "sequence",
}


def _normalize_freq(freq: str) -> str:
    value = str(freq or "5min").strip()
    if not value:
        return "5min"
    return value.replace("T", "min").replace("H", "h")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _coerce_market_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in TIMESTAMP_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    for col in out.columns:
        if (
            col in NUMERIC_COLUMNS
            or col.startswith(("bid_px_", "ask_px_", "bid_sz_", "ask_sz_", "bid_ct_", "ask_ct_"))
        ):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")

    return out


def _read_market_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    return _coerce_market_schema(df)


def _safe_unique_sample(series: pd.Series, limit: int = 20) -> list[Any]:
    if series.empty:
        return []
    values = series.dropna().astype(str).unique().tolist()
    return values[:limit]


def _diagnose_market_df(df: pd.DataFrame, name: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        valid_ts = ts.dropna()
        report.update(
            {
                "missing_ts_event": int(ts.isna().sum()),
                "duplicate_ts_event": int(ts.duplicated().sum()),
                "ts_event_monotonic_in_input_order": bool(valid_ts.is_monotonic_increasing),
                "time_start": str(valid_ts.min()) if len(valid_ts) else None,
                "time_end": str(valid_ts.max()) if len(valid_ts) else None,
            }
        )
        if len(valid_ts) > 1:
            report["backward_timestamp_steps"] = int((valid_ts.diff().dropna() < pd.Timedelta(0)).sum())

    if "symbol" in df.columns:
        report["symbols_sample"] = _safe_unique_sample(df["symbol"])
        report["symbol_count"] = int(df["symbol"].dropna().astype(str).nunique())

    if "instrument_id" in df.columns:
        report["instrument_ids_sample"] = _safe_unique_sample(df["instrument_id"])
        report["instrument_id_count"] = int(df["instrument_id"].dropna().nunique())

    if "price" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce")
        report["missing_price"] = int(price.isna().sum())
        report["non_positive_price"] = int((price.dropna() <= 0).sum())

    if "size" in df.columns:
        size = pd.to_numeric(df["size"], errors="coerce")
        report["missing_size"] = int(size.isna().sum())
        report["negative_size"] = int((size.dropna() < 0).sum())

    if "action" in df.columns:
        report["action_counts"] = {
            str(k): int(v) for k, v in df["action"].astype("string").value_counts(dropna=False).to_dict().items()
        }

    return report


def _write_parquet(df: pd.DataFrame, out_path: Path, compression: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression=compression)


def _bars_from_ticks(
    df: pd.DataFrame,
    *,
    price_col: str,
    freq: str,
    size_col: str = "size",
    count_col: str = "tick_count",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", count_col])

    needed = ["ts_event", price_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot aggregate bars; missing columns: {missing}")

    work = df.copy()
    work["ts_event"] = pd.to_datetime(work["ts_event"], utc=True, errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work[work["ts_event"].notna() & work[price_col].notna() & (work[price_col] > 0)]
    if work.empty:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", count_col])

    if size_col not in work.columns:
        work[size_col] = 0.0
    work[size_col] = pd.to_numeric(work[size_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    work = work.sort_values("ts_event").reset_index(drop=True)
    work["bar_ts"] = work["ts_event"].dt.floor(freq)
    grouped = work.groupby("bar_ts", sort=True)

    bars = grouped.agg(
        open=(price_col, "first"),
        high=(price_col, "max"),
        low=(price_col, "min"),
        close=(price_col, "last"),
        volume=(size_col, "sum"),
        **{count_col: (price_col, "size")},
    ).reset_index()
    bars = bars.rename(columns={"bar_ts": "ts_event"})
    return bars.dropna(subset=["open", "high", "low", "close"]).sort_values("ts_event").reset_index(drop=True)


def _prepare_mbo_bars(mbo: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "ts_event" not in mbo.columns:
        raise ValueError("MBO CSV must contain ts_event")
    if "price" not in mbo.columns:
        raise ValueError("MBO CSV must contain price")

    work = mbo.copy()
    if "action" in work.columns:
        actions = work["action"].astype("string").str.strip().str.upper()
        work = work[actions.isin(TRADE_ACTIONS)].copy()

    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["size"] = pd.to_numeric(work.get("size", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work[work["price"].notna() & (work["price"] > 0)]

    bars = _bars_from_ticks(work, price_col="price", freq=freq, size_col="size", count_col="tick_count")
    bars["source"] = "MBO"

    if not work.empty and "side" in work.columns:
        side = work["side"].astype("string").str.strip().str.upper()
        work["buy_volume"] = np.where(side.isin(BUY_SIDES), work["size"], 0.0)
        work["sell_volume"] = np.where(side.isin(SELL_SIDES), work["size"], 0.0)
        side_group = work.assign(bar_ts=work["ts_event"].dt.floor(freq)).groupby("bar_ts", sort=True)
        side_bars = side_group.agg(buy_volume=("buy_volume", "sum"), sell_volume=("sell_volume", "sum")).reset_index()
        side_bars = side_bars.rename(columns={"bar_ts": "ts_event"})
        bars = bars.merge(side_bars, on="ts_event", how="left")
    else:
        bars["buy_volume"] = 0.0
        bars["sell_volume"] = 0.0

    bars["buy_volume"] = bars["buy_volume"].fillna(0.0)
    bars["sell_volume"] = bars["sell_volume"].fillna(0.0)
    denom = (bars["buy_volume"] + bars["sell_volume"]).replace(0.0, np.nan)
    bars["buy_ratio"] = (bars["buy_volume"] / denom).fillna(0.5)

    info = {
        "mbo_trade_rows_used": int(len(work)),
        "mbo_bars": int(len(bars)),
    }
    return bars, info


def _prepare_mbp_bars(mbp: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "ts_event" not in mbp.columns:
        raise ValueError("MBP CSV must contain ts_event")

    work = mbp.copy()
    bid = pd.to_numeric(work.get("bid_px_00"), errors="coerce")
    ask = pd.to_numeric(work.get("ask_px_00"), errors="coerce")
    valid_bbo = bid.notna() & ask.notna() & (bid > 0) & (ask > 0) & (ask >= bid)
    mid = (bid + ask) / 2.0

    if "price" in work.columns:
        fallback = pd.to_numeric(work["price"], errors="coerce")
        mid = mid.where(valid_bbo, fallback)

    work["mid_price"] = mid
    work["size"] = 0.0
    bars = _bars_from_ticks(work, price_col="mid_price", freq=freq, size_col="size", count_col="snapshot_count")
    bars["source"] = "MBP"

    work["spread"] = (ask - bid).where(valid_bbo, np.nan)
    spread_bars = (
        work.assign(ts_event=pd.to_datetime(work["ts_event"], utc=True, errors="coerce"))
        .dropna(subset=["ts_event"])
        .assign(bar_ts=lambda x: x["ts_event"].dt.floor(freq))
        .groupby("bar_ts", sort=True)
        .agg(avg_spread=("spread", "mean"))
        .reset_index()
        .rename(columns={"bar_ts": "ts_event"})
    )
    bars = bars.merge(spread_bars, on="ts_event", how="left")

    info = {
        "mbp_snapshot_rows_used": int(work["mid_price"].notna().sum()),
        "mbp_bars": int(len(bars)),
    }
    return bars, info


def _combine_mbo_mbp_bars(mbo_bars: pd.DataFrame, mbp_bars: pd.DataFrame) -> pd.DataFrame:
    if mbo_bars.empty and mbp_bars.empty:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "atr_14"])
    if mbp_bars.empty:
        out = mbo_bars.copy()
        out["snapshot_count"] = 0
        out["avg_spread"] = np.nan
        return out
    if mbo_bars.empty:
        out = mbp_bars.copy()
        out["tick_count"] = 0
        out["buy_volume"] = 0.0
        out["sell_volume"] = 0.0
        out["buy_ratio"] = 0.5
        return out

    merged = pd.merge(mbp_bars, mbo_bars, on="ts_event", how="outer", suffixes=("_mbp", "_mbo"))
    out = pd.DataFrame({"ts_event": merged["ts_event"]})

    for col in ("open", "high", "low", "close"):
        out[col] = merged[f"{col}_mbo"].combine_first(merged[f"{col}_mbp"])

    out["volume"] = merged.get("volume_mbo", pd.Series(0.0, index=merged.index)).fillna(0.0)
    out["tick_count"] = merged.get("tick_count", pd.Series(0, index=merged.index)).fillna(0).astype(int)
    out["snapshot_count"] = merged.get("snapshot_count", pd.Series(0, index=merged.index)).fillna(0).astype(int)
    out["buy_volume"] = merged.get("buy_volume", pd.Series(0.0, index=merged.index)).fillna(0.0)
    out["sell_volume"] = merged.get("sell_volume", pd.Series(0.0, index=merged.index)).fillna(0.0)
    out["buy_ratio"] = merged.get("buy_ratio", pd.Series(0.5, index=merged.index)).fillna(0.5)
    out["avg_spread"] = merged.get("avg_spread", pd.Series(np.nan, index=merged.index))
    out["source"] = np.where(merged["open_mbo"].notna(), "MBO", "MBP")

    return out.dropna(subset=["open", "high", "low", "close"]).sort_values("ts_event").reset_index(drop=True)


def _add_causal_atr(bars: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = bars.copy().sort_values("ts_event").reset_index(drop=True)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(window=window, min_periods=1).mean()
    out["price"] = out["close"]
    return out


def build_threshold_features(mbo: pd.DataFrame, mbp: pd.DataFrame, *, freq: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    mbo_bars, mbo_info = _prepare_mbo_bars(mbo, freq)
    mbp_bars, mbp_info = _prepare_mbp_bars(mbp, freq)
    bars = _combine_mbo_mbp_bars(mbo_bars, mbp_bars)
    bars = _add_causal_atr(bars)

    required = {"open", "high", "low", "close", "atr_14", "ts_event"}
    missing_required = sorted(required - set(bars.columns))
    if missing_required:
        raise ValueError(f"Feature parquet missing required columns after build: {missing_required}")

    bars = bars.dropna(subset=["ts_event", "open", "high", "low", "close", "atr_14"])
    bars = bars.sort_values("ts_event").reset_index(drop=True)

    if bars.empty:
        raise ValueError("No threshold feature rows were produced. Check trade actions, prices, and timestamps.")

    report = {
        **mbo_info,
        **mbp_info,
        "feature_rows": int(len(bars)),
        "feature_time_start": str(bars["ts_event"].iloc[0]),
        "feature_time_end": str(bars["ts_event"].iloc[-1]),
        "feature_ts_event_monotonic": bool(pd.to_datetime(bars["ts_event"], utc=True).is_monotonic_increasing),
        "feature_duplicate_ts_event": int(pd.to_datetime(bars["ts_event"], utc=True).duplicated().sum()),
        "feature_sources": {str(k): int(v) for k, v in bars["source"].value_counts(dropna=False).to_dict().items()},
    }
    return bars, report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert rich_mbo.csv/rich_mbp.csv to parquet and build tools.diagnostics.analyze_thresholds input."
    )
    parser.add_argument("--mbo", default=str(root / "rich_mbo.csv"), help="Input MBO CSV path.")
    parser.add_argument("--mbp", default=str(root / "rich_mbp.csv"), help="Input MBP CSV path.")
    parser.add_argument("--out-dir", default=str(root), help="Directory for output parquet/report files.")
    parser.add_argument("--freq", default="5min", help="Bar frequency for threshold features, e.g. 1min or 5min.")
    parser.add_argument("--compression", default="snappy", help="Parquet compression codec.")
    parser.add_argument("--mbo-out", default="rich_mbo.parquet", help="Output raw MBO parquet filename/path.")
    parser.add_argument("--mbp-out", default="rich_mbp.parquet", help="Output raw MBP parquet filename/path.")
    parser.add_argument(
        "--features-out",
        default="rich_threshold_features.parquet",
        help="Output OHLC+ATR feature parquet filename/path for tools.diagnostics.analyze_thresholds.",
    )
    parser.add_argument("--report-out", default="rich_conversion_report.json", help="Output conversion report JSON.")
    return parser.parse_args()


def _resolve_output_path(value: str, out_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return out_dir / path


def main() -> None:
    args = parse_args()
    mbo_path = Path(args.mbo).expanduser().resolve()
    mbp_path = Path(args.mbp).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    freq = _normalize_freq(args.freq)

    mbo_out = _resolve_output_path(args.mbo_out, out_dir)
    mbp_out = _resolve_output_path(args.mbp_out, out_dir)
    features_out = _resolve_output_path(args.features_out, out_dir)
    report_out = _resolve_output_path(args.report_out, out_dir)

    mbo = _read_market_csv(mbo_path)
    mbp = _read_market_csv(mbp_path)

    report: dict[str, Any] = {
        "inputs": {"mbo": mbo_path, "mbp": mbp_path},
        "outputs": {"mbo_parquet": mbo_out, "mbp_parquet": mbp_out, "features_parquet": features_out},
        "bar_frequency": freq,
        "raw": {
            "mbo": _diagnose_market_df(mbo, "mbo"),
            "mbp": _diagnose_market_df(mbp, "mbp"),
        },
        "notes": [
            "raw MBO/MBP parquet files preserve input rows after timestamp/numeric coercion",
            "feature parquet uses MBO trade bars first and MBP mid-price bars only where no MBO trade bar exists",
            "atr_14 is a causal rolling mean over true range in timestamp order",
        ],
    }

    _write_parquet(mbo, mbo_out, args.compression)
    _write_parquet(mbp, mbp_out, args.compression)

    features, feature_report = build_threshold_features(mbo, mbp, freq=freq)
    _write_parquet(features, features_out, args.compression)
    report["features"] = feature_report

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2, ensure_ascii=False)

    print(f"Wrote raw MBO parquet     : {mbo_out}")
    print(f"Wrote raw MBP parquet     : {mbp_out}")
    print(f"Wrote threshold features  : {features_out}")
    print(f"Wrote conversion report   : {report_out}")
    print(f"Feature rows              : {len(features)}")
    print(f"Feature time range        : {features['ts_event'].iloc[0]} -> {features['ts_event'].iloc[-1]}")
    print()
    print("Next:")
    print(f"  python3 -m tools.diagnostics.analyze_thresholds --parquet {features_out} --no-plots")


if __name__ == "__main__":
    main()
