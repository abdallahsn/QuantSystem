"""Causal market-data cleaning with explicit row-count reports."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanConfig:
    sort_by_ts: bool = True
    drop_invalid_timestamps: bool = True
    drop_exact_duplicates: bool = False
    strict_positive_trade_prices: bool = False


@dataclass(frozen=True)
class CleanReport:
    input_rows: int
    output_rows: int
    invalid_timestamp_rows: int
    exact_duplicate_rows: int
    invalid_price_rows: int
    rows_dropped: int
    sorted_by_ts: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def clean_market_data(df: pd.DataFrame, config: CleanConfig | None = None) -> tuple[pd.DataFrame, CleanReport]:
    cfg = config or CleanConfig()
    out = df.copy()
    input_rows = int(len(out))
    invalid_ts = 0
    duplicate_rows = 0
    invalid_prices = 0

    if "ts_event" not in out.columns and "ts_recv" in out.columns:
        out["ts_event"] = out["ts_recv"]
    if "ts_event" not in out.columns:
        raise ValueError("clean_market_data requires ts_event or ts_recv")

    out["ts_event"] = _coerce_ts(out["ts_event"])
    invalid_ts = int(out["ts_event"].isna().sum())
    if invalid_ts and cfg.drop_invalid_timestamps:
        out = out.loc[out["ts_event"].notna()].copy()

    if cfg.sort_by_ts and "ts_event" in out.columns:
        out = out.sort_values("ts_event").reset_index(drop=True)

    duplicate_rows = int(out.duplicated().sum())
    if duplicate_rows and cfg.drop_exact_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)

    if "price" in out.columns:
        price = pd.to_numeric(out["price"], errors="coerce")
        invalid_prices = int((~np.isfinite(price.to_numpy(dtype=np.float64)) | (price <= 0)).sum())
        if invalid_prices and cfg.strict_positive_trade_prices:
            trade_mask = out.get("action", "").astype(str).str.upper().isin({"T", "F"})
            keep = ~(trade_mask & (price <= 0))
            out = out.loc[keep].reset_index(drop=True)

    report = CleanReport(
        input_rows=input_rows,
        output_rows=int(len(out)),
        invalid_timestamp_rows=invalid_ts,
        exact_duplicate_rows=duplicate_rows,
        invalid_price_rows=invalid_prices,
        rows_dropped=int(input_rows - len(out)),
        sorted_by_ts=bool(cfg.sort_by_ts),
    )
    return out, report
