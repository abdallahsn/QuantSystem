"""Databento MBO/MBP schema contracts for QuantSystem v20."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd


class FeedType(str, Enum):
    MBO = "mbo"
    MBP = "mbp"
    UNKNOWN = "unknown"


MBO_REQUIRED_COLUMNS = frozenset({"ts_event", "action", "side", "price", "size"})
MBP_REQUIRED_COLUMNS = frozenset({"ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"})
TIMESTAMP_COLUMNS = ("ts_event", "ts_recv")
SYMBOL_COLUMNS = ("symbol", "instrument_id", "contract_symbol")


def _coerce_feed_type(feed_type: FeedType | str) -> FeedType:
    if isinstance(feed_type, FeedType):
        return feed_type
    return FeedType(str(feed_type).lower())


@dataclass(frozen=True)
class FeedSchema:
    feed_type: FeedType
    required_columns: frozenset[str]
    timestamp_columns: tuple[str, ...] = TIMESTAMP_COLUMNS
    symbol_columns: tuple[str, ...] = SYMBOL_COLUMNS


def schema_for(feed_type: FeedType | str) -> FeedSchema:
    feed = _coerce_feed_type(feed_type)
    if feed == FeedType.MBO:
        return FeedSchema(feed_type=feed, required_columns=MBO_REQUIRED_COLUMNS)
    if feed == FeedType.MBP:
        return FeedSchema(feed_type=feed, required_columns=MBP_REQUIRED_COLUMNS)
    return FeedSchema(feed_type=FeedType.UNKNOWN, required_columns=frozenset({"ts_event"}))


def detect_feed_type(columns: Iterable[str]) -> FeedType:
    cols = {str(col) for col in columns}
    if MBP_REQUIRED_COLUMNS.issubset(cols):
        return FeedType.MBP
    if MBO_REQUIRED_COLUMNS.issubset(cols):
        return FeedType.MBO
    return FeedType.UNKNOWN


def validate_required_columns(df: pd.DataFrame, feed_type: FeedType | str) -> dict:
    schema = schema_for(feed_type)
    columns = {str(col) for col in df.columns}
    missing = sorted(schema.required_columns - columns)
    return {
        "feed_type": schema.feed_type.value,
        "rows": int(len(df)),
        "columns": sorted(columns),
        "required_columns": sorted(schema.required_columns),
        "missing_columns": missing,
        "passed": not missing,
    }
