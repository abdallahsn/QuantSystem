"""Memory-safe table reading helpers for QuantSystem v20."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd

from modules.feature_artifact_v19 import iter_table_chunks, read_table
from .databento_schema import FeedType, detect_feed_type, validate_required_columns


@dataclass(frozen=True)
class TableScanReport:
    path: str
    rows_scanned: int
    chunks_scanned: int
    feed_type: str
    missing_required_columns: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def read_table_columns(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    return read_table(path, columns=columns)


def iter_market_chunks(
    path: str,
    *,
    chunk_rows: int,
    columns: list[str] | None = None,
) -> Iterable[pd.DataFrame]:
    yield from iter_table_chunks(path, chunk_rows, columns=columns)


def scan_table_schema(
    path: str,
    *,
    chunk_rows: int = 100_000,
    expected_feed: FeedType | str | None = None,
    max_chunks: int = 1,
) -> TableScanReport:
    rows = 0
    chunks = 0
    detected = FeedType.UNKNOWN
    missing: tuple[str, ...] = ()
    for chunk in iter_market_chunks(path, chunk_rows=chunk_rows):
        chunks += 1
        rows += int(len(chunk))
        detected = detect_feed_type(chunk.columns)
        feed = expected_feed if isinstance(expected_feed, FeedType) else FeedType(str(expected_feed).lower()) if expected_feed else detected
        schema_report = validate_required_columns(chunk, feed)
        missing = tuple(schema_report.get("missing_columns", ()))
        if chunks >= int(max_chunks):
            break
    return TableScanReport(
        path=str(path),
        rows_scanned=int(rows),
        chunks_scanned=int(chunks),
        feed_type=detected.value,
        missing_required_columns=missing,
        passed=bool(chunks > 0 and not missing),
    )
