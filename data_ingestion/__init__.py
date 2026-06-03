"""Data ingestion interfaces for QuantSystem v20."""

from .databento_schema import FeedSchema, FeedType, detect_feed_type, validate_required_columns
from .table_reader import TableScanReport, iter_market_chunks, read_table_columns

__all__ = [
    "FeedSchema",
    "FeedType",
    "TableScanReport",
    "detect_feed_type",
    "iter_market_chunks",
    "read_table_columns",
    "validate_required_columns",
]
