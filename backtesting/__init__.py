"""Backtesting interfaces for QuantSystem v20."""

from .execution_model import ExecutionAssumptions, estimate_fill_probability

__all__ = ["ExecutionAssumptions", "estimate_fill_probability"]
