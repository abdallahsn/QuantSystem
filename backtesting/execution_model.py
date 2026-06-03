"""Execution assumptions and fill-probability approximation for v20 backtests."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ExecutionAssumptions:
    tick_size: float = 0.0001
    tick_value: float = 6.25
    commission_per_side: float = 0.75
    min_spread_ticks: float = 1.0
    min_slippage_ticks: float = 0.5
    latency_rows: int = 1
    passive_fill_model: str = "queue_decay"

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_fill_probability(
    *,
    queue_ahead: float,
    trade_through_size: float,
    visible_depth: float,
    latency_rows: int = 1,
    aggressiveness: float = 0.0,
) -> float:
    """Approximate passive fill probability without assuming guaranteed fills."""
    q = max(float(queue_ahead), 0.0)
    traded = max(float(trade_through_size), 0.0)
    depth = max(float(visible_depth), 1e-9)
    latency_penalty = np.exp(-0.08 * max(int(latency_rows), 0))
    queue_score = traded / max(q + depth, 1e-9)
    aggressive_bonus = max(float(aggressiveness), 0.0) * 0.25
    return float(np.clip((queue_score + aggressive_bonus) * latency_penalty, 0.0, 1.0))
