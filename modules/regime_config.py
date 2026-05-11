"""
Compatibility shim for centralized regime configuration.

Canonical configuration currently lives at project root: ``regime_config.py``.
This shim allows optional imports such as:
    from modules.regime_config import REGIME_EVENT_THRESHOLD
"""

from __future__ import annotations

from regime_config import (
    REGIMES,
    REGIME_EVENT_THRESHOLD,
    REGIME_TP_SL,
    REGIME_MAX_BARS,
    REGIME_MODEL_CONFIGS,
    REGIME_EXTRA_FEATURES,
    REGIME_DRIFT_CONFIG,
    REGIME_PRED_THRESHOLD,
    EVENT_SCORE_WEIGHTS,
    EVENT_ZSCORE_WINDOW,
    EVENT_ZSCORE_MIN_PERIODS,
    print_regime_summary,
)

__all__ = [
    "REGIMES",
    "REGIME_EVENT_THRESHOLD",
    "REGIME_TP_SL",
    "REGIME_MAX_BARS",
    "REGIME_MODEL_CONFIGS",
    "REGIME_EXTRA_FEATURES",
    "REGIME_DRIFT_CONFIG",
    "REGIME_PRED_THRESHOLD",
    "EVENT_SCORE_WEIGHTS",
    "EVENT_ZSCORE_WINDOW",
    "EVENT_ZSCORE_MIN_PERIODS",
    "print_regime_summary",
]

