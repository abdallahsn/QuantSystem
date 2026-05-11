"""
Compatibility shim for Section 10 live prediction pipeline.

Canonical implementation currently lives at project root: ``live_predictor.py``.
This module keeps documented imports stable:
    from modules.live_predictor import run_bar_pipeline
"""

from __future__ import annotations

from live_predictor import (
    BASE_FEATURES,
    get_regime_features,
    predict_live,
    run_bar_pipeline,
    evaluate_on_test_set,
)

__all__ = [
    "BASE_FEATURES",
    "get_regime_features",
    "predict_live",
    "run_bar_pipeline",
    "evaluate_on_test_set",
]

