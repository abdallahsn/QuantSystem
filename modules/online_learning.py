"""
Compatibility shim for Section 8 online learning module.

Canonical implementation currently lives at project root: ``online_learning.py``.
This module keeps the documented import path stable:
    from modules.online_learning import RegimeConditionalEnsemble
"""

from __future__ import annotations

from online_learning import DriftDetector, SlidingWindowTrainer, RegimeConditionalEnsemble

__all__ = [
    "DriftDetector",
    "SlidingWindowTrainer",
    "RegimeConditionalEnsemble",
]

