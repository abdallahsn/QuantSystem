"""Base model interfaces for QuantSystem v20."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelFitResult:
    model_id: str
    artifact_paths: tuple[str, ...]
    metrics: dict


class TabularModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight=None) -> ModelFitResult:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...


class SequenceModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight=None) -> ModelFitResult:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...
