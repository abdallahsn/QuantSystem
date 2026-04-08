"""
preprocessing_v19.py - Shared preprocessing for QuantSystem V19 inference
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.feature_factory_v19 import V19FeatureFactory


class V19FeaturePreprocessor:
    """
    Loads the V19 feature schema and scaler params, then applies the same
    feature ordering and numeric normalization used during training.
    """

    def __init__(self,
                 models_dir: str,
                 schema_file: str = 'feature_schema_v19.json',
                 scaler_file: str = 'scaler_params.json'):
        self.factory = V19FeatureFactory(
            models_dir=models_dir,
            schema_file=schema_file,
            scaler_file=scaler_file,
        )
        self.models_dir = models_dir
        self.schema = self.factory.schema
        self.scaler_params = self.factory.scaler_params
        self.stat_features = self.factory.stat_features
        self.meta_features = self.factory.meta_features
        self.input_dim = self.factory.input_dim
        self.seq_len = self.factory.seq_len

    def transform_frame(self, df: pd.DataFrame, already_scaled: bool = False) -> pd.DataFrame:
        out = self.factory.prepare_frame(df, already_scaled=already_scaled, include_meta=False)
        return out[self.stat_features].copy()

    def transform_dict(self, features: dict, already_scaled: bool = False) -> pd.DataFrame:
        out = self.factory.prepare_row(features, already_scaled=already_scaled, include_meta=False)
        return out[self.stat_features].copy()

    def to_matrix(self, df: pd.DataFrame, already_scaled: bool = False) -> np.ndarray:
        arr_df = self.transform_frame(df, already_scaled=already_scaled)
        return arr_df.values.astype(np.float32)
