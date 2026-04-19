"""
feature_factory_v19.py - Canonical feature assembly for QuantSystem V19
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import pandas as pd

EXPECTED_SCHEMA_VERSION = 'v19-event-binary'
EXPECTED_META_FEATURES = 6


DEFAULT_TIMESTAMP_COLS = ('ts_event', 'label_end_ts')
DEFAULT_PASSTHROUGH_COLS = [
    'ts_event',
    'label_end_ts',
    'price',
    'size',
    'bias_label',
    'setup_label',
    'conf_label',
    'signal_quality',
    'regime_label',
    'regime_cluster',
    'event_flag',
    'train_event_flag',
    'event_score',
    'event_trigger_count',
    'is_expansion',
    'liq_score',
    'forward_return',
    'label_horizon_steps',
]
DEFAULT_INT_COLS = {
    'bias_label',
    'setup_label',
    'signal_quality',
    'regime_label',
    'regime_cluster',
    'event_flag',
    'train_event_flag',
    'event_trigger_count',
    'is_expansion',
}


def _load_json(path: str, required: bool = True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f'Missing artifact: {path}')
        return None
    with open(path) as f:
        return json.load(f)


def normalize_timestamp_columns(
    df: pd.DataFrame,
    timestamp_cols: Iterable[str] = DEFAULT_TIMESTAMP_COLS,
) -> pd.DataFrame:
    out = df.copy()
    for col in timestamp_cols:
        if col not in out.columns:
            out[col] = pd.NaT
        out[col] = pd.to_datetime(out[col], utc=True, errors='coerce').dt.tz_localize(None)
    return out


def apply_scaler_params_to_frame(df: pd.DataFrame, scaler_params: dict) -> pd.DataFrame:
    out = df.copy()
    for col, p in scaler_params.items():
        if col not in out.columns:
            out[col] = 0.0
            continue
        s = pd.to_numeric(out[col], errors='coerce').fillna(0.0).astype(np.float32)
        typ = p.get('type', 'zero')
        if typ == 'binary':
            out[col] = s
        elif typ == 'robust':
            out[col] = ((s - p.get('median', 0.0)) / max(p.get('iqr', 0.0), 1e-8)).clip(-10, 10)
        elif typ == 'minmax':
            rng = max(float(p.get('max', 0.0)) - float(p.get('min', 0.0)), 1e-8)
            out[col] = ((s - float(p.get('min', 0.0))) / rng * 2 - 1).clip(-10, 10)
        else:
            out[col] = 0.0
    return out


def prepare_feature_frame(
    df: pd.DataFrame,
    stat_features: Iterable[str],
    scaler_params: dict | None = None,
    already_scaled: bool = False,
    passthrough_cols: Iterable[str] | None = None,
    timestamp_cols: Iterable[str] = DEFAULT_TIMESTAMP_COLS,
) -> pd.DataFrame:
    passthrough_cols = list(passthrough_cols or [])
    stat_features = list(stat_features)

    out = normalize_timestamp_columns(df, timestamp_cols=timestamp_cols)

    for col in passthrough_cols + stat_features:
        if col not in out.columns:
            out[col] = pd.NaT if col in timestamp_cols else 0.0

    numeric_cols = [c for c in stat_features if c not in timestamp_cols]
    numeric_meta = [c for c in passthrough_cols if c not in timestamp_cols]
    for col in numeric_cols + numeric_meta:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
        if col in DEFAULT_INT_COLS:
            out[col] = out[col].astype(np.int32)
        else:
            out[col] = out[col].astype(np.float32)

    if not already_scaled and scaler_params:
        out = apply_scaler_params_to_frame(out, scaler_params)

    final_cols = []
    for col in passthrough_cols + stat_features:
        if col not in final_cols:
            final_cols.append(col)
    return out[final_cols].copy()


def prepare_feature_row(
    row: dict,
    stat_features: Iterable[str],
    scaler_params: dict | None = None,
    already_scaled: bool = False,
    passthrough_cols: Iterable[str] | None = None,
    timestamp_cols: Iterable[str] = DEFAULT_TIMESTAMP_COLS,
    ts=None,
    extra: dict | None = None,
) -> pd.DataFrame:
    payload = dict(row or {})
    if ts is not None and 'ts_event' not in payload:
        payload['ts_event'] = ts
    if extra:
        payload.update(extra)
    return prepare_feature_frame(
        pd.DataFrame([payload]),
        stat_features=stat_features,
        scaler_params=scaler_params,
        already_scaled=already_scaled,
        passthrough_cols=passthrough_cols,
        timestamp_cols=timestamp_cols,
    )


class V19FeatureFactory:
    def __init__(
        self,
        models_dir: str,
        schema_file: str = 'feature_schema_v19.json',
        scaler_file: str = 'scaler_params.json',
    ):
        self.models_dir = models_dir
        self.schema_path = os.path.join(models_dir, schema_file)
        self.scaler_path = os.path.join(models_dir, scaler_file)

        self.schema = _load_json(self.schema_path)
        self.scaler_params = _load_json(self.scaler_path, required=False) or {}

        self.stat_features = list(self.schema.get('stat_features', []))
        self.meta_features = list(self.schema.get('meta_features', []))
        self.visual_features = list(self.schema.get('visual_features', []))
        self.passthrough_cols = list(self.schema.get('passthrough_cols', DEFAULT_PASSTHROUGH_COLS))
        self.timestamp_cols = tuple(self.schema.get('timestamp_cols', list(DEFAULT_TIMESTAMP_COLS)))
        self.input_dim = int(self.schema.get('input_dim', len(self.stat_features) + len(self.meta_features)))
        self.seq_len = int(self.schema.get('seq_len', 50))
        self._validate_schema()

    def _validate_schema(self) -> None:
        version = str(self.schema.get('version', '')).strip()
        if version != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f'Unsupported feature schema version: {version or "<missing>"}. '
                f'Expected {EXPECTED_SCHEMA_VERSION}.'
            )
        if len(self.meta_features) != EXPECTED_META_FEATURES:
            raise ValueError(
                f'Unsupported meta feature surface: expected {EXPECTED_META_FEATURES} dims, got {len(self.meta_features)}'
            )

    def prepare_frame(self, df: pd.DataFrame, already_scaled: bool = False, include_meta: bool = True) -> pd.DataFrame:
        return prepare_feature_frame(
            df,
            stat_features=self.stat_features,
            scaler_params=self.scaler_params,
            already_scaled=already_scaled,
            passthrough_cols=self.passthrough_cols if include_meta else [],
            timestamp_cols=self.timestamp_cols,
        )

    def prepare_row(
        self,
        row: dict,
        already_scaled: bool = False,
        include_meta: bool = True,
        ts=None,
        extra: dict | None = None,
    ) -> pd.DataFrame:
        return prepare_feature_row(
            row,
            stat_features=self.stat_features,
            scaler_params=self.scaler_params,
            already_scaled=already_scaled,
            passthrough_cols=self.passthrough_cols if include_meta else [],
            timestamp_cols=self.timestamp_cols,
            ts=ts,
            extra=extra,
        )

    def stat_matrix(self, df: pd.DataFrame, already_scaled: bool = False) -> np.ndarray:
        stat_df = self.prepare_frame(df, already_scaled=already_scaled, include_meta=False)
        return stat_df[self.stat_features].values.astype(np.float32)

    def zero_visual_embeddings(self, n_rows: int) -> np.ndarray:
        return np.zeros((int(n_rows), len(self.visual_features)), dtype=np.float32)

    def prepare_visual_embeddings(self, visual_embedding: np.ndarray | None, n_rows: int = 1) -> np.ndarray:
        if len(self.visual_features) == 0:
            return np.zeros((n_rows, 0), dtype=np.float32)
        if visual_embedding is None:
            return self.zero_visual_embeddings(n_rows)

        arr = np.asarray(visual_embedding, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] == 1 and n_rows > 1:
            arr = np.repeat(arr, n_rows, axis=0)
        if arr.shape[1] < len(self.visual_features):
            pad = np.zeros((arr.shape[0], len(self.visual_features) - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        return arr[:n_rows, :len(self.visual_features)]
