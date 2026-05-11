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
BASE_MODEL_META_GROUPS = (
    ('catboost', ('cb_prob_long', 'cb_prob_short')),
    ('xgboost', ('xgb_prob_long', 'xgb_prob_short')),
)
REGIME_META_FEATURES = (
    'cluster_0',
    'cluster_1',
    'cluster_2',
    'cluster_3',
    'regime_lowliq_score',
    'regime_trend_score',
    'regime_volatile_score',
)
LEGACY_META_FEATURES = tuple(BASE_MODEL_META_GROUPS[0][1]) + REGIME_META_FEATURES
FULL_META_FEATURES = tuple(
    [*BASE_MODEL_META_GROUPS[0][1], *BASE_MODEL_META_GROUPS[1][1], *REGIME_META_FEATURES]
)
SUPPORTED_META_FEATURES = {
    LEGACY_META_FEATURES: 'legacy_catboost_only',
    FULL_META_FEATURES: 'catboost_xgboost',
}
EXPECTED_META_FEATURES = len(FULL_META_FEATURES)
ROBUST_IQR_MIN = 1e-2


DEFAULT_TIMESTAMP_COLS = ('ts_event', 'label_end_ts')
DEFAULT_PASSTHROUGH_COLS = [
    'ts_event',
    'label_end_ts',
    'lob_tensor_id',
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
    'bias_label_raw',
    'effective_threshold_ticks',
    'effective_tp_long_ticks',
    'effective_tp_short_ticks',
    'effective_sl_ticks',
    'meta_trade_side',
    'meta_label',
    'meta_label_active',
    'meta_outcome_ticks',
    'path_outcome',
    'adverse_path_flag',
    'bias_label_detail',
    'neutral_reason',
    'timeout_move_exceeded_band',
    'label_dynamic_threshold',
    'effective_horizon',
    'soft_label',
    'label_confidence',
    'soft_label_long',
    'soft_label_short',
    'soft_sample_weight',
    'soft_label_confidence',
    'soft_label_entropy',
    'soft_label_scenarios',
    'mc_sample_weight',
    'label_stability',
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
    'bias_label_raw',
    'meta_trade_side',
    'meta_label',
    'meta_label_active',
    'path_outcome',
    'adverse_path_flag',
    'bias_label_detail',
    'neutral_reason',
    'timeout_move_exceeded_band',
    'effective_horizon',
    'soft_label_scenarios',
    'lob_tensor_id',
}
FORBIDDEN_STAT_FEATURES = {
    'forward_return',
    'label_end_ts',
    'ts_event',
    'bias_label',
    'conf_label',
    'signal_quality',
    'regime_label',
    'regime_cluster',
    'event_flag',
    'train_event_flag',
    'event_score',
    'event_trigger_count',
    'is_expansion',
    'label_horizon_steps',
    'bias_label_raw',
    'effective_threshold_ticks',
    'effective_tp_long_ticks',
    'effective_tp_short_ticks',
    'effective_sl_ticks',
    'meta_trade_side',
    'meta_label',
    'meta_label_active',
    'meta_outcome_ticks',
    'path_outcome',
    'adverse_path_flag',
    'bias_label_detail',
    'neutral_reason',
    'timeout_move_exceeded_band',
    'soft_label',
    'label_confidence',
    'soft_label_long',
    'soft_label_short',
    'soft_sample_weight',
    'soft_label_confidence',
    'soft_label_entropy',
    'soft_label_scenarios',
    'mc_sample_weight',
    'label_stability',
}


def resolve_meta_feature_names(
    *,
    include_xgboost: bool = True,
    meta_dim: int | None = None,
) -> list[str]:
    if meta_dim is not None:
        if int(meta_dim) == len(LEGACY_META_FEATURES):
            return list(LEGACY_META_FEATURES)
        if int(meta_dim) == len(FULL_META_FEATURES):
            return list(FULL_META_FEATURES)
        raise ValueError(f'Unsupported meta feature dimension: {meta_dim}')
    return list(FULL_META_FEATURES if include_xgboost else LEGACY_META_FEATURES)


def infer_meta_feature_layout(meta_features: Iterable[str]) -> dict:
    names = [str(col) for col in meta_features]
    if tuple(names) in SUPPORTED_META_FEATURES:
        base_models = []
        offset = 0
        for model_name, cols in BASE_MODEL_META_GROUPS:
            cols = list(cols)
            if names[offset:offset + len(cols)] != cols:
                continue
            base_models.append(
                {
                    'name': model_name,
                    'prob_cols': cols,
                    'start': int(offset),
                    'end': int(offset + len(cols)),
                }
            )
            offset += len(cols)
        return {
            'surface': SUPPORTED_META_FEATURES[tuple(names)],
            'base_models': base_models,
            'base_prob_dim': int(offset),
            'regime_meta_cols': list(REGIME_META_FEATURES),
        }

    if len(names) < len(REGIME_META_FEATURES):
        raise ValueError(
            f'Unsupported meta feature surface: expected at least {len(REGIME_META_FEATURES)} columns, got {len(names)}'
        )

    regime_cols = names[-len(REGIME_META_FEATURES):]
    if regime_cols != list(REGIME_META_FEATURES):
        raise ValueError(
            'Unsupported meta feature surface: invalid regime meta suffix '
            f'{regime_cols}, expected {list(REGIME_META_FEATURES)}'
        )

    base_block = names[:-len(REGIME_META_FEATURES)]
    offset = 0
    base_models = []
    for model_name, cols in BASE_MODEL_META_GROUPS:
        cols = list(cols)
        if base_block[offset:offset + len(cols)] != cols:
            break
        base_models.append(
            {
                'name': model_name,
                'prob_cols': cols,
                'start': int(offset),
                'end': int(offset + len(cols)),
            }
        )
        offset += len(cols)

    if offset != len(base_block) or not base_models:
        supported = [list(surface) for surface in SUPPORTED_META_FEATURES]
        raise ValueError(
            'Unsupported meta feature surface. '
            f'Got {names}. Supported surfaces: {supported}'
        )

    return {
        'surface': 'custom_supported_order',
        'base_models': base_models,
        'base_prob_dim': int(offset),
        'regime_meta_cols': list(REGIME_META_FEATURES),
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


def _clip_numeric_series(
    series: pd.Series,
    clip_range: tuple[float, float] | None,
) -> pd.Series:
    if clip_range is None:
        return series
    clip_low, clip_high = clip_range
    return series.clip(float(clip_low), float(clip_high))


def fit_numeric_scaler_param(
    series: pd.Series,
    *,
    robust_iqr_min: float = ROBUST_IQR_MIN,
    clip_rate_threshold: float | None = None,
) -> dict:
    s = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(np.float32)
    median_ = float(s.median())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    smin = float(s.min())
    smax = float(s.max())
    rng = smax - smin

    if np.isfinite(iqr) and iqr >= float(robust_iqr_min):
        payload = {
            'type': 'robust',
            'median': median_,
            'iqr': float(iqr),
            'min_robust_iqr': float(robust_iqr_min),
        }
        if clip_rate_threshold is not None:
            denom = max(float(iqr), float(robust_iqr_min))
            robust_scaled = ((s - median_) / denom).astype(np.float32)
            clip_rate = float((np.abs(robust_scaled) >= 9.5).mean()) if len(robust_scaled) else 0.0
            if clip_rate > float(clip_rate_threshold) and rng > 1e-8:
                return {
                    'type': 'minmax',
                    'min': smin,
                    'max': smax,
                    'fallback_from': 'robust',
                    'observed_iqr': float(iqr),
                    'min_robust_iqr': float(robust_iqr_min),
                    'robust_clip_rate': clip_rate,
                }
            payload['robust_clip_rate'] = clip_rate
        return payload

    if rng > 1e-8:
        return {
            'type': 'minmax',
            'min': smin,
            'max': smax,
            'fallback_from': 'low_iqr',
            'observed_iqr': float(iqr),
            'min_robust_iqr': float(robust_iqr_min),
        }

    return {'type': 'zero'}


def apply_scaler_params_to_frame(
    df: pd.DataFrame,
    scaler_params: dict,
    clip_range: tuple[float, float] | None = (-10.0, 10.0),
) -> pd.DataFrame:
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
            denom = max(
                float(p.get('iqr', 0.0)),
                float(p.get('min_robust_iqr', ROBUST_IQR_MIN)),
            )
            out[col] = _clip_numeric_series(
                (s - p.get('median', 0.0)) / denom,
                clip_range,
            )
        elif typ == 'minmax':
            rng = max(float(p.get('max', 0.0)) - float(p.get('min', 0.0)), 1e-8)
            out[col] = _clip_numeric_series(
                (s - float(p.get('min', 0.0))) / rng * 2 - 1,
                clip_range,
            )
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
        infer_meta_feature_layout(self.meta_features)
        leaked = sorted(set(self.stat_features) & FORBIDDEN_STAT_FEATURES)
        if leaked:
            raise ValueError(
                f'Stat feature schema contains forbidden leakage-prone columns: {leaked}'
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
