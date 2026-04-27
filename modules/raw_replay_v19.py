"""
raw_replay_v19.py - Raw market replay dataset builder for QuantSystem V19
"""

from __future__ import annotations

import os

import pandas as pd

from modules.feature_artifact_v19 import read_table, write_table
from prepare_training_data import (
    DEFAULT_V22_DIRECTION_THRESHOLD_TICKS,
    DEFAULT_V22_TP_MULT,
    run_refinery,
)


def read_market_data(path: str) -> pd.DataFrame:
    return read_table(path)


def normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'ts_event' not in out.columns and 'ts_recv' in out.columns:
        out['ts_event'] = out['ts_recv']
    out['ts_event'] = pd.to_datetime(out.get('ts_event'), utc=True, errors='coerce').dt.tz_localize(None)
    return out.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)


def filter_timerange(df: pd.DataFrame, start_ts=None, end_ts=None) -> pd.DataFrame:
    out = normalize_ts(df)
    if start_ts is not None:
        start_ts = pd.Timestamp(start_ts).tz_localize(None) if pd.Timestamp(start_ts).tzinfo is None else pd.Timestamp(start_ts).tz_convert(None)
        out = out[out['ts_event'] >= start_ts]
    if end_ts is not None:
        end_ts = pd.Timestamp(end_ts).tz_localize(None) if pd.Timestamp(end_ts).tzinfo is None else pd.Timestamp(end_ts).tz_convert(None)
        out = out[out['ts_event'] < end_ts]
    return out.reset_index(drop=True)


def write_market_slice(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return write_table(df, path)


def build_replay_dataset(
    mbo_path: str,
    mbp_path: str,
    output_dir: str,
    start_ts=None,
    end_ts=None,
    label_mode: str = 'v19',
    chunksize: int = 0,
    n_workers: int | None = None,
    target_bars: int = 500,
    label_horizon: int = 150,
    event_roll_window: int = 50,
    direction_threshold_ticks: float = DEFAULT_V22_DIRECTION_THRESHOLD_TICKS,
    causal_threshold_mode: str = 'expanding',
    tp_mult: float = DEFAULT_V22_TP_MULT,
    sl_mult: float = 1.0,
    kalman_slope_threshold: float = 0.05,
    trend_strength_min: float = 0.05,
    regime_mode: str = 'rules',
    regime_stride: int = 50,
    regime_window: int = 50,
    regime_progress_every: int = 25_000,
    lob_event_sample: int = 100000,
    merge_tolerance_ms: int = 500,
    external_scaler_path: str | None = None,
    fit_aux_models: bool = True,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, 'raw_slice')
    mbo_df = filter_timerange(read_market_data(mbo_path), start_ts=start_ts, end_ts=end_ts)
    mbp_df = filter_timerange(read_market_data(mbp_path), start_ts=start_ts, end_ts=end_ts) if mbp_path else pd.DataFrame()

    mbo_slice = write_market_slice(mbo_df, os.path.join(raw_dir, 'mbo_slice.parquet'))
    if len(mbp_df):
        mbp_slice = write_market_slice(mbp_df, os.path.join(raw_dir, 'mbp_slice.parquet'))
    else:
        mbp_slice = write_market_slice(pd.DataFrame(columns=['ts_event']), os.path.join(raw_dir, 'mbp_slice.parquet'))

    run_refinery(
        mbo_path=mbo_slice,
        mbp_path=mbp_slice,
        symbol='',
        output_dir=output_dir,
        chunksize=None if not chunksize else int(chunksize),
        label_mode=label_mode,
        n_workers=n_workers,
        target_bars=target_bars,
        label_horizon=label_horizon,
        event_roll_window=event_roll_window,
        direction_threshold_ticks=direction_threshold_ticks,
        causal_threshold_mode=causal_threshold_mode,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        kalman_slope_threshold=kalman_slope_threshold,
        trend_strength_min=trend_strength_min,
        regime_mode=regime_mode,
        regime_stride=regime_stride,
        regime_window=regime_window,
        regime_progress_every=regime_progress_every,
        lob_event_sample=lob_event_sample,
        merge_tolerance_ms=merge_tolerance_ms,
        external_scaler_path=external_scaler_path,
        fit_aux_models=fit_aux_models,
    )

    return {
        'output_dir': output_dir,
        'mbo_slice': mbo_slice,
        'mbp_slice': mbp_slice,
        'data': output_dir,
        'csv': output_dir,
        'lob': os.path.join(output_dir, 'lob_tensors.npy'),
        'lob_ts': os.path.join(output_dir, 'lob_tensor_timestamps.npy'),
        'scaler': os.path.join(output_dir, 'scaler_params.json'),
    }
