"""
raw_replay_v19.py - Raw market replay dataset builder for QuantSystem V19
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from modules.feature_artifact_v19 import (
    FINAL_FEATURE_DIR,
    load_feature_artifact,
    parquet_shard_paths,
    read_table,
    write_parquet_shards,
    write_table,
)
from prepare_training_data import (
    DEFAULT_V19_DIRECTION_THRESHOLD_TICKS,
    DEFAULT_V19_TP_MULT,
    _build_refinery_split_context,
    run_refinery,
)


def read_market_data(path: str) -> pd.DataFrame:
    return read_table(path)


def normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'ts_event' not in out.columns and 'ts_recv' in out.columns:
        out['ts_event'] = out['ts_recv']
    if 'ts_event' not in out.columns:
        raise ValueError('❌ Replay dataset builder requires ts_event or ts_recv in source data')
    out['ts_event'] = pd.to_datetime(out.get('ts_event'), utc=True, errors='coerce').dt.tz_localize(None)
    invalid_mask = out['ts_event'].isna()
    if bool(invalid_mask.any()):
        sample = invalid_mask[invalid_mask].index[:5].tolist()
        raise ValueError(
            "❌ Replay source contains invalid timestamps: "
            f"rows={int(invalid_mask.sum())} sample_indices={sample}"
        )
    return out.sort_values('ts_event').reset_index(drop=True)


def _coerce_optional_timestamp(value):
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def _rewrite_trimmed_feature_artifact(
    output_dir: str,
    *,
    start_ts=None,
    end_ts=None,
) -> dict:
    score_start = _coerce_optional_timestamp(start_ts)
    score_end = _coerce_optional_timestamp(end_ts)
    df = load_feature_artifact(output_dir)
    if len(df) == 0:
        raise RuntimeError('❌ Cannot trim an empty replay artifact')
    if 'ts_event' not in df.columns:
        raise ValueError('❌ Replay artifact is missing ts_event and cannot be trimmed safely')

    ts = pd.to_datetime(df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    invalid_mask = ts.isna()
    if bool(invalid_mask.any()):
        sample = invalid_mask[invalid_mask].index[:5].tolist()
        raise ValueError(
            "❌ Replay artifact contains invalid ts_event values after refinery: "
            f"rows={int(invalid_mask.sum())} sample_indices={sample}"
        )

    keep_mask = np.ones(len(df), dtype=bool)
    if score_start is not None:
        keep_mask &= (ts >= score_start).to_numpy(dtype=bool)
    if score_end is not None:
        keep_mask &= (ts < score_end).to_numpy(dtype=bool)
    trimmed = df.loc[keep_mask].reset_index(drop=True)
    if trimmed.empty:
        raise RuntimeError(
            '❌ Replay artifact trimming removed every row. '
            f'start_ts={score_start} end_ts={score_end}'
        )

    final_dir = os.path.join(output_dir, FINAL_FEATURE_DIR)
    for path in parquet_shard_paths(final_dir):
        os.remove(path)
    shard_records = write_parquet_shards(
        trimmed,
        final_dir,
        stem='features',
        rows_per_shard=250_000,
    )
    with open(os.path.join(output_dir, 'final_feature_shards.json'), 'w') as f:
        json.dump(shard_records, f, indent=2)

    split_ctx = _build_refinery_split_context(trimmed, train_frac=0.80)
    split_time = split_ctx['split_time']
    split_meta = {
        'split_idx': int(split_ctx['split_idx']),
        'split_time': None if pd.isna(split_time) else str(split_time),
        'train_rows': int(np.sum(split_ctx['train_row_ok'])),
        'holdout_rows': int(np.sum(split_ctx['holdout_row_ok'])),
        'purged_rows': int(np.sum(split_ctx['purged_row_ok'])),
    }
    with open(os.path.join(output_dir, 'refinery_split.json'), 'w') as f:
        json.dump(split_meta, f, indent=2)

    manifest_path = os.path.join(output_dir, 'artifact_manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        metrics = manifest.get('metrics', {}) or {}
        metrics['rows'] = int(len(trimmed))
        metrics['event_rows'] = int(
            pd.to_numeric(trimmed.get('event_flag', 0), errors='coerce')
            .fillna(0)
            .astype(np.int8)
            .sum()
        ) if len(trimmed) else 0
        manifest['metrics'] = metrics

        extra = manifest.get('extra', {}) or {}
        extra['rows'] = int(len(trimmed))
        extra['ts_min'] = str(ts.loc[keep_mask].min())
        extra['ts_max'] = str(ts.loc[keep_mask].max())
        extra['split_meta'] = split_meta
        extra['final_feature_shards'] = shard_records
        extra['score_window_trim'] = {
            'applied': True,
            'rows_before': int(len(df)),
            'rows_after': int(len(trimmed)),
            'score_start_ts': None if score_start is None else str(score_start),
            'score_end_ts': None if score_end is None else str(score_end),
        }
        manifest['extra'] = extra
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    return {
        'trimmed_rows_before': int(len(df)),
        'trimmed_rows_after': int(len(trimmed)),
        'split_meta': split_meta,
    }


def _slice_timerange_with_row_context(
    df: pd.DataFrame,
    *,
    start_ts=None,
    end_ts=None,
    warmup_rows: int = 0,
    tail_rows: int = 0,
) -> tuple[pd.DataFrame, dict]:
    out = normalize_ts(df)
    meta = {
        'requested_start_ts': None if start_ts is None else str(pd.Timestamp(start_ts)),
        'requested_end_ts': None if end_ts is None else str(pd.Timestamp(end_ts)),
        'warmup_rows_requested': int(max(warmup_rows, 0)),
        'tail_rows_requested': int(max(tail_rows, 0)),
        'scoring_rows_est': int(len(out)),
        'context_rows': int(len(out)),
        'warmup_rows_applied': 0,
        'tail_rows_applied': 0,
    }
    if out.empty:
        return out, meta

    ts = out['ts_event']
    start = None
    end = None
    if start_ts is not None:
        start = pd.Timestamp(start_ts)
        start = start.tz_localize(None) if start.tzinfo is None else start.tz_convert(None)
    if end_ts is not None:
        end = pd.Timestamp(end_ts)
        end = end.tz_localize(None) if end.tzinfo is None else end.tz_convert(None)

    start_pos = 0 if start is None else int(np.searchsorted(ts.values.astype('datetime64[ns]'), start.to_datetime64(), side='left'))
    end_pos = len(out) if end is None else int(np.searchsorted(ts.values.astype('datetime64[ns]'), end.to_datetime64(), side='left'))
    start_pos = min(max(start_pos, 0), len(out))
    end_pos = min(max(end_pos, start_pos), len(out))

    warmup_rows = int(max(warmup_rows, 0))
    tail_rows = int(max(tail_rows, 0))
    ctx_start = max(0, start_pos - warmup_rows)
    ctx_end = min(len(out), end_pos + tail_rows)
    sliced = out.iloc[ctx_start:ctx_end].reset_index(drop=True)

    meta.update({
        'scoring_rows_est': int(max(end_pos - start_pos, 0)),
        'context_rows': int(len(sliced)),
        'warmup_rows_applied': int(start_pos - ctx_start),
        'tail_rows_applied': int(ctx_end - end_pos),
    })
    return sliced, meta


def filter_timerange(df: pd.DataFrame, start_ts=None, end_ts=None) -> pd.DataFrame:
    out, _ = _slice_timerange_with_row_context(
        df,
        start_ts=start_ts,
        end_ts=end_ts,
        warmup_rows=0,
        tail_rows=0,
    )
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
    direction_threshold_ticks: float = DEFAULT_V19_DIRECTION_THRESHOLD_TICKS,
    causal_threshold_mode: str = 'expanding',
    tp_mult: float = DEFAULT_V19_TP_MULT,
    sl_mult: float = 1.0,
    kalman_slope_threshold: float = 0.05,
    trend_strength_min: float = 0.05,
    regime_mode: str = 'rules',
    regime_stride: int = 50,
    regime_window: int = 50,
    regime_progress_every: int = 25_000,
    lob_event_sample: int = 100000,
    merge_tolerance_ms: int = 100,
    external_scaler_path: str | None = None,
    fit_aux_models: bool = True,
    warmup_rows: int = 0,
    tail_rows: int = 0,
    trim_to_score_window: bool = True,
) -> dict:
    """Build replay artifact.

    Warmup rows are prepended before refinery only for causal feature context.
    ``trim_to_score_window=True`` (default) trims the finalized feature parquet
    shards back to ``[start_ts, end_ts)`` so warmup/tail rows never leak into train.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, 'raw_slice')
    mbo_df, mbo_window = _slice_timerange_with_row_context(
        read_market_data(mbo_path),
        start_ts=start_ts,
        end_ts=end_ts,
        warmup_rows=warmup_rows,
        tail_rows=tail_rows,
    )
    if mbp_path:
        mbp_df, mbp_window = _slice_timerange_with_row_context(
            read_market_data(mbp_path),
            start_ts=start_ts,
            end_ts=end_ts,
            warmup_rows=warmup_rows,
            tail_rows=tail_rows,
        )
    else:
        mbp_df, mbp_window = pd.DataFrame(), {
            'requested_start_ts': None if start_ts is None else str(pd.Timestamp(start_ts)),
            'requested_end_ts': None if end_ts is None else str(pd.Timestamp(end_ts)),
            'warmup_rows_requested': int(max(warmup_rows, 0)),
            'tail_rows_requested': int(max(tail_rows, 0)),
            'scoring_rows_est': 0,
            'context_rows': 0,
            'warmup_rows_applied': 0,
            'tail_rows_applied': 0,
        }

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

    trim_meta = None
    if trim_to_score_window:
        if start_ts is None and end_ts is None:
            print(
                "  ℹ️ trim_to_score_window=True but start_ts/end_ts unset — "
                "no time-based trim applied (full refinery output kept)."
            )
        else:
            trim_meta = _rewrite_trimmed_feature_artifact(
                output_dir,
                start_ts=start_ts,
                end_ts=end_ts,
            )
    elif int(max(warmup_rows, 0)) > 0 or int(max(tail_rows, 0)) > 0:
        print(
            "  ⚠️ warmup_rows/tail_rows were used but trim_to_score_window=False — "
            "context rows may remain in training shards. Prefer trim_to_score_window=True."
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
        'window': {
            'score_start_ts': None if start_ts is None else str(pd.Timestamp(start_ts)),
            'score_end_ts': None if end_ts is None else str(pd.Timestamp(end_ts)),
            'warmup_rows': int(max(warmup_rows, 0)),
            'tail_rows': int(max(tail_rows, 0)),
            'trim_to_score_window': bool(trim_to_score_window),
            'trim_meta': trim_meta,
            'mbo': mbo_window,
            'mbp': mbp_window,
        },
    }
