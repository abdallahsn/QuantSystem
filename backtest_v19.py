"""
backtest_v19.py - Causal replay backtester for QuantSystem V19
================================================================
This backtester replays rows one-by-one through V19PredictionEngine using the
same step-by-step path as live inference. It evaluates predictions against the
causal V19 labels and computes trading metrics by replaying the future price
path instead of settling directly on the stored oracle forward_return label.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.slippage_model import confidence_bet_size
from predict_v19 import V19PredictionEngine

try:
    from modules.html_reporter import generate_backtest_report
    HTML_REPORT_AVAILABLE = True
except ImportError:
    HTML_REPORT_AVAILABLE = False


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _series_or_default(df: pd.DataFrame, col: str, default, dtype=None) -> pd.Series:
    if col in df.columns:
        series = df[col]
    else:
        series = pd.Series([default] * len(df), index=df.index)
    if dtype is not None:
        series = pd.to_numeric(series, errors='coerce').fillna(default).astype(dtype)
    return series


def _simulate_trade_path(
    entry_idx: int,
    direction: str,
    prices: np.ndarray,
    horizons: np.ndarray,
    micro_atr: np.ndarray,
    tick_size: float,
    direction_threshold_ticks: float = 1.0,
    tp_mult: float = 1.2,
    sl_mult: float = 1.0,
    max_horizon_steps: int | None = None,
    # ── Wall data (اللايف بيستخدمها — الباك تست كان يتجاهلها) ──
    row_data: dict | None = None,
) -> dict | None:
    """
    FIX: يستخدم الآن DynamicTargetManager نفس اللايف.
    TP/SL محسوبان من جدران السيولة الحقيقية
    بدل ATR × multiplier الثابت.
    """
    if direction not in ('LONG', 'SHORT'):
        return None
    if entry_idx < 0 or entry_idx >= len(prices):
        return None

    entry_price = float(prices[entry_idx])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    horizon_steps = int(horizons[entry_idx]) if entry_idx < len(horizons) else 0
    if max_horizon_steps is not None and int(max_horizon_steps) > 0:
        horizon_steps = min(horizon_steps, int(max_horizon_steps)) if horizon_steps > 0 else int(max_horizon_steps)
    if horizon_steps <= 0:
        return None

    exit_cap_idx = min(entry_idx + horizon_steps, len(prices) - 1)
    if exit_cap_idx <= entry_idx:
        return None

    atr_now = float(micro_atr[entry_idx]) if entry_idx < len(micro_atr) else 0.0
    min_move = max(float(direction_threshold_ticks) * float(tick_size),
                   0.5 * max(atr_now, 0.0), float(tick_size))

    # ══════════════════════════════════════════════════════════════════
    # FIX: استخدم DynamicTargetManager مع بيانات الجدران الحقيقية
    # نفس المسار اللي بيسلكه اللايف
    # ══════════════════════════════════════════════════════════════════
    tp_level = sl_level = None

    if row_data is not None:
        try:
            from modules.dynamic_target import DynamicTargetManager
            dtm = DynamicTargetManager()

            # بناء scan_result من الـ CSV مباشرة
            scan_result = {
                'bid_wall_strength':  float(row_data.get('bid_wall_strength', 0.5) or 0.5),
                'ask_wall_strength':  float(row_data.get('ask_wall_strength', 0.5) or 0.5),
                'dist_to_bid_wall':   float(row_data.get('dist_to_bid_wall',  min_move * 1.5) or min_move * 1.5),
                'dist_to_ask_wall':   float(row_data.get('dist_to_ask_wall',  min_move * 1.5) or min_move * 1.5),
                'bid_wall_size_raw':  float(row_data.get('gap_size', 1.0) or 1.0),
                'ask_wall_size_raw':  float(row_data.get('gap_size', 1.0) or 1.0),
            }

            # بناء context
            remaining_fuel = max(
                float(row_data.get('micro_atr', min_move * 10) or min_move * 10) * 80,
                min_move * 20,
            )
            context = {
                'tick_size':      tick_size,
                'remaining_fuel': remaining_fuel,
                'adr_pips':       remaining_fuel / tick_size,
            }

            signal = {
                'bias':      direction,
                'price':     entry_price,
                'cvd_delta': float(row_data.get('cvd', 0.0) or 0.0),
            }

            levels = {
                'long_wall_size':  scan_result['bid_wall_size_raw'],
                'short_wall_size': scan_result['ask_wall_size_raw'],
            }

            trade = dtm.open_trade(signal, levels, scan_result, context)
            tp_level = trade.tp1
            sl_level = trade.sl

        except Exception as _e:
            # fallback للـ ATR إذا فشل DynamicTargetManager
            tp_level = None

    # Fallback: ATR-based (إذا مفيش wall data)
    if tp_level is None or sl_level is None:
        tp_distance = max(float(tp_mult) * min_move, float(tick_size))
        sl_distance = max(float(sl_mult) * min_move, float(tick_size))
        if direction == 'LONG':
            tp_level = entry_price + tp_distance
            sl_level = entry_price - sl_distance
        else:
            tp_level = entry_price - tp_distance
            sl_level = entry_price + sl_distance

    # ── Replay المسار الزمني ───────────────────────────────────────────
    future_prices = np.asarray(prices[entry_idx + 1:exit_cap_idx + 1], dtype=np.float64)
    if future_prices.size == 0:
        return None

    exit_idx = exit_cap_idx
    exit_reason = 'horizon'
    for offset, future_price in enumerate(future_prices, start=1):
        if direction == 'LONG':
            if future_price >= tp_level:
                exit_idx = entry_idx + offset; exit_reason = 'tp'; break
            if future_price <= sl_level:
                exit_idx = entry_idx + offset; exit_reason = 'sl'; break
        else:
            if future_price <= tp_level:
                exit_idx = entry_idx + offset; exit_reason = 'tp'; break
            if future_price >= sl_level:
                exit_idx = entry_idx + offset; exit_reason = 'sl'; break

    exit_price = float(prices[exit_idx])
    price_return   = (exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)
    path_moves     = (future_prices - entry_price) if direction == 'LONG' else (entry_price - future_prices)
    favourable_move = float(np.max(path_moves)) if path_moves.size else 0.0
    adverse_move    = float(np.min(path_moves)) if path_moves.size else 0.0

    return {
        'exit_idx':    int(exit_idx),
        'exit_price':  float(exit_price),
        'exit_reason': exit_reason,
        'hold_steps':  int(exit_idx - entry_idx),
        'price_return': float(price_return),
        'raw_pnl_pips': float(price_return / max(float(tick_size), 1e-8)),
        'mfe_pips':    float(favourable_move / max(float(tick_size), 1e-8)),
        'mae_pips':    float(adverse_move    / max(float(tick_size), 1e-8)),
        'tp_level':    round(tp_level, 5),
        'sl_level':    round(sl_level, 5),
        'tp_pips':     round(abs(tp_level - entry_price) / max(tick_size, 1e-8), 1),
        'sl_pips':     round(abs(sl_level - entry_price) / max(tick_size, 1e-8), 1),
        'rr_ratio':    round(abs(tp_level - entry_price) / max(abs(sl_level - entry_price), tick_size), 2),
    }


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ('ts_event', 'label_end_ts'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)
    return df


def _load_json_if_exists(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _load_backtest_contract(csv_path: str, models_dir: str) -> tuple[dict, dict]:
    model_manifest = _load_json_if_exists(os.path.join(models_dir, 'manifest.json')) or {}
    dataset_manifest = _load_json_if_exists(os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'artifact_manifest.json')) or {}
    return model_manifest, dataset_manifest


def _enforce_oos_backtest_guard(
    df: pd.DataFrame,
    *,
    csv_path: str,
    models_dir: str,
    allow_in_sample_data_override: bool = False,
) -> dict:
    info = {
        'checked': True,
        'allowed': True,
        'reason': 'no_overlap_detected',
    }
    if allow_in_sample_data_override:
        info['reason'] = 'override_enabled'
        return info

    if 'bias_label' not in df.columns and 'forward_return' not in df.columns:
        info['reason'] = 'unlabeled_dataset'
        return info

    model_manifest, dataset_manifest = _load_backtest_contract(csv_path, models_dir)
    source_contract = ((model_manifest.get('extra', {}) or {}).get('source_contract', {}) or {})
    source_csv = os.path.abspath(str(source_contract.get('source_csv') or model_manifest.get('inputs', {}).get('csv') or ''))
    backtest_csv = os.path.abspath(csv_path)
    source_dataset_id = str(source_contract.get('dataset_id') or '')
    backtest_dataset_id = str((dataset_manifest.get('extra', {}) or {}).get('dataset_id') or '')

    dataset_slice = pd.Series(df.get('dataset_slice', pd.Series([], dtype='object'))).astype(str).str.lower()
    is_pure_holdout = bool(len(dataset_slice) > 0 and dataset_slice.isin(['holdout']).all())

    same_path = bool(source_csv) and source_csv == backtest_csv
    same_dataset = bool(source_dataset_id) and source_dataset_id == backtest_dataset_id

    if (same_path or same_dataset) and not is_pure_holdout:
        info['allowed'] = False
        info['reason'] = 'same_training_dataset'
        raise ValueError(
            '❌ Refusing in-sample labeled backtest by default. '
            'Use a pure holdout/OOS dataset or pass --allow_in_sample_data_override intentionally.'
        )

    if same_dataset and is_pure_holdout:
        info['reason'] = 'same_dataset_holdout_only'
    elif same_path and is_pure_holdout:
        info['reason'] = 'same_csv_holdout_only'

    return info


def _filter_backtest_window(
    df: pd.DataFrame,
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> pd.DataFrame:
    if 'ts_event' not in df.columns or (start_ts is None and end_ts is None):
        return df.copy()

    out = df.copy()
    ts = pd.to_datetime(out['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    mask = pd.Series(True, index=out.index)

    if start_ts is not None:
        start = pd.to_datetime(start_ts, utc=True, errors='coerce')
        if not pd.isna(start):
            mask &= ts >= start.tz_localize(None)

    if end_ts is not None:
        end = pd.to_datetime(end_ts, utc=True, errors='coerce')
        if not pd.isna(end):
            mask &= ts < end.tz_localize(None)

    return out.loc[mask].reset_index(drop=True)


def _directional_event_mask(df: pd.DataFrame) -> np.ndarray:
    event_flag = pd.to_numeric(df.get('event_flag', 0), errors='coerce').fillna(0).astype(np.int8)
    train_event_flag = pd.to_numeric(df.get('train_event_flag', event_flag), errors='coerce').fillna(0).astype(np.int8)
    bias_label = pd.to_numeric(df.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int8)

    directional_mask = bias_label.isin([0, 1])
    mask = (train_event_flag == 1) & directional_mask

    if not mask.any():
        fallback_mask = (event_flag == 1) & directional_mask
        if fallback_mask.any():
            mask = fallback_mask

    if not mask.any() and directional_mask.any():
        mask = directional_mask

    return mask.to_numpy(dtype=bool)


def _coerce_row_aligned_array(arr: np.ndarray, expected_dim: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f'❌ array must be 2D, got {arr.shape}')

    if arr.shape[1] < expected_dim:
        out = np.zeros((len(arr), expected_dim), dtype=dtype)
        out[:, :arr.shape[1]] = arr
        return out

    return arr[:, :expected_dim]


def _expand_directional_rows(
    df: pd.DataFrame,
    arr: np.ndarray,
    expected_dim: int,
    *,
    kind: str,
    path: str,
    dtype=np.float32,
) -> np.ndarray | None:
    mask = _directional_event_mask(df)
    n_rows = int(mask.sum())
    if n_rows != len(arr):
        return None

    out = np.zeros((len(df), expected_dim), dtype=dtype)
    coerced = _coerce_row_aligned_array(arr, expected_dim, dtype=dtype)
    out[mask] = coerced
    print(
        f"  ℹ️ Expanded {kind} from directional event rows: "
        f"{n_rows:,} -> {len(df):,} ({os.path.basename(path)})"
    )
    return out


def _coverage_sidecar_candidates(
    path: str | None,
    models_dir: str | None,
    coverage_name: str,
) -> list[str]:
    seen = set()
    candidates = []
    for base in (
        os.path.dirname(os.path.abspath(path)) if path else None,
        os.path.abspath(models_dir) if models_dir else None,
    ):
        if not base:
            continue
        candidate = os.path.join(base, coverage_name)
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _expand_compact_rows_with_coverage(
    arr: np.ndarray,
    expected_dim: int,
    *,
    path: str,
    models_dir: str | None,
    coverage_name: str,
    kind: str,
    dtype=np.float32,
) -> np.ndarray:
    coerced = _coerce_row_aligned_array(arr, expected_dim, dtype=dtype)
    for coverage_path in _coverage_sidecar_candidates(path, models_dir, coverage_name):
        if not os.path.exists(coverage_path):
            continue

        coverage = np.asarray(np.load(coverage_path)).reshape(-1).astype(bool)
        if int(coverage.sum()) != len(coerced):
            continue

        out = np.zeros((len(coverage), expected_dim), dtype=dtype)
        out[coverage] = coerced
        print(
            f"  ℹ️ Expanded compact {kind} via coverage mask: "
            f"{len(coerced):,} -> {len(coverage):,} "
            f"({os.path.basename(path)} + {os.path.basename(coverage_path)})"
        )
        return out

    return coerced


def _load_visual_embeddings(
    df: pd.DataFrame,
    explicit_path: str | None,
    default_path: str | None,
    expected_dim: int,
    models_dir: str | None = None,
) -> np.ndarray:
    n = len(df)
    zero = np.zeros((n, expected_dim), dtype=np.float32)

    path = explicit_path if explicit_path else default_path
    if not path or not os.path.exists(path):
        return zero

    vis = np.load(path)
    vis = np.asarray(vis, dtype=np.float32)
    if vis.ndim != 2:
        raise ValueError(f'❌ visual embeddings must be 2D, got {vis.shape}')
    vis = _expand_compact_rows_with_coverage(
        vis,
        expected_dim,
        path=path,
        models_dir=models_dir,
        coverage_name='visual_coverage_v19.npy',
        kind='visual embeddings',
        dtype=np.float32,
    )

    if len(vis) == n:
        return vis

    expanded = _expand_directional_rows(
        df,
        vis,
        expected_dim,
        kind='visual embeddings',
        path=path,
        dtype=np.float32,
    )
    if expanded is not None:
        return expanded

    directional_rows = int(_directional_event_mask(df).sum())
    if explicit_path:
        raise ValueError(
            f'❌ visual embeddings rows ({len(vis)}) do not match CSV rows ({n}) '
            f'or directional-event rows ({directional_rows}) for explicit file {path}. '
            'This usually means the embeddings were produced from a different '
            'training CSV, or from a compact covered-rows artifact without a '
            'matching visual_coverage_v19.npy sidecar.'
        )

    if len(vis) < n:
        out = zero.copy()
        out[:len(vis)] = vis
        return out

    return vis[:n]


def _load_meta_features(
    df: pd.DataFrame,
    explicit_path: str | None,
    expected_dim: int,
    allow_in_sample_live_override: bool = False,
) -> np.ndarray | None:
    n = len(df)
    if not explicit_path or not os.path.exists(explicit_path):
        return None

    basename = os.path.basename(str(explicit_path)).lower()
    if (
        not allow_in_sample_live_override
        and 'live' in basename
        and ('bias_label' in df.columns or 'forward_return' in df.columns)
    ):
        # The final/live meta stack is fitted on all rows. Reusing it on a
        # labeled backtest slice would leak in-sample predictions back into the
        # evaluation unless the user explicitly overrides this guard.
        raise ValueError(
            '❌ Refusing to use live/final-fit meta features on a labeled backtest dataset. '
            'Use meta_features_oof_v19.npy or omit --meta_npy.'
        )

    meta = np.load(explicit_path)
    meta = np.asarray(meta, dtype=np.float32)
    if meta.ndim != 2:
        raise ValueError(f'❌ meta features must be 2D, got {meta.shape}')

    if len(meta) == n:
        meta = _coerce_row_aligned_array(meta, expected_dim, dtype=np.float32)
    else:
        expanded = _expand_directional_rows(
            df,
            meta,
            expected_dim,
            kind='meta features',
            path=explicit_path,
            dtype=np.float32,
        )
        if expanded is None:
            raise ValueError(
                f'❌ meta features rows ({len(meta)}) do not match CSV rows ({n}) '
                f'or directional-event rows for explicit file {explicit_path}'
            )
        meta = expanded

    if meta.shape[1] != expected_dim:
        raise ValueError(
            f'❌ meta features columns ({meta.shape[1]}) لا تطابق schema المطلوب ({expected_dim})'
        )
    return meta


def _equity_metrics(equity_curve: list[float]) -> tuple[float, float]:
    if not equity_curve:
        return 0.0, 0.0
    eq = np.asarray(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(eq)
    dd = peaks - eq
    mdd_abs = float(dd.max()) if len(dd) else 0.0
    mdd_pct = float((dd / np.maximum(peaks, 1e-8)).max()) if len(dd) else 0.0
    return mdd_abs, mdd_pct


def _trade_sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    arr = np.asarray(pnls, dtype=np.float64)
    std = float(arr.std())
    if std <= 1e-12:
        return 0.0
    return float(arr.mean() / std * math.sqrt(len(arr)))


def _directional_metrics(results_df: pd.DataFrame) -> dict:
    if results_df.empty or 'true_bias' not in results_df.columns:
        return {
            'directional_precision_macro': 0.0,
            'directional_recall_macro': 0.0,
            'directional_f1_macro': 0.0,
        }

    y_true = pd.to_numeric(results_df.get('true_bias', 2), errors='coerce').fillna(2).astype(int).values
    y_pred = pd.to_numeric(results_df.get('bias_idx', 2), errors='coerce').fillna(2).astype(int).values
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average='macro',
        zero_division=0,
    )
    return {
        'directional_precision_macro': round(float(precision), 4),
        'directional_recall_macro': round(float(recall), 4),
        'directional_f1_macro': round(float(f1), 4),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def _coverage_counts(mask: np.ndarray, covered_mask: np.ndarray) -> tuple[int, int, float]:
    mask = np.asarray(mask, dtype=bool)
    covered_mask = np.asarray(covered_mask, dtype=bool)
    total = int(mask.sum())
    covered = int(np.sum(mask & covered_mask))
    return total, covered, round(_ratio(covered, total), 4)


def _load_visual_training_reference(models_dir: str | None) -> dict | None:
    if not models_dir:
        return None
    path = os.path.join(models_dir, 'visual_metrics_v19.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            'path': path,
            'rows_total': int(data.get('n_rows', 0)),
            'rows_with_tensor': int(data.get('rows_with_tensor', 0)),
            'coverage_ratio': round(float(data.get('coverage_ratio', 0.0)), 4),
            'n_tensors': int(data.get('n_tensors', data.get('n_tensors_raw', 0))),
        }
    except Exception:
        return None


def _build_visual_diagnostics(
    df: pd.DataFrame,
    visual_embeddings: np.ndarray,
    *,
    models_dir: str | None = None,
    results_df: pd.DataFrame | None = None,
    eval_visual_diagnostics: dict | None = None,
) -> dict:
    n_rows = len(df)
    if n_rows == 0:
        base = {
            'rows_total': 0,
            'rows_with_visual': 0,
            'coverage_ratio': 0.0,
            'diagnosis_notes': ['لا توجد صفوف لتقييم التغطية البصرية.'],
        }
        if eval_visual_diagnostics:
            base.update(eval_visual_diagnostics)
        return base

    if visual_embeddings is None or np.size(visual_embeddings) == 0:
        covered_mask = np.zeros(n_rows, dtype=bool)
    else:
        vis = np.asarray(visual_embeddings, dtype=np.float32)
        if vis.ndim == 1:
            vis = vis.reshape(-1, 1)
        covered_mask = np.linalg.norm(vis, axis=1) > 0

    event_flag = pd.to_numeric(df.get('event_flag', 0), errors='coerce').fillna(0).astype(np.int8)
    train_event_flag = pd.to_numeric(df.get('train_event_flag', event_flag), errors='coerce').fillna(0).astype(np.int8)
    bias_label = pd.to_numeric(df.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int8)
    directional_mask = bias_label.isin([0, 1]).to_numpy(dtype=bool)
    full_mask = np.ones(n_rows, dtype=bool)
    event_mask = (event_flag == 1).to_numpy(dtype=bool)
    train_event_mask = (train_event_flag == 1).to_numpy(dtype=bool)
    directional_train_event_mask = train_event_mask & directional_mask

    tradeable_mask = np.zeros(n_rows, dtype=bool)
    executed_mask = np.zeros(n_rows, dtype=bool)
    if results_df is not None and len(results_df) == n_rows:
        tradeable_mask = pd.to_numeric(results_df.get('tradeable', 0), errors='coerce').fillna(0).astype(bool).to_numpy()
        executed_mask = pd.to_numeric(results_df.get('executed', 0), errors='coerce').fillna(0).astype(bool).to_numpy()

    full_total, full_covered, full_ratio = _coverage_counts(full_mask, covered_mask)
    event_total, event_covered, event_ratio = _coverage_counts(event_mask, covered_mask)
    train_event_total, train_event_covered, train_event_ratio = _coverage_counts(train_event_mask, covered_mask)
    directional_total, directional_covered, directional_ratio = _coverage_counts(directional_mask, covered_mask)
    directional_train_total, directional_train_covered, directional_train_ratio = _coverage_counts(
        directional_train_event_mask,
        covered_mask,
    )
    tradeable_total, tradeable_covered, tradeable_ratio = _coverage_counts(tradeable_mask, covered_mask)
    executed_total, executed_covered, executed_ratio = _coverage_counts(executed_mask, covered_mask)

    diagnostics = {
        'source': str((eval_visual_diagnostics or {}).get('source', 'provided')),
        'reason': str((eval_visual_diagnostics or {}).get('reason', 'ok')),
        'rows_total': int(full_total),
        'rows_with_visual': int(full_covered),
        'coverage_ratio': float(full_ratio),
        'event_rows': int(event_total),
        'event_rows_with_visual': int(event_covered),
        'event_coverage_ratio': float(event_ratio),
        'train_event_rows': int(train_event_total),
        'train_event_rows_with_visual': int(train_event_covered),
        'train_event_coverage_ratio': float(train_event_ratio),
        'directional_rows': int(directional_total),
        'directional_rows_with_visual': int(directional_covered),
        'directional_coverage_ratio': float(directional_ratio),
        'directional_train_event_rows': int(directional_train_total),
        'directional_train_event_rows_with_visual': int(directional_train_covered),
        'directional_train_event_coverage_ratio': float(directional_train_ratio),
        'tradeable_rows': int(tradeable_total),
        'tradeable_rows_with_visual': int(tradeable_covered),
        'tradeable_coverage_ratio': float(tradeable_ratio),
        'executed_rows': int(executed_total),
        'executed_rows_with_visual': int(executed_covered),
        'executed_coverage_ratio': float(executed_ratio),
    }

    if eval_visual_diagnostics:
        for key in (
            'lob_tensors_available',
            'lob_timestamps_available',
            'rows_with_tensor',
            'rows_with_tensor_ratio',
            'used_tensor_count',
            'visual_coverage_ratio',
        ):
            if key in eval_visual_diagnostics:
                diagnostics[key] = eval_visual_diagnostics[key]

    training_reference = _load_visual_training_reference(models_dir)
    if training_reference is not None:
        diagnostics['training_reference'] = training_reference
        diagnostics['coverage_gap_vs_train'] = round(
            diagnostics['coverage_ratio'] - float(training_reference.get('coverage_ratio', 0.0)),
            4,
        )

    notes: list[str] = []
    if training_reference is not None and training_reference.get('rows_total', 0) != diagnostics['rows_total']:
        notes.append(
            "تغطية التدريب المرجعية محسوبة على event rows فقط "
            f"({training_reference.get('rows_total', 0):,})، بينما ملخص الباكتيست الافتراضي هنا على كل الصفوف "
            f"({diagnostics['rows_total']:,})."
        )
    if training_reference is not None and diagnostics.get('coverage_gap_vs_train', 0.0) <= -0.25:
        notes.append(
            "هناك فجوة كبيرة بين تغطية الـ visual branch في التدريب والتقييم "
            f"({diagnostics['coverage_gap_vs_train']:+.1%})."
        )
    used_tensor_count = int(diagnostics.get('used_tensor_count', 0))
    directional_train_rows = int(diagnostics.get('directional_train_event_rows', 0))
    if directional_train_rows > 0 and _ratio(used_tensor_count, directional_train_rows) < 0.25:
        notes.append(
            "عدد الـ LOB tensors المستخدمة قليل جدًا مقارنةً بعدد directional train-event rows، "
            "وهذا يشير عادةً إلى أن لقطات الـ MBP في التقييم sparse أو أن عدة أحداث تنهار على نفس snapshot."
        )
    rows_with_tensor = int(diagnostics.get('rows_with_tensor', diagnostics['rows_with_visual']))
    if rows_with_tensor > 0 and diagnostics['rows_with_visual'] < rows_with_tensor:
        notes.append(
            "بعض الصفوف اصطفّت مع tensors زمنياً لكن خرجت embeddings صفرية؛ هذا يوحي بمشكلة إضافية بعد المحاذاة وليس في التوقيت فقط."
        )
    if diagnostics['tradeable_rows'] > 0 and diagnostics['tradeable_coverage_ratio'] < 0.25:
        notes.append(
            "حتى بين الصفوف tradeable، التغطية البصرية منخفضة؛ لذلك الـ MetaLearner غالبًا يتخذ قراراته على stat/meta فقط معظم الوقت."
        )
    if not notes:
        notes.append("لا يظهر خلل واضح في تغطية الـ visual branch من الملخص الحالي.")
    diagnostics['diagnosis_notes'] = notes
    return diagnostics


def run_causal_backtest(
    df: pd.DataFrame,
    models_dir: str,
    output_dir: str,
    visual_embeddings: np.ndarray,
    meta_features: np.ndarray | None,
    input_scaled: bool,
    tick_size: float,
    tick_value: float,
    round_trip_cost_pips: float,
    max_size: int,
    starting_equity: float,
    direction_threshold_ticks: float = 1.0,
    tp_mult: float = 1.2,
    sl_mult: float = 1.0,
    max_horizon_steps: int | None = None,
    allow_oracle_forward_return: bool = False,
    single_position_only: bool = True,
    cooldown_rows: int = 0,
    visual_diagnostics: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    engine = V19PredictionEngine(models_dir, run_mode='backtest')
    engine.reset_state()
    engine.loss_guard.update_equity(starting_equity)
    source_has_labels = 'bias_label' in df.columns
    source_has_fwd = 'forward_return' in df.columns
    replay_df = engine.factory.prepare_frame(df, already_scaled=input_scaled, include_meta=True)
    price_arr = _series_or_default(replay_df, 'price', 0.0, dtype=np.float64).values
    horizon_arr = _series_or_default(replay_df, 'label_horizon_steps', 0, dtype=np.int32).values
    if 'raw__micro_atr' in replay_df.columns:
        micro_atr_arr = _series_or_default(replay_df, 'raw__micro_atr', 0.0, dtype=np.float64).values
    else:
        micro_atr_arr = _series_or_default(replay_df, 'micro_atr', 0.0, dtype=np.float64).values
    ts_arr = pd.to_datetime(
        replay_df.get('ts_event', pd.Series([pd.NaT] * len(replay_df), index=replay_df.index)),
        utc=True,
        errors='coerce',
    ).dt.tz_localize(None)

    results = []
    trades = []
    equity_curve = [float(starting_equity)]
    equity = float(starting_equity)

    has_labels = source_has_labels
    blocked_count = 0
    replayed_trades = 0
    oracle_trades = 0
    skipped_trade_replays = 0
    active_trade = None
    cooldown_until = -1

    for i, (_, row) in enumerate(replay_df.iterrows()):
        ts = row.get('ts_event', None)
        if active_trade is not None and i >= int(active_trade['exit_idx']):
            equity += float(active_trade['pnl'])
            engine.loss_guard.update_equity(equity)
            engine.loss_guard.record_trade(float(active_trade['pnl']), ts=ts)
            equity_curve.append(float(equity))
            closed_trade = {
                **active_trade,
                'exit_ts': '' if pd.isna(ts) else str(ts),
                'equity_after': round(float(equity), 2),
            }
            trades.append(closed_trade)
            if active_trade.get('pnl_source') == 'path_replay':
                replayed_trades += 1
            else:
                oracle_trades += 1
            active_trade = None
            cooldown_until = i + max(int(cooldown_rows), 0)

        visual = visual_embeddings[i] if visual_embeddings.size else None
        pred = engine.predict_step(
            row.to_dict(),
            visual_embedding=visual,
            meta_override=meta_features[i] if meta_features is not None else None,
            ts=ts,
            already_scaled=True,
        )

        if pred.get('reason', '').startswith('Warming up'):
            continue

        pred['idx'] = i
        pred['ts_event'] = str(ts) if ts is not None and not pd.isna(ts) else ''
        pred['price'] = _safe_float(row.get('price', 0.0))

        if has_labels:
            true_bias = int(row.get('bias_label', 2))
            pred['true_bias'] = true_bias
            pred['true_label'] = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}.get(true_bias, '?')
            pred['correct'] = (pred.get('bias_idx') == true_bias)

        if pred.get('reason') and 'Warming up' not in pred.get('reason', ''):
            blocked_count += 1

        tradeable = bool(pred.get('tradeable', False))
        direction = pred.get('bias', 'NEUTRAL')
        pred['executed'] = False

        if active_trade is not None and bool(single_position_only):
            pred['trade_skip_reason'] = 'single_position_only'
            results.append(pred)
            continue
        if i < cooldown_until:
            pred['trade_skip_reason'] = 'cooldown_active'
            results.append(pred)
            continue

        if tradeable and direction in ('LONG', 'SHORT'):
            size = int(pred.get('position_size') or confidence_bet_size(
                float(pred.get('confidence', 0.0)),
                base_size=1,
                max_size=max_size,
            ))
            trade_path = _simulate_trade_path(
                entry_idx=i,
                direction=direction,
                prices=price_arr,
                horizons=horizon_arr,
                micro_atr=micro_atr_arr,
                tick_size=tick_size,
                direction_threshold_ticks=direction_threshold_ticks,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                max_horizon_steps=max_horizon_steps,
                row_data=row.to_dict(),  # FIX: تمرير بيانات الجدران للـ DynamicTargetManager
            )

            pnl_source = 'path_replay'
            if trade_path is None and allow_oracle_forward_return and source_has_fwd:
                forward_return = _safe_float(row.get('forward_return', 0.0))
                trade_path = {
                    'exit_idx': min(i + max(_safe_int(row.get('label_horizon_steps', 1), 1), 1), len(replay_df) - 1),
                    'exit_price': _safe_float(row.get('price', 0.0)) + (forward_return if direction == 'LONG' else -forward_return),
                    'exit_reason': 'oracle_forward_return',
                    'hold_steps': max(_safe_int(row.get('label_horizon_steps', 1), 1), 1),
                    'price_return': forward_return if direction == 'LONG' else -forward_return,
                    'raw_pnl_pips': (forward_return / tick_size) if direction == 'LONG' else (-forward_return / tick_size),
                    'mfe_pips': 0.0,
                    'mae_pips': 0.0,
                }
                pnl_source = 'oracle_forward_return'

            if trade_path is None:
                skipped_trade_replays += 1
                pred['trade_skip_reason'] = 'missing_exit_path'
                results.append(pred)
                continue

            raw_pnl_pips = float(trade_path['raw_pnl_pips'])
            net_pnl_pips = raw_pnl_pips - round_trip_cost_pips
            net_pnl_dollars = net_pnl_pips * tick_value * size

            result = 'WIN' if net_pnl_pips > 0 else ('LOSE' if net_pnl_pips < 0 else 'FLAT')
            exit_idx = int(trade_path['exit_idx'])
            exit_ts = ts_arr.iloc[exit_idx] if exit_idx < len(ts_arr) else pd.NaT
            trade = {
                'idx': i,
                'ts_event': pred['ts_event'],
                'dir': direction,
                'confidence': round(_safe_float(pred.get('confidence', 0.0)), 4),
                'size': size,
                'ep': _safe_float(row.get('price', 0.0)),
                'xp': round(float(trade_path['exit_price']), 6),
                'exit_idx': exit_idx,
                'exit_ts': '' if pd.isna(exit_ts) else str(exit_ts),
                'exit_reason': str(trade_path['exit_reason']),
                'dur_steps': int(trade_path['hold_steps']),
                'dur_min': int(trade_path['hold_steps']),
                'pips': round(net_pnl_pips, 4),
                'raw_pnl_pips': round(raw_pnl_pips, 4),
                'cost_pips': round(round_trip_cost_pips, 4),
                'mfe_pips': round(float(trade_path.get('mfe_pips', 0.0)), 4),
                'mae_pips': round(float(trade_path.get('mae_pips', 0.0)), 4),
                'pnl_source': pnl_source,
                'pnl': round(net_pnl_dollars, 2),
                'result': result,
                'cluster': pred.get('cluster', 0),
                'cluster_name': pred.get('cluster_name', 'Unknown'),
            }
            pred['executed'] = True
            pred['position_size'] = size
            pred['pending_exit_idx'] = exit_idx
            pred['pending_exit_ts'] = '' if pd.isna(exit_ts) else str(exit_ts)
            pred['exit_reason'] = str(trade_path['exit_reason'])
            pred['pnl_source'] = pnl_source

            if bool(single_position_only):
                active_trade = trade
            else:
                equity += net_pnl_dollars
                engine.loss_guard.update_equity(equity)
                engine.loss_guard.record_trade(net_pnl_dollars, ts=ts)
                equity_curve.append(float(equity))
                trade['equity_after'] = round(float(equity), 2)
                trades.append(trade)
                if pnl_source == 'path_replay':
                    replayed_trades += 1
                else:
                    oracle_trades += 1
                pred['net_pnl_pips'] = round(net_pnl_pips, 4)
                pred['net_pnl_dollars'] = round(net_pnl_dollars, 2)
                pred['equity_after'] = round(equity, 2)

        results.append(pred)

    results_df = pd.DataFrame(results)
    trades_df = pd.DataFrame(trades)
    visual_diag = _build_visual_diagnostics(
        replay_df,
        visual_embeddings,
        models_dir=models_dir,
        results_df=results_df,
        eval_visual_diagnostics=visual_diagnostics,
    )

    mdd_abs, mdd_pct = _equity_metrics(equity_curve)
    trade_pnls = trades_df['pnl'].tolist() if not trades_df.empty else []
    wins = trades_df[trades_df['pnl'] > 0] if not trades_df.empty else trades_df
    losses = trades_df[trades_df['pnl'] < 0] if not trades_df.empty else trades_df

    summary = {
        'predictions': int(len(results_df)),
        'trades': int(len(trades_df)),
        'tradeable_signals': int(results_df['tradeable'].sum()) if 'tradeable' in results_df.columns else 0,
        **_directional_metrics(results_df),
        'event_gate_rate': round(float(results_df['event_gate_passed'].mean()), 4)
            if 'event_gate_passed' in results_df.columns and len(results_df) else 0.0,
        'win_rate': round(float((trades_df['pnl'] > 0).mean()), 4) if len(trades_df) else 0.0,
        'total_pnl_dollars': round(float(trades_df['pnl'].sum()), 2) if len(trades_df) else 0.0,
        'avg_trade_pnl_dollars': round(float(trades_df['pnl'].mean()), 2) if len(trades_df) else 0.0,
        'avg_trade_pips': round(float(trades_df['pips'].mean()), 4) if len(trades_df) else 0.0,
        'profit_factor': round(float(wins['pnl'].sum() / abs(losses['pnl'].sum())), 4)
            if len(losses) and abs(float(losses['pnl'].sum())) > 1e-9 else 0.0,
        'trade_sharpe': round(_trade_sharpe(trade_pnls), 4),
        'max_drawdown_dollars': round(float(mdd_abs), 2),
        'max_drawdown_pct': round(float(mdd_pct), 4),
        'ending_equity': round(float(equity), 2),
        'blocked_predictions': int(blocked_count),
        'pnl_engine': 'path_replay',
        'replayed_trades': int(replayed_trades),
        'oracle_forward_return_trades': int(oracle_trades),
        'oracle_forward_return_used': bool(oracle_trades > 0),
        'skipped_trade_replays': int(skipped_trade_replays),
        'single_position_only': bool(single_position_only),
        'cooldown_rows': int(max(cooldown_rows, 0)),
        'visual_coverage': float(visual_diag.get('coverage_ratio', 0.0)),
        'visual_diagnostics': visual_diag,
    }

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'backtest_v19_results.csv'), index=False)
    trades_df.to_csv(os.path.join(output_dir, 'backtest_v19_trades.csv'), index=False)
    with open(os.path.join(output_dir, 'backtest_v19_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, 'visual_diagnostics_v19.json'), 'w') as f:
        json.dump(visual_diag, f, indent=2)

    if HTML_REPORT_AVAILABLE and len(trades_df):
        try:
            generate_backtest_report(
                output_dir=output_dir,
                trades=trades,
                equity=equity_curve,
                n_test_bars=len(results_df),
                model_acc=summary.get('directional_f1_macro', 0.0),
                n_features=0,
                n_dataset=len(replay_df),
                backtest_summary=summary,
                visual_diagnostics=visual_diag,
            )
        except Exception as e:
            print(f"  ⚠️ HTML report skipped: {e}")

    return results_df, trades_df, summary


def main():
    p = argparse.ArgumentParser(description='Causal replay backtester for QuantSystem V19')
    p.add_argument('--csv', required=True, help='training_features_ready.csv or compatible V19 feature CSV')
    p.add_argument('--models', default='outputs_v19', help='trained V19 models directory')
    p.add_argument('--output', default='outputs_v19', help='backtest output directory')
    p.add_argument('--visual_npy', default=None, help='optional row-aligned visual embeddings file')
    p.add_argument('--meta_npy', default=None, help='optional row-aligned stage-1 meta features file')
    p.add_argument('--allow_in_sample_live_meta_override', action='store_true',
                   help='dangerous: allow explicit live/final-fit meta features on labeled backtest data')
    p.add_argument('--allow_in_sample_data_override', action='store_true',
                   help='dangerous: allow backtesting directly on the training dataset / non-holdout labeled CSV')
    p.add_argument('--input_scaled', action='store_true',
                   help='set when CSV is already scaled like training_features_ready.csv')
    p.add_argument('--tick_size', type=float, default=0.0001)
    p.add_argument('--tick_value', type=float, default=10.0)
    p.add_argument('--round_trip_cost_pips', type=float, default=1.0)
    p.add_argument('--max_size', type=int, default=5)
    p.add_argument('--starting_equity', type=float, default=100000.0)
    p.add_argument('--direction_threshold_ticks', type=float, default=1.0)
    p.add_argument('--tp_mult', type=float, default=1.2)
    p.add_argument('--sl_mult', type=float, default=1.0)
    p.add_argument('--max_horizon_steps', type=int, default=0,
                   help='optional cap on replay horizon in rows; 0 uses label_horizon_steps as-is')
    p.add_argument('--allow_oracle_forward_return', action='store_true',
                   help='dangerous: fall back to stored forward_return when no causal replay window is available')
    p.add_argument('--disable_single_position_only', action='store_true',
                   help='allow overlapping trades; default keeps one active position at a time')
    p.add_argument('--cooldown_rows', type=int, default=0,
                   help='rows to wait after closing a trade before opening a new one')
    args = p.parse_args()

    df = _load_csv(args.csv)
    oos_guard = _enforce_oos_backtest_guard(
        df,
        csv_path=args.csv,
        models_dir=args.models,
        allow_in_sample_data_override=args.allow_in_sample_data_override,
    )
    engine = V19PredictionEngine(args.models)
    visual_embeddings = _load_visual_embeddings(
        df,
        explicit_path=args.visual_npy,
        default_path=engine.visual_emb_path,
        expected_dim=len(engine.visual_features),
        models_dir=args.models,
    )
    meta_features = _load_meta_features(
        df,
        explicit_path=args.meta_npy,
        expected_dim=len(engine.meta_features),
        allow_in_sample_live_override=args.allow_in_sample_live_meta_override,
    )

    _, _, summary = run_causal_backtest(
        df=df,
        models_dir=args.models,
        output_dir=args.output,
        visual_embeddings=visual_embeddings,
        meta_features=meta_features,
        input_scaled=args.input_scaled,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        round_trip_cost_pips=args.round_trip_cost_pips,
        max_size=args.max_size,
        starting_equity=args.starting_equity,
        direction_threshold_ticks=args.direction_threshold_ticks,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        max_horizon_steps=(args.max_horizon_steps if args.max_horizon_steps > 0 else None),
        allow_oracle_forward_return=args.allow_oracle_forward_return,
        single_position_only=(not args.disable_single_position_only),
        cooldown_rows=args.cooldown_rows,
    )
    summary['oos_guard'] = oos_guard

    print("\n✅ V19 causal backtest complete")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
