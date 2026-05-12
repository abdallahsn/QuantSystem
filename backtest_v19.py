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
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _configure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows consoles when printing non-ASCII log lines."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            try:
                reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


_configure_stdio_utf8()

from modules.feature_artifact_v19 import (
    load_artifact_manifest,
    load_feature_artifact,
    resolve_artifact_root,
)
from modules.slippage_model import SlippageModel, position_size_from_prediction
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


def _frame_symbol_counts(df: pd.DataFrame) -> tuple[str | None, dict[str, int]]:
    if 'symbol' in df.columns:
        col = 'symbol'
    elif 'instrument_id' in df.columns:
        col = 'instrument_id'
    else:
        return None, {}
    counts = df[col].fillna('UNKNOWN').astype(str).value_counts()
    return col, {str(key): int(value) for key, value in counts.items()}


def _assert_single_contract_df(df: pd.DataFrame, *, context: str) -> None:
    col, counts = _frame_symbol_counts(df)
    if col is None or len(counts) <= 1:
        return
    raise RuntimeError(
        "❌ Mixed-contract/symbol data is not allowed in V19 hardened backtests "
        f"| context={context} | column={col} | counts={counts}"
    )


def _binary_ece(y_true: np.ndarray, p_long: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    p_long = np.clip(np.asarray(p_long, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if y_true.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, int(max(n_bins, 2)) + 1)
    bucket = np.digitize(p_long, bins[1:-1], right=False)
    ece = 0.0
    for idx in range(len(bins) - 1):
        mask = bucket == idx
        if not np.any(mask):
            continue
        conf = float(np.mean(p_long[mask]))
        acc = float(np.mean(y_true[mask] == 0))
        ece += abs(conf - acc) * (float(np.sum(mask)) / float(y_true.size))
    return float(ece)


def _realized_fill_pricing(
    *,
    entry_row: dict,
    exit_row: dict,
    direction: str,
    size: int,
    raw_pnl_pips: float,
    tick_size: float,
    round_trip_cost_pips: float,
    tick_value: float,
    commission_per_side: float = 0.0,
    min_spread_ticks: float = 1.0,
    min_slippage_ticks: float = 1.0,
    spread_multiplier: float = 0.5,
) -> dict:
    tick = max(float(tick_size), 1e-8)
    raw_pnl_pips = float(raw_pnl_pips)
    floor_cost = max(float(round_trip_cost_pips), 0.0)

    def _spread_ticks(row_like: dict | None) -> float:
        row_like = row_like or {}
        best_bid = _safe_float(row_like.get('bid_px_00', 0.0))
        best_ask = _safe_float(row_like.get('ask_px_00', 0.0))
        if best_bid > 0 and best_ask > best_bid:
            return max((best_ask - best_bid) / tick, float(min_spread_ticks))
        return float(min_spread_ticks)

    commission_one_way_pips = max(float(commission_per_side), 0.0) / max(float(tick_value), 1e-8)
    fill_model = SlippageModel(
        tick_size=tick,
        tick_value=max(float(tick_value), 1e-8),
        commission=max(float(commission_per_side), 0.0),
    )
    entry_fill = fill_model.compute_fill(entry_row or {}, size=max(int(size), 1), direction=str(direction).lower())
    exit_direction = 'short' if str(direction).upper() == 'LONG' else 'long'
    exit_fill = fill_model.compute_fill(exit_row or {}, size=max(int(size), 1), direction=exit_direction)

    entry_price = _safe_float(entry_fill.get('fill_price', 0.0))
    exit_price = _safe_float(exit_fill.get('fill_price', 0.0))
    if entry_fill.get('filled', 0) <= 0 or exit_fill.get('filled', 0) <= 0 or entry_price <= 0 or exit_price <= 0:
        fallback_cost = max(
            floor_cost,
            2.0 * (
                max(float(min_slippage_ticks), float(spread_multiplier) * _spread_ticks(entry_row))
                + commission_one_way_pips
            ),
        )
        return {
            'used_dynamic_fill': False,
            'entry_fill': entry_fill,
            'exit_fill': exit_fill,
            'fill_pnl_pips': raw_pnl_pips,
            'dynamic_cost_pips': float(fallback_cost),
            'net_pnl_pips': float(raw_pnl_pips - fallback_cost),
        }

    if str(direction).upper() == 'LONG':
        fill_pnl_pips = float((exit_price - entry_price) / tick)
    else:
        fill_pnl_pips = float((entry_price - exit_price) / tick)

    entry_spread_ticks = _spread_ticks(entry_row)
    exit_spread_ticks = _spread_ticks(exit_row)
    entry_slip_floor = max(float(min_slippage_ticks), float(spread_multiplier) * entry_spread_ticks)
    exit_slip_floor = max(float(min_slippage_ticks), float(spread_multiplier) * exit_spread_ticks)
    realized_entry_cost = max(float(entry_fill.get('slippage_pips', 0.0) or 0.0), entry_slip_floor) + commission_one_way_pips
    realized_exit_cost = max(float(exit_fill.get('slippage_pips', 0.0) or 0.0), exit_slip_floor) + commission_one_way_pips
    dynamic_cost_pips = max(float(raw_pnl_pips - fill_pnl_pips), 0.0)
    total_cost_pips = max(dynamic_cost_pips, realized_entry_cost + realized_exit_cost, floor_cost)
    return {
        'used_dynamic_fill': True,
        'entry_fill': entry_fill,
        'exit_fill': exit_fill,
        'fill_pnl_pips': float(fill_pnl_pips),
        'dynamic_cost_pips': float(total_cost_pips),
        'net_pnl_pips': float(raw_pnl_pips - total_cost_pips),
    }


def _simulate_trade_path(
    entry_idx: int,
    direction: str,
    prices: np.ndarray,
    horizons: np.ndarray,
    micro_atr: np.ndarray,
    tick_size: float,
    opens: np.ndarray | None = None,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    direction_threshold_ticks: float = 1.0,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_horizon_steps: int | None = None,
    replay_horizon_steps: int | None = None,
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
    if replay_horizon_steps is not None and int(replay_horizon_steps) > 0:
        horizon_steps = int(replay_horizon_steps)
    elif max_horizon_steps is not None and int(max_horizon_steps) > 0:
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

    opens_arr = np.asarray(opens if opens is not None else prices, dtype=np.float64)
    highs_arr = np.asarray(highs if highs is not None else prices, dtype=np.float64)
    lows_arr = np.asarray(lows if lows is not None else prices, dtype=np.float64)
    future_prices = np.asarray(prices[entry_idx + 1:exit_cap_idx + 1], dtype=np.float64)
    future_highs = np.asarray(highs_arr[entry_idx + 1:exit_cap_idx + 1], dtype=np.float64)
    future_lows = np.asarray(lows_arr[entry_idx + 1:exit_cap_idx + 1], dtype=np.float64)
    future_opens = np.asarray(opens_arr[entry_idx + 1:exit_cap_idx + 1], dtype=np.float64)
    if future_prices.size == 0:
        return None

    exit_idx = exit_cap_idx
    exit_reason = 'horizon'
    exit_price = float(prices[exit_idx])
    for offset, future_price in enumerate(future_prices, start=1):
        h = float(future_highs[offset - 1]) if np.isfinite(future_highs[offset - 1]) else float(future_price)
        l = float(future_lows[offset - 1]) if np.isfinite(future_lows[offset - 1]) else float(future_price)
        o = float(future_opens[offset - 1]) if np.isfinite(future_opens[offset - 1]) else float(future_price)
        if direction == 'LONG':
            hit_tp = h >= tp_level
            hit_sl = l <= sl_level
            if hit_tp and hit_sl:
                exit_idx = entry_idx + offset
                if abs(tp_level - o) <= abs(o - sl_level):
                    exit_reason = 'tp'; exit_price = float(tp_level)
                else:
                    exit_reason = 'sl'; exit_price = float(sl_level)
                break
            if hit_tp:
                exit_idx = entry_idx + offset; exit_reason = 'tp'; exit_price = float(tp_level); break
            if hit_sl:
                exit_idx = entry_idx + offset; exit_reason = 'sl'; exit_price = float(sl_level); break
        else:
            hit_tp = l <= tp_level
            hit_sl = h >= sl_level
            if hit_tp and hit_sl:
                exit_idx = entry_idx + offset
                if abs(o - tp_level) <= abs(sl_level - o):
                    exit_reason = 'tp'; exit_price = float(tp_level)
                else:
                    exit_reason = 'sl'; exit_price = float(sl_level)
                break
            if hit_tp:
                exit_idx = entry_idx + offset; exit_reason = 'tp'; exit_price = float(tp_level); break
            if hit_sl:
                exit_idx = entry_idx + offset; exit_reason = 'sl'; exit_price = float(sl_level); break

    price_return   = (exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)
    if direction == 'LONG':
        path_moves = future_highs - entry_price
        adverse_moves = future_lows - entry_price
    else:
        path_moves = entry_price - future_lows
        adverse_moves = entry_price - future_highs
    favourable_move = float(np.max(path_moves)) if path_moves.size else 0.0
    adverse_move    = float(np.min(adverse_moves)) if adverse_moves.size else 0.0

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
    return load_feature_artifact(path)


def _parse_freq_to_bar_minutes(freq: str | None) -> float | None:
    """Parse prepare_day_trading freq like '5min', '15min', '1h' → minutes per bar."""
    if freq is None:
        return None
    s = str(freq).strip().lower().replace(' ', '')
    if not s:
        return None
    try:
        if s.endswith('min'):
            v = float(s[:-3] or 0)
            return v if v > 0 else None
        if s.endswith('h'):
            v = float(s[:-1] or 0) * 60.0
            return v if v > 0 else None
        if s.endswith('d'):
            v = float(s[:-1] or 0) * 1440.0
            return v if v > 0 else None
    except ValueError:
        return None
    return None


def _day_trading_manifest_path(data_arg: str) -> str | None:
    """path/to/day_trading_features.parquet → path/to/day_trading_manifest.json"""
    try:
        p = os.path.abspath(str(data_arg))
        root = os.path.dirname(p) if os.path.isfile(p) else p
        cand = os.path.join(root, 'day_trading_manifest.json')
        return cand if os.path.isfile(cand) else None
    except Exception:
        return None


def _read_day_trading_manifest_meta(manifest_path: str | None) -> tuple[int | None, float | None, str | None]:
    """Returns (horizon_bars, bar_minutes, freq_str) from day_trading_manifest.json."""
    payload = _load_json_if_exists(manifest_path or '')
    if not payload or str(payload.get('mode', '')).strip().lower() != 'day_trading':
        return None, None, None
    hb = payload.get('horizon_bars')
    try:
        hb_i = int(hb) if hb is not None else None
    except (TypeError, ValueError):
        hb_i = None
    if hb_i is not None and hb_i <= 0:
        hb_i = None
    fq = payload.get('freq')
    fq_s = str(fq).strip() if fq is not None else None
    bm = _parse_freq_to_bar_minutes(fq_s)
    return hb_i, bm, fq_s


def _ensure_trade_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    شموع الـ day trading تخرج عادةً بـ OHLC (`close`) فقط، بينما `feature_schema_v19`
    يمرّر `price` للباك تست. إن غاب `price` أو كان صفرًا، يُعبَّأ من `close` حتى تعمل
    محاكاة المسار (وإلا trade_path=None و skipped_trade_replays يرتفع بدون صفقات).
    """
    if df is None or df.empty:
        return df
    out = df
    use_close = False
    if 'close' not in out.columns:
        return out
    close = pd.to_numeric(out['close'], errors='coerce')
    if 'price' not in out.columns:
        use_close = True
    else:
        pr = pd.to_numeric(out['price'], errors='coerce')
        if (not np.isfinite(pr.to_numpy(dtype=np.float64)).any()) or float(pr.fillna(0.0).abs().max()) < 1e-12:
            use_close = True
    if use_close:
        out = out.copy()
        out['price'] = close
        print('  [backtest] price column: filled from `close` (day-trading OHLC artifact)', flush=True)
    return out


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
    dataset_manifest = load_artifact_manifest(csv_path) or {}
    return model_manifest, dataset_manifest


def _load_model_training_window(models_dir: str) -> dict:
    manifest = _load_json_if_exists(os.path.join(models_dir, 'manifest.json')) or {}
    extra = manifest.get('extra', {}) or {}
    training_window = extra.get('training_window', {}) or {}
    if training_window:
        return training_window
    source_contract = extra.get('source_contract', {}) or {}
    split_time = source_contract.get('split_time')
    return {'holdout_start_time': split_time} if split_time else {}


def _build_backtest_window_mask(
    df: pd.DataFrame,
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> pd.Series:
    if 'ts_event' not in df.columns or (start_ts is None and end_ts is None):
        return pd.Series(True, index=df.index)

    ts = pd.to_datetime(df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    mask = pd.Series(True, index=df.index)

    if start_ts is not None:
        start = pd.to_datetime(start_ts, utc=True, errors='coerce')
        if not pd.isna(start):
            mask &= ts >= start.tz_localize(None)

    if end_ts is not None:
        end = pd.to_datetime(end_ts, utc=True, errors='coerce')
        if not pd.isna(end):
            mask &= ts < end.tz_localize(None)

    return mask


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
    backtest_csv = os.path.abspath(resolve_artifact_root(csv_path) if os.path.isdir(csv_path) else csv_path)
    source_dataset_id = str(source_contract.get('dataset_id') or '')
    backtest_dataset_id = str((dataset_manifest.get('extra', {}) or {}).get('dataset_id') or '')
    source_split_time = str(source_contract.get('split_time') or '')
    source_schema_version = str(source_contract.get('schema_version') or '')
    backtest_schema_version = str((dataset_manifest.get('extra', {}) or {}).get('schema_version') or '')

    dataset_slice = pd.Series(df.get('dataset_slice', pd.Series([], dtype='object'))).astype(str).str.lower()
    is_pure_holdout = bool(len(dataset_slice) > 0 and dataset_slice.isin(['holdout']).all())

    missing_contract_bits = []
    if not model_manifest:
        missing_contract_bits.append('model manifest')
    if not dataset_manifest:
        missing_contract_bits.append('dataset manifest')
    if not source_csv:
        missing_contract_bits.append('source_contract.source_csv')
    if not source_dataset_id:
        missing_contract_bits.append('source_contract.dataset_id')
    if not source_split_time:
        missing_contract_bits.append('source_contract.split_time')
    if not source_schema_version:
        missing_contract_bits.append('source_contract.schema_version')
    if not backtest_dataset_id:
        missing_contract_bits.append('dataset_manifest.extra.dataset_id')
    if not backtest_schema_version:
        missing_contract_bits.append('dataset_manifest.extra.schema_version')

    if missing_contract_bits:
        info['allowed'] = False
        info['reason'] = 'missing_oos_contract'
        raise ValueError(
            '❌ Refusing labeled backtest because OOS contract metadata is incomplete: '
            + ', '.join(missing_contract_bits)
            + '. Rebuild the dataset/models with current manifests or pass '
            '--allow_in_sample_data_override intentionally.'
        )

    same_path = bool(source_csv) and source_csv == backtest_csv
    same_dataset = bool(source_dataset_id) and source_dataset_id == backtest_dataset_id

    if (same_path or same_dataset) and not is_pure_holdout:
        split_ts = pd.to_datetime(source_split_time, utc=True, errors='coerce')
        row_ts = pd.to_datetime(df.get('ts_event', pd.Series(dtype='datetime64[ns]')), utc=True, errors='coerce').dt.tz_localize(None).dropna()
        if not pd.isna(split_ts) and len(row_ts):
            split_ts = split_ts.tz_localize(None)
            if row_ts.min() >= split_ts:
                info['reason'] = 'post_split_window_only'
                return info
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
    mask = _build_backtest_window_mask(df, start_ts=start_ts, end_ts=end_ts)
    return df.loc[mask].reset_index(drop=True)


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
    source_kind = 'explicit file' if explicit_path else 'default artifact'
    raise ValueError(
        f'❌ visual embeddings rows ({len(vis)}) do not match CSV rows ({n}) '
        f'or directional-event rows ({directional_rows}) for {source_kind} {path}. '
        'Refusing silent truncate/pad because this usually means the embeddings '
        'were produced from a different dataset or are missing their matching '
        'visual_coverage_v19.npy sidecar.'
    )


def _load_lob_entrydata(
    df: pd.DataFrame,
    *,
    lob_path: str | None = None,
    lob_ts_path: str | None = None,
    lob_map_path: str | None = None,
    data_path: str | None = None,
    models_dir: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Load raw LOB tensors and row->tensor mapping for EntryData replay."""
    diagnostics = {
        'entrydata_enabled': False,
        'lob_path': None,
        'lob_map_source': None,
        'rows_with_lob_tensor': 0,
    }

    candidate_lob = lob_path
    if candidate_lob is None and data_path:
        try:
            root = resolve_artifact_root(data_path)
            p = os.path.join(root, 'lob_tensors.npy')
            candidate_lob = p if os.path.exists(p) else None
            if lob_ts_path is None:
                ts_p = os.path.join(root, 'lob_tensor_timestamps.npy')
                lob_ts_path = ts_p if os.path.exists(ts_p) else None
        except Exception:
            candidate_lob = None

    if not candidate_lob or not os.path.exists(candidate_lob):
        return None, None, diagnostics

    tensors = np.load(candidate_lob, mmap_mode='r')
    n = len(df)
    row_to_tensor = None

    if lob_map_path and os.path.exists(lob_map_path):
        row_to_tensor = np.asarray(np.load(lob_map_path), dtype=np.int32).reshape(-1)
        diagnostics['lob_map_source'] = lob_map_path
    elif models_dir:
        default_map = os.path.join(models_dir, 'row_to_lob_tensor_id_v19.npy')
        if os.path.exists(default_map):
            candidate = np.asarray(np.load(default_map), dtype=np.int32).reshape(-1)
            if len(candidate) == n:
                row_to_tensor = candidate
                diagnostics['lob_map_source'] = default_map

    if row_to_tensor is None and 'lob_tensor_id' in df.columns:
        row_to_tensor = pd.to_numeric(df['lob_tensor_id'], errors='coerce').fillna(-1).astype(np.int32).values
        diagnostics['lob_map_source'] = 'df.lob_tensor_id'

    if row_to_tensor is None and lob_ts_path and os.path.exists(lob_ts_path):
        ts_raw = np.load(lob_ts_path)
        lob_ts = pd.to_datetime(ts_raw.astype(np.int64), unit='ns', utc=True, errors='coerce').tz_localize(None)
        n_lob = min(len(lob_ts), len(tensors))
        row_ts = pd.to_datetime(
            df.get('ts_event', pd.Series([pd.NaT] * n)),
            utc=True,
            errors='coerce',
        ).dt.tz_localize(None)
        row_df = pd.DataFrame({'ts_event': row_ts, 'row_idx': np.arange(n, dtype=np.int32)})
        lob_df = pd.DataFrame({
            'ts_event': pd.Series(lob_ts).iloc[:n_lob],
            'tensor_idx': np.arange(n_lob, dtype=np.int32),
        }).dropna(subset=['ts_event']).sort_values('ts_event')
        merged = pd.merge_asof(
            row_df.sort_values('ts_event'),
            lob_df,
            on='ts_event',
            direction='backward',
            tolerance=pd.Timedelta('10min'),
        ).sort_values('row_idx')
        row_to_tensor = merged['tensor_idx'].fillna(-1).astype(np.int32).values
        diagnostics['lob_map_source'] = f'timestamp_asof:{lob_ts_path}'

    if row_to_tensor is None:
        return tensors, None, diagnostics

    if len(row_to_tensor) != n:
        print(
            "  ⚠️ EntryData LOB map row mismatch: "
            f"map={len(row_to_tensor):,}, rows={n:,}; disabling raw LOB replay.",
            flush=True,
        )
        return tensors, None, diagnostics

    valid = (row_to_tensor >= 0) & (row_to_tensor < len(tensors))
    diagnostics.update(
        {
            'entrydata_enabled': bool(valid.any()),
            'lob_path': str(candidate_lob),
            'rows_with_lob_tensor': int(valid.sum()),
            'rows_total': int(n),
            'coverage_ratio': float(valid.mean()) if n else 0.0,
        }
    )
    if valid.any():
        print(
            "  ✅ EntryData raw LOB enabled: "
            f"{int(valid.sum()):,}/{n:,} rows ({float(valid.mean()):.1%}) "
            f"| source={diagnostics['lob_map_source']}",
            flush=True,
        )
    return tensors, row_to_tensor, diagnostics


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
    scoring_mask: np.ndarray | None = None,
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

    if scoring_mask is None:
        scoring_mask = np.ones(n_rows, dtype=bool)
    else:
        scoring_mask = np.asarray(scoring_mask, dtype=bool)
        if len(scoring_mask) != n_rows:
            scoring_mask = np.ones(n_rows, dtype=bool)

    event_flag = pd.to_numeric(df.get('event_flag', 0), errors='coerce').fillna(0).astype(np.int8)
    train_event_flag = pd.to_numeric(df.get('train_event_flag', event_flag), errors='coerce').fillna(0).astype(np.int8)
    bias_label = pd.to_numeric(df.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int8)
    directional_mask = bias_label.isin([0, 1]).to_numpy(dtype=bool)
    full_mask = np.asarray(scoring_mask, dtype=bool)
    event_mask = (event_flag == 1).to_numpy(dtype=bool) & full_mask
    train_event_mask = (train_event_flag == 1).to_numpy(dtype=bool) & full_mask
    directional_train_event_mask = train_event_mask & directional_mask
    directional_mask = directional_mask & full_mask

    tradeable_mask = np.zeros(n_rows, dtype=bool)
    executed_mask = np.zeros(n_rows, dtype=bool)
    active_idx = np.flatnonzero(full_mask)
    if results_df is not None and len(results_df):
        mapped_idx = None
        if 'idx' in results_df.columns:
            idx_values = pd.to_numeric(results_df['idx'], errors='coerce').dropna().astype(int).to_numpy()
            if len(idx_values) == len(results_df):
                mapped_idx = idx_values
        if mapped_idx is None and len(results_df) == len(active_idx):
            mapped_idx = active_idx
        if mapped_idx is not None and len(mapped_idx):
            valid = (mapped_idx >= 0) & (mapped_idx < n_rows)
            mapped_idx = mapped_idx[valid]
            tradeable_values = pd.to_numeric(results_df.get('tradeable', 0), errors='coerce').fillna(0).astype(bool).to_numpy()[valid]
            executed_values = pd.to_numeric(results_df.get('executed', 0), errors='coerce').fillna(0).astype(bool).to_numpy()[valid]
            tradeable_mask[mapped_idx] = tradeable_values
            executed_mask[mapped_idx] = executed_values

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
    if not np.all(full_mask):
        context_total, context_covered, context_ratio = _coverage_counts(np.ones(n_rows, dtype=bool), covered_mask)
        diagnostics.update({
            'context_rows_total': int(context_total),
            'context_rows_with_visual': int(context_covered),
            'context_coverage_ratio': float(context_ratio),
        })

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
    commission_per_side: float = 0.0,
    min_spread_ticks: float = 1.0,
    min_slippage_ticks: float = 1.0,
    spread_multiplier: float = 0.5,
    max_size: int = 5,
    starting_equity: float = 100000.0,
    latency_rows: int = 1,
    max_daily_loss_pct: float = 0.02,
    direction_threshold_ticks: float = 1.0,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_horizon_steps: int | None = None,
    replay_horizon_steps: int | None = None,
    allow_oracle_forward_return: bool = False,
    single_position_only: bool = True,
    cooldown_rows: int = 0,
    visual_diagnostics: dict | None = None,
    score_start_ts: str | None = None,
    score_end_ts: str | None = None,
    scenario_name: str = 'base',
    relax_policy_ev: bool = False,
    policy_min_edge: float | None = None,
    skip_event_gate: bool = False,
    long_only: bool = False,
    lob_tensors: np.ndarray | None = None,
    row_to_lob_tensor: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    engine = V19PredictionEngine(
        models_dir,
        run_mode='backtest',
        policy_require_positive_ev=(not relax_policy_ev),
        policy_edge_prob_override=policy_min_edge,
        skip_event_gate=skip_event_gate,
    )
    engine.reset_state()
    engine.loss_guard.max_daily_loss_pct = float(max_daily_loss_pct)
    engine.loss_guard.update_equity(starting_equity)
    _assert_single_contract_df(df, context='run_causal_backtest_input')
    source_has_labels = 'bias_label' in df.columns
    source_has_fwd = 'forward_return' in df.columns
    print(f"  [backtest] Building replay feature frame ({len(df):,} rows, input_scaled={input_scaled})...", flush=True)
    t_pf0 = time.perf_counter()
    replay_df = engine.factory.prepare_frame(df, already_scaled=input_scaled, include_meta=True)
    print(f"  [backtest] replay_df ready: {replay_df.shape[0]:,} x {replay_df.shape[1]} in {time.perf_counter() - t_pf0:.1f}s", flush=True)
    price_arr = _series_or_default(replay_df, 'price', 0.0, dtype=np.float64).values
    open_arr = (
        _series_or_default(replay_df, 'open', 0.0, dtype=np.float64).values
        if 'open' in replay_df.columns else price_arr.copy()
    )
    high_arr = (
        _series_or_default(replay_df, 'high', 0.0, dtype=np.float64).values
        if 'high' in replay_df.columns else price_arr.copy()
    )
    low_arr = (
        _series_or_default(replay_df, 'low', 0.0, dtype=np.float64).values
        if 'low' in replay_df.columns else price_arr.copy()
    )
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
    scoring_mask = _build_backtest_window_mask(
        replay_df,
        start_ts=score_start_ts,
        end_ts=score_end_ts,
    ).to_numpy(dtype=bool)
    if not bool(scoring_mask.any()):
        raise ValueError('❌ نافذة التقييم المطلوبة لا تحتوي أي صفوف داخل replay_df.')

    n_replay = len(replay_df)
    _rh = f" | replay_horizon_steps={int(replay_horizon_steps)}" if replay_horizon_steps else ""
    print(
        f"  [backtest] Causal replay: {n_replay:,} rows | "
        f"scoring_mask={int(scoring_mask.sum()):,}"
        f"{_rh}"
        f"{' | long_only=True (SHORT signals not executed)' if long_only else ''}",
        flush=True,
    )
    progress_every = min(250, max(50, n_replay // 30))
    sl = int(getattr(engine, 'seq_len', 0) or 0)
    print(
        f"  [backtest] Replay log every {progress_every} rows "
        f"(after sequence warm-up rows 0 .. {max(sl - 1, 0)} each step carries full inference; wait for first '{progress_every}' line).",
        flush=True,
    )

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

    for i in range(n_replay):
        row = replay_df.iloc[i].to_dict()
        if i > 0 and i % progress_every == 0:
            print(f"  [backtest] progress: {i:,}/{n_replay:,}", flush=True)
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

        lob_tensor = None
        if row_to_lob_tensor is not None and lob_tensors is not None:
            tensor_idx = int(row_to_lob_tensor[i])
            if 0 <= tensor_idx < len(lob_tensors):
                lob_tensor = np.asarray(lob_tensors[tensor_idx], dtype=np.float32)
                row['lob_tensor_id'] = tensor_idx
        visual = None if lob_tensor is not None else (visual_embeddings[i] if visual_embeddings.size else None)
        pred = engine.predict_step(
            row,
            visual_embedding=visual,
            meta_override=meta_features[i] if meta_features is not None else None,
            lob_tensor=lob_tensor,
            ts=ts,
            already_scaled=True,
        )

        if pred.get('reason', '').startswith('Warming up'):
            continue

        row_in_score = bool(scoring_mask[i])
        if not row_in_score:
            continue

        pred['idx'] = i
        pred['ts_event'] = str(ts) if ts is not None and not pd.isna(ts) else ''
        pred['price'] = _safe_float(row.get('price', 0.0))
        if lob_tensor is not None:
            pred['lob_tensor_id'] = int(row.get('lob_tensor_id', -1))
            pred['entrydata_source'] = 'raw_lob_tensor'

        if has_labels:
            true_bias = int(row.get('bias_label', 2))
            pred['true_bias'] = true_bias
            pred['true_label'] = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}.get(true_bias, '?')
            pred['correct'] = (pred.get('bias_idx') == true_bias)

        # Count rows where the engine did not emit a tradeable directional signal
        # (not every non-empty policy `reason` — approved trades use reasons too).
        if not bool(pred.get('tradeable', False)):
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
            if long_only and direction == 'SHORT':
                pred['trade_skip_reason'] = 'long_only_backtest'
                results.append(pred)
                continue
            size = int(position_size_from_prediction(
                pred,
                base_size=1,
                max_size=max_size,
                fraction=0.25,
            ))
            if size <= 0:
                pred['trade_skip_reason'] = 'position_size_zero'
                pred['position_size'] = 0
                results.append(pred)
                continue
            entry_idx = min(i + max(int(latency_rows), 0), len(replay_df) - 1)
            entry_row_data = replay_df.iloc[entry_idx].to_dict() if 0 <= entry_idx < n_replay else row
            trade_path = _simulate_trade_path(
                entry_idx=entry_idx,
                direction=direction,
                prices=price_arr,
                horizons=horizon_arr,
                micro_atr=micro_atr_arr,
                tick_size=tick_size,
                opens=open_arr,
                highs=high_arr,
                lows=low_arr,
                direction_threshold_ticks=direction_threshold_ticks,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                max_horizon_steps=max_horizon_steps,
                replay_horizon_steps=replay_horizon_steps,
                row_data=entry_row_data,  # FIX: تمرير بيانات الجدران للـ DynamicTargetManager
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
            exit_idx = int(trade_path['exit_idx'])
            exit_ts = ts_arr.iloc[exit_idx] if exit_idx < len(ts_arr) else pd.NaT
            exit_row = replay_df.iloc[exit_idx].to_dict() if 0 <= exit_idx < n_replay else {}
            fill_pricing = _realized_fill_pricing(
                entry_row=entry_row_data,
                exit_row=exit_row,
                direction=direction,
                size=size,
                raw_pnl_pips=raw_pnl_pips,
                tick_size=tick_size,
                round_trip_cost_pips=round_trip_cost_pips,
                tick_value=tick_value,
                commission_per_side=commission_per_side,
                min_spread_ticks=min_spread_ticks,
                min_slippage_ticks=min_slippage_ticks,
                spread_multiplier=spread_multiplier,
            )
            net_pnl_pips = float(fill_pricing['net_pnl_pips'])
            net_pnl_dollars = net_pnl_pips * tick_value * size

            result = 'WIN' if net_pnl_pips > 0 else ('LOSE' if net_pnl_pips < 0 else 'FLAT')
            entry_fill = fill_pricing.get('entry_fill', {}) or {}
            exit_fill = fill_pricing.get('exit_fill', {}) or {}
            entry_price = _safe_float(entry_fill.get('fill_price', row.get('price', 0.0)))
            exit_price = _safe_float(exit_fill.get('fill_price', trade_path.get('exit_price', 0.0)))
            trade = {
                'idx': i,
                'ts_event': pred['ts_event'],
                'entry_idx': int(entry_idx),
                'dir': direction,
                'confidence': round(_safe_float(pred.get('confidence', 0.0)), 4),
                'size': size,
                'ep': round(float(entry_price), 6),
                'xp': round(float(exit_price), 6),
                'exit_idx': exit_idx,
                'exit_ts': '' if pd.isna(exit_ts) else str(exit_ts),
                'exit_reason': str(trade_path['exit_reason']),
                'dur_steps': int(trade_path['hold_steps']),
                'dur_min': int(trade_path['hold_steps']),
                'pips': round(net_pnl_pips, 4),
                'raw_pnl_pips': round(raw_pnl_pips, 4),
                'cost_pips': round(float(fill_pricing.get('dynamic_cost_pips', round_trip_cost_pips)), 4),
                'fill_pnl_pips': round(float(fill_pricing.get('fill_pnl_pips', raw_pnl_pips)), 4),
                'used_dynamic_fill_cost': bool(fill_pricing.get('used_dynamic_fill', False)),
                'entry_slippage_pips': round(float(entry_fill.get('slippage_pips', 0.0) or 0.0), 4),
                'exit_slippage_pips': round(float(exit_fill.get('slippage_pips', 0.0) or 0.0), 4),
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
            pred['entry_idx'] = int(entry_idx)
            pred['pending_exit_idx'] = exit_idx
            pred['pending_exit_ts'] = '' if pd.isna(exit_ts) else str(exit_ts)
            pred['exit_reason'] = str(trade_path['exit_reason'])
            pred['pnl_source'] = pnl_source
            pred['cost_pips'] = round(float(fill_pricing.get('dynamic_cost_pips', round_trip_cost_pips)), 4)
            pred['used_dynamic_fill_cost'] = bool(fill_pricing.get('used_dynamic_fill', False))

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
        scoring_mask=scoring_mask,
    )

    mdd_abs, mdd_pct = _equity_metrics(equity_curve)
    trade_pnls = trades_df['pnl'].tolist() if not trades_df.empty else []
    wins = trades_df[trades_df['pnl'] > 0] if not trades_df.empty else trades_df
    losses = trades_df[trades_df['pnl'] < 0] if not trades_df.empty else trades_df

    summary = {
        'predictions': int(len(results_df)),
        'trades': int(len(trades_df)),
        'scenario': str(scenario_name),
        'tradeable_signals': int(results_df['tradeable'].sum()) if 'tradeable' in results_df.columns else 0,
        **_directional_metrics(results_df),
        'event_gate_rate': round(float(results_df['event_gate_passed'].mean()), 4)
            if 'event_gate_passed' in results_df.columns and len(results_df) else 0.0,
        'win_rate': round(float((trades_df['pnl'] > 0).mean()), 4) if len(trades_df) else 0.0,
        'total_pnl_dollars': round(float(trades_df['pnl'].sum()), 2) if len(trades_df) else 0.0,
        'avg_trade_pnl_dollars': round(float(trades_df['pnl'].mean()), 2) if len(trades_df) else 0.0,
        'avg_trade_pips': round(float(trades_df['pips'].mean()), 4) if len(trades_df) else 0.0,
        'avg_trade_expectancy_dollars': round(float(trades_df['pnl'].mean()), 2) if len(trades_df) else 0.0,
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
        'latency_rows': int(max(latency_rows, 0)),
        'commission_per_side': float(commission_per_side),
        'min_spread_ticks': float(min_spread_ticks),
        'min_slippage_ticks': float(min_slippage_ticks),
        'spread_multiplier': float(spread_multiplier),
        'max_daily_loss_pct': float(max_daily_loss_pct),
        'context_rows': int(len(replay_df)),
        'scored_rows': int(scoring_mask.sum()),
        'score_start_ts': None if score_start_ts is None else str(score_start_ts),
        'score_end_ts': None if score_end_ts is None else str(score_end_ts),
        'visual_coverage': float(visual_diag.get('coverage_ratio', 0.0)),
        'visual_diagnostics': visual_diag,
        'entrydata_raw_lob_rows': int(
            (results_df.get('entrydata_source', pd.Series(dtype=object)) == 'raw_lob_tensor').sum()
        ) if len(results_df) else 0,
        'long_only': bool(long_only),
        'max_horizon_steps': None if max_horizon_steps is None else int(max_horizon_steps),
        'replay_horizon_steps': None if replay_horizon_steps is None else int(replay_horizon_steps),
    }
    if 'true_bias' in results_df.columns and 'direction_probs' in results_df.columns and len(results_df):
        directional = results_df[results_df['true_bias'].isin([0, 1])].copy()
        if len(directional):
            p_long = directional['direction_probs'].apply(
                lambda x: float((x or {}).get('LONG', 0.0)) if isinstance(x, dict) else 0.0
            ).to_numpy(dtype=np.float64)
            y_true = directional['true_bias'].to_numpy(dtype=np.int32)
            summary['brier_score'] = float(np.mean((p_long - (y_true == 0).astype(np.float64)) ** 2))
            summary['ece'] = _binary_ece(y_true, p_long)
        else:
            summary['brier_score'] = 0.0
            summary['ece'] = 0.0
    else:
        summary['brier_score'] = 0.0
        summary['ece'] = 0.0

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


_STAGE1_SOFT = 'soft_label'
_SOFT_BUNDLE_FALLBACK_POLICY_MIN_EDGE = 0.52  # mirrors predict_v19.V19PredictionEngine._base_fallback_min_edge()


def _infer_soft_bundle_default_policy_min_edge(models_dir: str) -> float | None:
    """When stage-1 trains on soft_label, pseudo-probs use a softer scale than cost-derived ~2/3 thresholds."""
    soft = False
    for fname in ('catboost_classes_v19.json', 'xgboost_classes_v19.json'):
        path = os.path.join(models_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                payload = json.load(f)
            if str(payload.get('stage1_target', '') or '').strip().lower() == _STAGE1_SOFT:
                soft = True
                break
        except Exception:
            continue
    if not soft:
        return None
    return float(_SOFT_BUNDLE_FALLBACK_POLICY_MIN_EDGE)


def main():
    p = argparse.ArgumentParser(description='Causal replay backtester for QuantSystem V19')
    p.add_argument('--data', '--csv', dest='data', required=True, help='stage1 artifact dir/manifest/parquet')
    p.add_argument('--models', default='outputs_v19', help='trained V19 models directory')
    p.add_argument('--output', default='outputs_v19', help='backtest output directory')
    p.add_argument('--visual_npy', default=None, help='optional row-aligned visual embeddings file')
    p.add_argument('--meta_npy', default=None, help='optional row-aligned stage-1 meta features file')
    p.add_argument('--lob', default=None, help='optional raw lob_tensors.npy for EntryData replay')
    p.add_argument('--lob_ts', default=None, help='optional lob_tensor_timestamps.npy for timestamp EntryData alignment')
    p.add_argument('--lob_map_npy', default=None, help='optional row-aligned row_to_lob_tensor_id_v19.npy')
    p.add_argument(
        '--live_like_runtime_inputs',
        action='store_true',
        help=(
            'disable auto OOF meta/visual artifacts so backtest uses runtime-computed '
            'tree/regime surfaces (closest to live behavior on the same feature rows)'
        ),
    )
    p.add_argument(
        '--disable_auto_oof_meta',
        action='store_true',
        help='do not auto-load meta_features_oof_v19.npy when --meta_npy is omitted',
    )
    p.add_argument(
        '--disable_auto_visual_artifact',
        action='store_true',
        help='do not auto-load cached visual embeddings when --visual_npy is omitted',
    )
    p.add_argument('--allow_in_sample_live_meta_override', action='store_true',
                   help='dangerous: allow explicit live/final-fit meta features on labeled backtest data')
    p.add_argument('--allow_in_sample_data_override', action='store_true',
                   help='dangerous: allow backtesting directly on the training dataset / non-holdout labeled artifact')
    p.add_argument('--input_scaled', action='store_true',
                   help='set when the artifact is already scaled like final stage1 features')
    p.add_argument('--tick_size', type=float, default=0.0001)
    p.add_argument('--tick_value', type=float, default=10.0)
    p.add_argument('--round_trip_cost_pips', type=float, default=1.0)
    p.add_argument('--commission_per_side', type=float, default=0.0)
    p.add_argument('--min_spread_ticks', type=float, default=1.0)
    p.add_argument('--min_slippage_ticks', type=float, default=1.0)
    p.add_argument('--spread_multiplier', type=float, default=0.5)
    p.add_argument('--max_size', type=int, default=5)
    p.add_argument('--starting_equity', type=float, default=100000.0)
    p.add_argument('--latency_rows', type=int, default=1)
    p.add_argument('--max_daily_loss_pct', type=float, default=0.02)
    p.add_argument('--direction_threshold_ticks', type=float, default=1.0)
    p.add_argument('--tp_mult', type=float, default=1.5)
    p.add_argument('--sl_mult', type=float, default=1.0)
    p.add_argument('--start_ts', default=None, help='optional inclusive start timestamp for the backtest window')
    p.add_argument('--end_ts', default=None, help='optional exclusive end timestamp for the backtest window')
    p.add_argument('--max_horizon_steps', type=int, default=0,
                   help='optional cap on replay horizon in rows; 0 uses label_horizon_steps as-is (cannot extend below label)')
    p.add_argument(
        '--fixed_horizon',
        type=float,
        default=None,
        help=(
            'override path-replay holding window in minutes (day bars: set --bar_minutes, e.g. 60 min / 5 min bar => 12 steps). '
            'Replaces label_horizon_steps for TP/SL path replay.'
        ),
    )
    p.add_argument(
        '--bar_minutes',
        type=float,
        default=None,
        help=(
            'minutes per bar when using --fixed_horizon; '
            'default: read from day_trading_manifest.json freq next to --data, else 5'
        ),
    )
    p.add_argument('--allow_oracle_forward_return', action='store_true',
                   help='dangerous: fall back to stored forward_return when no causal replay window is available')
    p.add_argument('--disable_single_position_only', action='store_true',
                   help='allow overlapping trades; default keeps one active position at a time')
    p.add_argument('--cooldown_rows', type=int, default=0,
                   help='rows to wait after closing a trade before opening a new one')
    p.add_argument('--relax_policy_ev', action='store_true',
                   help='cost-aware policy: do not require positive EV (recommended when trades=0: placeholder symmetric win/loss in policy vs cost)')
    p.add_argument('--policy_min_edge', type=float, default=None,
                   help=(
                       'override LONG/SHORT prob threshold vs policy-derived cost floor; '
                       'omit to auto-use 0.52 for soft_label CatBoost/XGB bundles (matches live predict fallback)'
                   ))
    p.add_argument('--skip_event_gate', action='store_true',
                   help='disable EventGate for this run (diagnostic only)')
    p.add_argument(
        '--long_only',
        action='store_true',
        help='open only LONG positions; SHORT signals stay in logs (trade_skip_reason=long_only_backtest)',
    )
    args = p.parse_args()

    inferred_policy_edge = _infer_soft_bundle_default_policy_min_edge(args.models)
    if args.policy_min_edge is None and inferred_policy_edge is not None:
        args.policy_min_edge = inferred_policy_edge
        print(
            f"  [backtest] soft_label artifacts: policy_min_edge={args.policy_min_edge} "
            f"(aligned with predict_v19 soft pseudo-prob scale; override with --policy_min_edge)",
            flush=True,
        )
        if not args.relax_policy_ev:
            print(
                '  [backtest] hint: symmetric placeholder win/loss in decision_policy often makes '
                'EV>0 impossible at any probability — if you see zero trades, add --relax_policy_ev',
                flush=True,
            )

    manifest_path = _day_trading_manifest_path(args.data)
    mt_h, mt_bm_manifest, mt_freq = _read_day_trading_manifest_meta(manifest_path)
    bar_minutes_eff = float(args.bar_minutes) if args.bar_minutes is not None else None
    if bar_minutes_eff is None:
        bar_minutes_eff = float(mt_bm_manifest) if mt_bm_manifest is not None else 5.0
        if mt_bm_manifest is not None:
            print(
                f"  [backtest] bar_minutes={bar_minutes_eff:g} (from day_trading_manifest freq={mt_freq!r})",
                flush=True,
            )
        else:
            print(f"  [backtest] bar_minutes={bar_minutes_eff:g} (default; no day_trading_manifest freq)", flush=True)

    replay_horizon_steps = None
    if args.fixed_horizon is not None and float(args.fixed_horizon) > 0:
        bm = max(float(bar_minutes_eff), 1e-6)
        replay_horizon_steps = max(1, int(round(float(args.fixed_horizon) / bm)))
        print(
            f"  [backtest] fixed_horizon: {float(args.fixed_horizon)} min / {bm} min per bar "
            f"=> replay_horizon_steps={replay_horizon_steps}",
            flush=True,
        )
        if mt_h is not None and int(replay_horizon_steps) != int(mt_h):
            print(
                f"  [backtest] WARNING: replay_horizon_steps={replay_horizon_steps} != manifest horizon_bars={mt_h} "
                f"(labels trained at {mt_h} bars × ~{bar_minutes_eff:g} min ≈ {int(mt_h) * bar_minutes_eff:g} min). "
                f"For aligned train/backtest: omit --fixed_horizon or set "
                f"--fixed_horizon {int(mt_h) * bar_minutes_eff:g} with this bar size, or regenerate data with "
                f"prepare_day_trading --horizon {replay_horizon_steps}.",
                flush=True,
            )
    else:
        if mt_h is not None:
            approx_min = float(mt_h) * float(bar_minutes_eff)
            print(
                f"  [backtest] day_trading manifest: horizon_bars={mt_h}, freq={mt_freq!r} "
                f"(~{bar_minutes_eff:g} min/bar, ~{approx_min:g} min label horizon). "
                f"Path replay uses per-row label_horizon_steps (override with --fixed_horizon minutes).",
                flush=True,
            )

    df_raw = _ensure_trade_price_column(_load_csv(args.data))
    if mt_h is not None and 'label_horizon_steps' in df_raw.columns:
        med = pd.to_numeric(df_raw['label_horizon_steps'], errors='coerce').dropna()
        if len(med):
            med_v = int(round(float(med.median())))
            if med_v != int(mt_h):
                print(
                    f"  [backtest] WARNING: data label_horizon_steps median={med_v} != manifest horizon_bars={mt_h} "
                    f"— use matching prepare_day_trading output or --fixed_horizon.",
                    flush=True,
                )

    model_window = _load_model_training_window(args.models)
    align_start_ts = model_window.get('train_start_time')
    align_end_ts = model_window.get('holdout_end_time_exclusive')
    align_mask = _build_backtest_window_mask(
        df_raw,
        start_ts=align_start_ts,
        end_ts=align_end_ts,
    )
    df_aligned = df_raw.loc[align_mask].reset_index(drop=True)

    start_ts = args.start_ts if args.start_ts is not None else model_window.get('holdout_start_time')
    end_ts = args.end_ts if args.end_ts is not None else model_window.get('holdout_end_time_exclusive')
    eval_mask = _build_backtest_window_mask(
        df_aligned,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    df_score = df_aligned.loc[eval_mask].reset_index(drop=True)
    if df_score.empty:
        raise ValueError('❌ نافذة الباك تست المطلوبة فارغة. راجع start_ts/end_ts أو training_window في manifest.')

    print(
        f"  [backtest] Replay timeline rows: {len(df_aligned):,} | "
        f"holdout/scored window: {len(df_score):,}",
        flush=True,
    )

    oos_guard = _enforce_oos_backtest_guard(
        df_score,
        csv_path=args.data,
        models_dir=args.models,
        allow_in_sample_data_override=args.allow_in_sample_data_override,
    )
    engine = V19PredictionEngine(
        args.models,
        run_mode='backtest',
        policy_require_positive_ev=(not args.relax_policy_ev),
        policy_edge_prob_override=args.policy_min_edge,
        skip_event_gate=args.skip_event_gate,
    )
    disable_auto_meta = bool(args.disable_auto_oof_meta or args.live_like_runtime_inputs)
    disable_auto_visual = bool(args.disable_auto_visual_artifact or args.live_like_runtime_inputs)
    if args.live_like_runtime_inputs:
        print(
            "  [backtest] live_like_runtime_inputs=True → disable auto OOF meta/visual surfaces "
            "(runtime tree/regime inference path).",
            flush=True,
        )
    visual_default_path = None if disable_auto_visual else engine.visual_emb_path
    visual_full = _load_visual_embeddings(
        df_aligned,
        explicit_path=args.visual_npy,
        default_path=visual_default_path,
        expected_dim=len(engine.visual_features),
        models_dir=args.models,
    )

    meta_path = args.meta_npy
    if meta_path is None and not disable_auto_meta:
        default_meta_oof = os.path.join(args.models, 'meta_features_oof_v19.npy')
        if os.path.exists(default_meta_oof):
            meta_path = default_meta_oof

    meta_full = _load_meta_features(
        df_aligned,
        explicit_path=meta_path,
        expected_dim=len(engine.meta_features),
        allow_in_sample_live_override=args.allow_in_sample_live_meta_override,
    )
    lob_tensors, row_to_lob_tensor, entrydata_diag = _load_lob_entrydata(
        df_aligned,
        lob_path=args.lob,
        lob_ts_path=args.lob_ts,
        lob_map_path=args.lob_map_npy,
        data_path=args.data,
        models_dir=args.models,
    )

    _, _, summary = run_causal_backtest(
        df=df_aligned,
        models_dir=args.models,
        output_dir=args.output,
        visual_embeddings=visual_full,
        meta_features=meta_full,
        input_scaled=args.input_scaled,
        lob_tensors=lob_tensors,
        row_to_lob_tensor=row_to_lob_tensor,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        round_trip_cost_pips=args.round_trip_cost_pips,
        commission_per_side=args.commission_per_side,
        min_spread_ticks=args.min_spread_ticks,
        min_slippage_ticks=args.min_slippage_ticks,
        spread_multiplier=args.spread_multiplier,
        max_size=args.max_size,
        starting_equity=args.starting_equity,
        latency_rows=args.latency_rows,
        max_daily_loss_pct=args.max_daily_loss_pct,
        direction_threshold_ticks=args.direction_threshold_ticks,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        max_horizon_steps=(args.max_horizon_steps if args.max_horizon_steps > 0 else None),
        replay_horizon_steps=replay_horizon_steps,
        allow_oracle_forward_return=args.allow_oracle_forward_return,
        single_position_only=(not args.disable_single_position_only),
        cooldown_rows=args.cooldown_rows,
        score_start_ts=start_ts,
        score_end_ts=end_ts,
        relax_policy_ev=args.relax_policy_ev,
        policy_min_edge=args.policy_min_edge,
        skip_event_gate=args.skip_event_gate,
        long_only=args.long_only,
    )
    summary['oos_guard'] = oos_guard
    summary['entrydata'] = entrydata_diag
    summary['backtest_window'] = {
        'aligned_start_ts': align_start_ts,
        'aligned_end_ts': align_end_ts,
        'start_ts': start_ts,
        'end_ts': end_ts,
        'rows': int(len(df_score)),
    }
    summary['horizon_alignment'] = {
        'day_trading_manifest_path': manifest_path,
        'manifest_horizon_bars': mt_h,
        'manifest_freq': mt_freq,
        'bar_minutes_effective': float(bar_minutes_eff),
        'fixed_horizon_minutes': None if args.fixed_horizon is None else float(args.fixed_horizon),
        'replay_horizon_steps': summary.get('replay_horizon_steps'),
        'note': (
            'replay uses label_horizon_steps per row when replay_horizon_steps is null; '
            'else --fixed_horizon overrides. Align minutes: horizon_bars * bar_minutes.'
        ),
    }
    summary['runtime_input_mode'] = {
        'live_like_runtime_inputs': bool(args.live_like_runtime_inputs),
        'disable_auto_oof_meta': bool(disable_auto_meta),
        'disable_auto_visual_artifact': bool(disable_auto_visual),
        'meta_path': None if meta_path is None else str(meta_path),
        'visual_path': None if args.visual_npy is None else str(args.visual_npy),
    }
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'backtest_v19_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n[backtest] V19 causal backtest complete")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
