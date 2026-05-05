"""
labels_v19.py - Unified causal labels for QuantSystem
======================================================
V19 includes three structural fixes over the older labeling path:

  FIX-9   Adaptive horizon ∝ ATR (was static ×1.5)
          horizon now scales per volatility regime:
            slow market  → shorter horizon (don't wait forever)
            fast market  → longer horizon  (give the move space)
          Formula: effective_horizon_i = clip(base × (ATR_i / ATR_median), min, max)

  FIX-10  Per-sample dynamic threshold in forward scan
          Older code used median(dynamic_threshold) — one scalar for the entire dataset.
          V19 patches label_with_forward_scan per-row via a vectorised pre-pass
          that computes TP/SL hit indices directly, then assembles labels without
          needing a scalar threshold. The forward scan is now truly ATR-per-row.

  FIX-11  Kalman trend actively filters labeling (not just a feature)
          Older code computed trend_label / trend_entry_label but never used them to
          gate bias_label. V19 applies a post-labeling mask:
            - LONG  labels where Kalman trend == DOWN  → forced NEUTRAL
            - SHORT labels where Kalman trend == UP    → forced NEUTRAL
          This removes counter-trend noise before the model ever sees the data.
          Trend strength (kalman_slope magnitude) is exposed as trend_strength
          so the model can learn "aligned + strong trend = higher conviction".

All earlier fixes (FIX-1 … FIX-8) are preserved unchanged.
External API is 100% backwards-compatible: build_causal_event_labels(df, ...).

New optional parameters:
  adaptive_horizon     : bool  = False  — enable FIX-9 only when explicitly requested
  horizon_min_mult     : float = 0.5    — floor: horizon never < base × 0.5
  horizon_max_mult     : float = 3.0    — ceiling: horizon never > base × 3.0
  trend_filter         : bool  = False  — enable FIX-11 only when explicitly requested
  trend_filter_strict  : bool  = False  — if True, neutral rows also filtered by trend
  tp_sl_anchor_window  : int   = 1000   — anchor rows for stationary ATR-based TP/SL
"""

from __future__ import annotations

import multiprocessing
import sys
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from modules.dynamic_labels import (
    DIR_LONG,
    DIR_NEUTRAL,
    DIR_SHORT,
    EVENT_SHIFT_COLS,
    LABEL_CANCEL,
    LABEL_LOSE,
    LABEL_WIN,
    QUALITY_NONE,
    QUALITY_STRONG,
    QUALITY_WEAK,
    REGIME_RANGING,
    TREND_DOWN,
    TREND_UP,
    TREND_NEUTRAL,
    append_dynamic_orderbook_features,
    build_event_filter,
    engineer_features,
    label_with_forward_scan,
    kalman_trend,           # returns (trend_label, trend_strength, kalman_price)
)

# ── public aliases (backwards-compat) ─────────────────────────────────────────
BIAS_LONG    = DIR_LONG
BIAS_SHORT   = DIR_SHORT
BIAS_NEUTRAL = DIR_NEUTRAL

SETUP_ABSORPTION = 0
SETUP_SPOOFING   = 1
SETUP_OBI        = 2
SETUP_MIXED      = 3

PATH_LONG_TP_FIRST  = 0
PATH_SHORT_TP_FIRST = 1
PATH_LONG_SL_FIRST  = 2
PATH_SHORT_SL_FIRST = 3
PATH_TIMEOUT        = 4

DETAIL_LONG = 0
DETAIL_SHORT = 1
DETAIL_TIMEOUT = 2
DETAIL_LONG_SL = 3
DETAIL_SHORT_SL = 4

NEUTRAL_REASON_NONE = 0
NEUTRAL_REASON_TIMEOUT = 1
NEUTRAL_REASON_LONG_SL = 2
NEUTRAL_REASON_SHORT_SL = 3

DEFAULT_V19_DIRECTION_THRESHOLD_TICKS = 1.5
DEFAULT_V19_TP_MULT = 1.5
DEFAULT_RAW_EVENT_TARGET_RATE = 0.70
DEFAULT_TRAINING_EVENT_TARGET_RATE = 0.25


_FORWARD_SCAN_SHARED: dict[str, object] = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_duration_brief(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _coerce_label_timestamp_series(values, *, context: str) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    ts = pd.to_datetime(series, utc=True, errors='coerce')
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts, index=series.index)
    ts = ts.dt.tz_localize(None)
    invalid_mask = ts.isna()
    if bool(invalid_mask.any()):
        sample = invalid_mask[invalid_mask].index[:5].tolist()
        raise ValueError(
            f"❌ Invalid timestamps in {context}: "
            f"rows={int(invalid_mask.sum())} sample_indices={sample}"
        )
    return ts


def _estimate_remaining_seconds(done: int, total: int, elapsed_seconds: float) -> float | None:
    done = int(done)
    total = int(total)
    if done <= 0 or total <= done:
        return 0.0 if total == done and total > 0 else None
    elapsed_seconds = float(elapsed_seconds)
    if elapsed_seconds <= 0:
        return None
    rate = done / elapsed_seconds
    if rate <= 0:
        return None
    return (total - done) / rate


def _resolve_economic_tp_floor_ticks(
    *,
    direction_threshold_ticks: float,
    tp_mult: float,
    sl_mult: float,
    execution_cost_pips: float,
    enforce_economic_tp_floor: bool,
) -> dict[str, float | bool]:
    floor_ticks = max(float(direction_threshold_ticks or 0.0), 0.0)
    base_tp_floor_ticks = floor_ticks * max(float(tp_mult or 0.0), 0.0)
    stop_floor_ticks = floor_ticks * max(float(sl_mult or 0.0), 0.0)
    effective_cost_ticks = max(float(execution_cost_pips or 0.0), 0.0)
    economic_tp_floor_ticks = base_tp_floor_ticks
    if enforce_economic_tp_floor and effective_cost_ticks > 0.0:
        economic_tp_floor_ticks = max(base_tp_floor_ticks, stop_floor_ticks + effective_cost_ticks)
    uplift_ticks = max(economic_tp_floor_ticks - base_tp_floor_ticks, 0.0)
    return {
        "base_tp_floor_ticks": float(base_tp_floor_ticks),
        "stop_floor_ticks": float(stop_floor_ticks),
        "effective_cost_ticks": float(effective_cost_ticks),
        "economic_tp_floor_ticks": float(economic_tp_floor_ticks),
        "tp_floor_uplift_ticks": float(uplift_ticks),
        "economic_floor_applied": bool(enforce_economic_tp_floor and uplift_ticks > 1e-9),
    }


def _fenwick_add(tree: np.ndarray, idx_zero_based: int, delta: int) -> None:
    i = int(idx_zero_based) + 1
    size = len(tree)
    while i < size:
        tree[i] += int(delta)
        i += i & -i


def _fenwick_find_by_order(tree: np.ndarray, order_one_based: int) -> int:
    if order_one_based <= 0:
        return 0
    idx = 0
    bit = 1 << (len(tree).bit_length() - 1)
    while bit:
        nxt = idx + bit
        if nxt < len(tree) and tree[nxt] < order_one_based:
            order_one_based -= int(tree[nxt])
            idx = nxt
        bit >>= 1
    return min(idx, len(tree) - 2)


def _fenwick_quantile_linear(
    tree: np.ndarray,
    sorted_values: np.ndarray,
    q: float,
    count: int,
) -> float:
    count = int(count)
    if count <= 0:
        return 0.0
    if count == 1:
        only_idx = _fenwick_find_by_order(tree, 1)
        return float(sorted_values[only_idx])

    q = min(max(float(q), 0.0), 1.0)
    h = (count - 1) * q
    lo_rank = int(np.floor(h))
    hi_rank = int(np.ceil(h))
    lo_idx = _fenwick_find_by_order(tree, lo_rank + 1)
    lo_val = float(sorted_values[lo_idx])
    if hi_rank == lo_rank:
        return lo_val
    hi_idx = _fenwick_find_by_order(tree, hi_rank + 1)
    hi_val = float(sorted_values[hi_idx])
    frac = float(h - lo_rank)
    return lo_val + frac * (hi_val - lo_val)

def _safe_series(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    """Return a numeric numpy array for *col*, falling back to zeros."""
    return (
        pd.to_numeric(df.get(col, pd.Series(np.zeros(n))), errors="coerce")
        .fillna(0.0)
        .values
    )


def _infer_setup_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Classify each row into one of four microstructure setups.
    FIX-7: thresholds loosened (0.30→0.20, 0.20→0.15, 0.30→0.20).
    """
    n = len(df)
    absorb = np.abs(_safe_series(df, "absorption_intensity", n))
    spoof  = np.abs(_safe_series(df, "spoofing_ratio",       n))
    obi    = np.abs(_safe_series(df, "obi",                  n))

    absorb_sig = absorb > 0.20
    spoof_sig  = spoof  > 0.15
    obi_sig    = obi    > 0.20

    sig_count = (
        absorb_sig.astype(np.int8)
        + spoof_sig.astype(np.int8)
        + obi_sig.astype(np.int8)
    )

    setup = np.full(n, SETUP_MIXED, dtype=np.int8)
    setup[(sig_count == 1) & absorb_sig] = SETUP_ABSORPTION
    setup[(sig_count == 1) & spoof_sig]  = SETUP_SPOOFING
    setup[(sig_count == 1) & obi_sig]    = SETUP_OBI
    return setup


def _liquidity_score(df: pd.DataFrame) -> np.ndarray:
    """Composite liquidity score [0, 100]."""
    n = len(df)
    volume   = np.clip(_safe_series(df, "volume",            n), 0.0, None)
    depth    = np.clip(_safe_series(df, "liquidity_density", n), 0.0, None)
    gap      = np.clip(_safe_series(df, "gap_size",          n), 0.0, None)
    wall_bid = np.clip(_safe_series(df, "bid_wall_strength", n), 0.0, None)
    wall_ask = np.clip(_safe_series(df, "ask_wall_strength", n), 0.0, None)
    obi      = np.abs( _safe_series(df, "obi",               n))

    if "volume" not in df.columns and "size" in df.columns:
        volume = np.clip(_safe_series(df, "size", n), 0.0, None)

    wall = np.maximum(wall_bid, wall_ask)
    raw  = (
        np.log1p(volume) * (1.0 + obi) * (1.0 + wall)
        + 0.25 * depth
        + 0.10 * gap
    )
    return np.clip(raw, 0.0, 100.0).astype(np.float32)


def _quality_to_conf(signal_quality: pd.Series | np.ndarray) -> np.ndarray:
    """Map quality enum → float confidence. FIX-3: QUALITY_NONE → 0.25."""
    q   = pd.Series(signal_quality).fillna(QUALITY_NONE).astype(np.int8).values
    out = np.full(len(q), 0.5, dtype=np.float32)
    out[q == QUALITY_STRONG] = 1.0
    out[q == QUALITY_NONE]   = 0.25
    return out


def _empty_output(out: pd.DataFrame) -> pd.DataFrame:
    empty_int8    = np.array([], dtype=np.int8)
    empty_float32 = np.array([], dtype=np.float32)
    empty_int32   = np.array([], dtype=np.int32)
    out["bias_label"]           = empty_int8
    out["setup_label"]          = empty_int8
    out["conf_label"]           = empty_float32
    out["signal_quality"]       = empty_int8
    out["is_expansion"]         = empty_int8
    out["liq_score"]            = empty_float32
    out["regime_label"]         = empty_int8
    out["label_end_ts"]         = pd.to_datetime([])
    out["forward_return"]       = empty_float32
    out["label_horizon_steps"]  = empty_int32
    out["long_label"]           = empty_int8
    out["short_label"]          = empty_int8
    out["path_outcome"]         = empty_int8
    out["adverse_path_flag"]    = empty_int8
    out["bias_label_detail"]    = empty_int8
    out["neutral_reason"]       = empty_int8
    out["timeout_move_exceeded_band"] = empty_int8
    out["event_flag"]           = empty_int8
    out["train_event_flag"]     = empty_int8
    out["event_score"]          = empty_float32
    out["event_trigger_count"]  = empty_int8
    out["is_event"]             = empty_int8
    out["trend_label"]          = empty_int8
    out["trend_strength"]       = empty_float32
    out["kalman_trend_label"]   = empty_int8
    out["kalman_trend_strength"] = empty_float32
    out["kalman_price"]         = empty_float32
    out["effective_horizon"]    = empty_int32
    return out


def _compute_micro_atr(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Approximate ATR on tick/bar data when no high/low is available.
    Uses rolling std of returns as volatility proxy.
    """
    returns = np.diff(prices, prepend=prices[0])
    atr     = np.zeros_like(prices, dtype=np.float64)
    for i in range(len(prices)):
        lo = max(0, i - window + 1)
        atr[i] = np.std(returns[lo : i + 1]) if i >= 1 else abs(returns[i])
    atr_hist = np.where(atr > 1e-10, atr, np.nan)
    causal_med = _causal_expanding_median(
        atr_hist,
        min_periods=max(5, min(int(window), 20)),
        fallback=1e-5,
    )
    atr = np.where(atr < 1e-10, causal_med, atr)
    return atr


def _causal_expanding_median(
    values: np.ndarray,
    min_periods: int = 20,
    fallback: float = 1e-5,
) -> np.ndarray:
    """Causal expanding median with no future-row contamination."""
    series = pd.Series(np.asarray(values, dtype=np.float64)).replace([np.inf, -np.inf], np.nan)
    primary = series.expanding(min_periods=max(int(min_periods), 1)).median()
    bootstrap = series.expanding(min_periods=1).median()
    med = primary.where(primary.notna(), bootstrap).ffill().fillna(float(fallback))
    return med.to_numpy(dtype=np.float64, copy=False)


def _rolling_zscore_np(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score used by the stronger training-event gate."""
    series = pd.Series(np.asarray(values, dtype=np.float64))
    roll_window = max(int(window), 1)
    roll_mean = series.rolling(roll_window, min_periods=1).mean()
    roll_std = series.rolling(roll_window, min_periods=1).std().replace(0.0, 1e-9)
    return ((series - roll_mean) / roll_std).fillna(0.0).to_numpy(dtype=np.float64, copy=False)


def _build_training_event_gate(
    df: pd.DataFrame,
    base_event_mask: pd.Series | np.ndarray,
    roll_window: int,
    vol_mult: float,
    obi_thr: float,
    wall_thr: float,
    shift_z_thr: float = 0.75,
    target_rate: float = 0.25,
    causal_threshold_mode: str = "expanding",
    fixed_score_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build a stricter causal gate for dataset selection.

    `event_flag` remains a broad activity/anomaly marker. `train_event_flag`
    is narrower and keeps roughly the strongest quarter of rows using only
    current-time information.
    """
    n = len(df)
    if n == 0:
        empty_int8 = np.array([], dtype=np.int8)
        empty_float32 = np.array([], dtype=np.float32)
        return empty_int8, empty_float32, empty_int8, 0.0

    if "volume" in df.columns:
        volume = _safe_series(df, "volume", n).astype(np.float64)
    elif "size" in df.columns:
        volume = _safe_series(df, "size", n).astype(np.float64)
    else:
        volume = np.zeros(n, dtype=np.float64)

    roll_vol = (
        pd.Series(volume)
        .rolling(max(int(roll_window), 1), min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64, copy=False)
    )
    vol_ratio = volume / np.maximum(roll_vol, 1e-9)
    obi_abs = np.abs(_safe_series(df, "obi", n).astype(np.float64))
    wall_strength = np.maximum(
        _safe_series(df, "bid_wall_strength", n).astype(np.float64),
        _safe_series(df, "ask_wall_strength", n).astype(np.float64),
    )

    shift_peak = np.zeros(n, dtype=np.float64)
    for col in EVENT_SHIFT_COLS:
        z_col = f"{col}_zscore"
        if z_col in df.columns:
            zscore = np.abs(_safe_series(df, z_col, n).astype(np.float64))
        elif col in df.columns:
            zscore = np.abs(_rolling_zscore_np(_safe_series(df, col, n), window=roll_window))
        else:
            continue
        shift_peak = np.maximum(shift_peak, zscore)

    cond_vol = vol_ratio > max(float(vol_mult), 1e-6)
    cond_obi = obi_abs > max(float(obi_thr), 1e-6)
    cond_wall = wall_strength > max(float(wall_thr), 1e-6)
    cond_shift = shift_peak > max(float(shift_z_thr), 1e-6)
    trigger_count = (
        cond_vol.astype(np.int8)
        + cond_obi.astype(np.int8)
        + cond_wall.astype(np.int8)
        + cond_shift.astype(np.int8)
    ).astype(np.int8)

    vol_excess = np.clip((vol_ratio / max(float(vol_mult), 1e-6)) - 1.0, 0.0, None)
    obi_excess = np.clip((obi_abs / max(float(obi_thr), 1e-6)) - 1.0, 0.0, None)
    wall_excess = np.clip((wall_strength / max(float(wall_thr), 1e-6)) - 1.0, 0.0, None)
    shift_excess = np.clip((shift_peak / max(float(shift_z_thr), 1e-6)) - 1.0, 0.0, None)
    score = (
        0.30 * vol_excess
        + 0.30 * obi_excess
        + 0.20 * wall_excess
        + 0.20 * shift_excess
        + 0.50 * np.clip(trigger_count.astype(np.float32) - 1.0, 0.0, None)
    ).astype(np.float32)

    base_mask = np.asarray(base_event_mask, dtype=bool)
    candidate = np.zeros(n, dtype=bool)
    threshold = 0.0
    if base_mask.any():
        mode = str(causal_threshold_mode or "expanding").strip().lower()
        if mode == "fixed":
            threshold = float(max(fixed_score_threshold or 0.0, 0.0))
            candidate = base_mask & (score >= threshold)
        else:
            score64 = np.asarray(score, dtype=np.float64)
            base_scores_all = score64[base_mask]
            if base_scores_all.size == 0:
                return candidate.astype(np.int8), score, trigger_count, threshold
            sorted_values = np.unique(base_scores_all)
            fenwick_tree = np.zeros(len(sorted_values) + 1, dtype=np.int32)
            compressed = np.searchsorted(sorted_values, score64, side="left")
            past_base_count = 0
            for i in range(n):
                if not base_mask[i]:
                    continue
                if past_base_count <= 0 or past_base_count < max(int(roll_window), 10):
                    current_threshold = 0.0
                else:
                    base_rate = float(past_base_count) / max(float(i), 1.0)
                    desired_rate = min(max(float(target_rate), 0.05), base_rate)
                    keep_inside_base = min(max(desired_rate / max(base_rate, 1e-9), 0.0), 1.0)
                    if keep_inside_base >= 0.999:
                        current_threshold = _fenwick_quantile_linear(
                            fenwick_tree,
                            sorted_values,
                            q=0.0,
                            count=past_base_count,
                        )
                    else:
                        q = min(max(1.0 - keep_inside_base, 0.0), 1.0)
                        current_threshold = _fenwick_quantile_linear(
                            fenwick_tree,
                            sorted_values,
                            q=q,
                            count=past_base_count,
                        )
                threshold = float(max(current_threshold, 0.0))
                candidate[i] = bool(score[i] >= threshold)
                _fenwick_add(fenwick_tree, int(compressed[i]), 1)
                past_base_count += 1
        if not candidate.any():
            fallback = base_mask & (trigger_count >= 2)
            candidate = fallback if fallback.any() else base_mask.copy()
            threshold = float(np.nanmin(score[candidate])) if candidate.any() else threshold

    return candidate.astype(np.int8), score, trigger_count, threshold


def _build_broad_event_gate(
    df: pd.DataFrame,
    base_event_mask: pd.Series | np.ndarray,
    roll_window: int,
    vol_mult: float,
    obi_thr: float,
    wall_thr: float,
    shift_z_thr: float = 0.75,
    target_rate: float = 0.70,
    causal_threshold_mode: str = "expanding",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build a broad-but-not-trivial causal event gate.

    The legacy raw event_flag was a pure OR across several conditions and could
    easily light up >90% of rows on volatile datasets. We keep it broader than
    `train_event_flag`, but still causal and rate-controlled.
    """
    return _build_training_event_gate(
        df,
        base_event_mask=base_event_mask,
        roll_window=roll_window,
        vol_mult=vol_mult,
        obi_thr=obi_thr,
        wall_thr=wall_thr,
        shift_z_thr=shift_z_thr,
        target_rate=target_rate,
        causal_threshold_mode=causal_threshold_mode,
        fixed_score_threshold=None,
    )


# ── FIX-9: Adaptive per-row horizon ──────────────────────────────────────────

def _compute_adaptive_horizons(
    atr: np.ndarray,
    base_horizon: int,
    min_mult: float = 0.5,
    max_mult: float = 3.0,
) -> np.ndarray:
    """
    FIX-9: Compute per-row effective horizon proportional to ATR.

    Intuition:
      - When current ATR is HIGH (fast market):  horizon is LONGER
        → give the price move space to reach the TP target
      - When current ATR is LOW  (slow market):  horizon is SHORTER
        → don't wait forever for a move that isn't coming

    Formula:
        h_i = base × clip(ATR_i / ATR_median_t, min_mult, max_mult)

    ATR_median_t is causal/expanding, so row i never sees future volatility.
    This is proportional to ATR: slow = short horizon, fast = long horizon.
    Result is rounded to the nearest integer and bounded.

    Returns
    -------
    np.ndarray of int32, shape (n,)
    """
    atr = np.asarray(atr, dtype=np.float64)
    atr_hist = np.where(atr > 1e-10, atr, np.nan)
    if np.isnan(atr_hist).all():
        # degenerate case: flat price, return base horizon everywhere
        return np.full(len(atr), base_horizon, dtype=np.int32)

    atr_med = _causal_expanding_median(atr_hist, min_periods=20, fallback=float(np.nanmedian(atr_hist)))
    safe_atr = np.where(atr < 1e-10, atr_med, atr)
    denom = np.where(atr_med < 1e-10, safe_atr, atr_med)
    ratio = np.clip(safe_atr / np.where(denom < 1e-10, safe_atr, denom), min_mult, max_mult)
    horizons = np.round(base_horizon * ratio).astype(np.int32)
    return horizons


# ── Liquidity-aware TP cap ────────────────────────────────────────────────────

def _find_liquidity_tp_distance(
    current_price: float,
    direction: int,
    tick_size: float,
    bid_wall_px: float = np.nan,
    ask_wall_px: float = np.nan,
    bid_wall_strength: float = np.nan,
    ask_wall_strength: float = np.nan,
    min_wall_strength: float = 2.5,
    wall_exit_buffer_ticks: float = 1.0,
    min_tp_ticks: float = 2.0,
) -> Optional[float]:
    """
    Return a causal TP distance derived from the nearest structural wall.

    Scanner strengths may arrive either normalized to [0, 1] or as raw
    size/mean-size ratios. We accept both scales by using a 0.5 floor for
    normalized strengths and a higher default floor for ratio-style values.
    """
    tick = max(float(tick_size or 0.0), 1e-9)
    buffer_px = max(float(wall_exit_buffer_ticks), 0.0) * tick
    min_tp_px = max(float(min_tp_ticks), 0.0) * tick

    if direction == DIR_LONG:
        wall_px = float(ask_wall_px) if np.isfinite(ask_wall_px) else np.nan
        strength = float(ask_wall_strength) if np.isfinite(ask_wall_strength) else np.nan
        if not np.isfinite(wall_px) or wall_px <= current_price:
            return None
        target_price = wall_px - buffer_px
        distance = target_price - current_price
    elif direction == DIR_SHORT:
        wall_px = float(bid_wall_px) if np.isfinite(bid_wall_px) else np.nan
        strength = float(bid_wall_strength) if np.isfinite(bid_wall_strength) else np.nan
        if not np.isfinite(wall_px) or wall_px >= current_price:
            return None
        target_price = wall_px + buffer_px
        distance = current_price - target_price
    else:
        return None

    if np.isfinite(strength) and strength > 0.0:
        strength_floor = min(float(min_wall_strength), 0.5) if strength <= 1.0 else float(min_wall_strength)
        if strength < strength_floor:
            return None

    if not np.isfinite(distance) or distance < min_tp_px:
        return None

    return float(distance)


# ── FIX-10: Per-row forward scan ──────────────────────────────────────────────


def _forward_scan_rows(
    start: int,
    stop: int,
    *,
    prices: np.ndarray,
    dynamic_threshold: np.ndarray,
    adaptive_horizons: np.ndarray,
    tick_size: float,
    tp_mult: float = DEFAULT_V19_TP_MULT,
    sl_mult: float = 1.0,
    bid_wall_px: Optional[np.ndarray] = None,
    ask_wall_px: Optional[np.ndarray] = None,
    bid_wall_strength: Optional[np.ndarray] = None,
    ask_wall_strength: Optional[np.ndarray] = None,
    min_wall_strength: float = 2.5,
    wall_exit_buffer_ticks: float = 1.0,
    min_wall_tp_ticks: float = 2.0,
    economic_tp_floor_ticks: float = 0.0,
    progress_every_rows: int = 0,
    progress_prefix: str = "Step 4 forward scan",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a contiguous row range using the full read-only price arrays."""
    n = len(prices)
    size = max(0, int(stop) - int(start))
    bias_arr = np.full(size, DIR_NEUTRAL, dtype=np.int8)
    quality_arr = np.full(size, QUALITY_WEAK, dtype=np.int8)
    end_idx_arr = np.arange(int(start), int(stop), dtype=np.int32)
    path_outcome_arr = np.full(size, PATH_TIMEOUT, dtype=np.int8)
    loop_started_at = time.perf_counter() if progress_every_rows > 0 and size > 0 else 0.0

    for local_idx, i in enumerate(range(int(start), int(stop))):
        p0 = prices[i]
        thr = max(dynamic_threshold[i], tick_size)
        economic_tp_floor_px = max(float(economic_tp_floor_ticks), 0.0) * float(tick_size)
        tp = max(thr * tp_mult, economic_tp_floor_px)
        sl = thr * sl_mult
        h = int(adaptive_horizons[i])
        end = min(i + h, n - 1)
        tp_long = tp
        tp_short = tp
        liquidity_tp_min_ticks = max(float(min_wall_tp_ticks), float(economic_tp_floor_ticks))

        liquidity_tp_long = _find_liquidity_tp_distance(
            current_price=p0,
            direction=DIR_LONG,
            tick_size=tick_size,
            bid_wall_px=np.nan if bid_wall_px is None else bid_wall_px[i],
            ask_wall_px=np.nan if ask_wall_px is None else ask_wall_px[i],
            bid_wall_strength=np.nan if bid_wall_strength is None else bid_wall_strength[i],
            ask_wall_strength=np.nan if ask_wall_strength is None else ask_wall_strength[i],
            min_wall_strength=min_wall_strength,
            wall_exit_buffer_ticks=wall_exit_buffer_ticks,
            min_tp_ticks=liquidity_tp_min_ticks,
        )
        if liquidity_tp_long is not None:
            tp_long = min(tp_long, liquidity_tp_long)

        liquidity_tp_short = _find_liquidity_tp_distance(
            current_price=p0,
            direction=DIR_SHORT,
            tick_size=tick_size,
            bid_wall_px=np.nan if bid_wall_px is None else bid_wall_px[i],
            ask_wall_px=np.nan if ask_wall_px is None else ask_wall_px[i],
            bid_wall_strength=np.nan if bid_wall_strength is None else bid_wall_strength[i],
            ask_wall_strength=np.nan if ask_wall_strength is None else ask_wall_strength[i],
            min_wall_strength=min_wall_strength,
            wall_exit_buffer_ticks=wall_exit_buffer_ticks,
            min_tp_ticks=liquidity_tp_min_ticks,
        )
        if liquidity_tp_short is not None:
            tp_short = min(tp_short, liquidity_tp_short)

        tp_long_hit = False
        sl_long_hit = False
        tp_short_hit = False
        sl_short_hit = False
        first_long_hit_idx = end
        first_short_hit_idx = end

        for j in range(i + 1, end + 1):
            move = prices[j] - p0
            if not tp_long_hit and move >= tp_long:
                tp_long_hit = True
                first_long_hit_idx = j
                break
            if not sl_long_hit and move <= -sl:
                sl_long_hit = True
                first_long_hit_idx = j
                break

        for j in range(i + 1, end + 1):
            move = prices[j] - p0
            if not tp_short_hit and move <= -tp_short:
                tp_short_hit = True
                first_short_hit_idx = j
                break
            if not sl_short_hit and move >= sl:
                sl_short_hit = True
                first_short_hit_idx = j
                break

        if tp_long_hit and (not tp_short_hit or first_long_hit_idx <= first_short_hit_idx):
            bias_arr[local_idx] = DIR_LONG
            quality_arr[local_idx] = QUALITY_STRONG
            end_idx_arr[local_idx] = first_long_hit_idx
            path_outcome_arr[local_idx] = PATH_LONG_TP_FIRST
        elif tp_short_hit:
            bias_arr[local_idx] = DIR_SHORT
            quality_arr[local_idx] = QUALITY_STRONG
            end_idx_arr[local_idx] = first_short_hit_idx
            path_outcome_arr[local_idx] = PATH_SHORT_TP_FIRST
        elif sl_long_hit or sl_short_hit:
            quality_arr[local_idx] = QUALITY_WEAK
            if sl_long_hit and sl_short_hit:
                if first_long_hit_idx <= first_short_hit_idx:
                    end_idx_arr[local_idx] = first_long_hit_idx
                    path_outcome_arr[local_idx] = PATH_LONG_SL_FIRST
                else:
                    end_idx_arr[local_idx] = first_short_hit_idx
                    path_outcome_arr[local_idx] = PATH_SHORT_SL_FIRST
            elif sl_long_hit:
                end_idx_arr[local_idx] = first_long_hit_idx
                path_outcome_arr[local_idx] = PATH_LONG_SL_FIRST
            else:
                end_idx_arr[local_idx] = first_short_hit_idx
                path_outcome_arr[local_idx] = PATH_SHORT_SL_FIRST

        processed = local_idx + 1
        if (
            progress_every_rows > 0 and
            (processed % int(progress_every_rows) == 0 or processed == size)
        ):
            elapsed = time.perf_counter() - loop_started_at
            eta = _estimate_remaining_seconds(processed, size, elapsed)
            print(
                f"  ⏳ {progress_prefix}: rows={processed:,}/{size:,} "
                f"({processed/max(size, 1):.1%}) | elapsed={_format_duration_brief(elapsed)} "
                f"| eta≈{_format_duration_brief(eta)}",
                flush=True,
            )

    return bias_arr, quality_arr, end_idx_arr, path_outcome_arr


def _forward_scan_pool_worker(bounds: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start, stop = bounds
    if not _FORWARD_SCAN_SHARED:
        raise RuntimeError("forward scan pool worker missing shared arrays")
    bias_arr, quality_arr, end_idx_arr, path_outcome_arr = _forward_scan_rows(
        start,
        stop,
        **_FORWARD_SCAN_SHARED,
    )
    return start, bias_arr, quality_arr, end_idx_arr, path_outcome_arr

def _forward_scan_per_row(
    prices: np.ndarray,
    dynamic_threshold: np.ndarray,
    adaptive_horizons: np.ndarray,
    tick_size: float,
    tp_mult: float = DEFAULT_V19_TP_MULT,
    sl_mult: float = 1.0,
    bid_wall_px: Optional[np.ndarray] = None,
    ask_wall_px: Optional[np.ndarray] = None,
    bid_wall_strength: Optional[np.ndarray] = None,
    ask_wall_strength: Optional[np.ndarray] = None,
    min_wall_strength: float = 2.5,
    wall_exit_buffer_ticks: float = 1.0,
    min_wall_tp_ticks: float = 2.0,
    economic_tp_floor_ticks: float = 0.0,
    n_workers: int | None = None,
    min_parallel_rows: int = 250_000,
    progress_every_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FIX-10: Vectorised per-row forward scan with row-specific threshold AND horizon.

    v21 problem:
        label_with_forward_scan() expected a single scalar threshold.
        np.median(dynamic_threshold) collapsed all per-row ATR information
        into one number — making the "dynamic" threshold effectively static.

    current solution:
        We pre-compute for each row whether price hits TP or SL
        within its own adaptive_horizon using its own ATR-derived threshold.
        This replaces the downstream call to label_with_forward_scan for the
        direction/quality signal (we still use it for ancillary columns).

    Parameters
    ----------
    prices            : close price array, shape (n,)
    dynamic_threshold : per-row threshold in price units, shape (n,)
    adaptive_horizons : per-row max bars to scan forward, shape (n,)
    tick_size         : minimum price increment (for fallback)
    tp_mult           : TP = entry ± tp_mult × threshold
    sl_mult           : SL = entry ∓ sl_mult × threshold

    Returns
    -------
    bias_arr         : int8 array  — DIR_LONG / DIR_SHORT / DIR_NEUTRAL
    quality_arr      : int8 array  — QUALITY_STRONG / QUALITY_WEAK
    end_idx_arr      : int32 array — index where scan terminated
    path_outcome_arr : int8 array  — TP/SL/timeout path diagnostics
    """
    n = len(prices)
    workers = 1 if n_workers is None else max(1, int(n_workers))
    platform_note = ""
    if sys.platform == "win32":
        platform_note = " | platform=win32 (parallel Step 4 currently unavailable because fork is not supported)"
    if progress_every_rows is None:
        auto_progress_every_rows = 0 if n < 10_000 else min(250_000, max(2_500, n // 10))
    else:
        auto_progress_every_rows = max(0, int(progress_every_rows))
    shared_payload = {
        'prices': prices,
        'dynamic_threshold': dynamic_threshold,
        'adaptive_horizons': adaptive_horizons,
        'tick_size': tick_size,
        'tp_mult': tp_mult,
        'sl_mult': sl_mult,
        'bid_wall_px': bid_wall_px,
        'ask_wall_px': ask_wall_px,
        'bid_wall_strength': bid_wall_strength,
        'ask_wall_strength': ask_wall_strength,
        'min_wall_strength': min_wall_strength,
        'wall_exit_buffer_ticks': wall_exit_buffer_ticks,
        'min_wall_tp_ticks': min_wall_tp_ticks,
        'economic_tp_floor_ticks': economic_tp_floor_ticks,
    }

    if workers <= 1:
        print(
            f"  ℹ️ Step 4 forward scan sequential: workers={workers} | rows={n:,}{platform_note}",
            flush=True,
        )
        return _forward_scan_rows(
            0,
            n,
            progress_every_rows=auto_progress_every_rows,
            progress_prefix="Step 4 forward scan",
            **shared_payload,
        )
    if n < int(min_parallel_rows):
        print(
            f"  ℹ️ Step 4 forward scan sequential: rows={n:,} "
            f"< min_parallel_rows={int(min_parallel_rows):,} | requested_workers={workers}{platform_note}",
            flush=True,
        )
        return _forward_scan_rows(
            0,
            n,
            progress_every_rows=auto_progress_every_rows,
            progress_prefix="Step 4 forward scan",
            **shared_payload,
        )

    available_methods = set(multiprocessing.get_all_start_methods())
    use_fork = sys.platform != "win32" and "fork" in available_methods
    if not use_fork:
        warnings.warn(
            "Parallel Step 4 forward scan requires fork-capable multiprocessing; "
            "falling back to sequential scan on this platform.",
            RuntimeWarning,
        )
        print(
            f"  ℹ️ Step 4 forward scan sequential: fork start-method unavailable "
            f"| requested_workers={workers} | rows={n:,}{platform_note}",
            flush=True,
        )
        return _forward_scan_rows(
            0,
            n,
            progress_every_rows=auto_progress_every_rows,
            progress_prefix="Step 4 forward scan",
            **shared_payload,
        )

    workers = min(workers, n)
    task_count = min(n, max(workers, workers * 2))
    chunk_rows = max(1, (n + task_count - 1) // task_count)
    bounds = [(start, min(start + chunk_rows, n)) for start in range(0, n, chunk_rows)]
    if len(bounds) <= 1:
        return _forward_scan_rows(
            0,
            n,
            progress_every_rows=auto_progress_every_rows,
            progress_prefix="Step 4 forward scan",
            **shared_payload,
        )

    print(
        f"  ⚡ Step 4 forward scan parallel: workers={workers} "
        f"| chunks={len(bounds)} | rows={n:,}",
        flush=True,
    )

    global _FORWARD_SCAN_SHARED
    _FORWARD_SCAN_SHARED = shared_payload
    ctx_mp = multiprocessing.get_context("fork")
    parts: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    scan_started_at = time.perf_counter()
    done_rows = 0
    total_chunks = len(bounds)
    try:
        with ctx_mp.Pool(processes=min(workers, len(bounds))) as pool:
            for part in pool.imap_unordered(_forward_scan_pool_worker, bounds):
                parts.append(part)
                done_rows += len(part[1])
                elapsed = time.perf_counter() - scan_started_at
                eta = _estimate_remaining_seconds(done_rows, n, elapsed)
                print(
                    f"  ⏳ Step 4 forward scan: chunks={len(parts):,}/{total_chunks:,} "
                    f"| rows={done_rows:,}/{n:,} ({done_rows/max(n, 1):.1%}) "
                    f"| elapsed={_format_duration_brief(elapsed)} "
                    f"| eta≈{_format_duration_brief(eta)}",
                    flush=True,
                )
    finally:
        _FORWARD_SCAN_SHARED = {}

    parts.sort(key=lambda item: int(item[0]))
    bias_arr = np.empty(n, dtype=np.int8)
    quality_arr = np.empty(n, dtype=np.int8)
    end_idx_arr = np.empty(n, dtype=np.int32)
    path_outcome_arr = np.empty(n, dtype=np.int8)
    for start, bias_part, quality_part, end_part, path_part in parts:
        stop = start + len(bias_part)
        bias_arr[start:stop] = bias_part
        quality_arr[start:stop] = quality_part
        end_idx_arr[start:stop] = end_part
        path_outcome_arr[start:stop] = path_part
    return bias_arr, quality_arr, end_idx_arr, path_outcome_arr


# ── FIX-11: Kalman trend gate ─────────────────────────────────────────────────

def _apply_trend_filter(
    bias_arr: np.ndarray,
    trend_lbl: np.ndarray,
    trend_strength: Optional[np.ndarray] = None,
    strict: bool = False,
    trend_strength_min: float = 0.05,
) -> np.ndarray:
    """
    FIX-11: Suppress counter-trend labels using Kalman trend direction.

    Rules:
      - LONG  label where Kalman trend == TREND_DOWN and the opposite trend is
        sufficiently strong → NEUTRAL
      - SHORT label where Kalman trend == TREND_UP and the opposite trend is
        sufficiently strong → NEUTRAL
      - NEUTRAL labels: unchanged (unless strict=True, then also filtered)

    This is a post-labeling mask, not a feature — it fires before the model
    ever sees the data, removing label noise at the source.

    With strict=False (default) the model still sees NEUTRAL rows in both
    trend directions, letting it learn regime transitions.
    With strict=True every neutral row is also aligned to trend direction,
    further reducing noise at the cost of sample count.

    Parameters
    ----------
    bias_arr  : int8 array (DIR_LONG / DIR_SHORT / DIR_NEUTRAL)
    trend_lbl : int8 array (TREND_UP / TREND_DOWN / TREND_NEUTRAL)
    trend_strength : normalized trend strength in [0, 1], optional.
    strict    : bool — see above
    trend_strength_min : minimum opposite-trend strength required to veto a
                         directional label. Weak / neutral trends no longer
                         erase directional labels by themselves.

    Returns
    -------
    filtered bias_arr (new array, original untouched)
    """
    filtered = np.asarray(bias_arr, dtype=np.int8).copy()
    trend_lbl = np.asarray(trend_lbl, dtype=np.int8)

    if trend_strength is not None:
        strength_arr = np.asarray(trend_strength, dtype=np.float32)
        strong_counter_mask = strength_arr >= max(float(trend_strength_min), 0.0)
    else:
        strong_counter_mask = np.ones_like(filtered, dtype=bool)

    mask_bad_long  = (filtered == DIR_LONG)  & (trend_lbl == TREND_DOWN)
    mask_bad_short = (filtered == DIR_SHORT) & (trend_lbl == TREND_UP)
    filtered[(mask_bad_long | mask_bad_short) & strong_counter_mask] = DIR_NEUTRAL

    if strict:
        filtered[(filtered != DIR_NEUTRAL) & (trend_lbl == TREND_NEUTRAL)] = DIR_NEUTRAL

    return filtered


def _bias_slice_counts(values: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(values, dtype=np.int8)
    total = int(arr.size)
    n_long = int((arr == BIAS_LONG).sum())
    n_short = int((arr == BIAS_SHORT).sum())
    n_neutral = int((arr == BIAS_NEUTRAL).sum())
    return total, n_long, n_short, n_neutral


def _format_bias_line(
    tag: str,
    total: int,
    n_long: int,
    n_short: int,
    n_neutral: int | None = None,
) -> str:
    display_total = int(total)
    total = max(display_total, 1)
    text = (
        f"[v19] {tag:<7}→ LONG={n_long:,} ({n_long/total:.1%})  "
        f"SHORT={n_short:,} ({n_short/total:.1%})"
    )
    if n_neutral is not None:
        text += f"  NEUTRAL={n_neutral:,} ({n_neutral/total:.1%})"
    else:
        text += f"  | rows={display_total:,}"
    return text


# ── main entry point ──────────────────────────────────────────────────────────

def build_causal_event_labels(
    df: pd.DataFrame,
    horizon: int = 50,
    event_roll_window: int = 30,
    feature_roll_window: int = 150,
    direction_threshold_ticks: float = DEFAULT_V19_DIRECTION_THRESHOLD_TICKS,
    tp_mult: float = DEFAULT_V19_TP_MULT,
    sl_mult: float = 1.0,
    neutral_mult: float = 0.45,
    tick_size: float = 1e-4,
    tp_sl_threshold_mode: str = "fixed",
    tp_sl_anchor_window: int = 1000,
    # ── FIX-9 ────────────────────────────────────────────────────────────
    adaptive_horizon: bool = False,
    horizon_min_mult: float = 0.5,
    horizon_max_mult: float = 3.0,
    # ── FIX-11 ───────────────────────────────────────────────────────────
    trend_filter: bool = False,
    trend_filter_strict: bool = False,
    # ── FIX-Kalman ───────────────────────────────────────────────────────
    kalman_slope_threshold: float = 0.05,
    # رُفع من 1e-5 (≈ صفر بعد التطبيع) → 0.05 = 5% من أقوى ميل مرصود
    # يجعل الكالمان يُصنّف فقط الترندات الواضحة كـ UP/DOWN بدلاً من 97%
    trend_strength_min: float = 0.05,
    causal_threshold_mode: str = "fixed",
    raw_event_target_rate: float = DEFAULT_RAW_EVENT_TARGET_RATE,
    training_event_target_rate: float = DEFAULT_TRAINING_EVENT_TARGET_RATE,
    training_event_score_threshold: float | None = 0.0,
    execution_cost_pips: float = 0.0,
    enforce_economic_tp_floor: bool = False,
    n_workers: int | None = None,
    min_parallel_rows: int = 250_000,
    # الحد الأدنى لقوة الترند المعاكس لتفعيل الحذف في trend filter
    # 0.05 = نحذف counter-trend الواضح فقط، ولا نمسح الإشارات في الترند الضعيف/المحايد
) -> pd.DataFrame:
    """
    Build causal labels using unified order-book features + price-action forward scan.

    V19 includes:

    FIX-9   Adaptive horizon ∝ ATR
            effective_horizon_i = base × clip(ATR_i / ATR_median, min_mult, max_mult)
            → fast markets get longer horizon, slow markets get shorter.

    FIX-10  Per-sample dynamic threshold in forward scan
            Each row uses its own ATR-derived threshold for TP/SL calculation.
            Replaces the scalar median(dynamic_threshold) used before.

    FIX-11  Kalman trend filters labeling (not just a feature)
            LONG labels where trend==DOWN → NEUTRAL.
            SHORT labels where trend==UP  → NEUTRAL.
            trend_strength exposed for model weighting.

    Parameters
    ----------
    df                    : Raw tick/bar data.
    horizon               : Base forward-scan horizon in bars.
    event_roll_window     : Rolling window for event detection (FIX-8).
    feature_roll_window   : Rolling window for feature engineering (FIX-8).
    direction_threshold_ticks : Floor threshold in ticks (FIX-4 floor).
    tp_mult               : TP multiplier × threshold.
    sl_mult               : SL multiplier × threshold.
    neutral_mult          : Unused post-FIX-1 but kept for API compat.
    tick_size             : Minimum price increment.
    tp_sl_threshold_mode  : `fixed`, `anchor_atr`, or `atr`.
    tp_sl_anchor_window   : Anchor rows used by `anchor_atr` mode.
    adaptive_horizon      : Enable FIX-9 (default False).
    horizon_min_mult      : Floor multiplier for adaptive horizon.
    horizon_max_mult      : Ceiling multiplier for adaptive horizon.
    trend_filter          : Enable FIX-11 Kalman trend gate (default False).
    trend_filter_strict   : If True, also filter NEUTRAL rows by trend.
    trend_strength_min    : Minimum opposite-trend strength required to veto a
                            directional label.
    causal_threshold_mode : `fixed` (default) أو `expanding` للـ training-event gate.
    raw_event_target_rate : Target keep-rate for the broader `event_flag` mask.
    training_event_target_rate : Target keep-rate for the narrower
                                 `train_event_flag` mask داخل `event_flag`.
    execution_cost_pips    : Effective round-trip execution cost in tick-sized
                             price units. Used to prevent economically tiny TP
                             labels that the live policy would later reject.
    enforce_economic_tp_floor : If True, raise the TP floor so it cannot fall
                                below `stop_floor + execution_cost_pips`.
    """

    out = df.copy()
    n   = len(out)
    if n == 0:
        return _empty_output(out)
    step4_t0 = time.perf_counter()

    def _log_step4(msg: str) -> None:
        elapsed = time.perf_counter() - step4_t0
        print(f"  ⏳ Step 4: {msg} | t={elapsed:,.1f}s", flush=True)

    # ── 0. resolve price column ───────────────────────────────────────────────
    _log_step4(f"start rows={n:,}")
    price_col = (
        "price"       if "price"       in out.columns else
        "close"       if "close"       in out.columns else
        "micro_price"
    )
    out[price_col] = (
        pd.to_numeric(out.get(price_col, pd.Series(np.zeros(n))), errors="coerce")
        .ffill().fillna(0.0).astype(np.float32)
    )
    out["close"]  = out[price_col].astype(np.float32)
    out["volume"] = (
        pd.to_numeric(
            out.get("size", out.get("volume", pd.Series(np.zeros(n)))),
            errors="coerce",
        )
        .fillna(0.0).astype(np.float32)
    )
    out["cvd"] = (
        pd.to_numeric(out.get("cvd", pd.Series(np.zeros(n))), errors="coerce")
        .fillna(0.0).astype(np.float32)
    )

    # ── 1. dynamic order-book features ───────────────────────────────────────
    dynamic_cols = [
        "micro_price", "bid_wall_strength", "ask_wall_strength",
        "distance_to_wall", "gap_size", "liquidity_density",
    ]
    has_dynamic = all(c in out.columns for c in dynamic_cols)
    has_depth   = (
        any(c.startswith("bid_px_") for c in out.columns) and
        any(c.startswith("ask_px_") for c in out.columns)
    )

    if (not has_dynamic) and has_depth:
        out = append_dynamic_orderbook_features(
            out,
            tick_size=tick_size,
            price_col=price_col,
            cvd_col="cvd",
            volume_col="volume",
        )
    else:
        for col in dynamic_cols:
            if col not in out.columns:
                out[col] = 0.0
        if "micro_price" in out.columns:
            out["micro_price"] = (
                pd.to_numeric(out["micro_price"], errors="coerce")
                .fillna(out["close"]).astype(np.float32)
            )

    # ── 2. FIX-8: DUAL-WINDOW ────────────────────────────────────────────────
    feat_window = max(int(feature_roll_window), 20)
    ev_window   = max(int(event_roll_window),   10)

    _log_step4(f"engineer_features window={feat_window}")
    out = engineer_features(out, roll_window=feat_window)
    if "trend_strength" not in out.columns:
        out["trend_strength"] = np.zeros(n, dtype=np.float32)
    _log_step4("engineer_features done")

    # ── 3. FIX-1: broad event filter + stronger training gate ───────────────
    vol_mult = 1.10
    obi_thr  = 0.08
    wall_thr = 0.70

    _log_step4(f"event gates window={ev_window}")
    _log_step4("event gates 1/3 raw_event_filter start")
    raw_event_mask = build_event_filter(
        out,
        vol_mult=vol_mult,
        obi_thr=obi_thr,
        wall_str_thr=wall_thr,
        roll_window=ev_window,
    )
    raw_keep_rate = float(np.mean(np.asarray(raw_event_mask, dtype=np.float32))) if len(raw_event_mask) else 0.0
    _log_step4(f"event gates 1/3 raw_event_filter done keep={raw_keep_rate:.1%}")
    _log_step4("event gates 2/3 broad_event_gate start")
    event_flag, _, _, raw_event_score_threshold = _build_broad_event_gate(
        out,
        base_event_mask=raw_event_mask,
        roll_window=ev_window,
        vol_mult=vol_mult,
        obi_thr=obi_thr,
        wall_thr=wall_thr,
        target_rate=raw_event_target_rate,
        causal_threshold_mode=causal_threshold_mode,
    )
    broad_keep_rate = float(np.mean(np.asarray(event_flag, dtype=np.float32))) if len(event_flag) else 0.0
    _log_step4(
        f"event gates 2/3 broad_event_gate done keep={broad_keep_rate:.1%} "
        f"| score_thr={float(raw_event_score_threshold):.3f}"
    )
    _log_step4("event gates 3/3 training_event_gate start")
    train_event_flag, event_score, event_trigger_count, event_score_threshold = _build_training_event_gate(
        out,
        base_event_mask=event_flag.astype(bool),
        roll_window=ev_window,
        vol_mult=vol_mult,
        obi_thr=obi_thr,
        wall_thr=wall_thr,
        target_rate=training_event_target_rate,
        causal_threshold_mode=causal_threshold_mode,
        fixed_score_threshold=training_event_score_threshold,
    )
    train_keep_rate = float(np.mean(np.asarray(train_event_flag, dtype=np.float32))) if len(train_event_flag) else 0.0
    _log_step4(
        f"event gates 3/3 training_event_gate done keep={train_keep_rate:.1%} "
        f"| score_thr={float(event_score_threshold):.3f}"
    )
    _log_step4("event gates done")

    # ── 4. ATR: compute once, used by FIX-9 and FIX-10 ───────────────────────
    _log_step4("ATR + adaptive horizons")
    prices_arr = out["close"].astype(np.float64).values
    micro_atr  = (
        pd.to_numeric(out.get("micro_atr", pd.Series(np.ones(n))), errors="coerce")
        .fillna(np.nan).values
    )
    if np.isnan(micro_atr).all():
        micro_atr = _compute_micro_atr(prices_arr, window=feat_window)

    # FIX-10 ablation/stability mode:
    #   - fixed:      use one stationary tick-floor threshold across rows
    #   - anchor_atr: use one stationary threshold anchored to early ATR only
    #   - atr:        keep the older per-row ATR-adaptive threshold
    fixed_floor       = direction_threshold_ticks * tick_size
    tp_sl_mode = str(tp_sl_threshold_mode or "fixed").strip().lower()
    anchor_rows = int(max(tp_sl_anchor_window, 32))
    anchor_limit = int(min(max(anchor_rows, 1), n))
    anchor_slice = np.asarray(micro_atr[:anchor_limit], dtype=np.float64)
    anchor_valid = anchor_slice[np.isfinite(anchor_slice) & (anchor_slice > 0)]
    anchor_atr = float(np.nanmedian(anchor_valid)) if anchor_valid.size else float('nan')
    if tp_sl_mode == "atr":
        dynamic_threshold = np.maximum(fixed_floor, 0.5 * micro_atr)
    elif tp_sl_mode == "fixed":
        dynamic_threshold = np.full(n, max(fixed_floor, tick_size), dtype=np.float64)
    elif tp_sl_mode == "anchor_atr":
        anchor_threshold = max(fixed_floor, 0.5 * anchor_atr) if np.isfinite(anchor_atr) and anchor_atr > 0 else fixed_floor
        dynamic_threshold = np.full(n, max(anchor_threshold, tick_size), dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported tp_sl_threshold_mode={tp_sl_threshold_mode!r}. "
            "Use 'fixed', 'anchor_atr', or 'atr'."
        )
    label_economics = _resolve_economic_tp_floor_ticks(
        direction_threshold_ticks=direction_threshold_ticks,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        execution_cost_pips=execution_cost_pips,
        enforce_economic_tp_floor=enforce_economic_tp_floor,
    )
    economic_tp_floor_ticks = float(label_economics["economic_tp_floor_ticks"])
    if bool(enforce_economic_tp_floor) and bool(label_economics["economic_floor_applied"]):
        _log_step4(
            "economic TP floor uplift "
            f"cost_ticks={float(label_economics['effective_cost_ticks']):.2f} | "
            f"stop_floor_ticks={float(label_economics['stop_floor_ticks']):.2f} | "
            f"base_tp_floor_ticks={float(label_economics['base_tp_floor_ticks']):.2f} -> "
            f"economic_tp_floor_ticks={economic_tp_floor_ticks:.2f}"
        )
    else:
        _log_step4(
            "economic TP guard "
            f"cost_ticks={float(label_economics['effective_cost_ticks']):.2f} | "
            f"base_tp_floor_ticks={float(label_economics['base_tp_floor_ticks']):.2f} | "
            f"economic_tp_floor_ticks={economic_tp_floor_ticks:.2f} | "
            f"enforced={bool(enforce_economic_tp_floor)}"
        )

    # ── 5. FIX-9: adaptive horizon ∝ ATR ─────────────────────────────────────
    #
    #  v21: effective_horizon = int(horizon × 1.5)  — same for all rows
    #  v19: effective_horizon_i = base × clip(ATR_i / ATR_med, min, max)
    #
    #  Why proportional ATR?
    #    High ATR → price moves fast → needs MORE bars to reach TP → longer horizon
    #    Low ATR  → price moves slow → DON'T wait → shorter horizon
    #
    if adaptive_horizon:
        adaptive_horizons = _compute_adaptive_horizons(
            micro_atr,
            base_horizon=horizon,
            min_mult=horizon_min_mult,
            max_mult=horizon_max_mult,
        )
    else:
        # FIX-5 fallback: static ×1.5 (v21 behaviour)
        adaptive_horizons = np.full(n, int(horizon * 1.5), dtype=np.int32)

    nan_series = pd.Series(np.full(n, np.nan), index=out.index, dtype=np.float64)
    bid_wall_px_arr = pd.to_numeric(out.get("bid_wall_px", nan_series), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ask_wall_px_arr = pd.to_numeric(out.get("ask_wall_px", nan_series), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    bid_wall_strength_arr = pd.to_numeric(out.get("bid_wall_strength", nan_series), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ask_wall_strength_arr = pd.to_numeric(out.get("ask_wall_strength", nan_series), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    _log_step4("ATR + adaptive horizons done")
    approx_price_checks = int(np.asarray(adaptive_horizons, dtype=np.int64).sum()) * 2
    requested_workers = 1 if n_workers is None else max(1, int(n_workers))
    _log_step4(
        "forward scan plan "
        f"workers={requested_workers} | rows={n:,} | "
        f"horizon_med={int(np.median(adaptive_horizons)):,} | "
        f"horizon_max={int(np.max(adaptive_horizons)):,} | "
        f"approx_price_checks≈{approx_price_checks:,}"
    )

    # ── 6. FIX-10: per-row forward scan ──────────────────────────────────────
    #
    #  v21: label_with_forward_scan(scalar_threshold)
    #       → median collapsed all per-row ATR info into one number
    #
    #  v19: _forward_scan_per_row(dynamic_threshold[i], adaptive_horizons[i])
    #       → each row evaluated against its own ATR threshold and horizon
    #
    _log_step4("forward scan start")
    bias_raw, quality_raw, end_idx_arr, path_outcome_raw = _forward_scan_per_row(
        prices     = prices_arr,
        dynamic_threshold = dynamic_threshold,
        adaptive_horizons = adaptive_horizons,
        tick_size  = tick_size,
        tp_mult    = tp_mult,
        sl_mult    = sl_mult,
        bid_wall_px = bid_wall_px_arr,
        ask_wall_px = ask_wall_px_arr,
        bid_wall_strength = bid_wall_strength_arr,
        ask_wall_strength = ask_wall_strength_arr,
        economic_tp_floor_ticks = economic_tp_floor_ticks,
        n_workers = n_workers,
        min_parallel_rows = min_parallel_rows,
    )
    _log_step4("forward scan done")

    # Merge into labeled DataFrame (keeping all columns from engineer_features)
    labeled = out.copy()
    labeled["bias_label"]     = bias_raw.astype(np.int8)
    labeled["signal_quality"] = quality_raw.astype(np.int8)
    labeled["path_outcome"]   = path_outcome_raw.astype(np.int8)
    labeled["adverse_path_flag"] = np.isin(
        path_outcome_raw,
        np.array([PATH_LONG_SL_FIRST, PATH_SHORT_SL_FIRST], dtype=np.int8),
    ).astype(np.int8)
    labeled["bias_label_detail"] = np.select(
        [
            path_outcome_raw == PATH_LONG_TP_FIRST,
            path_outcome_raw == PATH_SHORT_TP_FIRST,
            path_outcome_raw == PATH_TIMEOUT,
            path_outcome_raw == PATH_LONG_SL_FIRST,
            path_outcome_raw == PATH_SHORT_SL_FIRST,
        ],
        [
            DETAIL_LONG,
            DETAIL_SHORT,
            DETAIL_TIMEOUT,
            DETAIL_LONG_SL,
            DETAIL_SHORT_SL,
        ],
        default=DETAIL_TIMEOUT,
    ).astype(np.int8)
    labeled["neutral_reason"] = np.select(
        [
            path_outcome_raw == PATH_TIMEOUT,
            path_outcome_raw == PATH_LONG_SL_FIRST,
            path_outcome_raw == PATH_SHORT_SL_FIRST,
        ],
        [
            NEUTRAL_REASON_TIMEOUT,
            NEUTRAL_REASON_LONG_SL,
            NEUTRAL_REASON_SHORT_SL,
        ],
        default=NEUTRAL_REASON_NONE,
    ).astype(np.int8)

    # Populate ancillary forward-scan columns expected downstream
    labeled["long_label"]  = np.where(
        path_outcome_raw == PATH_LONG_TP_FIRST,
        LABEL_WIN,
        np.where(path_outcome_raw == PATH_LONG_SL_FIRST, LABEL_LOSE, LABEL_CANCEL),
    ).astype(np.int8)
    labeled["short_label"] = np.where(
        path_outcome_raw == PATH_SHORT_TP_FIRST,
        LABEL_WIN,
        np.where(path_outcome_raw == PATH_SHORT_SL_FIRST, LABEL_LOSE, LABEL_CANCEL),
    ).astype(np.int8)

    # ── 7. FIX-3: replace any residual QUALITY_NONE with QUALITY_WEAK ─────────
    if "signal_quality" in labeled.columns:
        sq = labeled["signal_quality"].astype(np.int8)
        labeled["signal_quality"] = np.where(
            sq == QUALITY_NONE, QUALITY_WEAK, sq
        ).astype(np.int8)

    # ── 8. FIX-11: Kalman trend — compute THEN filter labels ─────────────────
    #
    #  v21: kalman_trend() computed trend_label but never used it in labeling.
    #       It was stored as a feature and ignored during label generation.
    #
    #  v19: THREE-step process:
    #    (a) Compute Kalman trend (smoothed price slope → direction)
    #    (b) Store trend_label + trend_strength as ML features
    #    (c) Apply post-label mask: counter-trend labels → NEUTRAL
    #
    #  Scientific basis:
    #    An absorption signal in a DOWN trend is NOT a high-probability LONG.
    #    Including it as LONG adds noise. Filtering it:
    #      - Reduces label noise (cleaner training signal)
    #      - Improves precision at the cost of recall (acceptable trade-off)
    #      - Lets model learn: "signal + trend alignment = conviction"
    #
    trend_runtime_available = bool(callable(kalman_trend))
    try:
        if not trend_runtime_available:
            raise RuntimeError("kalman_trend unavailable in modules.dynamic_labels")
        trend_lbl, trend_strength, kalman_price = kalman_trend(prices_arr, slope_threshold=kalman_slope_threshold)
    except Exception:
        # kalman_trend not yet implemented → graceful fallback
        trend_runtime_available = False
        trend_lbl      = np.full(n, TREND_NEUTRAL, dtype=np.int8)
        trend_strength = np.zeros(n, dtype=np.float32)
        kalman_price   = prices_arr.astype(np.float32)

    labeled["kalman_trend_label"] = trend_lbl.astype(np.int8)
    labeled["kalman_trend_strength"] = trend_strength.astype(np.float32)
    labeled["kalman_price"] = kalman_price.astype(np.float32)

    trend_filter_applied = bool(trend_filter and trend_runtime_available)
    if trend_filter_applied:
        bias_filtered = _apply_trend_filter(
            labeled["bias_label"].values.astype(np.int8),
            trend_lbl,
            trend_strength=trend_strength,
            strict=trend_filter_strict,
            trend_strength_min=trend_strength_min,
        )
        labeled["bias_label"] = bias_filtered.astype(np.int8)
        # Re-sync long_label / short_label after trend filter
        labeled["long_label"]  = np.where(
            labeled["bias_label"] == DIR_LONG,
            np.where(labeled["path_outcome"].astype(np.int8) == PATH_LONG_TP_FIRST, LABEL_WIN, LABEL_CANCEL),
            LABEL_CANCEL,
        ).astype(np.int8)
        labeled["short_label"] = np.where(
            labeled["bias_label"] == DIR_SHORT,
            np.where(labeled["path_outcome"].astype(np.int8) == PATH_SHORT_TP_FIRST, LABEL_WIN, LABEL_CANCEL),
            LABEL_CANCEL,
        ).astype(np.int8)

    # ── 9. timestamps ─────────────────────────────────────────────────────────
    ts_raw = labeled.get("ts_event", pd.Series(pd.RangeIndex(n)))
    ts = _coerce_label_timestamp_series(ts_raw, context='labels_v19.ts_event')

    end_idx = np.minimum(end_idx_arr, n - 1).astype(np.int32)
    prices  = labeled["close"].astype(np.float64).values
    terminal_abs_move = np.abs(prices[end_idx] - prices)
    timeout_band = np.maximum(float(neutral_mult), 0.0) * dynamic_threshold
    timeout_exceeded_mask = (
        (labeled["path_outcome"].to_numpy(dtype=np.int8, copy=False) == PATH_TIMEOUT)
        & (terminal_abs_move >= timeout_band)
    )
    labeled["timeout_move_exceeded_band"] = timeout_exceeded_mask.astype(np.int8)

    # ── 10. derived columns ───────────────────────────────────────────────────
    labeled["setup_label"] = _infer_setup_labels(labeled)
    labeled["conf_label"]  = _quality_to_conf(labeled["signal_quality"])

    labeled["is_expansion"] = (
        (labeled["bias_label"].astype(np.int8)      != BIAS_NEUTRAL)
        & (labeled["signal_quality"].astype(np.int8) == QUALITY_STRONG)
    ).astype(np.int8)

    labeled["liq_score"]           = _liquidity_score(labeled)
    labeled["regime_label"]        = (
        pd.to_numeric(labeled.get("regime", pd.Series(np.full(n, REGIME_RANGING, dtype=np.int8))), errors="coerce")
        .fillna(REGIME_RANGING).astype(np.int8)
    )
    labeled["label_end_ts"]        = pd.to_datetime(ts.iloc[end_idx].to_numpy())
    labeled["forward_return"]      = (prices[end_idx] - prices).astype(np.float32)
    labeled["label_horizon_steps"] = (end_idx - np.arange(n)).astype(np.int32)
    labeled["effective_horizon"]   = adaptive_horizons.astype(np.int32)   # FIX-9: expose per-row

    # FIX-6: broad event flag as context feature + stricter train-event gate
    labeled["event_flag"] = np.asarray(event_flag, dtype=np.int8)
    labeled["train_event_flag"] = train_event_flag.astype(np.int8)
    labeled["event_score"] = event_score.astype(np.float32)
    labeled["event_trigger_count"] = event_trigger_count.astype(np.int8)
    labeled["is_event"]   = labeled["event_flag"]

    # ── 11. diagnostics ───────────────────────────────────────────────────────
    total       = max(n, 1)
    bias_counts = labeled["bias_label"].value_counts()
    qual_counts = labeled["signal_quality"].value_counts()

    n_long    = bias_counts.get(BIAS_LONG,    0)
    n_short   = bias_counts.get(BIAS_SHORT,   0)
    n_neutral = bias_counts.get(BIAS_NEUTRAL, 0)
    n_strong  = qual_counts.get(QUALITY_STRONG, 0)
    n_weak    = qual_counts.get(QUALITY_WEAK,   0)
    n_none    = qual_counts.get(QUALITY_NONE,   0)
    n_events  = int(labeled["event_flag"].sum())
    n_train_events = int(labeled["train_event_flag"].sum())
    n_up      = int((labeled["kalman_trend_label"] == TREND_UP).sum())
    n_down    = int((labeled["kalman_trend_label"] == TREND_DOWN).sum())
    n_trend_neutral = int((labeled["kalman_trend_label"] == TREND_NEUTRAL).sum())
    n_directional = int(n_long + n_short)
    event_slice = labeled[labeled["event_flag"].astype(np.int8) == 1]
    train_event_slice = labeled[labeled["train_event_flag"].astype(np.int8) == 1]
    directional_event_slice = event_slice[event_slice["bias_label"].astype(np.int8) != BIAS_NEUTRAL]
    directional_train_slice = train_event_slice[train_event_slice["bias_label"].astype(np.int8) != BIAS_NEUTRAL]

    evt_total, evt_long, evt_short, evt_neutral = _bias_slice_counts(
        event_slice["bias_label"].values.astype(np.int8)
        if len(event_slice)
        else np.array([], dtype=np.int8)
    )
    evt_dir_total, evt_dir_long, evt_dir_short, _ = _bias_slice_counts(
        directional_event_slice["bias_label"].values.astype(np.int8)
        if len(directional_event_slice)
        else np.array([], dtype=np.int8)
    )
    train_total, train_long, train_short, train_neutral = _bias_slice_counts(
        train_event_slice["bias_label"].values.astype(np.int8)
        if len(train_event_slice)
        else np.array([], dtype=np.int8)
    )
    train_dir_total, train_dir_long, train_dir_short, _ = _bias_slice_counts(
        directional_train_slice["bias_label"].values.astype(np.int8)
        if len(directional_train_slice)
        else np.array([], dtype=np.int8)
    )

    h_med = int(np.median(adaptive_horizons))
    h_min = int(adaptive_horizons.min())
    h_max = int(adaptive_horizons.max())

    print(
        "[v19] Note   → Step 4 shows raw row-level causal labels, "
        "not 5m CatBoost prediction mix"
    )
    print(
        f"[v19] Config → thr_ticks={direction_threshold_ticks:.2f}  "
        f"tp_mult={tp_mult:.2f}  sl_mult={sl_mult:.2f}  "
        f"tp_sl_mode={tp_sl_mode}  "
        f"kalman_thr={kalman_slope_threshold:.2f}  trend_min={trend_strength_min:.2f}"
    )
    if tp_sl_mode == "anchor_atr":
        print(
            f"[v19] TP/SL  → anchor_rows={anchor_limit:,}  "
            f"anchor_atr={0.0 if not np.isfinite(anchor_atr) else float(anchor_atr):.8f}"
        )
    print(
        _format_bias_line("BiasAll", total, n_long, n_short, n_neutral)
    )
    print(
        _format_bias_line("BiasEvt", evt_total, evt_long, evt_short, evt_neutral)
    )
    print(
        _format_bias_line("BiasDir", evt_dir_total, evt_dir_long, evt_dir_short)
    )
    print(
        _format_bias_line("BiasTrn", train_total, train_long, train_short, train_neutral)
    )
    print(
        _format_bias_line("BiasSel", train_dir_total, train_dir_long, train_dir_short)
    )
    print(
        f"[v19] Qual   → STRONG={n_strong:,} ({n_strong/total:.1%})  "
        f"WEAK={n_weak:,} ({n_weak/total:.1%})  "
        f"NONE={n_none:,} (target=0)"
    )
    print(
        f"[v19] Events → raw={n_events:,}/{total:,} ({n_events/total:.1%})  "
        f"train={n_train_events:,}/{total:,} ({n_train_events/total:.1%})  "
        f"| feat_win={feat_window}  ev_win={ev_window}"
    )
    print(
        f"[v19] Gate   → raw_score_thr={raw_event_score_threshold:.3f}  "
        f"train_score_thr={event_score_threshold:.3f}  "
        f"raw_target={float(raw_event_target_rate):.2f}  "
        f"train_target={float(training_event_target_rate):.2f}  "
        f"avg_train_score={float(np.nanmean(event_score)):.3f}  "
        f"max_triggers={int(event_trigger_count.max()) if len(event_trigger_count) else 0}"
    )
    print(
        f"[v19] Trend  → status={'APPLIED' if trend_filter_applied else ('UNAVAILABLE' if trend_filter else 'OFF')}  "
        f"UP={n_up:,} ({n_up/total:.1%})  "
        f"DOWN={n_down:,} ({n_down/total:.1%})  "
        f"NEUTRAL={n_trend_neutral:,} ({n_trend_neutral/total:.1%})"
    )
    print(
        f"[v19] Paths  → long_tp={int((labeled['path_outcome'] == PATH_LONG_TP_FIRST).sum()):,}  "
        f"short_tp={int((labeled['path_outcome'] == PATH_SHORT_TP_FIRST).sum()):,}  "
        f"long_sl={int((labeled['path_outcome'] == PATH_LONG_SL_FIRST).sum()):,}  "
        f"short_sl={int((labeled['path_outcome'] == PATH_SHORT_SL_FIRST).sum()):,}  "
        f"timeout={int((labeled['path_outcome'] == PATH_TIMEOUT).sum()):,}  "
        f"| timeout>|neutral_band|={int(timeout_exceeded_mask.sum()):,}"
    )
    print(
        f"[v19] Horizon→ adaptive={'ON' if adaptive_horizon else 'OFF'}  "
        f"med={h_med}  min={h_min}  max={h_max}  base={horizon}"
    )

    if n_directional / total < 0.05:
        print(
            f"[v19] Warn   → directional labels are sparse: "
            f"{n_directional:,}/{total:,} ({n_directional/total:.1%})"
        )
    if n_events / total > 0.90:
        print(
            f"[v19] Warn   → event filter is permissive: "
            f"{n_events:,}/{total:,} ({n_events/total:.1%})"
        )
    if 0 < n_train_events / total < 0.05:
        print(
            f"[v19] Warn   → training-event gate is too sparse: "
            f"{n_train_events:,}/{total:,} ({n_train_events/total:.1%})"
        )
    if trend_filter and not trend_runtime_available:
        print("[v19] Warn   → kalman_trend unavailable in dynamic_labels.py; trend filter skipped")

    if n_none > 0:
        warnings.warn(
            f"[v19] {n_none} rows still have QUALITY_NONE after FIX-3 — "
            "check label_with_forward_scan internals.",
            RuntimeWarning,
            stacklevel=2,
        )

    return labeled
