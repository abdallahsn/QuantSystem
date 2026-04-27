"""
labels_v22.py - Unified causal labels for QuantSystem
======================================================
v22 fixes three structural flaws that remained in v21:

  FIX-9   Adaptive horizon ∝ ATR (was static ×1.5)
          horizon now scales per volatility regime:
            slow market  → shorter horizon (don't wait forever)
            fast market  → longer horizon  (give the move space)
          Formula: effective_horizon_i = clip(base × (ATR_i / ATR_median), min, max)

  FIX-10  Per-sample dynamic threshold in forward scan
          v21 used median(dynamic_threshold) — one scalar for the entire dataset.
          v22 patches label_with_forward_scan per-row via a vectorised pre-pass
          that computes TP/SL hit indices directly, then assembles labels without
          needing a scalar threshold. The forward scan is now truly ATR-per-row.

  FIX-11  Kalman trend actively filters labeling (not just a feature)
          v21 computed trend_label / trend_entry_label but never used them to
          gate bias_label. v22 applies a post-labeling mask:
            - LONG  labels where Kalman trend == DOWN  → forced NEUTRAL
            - SHORT labels where Kalman trend == UP    → forced NEUTRAL
          This removes counter-trend noise before the model ever sees the data.
          Trend strength (kalman_slope magnitude) is exposed as trend_strength
          so the model can learn "aligned + strong trend = higher conviction".

All v21 fixes (FIX-1 … FIX-8) are preserved unchanged.
External API is 100% backwards-compatible: build_causal_event_labels(df, ...).

New optional parameters:
  adaptive_horizon     : bool  = True   — enable FIX-9 (disable for ablation)
  horizon_min_mult     : float = 0.5    — floor: horizon never < base × 0.5
  horizon_max_mult     : float = 3.0    — ceiling: horizon never > base × 3.0
  trend_filter         : bool  = True   — enable FIX-11 (disable for ablation)
  trend_filter_strict  : bool  = False  — if True, neutral rows also filtered by trend
"""

from __future__ import annotations

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

DEFAULT_V22_DIRECTION_THRESHOLD_TICKS = 1.5
DEFAULT_V22_TP_MULT = 1.5


# ── helpers ───────────────────────────────────────────────────────────────────

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
    out["event_flag"]           = empty_int8
    out["train_event_flag"]     = empty_int8
    out["event_score"]          = empty_float32
    out["event_trigger_count"]  = empty_int8
    out["is_event"]             = empty_int8
    out["trend_label"]          = empty_int8
    out["trend_strength"]       = empty_float32
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
            past_scores: list[float] = []
            past_base_count = 0
            for i in range(n):
                if not base_mask[i]:
                    continue
                if past_base_count <= 0 or len(past_scores) < max(int(roll_window), 10):
                    current_threshold = 0.0
                else:
                    base_rate = float(past_base_count) / max(float(i), 1.0)
                    desired_rate = min(max(float(target_rate), 0.05), base_rate)
                    keep_inside_base = min(max(desired_rate / max(base_rate, 1e-9), 0.0), 1.0)
                    if keep_inside_base >= 0.999:
                        current_threshold = float(np.nanmin(past_scores))
                    else:
                        q = min(max(1.0 - keep_inside_base, 0.0), 1.0)
                        current_threshold = float(np.nanquantile(np.asarray(past_scores, dtype=np.float64), q))
                threshold = float(max(current_threshold, 0.0))
                candidate[i] = bool(score[i] >= threshold)
                past_scores.append(float(score[i]))
                past_base_count += 1
        if not candidate.any():
            fallback = base_mask & (trigger_count >= 2)
            candidate = fallback if fallback.any() else base_mask.copy()
            threshold = float(np.nanmin(score[candidate])) if candidate.any() else threshold

    return candidate.astype(np.int8), score, trigger_count, threshold


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

def _forward_scan_per_row(
    prices: np.ndarray,
    dynamic_threshold: np.ndarray,
    adaptive_horizons: np.ndarray,
    tick_size: float,
    tp_mult: float = 2.5,                  # FIX: كان 1.5 → TP ≈ 20-30pip
    sl_mult: float = 1.0,
    bid_wall_px: Optional[np.ndarray] = None,
    ask_wall_px: Optional[np.ndarray] = None,
    bid_wall_strength: Optional[np.ndarray] = None,
    ask_wall_strength: Optional[np.ndarray] = None,
    min_wall_strength: float = 2.5,
    wall_exit_buffer_ticks: float = 1.0,
    min_wall_tp_ticks: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FIX-10: Vectorised per-row forward scan with row-specific threshold AND horizon.

    v21 problem:
        label_with_forward_scan() expected a single scalar threshold.
        np.median(dynamic_threshold) collapsed all per-row ATR information
        into one number — making the "dynamic" threshold effectively static.

    v22 solution:
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
    bias_arr     : int8 array  — DIR_LONG / DIR_SHORT / DIR_NEUTRAL
    quality_arr  : int8 array  — QUALITY_STRONG / QUALITY_WEAK
    end_idx_arr  : int32 array — index where scan terminated
    """
    n          = len(prices)
    bias_arr   = np.full(n, DIR_NEUTRAL,    dtype=np.int8)
    quality_arr= np.full(n, QUALITY_WEAK,   dtype=np.int8)
    end_idx_arr= np.arange(n,              dtype=np.int32)

    for i in range(n):
        p0  = prices[i]
        thr = max(dynamic_threshold[i], tick_size)
        tp  = thr * tp_mult
        sl  = thr * sl_mult
        h   = int(adaptive_horizons[i])
        end = min(i + h, n - 1)
        tp_long = tp
        tp_short = tp

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
            min_tp_ticks=min_wall_tp_ticks,
        )
        if liquidity_tp_long is not None:
            # Wall acts as a TP cap: don't force price through the first barrier.
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
            min_tp_ticks=min_wall_tp_ticks,
        )
        if liquidity_tp_short is not None:
            tp_short = min(tp_short, liquidity_tp_short)

        tp_long_hit  = False
        sl_long_hit  = False
        tp_short_hit = False
        sl_short_hit = False
        first_long_hit_idx  = end
        first_short_hit_idx = end

        for j in range(i + 1, end + 1):
            move = prices[j] - p0
            if not tp_long_hit and move >= tp_long:
                tp_long_hit = True
                first_long_hit_idx = j
                break                          # LONG TP → best case, stop here
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

        # Determine direction
        if tp_long_hit and (not tp_short_hit or first_long_hit_idx <= first_short_hit_idx):
            bias_arr[i]    = DIR_LONG
            quality_arr[i] = QUALITY_STRONG
            end_idx_arr[i] = first_long_hit_idx
        elif tp_short_hit:
            bias_arr[i]    = DIR_SHORT
            quality_arr[i] = QUALITY_STRONG
            end_idx_arr[i] = first_short_hit_idx
        elif sl_long_hit or sl_short_hit:
            # SL hit first → WEAK signal in whichever direction lost
            quality_arr[i] = QUALITY_WEAK
            end_idx_arr[i] = min(first_long_hit_idx, first_short_hit_idx)
        # else: no move within horizon → NEUTRAL / WEAK (defaults above)

    return bias_arr, quality_arr, end_idx_arr


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
        f"[v22] {tag:<7}→ LONG={n_long:,} ({n_long/total:.1%})  "
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
    direction_threshold_ticks: float = DEFAULT_V22_DIRECTION_THRESHOLD_TICKS,
    tp_mult: float = DEFAULT_V22_TP_MULT,
    sl_mult: float = 1.0,
    neutral_mult: float = 0.45,
    tick_size: float = 1e-4,
    # ── FIX-9 ────────────────────────────────────────────────────────────
    adaptive_horizon: bool = True,
    horizon_min_mult: float = 0.5,
    horizon_max_mult: float = 3.0,
    # ── FIX-11 ───────────────────────────────────────────────────────────
    trend_filter: bool = True,
    trend_filter_strict: bool = False,
    # ── FIX-Kalman ───────────────────────────────────────────────────────
    kalman_slope_threshold: float = 0.05,
    # رُفع من 1e-5 (≈ صفر بعد التطبيع) → 0.05 = 5% من أقوى ميل مرصود
    # يجعل الكالمان يُصنّف فقط الترندات الواضحة كـ UP/DOWN بدلاً من 97%
    trend_strength_min: float = 0.05,
    causal_threshold_mode: str = "expanding",
    training_event_score_threshold: float | None = None,
    # الحد الأدنى لقوة الترند المعاكس لتفعيل الحذف في trend filter
    # 0.05 = نحذف counter-trend الواضح فقط، ولا نمسح الإشارات في الترند الضعيف/المحايد
) -> pd.DataFrame:
    """
    Build causal labels using unified order-book features + price-action forward scan.

    v22 fixes (on top of all v21 fixes):

    FIX-9   Adaptive horizon ∝ ATR
            effective_horizon_i = base × clip(ATR_i / ATR_median, min_mult, max_mult)
            → fast markets get longer horizon, slow markets get shorter.

    FIX-10  Per-sample dynamic threshold in forward scan
            Each row uses its own ATR-derived threshold for TP/SL calculation.
            Replaces the scalar median(dynamic_threshold) used in v21.

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
    tp_mult               : TP multiplier × ATR threshold.
    sl_mult               : SL multiplier × ATR threshold.
    neutral_mult          : Unused post-FIX-1 but kept for API compat.
    tick_size             : Minimum price increment.
    adaptive_horizon      : Enable FIX-9 (default True).
    horizon_min_mult      : Floor multiplier for adaptive horizon.
    horizon_max_mult      : Ceiling multiplier for adaptive horizon.
    trend_filter          : Enable FIX-11 Kalman trend gate (default True).
    trend_filter_strict   : If True, also filter NEUTRAL rows by trend.
    trend_strength_min    : Minimum opposite-trend strength required to veto a
                            directional label.
    causal_threshold_mode : `expanding` (default) أو `fixed` للـ training-event gate.
    """

    out = df.copy()
    n   = len(out)
    if n == 0:
        return _empty_output(out)

    # ── 0. resolve price column ───────────────────────────────────────────────
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

    out = engineer_features(out, roll_window=feat_window)

    # ── 3. FIX-1: broad event filter + stronger training gate ───────────────
    vol_mult = 1.10
    obi_thr  = 0.08
    wall_thr = 0.70

    event_mask = build_event_filter(
        out,
        vol_mult=vol_mult,
        obi_thr=obi_thr,
        wall_str_thr=wall_thr,
        roll_window=ev_window,
    )
    train_event_flag, event_score, event_trigger_count, event_score_threshold = _build_training_event_gate(
        out,
        base_event_mask=event_mask,
        roll_window=ev_window,
        vol_mult=vol_mult,
        obi_thr=obi_thr,
        wall_thr=wall_thr,
        causal_threshold_mode=causal_threshold_mode,
        fixed_score_threshold=training_event_score_threshold,
    )

    # ── 4. ATR: compute once, used by FIX-9 and FIX-10 ───────────────────────
    prices_arr = out["close"].astype(np.float64).values
    micro_atr  = (
        pd.to_numeric(out.get("micro_atr", pd.Series(np.ones(n))), errors="coerce")
        .fillna(np.nan).values
    )
    if np.isnan(micro_atr).all():
        micro_atr = _compute_micro_atr(prices_arr, window=feat_window)

    # FIX-4: per-row dynamic threshold (floor = fixed ticks, adaptive = 0.5×ATR)
    fixed_floor       = direction_threshold_ticks * tick_size
    dynamic_threshold = np.maximum(fixed_floor, 0.5 * micro_atr)   # shape (n,)

    # ── 5. FIX-9: adaptive horizon ∝ ATR ─────────────────────────────────────
    #
    #  v21: effective_horizon = int(horizon × 1.5)  — same for all rows
    #  v22: effective_horizon_i = base × clip(ATR_i / ATR_med, min, max)
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

    # ── 6. FIX-10: per-row forward scan ──────────────────────────────────────
    #
    #  v21: label_with_forward_scan(scalar_threshold)
    #       → median collapsed all per-row ATR info into one number
    #
    #  v22: _forward_scan_per_row(dynamic_threshold[i], adaptive_horizons[i])
    #       → each row evaluated against its own ATR threshold and horizon
    #
    bias_raw, quality_raw, end_idx_arr = _forward_scan_per_row(
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
    )

    # Merge into labeled DataFrame (keeping all columns from engineer_features)
    labeled = out.copy()
    labeled["bias_label"]     = bias_raw.astype(np.int8)
    labeled["signal_quality"] = quality_raw.astype(np.int8)

    # Populate ancillary forward-scan columns expected downstream
    labeled["long_label"]  = np.where(
        bias_raw == DIR_LONG,  QUALITY_STRONG, LABEL_CANCEL
    ).astype(np.int8)
    labeled["short_label"] = np.where(
        bias_raw == DIR_SHORT, QUALITY_STRONG, LABEL_CANCEL
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
    #  v22: THREE-step process:
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

    labeled["trend_label"]    = trend_lbl.astype(np.int8)
    labeled["trend_strength"] = trend_strength.astype(np.float32)
    labeled["kalman_price"]   = kalman_price.astype(np.float32)

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
            labeled["bias_label"] == DIR_LONG,  QUALITY_STRONG, LABEL_CANCEL
        ).astype(np.int8)
        labeled["short_label"] = np.where(
            labeled["bias_label"] == DIR_SHORT, QUALITY_STRONG, LABEL_CANCEL
        ).astype(np.int8)

    # ── 9. timestamps ─────────────────────────────────────────────────────────
    ts_raw = labeled.get("ts_event", pd.Series(pd.RangeIndex(n)))
    ts     = pd.to_datetime(ts_raw, utc=True, errors="coerce").dt.tz_localize(None)
    if ts.isna().all():
        ts = pd.Series(pd.date_range("2026-01-01", periods=n, freq="s"))
    else:
        ts = ts.ffill().bfill()

    end_idx = np.minimum(end_idx_arr, n - 1).astype(np.int32)
    prices  = labeled["close"].astype(np.float64).values

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
    labeled["event_flag"] = event_mask.fillna(False).astype(np.int8)
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
    n_up      = int((labeled["trend_label"] == TREND_UP).sum())
    n_down    = int((labeled["trend_label"] == TREND_DOWN).sum())
    n_trend_neutral = int((labeled["trend_label"] == TREND_NEUTRAL).sum())
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
        "[v22] Note   → Step 4 shows raw row-level causal labels, "
        "not 5m CatBoost prediction mix"
    )
    print(
        f"[v22] Config → thr_ticks={direction_threshold_ticks:.2f}  "
        f"tp_mult={tp_mult:.2f}  sl_mult={sl_mult:.2f}  "
        f"kalman_thr={kalman_slope_threshold:.2f}  trend_min={trend_strength_min:.2f}"
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
        f"[v22] Qual   → STRONG={n_strong:,} ({n_strong/total:.1%})  "
        f"WEAK={n_weak:,} ({n_weak/total:.1%})  "
        f"NONE={n_none:,} (target=0)"
    )
    print(
        f"[v22] Events → raw={n_events:,}/{total:,} ({n_events/total:.1%})  "
        f"train={n_train_events:,}/{total:,} ({n_train_events/total:.1%})  "
        f"| feat_win={feat_window}  ev_win={ev_window}"
    )
    print(
        f"[v22] Gate   → score_thr={event_score_threshold:.3f}  "
        f"avg_score={float(np.nanmean(event_score)):.3f}  "
        f"max_triggers={int(event_trigger_count.max()) if len(event_trigger_count) else 0}"
    )
    print(
        f"[v22] Trend  → status={'APPLIED' if trend_filter_applied else ('UNAVAILABLE' if trend_filter else 'OFF')}  "
        f"UP={n_up:,} ({n_up/total:.1%})  "
        f"DOWN={n_down:,} ({n_down/total:.1%})  "
        f"NEUTRAL={n_trend_neutral:,} ({n_trend_neutral/total:.1%})"
    )
    print(
        f"[v22] Horizon→ adaptive={'ON' if adaptive_horizon else 'OFF'}  "
        f"med={h_med}  min={h_min}  max={h_max}  base={horizon}"
    )

    if n_directional / total < 0.05:
        print(
            f"[v22] Warn   → directional labels are sparse: "
            f"{n_directional:,}/{total:,} ({n_directional/total:.1%})"
        )
    if n_events / total > 0.90:
        print(
            f"[v22] Warn   → event filter is permissive: "
            f"{n_events:,}/{total:,} ({n_events/total:.1%})"
        )
    if 0 < n_train_events / total < 0.05:
        print(
            f"[v22] Warn   → training-event gate is too sparse: "
            f"{n_train_events:,}/{total:,} ({n_train_events/total:.1%})"
        )
    if trend_filter and not trend_runtime_available:
        print("[v22] Warn   → kalman_trend unavailable in dynamic_labels.py; trend filter skipped")

    if n_none > 0:
        warnings.warn(
            f"[v22] {n_none} rows still have QUALITY_NONE after FIX-3 — "
            "check label_with_forward_scan internals.",
            RuntimeWarning,
            stacklevel=2,
        )

    return labeled
