"""
mc_label_weights.py — Structural prior weights for wall / ATR-adjusted TP–SL labeling
================================================================================
Gambler's-ruin style P(hit TP before SL) from distances (tp_dist, sl_dist) and σ,
optionally scaled by horizon and wall-confidence. Separate from Monte-Carlo perturbation
on the realised path (sensitivity analysis).

Also provides a lightweight *label neighbour stability* score: agreement of bias_label
with nearby rows across fixed index shifts (heuristic smoothing / ambiguity proxy).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2

QUALITY_NONE = 0
QUALITY_WEAK = 1
QUALITY_STRONG = 2


@dataclass
class MCWeightConfig:
    """Tunable knobs for geometric prior × wall × horizon scaling."""

    drift_window: int = 20
    drift_clip_frac: float = 0.3  # clip |μ| to this fraction of σ per row
    use_drift_in_gambler_ruin: bool = False

    horizon_penalty_power: float = 0.5

    wall_strength_scale: float = 3.0
    wall_weight_min: float = 0.40
    wall_weight_max: float = 1.20

    quality_strong_mult: float = 1.40
    quality_weak_mult: float = 1.00
    quality_none_mult: float = 0.60

    weight_min: float = 0.10
    weight_max: float = 3.50
    normalize: bool = True


def _gambler_ruin_p_tp(
    tp_dist: np.ndarray,
    sl_dist: np.ndarray,
    micro_sigma: np.ndarray,
    drift: Optional[np.ndarray] = None,
    *,
    drift_clip_frac: float = 0.3,
) -> np.ndarray:
    """P(hit TP before SL) with barriers ±tp / ∓sl reduced to symmetric 1D formula."""
    a = np.maximum(tp_dist.astype(np.float64), 1e-12)
    b = np.maximum(sl_dist.astype(np.float64), 1e-12)
    s = np.maximum(micro_sigma.astype(np.float64), 1e-12)

    if drift is None or not np.any(np.abs(drift) > 1e-12):
        return (b / (a + b)).astype(np.float32)

    # μ, σ must share units (price delta per bar).
    mu = np.clip(np.asarray(drift, dtype=np.float64), -s * drift_clip_frac, s * drift_clip_frac)
    exp_b = np.exp(np.clip(-2.0 * mu * b / (s * s), -500.0, 500.0))
    exp_tot = np.exp(np.clip(-2.0 * mu * (a + b) / (s * s), -500.0, 500.0))
    den = 1.0 - exp_tot
    degenerate = np.abs(den) < 1e-9
    numer = 1.0 - exp_b
    p_full = np.where(degenerate, b / (a + b), numer / (den + 1e-15))
    return np.clip(p_full.astype(np.float32), 0.005, 0.995)


def _estimate_local_drift_px(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """Trailing mean bar-to-bar price change using only past observations (causal)."""
    px = pd.Series(np.asarray(prices, dtype=np.float64), copy=False)
    dpx = px.diff().fillna(0.0)
    return dpx.shift(1).rolling(window=int(window), min_periods=1).mean().fillna(0.0).to_numpy(dtype=np.float32)


def _wall_confidence(df: pd.DataFrame, cfg: MCWeightConfig) -> np.ndarray:
    """
    Confidence from aggregated wall strength. Fixed logic:
      - Prefer max(bid_wall_strength, ask_wall_strength) when both exist
      - Else single-sided wall column
      - Else generic wall_strength
    """
    n = len(df)
    mid = np.float32(0.5 * (cfg.wall_weight_min + cfg.wall_weight_max))

    if "bid_wall_strength" in df.columns and "ask_wall_strength" in df.columns:
        w_bid = pd.to_numeric(df["bid_wall_strength"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        w_ask = pd.to_numeric(df["ask_wall_strength"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        wall_str = np.maximum(w_bid, w_ask).astype(np.float32)
    elif "bid_wall_strength" in df.columns:
        wall_str = pd.to_numeric(df["bid_wall_strength"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    elif "ask_wall_strength" in df.columns:
        wall_str = pd.to_numeric(df["ask_wall_strength"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    elif "wall_strength" in df.columns:
        wall_str = pd.to_numeric(df["wall_strength"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        return np.full(n, mid, dtype=np.float32)

    x = wall_str / max(np.float32(cfg.wall_strength_scale), np.float32(1e-6))
    sig = 1.0 / (1.0 + np.exp(-x * np.float32(3.0)))
    return (
        np.float32(cfg.wall_weight_min)
        + (np.float32(cfg.wall_weight_max) - np.float32(cfg.wall_weight_min)) * sig.astype(np.float32)
    ).astype(np.float32)


def compute_label_stability(
    labeled_df: pd.DataFrame,
    *,
    bias_col: str = "bias_label",
    shift_steps: Iterable[int] = (-5, -3, -1, 1, 3, 5),
) -> np.ndarray:
    """
    Fraction of neighbouring rows (index shifts ±k inside bounds) that share the same
    bias_label. Not a full re-scan of the labelling window; cheap ambiguity proxy.
    """
    shifts = tuple(int(x) for x in shift_steps)
    n = len(labeled_df)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    bias = pd.to_numeric(labeled_df[bias_col], errors="coerce").fillna(DIR_NEUTRAL).astype(np.int8).to_numpy(copy=False)

    match_cnt = np.zeros(n, dtype=np.float32)
    valid_cnt = np.zeros(n, dtype=np.float32)
    ix = np.arange(n, dtype=np.int32)

    for s in shifts:
        j = ix + np.int32(s)
        mask = (j >= 0) & (j < n)
        if not np.any(mask):
            continue
        same = bias[j[mask]] == bias[ix[mask]]
        match_cnt[mask] += same.astype(np.float32)
        valid_cnt += mask.astype(np.float32)

    return np.where(valid_cnt > 0.0, match_cnt / np.maximum(valid_cnt, 1.0), 1.0).astype(np.float32)


def compute_mc_soft_weights(
    labeled_df: pd.DataFrame,
    *,
    tp_mult: float,
    sl_mult: float,
    config: Optional[MCWeightConfig] = None,
) -> np.ndarray:
    """
    Structural training weights ∝ gambler ruin P × horizon factor × wall confidence × quality.

    Required / used columns when present:
      label_dynamic_threshold, micro_atr, effective_horizon, bias_label,
      signal_quality, close OR mid_price, bid/ask wall strength.
    """
    cfg = config or MCWeightConfig()
    n = len(labeled_df)
    if n == 0:
        return np.array([], dtype=np.float32)

    bias = (
        pd.to_numeric(labeled_df.get("bias_label", DIR_NEUTRAL), errors="coerce")
        .fillna(DIR_NEUTRAL)
        .astype(np.int8)
        .to_numpy()
    )
    quality = (
        pd.to_numeric(labeled_df.get("signal_quality", QUALITY_WEAK), errors="coerce")
        .fillna(QUALITY_WEAK)
        .astype(np.int8)
        .to_numpy()
    )

    if "label_dynamic_threshold" in labeled_df.columns:
        dyn_thr = pd.to_numeric(labeled_df["label_dynamic_threshold"], errors="coerce").fillna(np.nan).to_numpy(np.float32)
        med = float(np.nanmedian(dyn_thr)) if np.any(np.isfinite(dyn_thr)) else 1e-4
        dyn_thr = np.where(np.isfinite(dyn_thr), dyn_thr, med).astype(np.float32)
    else:
        dyn_thr = np.full(n, 1e-4, dtype=np.float32)

    tp_m = tp_mult if np.isfinite(tp_mult) and tp_mult > 0 else 1.5
    sl_m = sl_mult if np.isfinite(sl_mult) and sl_mult > 0 else 1.0

    tp_dist = dyn_thr * np.float32(tp_m)
    sl_dist = dyn_thr * np.float32(sl_m)

    if "micro_atr" in labeled_df.columns:
        sig = pd.to_numeric(labeled_df["micro_atr"], errors="coerce").fillna(np.nan).to_numpy(np.float32)
        smed = float(np.nanmedian(sig)) if np.any(np.isfinite(sig)) else float(np.median(dyn_thr))
        sig = np.where(np.isfinite(sig), sig, smed)
        sigma = np.maximum(sig, np.float32(1e-9))
    else:
        sigma = np.maximum(dyn_thr, np.float32(1e-9))

    price_col = "close" if "close" in labeled_df.columns else ("mid_price" if "mid_price" in labeled_df.columns else None)
    drift: Optional[np.ndarray] = None
    if cfg.use_drift_in_gambler_ruin and price_col is not None:
        prices = pd.to_numeric(labeled_df[price_col], errors="coerce").to_numpy(dtype=np.float64)
        drift = _estimate_local_drift_px(prices, window=int(cfg.drift_window))

    p_gam = _gambler_ruin_p_tp(
        tp_dist,
        sl_dist,
        sigma,
        drift,
        drift_clip_frac=float(cfg.drift_clip_frac),
    )
    p_gam[np.asarray(bias) == DIR_NEUTRAL] = 0.50

    if "effective_horizon" in labeled_df.columns:
        hor = (
            pd.to_numeric(labeled_df["effective_horizon"], errors="coerce")
            .fillna(20.0)
            .to_numpy(np.float32)
        )
        hor = np.maximum(hor, 1.0)
        med_hor = float(np.median(hor))
        norm_hor = hor / max(med_hor, 1.0)
        horizon_penalty = np.power((1.0 / np.maximum(norm_hor, np.float32(0.1))), np.float32(cfg.horizon_penalty_power))
        horizon_penalty = np.clip(horizon_penalty, 0.40, 1.60).astype(np.float32)
    else:
        horizon_penalty = np.ones(n, dtype=np.float32)

    wall_conf = _wall_confidence(labeled_df, cfg)

    quality_mult = np.where(
        quality == QUALITY_STRONG,
        np.float32(cfg.quality_strong_mult),
        np.where(
            quality == QUALITY_WEAK,
            np.float32(cfg.quality_weak_mult),
            np.float32(cfg.quality_none_mult),
        ),
    ).astype(np.float32)

    raw_weight = (p_gam * horizon_penalty * wall_conf * quality_mult).astype(np.float32)
    raw_weight = np.clip(raw_weight, np.float32(cfg.weight_min), np.float32(cfg.weight_max))

    if cfg.normalize:
        mean_w = float(np.mean(raw_weight))
        if mean_w > 1e-8:
            raw_weight = raw_weight / np.float32(mean_w)
        raw_weight = np.clip(raw_weight, np.float32(cfg.weight_min), np.float32(cfg.weight_max))

    return raw_weight.astype(np.float32)


def validate_and_report(labeled_df: pd.DataFrame, weights: np.ndarray) -> bool:
    """Smoke checks without hard assertions — logs PASS/FAIL lines."""
    if len(weights) == 0:
        return True

    quality = pd.to_numeric(labeled_df.get("signal_quality", 1), errors="coerce").fillna(1).to_numpy()
    strong = quality == QUALITY_STRONG
    weak = quality == QUALITY_WEAK

    w_mean = float(np.mean(weights))
    w_std = float(np.std(weights))
    low_w = float((weights < 0.30).mean())
    w_strong = float(weights[strong].mean()) if np.any(strong) else float("nan")
    w_weak = float(weights[weak].mean()) if np.any(weak) else float("nan")

    print("=" * 65)
    print("MC Soft Weights — diagnostic report")
    print("=" * 65)
    print(f" rows         : {len(weights):,}")
    print(f" mean weight   : {w_mean:.4f}  (target ~1.0)")
    print(f" std weight    : {w_std:.4f}")
    print(f" pct w<0.3     : {low_w:.1%}")
    if np.isfinite(w_strong) and np.isfinite(w_weak):
        print(f" mean STRONG   : {w_strong:.4f}")
        print(f" mean WEAK     : {w_weak:.4f}")

    ok = True
    if not (0.80 < w_mean < 1.20):
        print(f" WARN: mean weight out of band: {w_mean:.4f}")
        ok = False
    if low_w < 0.05:
        print(f" WARN: pct w<0.3 very low ({low_w:.1%}); downweight might be negligible")
        ok = False
    if np.isfinite(w_strong) and np.isfinite(w_weak) and (w_weak > 1e-8) and w_strong < w_weak * 1.08:
        print(" WARN: STRONG vs WEAK separation weak")

    print("=" * 65)
    return ok


def attach_mc_prior_columns(
    labeled_df: pd.DataFrame,
    *,
    tp_mult: float,
    sl_mult: float,
    stability_shifts: Iterable[int] = (-5, -3, -1, 1, 3, 5),
    config: Optional[MCWeightConfig] = None,
    mc_column: str = "mc_sample_weight",
    stability_column: str = "label_stability",
    quiet: bool = False,
) -> pd.DataFrame:
    """Adds label_stability and mc_sample_weight; optional printed validation."""
    df = labeled_df.copy()
    if len(df) == 0:
        df[stability_column] = np.zeros(0, dtype=np.float32)
        df[mc_column] = np.zeros(0, dtype=np.float32)
        return df

    stab = compute_label_stability(df, shift_steps=stability_shifts)
    weights = compute_mc_soft_weights(df, tp_mult=tp_mult, sl_mult=sl_mult, config=config)

    df[stability_column] = stab
    df[mc_column] = weights

    if not quiet:
        valid_ok = validate_and_report(df, weights)
        if not valid_ok:
            warnings.warn(
                "[mc_label_weights] validation warnings — inspect columns "
                "(label_dynamic_threshold, micro_atr, effective_horizon, bias_label)",
                RuntimeWarning,
                stacklevel=2,
            )

    return df
