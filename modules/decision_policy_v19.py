"""
decision_policy_v19.py - Cost-aware decision policy helpers for QuantSystem V19
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

POLICY_VERSION = "v19-cost-aware-v1"
DEFAULT_DECISION_POLICY_ARTIFACT = "decision_policy_v19.json"
MIN_POLICY_SUPPORT = 24
MIN_POLICY_COVERAGE_RATIO = 0.50


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def normalize_regime_probs(regime_probs: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray:
    arr = np.asarray(regime_probs if regime_probs is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1)
    if arr.size == 0:
        arr = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        arr = np.ones(max(len(arr), 1), dtype=np.float64)
        total = float(arr.sum())
    return (arr / total).astype(np.float64, copy=False)


def normalized_entropy(probs: np.ndarray | list[float] | tuple[float, ...] | None) -> float:
    arr = normalize_regime_probs(probs)
    if arr.size <= 1:
        return 0.0
    entropy = -float(np.sum(arr * np.log(np.clip(arr, 1e-12, 1.0))))
    return float(entropy / math.log(float(arr.size)))


def effective_round_trip_cost_pips(cost_config: dict | None = None) -> dict:
    cfg = dict(cost_config or {})
    round_trip_cost_pips = max(_safe_float(cfg.get("round_trip_cost_pips", 1.0), 1.0), 0.0)
    commission_per_side = max(_safe_float(cfg.get("commission_per_side", 0.0), 0.0), 0.0)
    tick_value = max(_safe_float(cfg.get("tick_value", 10.0), 10.0), 1e-8)
    min_spread_ticks = max(_safe_float(cfg.get("min_spread_ticks", 1.0), 1.0), 0.0)
    min_slippage_ticks = max(_safe_float(cfg.get("min_slippage_ticks", 1.0), 1.0), 0.0)
    spread_multiplier = max(_safe_float(cfg.get("spread_multiplier", 0.5), 0.5), 0.0)

    commission_one_way_pips = commission_per_side / tick_value
    effective_cost_pips = max(
        round_trip_cost_pips,
        2.0 * (min_slippage_ticks + spread_multiplier * min_spread_ticks + commission_one_way_pips),
    )
    return {
        "round_trip_cost_pips": float(round_trip_cost_pips),
        "commission_per_side": float(commission_per_side),
        "tick_value": float(tick_value),
        "min_spread_ticks": float(min_spread_ticks),
        "min_slippage_ticks": float(min_slippage_ticks),
        "spread_multiplier": float(spread_multiplier),
        "commission_one_way_pips": float(commission_one_way_pips),
        "effective_cost_pips": float(effective_cost_pips),
    }


def structure_bucket_from_row(row: dict | pd.Series) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    trend_strength = _safe_float(getter("trend_strength", 0.0))
    correction_depth = _safe_float(getter("correction_depth", 0.0))
    liquidity_sweep = _safe_float(getter("liquidity_sweep", 0.0))
    kalman_strength = _safe_float(getter("kalman_trend_strength", 0.0))
    event_score = _safe_float(getter("event_score", 0.0))
    price_position = _safe_float(getter("price_position", 0.5))

    if abs(liquidity_sweep) >= 0.35:
        return "liquidity_sweep"
    if trend_strength >= 0.55 and correction_depth <= 0.20 and kalman_strength >= 0.20:
        return "impulse_extension"
    if trend_strength >= 0.35 and correction_depth >= 0.60:
        return "deep_pullback"
    if trend_strength >= 0.30 and correction_depth >= 0.25:
        return "trend_pullback"
    if kalman_strength >= 0.20 and event_score >= 1.0:
        return "trend_confirmation"
    if 0.20 <= price_position <= 0.80:
        return "mid_range_probe"
    return "transition"


def structure_bucket_series(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype="object")
    return df.apply(structure_bucket_from_row, axis=1).astype(str)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size != weights.size:
        raise ValueError(f"weighted mean size mismatch: {values.size} vs {weights.size}")
    weight_sum = float(np.sum(np.clip(weights, 0.0, None)))
    if weight_sum <= 1e-12:
        return float(np.mean(values))
    return float(np.sum(values * np.clip(weights, 0.0, None)) / weight_sum)


def _side_stats(frame: pd.DataFrame, side_label: int, cost_pips: float, side_probs: np.ndarray | None = None) -> dict:
    if frame.empty:
        baseline = max(float(cost_pips), 1.0)
        threshold = (baseline + float(cost_pips)) / max(2.0 * baseline + float(cost_pips), 1e-8)
        return {
            "support": 0,
            "win_support": 0,
            "loss_support": 0,
            "coverage_ratio": 0.0,
            "avg_win_pips": float(baseline),
            "avg_loss_pips": float(baseline),
            "threshold_from_cost": float(threshold),
            "realized_prior": 0.5,
            "avg_abs_return_pips": float(baseline),
        }

    if "_abs_forward_pips" in frame.columns:
        forward_pips = pd.to_numeric(frame["_abs_forward_pips"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        forward_pips = np.abs(pd.to_numeric(frame.get("forward_return", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
        tick_size = max(_safe_float(frame.attrs.get("tick_size", 1.0), 1.0), 1e-8)
        forward_pips = forward_pips / tick_size
    labels = pd.to_numeric(frame.get("bias_label", 1), errors="coerce").fillna(1).astype(np.int32).to_numpy()
    covered = pd.to_numeric(frame.get("policy_covered", 0), errors="coerce").fillna(0).astype(np.int8).to_numpy()
    weights = (
        np.clip(np.asarray(side_probs, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if side_probs is not None
        else np.ones(len(frame), dtype=np.float64)
    )
    if len(weights) != len(frame):
        raise ValueError(f"side_probs length mismatch: {len(weights)} vs {len(frame)}")

    win_mask = labels == int(side_label)
    loss_mask = labels == int(1 - side_label)
    avg_win = _weighted_mean(forward_pips[win_mask], weights[win_mask]) if np.any(win_mask) else 0.0
    avg_loss = _weighted_mean(forward_pips[loss_mask], weights[loss_mask]) if np.any(loss_mask) else 0.0
    baseline = max(float(cost_pips), 1.0)
    avg_win = max(avg_win, baseline)
    avg_loss = max(avg_loss, baseline)
    threshold = (avg_loss + float(cost_pips)) / max(avg_win + avg_loss + float(cost_pips), 1e-8)
    weighted_support = float(np.sum(weights))
    return {
        "support": int(len(frame)),
        "weighted_support": float(weighted_support),
        "win_support": int(np.sum(win_mask)),
        "loss_support": int(np.sum(loss_mask)),
        "coverage_ratio": float(np.mean(covered.astype(np.float32))) if len(covered) else 0.0,
        "avg_win_pips": float(avg_win),
        "avg_loss_pips": float(avg_loss),
        "threshold_from_cost": float(np.clip(threshold, 0.0, 1.0)),
        "realized_prior": _weighted_mean(win_mask.astype(np.float32), weights) if len(win_mask) else 0.5,
        "avg_abs_return_pips": _weighted_mean(forward_pips, weights) if len(forward_pips) else float(baseline),
    }


def build_decision_policy(
    df: pd.DataFrame,
    bias_probs: np.ndarray,
    regime_meta: np.ndarray,
    coverage_mask: np.ndarray,
    *,
    cost_config: dict | None = None,
    source: str = "stage1_oof",
    min_support: int = MIN_POLICY_SUPPORT,
    min_coverage_ratio: float = MIN_POLICY_COVERAGE_RATIO,
) -> dict:
    frame = df.copy().reset_index(drop=True)
    probs = np.asarray(bias_probs, dtype=np.float64)
    regime_arr = np.asarray(regime_meta, dtype=np.float64)
    coverage = np.asarray(coverage_mask, dtype=bool).reshape(-1)

    if len(frame) == 0:
        return {
            "policy_version": POLICY_VERSION,
            "source": str(source),
            "coverage_ratio": 0.0,
            "minimum_support": int(min_support),
            "minimum_coverage_ratio": float(min_coverage_ratio),
            "cost_model": effective_round_trip_cost_pips(cost_config),
            "global": {},
            "structures": {},
            "regime_priors": [1.0, 0.0, 0.0, 0.0],
            "structure_buckets": [],
        }

    if probs.shape != (len(frame), 2):
        raise ValueError(f"bias_probs must be shaped {(len(frame), 2)}, got {probs.shape}")
    if regime_arr.ndim != 2 or regime_arr.shape[0] != len(frame) or regime_arr.shape[1] < 4:
        raise ValueError(
            "regime_meta must be 2D with at least 4 columns matching the frame rows; "
            f"got {regime_arr.shape}"
        )
    if len(coverage) != len(frame):
        raise ValueError(f"coverage_mask length mismatch: {len(coverage)} vs {len(frame)}")

    cost_model = effective_round_trip_cost_pips(cost_config)
    frame["policy_covered"] = coverage.astype(np.int8)
    frame["structure_bucket"] = structure_bucket_series(frame)
    frame["dominant_regime"] = np.argmax(regime_arr[:, :4], axis=1).astype(np.int32)
    frame["prob_long"] = np.clip(probs[:, 0], 0.0, 1.0)
    frame["prob_short"] = np.clip(probs[:, 1], 0.0, 1.0)
    tick_size = max(_safe_float((cost_config or {}).get("tick_size", 1.0), 1.0), 1e-8)
    frame["_abs_forward_pips"] = (
        np.abs(pd.to_numeric(frame.get("forward_return", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
        / tick_size
    )

    global_block = {
        "LONG": _side_stats(frame, 0, cost_model["effective_cost_pips"], frame["prob_long"].to_numpy(dtype=np.float64)),
        "SHORT": _side_stats(frame, 1, cost_model["effective_cost_pips"], frame["prob_short"].to_numpy(dtype=np.float64)),
    }

    structures: dict[str, dict[str, Any]] = {}
    for bucket, bucket_df in frame.groupby("structure_bucket", sort=True):
        bucket_payload: dict[str, Any] = {
            "support": int(len(bucket_df)),
            "coverage_ratio": float(bucket_df["policy_covered"].mean()) if len(bucket_df) else 0.0,
            "global": {
                "LONG": _side_stats(bucket_df, 0, cost_model["effective_cost_pips"], bucket_df["prob_long"].to_numpy(dtype=np.float64)),
                "SHORT": _side_stats(bucket_df, 1, cost_model["effective_cost_pips"], bucket_df["prob_short"].to_numpy(dtype=np.float64)),
            },
            "by_regime": {},
        }
        for regime_id in range(4):
            regime_df = bucket_df[bucket_df["dominant_regime"] == regime_id]
            bucket_payload["by_regime"][str(regime_id)] = {
                "support": int(len(regime_df)),
                "LONG": _side_stats(regime_df, 0, cost_model["effective_cost_pips"], regime_df["prob_long"].to_numpy(dtype=np.float64)),
                "SHORT": _side_stats(regime_df, 1, cost_model["effective_cost_pips"], regime_df["prob_short"].to_numpy(dtype=np.float64)),
            }
        structures[str(bucket)] = bucket_payload

    regime_priors = normalize_regime_probs(np.mean(regime_arr[:, :4], axis=0)).tolist()
    out_pol = {
        "policy_version": POLICY_VERSION,
        "source": str(source),
        "coverage_ratio": float(np.mean(coverage.astype(np.float32))) if len(coverage) else 0.0,
        "minimum_support": int(min_support),
        "minimum_coverage_ratio": float(min_coverage_ratio),
        "cost_model": cost_model,
        "global": global_block,
        "structures": structures,
        "regime_priors": [float(v) for v in regime_priors],
        "structure_buckets": sorted(structures),
    }
    return out_pol


def _coerce_side_block(block: dict | None, fallback: dict) -> dict:
    if not isinstance(block, dict):
        return dict(fallback)
    out = dict(fallback)
    out.update(block)
    return out


def _resolve_side_policy(
    policy: dict,
    side: str,
    structure_bucket: str,
    regime_probs: np.ndarray | list[float] | tuple[float, ...] | None,
) -> dict:
    side = str(side).upper()
    global_side = _coerce_side_block((policy.get("global") or {}).get(side), {})
    bucket_block = (policy.get("structures") or {}).get(str(structure_bucket), {})
    bucket_side = _coerce_side_block((bucket_block.get("global") or {}).get(side), global_side)
    min_support = _safe_int(policy.get("minimum_support", MIN_POLICY_SUPPORT), MIN_POLICY_SUPPORT)
    probs = normalize_regime_probs(regime_probs)

    weighted_values = {
        "avg_win_pips": 0.0,
        "avg_loss_pips": 0.0,
        "threshold_from_cost": 0.0,
        "realized_prior": 0.0,
        "coverage_ratio": 0.0,
    }
    total_weight = 0.0
    for regime_id, weight in enumerate(probs[:4]):
        if weight <= 0:
            continue
        regime_payload = ((bucket_block.get("by_regime") or {}).get(str(regime_id)) or {})
        regime_side = _coerce_side_block(regime_payload.get(side), bucket_side)
        if _safe_int(regime_side.get("support", 0), 0) < min_support:
            regime_side = bucket_side if _safe_int(bucket_side.get("support", 0), 0) >= min_support else global_side
        total_weight += float(weight)
        for key in weighted_values:
            weighted_values[key] += float(weight) * _safe_float(regime_side.get(key, bucket_side.get(key, global_side.get(key, 0.0))), 0.0)

    if total_weight <= 0:
        resolved = bucket_side if _safe_int(bucket_side.get("support", 0), 0) >= min_support else global_side
    else:
        resolved = dict(bucket_side if _safe_int(bucket_side.get("support", 0), 0) >= min_support else global_side)
        for key, value in weighted_values.items():
            resolved[key] = float(value / total_weight)
    resolved["support"] = int(max(_safe_int(bucket_side.get("support", 0), 0), _safe_int(global_side.get("support", 0), 0)))
    return resolved


def evaluate_decision_policy(
    policy: dict | None,
    *,
    direction_probs: dict[str, float] | None,
    regime_probs: np.ndarray | list[float] | tuple[float, ...] | None,
    structure_bucket: str,
    uncertainty: float = 0.0,
    runtime_penalty: float = 1.0,
    coverage_ratio: float | None = None,
    require_positive_ev: bool = True,
    edge_prob_override: float | None = None,
) -> dict | None:
    if not isinstance(policy, dict) or not policy:
        return None

    p_long = _clip01((direction_probs or {}).get("LONG", 0.0))
    p_short = _clip01((direction_probs or {}).get("SHORT", 0.0))
    regime_arr_raw = normalize_regime_probs(regime_probs if regime_probs is not None else policy.get("regime_priors"))
    regime_prior = normalize_regime_probs(policy.get("regime_priors"))
    regime_entropy = normalized_entropy(regime_arr_raw)
    entropy_blend = float(np.clip((regime_entropy - 0.70) / 0.25, 0.0, 1.0))
    regime_arr = normalize_regime_probs(
        (1.0 - entropy_blend) * regime_arr_raw + entropy_blend * regime_prior
    )
    long_side = _resolve_side_policy(policy, "LONG", structure_bucket, regime_arr)
    short_side = _resolve_side_policy(policy, "SHORT", structure_bucket, regime_arr)
    cost_pips = _safe_float(((policy.get("cost_model") or {}).get("effective_cost_pips")), 0.0)
    ev_long = p_long * _safe_float(long_side.get("avg_win_pips", 0.0), 0.0) - (1.0 - p_long) * _safe_float(long_side.get("avg_loss_pips", 0.0), 0.0) - cost_pips
    ev_short = p_short * _safe_float(short_side.get("avg_win_pips", 0.0), 0.0) - (1.0 - p_short) * _safe_float(short_side.get("avg_loss_pips", 0.0), 0.0) - cost_pips
    effective_coverage = float(policy.get("coverage_ratio", 0.0)) if coverage_ratio is None else float(coverage_ratio)
    coverage_ok = effective_coverage >= _safe_float(policy.get("minimum_coverage_ratio", MIN_POLICY_COVERAGE_RATIO), MIN_POLICY_COVERAGE_RATIO)
    runtime_penalty = _clip01(runtime_penalty)
    uncertainty = _clip01(uncertainty)

    long_thr = float(_safe_float(long_side.get("threshold_from_cost", 0.5), 0.5))
    short_thr = float(_safe_float(short_side.get("threshold_from_cost", 0.5), 0.5))
    if edge_prob_override is not None:
        o = _clip01(float(edge_prob_override))
        long_thr = o
        short_thr = o
    decision = {
        "policy_available": True,
        "policy_version": str(policy.get("policy_version", POLICY_VERSION)),
        "structure_bucket": str(structure_bucket),
        "policy_coverage_ratio": float(effective_coverage),
        "coverage_ok": bool(coverage_ok),
        "runtime_ok": bool(runtime_penalty >= 0.5),
        "regime_entropy": float(regime_entropy),
        "regime_entropy_blend": float(entropy_blend),
        "cost_pips": float(cost_pips),
        "expected_value_long_pips": float(ev_long),
        "expected_value_short_pips": float(ev_short),
        "long_threshold": float(long_thr),
        "short_threshold": float(short_thr),
        "long_policy": long_side,
        "short_policy": short_side,
        "require_positive_ev": bool(require_positive_ev),
        "edge_prob_override": None if edge_prob_override is None else float(_clip01(float(edge_prob_override))),
    }

    passes_long = (
        coverage_ok
        and runtime_penalty >= 0.5
        and p_long >= long_thr
        and (ev_long > 0.0 if require_positive_ev else True)
    )
    passes_short = (
        coverage_ok
        and runtime_penalty >= 0.5
        and p_short >= short_thr
        and (ev_short > 0.0 if require_positive_ev else True)
    )
    long_threshold_ok = bool(p_long >= long_thr)
    short_threshold_ok = bool(p_short >= short_thr)
    long_ev_positive = bool(ev_long > 0.0)
    short_ev_positive = bool(ev_short > 0.0)
    decision["long_threshold_ok"] = long_threshold_ok
    decision["short_threshold_ok"] = short_threshold_ok
    decision["long_ev_positive"] = long_ev_positive
    decision["short_ev_positive"] = short_ev_positive

    if passes_long and (not passes_short or ev_long >= ev_short):
        decision.update(
            {
                "bias": "LONG",
                "bias_idx": 0,
                "tradeable": True,
                "expected_value_pips": float(ev_long),
                "edge_prob": float(p_long),
                "chosen_threshold": float(decision["long_threshold"]),
                "selected_avg_win_pips": float(_safe_float(long_side.get("avg_win_pips", 0.0), 0.0)),
                "selected_avg_loss_pips": float(_safe_float(long_side.get("avg_loss_pips", 0.0), 0.0)),
                "reason": "cost_aware_ev_long",
            }
        )
    elif passes_short:
        decision.update(
            {
                "bias": "SHORT",
                "bias_idx": 1,
                "tradeable": True,
                "expected_value_pips": float(ev_short),
                "edge_prob": float(p_short),
                "chosen_threshold": float(decision["short_threshold"]),
                "selected_avg_win_pips": float(_safe_float(short_side.get("avg_win_pips", 0.0), 0.0)),
                "selected_avg_loss_pips": float(_safe_float(short_side.get("avg_loss_pips", 0.0), 0.0)),
                "reason": "cost_aware_ev_short",
            }
        )
    else:
        if not coverage_ok:
            reject_reason = "policy_coverage_too_low"
            reject_bucket = "coverage"
        elif runtime_penalty < 0.5:
            reject_reason = "runtime_penalty_too_low"
            reject_bucket = "runtime"
        elif not long_threshold_ok and not short_threshold_ok:
            reject_reason = "probability_below_threshold"
            reject_bucket = "threshold"
        elif not long_ev_positive and not short_ev_positive:
            reject_reason = "expected_value_non_positive"
            reject_bucket = "ev"
        else:
            reject_reason = "mixed_policy_constraints"
            reject_bucket = "mixed"
        decision.update(
            {
                "bias": "NEUTRAL",
                "bias_idx": 2,
                "tradeable": False,
                "expected_value_pips": float(max(ev_long, ev_short)),
                "edge_prob": float(max(p_long, p_short)),
                "chosen_threshold": float(max(decision["long_threshold"], decision["short_threshold"])),
                "selected_avg_win_pips": float(max(_safe_float(long_side.get("avg_win_pips", 0.0), 0.0), _safe_float(short_side.get("avg_win_pips", 0.0), 0.0))),
                "selected_avg_loss_pips": float(max(_safe_float(long_side.get("avg_loss_pips", 0.0), 0.0), _safe_float(short_side.get("avg_loss_pips", 0.0), 0.0))),
                "reason": str(reject_reason),
                "reject_bucket": str(reject_bucket),
            }
        )

    decision["sizing_penalty"] = float(
        np.clip((1.0 - uncertainty) * (1.0 - regime_entropy) * max(runtime_penalty, 0.25), 0.0, 1.0)
    )
    return decision
