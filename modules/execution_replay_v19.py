"""
execution_replay_v19.py - Shared execution replay helpers for V19
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modules.slippage_model import SlippageModel


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def realized_fill_pricing(
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
        best_bid = safe_float(row_like.get("bid_px_00", 0.0))
        best_ask = safe_float(row_like.get("ask_px_00", 0.0))
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
    exit_direction = "short" if str(direction).upper() == "LONG" else "long"
    exit_fill = fill_model.compute_fill(exit_row or {}, size=max(int(size), 1), direction=exit_direction)

    entry_price = safe_float(entry_fill.get("fill_price", 0.0))
    exit_price = safe_float(exit_fill.get("fill_price", 0.0))
    if entry_fill.get("filled", 0) <= 0 or exit_fill.get("filled", 0) <= 0 or entry_price <= 0 or exit_price <= 0:
        fallback_cost = max(
            floor_cost,
            2.0 * (
                max(float(min_slippage_ticks), float(spread_multiplier) * _spread_ticks(entry_row))
                + commission_one_way_pips
            ),
        )
        return {
            "used_dynamic_fill": False,
            "entry_fill": entry_fill,
            "exit_fill": exit_fill,
            "fill_pnl_pips": raw_pnl_pips,
            "dynamic_cost_pips": float(fallback_cost),
            "net_pnl_pips": float(raw_pnl_pips - fallback_cost),
        }

    if str(direction).upper() == "LONG":
        fill_pnl_pips = float((exit_price - entry_price) / tick)
    else:
        fill_pnl_pips = float((entry_price - exit_price) / tick)

    entry_spread_ticks = _spread_ticks(entry_row)
    exit_spread_ticks = _spread_ticks(exit_row)
    entry_slip_floor = max(float(min_slippage_ticks), float(spread_multiplier) * entry_spread_ticks)
    exit_slip_floor = max(float(min_slippage_ticks), float(spread_multiplier) * exit_spread_ticks)
    realized_entry_cost = max(float(entry_fill.get("slippage_pips", 0.0) or 0.0), entry_slip_floor) + commission_one_way_pips
    realized_exit_cost = max(float(exit_fill.get("slippage_pips", 0.0) or 0.0), exit_slip_floor) + commission_one_way_pips
    dynamic_cost_pips = max(float(raw_pnl_pips - fill_pnl_pips), 0.0)
    total_cost_pips = max(dynamic_cost_pips, realized_entry_cost + realized_exit_cost, floor_cost)
    return {
        "used_dynamic_fill": True,
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "fill_pnl_pips": float(fill_pnl_pips),
        "dynamic_cost_pips": float(total_cost_pips),
        "net_pnl_pips": float(raw_pnl_pips - total_cost_pips),
    }


def simulate_trade_path(
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
    row_data: dict | None = None,
) -> dict | None:
    if direction not in ("LONG", "SHORT"):
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
    min_move = max(float(direction_threshold_ticks) * float(tick_size), 0.5 * max(atr_now, 0.0), float(tick_size))

    tp_level = sl_level = None
    if row_data is not None:
        try:
            from modules.dynamic_target import DynamicTargetManager

            dtm = DynamicTargetManager()
            scan_result = {
                "bid_wall_strength": float(row_data.get("bid_wall_strength", 0.5) or 0.5),
                "ask_wall_strength": float(row_data.get("ask_wall_strength", 0.5) or 0.5),
                "dist_to_bid_wall": float(row_data.get("dist_to_bid_wall", min_move * 1.5) or min_move * 1.5),
                "dist_to_ask_wall": float(row_data.get("dist_to_ask_wall", min_move * 1.5) or min_move * 1.5),
                "bid_wall_size_raw": float(row_data.get("gap_size", 1.0) or 1.0),
                "ask_wall_size_raw": float(row_data.get("gap_size", 1.0) or 1.0),
            }
            remaining_fuel = max(
                float(row_data.get("micro_atr", min_move * 10) or min_move * 10) * 80,
                min_move * 20,
            )
            context = {
                "tick_size": tick_size,
                "remaining_fuel": remaining_fuel,
                "adr_pips": remaining_fuel / tick_size,
            }
            signal = {
                "bias": direction,
                "price": entry_price,
                "cvd_delta": float(row_data.get("cvd", 0.0) or 0.0),
            }
            levels = {
                "long_wall_size": scan_result["bid_wall_size_raw"],
                "short_wall_size": scan_result["ask_wall_size_raw"],
            }
            trade = dtm.open_trade(signal, levels, scan_result, context)
            tp_level = trade.tp1
            sl_level = trade.sl
        except Exception:
            tp_level = None

    if tp_level is None or sl_level is None:
        tp_distance = max(float(tp_mult) * min_move, float(tick_size))
        sl_distance = max(float(sl_mult) * min_move, float(tick_size))
        if direction == "LONG":
            tp_level = entry_price + tp_distance
            sl_level = entry_price - sl_distance
        else:
            tp_level = entry_price - tp_distance
            sl_level = entry_price + sl_distance

    future_prices = np.asarray(prices[entry_idx + 1:exit_cap_idx + 1], dtype=np.float64)
    if future_prices.size == 0:
        return None

    exit_idx = exit_cap_idx
    exit_reason = "horizon"
    for offset, future_price in enumerate(future_prices, start=1):
        if direction == "LONG":
            if future_price >= tp_level:
                exit_idx = entry_idx + offset
                exit_reason = "tp"
                break
            if future_price <= sl_level:
                exit_idx = entry_idx + offset
                exit_reason = "sl"
                break
        else:
            if future_price <= tp_level:
                exit_idx = entry_idx + offset
                exit_reason = "tp"
                break
            if future_price >= sl_level:
                exit_idx = entry_idx + offset
                exit_reason = "sl"
                break

    exit_price = float(prices[exit_idx])
    price_return = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
    path_moves = (future_prices - entry_price) if direction == "LONG" else (entry_price - future_prices)
    favourable_move = float(np.max(path_moves)) if path_moves.size else 0.0
    adverse_move = float(np.min(path_moves)) if path_moves.size else 0.0

    return {
        "exit_idx": int(exit_idx),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "hold_steps": int(exit_idx - entry_idx),
        "price_return": float(price_return),
        "raw_pnl_pips": float(price_return / max(float(tick_size), 1e-8)),
        "mfe_pips": float(favourable_move / max(float(tick_size), 1e-8)),
        "mae_pips": float(adverse_move / max(float(tick_size), 1e-8)),
        "tp_level": round(tp_level, 5),
        "sl_level": round(sl_level, 5),
        "tp_pips": round(abs(tp_level - entry_price) / max(tick_size, 1e-8), 1),
        "sl_pips": round(abs(sl_level - entry_price) / max(tick_size, 1e-8), 1),
        "rr_ratio": round(abs(tp_level - entry_price) / max(abs(sl_level - entry_price), tick_size), 2),
    }


def build_realized_policy_frame(
    df: pd.DataFrame,
    coverage_mask: np.ndarray,
    *,
    cost_config: dict | None = None,
    replay_config: dict | None = None,
) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    if len(frame) == 0:
        frame["_realized_long_net_pnl_pips"] = pd.Series(dtype=np.float32)
        frame["_realized_short_net_pnl_pips"] = pd.Series(dtype=np.float32)
        return frame

    cfg_cost = dict(cost_config or {})
    cfg_replay = dict(replay_config or {})
    tick_size = max(safe_float(cfg_cost.get("tick_size", 1.0), 1.0), 1e-8)
    tick_value = max(safe_float(cfg_cost.get("tick_value", 10.0), 10.0), 1e-8)
    round_trip_cost_pips = max(safe_float(cfg_cost.get("round_trip_cost_pips", 1.0), 1.0), 0.0)
    commission_per_side = max(safe_float(cfg_cost.get("commission_per_side", 0.0), 0.0), 0.0)
    min_spread_ticks = max(safe_float(cfg_cost.get("min_spread_ticks", 1.0), 1.0), 0.0)
    min_slippage_ticks = max(safe_float(cfg_cost.get("min_slippage_ticks", 1.0), 1.0), 0.0)
    spread_multiplier = max(safe_float(cfg_cost.get("spread_multiplier", 0.5), 0.5), 0.0)
    direction_threshold_ticks = max(safe_float(cfg_replay.get("direction_threshold_ticks", 1.0), 1.0), 0.0)
    tp_mult = max(safe_float(cfg_replay.get("tp_mult", 1.2), 1.2), 0.0)
    sl_mult = max(safe_float(cfg_replay.get("sl_mult", 1.0), 1.0), 0.0)
    max_horizon_steps = int(cfg_replay.get("max_horizon_steps", 0) or 0) or None

    prices = pd.to_numeric(frame.get("price", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    horizons = pd.to_numeric(frame.get("label_horizon_steps", 0), errors="coerce").fillna(0).astype(np.int32).to_numpy()
    micro_atr_source = frame.get("raw__micro_atr", frame.get("micro_atr", 0.0))
    micro_atr = pd.to_numeric(micro_atr_source, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    covered = np.asarray(coverage_mask, dtype=bool).reshape(-1)
    if len(covered) != len(frame):
        raise ValueError(f"coverage_mask length mismatch: {len(covered)} vs {len(frame)}")

    long_pnl = np.full(len(frame), np.nan, dtype=np.float32)
    short_pnl = np.full(len(frame), np.nan, dtype=np.float32)
    long_valid = np.zeros(len(frame), dtype=np.int8)
    short_valid = np.zeros(len(frame), dtype=np.int8)

    for idx in range(len(frame)):
        if not covered[idx]:
            continue
        row_dict = frame.iloc[idx].to_dict()
        for side, target, valid in (
            ("LONG", long_pnl, long_valid),
            ("SHORT", short_pnl, short_valid),
        ):
            trade = simulate_trade_path(
                idx,
                side,
                prices,
                horizons,
                micro_atr,
                tick_size=tick_size,
                direction_threshold_ticks=direction_threshold_ticks,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                max_horizon_steps=max_horizon_steps,
                row_data=row_dict,
            )
            if trade is None:
                continue
            exit_idx = int(trade.get("exit_idx", idx))
            if exit_idx < 0 or exit_idx >= len(frame):
                continue
            exit_row = frame.iloc[exit_idx].to_dict()
            fill = realized_fill_pricing(
                entry_row=row_dict,
                exit_row=exit_row,
                direction=side,
                size=1,
                raw_pnl_pips=float(trade.get("raw_pnl_pips", 0.0) or 0.0),
                tick_size=tick_size,
                round_trip_cost_pips=round_trip_cost_pips,
                tick_value=tick_value,
                commission_per_side=commission_per_side,
                min_spread_ticks=min_spread_ticks,
                min_slippage_ticks=min_slippage_ticks,
                spread_multiplier=spread_multiplier,
            )
            target[idx] = float(fill.get("net_pnl_pips", np.nan))
            valid[idx] = 1

    frame["_realized_long_net_pnl_pips"] = long_pnl
    frame["_realized_short_net_pnl_pips"] = short_pnl
    frame["_realized_long_valid"] = long_valid
    frame["_realized_short_valid"] = short_valid
    frame["_policy_tick_size"] = float(tick_size)
    return frame
