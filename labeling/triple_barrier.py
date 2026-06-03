"""Cost-aware triple-barrier labels with label_end_ts."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon_rows: int = 150
    tick_size: float = 0.0001
    tp_vol_mult: float = 1.5
    sl_vol_mult: float = 1.0
    neutral_mult: float = 0.5
    min_barrier_ticks: float = 1.0
    round_trip_cost_ticks: float = 1.0
    spread_cost_mult: float = 1.0
    volatility_col: str = "realized_vol"
    price_col: str = "mid_price"

    def to_dict(self) -> dict:
        return asdict(self)


def _series(df: pd.DataFrame, col: str, fallback: str | None = None) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if fallback and fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    raise ValueError(f"Missing required price column: {col}")


def build_triple_barrier_labels(df: pd.DataFrame, config: TripleBarrierConfig | None = None) -> pd.DataFrame:
    cfg = config or TripleBarrierConfig()
    if "ts_event" not in df.columns:
        raise ValueError("build_triple_barrier_labels requires ts_event")

    out = df.copy()
    ts = pd.to_datetime(out["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    if ts.isna().any():
        raise ValueError(f"Invalid ts_event rows for labeling: {int(ts.isna().sum())}")

    price_series = _series(out, cfg.price_col, fallback="price").ffill()
    if price_series.isna().any():
        bad = int(price_series.isna().sum())
        raise ValueError(f"Cannot build labels with missing causal price rows: {bad}")
    price = price_series.to_numpy(dtype=np.float64)
    vol = (
        pd.to_numeric(out[cfg.volatility_col], errors="coerce")
        if cfg.volatility_col in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    fallback_vol = pd.Series(price).diff().abs().rolling(50, min_periods=5).median()
    vol = vol.fillna(fallback_vol).fillna(float(cfg.tick_size)).clip(lower=float(cfg.tick_size))
    spread_ticks = (
        pd.to_numeric(out.get("spread", 0.0), errors="coerce").fillna(0.0) / max(float(cfg.tick_size), 1e-12)
    )
    barrier_floor_ticks = np.maximum(
        float(cfg.min_barrier_ticks),
        float(cfg.round_trip_cost_ticks) + float(cfg.spread_cost_mult) * spread_ticks.to_numpy(dtype=np.float64),
    )
    base = np.maximum(vol.to_numpy(dtype=np.float64), barrier_floor_ticks * float(cfg.tick_size))
    upper = price + np.maximum(base * float(cfg.tp_vol_mult), barrier_floor_ticks * float(cfg.tick_size))
    lower = price - np.maximum(base * float(cfg.sl_vol_mult), barrier_floor_ticks * float(cfg.tick_size))
    neutral_abs = np.maximum(base * float(cfg.neutral_mult), barrier_floor_ticks * float(cfg.tick_size))

    n = len(out)
    label = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    tradeable = np.zeros(n, dtype=np.int8)
    end_idx = np.arange(n, dtype=np.int32)
    outcome = np.full(n, "timeout_neutral", dtype=object)

    for i in range(n):
        stop = min(i + int(cfg.horizon_rows), n - 1)
        for j in range(i + 1, stop + 1):
            if price[j] >= upper[i]:
                label[i] = DIR_LONG
                tradeable[i] = 1
                end_idx[i] = j
                outcome[i] = "upper_hit"
                break
            if price[j] <= lower[i]:
                label[i] = DIR_SHORT
                tradeable[i] = 1
                end_idx[i] = j
                outcome[i] = "lower_hit"
                break
        if end_idx[i] == i and stop > i:
            end_idx[i] = stop
            terminal_move = price[stop] - price[i]
            if terminal_move >= neutral_abs[i]:
                label[i] = DIR_LONG
                outcome[i] = "timeout_directional_up"
            elif terminal_move <= -neutral_abs[i]:
                label[i] = DIR_SHORT
                outcome[i] = "timeout_directional_down"
        elif stop == i:
            outcome[i] = "no_future"

    realized_return = price[end_idx] - price

    out["direction_label"] = label
    out["tradeability_label"] = tradeable
    out["bias_label"] = label
    out["event_flag"] = (label != DIR_NEUTRAL).astype(np.int8)
    out["train_event_flag"] = tradeable.astype(np.int8)
    out["label_end_ts"] = pd.to_datetime(ts.iloc[end_idx].to_numpy())
    out["horizon_end_ts"] = pd.to_datetime(ts.iloc[np.minimum(np.arange(n) + int(cfg.horizon_rows), n - 1)].to_numpy())
    out["label_horizon_steps"] = (end_idx - np.arange(n)).astype(np.int32)
    out["label_outcome"] = outcome
    out["barrier_hit_type"] = outcome
    out["realized_return"] = realized_return.astype(np.float32)
    out["forward_return"] = out["realized_return"]
    out["label_barrier_upper"] = upper.astype(np.float32)
    out["label_barrier_lower"] = lower.astype(np.float32)
    out["label_neutral_abs"] = neutral_abs.astype(np.float32)
    return out
