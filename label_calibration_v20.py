"""Calibrate v20 triple-barrier label parameters from a prepared artifact.

This utility is intentionally preparation-only. It reads causal v20 features,
recomputes cost/spread-aware labels across a parameter grid, and writes reports
that help choose a label configuration before starting Phase 3 training work.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import pandas as pd


DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2


def _utc_now() -> str:
    return pd.Timestamp.utcnow().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _parse_int_grid(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Grid cannot be empty")
    return values


def _parse_float_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Grid cannot be empty")
    return values


def _artifact_features_path(path: str) -> str:
    if os.path.isdir(path):
        direct = os.path.join(path, "features.parquet")
        if os.path.exists(direct):
            return direct
        final_dir = os.path.join(path, "final")
        if os.path.isdir(final_dir):
            shards = sorted(
                os.path.join(final_dir, name)
                for name in os.listdir(final_dir)
                if name.endswith(".parquet")
            )
            if shards:
                return shards[0]
        raise FileNotFoundError(f"No features.parquet or final parquet shard found under {path}")
    return path


def _load_features(path: str, *, max_rows: int = 0) -> pd.DataFrame:
    features_path = _artifact_features_path(path)
    df = pd.read_parquet(features_path)
    if max_rows > 0:
        df = df.head(max_rows).copy()
    required = ("ts_event", "mid_price")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for calibration: {missing}")
    df = df.sort_values("ts_event").reset_index(drop=True)
    return df


def _numeric_array(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        series = pd.Series(default, index=df.index)
    return series.fillna(default).to_numpy(dtype=np.float64)


def _calibrate_one(
    df: pd.DataFrame,
    *,
    horizon: int,
    tick_size: float,
    tp_mult: float,
    sl_mult: float,
    neutral_mult: float,
    round_trip_cost_ticks: float,
    spread_cost_mult: float,
    min_barrier_ticks: float,
    chunk_rows: int,
) -> dict:
    n = int(len(df))
    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    if ts.isna().any():
        raise ValueError(f"Invalid ts_event rows in calibration artifact: {int(ts.isna().sum())}")
    price = _numeric_array(df, "mid_price", np.nan)
    if np.isnan(price).any():
        raise ValueError(f"Invalid mid_price rows in calibration artifact: {int(np.isnan(price).sum())}")

    vol = _numeric_array(df, "realized_vol", np.nan)
    fallback_vol = pd.Series(price).diff().abs().rolling(50, min_periods=5).median().to_numpy(dtype=np.float64)
    vol = np.where(np.isfinite(vol), vol, fallback_vol)
    vol = np.nan_to_num(vol, nan=float(tick_size), posinf=float(tick_size), neginf=float(tick_size))
    vol = np.maximum(vol, float(tick_size))

    spread = _numeric_array(df, "spread", 0.0)
    spread_ticks = spread / max(float(tick_size), 1e-12)
    barrier_floor_ticks = np.maximum(
        float(min_barrier_ticks),
        float(round_trip_cost_ticks) + float(spread_cost_mult) * spread_ticks,
    )
    base = np.maximum(vol, barrier_floor_ticks * float(tick_size))
    upper = price + np.maximum(base * float(tp_mult), barrier_floor_ticks * float(tick_size))
    lower = price - np.maximum(base * float(sl_mult), barrier_floor_ticks * float(tick_size))
    neutral_abs = np.maximum(base * float(neutral_mult), barrier_floor_ticks * float(tick_size))

    labels = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    tradeable = np.zeros(n, dtype=np.int8)
    end_idx = np.arange(n, dtype=np.int64)
    outcomes = np.full(n, "timeout_neutral", dtype=object)
    offsets = np.arange(1, int(horizon) + 1, dtype=np.int64)
    rows_all = np.arange(n, dtype=np.int64)
    chunk_size = max(int(chunk_rows), 1)

    for start in range(0, n, chunk_size):
        rows = rows_all[start : start + chunk_size]
        if len(rows) == 0:
            continue
        future_idx = np.minimum(rows[:, None] + offsets[None, :], n - 1)
        future_price = price[future_idx]
        upper_hit = future_price >= upper[rows, None]
        lower_hit = future_price <= lower[rows, None]
        upper_any = upper_hit.any(axis=1)
        lower_any = lower_hit.any(axis=1)
        upper_first = np.where(upper_any, upper_hit.argmax(axis=1), int(horizon) + 1)
        lower_first = np.where(lower_any, lower_hit.argmax(axis=1), int(horizon) + 1)

        use_upper = upper_any & (upper_first <= lower_first)
        use_lower = lower_any & (lower_first < upper_first)
        hit = use_upper | use_lower
        first_hit = np.minimum(upper_first, lower_first)
        chunk_end = np.minimum(rows + np.where(hit, first_hit + 1, int(horizon)), n - 1)

        terminal_move = price[chunk_end] - price[rows]
        labels[rows[use_upper]] = DIR_LONG
        labels[rows[use_lower]] = DIR_SHORT
        tradeable[rows[hit]] = 1
        outcomes[rows[use_upper]] = "upper_hit"
        outcomes[rows[use_lower]] = "lower_hit"

        no_hit = ~hit
        no_hit_rows = rows[no_hit]
        directional_up = no_hit & (terminal_move >= neutral_abs[rows])
        directional_down = no_hit & (terminal_move <= -neutral_abs[rows])
        labels[rows[directional_up]] = DIR_LONG
        labels[rows[directional_down]] = DIR_SHORT
        outcomes[rows[directional_up]] = "timeout_directional_up"
        outcomes[rows[directional_down]] = "timeout_directional_down"
        if len(no_hit_rows):
            no_future = no_hit_rows >= n - 1
            outcomes[no_hit_rows[no_future]] = "no_future"
        end_idx[rows] = chunk_end

    realized_return = price[end_idx] - price
    label_end_ts = ts.iloc[end_idx].reset_index(drop=True)
    feature_ts = ts.reset_index(drop=True)
    leakage_passed = bool(label_end_ts.ge(feature_ts).all())
    long_count = int(np.sum(labels == DIR_LONG))
    short_count = int(np.sum(labels == DIR_SHORT))
    neutral_count = int(np.sum(labels == DIR_NEUTRAL))
    directional_count = long_count + short_count
    outcome_counts = pd.Series(outcomes).value_counts(dropna=False).to_dict()
    return {
        "horizon": int(horizon),
        "tp_mult": float(tp_mult),
        "sl_mult": float(sl_mult),
        "neutral_mult": float(neutral_mult),
        "rows": n,
        "long_count": long_count,
        "short_count": short_count,
        "neutral_count": neutral_count,
        "long_share": float(long_count / max(n, 1)),
        "short_share": float(short_count / max(n, 1)),
        "neutral_share": float(neutral_count / max(n, 1)),
        "directional_share": float(directional_count / max(n, 1)),
        "event_share": float(np.mean(tradeable)) if n else 0.0,
        "long_short_imbalance": float(abs(long_count - short_count) / max(directional_count, 1)),
        "barrier_hit_type_distribution": {str(k): int(v) for k, v in outcome_counts.items()},
        "realized_return_min": float(np.min(realized_return)) if n else None,
        "realized_return_mean": float(np.mean(realized_return)) if n else None,
        "realized_return_median": float(np.median(realized_return)) if n else None,
        "realized_return_std": float(np.std(realized_return)) if n else None,
        "realized_return_p95": float(np.quantile(realized_return, 0.95)) if n else None,
        "realized_return_p99": float(np.quantile(realized_return, 0.99)) if n else None,
        "realized_return_max": float(np.max(realized_return)) if n else None,
        "leakage_precheck_passed": leakage_passed,
    }


def _score_row(row: dict) -> float:
    directional = float(row["directional_share"])
    imbalance = float(row["long_short_imbalance"])
    neutral = float(row["neutral_share"])
    event_share = float(row["event_share"])
    score = abs(directional - 0.12) + 0.25 * imbalance + abs(event_share - min(directional, 0.12))
    if not (0.05 <= directional <= 0.25):
        score += 1.0 + abs(directional - 0.12)
    if neutral <= 0.0:
        score += 1.0
    if imbalance > 0.60:
        score += 0.5
    if not row.get("leakage_precheck_passed", False):
        score += 10.0
    return float(score)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    output = args.output or (args.artifact if os.path.isdir(args.artifact) else os.path.dirname(os.path.abspath(args.artifact)))
    os.makedirs(output, exist_ok=True)
    df = _load_features(args.artifact, max_rows=int(args.max_rows or 0))
    horizons = _parse_int_grid(args.horizons)
    tp_values = _parse_float_grid(args.tp_mults)
    sl_values = _parse_float_grid(args.sl_mults)
    neutral_values = _parse_float_grid(args.neutral_mults)
    rows: list[dict] = []
    for horizon, tp_mult, sl_mult, neutral_mult in itertools.product(horizons, tp_values, sl_values, neutral_values):
        rows.append(
            _calibrate_one(
                df,
                horizon=horizon,
                tick_size=float(args.tick_size),
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                neutral_mult=neutral_mult,
                round_trip_cost_ticks=float(args.round_trip_cost_ticks),
                spread_cost_mult=float(args.spread_cost_mult),
                min_barrier_ticks=float(args.min_barrier_ticks),
                chunk_rows=int(args.chunk_rows),
            )
        )
    for row in rows:
        row["selection_score"] = _score_row(row)
        row["meets_target_band"] = bool(
            row["leakage_precheck_passed"]
            and 0.05 <= float(row["directional_share"]) <= 0.25
            and float(row["neutral_share"]) > 0.0
            and float(row["long_short_imbalance"]) <= 0.60
        )
    grid = pd.DataFrame(rows).sort_values(["selection_score", "horizon", "tp_mult", "sl_mult", "neutral_mult"]).reset_index(drop=True)
    grid_path = os.path.join(output, "label_calibration_grid.csv")
    grid.to_csv(grid_path, index=False)
    eligible = grid.loc[grid["meets_target_band"]]
    best = (eligible.iloc[0] if len(eligible) else grid.iloc[0]).to_dict() if len(grid) else {}
    report = {
        "generated_at": _utc_now(),
        "artifact": os.path.abspath(args.artifact),
        "features_path": os.path.abspath(_artifact_features_path(args.artifact)),
        "rows": int(len(df)),
        "date_range": {
            "start": str(pd.to_datetime(df["ts_event"], errors="coerce").min()) if len(df) else None,
            "end": str(pd.to_datetime(df["ts_event"], errors="coerce").max()) if len(df) else None,
        },
        "grid": {
            "horizons": horizons,
            "tp_mults": tp_values,
            "sl_mults": sl_values,
            "neutral_mults": neutral_values,
            "combinations": int(len(rows)),
        },
        "target": {
            "directional_share_min": 0.05,
            "directional_share_max": 0.25,
            "long_short_imbalance_max": 0.60,
            "neutral_required": True,
        },
        "recommended_config": {
            "horizon": int(best.get("horizon")) if best else None,
            "tp_mult": float(best.get("tp_mult")) if best else None,
            "sl_mult": float(best.get("sl_mult")) if best else None,
            "neutral_mult": float(best.get("neutral_mult")) if best else None,
            "directional_share": float(best.get("directional_share")) if best else None,
            "event_share": float(best.get("event_share")) if best else None,
            "long_short_imbalance": float(best.get("long_short_imbalance")) if best else None,
            "meets_target_band": bool(best.get("meets_target_band")) if best else False,
        },
        "top_10": grid.head(10).to_dict(orient="records"),
        "leakage_precheck_all_passed": bool(grid["leakage_precheck_passed"].all()) if len(grid) else False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "outputs": {
            "label_calibration_report": os.path.abspath(os.path.join(output, "label_calibration_report.json")),
            "label_calibration_grid": os.path.abspath(grid_path),
        },
    }
    report_path = os.path.join(output, "label_calibration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(report), f, indent=2)
    print(
        "label calibration written: "
        f"{os.path.abspath(output)} combinations={len(rows)} recommended={report['recommended_config']}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate v20 cost-aware triple-barrier labels")
    p.add_argument("--artifact", required=True, help="v20 artifact directory or features.parquet path")
    p.add_argument("--output", default=None, help="Report output directory; defaults to artifact directory")
    p.add_argument("--tick_size", type=float, default=0.0001)
    p.add_argument("--horizons", default="50,100,200")
    p.add_argument("--tp_mults", default="0.5,0.75,1.0,1.5")
    p.add_argument("--sl_mults", default="0.5,0.75,1.0")
    p.add_argument("--neutral_mults", default="0.1,0.2,0.3,0.45")
    p.add_argument("--round_trip_cost_ticks", type=float, default=1.0)
    p.add_argument("--spread_cost_mult", type=float, default=1.0)
    p.add_argument("--min_barrier_ticks", type=float, default=1.0)
    p.add_argument("--chunk_rows", type=int, default=50_000)
    p.add_argument("--max_rows", type=int, default=0, help="Optional calibration row cap for smoke tests")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
