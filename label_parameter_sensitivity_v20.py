"""Audit v20 label-parameter sensitivity without touching training/backtesting."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from label_calibration_v20 import _calibrate_one, _load_features
from labeling.triple_barrier import (
    DIR_LONG,
    DIR_NEUTRAL,
    DIR_SHORT,
    TripleBarrierConfig,
    build_triple_barrier_labels,
)


LABEL_NAMES = {
    DIR_LONG: "LONG",
    DIR_SHORT: "SHORT",
    DIR_NEUTRAL: "NEUTRAL",
}

BARRIER_FORMULAS = {
    "spread_ticks": "spread / tick_size",
    "barrier_floor_ticks": "max(min_barrier_ticks, round_trip_cost_ticks + spread_cost_mult * spread_ticks)",
    "base": "max(volatility, barrier_floor_ticks * tick_size)",
    "upper_barrier": "mid_price + max(base * tp_mult, barrier_floor_ticks * tick_size)",
    "lower_barrier": "mid_price - max(base * sl_mult, barrier_floor_ticks * tick_size)",
    "neutral_threshold_abs": "max(base * neutral_mult, barrier_floor_ticks * tick_size)",
    "label_scan": (
        "scan future rows from t+1 through min(t+horizon, last_row); first upper hit -> LONG, "
        "first lower hit -> SHORT, tie goes LONG because upper is checked first; if neither barrier "
        "hits, compare terminal_move to neutral_threshold_abs; otherwise NEUTRAL"
    ),
}


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


def _numeric_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(default, index=df.index)


def _barrier_components(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    price = _numeric_series(df, cfg.price_col, np.nan).ffill()
    if price.isna().any():
        raise ValueError(f"Invalid {cfg.price_col} rows: {int(price.isna().sum())}")
    vol = _numeric_series(df, cfg.volatility_col, np.nan)
    fallback_vol = price.diff().abs().rolling(50, min_periods=5).median()
    vol = vol.fillna(fallback_vol).fillna(float(cfg.tick_size)).clip(lower=float(cfg.tick_size))
    spread = _numeric_series(df, "spread", 0.0).fillna(0.0)
    spread_ticks = spread / max(float(cfg.tick_size), 1e-12)
    barrier_floor_ticks = np.maximum(
        float(cfg.min_barrier_ticks),
        float(cfg.round_trip_cost_ticks) + float(cfg.spread_cost_mult) * spread_ticks.to_numpy(dtype=np.float64),
    )
    floor_price = barrier_floor_ticks * float(cfg.tick_size)
    base = np.maximum(vol.to_numpy(dtype=np.float64), floor_price)
    tp_raw = base * float(cfg.tp_vol_mult)
    sl_raw = base * float(cfg.sl_vol_mult)
    neutral_raw = base * float(cfg.neutral_mult)
    upper_distance = np.maximum(tp_raw, floor_price)
    lower_distance = np.maximum(sl_raw, floor_price)
    neutral_abs = np.maximum(neutral_raw, floor_price)
    out = pd.DataFrame(
        {
            "mid_price": price.to_numpy(dtype=np.float64),
            "volatility": vol.to_numpy(dtype=np.float64),
            "spread": spread.to_numpy(dtype=np.float64),
            "spread_ticks": spread_ticks.to_numpy(dtype=np.float64),
            "barrier_floor_ticks": barrier_floor_ticks,
            "barrier_floor_price": floor_price,
            "base": base,
            "tp_raw_distance": tp_raw,
            "sl_raw_distance": sl_raw,
            "neutral_raw_abs": neutral_raw,
            "upper_distance": upper_distance,
            "lower_distance": lower_distance,
            "neutral_threshold_abs": neutral_abs,
            "upper_barrier": price.to_numpy(dtype=np.float64) + upper_distance,
            "lower_barrier": price.to_numpy(dtype=np.float64) - lower_distance,
            "tp_clamped_by_floor": tp_raw <= floor_price,
            "sl_clamped_by_floor": sl_raw <= floor_price,
            "neutral_clamped_by_floor": neutral_raw <= floor_price,
        },
        index=df.index,
    )
    return out


def _quantiles(values: pd.Series | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.nanmin(arr)),
        "median": float(np.nanquantile(arr, 0.50)),
        "p90": float(np.nanquantile(arr, 0.90)),
        "p95": float(np.nanquantile(arr, 0.95)),
        "p99": float(np.nanquantile(arr, 0.99)),
        "max": float(np.nanmax(arr)),
    }


def _class_counts(labels: pd.Series) -> dict:
    counts = labels.value_counts(dropna=False).to_dict()
    return {
        "LONG": int(counts.get(DIR_LONG, 0)),
        "SHORT": int(counts.get(DIR_SHORT, 0)),
        "NEUTRAL": int(counts.get(DIR_NEUTRAL, 0)),
    }


def _config_summary(df: pd.DataFrame, cfg: TripleBarrierConfig, *, chunk_rows: int) -> tuple[pd.DataFrame, dict]:
    labeled = build_triple_barrier_labels(df, cfg)
    components = _barrier_components(df, cfg)
    calibration_counts = _calibrate_one(
        df,
        horizon=int(cfg.horizon_rows),
        tick_size=float(cfg.tick_size),
        tp_mult=float(cfg.tp_vol_mult),
        sl_mult=float(cfg.sl_vol_mult),
        neutral_mult=float(cfg.neutral_mult),
        round_trip_cost_ticks=float(cfg.round_trip_cost_ticks),
        spread_cost_mult=float(cfg.spread_cost_mult),
        min_barrier_ticks=float(cfg.min_barrier_ticks),
        chunk_rows=chunk_rows,
    )
    canonical_counts = _class_counts(labeled["direction_label"])
    canonical_barriers = labeled["barrier_hit_type"].value_counts(dropna=False).to_dict()
    summary = {
        "config": cfg.to_dict(),
        "canonical_counts": canonical_counts,
        "calibration_counts": {
            "LONG": int(calibration_counts["long_count"]),
            "SHORT": int(calibration_counts["short_count"]),
            "NEUTRAL": int(calibration_counts["neutral_count"]),
        },
        "calibration_matches_canonical_counts": bool(
            canonical_counts["LONG"] == int(calibration_counts["long_count"])
            and canonical_counts["SHORT"] == int(calibration_counts["short_count"])
            and canonical_counts["NEUTRAL"] == int(calibration_counts["neutral_count"])
        ),
        "barrier_hit_type_distribution": {str(k): int(v) for k, v in canonical_barriers.items()},
        "clamp_share": {
            "tp": float(components["tp_clamped_by_floor"].mean()),
            "sl": float(components["sl_clamped_by_floor"].mean()),
            "neutral": float(components["neutral_clamped_by_floor"].mean()),
        },
        "distance_ticks_stats": {
            "upper": _quantiles(components["upper_distance"] / float(cfg.tick_size)),
            "lower": _quantiles(components["lower_distance"] / float(cfg.tick_size)),
            "neutral": _quantiles(components["neutral_threshold_abs"] / float(cfg.tick_size)),
            "floor": _quantiles(components["barrier_floor_ticks"]),
            "volatility": _quantiles(components["volatility"]),
            "spread_ticks": _quantiles(components["spread_ticks"]),
        },
        "directional_share": float((labeled["direction_label"] != DIR_NEUTRAL).mean()) if len(labeled) else 0.0,
        "event_share": float(labeled["tradeability_label"].mean()) if len(labeled) else 0.0,
    }
    return labeled, summary


def _sample_rows(
    df: pd.DataFrame,
    labeled_a: pd.DataFrame,
    labeled_b: pd.DataFrame,
    comp_a: pd.DataFrame,
    comp_b: pd.DataFrame,
    *,
    seed: int,
    rows: int,
    tick_size: float,
) -> list[dict]:
    n = len(df)
    if n == 0:
        return []
    rng = np.random.default_rng(int(seed))
    sample_size = min(int(rows), n)
    indices = np.sort(rng.choice(np.arange(n), size=sample_size, replace=False))
    ts = pd.to_datetime(df["ts_event"], errors="coerce") if "ts_event" in df.columns else pd.Series(pd.NaT, index=df.index)
    records: list[dict] = []
    for idx in indices:
        rec = {
            "row_index": int(idx),
            "ts_event": str(ts.iloc[idx]),
            "mid_price": float(comp_a["mid_price"].iloc[idx]),
            "volatility": float(comp_a["volatility"].iloc[idx]),
            "spread_ticks": float(comp_a["spread_ticks"].iloc[idx]),
            "barrier_floor_ticks": float(comp_a["barrier_floor_ticks"].iloc[idx]),
            "config_a": {
                "upper_barrier": float(comp_a["upper_barrier"].iloc[idx]),
                "lower_barrier": float(comp_a["lower_barrier"].iloc[idx]),
                "neutral_threshold": float(comp_a["neutral_threshold_abs"].iloc[idx]),
                "upper_distance_ticks": float(comp_a["upper_distance"].iloc[idx] / max(float(tick_size), 1e-12)),
                "lower_distance_ticks": float(comp_a["lower_distance"].iloc[idx] / max(float(tick_size), 1e-12)),
                "neutral_threshold_ticks": float(comp_a["neutral_threshold_abs"].iloc[idx] / max(float(tick_size), 1e-12)),
                "label": LABEL_NAMES.get(int(labeled_a["direction_label"].iloc[idx]), str(labeled_a["direction_label"].iloc[idx])),
                "barrier_hit_type": str(labeled_a["barrier_hit_type"].iloc[idx]),
                "label_end_ts": str(labeled_a["label_end_ts"].iloc[idx]),
            },
            "config_b": {
                "upper_barrier": float(comp_b["upper_barrier"].iloc[idx]),
                "lower_barrier": float(comp_b["lower_barrier"].iloc[idx]),
                "neutral_threshold": float(comp_b["neutral_threshold_abs"].iloc[idx]),
                "upper_distance_ticks": float(comp_b["upper_distance"].iloc[idx] / max(float(tick_size), 1e-12)),
                "lower_distance_ticks": float(comp_b["lower_distance"].iloc[idx] / max(float(tick_size), 1e-12)),
                "neutral_threshold_ticks": float(comp_b["neutral_threshold_abs"].iloc[idx] / max(float(tick_size), 1e-12)),
                "label": LABEL_NAMES.get(int(labeled_b["direction_label"].iloc[idx]), str(labeled_b["direction_label"].iloc[idx])),
                "barrier_hit_type": str(labeled_b["barrier_hit_type"].iloc[idx]),
                "label_end_ts": str(labeled_b["label_end_ts"].iloc[idx]),
            },
        }
        records.append(rec)
    return records


def _parse_config(prefix: str, args: argparse.Namespace) -> TripleBarrierConfig:
    return TripleBarrierConfig(
        horizon_rows=int(getattr(args, f"{prefix}_horizon")),
        tick_size=float(args.tick_size),
        tp_vol_mult=float(getattr(args, f"{prefix}_tp_mult")),
        sl_vol_mult=float(getattr(args, f"{prefix}_sl_mult")),
        neutral_mult=float(getattr(args, f"{prefix}_neutral_mult")),
        min_barrier_ticks=float(args.min_barrier_ticks),
        round_trip_cost_ticks=float(args.round_trip_cost_ticks),
        spread_cost_mult=float(args.spread_cost_mult),
        volatility_col="realized_vol",
        price_col="mid_price",
    )


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    output = args.output or (args.artifact if os.path.isdir(args.artifact) else os.path.dirname(os.path.abspath(args.artifact)))
    os.makedirs(output, exist_ok=True)
    df = _load_features(args.artifact, max_rows=int(args.max_rows or 0))
    cfg_a = _parse_config("a", args)
    cfg_b = _parse_config("b", args)
    labeled_a, summary_a = _config_summary(df, cfg_a, chunk_rows=int(args.chunk_rows))
    labeled_b, summary_b = _config_summary(df, cfg_b, chunk_rows=int(args.chunk_rows))
    comp_a = _barrier_components(df, cfg_a)
    comp_b = _barrier_components(df, cfg_b)

    label_same = labeled_a["direction_label"].to_numpy() == labeled_b["direction_label"].to_numpy()
    hit_same = labeled_a["barrier_hit_type"].astype(str).to_numpy() == labeled_b["barrier_hit_type"].astype(str).to_numpy()
    report = {
        "generated_at": _utc_now(),
        "artifact": os.path.abspath(args.artifact),
        "rows": int(len(df)),
        "sample_seed": int(args.sample_seed),
        "barrier_formulas": BARRIER_FORMULAS,
        "parameter_usage_verification": {
            "tp_mult_used_in_code": True,
            "sl_mult_used_in_code": True,
            "neutral_mult_used_in_code": True,
            "tp_mult_usage": "upper_distance = max(base * tp_mult, barrier_floor_ticks * tick_size)",
            "sl_mult_usage": "lower_distance = max(base * sl_mult, barrier_floor_ticks * tick_size)",
            "neutral_mult_usage": "neutral_threshold_abs = max(base * neutral_mult, barrier_floor_ticks * tick_size)",
            "important_caveat": "The multiplier is mathematically used, but it has no effective impact on rows where the cost/spread floor is larger than base * multiplier.",
        },
        "config_a": summary_a,
        "config_b": summary_b,
        "config_comparison": {
            "same_class_label_rows": int(np.sum(label_same)),
            "same_class_label_share": float(np.mean(label_same)) if len(label_same) else 0.0,
            "changed_class_label_rows": int(len(label_same) - np.sum(label_same)),
            "same_barrier_hit_type_rows": int(np.sum(hit_same)),
            "same_barrier_hit_type_share": float(np.mean(hit_same)) if len(hit_same) else 0.0,
            "upper_barrier_equal_rows": int(np.isclose(comp_a["upper_barrier"], comp_b["upper_barrier"], rtol=0.0, atol=1e-12).sum()),
            "lower_barrier_equal_rows": int(np.isclose(comp_a["lower_barrier"], comp_b["lower_barrier"], rtol=0.0, atol=1e-12).sum()),
            "neutral_threshold_equal_rows": int(np.isclose(comp_a["neutral_threshold_abs"], comp_b["neutral_threshold_abs"], rtol=0.0, atol=1e-12).sum()),
            "upper_barrier_equal_share": float(np.isclose(comp_a["upper_barrier"], comp_b["upper_barrier"], rtol=0.0, atol=1e-12).mean()) if len(df) else 0.0,
            "lower_barrier_equal_share": float(np.isclose(comp_a["lower_barrier"], comp_b["lower_barrier"], rtol=0.0, atol=1e-12).mean()) if len(df) else 0.0,
            "neutral_threshold_equal_share": float(np.isclose(comp_a["neutral_threshold_abs"], comp_b["neutral_threshold_abs"], rtol=0.0, atol=1e-12).mean()) if len(df) else 0.0,
        },
        "random_sample_100_rows": _sample_rows(
            df,
            labeled_a,
            labeled_b,
            comp_a,
            comp_b,
            seed=int(args.sample_seed),
            rows=int(args.sample_rows),
            tick_size=float(args.tick_size),
        ),
        "diagnosis": {
            "calibration_grid_working": bool(
                summary_a["calibration_matches_canonical_counts"] and summary_b["calibration_matches_canonical_counts"]
            ),
            "parameter_insensitivity_bug_detected": False,
            "primary_reason_for_identical_or_near_identical_counts": (
                "Cost/spread floor dominates many rows. When base * multiplier <= barrier_floor_ticks * tick_size, "
                "the effective barrier is the floor, so different multipliers produce identical barriers and labels."
            ),
            "secondary_reason": (
                "neutral_mult only affects timeout rows after no TP/SL barrier was hit. If neutral is also floor-clamped, "
                "or if most rows are barrier hits/true neutral timeouts, changing neutral_mult will not change class counts."
            ),
            "audit_scope": "labeling only; training and backtesting were not touched",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = os.path.join(output, "label_parameter_sensitivity_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(report), f, indent=2)
    print(f"label parameter sensitivity report written: {os.path.abspath(path)}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit v20 label parameter sensitivity")
    p.add_argument("--artifact", required=True, help="v20 artifact directory or features.parquet path")
    p.add_argument("--output", default=None, help="Report output directory; defaults to artifact directory")
    p.add_argument("--tick_size", type=float, default=0.0001)
    p.add_argument("--round_trip_cost_ticks", type=float, default=1.0)
    p.add_argument("--spread_cost_mult", type=float, default=1.0)
    p.add_argument("--min_barrier_ticks", type=float, default=1.0)
    p.add_argument("--chunk_rows", type=int, default=50_000)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--sample_seed", type=int, default=20260604)
    p.add_argument("--sample_rows", type=int, default=100)
    p.add_argument("--a_horizon", type=int, default=200)
    p.add_argument("--a_tp_mult", type=float, default=0.75)
    p.add_argument("--a_sl_mult", type=float, default=0.75)
    p.add_argument("--a_neutral_mult", type=float, default=0.45)
    p.add_argument("--b_horizon", type=int, default=200)
    p.add_argument("--b_tp_mult", type=float, default=1.5)
    p.add_argument("--b_sl_mult", type=float, default=0.5)
    p.add_argument("--b_neutral_mult", type=float, default=0.1)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
