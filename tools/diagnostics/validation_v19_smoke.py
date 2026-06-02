"""
Smoke checks for QuantSystem V19 validation guards.

This script avoids model imports. It verifies that chronology, label leakage,
label distribution, market-data, and backtest-realism checks behave as expected.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.validation_v19 import (
    assert_no_label_leakage,
    assert_label_timestamps,
    summarize_label_distribution,
    validate_backtest_realism_config,
    validate_market_data_frame,
    validate_time_splits,
)


def _sample_frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=20, freq="s")
    return pd.DataFrame(
        {
            "ts_event": ts,
            "label_end_ts": ts + pd.to_timedelta([1] * 10 + [0] * 10, unit="s"),
            "bias_label": [0, 1, 2, 2, 0, 1, 2, 2, 0, 1] + [2] * 10,
            "train_event_flag": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1] + [0] * 10,
            "price": np.linspace(1.20, 1.21, 20),
            "bid_px_00": np.linspace(1.1999, 1.2099, 20),
            "ask_px_00": np.linspace(1.2001, 1.2101, 20),
        }
    )


def main() -> int:
    df = _sample_frame()
    assert_label_timestamps(df, context="smoke.clean_labels")
    market = validate_market_data_frame(df, context="smoke.market", tick_size=0.0001, strict=True)
    labels = summarize_label_distribution(df, context="smoke.labels")

    train_idx = np.arange(0, 9, dtype=np.int32)
    test_idx = np.arange(10, 20, dtype=np.int32)
    split_report = validate_time_splits(
        [(train_idx, test_idx)],
        df["ts_event"],
        df["label_end_ts"],
        context="smoke.split",
        strict=True,
    )

    assert_no_label_leakage(df.iloc[:9], cutoff_ts=df["ts_event"].iloc[10], context="smoke.no_leak")

    leaked = df.iloc[:11].copy()
    leaked.loc[leaked.index[-1], "label_end_ts"] = df["ts_event"].iloc[12]
    try:
        assert_no_label_leakage(leaked, cutoff_ts=df["ts_event"].iloc[10], context="smoke.expected_leak")
    except ValueError:
        leak_detected = True
    else:
        leak_detected = False

    if not leak_detected:
        raise AssertionError("expected label leakage was not detected")

    realism = validate_backtest_realism_config(
        {
            "tick_size": 0.0001,
            "tick_value": 10.0,
            "round_trip_cost_pips": 1.0,
            "commission_per_side": 2.5,
            "min_spread_ticks": 1.0,
            "min_slippage_ticks": 1.0,
            "latency_rows": 1,
            "max_size": 2,
        },
        context="smoke.backtest_realism",
        strict=True,
    )

    print(
        "validation_v19_smoke OK | "
        f"market_passed={market['passed']} | "
        f"directional_rows={labels['directional_rows']} | "
        f"split_passed={split_report['passed']} | "
        f"realism_passed={realism['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
