#!/usr/bin/env python3
"""Smoke checks for day-trading hardening patches."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prepare_day_trading import (
    add_day_trading_features,
    apply_mbp_lob_imbalance,
    build_rolling_lob_tensors_from_mbp,
    detect_microstructure_events,
)
from tools.diagnostics.verify_day_trading_dataset import LABEL_DERIVED_FEATURES


def _base_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2024-01-01 06:55",
                    "2024-01-01 07:00",
                    "2024-01-01 07:05",
                    "2024-01-01 12:00",
                    "2024-01-01 13:30",
                    "2024-01-01 13:35",
                ],
                utc=True,
            ).tz_localize(None),
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "volume": [100, 120, 130, 140, 150, 160],
            "cvd": [1, 3, 5, 9, 11, 13],
            "current_vwap": [10, 11, 12, 13, 14, 15],
        }
    )


def test_cvd_session_features() -> None:
    feat = add_day_trading_features(_base_bars())
    assert feat["cvd_prev_session"].tolist()[:4] == [0.0, 1.0, 1.0, 5.0]
    assert feat.loc[1, "cvd_session_open_delta"] == 0.0
    assert feat.loc[2, "cvd_session_open_delta"] == 2.0
    assert "cvd_slope_3b" in feat.columns
    assert "cvd_velocity_norm_by_volume" in feat.columns


def test_lob_depth_source() -> None:
    df = pd.DataFrame(
        {
            "order_flow_imbalance": [0.9, 0.8, -0.8],
            "mbp_imbalance_signed_peak": [-0.2, 0.4, 0.1],
            "mbp_bar_coverage": [1.0, 1.0, 0.0],
        }
    )
    out = apply_mbp_lob_imbalance(df)
    assert np.allclose(out["lob_depth_imbalance"].to_numpy(dtype=float), [-0.2, 0.4, 0.0])
    assert np.allclose(out["lob_imbalance"].to_numpy(dtype=float), [-0.2, 0.4, -0.8])
    assert out["lob_imbalance_is_depth"].tolist() == [1, 1, 0]


def test_lob_tensor_near_to_far() -> None:
    bars = _base_bars().iloc[:4].copy()
    mbp_rows = []
    for t in bars["ts_event"]:
        row = {"ts_event": t}
        for i in range(10):
            row[f"bid_px_{i:02d}"] = 100 - i
            row[f"ask_px_{i:02d}"] = 101 + i
            row[f"bid_sz_{i:02d}"] = 100 - i * 5
            row[f"ask_sz_{i:02d}"] = 80 - i * 4
        mbp_rows.append(row)
    mbp = pd.DataFrame(mbp_rows)
    mbo = pd.DataFrame(
        {
            "ts_event": bars["ts_event"],
            "action": ["T"] * len(bars),
            "side": ["A", "B", "A", "B"],
            "price": [101, 100, 101, 100],
            "size": [10, 20, 30, 40],
        }
    )
    t, _, _ = build_rolling_lob_tensors_from_mbp(mbo, mbp, bars, freq="5min", lookback_bars=1, levels=10)
    assert t.shape == (4, 1, 20, 3)
    assert t[0, 0, 0, 0] > t[0, 0, 9, 0]
    assert t[0, 0, 10, 0] > t[0, 0, 19, 0]
    assert t[1, 0, 0, 2] > 0.0


def test_causal_event_gate_smoke() -> None:
    n = 80
    df = pd.DataFrame(
        {
            "hawkes_intensity": np.r_[np.zeros(40), np.linspace(0, 10, 40)],
            "absorption_intensity": np.r_[np.zeros(40), np.linspace(0, 5, 40)],
            "kyle_lambda": np.r_[np.zeros(40), np.linspace(0, 3, 40)],
            "cvd_direction_pct": np.r_[np.zeros(40), np.ones(40)],
            "regime_label": ["ranging"] * n,
        }
    )
    out = detect_microstructure_events(df, event_target_rate=0.2, event_min_score_floor=0.2)
    assert {"event_score", "event_threshold", "is_event"}.issubset(out.columns)
    assert out["is_event"].iloc[:10].sum() == 0


def test_label_artifact_raw_names_are_detectable() -> None:
    leaked = [f"raw__{c}" for c in ("path_outcome", "effective_horizon", "forward_return")]
    detected = [c for c in leaked if c[5:] in set(LABEL_DERIVED_FEATURES)]
    assert detected == leaked


def main() -> int:
    test_cvd_session_features()
    test_lob_depth_source()
    test_lob_tensor_near_to_far()
    test_causal_event_gate_smoke()
    test_label_artifact_raw_names_are_detectable()
    print("✅ day-trading hardening smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
