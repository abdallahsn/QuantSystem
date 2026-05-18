from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import feature_intelligence_report as fir
from modules.market_research_features import KylesLambdaEngine
from modules.microstructure import AbsorptionIntensityEngine
from prepare_day_trading import aggregate_mbo_to_bars, detect_microstructure_events


def test_absorption_engine_does_not_collapse_tail_to_legacy_cap() -> None:
    engine = AbsorptionIntensityEngine(window=5, min_price_move=0.0001)
    vals: list[float] = []
    cvd = 0.0
    price = 1.2500

    for i in range(90):
        price += 0.0001 if i % 2 == 0 else -0.0001
        cvd += 2.0 + (i * 0.5)
        vals.append(engine.update(price, cvd))

    s = pd.Series(vals[10:], dtype=float)
    assert s.nunique() > 20
    assert float(np.mean(np.isclose(s.to_numpy(), 10.0, atol=1e-9))) == 0.0


def test_kyle_zscore_is_signed_and_bar_aggregation_preserves_abs_peak() -> None:
    engine = KylesLambdaEngine(window=12, output_mode="zscore")
    prices = [100, 101, 100, 102, 99, 101, 98, 100, 97, 99, 96, 98, 95, 97, 94]
    vals = [engine.update(price, 1.0) for price in prices]

    assert any(v > 0 for v in vals)
    assert any(v < 0 for v in vals)

    ticks = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:02:00"],
                utc=True,
            ).tz_localize(None),
            "price": [100.0, 100.1, 100.0],
            "size": [1.0, 1.0, 1.0],
            "action": ["T", "T", "T"],
            "side": ["A", "B", "B"],
            "kyle_lambda": [0.2, -0.8, 0.4],
        }
    )
    bars = aggregate_mbo_to_bars(ticks, freq="5min")

    assert float(bars.loc[0, "kyle_lambda"]) == pytest.approx(-0.8)


def test_bar_ohlcv_uses_valid_trade_prices_only() -> None:
    ticks = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:01:00",
                    "2024-01-01 00:02:00",
                    "2024-01-01 00:03:00",
                ],
                utc=True,
            ).tz_localize(None),
            "price": [np.nan, 1.2500, 0.0, 1.2510],
            "size": [100.0, 2.0, 50.0, 3.0],
            "action": ["R", "T", "A", "F"],
            "side": ["N", "A", "B", "B"],
        }
    )

    bars = aggregate_mbo_to_bars(ticks, freq="5min")

    assert len(bars) == 1
    assert float(bars.loc[0, "open"]) == pytest.approx(1.2500)
    assert float(bars.loc[0, "close"]) == pytest.approx(1.2510)
    assert float(bars.loc[0, "volume"]) == pytest.approx(5.0)
    assert int(bars.loc[0, "tick_count"]) == 2


def test_causal_event_gate_is_not_blocked_by_fixed_threshold() -> None:
    n = 80
    df = pd.DataFrame(
        {
            "hawkes_intensity": np.r_[np.zeros(40), np.linspace(0, 10, 40)],
            "absorption_intensity": np.r_[np.zeros(40), np.linspace(0, 5, 40)],
            "kyle_lambda": np.r_[np.zeros(40), np.linspace(-2, 3, 40)],
            "cvd_direction_pct": np.r_[np.zeros(40), np.ones(40)],
            "regime_label": ["ranging"] * n,
        }
    )

    out = detect_microstructure_events(df, event_target_rate=0.2, event_min_score_floor=0.2)

    assert int(out["is_event"].sum()) > 0
    assert float(out["event_threshold"].min()) < 0.6


def test_feature_intelligence_mutual_information_returns_score_when_available() -> None:
    if fir.mutual_info_classif is None:
        pytest.skip("sklearn mutual_info_classif is unavailable")

    x = pd.Series([0, 0, 0, 1, 1, 1], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    score = fir._mi_score(x, y)

    assert np.isfinite(score)
    assert score >= 0.0
