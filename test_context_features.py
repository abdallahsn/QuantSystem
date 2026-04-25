import pytest
import pandas as pd

from modules.context_features import DailyContextEngine


def test_daily_context_engine_uses_default_adr_until_min_samples():
    eng = DailyContextEngine(
        default_adr=80.0 * 0.0001,
        adr_lookback_days=20,
        min_adr_samples=3,
        tick_size=0.0001,
    )

    day1 = pd.Timestamp("2025-01-01 08:00:00")
    eng.update(day1, 1.2000)
    eng.update(day1 + pd.Timedelta(hours=2), 1.2040)

    day2 = pd.Timestamp("2025-01-02 08:00:00")
    _, remaining_fuel, _, adr_pips = eng.update(day2, 1.3000)

    assert adr_pips == pytest.approx(80.0)
    assert remaining_fuel == pytest.approx(0.0080, abs=1e-6)


def test_daily_context_engine_switches_to_median_adr_after_min_samples():
    eng = DailyContextEngine(
        default_adr=80.0 * 0.0001,
        adr_lookback_days=20,
        min_adr_samples=3,
        tick_size=0.0001,
    )

    samples = [
        ("2025-01-01", 1.2000, 1.2040),  # 0.0040
        ("2025-01-02", 1.3000, 1.3060),  # 0.0060
        ("2025-01-03", 1.4000, 1.4050),  # 0.0050
    ]

    for date_str, open_px, high_px in samples:
        day = pd.Timestamp(f"{date_str} 08:00:00")
        eng.update(day, open_px)
        eng.update(day + pd.Timedelta(hours=2), high_px)

    day4 = pd.Timestamp("2025-01-04 08:00:00")
    _, remaining_fuel, _, adr_pips = eng.update(day4, 1.5000)

    assert adr_pips == pytest.approx(50.0)
    assert remaining_fuel == pytest.approx(0.0050, abs=1e-6)
