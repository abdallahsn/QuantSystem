from __future__ import annotations

import pandas as pd

from prepare_day_trading import build_day_trading_labels, label_by_outcome


def _utc_ts(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _bars(*, highs, lows, closes=None, opens=None):
    n = len(highs)
    closes = closes or [100.0] * n
    opens = opens or closes
    return pd.DataFrame(
        {
            "ts_event": pd.date_range("2024-01-01 07:00:00", periods=n, freq="5min", tz="UTC"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr_14": [1.0] * n,
            "is_london": [1] * n,
            "is_overlap": [0] * n,
        }
    )


def test_long_tp_first_produces_long_label_and_first_hit_end_ts():
    df = _bars(highs=[100.0, 100.6, 100.0, 100.0], lows=[100.0, 99.8, 100.0, 100.0])

    out = build_day_trading_labels(df, horizon_bars=3, tp_atr_mult=0.5, sl_atr_mult=1.0)

    assert int(out.loc[0, "bias_label"]) == 0
    assert int(out.loc[0, "path_outcome"]) == 0
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[1, "ts_event"])


def test_short_tp_first_produces_short_label_and_first_hit_end_ts():
    df = _bars(highs=[100.0, 100.2, 100.0, 100.0], lows=[100.0, 99.4, 100.0, 100.0])

    out = build_day_trading_labels(df, horizon_bars=3, tp_atr_mult=0.5, sl_atr_mult=1.0)

    assert int(out.loc[0, "bias_label"]) == 1
    assert int(out.loc[0, "path_outcome"]) == 1
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[1, "ts_event"])


def test_long_sl_first_produces_neutral_with_sl_metadata():
    df = _bars(highs=[100.0, 100.4, 100.0, 100.0], lows=[100.0, 98.9, 100.0, 100.0])

    out = build_day_trading_labels(df, horizon_bars=3, tp_atr_mult=1.5, sl_atr_mult=1.0)

    assert int(out.loc[0, "bias_label"]) == 2
    assert int(out.loc[0, "signal_quality"]) == 0
    assert int(out.loc[0, "path_outcome"]) == 2
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[1, "ts_event"])


def test_short_sl_first_produces_neutral_with_sl_metadata():
    df = _bars(highs=[100.0, 101.1, 100.0, 100.0], lows=[100.0, 99.8, 100.0, 100.0])

    out = build_day_trading_labels(df, horizon_bars=3, tp_atr_mult=1.5, sl_atr_mult=1.0)

    assert int(out.loc[0, "bias_label"]) == 2
    assert int(out.loc[0, "signal_quality"]) == 0
    assert int(out.loc[0, "path_outcome"]) == 3
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[1, "ts_event"])


def test_timeout_produces_neutral_and_horizon_end_ts():
    df = _bars(
        highs=[100.0, 100.2, 100.2, 100.2],
        lows=[100.0, 99.8, 99.8, 99.8],
    )

    out = build_day_trading_labels(df, horizon_bars=3, tp_atr_mult=1.5, sl_atr_mult=1.0)

    assert int(out.loc[0, "bias_label"]) == 2
    assert int(out.loc[0, "path_outcome"]) == 4
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[3, "ts_event"])


def _event_bars(*, highs, lows, closes=None, opens=None, event_direction=1):
    df = _bars(highs=highs, lows=lows, closes=closes, opens=opens)
    df["is_event"] = 1
    df["event_score"] = 0.9
    df["event_direction"] = event_direction
    df["kalman_direction"] = 0
    return df


def test_label_by_outcome_records_first_hit_end_ts_and_horizon_steps():
    df = _event_bars(highs=[100.0, 101.2, 100.0, 100.0], lows=[100.0, 99.8, 100.0, 100.0])

    out = label_by_outcome(df, default_tp_mult=0.5, default_sl_mult=1.0, default_max_bars=3, min_atr=0.1)

    assert int(out.loc[0, "bias_label"]) == 0
    assert int(out.loc[0, "path_outcome"]) == 0
    assert int(out.loc[0, "label_horizon_steps"]) == 1
    assert int(out.loc[0, "effective_horizon"]) == 1
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[1, "ts_event"])


def test_label_by_outcome_timeout_records_session_cap_end_ts():
    df = _event_bars(highs=[100.0, 100.2, 100.2, 100.2], lows=[100.0, 99.8, 99.8, 99.8])

    out = label_by_outcome(df, default_tp_mult=1.5, default_sl_mult=1.0, default_max_bars=3, min_atr=0.1)

    assert int(out.loc[0, "bias_label"]) == 2
    assert int(out.loc[0, "path_outcome"]) == 4
    assert int(out.loc[0, "label_horizon_steps"]) == 3
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[3, "ts_event"])


def test_label_by_outcome_non_event_without_weak_conversion_is_zero_horizon():
    df = _event_bars(highs=[100.0, 101.0, 101.0, 101.0], lows=[100.0, 99.0, 99.0, 99.0])
    df["is_event"] = 0

    out = label_by_outcome(df, default_tp_mult=0.5, default_sl_mult=1.0, default_max_bars=3, min_atr=0.1)

    assert int(out.loc[0, "bias_label"]) == 2
    assert int(out.loc[0, "label_horizon_steps"]) == 0
    assert _utc_ts(out.loc[0, "label_end_ts"]) == _utc_ts(df.loc[0, "ts_event"])
