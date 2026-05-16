from __future__ import annotations

import pandas as pd

from prepare_day_trading import build_day_trading_labels


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
