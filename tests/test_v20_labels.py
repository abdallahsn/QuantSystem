import pandas as pd
import pytest

from labeling.triple_barrier import DIR_LONG, DIR_NEUTRAL, TripleBarrierConfig, build_triple_barrier_labels


def test_v20_labels_include_required_barrier_metadata():
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-01-01", periods=8, freq="s"),
            "mid_price": [1.2500, 1.2501, 1.2502, 1.2504, 1.2503, 1.2502, 1.2502, 1.2502],
            "spread": [0.0001] * 8,
            "realized_vol": [0.0001] * 8,
        }
    )
    labels = build_triple_barrier_labels(
        df,
        TripleBarrierConfig(
            horizon_rows=4,
            tick_size=0.0001,
            tp_vol_mult=1.0,
            sl_vol_mult=1.0,
            round_trip_cost_ticks=0.0,
            spread_cost_mult=0.0,
        ),
    )

    required = {
        "direction_label",
        "tradeability_label",
        "bias_label",
        "event_flag",
        "train_event_flag",
        "label_end_ts",
        "horizon_end_ts",
        "realized_return",
        "forward_return",
        "barrier_hit_type",
    }
    assert required.issubset(labels.columns)
    assert labels["label_end_ts"].ge(labels["ts_event"]).all()
    assert int(labels["bias_label"].iloc[0]) == DIR_LONG
    assert int(labels["train_event_flag"].iloc[0]) == 1


def test_timeout_direction_can_be_non_tradeable():
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-01-01", periods=5, freq="s"),
            "mid_price": [1.2500, 1.2501, 1.2501, 1.2502, 1.2502],
            "spread": [0.0001] * 5,
            "realized_vol": [0.0001] * 5,
        }
    )
    labels = build_triple_barrier_labels(
        df,
        TripleBarrierConfig(
            horizon_rows=4,
            tick_size=0.0001,
            tp_vol_mult=10.0,
            sl_vol_mult=10.0,
            neutral_mult=0.5,
            round_trip_cost_ticks=0.0,
            spread_cost_mult=0.0,
        ),
    )

    assert int(labels["bias_label"].iloc[0]) == DIR_LONG
    assert int(labels["tradeability_label"].iloc[0]) == 0
    assert labels["barrier_hit_type"].iloc[0] == "timeout_directional_up"


def test_labeler_refuses_missing_causal_price_rows():
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-01-01", periods=3, freq="s"),
            "mid_price": [None, 1.2501, 1.2502],
            "spread": [0.0001] * 3,
            "realized_vol": [0.0001] * 3,
        }
    )
    with pytest.raises(ValueError, match="missing causal price"):
        build_triple_barrier_labels(df, TripleBarrierConfig(horizon_rows=2))
