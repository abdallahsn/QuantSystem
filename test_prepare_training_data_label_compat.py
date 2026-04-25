import pandas as pd

import prepare_training_data as ptd


def test_label_builder_compat_keeps_supported_kwargs(monkeypatch):
    received = {}

    def fake_builder(df, horizon=0, event_roll_window=0, kalman_slope_threshold=0.0):
        received["df"] = df
        received["horizon"] = horizon
        received["event_roll_window"] = event_roll_window
        received["kalman_slope_threshold"] = kalman_slope_threshold
        return df.assign(ok=1)

    monkeypatch.setattr(ptd, "build_causal_event_labels", fake_builder)

    df = pd.DataFrame({"price": [1.0]})
    out = ptd._call_build_causal_event_labels(
        df,
        horizon=64,
        event_roll_window=50,
        kalman_slope_threshold=0.05,
    )

    assert received["df"] is df
    assert received["horizon"] == 64
    assert received["event_roll_window"] == 50
    assert received["kalman_slope_threshold"] == 0.05
    assert "ok" in out.columns


def test_label_builder_compat_drops_unsupported_kwargs(monkeypatch):
    received = {}

    def old_builder(df, horizon=0, event_roll_window=0):
        received["df"] = df
        received["horizon"] = horizon
        received["event_roll_window"] = event_roll_window
        return df.assign(ok=1)

    monkeypatch.setattr(ptd, "build_causal_event_labels", old_builder)

    df = pd.DataFrame({"price": [1.0]})
    out = ptd._call_build_causal_event_labels(
        df,
        horizon=32,
        event_roll_window=25,
        kalman_slope_threshold=0.05,
        trend_strength_min=0.10,
    )

    assert received["df"] is df
    assert received["horizon"] == 32
    assert received["event_roll_window"] == 25
    assert "ok" in out.columns

