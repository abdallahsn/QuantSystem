from __future__ import annotations

import numpy as np
import pandas as pd

from train_v19 import (
    TRAIN_MODE_DIRECTIONAL_ALL,
    TRAIN_MODE_EVENT_BINARY,
    _fit_long_isotonic_calibrator,
    _oof_isotonic_enabled,
    _apply_long_calibrator,
    _stable_sort_by_ts_event,
    build_event_training_view,
)


def _sample_training_frame(n: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2025-04-01", periods=n, freq="s")
    bias = np.full(n, 2, dtype=np.int8)
    bias[::3] = 0
    bias[1::3] = 1
    train_event = np.zeros(n, dtype=np.int8)
    train_event[::12] = 1
    train_event[1::12] = 1
    signal_quality = np.where(bias == 2, 0, 2).astype(np.int8)
    return pd.DataFrame(
        {
            "ts_event": ts,
            "label_end_ts": ts,
            "bias_label": bias,
            "event_flag": train_event,
            "train_event_flag": train_event,
            "signal_quality": signal_quality,
            "event_score": np.linspace(0.0, 1.0, n),
        }
    )


def test_directional_all_uses_all_long_short_rows():
    df = _sample_training_frame()
    event_df, info = build_event_training_view(df, mode=TRAIN_MODE_DIRECTIONAL_ALL)

    expected = int(pd.Series(df["bias_label"]).isin([0, 1]).sum())
    assert len(event_df) == expected
    assert set(event_df["bias_label"].unique()) == {0, 1}
    assert info["event_col"] == "bias_label"
    assert info["fallback_reason"] == "directional_all_requested"


def test_event_binary_keeps_selected_directional_events_only():
    df = _sample_training_frame()
    event_df, info = build_event_training_view(df, mode=TRAIN_MODE_EVENT_BINARY)

    expected = int(((df["train_event_flag"] == 1) & df["bias_label"].isin([0, 1])).sum())
    assert len(event_df) == expected
    assert info["event_col"] == "train_event_flag"


def test_isotonic_guard_accepts_when_future_half_improves():
    y = np.array(([0, 1] * 40), dtype=np.int32)
    p_long = np.where(y == 0, 0.60, 0.40).astype(np.float32)

    calibrator, report = _fit_long_isotonic_calibrator(y, p_long)

    assert calibrator is not None
    assert report["enabled"] is True
    raw = np.column_stack([p_long, 1.0 - p_long])
    calibrated = _apply_long_calibrator(calibrator, raw)
    assert float(calibrated[y == 0, 0].mean()) > float(raw[y == 0, 0].mean())
    assert float(calibrated[y == 1, 0].mean()) < float(raw[y == 1, 0].mean())


def test_isotonic_guard_rejects_when_future_half_worsens():
    first_half_y = np.array(([0, 1] * 20), dtype=np.int32)
    first_half_p = np.where(first_half_y == 0, 0.60, 0.40)
    second_half_y = np.array(([0, 1] * 20), dtype=np.int32)
    second_half_p = np.where(second_half_y == 0, 0.40, 0.60)
    y = np.concatenate([first_half_y, second_half_y]).astype(np.int32)
    p_long = np.concatenate([first_half_p, second_half_p]).astype(np.float32)

    calibrator, report = _fit_long_isotonic_calibrator(y, p_long)

    assert calibrator is None
    assert report["enabled"] is False
    assert report["reason"] == "calibration_guard_rejected"


def test_stable_ts_sort_preserves_duplicate_timestamp_order():
    ts = pd.to_datetime(
        [
            "2025-04-01 00:00:02",
            "2025-04-01 00:00:01",
            "2025-04-01 00:00:01",
            "2025-04-01 00:00:02",
        ]
    )
    df = pd.DataFrame({"ts_event": ts, "row_id": [0, 1, 2, 3]})

    sorted_df = _stable_sort_by_ts_event(df)

    assert sorted_df["row_id"].tolist() == [1, 2, 0, 3]


def test_oof_isotonic_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QS_ENABLE_OOF_ISOTONIC", raising=False)
    assert _oof_isotonic_enabled() is False
    monkeypatch.setenv("QS_ENABLE_OOF_ISOTONIC", "1")
    assert _oof_isotonic_enabled() is True
