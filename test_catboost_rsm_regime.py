from __future__ import annotations

import pandas as pd
import pytest

from modules.range_state_machine import apply_range_filter_to_dataframe, RangeStateMachine


def test_apply_range_filter_falls_back_to_regime_label(monkeypatch):
    seen_regimes: list[str] = []

    def fake_process(self, **kwargs):
        seen_regimes.append(kwargs["regime"])
        return {
            "action": "HOLD",
            "direction": None,
            "reason": "test",
            "market_state": "TEST",
            "confirmations": 0,
        }

    monkeypatch.setattr(RangeStateMachine, "process", fake_process)

    df = pd.DataFrame(
        {
            "close": [1.0, 1.1],
            "cb_direction": ["SHORT", "SHORT"],
            "cb_confidence": [0.8, 0.9],
            "regime_label": ["Trending", "Ranging"],
        }
    )

    apply_range_filter_to_dataframe(df, regime_col="regime")

    assert seen_regimes == ["Trending", "Ranging"]


def test_generate_report_prefers_regime_label(monkeypatch, tmp_path):
    plotly = pytest.importorskip("plotly")
    assert plotly is not None
    from modules import catboost_5m_report

    bars = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2025-01-15 00:00:00"]),
            "signal_time": pd.to_datetime(["2025-01-15 00:05:00"]),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [1.0],
            "event_count": [1],
            "cvd": [0.0],
            "cvd_delta": [0.0],
            "obi": [0.0],
            "absorption_intensity": [0.0],
            "kyle_lambda": [0.0],
            "hawkes_intensity": [0.0],
            "regime_label": ["Trending"],
            "cb_prob_long": [0.2],
            "cb_prob_short": [0.8],
            "cb_direction_idx": [1],
            "cb_direction": ["SHORT"],
            "cb_confidence": [0.8],
            "cb_change_flag": [1],
        }
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(catboost_5m_report, "RSM_AVAILABLE", True)
    monkeypatch.setattr(catboost_5m_report, "predict_catboost_frame", lambda *args, **kwargs: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(catboost_5m_report, "_resample_catboost_bars", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr(catboost_5m_report, "_load_optional_market_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(catboost_5m_report, "_compute_signal_stats", lambda *args, **kwargs: {"total": 1, "n_long": 0, "n_short": 1, "pct_long": 0.0, "pct_short": 100.0, "long_hit_rate": 0.0, "short_hit_rate": 100.0, "note": "test"})
    monkeypatch.setattr(catboost_5m_report, "_build_turns_table", lambda bars: pd.DataFrame())
    monkeypatch.setattr(catboost_5m_report, "_build_dashboard", lambda *args, **kwargs: type("Fig", (), {"write_html": lambda self, *a, **k: None, "to_html": lambda self, *a, **k: "<div></div>"})())
    monkeypatch.setattr(catboost_5m_report, "_build_confusion_chart", lambda *args, **kwargs: type("Fig", (), {"to_html": lambda self, *a, **k: "<div></div>"})())

    def fake_apply(df, **kwargs):
        captured["regime_col"] = kwargs["regime_col"]
        out = df.copy()
        out["rsm_action"] = ["ENTER"]
        out["rsm_direction"] = ["SHORT"]
        out["rsm_reason"] = ["test"]
        out["rsm_state"] = ["TRENDING"]
        out["rsm_confs"] = [2]
        return out

    monkeypatch.setattr(catboost_5m_report, "apply_range_filter_to_dataframe", fake_apply)

    summary = catboost_5m_report.generate_catboost_5m_report(
        csv_path="unused.csv",
        models_dir=str(tmp_path),
        output_dir=str(tmp_path),
        report_name="report",
    )

    assert captured["regime_col"] == "regime_label"
    assert summary["direction_counts"] == {"SHORT": 1}
