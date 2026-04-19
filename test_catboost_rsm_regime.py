from __future__ import annotations

import pandas as pd
import pytest

from modules.catboost_5m_report import _build_price_hover_trace
from modules.range_state_machine import SignalState, apply_range_filter_to_dataframe, RangeStateMachine


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


def test_trending_mode_allows_high_confidence_edge_entry_early():
    rsm = RangeStateMachine(
        min_confirmations=3,
        confirmation_window=6,
        min_candles_between=0,
        cvd_window=2,
        obi_window=2,
        hawkes_window=4,
        trend_early_entry_confidence=0.78,
        trend_early_entry_range_pos=0.85,
        trend_early_entry_min_reversal=1.0,
    )

    warmup = [
        {"price": 1.2200, "raw_signal": "NEUTRAL", "confidence": 0.0, "hawkes": 0.01},
        {"price": 1.2204, "raw_signal": "NEUTRAL", "confidence": 0.0, "hawkes": 0.01},
        {"price": 1.2208, "raw_signal": "NEUTRAL", "confidence": 0.0, "hawkes": 0.01},
        {"price": 1.2212, "raw_signal": "NEUTRAL", "confidence": 0.0, "hawkes": 0.01},
    ]
    for row in warmup:
        rsm.process(
            regime="Trending",
            cvd=0.0,
            obi=0.0,
            absorption=0.0,
            **row,
        )

    decision = rsm.process(
        price=1.2230,
        raw_signal="SHORT",
        confidence=0.82,
        cvd=-0.05,
        obi=-0.08,
        hawkes=0.20,
        absorption=0.0,
        regime="Trending",
    )

    assert decision["action"] == "ENTER"
    assert decision["direction"] == "SHORT"
    assert decision["reason"] == "trend_confirmed_1confs_early"


def test_price_hover_trace_exposes_raw_signal_when_rsm_masks_it():
    bars = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2025-01-15 07:55:00"]),
            "open": [1.2224],
            "high": [1.2234],
            "low": [1.2219],
            "close": [1.2233],
            "event_count": [12],
            "volume": [42.0],
            "cb_direction": ["NEUTRAL"],
            "cb_direction_raw": ["SHORT"],
            "cb_confidence": [0.79217],
            "cb_prob_long": [0.20783],
            "cb_prob_short": [0.79217],
            "cvd_delta": [-0.116959],
            "obi": [-0.15711],
            "absorption_intensity": [0.515036],
            "kyle_lambda": [0.680569],
            "hawkes_intensity": [0.035122],
            "regime_label": ["Trending"],
            "rsm_action": ["HOLD"],
            "rsm_reason": ["trend_accumulating_1/2"],
        }
    )

    trace = _build_price_hover_trace(bars)

    assert trace.customdata[0][6] == "NEUTRAL"
    assert trace.customdata[0][16] == "SHORT"
    assert trace.customdata[0][18] == "trend_accumulating_1/2"


def test_locked_state_times_out_and_reprocesses_current_bar():
    rsm = RangeStateMachine(
        min_confirmations=3,
        confirmation_window=6,
        min_candles_between=0,
        cvd_window=2,
        obi_window=2,
        hawkes_window=4,
        max_lock_bars=3,
        trend_early_entry_confidence=0.75,
        trend_early_entry_range_pos=0.80,
        trend_early_entry_min_reversal=1.0,
    )

    for row in [
        {"price": 1.2208, "hawkes": 0.01},
        {"price": 1.2206, "hawkes": 0.01},
        {"price": 1.2204, "hawkes": 0.01},
        {"price": 1.2202, "hawkes": 0.01},
    ]:
        rsm.process(
            price=row["price"],
            raw_signal="NEUTRAL",
            confidence=0.0,
            cvd=0.0,
            obi=0.0,
            hawkes=row["hawkes"],
            absorption=0.0,
            regime="Trending",
        )

    rsm.signal_state = SignalState.LOCKED
    rsm._last_trade_direction = "SHORT"
    rsm._last_trade_price = 1.2208
    rsm._last_trade_candle = rsm.candle_idx - 3

    decision = rsm.process(
        price=1.2190,
        raw_signal="LONG",
        confidence=0.84,
        cvd=0.0,
        obi=0.0,
        hawkes=0.20,
        absorption=0.0,
        regime="Trending",
    )

    assert decision["action"] == "ENTER"
    assert decision["direction"] == "LONG"
