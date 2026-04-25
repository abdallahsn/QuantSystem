import numpy as np
import pytest

from modules.dynamic_target import DynamicTargetManager, TradeState


def _base_levels() -> dict:
    return {
        "long_wall_size": 1200.0,
        "short_wall_size": 1200.0,
    }


def _base_context() -> dict:
    return {
        "tick_size": 0.0001,
        "remaining_fuel": 0.0200,
        "adr_pips": 80.0,
    }


def _long_scan() -> dict:
    return {
        "dist_to_bid_wall": 0.0010,
        "bid_wall_strength": 0.8,
        "bid_wall_size_raw": 1200.0,
        "dist_to_ask_wall": 0.0040,
        "ask_wall_strength": 1.2,
        "ask_wall_size_raw": 1100.0,
    }


def test_open_trade_aims_before_opposing_wall_when_available():
    mgr = DynamicTargetManager()
    trade = mgr.open_trade(
        {"bias": "LONG", "price": 1.2000, "cvd_delta": 100.0},
        _base_levels(),
        _long_scan(),
        _base_context(),
        total_size=4,
    )

    assert trade.sl < trade.entry_price
    assert trade.tp1 > trade.entry_price
    assert trade.tp1 < trade.entry_price + _long_scan()["dist_to_ask_wall"]


def test_raw_visual_embedding_does_not_override_execution_logic():
    base_signal = {"bias": "LONG", "price": 1.2000, "cvd_delta": 100.0}

    mgr_plain = DynamicTargetManager()
    trade_plain = mgr_plain.open_trade(
        base_signal,
        _base_levels(),
        _long_scan(),
        _base_context(),
        total_size=4,
    )

    mgr_raw_emb = DynamicTargetManager()
    trade_raw_emb = mgr_raw_emb.open_trade(
        {
            **base_signal,
            "visual_embedding": np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32),
        },
        _base_levels(),
        _long_scan(),
        _base_context(),
        total_size=4,
    )

    assert trade_raw_emb.sl == pytest.approx(trade_plain.sl)
    assert trade_raw_emb.tp1 == pytest.approx(trade_plain.tp1)


def test_execution_hints_absorption_tightens_stop_loss():
    base_signal = {"bias": "LONG", "price": 1.2000, "cvd_delta": 100.0}

    mgr_plain = DynamicTargetManager()
    trade_plain = mgr_plain.open_trade(
        base_signal,
        _base_levels(),
        _long_scan(),
        _base_context(),
        total_size=4,
    )

    mgr_hinted = DynamicTargetManager()
    trade_hinted = mgr_hinted.open_trade(
        {
            **base_signal,
            "execution_hints": {"absorption": 1.0},
        },
        _base_levels(),
        _long_scan(),
        _base_context(),
        total_size=4,
    )

    assert abs(trade_hinted.entry_price - trade_hinted.sl) < abs(trade_plain.entry_price - trade_plain.sl)


def test_close_all_tp1_counts_full_position_pnl():
    mgr = DynamicTargetManager()
    ctx = _base_context()
    trade = mgr.open_trade(
        {"bias": "LONG", "price": 1.2000, "cvd_delta": 100.0},
        _base_levels(),
        _long_scan(),
        ctx,
        total_size=4,
    )

    result = mgr.update(
        current_price=trade.tp1,
        current_cvd_delta=trade.entry_cvd,
        current_wall_size=trade.wall_size_orig,
        vwap_zscore=0.0,
        regime_tradeable=False,
        tick_size=ctx["tick_size"],
    )

    pips_tp1 = abs(trade.tp1 - trade.entry_price) / ctx["tick_size"]
    assert result["action"] == "close_all_tp1"
    assert trade.state == TradeState.CLOSED
    assert trade.realized_pips == pytest.approx(pips_tp1 * trade.total_size)

