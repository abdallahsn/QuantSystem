import json

import pandas as pd
import pytest

import prepare_training_data as refinery


def test_mbo_trade_processing_preserves_contract_metadata():
    df = pd.DataFrame(
        [
            {
                "ts_event": "2025-04-01T00:00:00.000000001Z",
                "price": 1.2500,
                "size": 3,
                "action": "T",
                "side": "A",
                "order_id": 100,
                "symbol": "6BM5",
                "__emit": 1,
                "__chunk_id": 0,
            }
        ]
    )

    out = refinery._process_mbo_chunk(
        (
            df,
            {
                "tick_size": 0.0001,
                "min_price_move": 0.0001,
                "sweep_thresh": 0.05,
            },
        )
    )

    assert out["symbol"].tolist() == ["6BM5"]


def test_final_contract_check_fails_when_expected_symbol_is_missing():
    with pytest.raises(RuntimeError, match="metadata is missing"):
        refinery._assert_single_contract(
            symbol_counts={},
            context="final_labeled_artifact",
            expected_symbol="6BM5",
        )


def test_contract_report_does_not_pass_empty_final_symbol_counts(tmp_path):
    report_path = refinery._write_contract_consistency_report(
        output_dir=str(tmp_path),
        expected_symbol="6BM5",
        input_symbol_counts={"6BM5": 10},
        final_symbol_counts={},
    )

    report = json.loads((tmp_path / "contract_consistency_report.json").read_text())
    assert report_path.endswith("contract_consistency_report.json")
    assert report["passed"] is False
    assert report["failure_reason"] == "final_contract_metadata_missing"


def test_lob_emit_mode_train_events_uses_selected_directional_events_only():
    labeled = pd.DataFrame(
        {
            "ts_event": pd.date_range("2025-04-01", periods=4, freq="s"),
            "event_flag": [1, 1, 1, 0],
            "train_event_flag": [0, 1, 1, 0],
            "bias_label": [0, 1, refinery.DIR_NEUTRAL, 0],
            "signal_quality": [refinery.QUALITY_STRONG] * 4,
        }
    )
    mbp = pd.DataFrame({"ts_event": pd.date_range("2025-04-01", periods=4, freq="s")})

    positions, meta = refinery._select_event_rich_lob_emit_positions(
        labeled,
        mbp,
        emit_mode=refinery.LOB_EMIT_MODE_TRAIN_EVENTS,
    )

    assert positions.tolist() == [1]
    assert meta["emit_mode"] == refinery.LOB_EMIT_MODE_TRAIN_EVENTS
    assert meta["event_col"] == "train_event_flag"


def test_lob_emit_mode_directional_all_uses_all_long_short_labels():
    labeled = pd.DataFrame(
        {
            "ts_event": pd.date_range("2025-04-01", periods=4, freq="s"),
            "event_flag": [1, 1, 1, 0],
            "train_event_flag": [0, 1, 1, 0],
            "bias_label": [0, 1, refinery.DIR_NEUTRAL, 0],
            "signal_quality": [refinery.QUALITY_STRONG] * 4,
        }
    )
    mbp = pd.DataFrame({"ts_event": pd.date_range("2025-04-01", periods=4, freq="s")})

    positions, meta = refinery._select_event_rich_lob_emit_positions(
        labeled,
        mbp,
        emit_mode=refinery.LOB_EMIT_MODE_DIRECTIONAL_ALL,
    )

    assert positions.tolist() == [0, 1, 3]
    assert meta["emit_mode"] == refinery.LOB_EMIT_MODE_DIRECTIONAL_ALL
    assert meta["event_col"] == "bias_label"
