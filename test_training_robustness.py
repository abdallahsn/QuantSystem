import json

import numpy as np
import pandas as pd

from prepare_training_data import _select_event_rich_lob_emit_positions
from train_v19 import _resolve_default_lob_paths, _resolve_meta_learner_profile
from walkforward_v19 import aggregate_fold_metrics


def test_meta_learner_profile_compacts_on_small_data():
    profile = _resolve_meta_learner_profile(train_sequences=1837, visual_seq_coverage=0.80)

    assert profile["name"] == "compact"
    assert profile["lstm_units_1"] == 48
    assert profile["lstm_units_2"] == 24
    assert profile["force_rebuild"] is True
    assert profile["visual_dropout_rate"] >= 0.35


def test_emit_positions_expand_neighbors_when_events_collapse_to_few_snapshots():
    df_labeled = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 00:00:31",
                    "2025-01-01 00:00:32",
                    "2025-01-01 00:00:33",
                    "2025-01-01 00:00:34",
                    "2025-01-01 00:00:35",
                    "2025-01-01 00:00:36",
                ]
            ),
            "event_flag": [1] * 6,
            "train_event_flag": [1] * 6,
            "bias_label": [0] * 6,
            "signal_quality": [2] * 6,
        }
    )
    lob_mbp_src = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:00:10",
                    "2025-01-01 00:00:20",
                    "2025-01-01 00:00:30",
                    "2025-01-01 00:00:40",
                    "2025-01-01 00:00:50",
                    "2025-01-01 00:01:00",
                ]
            )
        }
    )

    emit_positions, meta = _select_event_rich_lob_emit_positions(
        df_labeled,
        lob_mbp_src,
        max_events=6,
        max_positions=3,
    )

    assert meta["base_emit_positions"] == 1
    assert meta["emit_neighbor_radius"] == 2
    assert meta["emit_neighbor_positions_capped"] is True
    assert meta["selected_emit_positions"] == 3
    assert len(emit_positions) == 3


def test_resolve_default_lob_paths_uses_artifact_root(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps({"kind": "refinery"}), encoding="utf-8")
    np.save(artifact_dir / "lob_tensors.npy", np.zeros((2, 3, 4, 5), dtype=np.float32))
    np.save(artifact_dir / "lob_tensor_timestamps.npy", np.array([1, 2], dtype=np.int64))

    lob_path, lob_ts_path = _resolve_default_lob_paths(str(artifact_dir))

    assert lob_path == str(artifact_dir / "lob_tensors.npy")
    assert lob_ts_path == str(artifact_dir / "lob_tensor_timestamps.npy")


def test_aggregate_fold_metrics_exposes_profit_factor_and_trade_pnl():
    fold_reports = [
        {
            "backtest": {
                "directional_precision_macro": 0.41,
                "directional_recall_macro": 0.40,
                "directional_f1_macro": 0.39,
                "event_gate_rate": 0.22,
                "win_rate": 0.48,
                "profit_factor": 1.20,
                "trade_sharpe": 0.11,
                "avg_trade_pnl_dollars": 12.5,
                "max_drawdown_pct": 0.08,
                "trades": 18,
                "total_pnl_dollars": 225.0,
            }
        },
        {
            "backtest": {
                "directional_precision_macro": 0.45,
                "directional_recall_macro": 0.43,
                "directional_f1_macro": 0.44,
                "event_gate_rate": 0.26,
                "win_rate": 0.50,
                "profit_factor": 1.35,
                "trade_sharpe": 0.15,
                "avg_trade_pnl_dollars": 15.0,
                "max_drawdown_pct": 0.10,
                "trades": 20,
                "total_pnl_dollars": 300.0,
            }
        },
    ]

    out = aggregate_fold_metrics(fold_reports)

    assert out["mean_profit_factor"] == 1.275
    assert out["mean_trade_pnl_dollars"] == 13.75
    assert out["total_trades"] == 38
    assert out["total_pnl_dollars"] == 525.0
