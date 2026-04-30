import json

import numpy as np
import pandas as pd

from prepare_training_data import _build_refinery_split_context, _fit_regime_surface, _select_event_rich_lob_emit_positions
from modules.regime_classifier import RegimeClassifier, _build_regime_features
from modules.meta_learner import MetaLearnerLSTM
from train_v19 import (
    _align_lob_to_rows,
    _resolve_default_lob_paths,
    _resolve_meta_learner_profile,
    build_event_training_view,
)
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


def test_align_lob_to_rows_prefers_directional_targets_over_raw_obi():
    df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 00:00:01",
                    "2025-01-01 00:00:02",
                    "2025-01-01 00:00:03",
                ]
            ),
            "bias_label": [0, 1, 0],
            "signal_quality": [2, 1, 2],
            "train_event_flag": [1, 1, 0],
            "event_flag": [1, 1, 1],
            "obi": [-9.0, 9.0, -9.0],
        }
    )
    lob_timestamps = pd.Series(df["ts_event"].copy())

    row_to_tensor, tensor_targets, tensor_seen = _align_lob_to_rows(df, lob_timestamps, max_age="1s")

    assert row_to_tensor.tolist() == [0, 1, 2]
    assert tensor_seen.tolist() == [True, True, True]
    assert np.isclose(tensor_targets[0], 1.0)
    assert np.isclose(tensor_targets[1], -0.6)
    assert np.isclose(tensor_targets[2], 0.5)


def test_meta_threshold_calibration_can_shift_off_argmax_default():
    long_probs = np.array([0.70, 0.60, 0.55, 0.52, 0.48, 0.45], dtype=np.float32)
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    threshold, metrics = MetaLearnerLSTM.choose_bias_long_threshold(long_probs, y_true)

    assert threshold > 0.50
    assert metrics["macro_f1"] >= 0.99
    preds = MetaLearnerLSTM._labels_from_long_probs(long_probs, threshold)
    assert preds.tolist() == y_true.tolist()


def test_build_event_training_view_uses_continuous_conf_target():
    df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 00:00:01",
                    "2025-01-01 00:00:02",
                    "2025-01-01 00:00:03",
                    "2025-01-01 00:00:04",
                ]
            ),
            "event_flag": [1, 1, 1, 1],
            "train_event_flag": [1, 1, 1, 1],
            "bias_label": [0, 1, 0, 1],
            "signal_quality": [2, 1, 2, 1],
            "event_score": [0.5, 0.8, 1.0, 1.4],
        }
    )

    event_df, info = build_event_training_view(df)

    assert "normalized_event_score" in event_df.columns
    assert event_df["conf_target"].between(0.0, 1.0).all()
    assert event_df["conf_target"].nunique() > 2
    assert info["conf_target_std"] > 0.0


def test_regime_features_build_cvd_persistence_from_constant_cvd_delta():
    df = pd.DataFrame(
        {
            "price": np.linspace(100.0, 102.0, 80),
            "size": np.full(80, 10.0),
            "cvd_delta": np.ones(80, dtype=np.float64),
            "inter_event_time": np.full(80, 0.1),
            "obi": np.linspace(0.1, 0.9, 80),
            "raw__micro_atr": np.linspace(0.01, 0.05, 80),
        }
    )

    features = _build_regime_features(df)

    assert float(features["cvd_persistence"].iloc[-1]) > 0.0
    assert float(features["volatility"].iloc[-1]) > 0.0


def test_regime_rules_prioritize_trending_over_volatile_overlap():
    clf = RegimeClassifier(model_type="rules")
    X = pd.DataFrame(
        {
            "volatility": [0.20, 0.95, 0.12, 0.18, 0.08, 0.05],
            "activity": [0.75, 1.00, 0.25, 0.30, 0.12, 0.10],
            "volume_ratio": [0.80, 1.00, 0.28, 0.35, 0.14, 0.10],
            "cvd_impulse": [0.35, 0.90, 0.08, 0.12, 0.03, 0.02],
            "cvd_persistence": [0.70, 0.98, 0.10, 0.15, 0.04, 0.03],
            "trend_efficiency": [0.72, 0.99, 0.12, 0.18, 0.05, 0.04],
            "imbalance": [0.55, 0.85, 0.12, 0.18, 0.04, 0.02],
        }
    )

    clf._fit_rules(X)
    labels = clf._predict_rules(X)

    assert int(labels[1]) == 0
    assert clf.last_rule_diagnostics["trend_volatile_overlap_count"] >= 1


def test_fit_regime_surface_rules_uses_full_train_and_full_prediction(tmp_path):
    n = 240
    df = pd.DataFrame(
        {
            "price": np.linspace(100.0, 110.0, n),
            "size": np.linspace(5.0, 25.0, n),
            "cvd": np.cumsum(np.where(np.arange(n) % 3 == 0, 2.0, 1.0)),
            "inter_event_time": np.where(np.arange(n) % 5 == 0, 0.05, 0.20),
            "obi": np.sin(np.linspace(0.0, 5.0, n)),
            "raw__micro_atr": np.linspace(0.01, 0.08, n),
            "ts_event": pd.date_range("2025-01-01", periods=n, freq="s"),
            "label_end_ts": pd.date_range("2025-01-01 00:00:01", periods=n, freq="s"),
        }
    )
    split_ctx = _build_refinery_split_context(df, train_frac=0.80)

    labels, info = _fit_regime_surface(
        df,
        train_idx=split_ctx["train_idx"],
        split_ctx=split_ctx,
        output_dir=str(tmp_path),
        regime_mode="rules",
        regime_stride=50,
    )

    assert len(labels) == n
    assert info["effective_stride"] == 1
    assert info["train_sample_rows"] == info["train_rows"]
    assert info["full_sample_rows"] == info["full_rows"]
