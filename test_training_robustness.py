import json

import numpy as np
import pandas as pd

from prepare_training_data import _build_refinery_split_context, _fit_regime_surface, _select_event_rich_lob_emit_positions
from modules.decision_policy_v19 import build_decision_policy, evaluate_decision_policy
from modules.feature_factory_v19 import apply_scaler_params_to_frame
from modules.regime_classifier import RegimeClassifier, _build_regime_features
from modules.meta_learner import MetaLearnerLSTM
from modules.slippage_model import fractional_kelly_bet_size
from train_v19 import (
    _assert_lob_event_alignment,
    _align_lob_to_rows,
    _adaptive_tree_early_stopping_rounds,
    _event_gate_train_prefix,
    _infer_event_gate_schema,
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
    assert profile["compact_due_to_small_data"] is True
    assert profile["compact_due_to_visual_coverage"] is True
    assert profile["profile_reason"] == "small_training_set+low_visual_coverage"


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


def test_lob_event_alignment_guard_passes_and_fails_with_clear_diagnostics():
    df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 00:00:01",
                    "2025-01-01 00:00:02",
                    "2025-01-01 00:00:03",
                ]
            )
        }
    )
    good_ts = pd.Series(df["ts_event"].copy())
    stats = _assert_lob_event_alignment(df, good_ts, tolerance="1s", min_overlap_ratio=0.90)
    assert stats["matched_ratio"] == 1.0

    bad_ts = pd.Series(pd.to_datetime(["2025-01-01 01:00:00", "2025-01-01 01:00:01"]))
    try:
        _assert_lob_event_alignment(df, bad_ts, tolerance="1s", min_overlap_ratio=0.90)
    except RuntimeError as exc:
        assert "LOB/event timestamp alignment too low" in str(exc)
        assert "matched_rows=0/3" in str(exc)
    else:
        raise AssertionError("Expected low-overlap LOB alignment to fail")


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


def test_build_event_training_view_excludes_neutral_rows_from_stage1_surface():
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2025-01-01", periods=5, freq="s"),
            "event_flag": [1, 1, 1, 1, 1],
            "train_event_flag": [1, 1, 1, 1, 1],
            "bias_label": [0, 1, 2, 0, 2],
            "signal_quality": [2, 1, 2, 1, 0],
            "event_score": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    event_df, info = build_event_training_view(df)

    assert sorted(event_df["bias_label"].unique().tolist()) == [0, 1]
    assert info["bias_counts"] == {"0": 2, "1": 1}


def test_build_event_training_view_fits_score_normalization_on_train_prefix_only():
    base = pd.DataFrame(
        {
            "ts_event": pd.date_range("2025-01-01", periods=4, freq="s"),
            "event_flag": [1, 1, 1, 1],
            "train_event_flag": [1, 1, 1, 1],
            "bias_label": [0, 1, 0, 1],
            "signal_quality": [2, 2, 2, 2],
            "event_score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    shifted = base.copy()
    shifted.loc[3, "event_score"] = 40.0

    base_view, base_info = build_event_training_view(base)
    shifted_view, shifted_info = build_event_training_view(shifted)

    np.testing.assert_allclose(
        base_view["conf_target"].iloc[:3].to_numpy(dtype=np.float32),
        shifted_view["conf_target"].iloc[:3].to_numpy(dtype=np.float32),
        atol=1e-7,
    )
    assert base_info["score_fit_rows"] == shifted_info["score_fit_rows"]
    assert base_info["score_fit_max"] == shifted_info["score_fit_max"]


def test_infer_event_gate_schema_uses_train_only_prefix_rows():
    base = pd.DataFrame(
        {
            "ts_event": pd.date_range("2025-01-01", periods=10, freq="s"),
            "train_event_flag": [1] * 10,
            "event_flag": [1] * 10,
            "event_score": [0.5] * 8 + [4.0, 5.0],
        }
    )
    shifted = base.copy()
    shifted.loc[8:, "event_score"] = [40.0, 50.0]

    base_train = _event_gate_train_prefix(base, train_frac=0.80, split_time=None)
    shifted_train = _event_gate_train_prefix(shifted, train_frac=0.80, split_time=None)
    base_cfg = _infer_event_gate_schema(base_train)
    shifted_cfg = _infer_event_gate_schema(shifted_train)

    assert len(base_train) == len(shifted_train)
    assert base_cfg["score_threshold"] == shifted_cfg["score_threshold"] == 0.5
    assert base_cfg["score_threshold_source"] == "train_only"
    assert base_cfg["rows_used"] == len(base_train)


def test_apply_scaler_params_can_skip_clipping_for_tree_models():
    frame = pd.DataFrame({"cvd": [0.0, 1000.0]})
    scaler_params = {"cvd": {"type": "robust", "median": 0.0, "iqr": 1.0}}

    clipped = apply_scaler_params_to_frame(frame, scaler_params)
    unclipped = apply_scaler_params_to_frame(frame, scaler_params, clip_range=None)

    assert float(clipped["cvd"].iloc[1]) == 10.0
    assert float(unclipped["cvd"].iloc[1]) == 1000.0


def test_adaptive_tree_early_stopping_rounds_shrinks_for_small_folds():
    assert _adaptive_tree_early_stopping_rounds(300, True) == 30
    assert _adaptive_tree_early_stopping_rounds(1500, True) == 50
    assert _adaptive_tree_early_stopping_rounds(5000, True) == 75
    assert _adaptive_tree_early_stopping_rounds(15000, True) == 100
    assert _adaptive_tree_early_stopping_rounds(300, False) is None


def test_decision_policy_abstains_when_expected_value_is_negative():
    df = pd.DataFrame(
        {
            'bias_label': [0, 1, 0, 1],
            'forward_return': [0.0001, -0.0001, 0.0001, -0.0001],
            'trend_strength': [0.6, 0.6, 0.6, 0.6],
            'correction_depth': [0.7, 0.7, 0.7, 0.7],
        }
    )
    probs = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
            [0.85, 0.15],
            [0.15, 0.85],
        ],
        dtype=np.float32,
    )
    regime_meta = np.tile(np.array([[0.7, 0.2, 0.05, 0.05]], dtype=np.float32), (len(df), 1))
    coverage = np.ones(len(df), dtype=bool)

    policy = build_decision_policy(
        df,
        probs,
        regime_meta,
        coverage,
        cost_config={'tick_size': 0.0001, 'round_trip_cost_pips': 5.0},
        min_support=1,
    )
    decision = evaluate_decision_policy(
        policy,
        direction_probs={'LONG': 0.90, 'SHORT': 0.10},
        regime_probs=regime_meta[0],
        structure_bucket='deep_pullback',
        uncertainty=0.05,
        runtime_penalty=1.0,
    )

    assert decision is not None
    assert decision['bias'] == 'NEUTRAL'
    assert decision['tradeable'] is False
    assert decision['expected_value_long_pips'] <= 0.0


def test_decision_policy_build_uses_model_probabilities_to_weight_side_stats():
    df = pd.DataFrame(
        {
            'bias_label': [0, 0, 1, 1],
            'forward_return': [0.0001, 0.0006, -0.0001, -0.0001],
            'trend_strength': [0.6, 0.6, 0.6, 0.6],
            'correction_depth': [0.7, 0.7, 0.7, 0.7],
        }
    )
    probs = np.array(
        [
            [0.10, 0.90],
            [0.95, 0.05],
            [0.20, 0.80],
            [0.20, 0.80],
        ],
        dtype=np.float32,
    )
    regime_meta = np.tile(np.array([[0.7, 0.2, 0.05, 0.05]], dtype=np.float32), (len(df), 1))
    coverage = np.ones(len(df), dtype=bool)

    policy = build_decision_policy(
        df,
        probs,
        regime_meta,
        coverage,
        cost_config={'tick_size': 0.0001, 'round_trip_cost_pips': 1.0},
        min_support=1,
    )

    long_stats = policy['global']['LONG']
    short_stats = policy['global']['SHORT']

    assert long_stats['avg_win_pips'] > 3.0
    assert long_stats['avg_win_pips'] > short_stats['avg_win_pips']
    assert long_stats['weighted_support'] > 0.0


def test_fractional_kelly_size_shrinks_with_uncertainty():
    low_uncertainty = fractional_kelly_bet_size(
        0.65,
        4.0,
        2.0,
        uncertainty=0.05,
        coverage_ratio=1.0,
        regime_entropy=0.05,
        runtime_penalty=1.0,
        max_size=5,
    )
    high_uncertainty = fractional_kelly_bet_size(
        0.65,
        4.0,
        2.0,
        uncertainty=0.85,
        coverage_ratio=1.0,
        regime_entropy=0.05,
        runtime_penalty=1.0,
        max_size=5,
    )

    assert low_uncertainty >= high_uncertainty


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
