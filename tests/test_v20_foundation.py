import numpy as np
import pandas as pd

from backtesting.execution_model import estimate_fill_probability
from feature_engineering.lob_features import compute_lob_features
from feature_engineering.ofi import compute_mlofi
from labeling.triple_barrier import DIR_LONG, DIR_NEUTRAL, TripleBarrierConfig, build_triple_barrier_labels
from training.splits import PurgedWalkForwardConfig, build_purged_walkforward_splits
from validation.artifact_schema import validate_artifact_schema


def _mbp_frame():
    return pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-01-01", periods=4, freq="s"),
            "bid_px_00": [1.2500, 1.2500, 1.2501, 1.2501],
            "ask_px_00": [1.2501, 1.2501, 1.2502, 1.2502],
            "bid_sz_00": [10, 12, 8, 10],
            "ask_sz_00": [11, 9, 9, 7],
        }
    )


def test_lob_features_are_current_snapshot_only():
    features = compute_lob_features(_mbp_frame(), levels=1)
    assert "mid_price" in features.columns
    assert np.isclose(float(features["spread"].iloc[0]), 0.0001, atol=1e-7)
    assert float(features["order_book_imbalance"].iloc[1]) > 0.0


def test_mlofi_uses_previous_snapshot_not_future_rows():
    features = compute_mlofi(_mbp_frame(), levels=1)
    assert "mlofi_00" in features.columns
    assert np.isclose(float(features["mlofi_00"].iloc[1]), 4.0)
    assert float(features["mlofi_00"].iloc[2]) != float(features["mlofi_00"].iloc[3])


def test_triple_barrier_labels_have_label_end_ts():
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-01-01", periods=6, freq="s"),
            "mid_price": [1.2500, 1.2501, 1.2503, 1.2504, 1.2502, 1.2501],
            "spread": [0.0001] * 6,
            "realized_vol": [0.0001] * 6,
        }
    )
    labeled = build_triple_barrier_labels(
        df,
        TripleBarrierConfig(horizon_rows=3, tp_vol_mult=1.0, sl_vol_mult=1.0, round_trip_cost_ticks=0.0),
    )
    assert "label_end_ts" in labeled.columns
    assert labeled["label_end_ts"].ge(labeled["ts_event"]).all()
    assert int(labeled["bias_label"].iloc[0]) == DIR_LONG
    assert int(labeled["bias_label"].iloc[-1]) == DIR_NEUTRAL


def test_purged_walkforward_removes_label_overlap():
    ts = pd.Series(pd.date_range("2026-01-01", periods=200, freq="s"))
    label_end = ts + pd.to_timedelta(5, unit="s")
    splits, report = build_purged_walkforward_splits(
        ts,
        label_end,
        PurgedWalkForwardConfig(n_folds=2, initial_train_frac=0.5, test_frac=0.2, min_train_rows=20, min_test_rows=10),
    )
    assert report["passed"]
    for train_idx, test_idx in splits:
        assert label_end.iloc[train_idx].max() < ts.iloc[test_idx].min()


def test_artifact_schema_rejects_future_raw_features():
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-01-01", periods=2, freq="s"),
            "label_end_ts": pd.date_range("2026-01-01 00:00:01", periods=2, freq="s"),
            "bias_label": [2, 0],
            "raw__forward_return": [0.0, 0.1],
        }
    )
    report = validate_artifact_schema(df)
    assert not report.passed
    assert "raw__forward_return" in report.forbidden_feature_columns


def test_fill_probability_is_bounded_and_latency_sensitive():
    fast = estimate_fill_probability(queue_ahead=5, trade_through_size=10, visible_depth=10, latency_rows=0)
    slow = estimate_fill_probability(queue_ahead=5, trade_through_size=10, visible_depth=10, latency_rows=10)
    assert 0.0 <= slow <= fast <= 1.0
