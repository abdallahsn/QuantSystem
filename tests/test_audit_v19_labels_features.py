from __future__ import annotations

import numpy as np
import pandas as pd

import audit_v19_labels_features as audit


def _audit_frame(n: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2025-04-01", periods=n, freq="s")
    y_long = np.array(([1, 0] * (n // 2)), dtype=np.int8)
    bias = np.where(y_long == 1, audit.DIR_LONG, audit.DIR_SHORT).astype(np.int8)
    train_event = np.zeros(n, dtype=np.int8)
    train_event[::2] = 1
    return pd.DataFrame(
        {
            "ts_event": ts,
            "label_end_ts": ts + pd.Timedelta(seconds=1),
            "bias_label": bias,
            "event_flag": np.ones(n, dtype=np.int8),
            "train_event_flag": train_event,
            "signal_quality": np.full(n, 2, dtype=np.int8),
            "label_horizon_steps": np.ones(n, dtype=np.int32),
            "effective_horizon": np.full(n, 5, dtype=np.int32),
            "event_score": np.linspace(0.0, 1.0, n),
            "cvd": y_long.astype(float),
            "obi": (1 - y_long).astype(float),
            "micro_atr": np.linspace(0.1, 1.0, n),
        }
    )


def test_label_report_counts_directional_and_event_views():
    df = _audit_frame()
    report = audit.build_label_report(df, [audit.VIEW_EVENT_BINARY, audit.VIEW_DIRECTIONAL_ALL])
    slices = {row["slice"]: row for row in report["slices"]}

    assert slices["directional_all"]["rows"] == len(df)
    assert slices[audit.VIEW_EVENT_BINARY]["rows"] == len(df) // 2
    assert report["invalid_label_end_before_ts"] == 0


def test_feature_fold_metrics_detect_predictive_feature():
    df = audit.stable_sort_by_ts(_audit_frame())
    fold_rows, summary_rows, split_reports = audit.compute_feature_fold_metrics(
        df,
        ["cvd", "obi", "micro_atr"],
        [audit.VIEW_DIRECTIONAL_ALL],
        n_folds=3,
        test_size=0.20,
        embargo_pct=0.01,
        min_train_pct=0.20,
    )
    by_feature = {row["feature"]: row for row in summary_rows if row["view"] == audit.VIEW_DIRECTIONAL_ALL}

    assert fold_rows
    assert split_reports[audit.VIEW_DIRECTIONAL_ALL]["n_splits"] >= 1
    assert by_feature["cvd"]["auc_edge_mean"] > 0.99
    assert by_feature["cvd"]["train_signed_auc_mean"] > 0.99
    assert by_feature["obi"]["auc_edge_mean"] > 0.99
    assert by_feature["obi"]["train_signed_auc_mean"] > 0.99
    assert by_feature["cvd"]["top_side"] == "long"
    assert by_feature["obi"]["top_side"] == "short"


def test_train_signed_auc_penalizes_direction_flip():
    df = _audit_frame()
    half = len(df) // 2
    y_long = (df["bias_label"].to_numpy(dtype=np.int8) == audit.DIR_LONG).astype(float)
    flipping_feature = y_long.copy()
    flipping_feature[half:] = 1.0 - flipping_feature[half:]
    df["flip_feature"] = flipping_feature

    _, summary_rows, _ = audit.compute_feature_fold_metrics(
        audit.stable_sort_by_ts(df),
        ["flip_feature"],
        [audit.VIEW_DIRECTIONAL_ALL],
        n_folds=3,
        test_size=0.20,
        embargo_pct=0.01,
        min_train_pct=0.20,
    )
    row = summary_rows[0]

    assert row["auc_edge_mean"] > 0.99
    assert row["train_signed_auc_mean"] < 0.50
