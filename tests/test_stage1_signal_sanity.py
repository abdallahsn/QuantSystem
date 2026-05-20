from __future__ import annotations

import numpy as np

import audit_v19_labels_features as audit
import stage1_signal_sanity as sanity


def _audit_frame(n: int = 240):
    import pandas as pd

    ts = pd.date_range("2025-04-01", periods=n, freq="s")
    y_long = np.array(([1, 0] * (n // 2)), dtype=np.int8)
    bias = np.where(y_long == 1, audit.DIR_LONG, audit.DIR_SHORT).astype(np.int8)
    return pd.DataFrame(
        {
            "ts_event": ts,
            "label_end_ts": ts + pd.Timedelta(seconds=1),
            "bias_label": bias,
            "event_flag": np.ones(n, dtype=np.int8),
            "train_event_flag": np.ones(n, dtype=np.int8),
            "signal_quality": np.full(n, 2, dtype=np.int8),
            "label_horizon_steps": np.ones(n, dtype=np.int32),
            "cvd": y_long.astype(float),
            "micro_atr": np.linspace(0.1, 1.0, n),
        }
    )


def test_stage1_signal_sanity_finds_simple_predictive_feature():
    df = audit.stable_sort_by_ts(_audit_frame(360))

    _, summary = sanity.run_sanity(
        df,
        views=[audit.VIEW_DIRECTIONAL_ALL],
        feature_sets={"cvd_only": ["cvd"], "noise": ["micro_atr"]},
        n_folds=3,
        test_size=0.20,
        embargo_pct=0.01,
        min_train_pct=0.20,
    )

    by_set = {row["feature_set"]: row for row in summary}
    assert by_set["cvd_only"]["macro_f1_mean"] > 0.95
    assert by_set["cvd_only"]["auc_edge_mean"] > 0.99


def test_stage1_signal_sanity_penalizes_flipped_feature():
    df = audit.stable_sort_by_ts(_audit_frame(360))
    half = len(df) // 2
    y_long = (df["bias_label"].to_numpy(dtype=np.int8) == audit.DIR_LONG).astype(float)
    df["flip_feature"] = y_long
    df.loc[df.index[half:], "flip_feature"] = 1.0 - df.loc[df.index[half:], "flip_feature"]

    _, summary = sanity.run_sanity(
        df,
        views=[audit.VIEW_DIRECTIONAL_ALL],
        feature_sets={"flip": ["flip_feature"]},
        n_folds=3,
        test_size=0.20,
        embargo_pct=0.01,
        min_train_pct=0.20,
    )

    assert summary[0]["macro_f1_mean"] < 0.50
