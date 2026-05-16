from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest_v19 import _enforce_oos_backtest_guard


def _write_contracts(tmp_path, *, source_csv):
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir()
    models_dir.mkdir()
    csv_path = dataset_dir / "features.csv"
    csv_path.write_text("ts_event,bias_label\n2024-01-01T00:00:00Z,0\n", encoding="utf-8")

    dataset_manifest = {
        "kind": "refinery_v19",
        "extra": {
            "dataset_id": "dataset-1",
            "schema_version": "v19-event-binary",
        },
    }
    model_manifest = {
        "kind": "train_v19",
        "inputs": {"csv": str(source_csv)},
        "extra": {
            "source_contract": {
                "source_csv": str(source_csv),
                "dataset_id": "dataset-1",
                "split_time": "2024-01-02T00:00:00Z",
                "schema_version": "v19-event-binary",
            }
        },
    }
    (dataset_dir / "artifact_manifest.json").write_text(json.dumps(dataset_manifest), encoding="utf-8")
    (models_dir / "manifest.json").write_text(json.dumps(model_manifest), encoding="utf-8")
    return csv_path, models_dir


def test_oos_guard_rejects_same_training_dataset(tmp_path):
    csv_path, models_dir = _write_contracts(tmp_path, source_csv=tmp_path / "dataset" / "features.csv")
    df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2024-01-01T12:00:00Z"]),
            "bias_label": [0],
        }
    )

    with pytest.raises(ValueError, match="Refusing in-sample labeled backtest"):
        _enforce_oos_backtest_guard(df, csv_path=str(csv_path), models_dir=str(models_dir))


def test_oos_guard_accepts_post_split_holdout_window(tmp_path):
    csv_path, models_dir = _write_contracts(tmp_path, source_csv=tmp_path / "dataset" / "features.csv")
    df = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2024-01-02T00:05:00Z"]),
            "bias_label": [0],
        }
    )

    info = _enforce_oos_backtest_guard(df, csv_path=str(csv_path), models_dir=str(models_dir))

    assert info["allowed"] is True
    assert info["reason"] == "post_split_window_only"
