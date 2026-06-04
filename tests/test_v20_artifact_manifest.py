from argparse import Namespace
import json

import pandas as pd

from prepare_v20 import run


def test_v20_manifest_contains_required_phase2_contract(tmp_path):
    rows = 30
    ts = pd.date_range("2026-01-02 10:00:00", periods=rows, freq="s")
    mid = 1.2600 + (pd.Series(range(rows)) // 6).to_numpy() * 0.0001
    mbp = {
        "ts_event": ts,
        "ts_recv": ts,
        "symbol": ["6B"] * rows,
        "contract_symbol": ["6BH6"] * rows,
    }
    for level in range(10):
        mbp[f"bid_px_{level:02d}"] = mid - (level + 0.5) * 0.0001
        mbp[f"ask_px_{level:02d}"] = mid + (level + 0.5) * 0.0001
        mbp[f"bid_sz_{level:02d}"] = 12 + level
        mbp[f"ask_sz_{level:02d}"] = 10 + level
    mbp_path = tmp_path / "mbp.csv"
    pd.DataFrame(mbp).to_csv(mbp_path, index=False)

    mbo_path = tmp_path / "mbo.csv"
    pd.DataFrame(
        {
            "ts_event": ts,
            "ts_recv": ts,
            "symbol": ["6B"] * rows,
            "contract_symbol": ["6BH6"] * rows,
            "action": ["T"] * rows,
            "side": ["A"] * rows,
            "price": mid,
            "size": [1] * rows,
        }
    ).to_csv(mbo_path, index=False)

    output = tmp_path / "artifact"
    run(
        Namespace(
            mbo=str(mbo_path),
            mbp=str(mbp_path),
            output=str(output),
            symbol="6B",
            start=None,
            end=None,
            tick_size=0.0001,
            horizon=5,
            tp_mult=1.0,
            sl_mult=1.0,
            neutral_mult=0.5,
            round_trip_cost_ticks=0.0,
            spread_cost_mult=0.0,
            chunk_rows=20,
            sample_rows=0,
            max_rows=0,
            max_memory_gb=0.0,
            rows_per_shard=100,
            write_partitions=False,
            dry_run=False,
            validation_only=False,
            strict=False,
        )
    )

    with open(output / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "v20.0"
    assert manifest["inputs"]["mbo"] == str(mbo_path)
    assert manifest["inputs"]["mbp"] == str(mbp_path)
    assert manifest["tick_size"] == 0.0001
    assert manifest["horizon"] == 5
    assert manifest["label_end_ts_column"] == "label_end_ts"
    assert "feature_columns" in manifest
    assert "absorption_intensity" not in manifest["feature_columns"]
    assert "raw__absorption_intensity" not in manifest["feature_columns"]
    assert "absorption_intensity" in manifest["compatibility_only_feature_columns"]
    assert "raw__absorption_intensity" in manifest["compatibility_only_feature_columns"]
    assert manifest["extra"]["placeholder_features_excluded_from_training"]
    assert "label_columns" in manifest
    assert "metadata_columns" in manifest
    assert "final_feature_shards" in manifest["extra"]
    assert manifest["reports"]["train_v19_compatibility_report"] == "train_v19_compatibility_report.json"
