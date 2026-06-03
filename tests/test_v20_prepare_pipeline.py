from argparse import Namespace
import json

import pandas as pd

from modules.feature_artifact_v19 import load_feature_artifact
from prepare_v20 import run


def _write_synthetic_feeds(tmp_path, rows: int = 80):
    ts = pd.date_range("2026-01-01 09:00:00", periods=rows, freq="s")
    mid = 1.2500 + (pd.Series(range(rows)) // 8).to_numpy() * 0.0001
    mbp = {
        "ts_event": ts,
        "ts_recv": ts + pd.to_timedelta(1, unit="ms"),
        "symbol": ["6B"] * rows,
        "contract_symbol": ["6BH6"] * rows,
    }
    for level in range(10):
        distance = (level + 0.5) * 0.0001
        mbp[f"bid_px_{level:02d}"] = mid - distance
        mbp[f"ask_px_{level:02d}"] = mid + distance
        mbp[f"bid_sz_{level:02d}"] = 10 + level
        mbp[f"ask_sz_{level:02d}"] = 9 + level
    mbp_path = tmp_path / "mbp.csv"
    pd.DataFrame(mbp).to_csv(mbp_path, index=False)

    mbo = pd.DataFrame(
        {
            "ts_event": ts[::2],
            "ts_recv": ts[::2] + pd.to_timedelta(1, unit="ms"),
            "symbol": ["6B"] * len(ts[::2]),
            "contract_symbol": ["6BH6"] * len(ts[::2]),
            "action": ["T"] * len(ts[::2]),
            "side": ["A" if i % 3 else "B" for i in range(len(ts[::2]))],
            "price": mid[::2],
            "size": [2 + (i % 4) for i in range(len(ts[::2]))],
        }
    )
    mbo_path = tmp_path / "mbo.csv"
    mbo.to_csv(mbo_path, index=False)
    return mbo_path, mbp_path


def _args(tmp_path, mbo_path, mbp_path, **overrides):
    values = {
        "mbo": str(mbo_path),
        "mbp": str(mbp_path),
        "output": str(tmp_path / "artifact"),
        "symbol": "6B",
        "start": None,
        "end": None,
        "tick_size": 0.0001,
        "horizon": 10,
        "tp_mult": 1.0,
        "sl_mult": 1.0,
        "neutral_mult": 0.5,
        "round_trip_cost_ticks": 0.0,
        "spread_cost_mult": 0.0,
        "chunk_rows": 25,
        "sample_rows": 0,
        "rows_per_shard": 50,
        "dry_run": False,
        "validation_only": False,
        "strict": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_prepare_v20_writes_trainable_artifact(tmp_path):
    mbo_path, mbp_path = _write_synthetic_feeds(tmp_path)
    summary = run(_args(tmp_path, mbo_path, mbp_path))

    assert summary["train_v19_compatible"]
    out = tmp_path / "artifact"
    for name in (
        "features.parquet",
        "labels.parquet",
        "metadata.parquet",
        "manifest.json",
        "artifact_manifest.json",
        "data_validation_report.json",
        "feature_validation_report.json",
        "label_distribution_report.json",
        "leakage_precheck_report.json",
        "train_v19_compatibility_report.json",
    ):
        assert (out / name).exists(), name

    loaded = load_feature_artifact(str(out))
    assert len(loaded) == 80
    assert {"ts_event", "label_end_ts", "bias_label", "train_event_flag", "mlofi_sum", "raw__cvd"}.issubset(loaded.columns)
    assert pd.to_datetime(loaded["label_end_ts"]).ge(pd.to_datetime(loaded["ts_event"])).all()
    matched = loaded.loc[pd.to_datetime(loaded["mbo_state_ts"], errors="coerce").notna()]
    assert pd.to_datetime(matched["mbo_state_ts"]).le(pd.to_datetime(matched["ts_event"])).all()


def test_prepare_v20_dry_run_reports_without_final_artifact(tmp_path):
    mbo_path, mbp_path = _write_synthetic_feeds(tmp_path, rows=20)
    summary = run(_args(tmp_path, mbo_path, mbp_path, dry_run=True, output=str(tmp_path / "dry")))

    out = tmp_path / "dry"
    assert summary["dry_run"]
    assert (out / "data_validation_report.json").exists()
    assert not (out / "features.parquet").exists()
    assert not (out / "final").exists()


def test_prepare_v20_dry_run_handles_empty_symbol_filter(tmp_path):
    mbo_path, mbp_path = _write_synthetic_feeds(tmp_path, rows=20)
    output = tmp_path / "empty_symbol"
    summary = run(_args(tmp_path, mbo_path, mbp_path, dry_run=True, output=str(output), symbol="6BM6"))

    assert summary["dry_run"]
    assert not summary["passed"]
    with open(output / "data_validation_report.json", encoding="utf-8") as f:
        report = json.load(f)
    assert report["row_counts"]["mbp_rows_after_filters"] == 0
    assert "empty_mbp_after_filters" in report["mbp_quality"]["warnings"]
    assert report["mbp"]["pre_filter_value_counts"]["symbol"]["6B"] == 20
    assert not (output / "features.parquet").exists()
