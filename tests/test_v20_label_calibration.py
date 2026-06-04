from argparse import Namespace
import json

import numpy as np
import pandas as pd

from label_calibration_v20 import run


def test_label_calibration_writes_grid_and_report(tmp_path):
    rows = 120
    ts = pd.date_range("2026-01-02 09:00:00", periods=rows, freq="s")
    mid = 1.2500 + np.sin(np.arange(rows) / 6.0) * 0.0005 + np.arange(rows) * 0.000002
    features = pd.DataFrame(
        {
            "ts_event": ts,
            "feature_ts": ts,
            "mid_price": mid,
            "spread": np.full(rows, 0.0001),
            "realized_vol": np.full(rows, 0.0001),
        }
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    features.to_parquet(artifact / "features.parquet")
    report = run(
        Namespace(
            artifact=str(artifact),
            output=str(artifact),
            tick_size=0.0001,
            horizons="10,20",
            tp_mults="0.5,1.0",
            sl_mults="0.5",
            neutral_mults="0.1,0.3",
            round_trip_cost_ticks=0.0,
            spread_cost_mult=0.0,
            min_barrier_ticks=0.5,
            chunk_rows=50,
            max_rows=0,
        )
    )

    assert report["grid"]["combinations"] == 8
    assert report["leakage_precheck_all_passed"]
    assert (artifact / "label_calibration_report.json").exists()
    assert (artifact / "label_calibration_grid.csv").exists()
    with open(artifact / "label_calibration_report.json", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["recommended_config"]["horizon"] in {10, 20}
    grid = pd.read_csv(artifact / "label_calibration_grid.csv")
    assert {"directional_share", "event_share", "long_short_imbalance", "leakage_precheck_passed"}.issubset(grid.columns)
