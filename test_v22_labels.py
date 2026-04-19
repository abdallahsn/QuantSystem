import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QUANTSYSTEM_SKIP_HEAVY_ML", "1")
os.environ.setdefault("QUANTSYSTEM_SKIP_GPU_DETECT", "1")

import numpy as np
import pandas as pd

from modules.catboost_5m_report import _build_dashboard, _compute_signal_stats
from modules.labels_v22 import (
    DIR_LONG,
    DIR_NEUTRAL,
    DIR_SHORT,
    _forward_scan_per_row,
    build_causal_event_labels,
)
from prepare_training_data import (
    run_refinery,
)
from train_v19 import build_event_training_view


ROOT = Path(__file__).resolve().parent
SAMPLE_MBO = ROOT / "sample_mbo.csv"
SAMPLE_MBP = ROOT / "sample_mbp10.csv"
RICH_MBO = ROOT / "rich_mbo.csv"
RICH_MBP = ROOT / "rich_mbp.csv"


def _capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def _run_sample_refinery(**kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        result, stdout = _capture_stdout(
            run_refinery,
            str(RICH_MBO),
            str(RICH_MBP),
            output_dir=tmpdir,
            chunksize=0,
            label_mode="v22",
            n_workers=1,
            target_bars=500,
            lob_event_sample=256,
            fit_aux_models=False,
            **kwargs,
        )
    return result, stdout


class V22LabelsRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QUANTSYSTEM_SKIP_HEAVY_ML", "1")

    def test_build_causal_event_labels_no_typeerror(self):
        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2026-01-01", periods=8, freq="s", tz="UTC"),
                "price": [1.2000, 1.2002, 1.2004, 1.2001, 1.2006, 1.2003, 1.2008, 1.2005],
                "size": [1, 2, 1, 3, 2, 1, 2, 1],
                "cvd": [0, 2, 3, 1, 4, 2, 5, 3],
            }
        )
        out = build_causal_event_labels(
            df,
            horizon=4,
            direction_threshold_ticks=1.0,
            tp_mult=1.2,
            sl_mult=1.0,
            tick_size=0.0001,
            trend_strength_min=0.05,
        )
        self.assertIn("bias_label", out.columns)
        self.assertIn("trend_strength", out.columns)
        self.assertIn("train_event_flag", out.columns)
        self.assertIn("event_score", out.columns)
        self.assertEqual(len(out), len(df))

    def test_build_causal_event_labels_caps_tp_at_liquidity_wall(self):
        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2026-01-01", periods=4, freq="s", tz="UTC"),
                "price": [100.00, 100.12, 100.50, 100.32],
                "size": [1, 1, 1, 1],
                "cvd": [0, 0, 0, 0],
                "micro_price": [100.00, 100.12, 100.50, 100.32],
                "bid_wall_strength": [0.0, 0.0, 0.0, 0.0],
                "ask_wall_strength": [0.9, 0.0, 0.0, 0.0],
                "distance_to_wall": [5.0, 0.0, 0.0, 0.0],
                "gap_size": [0.0, 0.0, 0.0, 0.0],
                "liquidity_density": [1.0, 1.0, 1.0, 1.0],
                "bid_wall_px": [np.nan, np.nan, np.nan, np.nan],
                "ask_wall_px": [100.50, np.nan, np.nan, np.nan],
            }
        )

        out = build_causal_event_labels(
            df,
            horizon=3,
            direction_threshold_ticks=40.0,
            tp_mult=2.0,
            sl_mult=1.0,
            tick_size=0.01,
            adaptive_horizon=False,
            trend_filter=False,
        )

        self.assertEqual(int(out.loc[0, "bias_label"]), DIR_LONG)
        self.assertEqual(int(out.loc[0, "signal_quality"]), 2)

    def test_forward_scan_ignores_weak_liquidity_wall_and_falls_back_to_atr(self):
        prices = np.array([100.00, 99.86, 99.70, 99.82], dtype=np.float64)
        dynamic_threshold = np.full(prices.shape[0], 0.40, dtype=np.float64)
        adaptive_horizons = np.full(prices.shape[0], 3, dtype=np.int32)

        bias_arr, quality_arr, end_idx_arr = _forward_scan_per_row(
            prices=prices,
            dynamic_threshold=dynamic_threshold,
            adaptive_horizons=adaptive_horizons,
            tick_size=0.01,
            tp_mult=2.0,
            sl_mult=1.0,
            bid_wall_px=np.array([99.70, np.nan, np.nan, np.nan], dtype=np.float64),
            ask_wall_px=np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64),
            bid_wall_strength=np.array([0.30, np.nan, np.nan, np.nan], dtype=np.float64),
            ask_wall_strength=np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64),
        )

        self.assertEqual(int(bias_arr[0]), DIR_NEUTRAL)
        self.assertEqual(int(quality_arr[0]), 1)
        self.assertEqual(int(end_idx_arr[0]), 0)

    def test_sample_step4_prints_bias_layers_and_aggressive_is_more_directional(self):
        legacy, legacy_stdout = _run_sample_refinery(
            label_horizon=150,
            event_roll_window=50,
            direction_threshold_ticks=2.0,
            tp_mult=1.5,
            sl_mult=1.0,
            kalman_slope_threshold=0.05,
            trend_strength_min=0.20,
        )
        aggressive, aggressive_stdout = _run_sample_refinery(
            label_horizon=150,
            event_roll_window=50,
            direction_threshold_ticks=1.0,
            tp_mult=1.2,
            sl_mult=1.0,
            kalman_slope_threshold=0.05,
            trend_strength_min=0.05,
        )

        aggressive_directional = int((aggressive["bias_label"] != 2).sum())
        aggressive_selected_directional = int(((aggressive["train_event_flag"] == 1) & (aggressive["bias_label"] != 2)).sum())

        self.assertIn("BiasAll", aggressive_stdout)
        self.assertIn("BiasEvt", aggressive_stdout)
        self.assertIn("BiasDir", aggressive_stdout)
        self.assertIn("BiasTrn", aggressive_stdout)
        self.assertIn("BiasSel", aggressive_stdout)
        self.assertIn("raw row-level causal labels", aggressive_stdout)
        self.assertGreater(aggressive_directional, 0)
        self.assertGreater(aggressive_selected_directional, 0)
        self.assertGreaterEqual(
            int((aggressive["bias_label"] == DIR_LONG).sum()) + int((aggressive["bias_label"] == DIR_SHORT).sum()),
            aggressive_directional,
        )
        self.assertLessEqual(
            int(aggressive["train_event_flag"].sum()),
            int(aggressive["event_flag"].sum()),
        )
        self.assertLess(
            float(aggressive["train_event_flag"].mean()),
            float(aggressive["event_flag"].mean()),
        )
        self.assertNotEqual(legacy_stdout, "")

    def test_event_training_view_prefers_train_event_gate(self):
        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2026-01-01", periods=4, freq="s"),
                "event_flag": [1, 1, 1, 1],
                "train_event_flag": [0, 1, 0, 1],
                "bias_label": [DIR_LONG, DIR_LONG, DIR_SHORT, DIR_SHORT],
                "signal_quality": [2, 2, 1, 2],
            }
        )
        event_df, info = build_event_training_view(df)
        self.assertEqual(info["event_col"], "train_event_flag")
        self.assertEqual(len(event_df), 2)
        self.assertAlmostEqual(info["raw_event_rate_full"], 1.0)
        self.assertAlmostEqual(info["event_rate_full"], 0.5)

    def test_run_refinery_cli_knobs_and_catboost_note(self):
        out_df, stdout = _run_sample_refinery(
            label_horizon=150,
            event_roll_window=50,
            direction_threshold_ticks=1.0,
            tp_mult=1.2,
            sl_mult=1.0,
            kalman_slope_threshold=0.05,
            trend_strength_min=0.05,
        )
        self.assertIn("BiasAll", stdout)
        self.assertIn("BiasEvt", stdout)
        self.assertIn("BiasTrn", stdout)
        self.assertTrue({"event_flag", "train_event_flag", "event_score", "bias_label", "signal_quality"}.issubset(out_df.columns))

        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["QUANTSYSTEM_SKIP_HEAVY_ML"] = "1"
            env["QUANTSYSTEM_SKIP_GPU_DETECT"] = "1"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "prepare_training_data.py"),
                    "--mbo", str(RICH_MBO),
                    "--mbp", str(RICH_MBP),
                    "--output", tmpdir,
                    "--chunksize", "0",
                    "--label_mode", "v22",
                    "--n_workers", "1",
                    "--label_horizon", "60",
                    "--event_roll_window", "25",
                    "--direction_threshold_ticks", "0.75",
                    "--tp_mult", "1.10",
                    "--sl_mult", "0.90",
                    "--kalman_slope_threshold", "0.07",
                    "--trend_strength_min", "0.03",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            cli_output = proc.stdout + proc.stderr
            self.assertIn("thr_ticks=0.75", cli_output)
            self.assertIn("tp_mult=1.10", cli_output)
            self.assertIn("sl_mult=0.90", cli_output)
            self.assertIn("kalman_thr=0.07", cli_output)
            self.assertIn("trend_min=0.03", cli_output)

        bars = pd.DataFrame(
            {
                "ts_event": pd.date_range("2026-01-01", periods=4, freq="5min"),
                "signal_time": pd.date_range("2026-01-01 00:05:00", periods=4, freq="5min"),
                "open": [1.0, 1.1, 1.2, 1.3],
                "high": [1.1, 1.2, 1.3, 1.4],
                "low": [0.9, 1.0, 1.1, 1.2],
                "close": [1.05, 1.15, 1.18, 1.28],
                "volume": [10, 11, 12, 13],
                "event_count": [1, 1, 1, 1],
                "cvd": [1, 2, 3, 4],
                "cvd_delta": [1, 1, 1, 1],
                "obi": [0.1, 0.2, 0.1, 0.3],
                "absorption_intensity": [0.0, 0.1, 0.2, 0.1],
                "kyle_lambda": [0.1, 0.2, 0.1, 0.3],
                "hawkes_intensity": [0.2, 0.1, 0.2, 0.3],
                "regime_label": ["Trending", "Trending", "Ranging", "Trending"],
                "cb_prob_long": [0.7, 0.2, 0.8, 0.4],
                "cb_prob_short": [0.3, 0.8, 0.2, 0.6],
                "cb_confidence": [0.7, 0.8, 0.8, 0.6],
                "cb_direction_idx": [0, 1, 0, 1],
                "cb_direction": ["LONG", "SHORT", "LONG", "SHORT"],
                "cb_change_flag": [1, 1, 1, 1],
            }
        )
        stats = _compute_signal_stats(bars, future_bars=1)
        fig = _build_dashboard(bars, stats)
        annotation_texts = [str(item.text) for item in fig.layout.annotations]
        self.assertTrue(
            any("not Step 4 raw causal labels" in text for text in annotation_texts),
            annotation_texts,
        )


if __name__ == "__main__":
    unittest.main()
