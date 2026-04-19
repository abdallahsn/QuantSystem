import os
import tempfile
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QUANTSYSTEM_SKIP_HEAVY_ML", "1")
os.environ.setdefault("QUANTSYSTEM_SKIP_GPU_DETECT", "1")

from backtest_v19 import _load_meta_features, _simulate_trade_path
from prepare_training_data import _require_causal_label_runtime
from modules.dynamic_labels import EventGate
from modules.labels_v22 import _compute_adaptive_horizons
from modules.regime_classifier import RegimeClassifier
from train_v19 import _project_sequence_aux_context, _raw_stat_frame, build_inference_scaler_params


class LeakageGuardTests(unittest.TestCase):
    def test_adaptive_horizons_are_causal(self):
        atr = np.full(64, 2.0, dtype=np.float64)
        future_spike = atr.copy()
        future_spike[40:] = 200.0

        base = _compute_adaptive_horizons(atr, base_horizon=100)
        spiked = _compute_adaptive_horizons(future_spike, base_horizon=100)

        np.testing.assert_array_equal(base[:40], spiked[:40])

    def test_raw_stat_frame_rejects_forward_return(self):
        df = pd.DataFrame(
            {
                "forward_return": [0.1, -0.2, 0.05],
                "raw__forward_return": [0.1, -0.2, 0.05],
            }
        )
        with self.assertRaises(ValueError):
            _raw_stat_frame(df, ["forward_return"])
        with self.assertRaises(ValueError):
            _raw_stat_frame(df, ["raw__forward_return"])

    def test_backtest_rejects_live_meta_on_labeled_df(self):
        df = pd.DataFrame(
            {
                "bias_label": [0, 1, 2],
                "forward_return": [0.1, -0.1, 0.0],
            }
        )
        meta = np.zeros((len(df), 6), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta_features_live_v19.npy")
            np.save(path, meta)
            with self.assertRaises(ValueError):
                _load_meta_features(
                    df,
                    explicit_path=path,
                    expected_dim=6,
                    allow_in_sample_live_override=False,
                )

    def test_sequence_aux_context_uses_last_step_only(self):
        window = np.arange(30, dtype=np.float32).reshape(5, 6)
        out = _project_sequence_aux_context(
            window,
            n_stat_feat=3,
            sequence_aux_mode="last_step_only",
        )
        self.assertTrue(np.allclose(out[:-1, 3:], 0.0))
        np.testing.assert_allclose(out[:, :3], window[:, :3])
        np.testing.assert_allclose(out[-1, 3:], window[-1, 3:])

    def test_backtest_replays_path_instead_of_terminal_oracle(self):
        prices = np.array([100.0, 99.0, 103.0], dtype=np.float64)
        horizons = np.array([2, 1, 0], dtype=np.int32)
        micro_atr = np.array([2.0, 0.0, 0.0], dtype=np.float64)

        trade = _simulate_trade_path(
            entry_idx=0,
            direction='LONG',
            prices=prices,
            horizons=horizons,
            micro_atr=micro_atr,
            tick_size=1.0,
            direction_threshold_ticks=1.0,
            tp_mult=1.2,
            sl_mult=1.0,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade['exit_reason'], 'sl')
        self.assertAlmostEqual(trade['raw_pnl_pips'], -1.0)

    def test_inference_scaler_can_reuse_refinery_split_time(self):
        n = 120
        ts = pd.date_range('2026-01-01', periods=n, freq='min')
        df = pd.DataFrame(
            {
                'ts_event': ts,
                'label_end_ts': ts + pd.Timedelta(seconds=1),
                'raw__cvd': np.concatenate([
                    np.linspace(1.0, 60.0, 60, dtype=np.float32),
                    np.full(40, 100.0, dtype=np.float32),
                    np.full(20, 200.0, dtype=np.float32),
                ]),
            }
        )

        params, info = build_inference_scaler_params(
            df,
            ['cvd'],
            train_frac=0.80,
            split_time=ts[60],
        )

        self.assertEqual(info['split_source'], 'explicit_time')
        self.assertEqual(info['scaler_train_rows'], 60)
        self.assertEqual(params['cvd']['type'], 'robust')
        self.assertAlmostEqual(params['cvd']['median'], 30.5)

    def test_v19_label_runtime_requires_explicit_override_for_fallback(self):
        import prepare_training_data as prep

        old_available = prep.V19_LABELS_AVAILABLE
        old_error = prep.V19_LABELS_IMPORT_ERROR
        old_override = os.environ.get('QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS')
        try:
            prep.V19_LABELS_AVAILABLE = False
            prep.V19_LABELS_IMPORT_ERROR = RuntimeError('missing labels runtime')
            os.environ.pop('QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS', None)
            with self.assertRaises(RuntimeError):
                _require_causal_label_runtime('v19')
        finally:
            prep.V19_LABELS_AVAILABLE = old_available
            prep.V19_LABELS_IMPORT_ERROR = old_error
            if old_override is None:
                os.environ.pop('QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS', None)
            else:
                os.environ['QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS'] = old_override

    def test_regime_rules_are_causal_across_prefixes(self):
        rng = np.random.default_rng(7)
        n = 180
        df = pd.DataFrame(
            {
                'price': 100 + np.cumsum(rng.normal(0.0, 0.2, n)),
                'size': rng.integers(1, 8, n),
                'cvd': np.cumsum(rng.normal(0.0, 1.0, n)),
                'obi': rng.uniform(-1.0, 1.0, n),
                'inter_event_time': rng.exponential(0.3, n),
                'micro_atr': np.abs(rng.normal(0.2, 0.05, n)),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clf = RegimeClassifier(n_regimes=4)
            clf.fit(df.iloc[:130], output_dir=tmpdir)
            full = clf.predict(df)
            prefix_last = np.array([clf.predict(df.iloc[:i + 1])[-1] for i in range(130, n)], dtype=np.int8)

        np.testing.assert_array_equal(full[130:], prefix_last)

    def test_event_gate_respects_training_score_threshold(self):
        gate = EventGate(
            roll_window=3,
            vol_mult=1.10,
            obi_thr=0.08,
            wall_str_thr=0.70,
            shift_z_thr=0.75,
            score_threshold=1.5,
        )
        quiet = {
            'size': 1.0,
            'obi': 0.0,
            'bid_wall_strength': 0.0,
            'ask_wall_strength': 0.0,
            'cvd': 0.0,
            'kyle_lambda': 0.0,
            'hawkes_intensity': 0.0,
            'vnet': 0.0,
        }
        for _ in range(5):
            gate.evaluate(quiet)

        weak = {
            'size': 2.0,
            'obi': 0.09,
            'bid_wall_strength': 0.0,
            'ask_wall_strength': 0.0,
            'cvd': 0.0,
            'kyle_lambda': 0.0,
            'hawkes_intensity': 0.0,
            'vnet': 0.0,
        }
        weak_result = gate.evaluate(weak)
        self.assertFalse(weak_result['passed'])
        self.assertIn('score', weak_result['reason'])

        strong = {
            'size': 5.0,
            'obi': 0.40,
            'bid_wall_strength': 1.50,
            'ask_wall_strength': 0.0,
            'cvd': 5.0,
            'kyle_lambda': 5.0,
            'hawkes_intensity': 5.0,
            'vnet': 5.0,
        }
        strong_result = gate.evaluate(strong)
        self.assertTrue(strong_result['passed'])


if __name__ == "__main__":
    unittest.main()
