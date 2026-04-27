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
from prepare_training_data import _fit_regime_surface, _require_causal_label_runtime
from modules.dynamic_labels import EventGate
from modules.feature_factory_v19 import V19FeatureFactory
from modules.labels_v22 import _compute_adaptive_horizons
from modules.meta_learner import _input_shape_matches
from modules.regime_classifier import RegimeClassifier, REGIME_META_SCORE_COLS, REGIME_ONE_HOT_COLS
from train_v19 import _load_required_stage1_artifacts, _project_sequence_aux_context, _raw_stat_frame, build_inference_scaler_params


class LeakageGuardTests(unittest.TestCase):
    def test_meta_learner_rejects_stale_input_shape(self):
        class FakeModel:
            input_shape = (None, 50, 45)

        self.assertFalse(_input_shape_matches(FakeModel(), seq_len=50, n_total=48))
        self.assertTrue(_input_shape_matches(FakeModel(), seq_len=50, n_total=45))

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
        meta = np.zeros((len(df), 9), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta_features_live_v19.npy")
            np.save(path, meta)
            with self.assertRaises(ValueError):
                _load_meta_features(
                    df,
                    explicit_path=path,
                    expected_dim=9,
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

    def test_regime_scores_are_causal_across_prefixes(self):
        rng = np.random.default_rng(11)
        n = 180
        df = pd.DataFrame(
            {
                'price': 100 + np.cumsum(rng.normal(0.0, 0.15, n)),
                'size': rng.integers(1, 10, n),
                'cvd': np.cumsum(rng.normal(0.0, 0.8, n)),
                'obi': rng.uniform(-1.0, 1.0, n),
                'inter_event_time': rng.exponential(0.4, n),
                'micro_atr': np.abs(rng.normal(0.2, 0.04, n)),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clf = RegimeClassifier(n_regimes=4)
            clf.fit(df.iloc[:130], output_dir=tmpdir)
            full_scores = clf.predict_scores(df)
            prefix_last = np.vstack(
                [clf.predict_scores(df.iloc[:i + 1]).iloc[-1].values for i in range(130, n)]
            ).astype(np.float64)

        np.testing.assert_allclose(full_scores.iloc[130:].values, prefix_last, atol=1e-8)
        self.assertTrue(((full_scores.values >= 0.0) & (full_scores.values <= 1.0)).all())

    def test_regime_meta_surface_has_expected_order(self):
        rng = np.random.default_rng(13)
        n = 64
        df = pd.DataFrame(
            {
                'price': 100 + np.cumsum(rng.normal(0.0, 0.1, n)),
                'size': rng.integers(1, 6, n),
                'cvd': np.cumsum(rng.normal(0.0, 0.6, n)),
                'obi': rng.uniform(-1.0, 1.0, n),
                'inter_event_time': rng.exponential(0.5, n),
                'micro_atr': np.abs(rng.normal(0.2, 0.03, n)),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            clf = RegimeClassifier(n_regimes=4)
            clf.fit(df, output_dir=tmpdir)
            meta = clf.predict_regime_meta(df)

        expected_cols = [*REGIME_ONE_HOT_COLS, *REGIME_META_SCORE_COLS]
        self.assertEqual(list(meta.columns), expected_cols)
        self.assertEqual(meta.shape, (n, len(expected_cols)))
        np.testing.assert_allclose(meta.loc[:, list(REGIME_ONE_HOT_COLS)].sum(axis=1).values, 1.0)
        self.assertTrue(((meta.loc[:, list(REGIME_META_SCORE_COLS)].values >= 0.0) & (meta.loc[:, list(REGIME_META_SCORE_COLS)].values <= 1.0)).all())

    def test_regime_surface_rules_expand_from_coarse_stride(self):
        rng = np.random.default_rng(23)
        n = 6000
        df = pd.DataFrame(
            {
                'price': 100 + np.cumsum(rng.normal(0.0, 0.15, n)),
                'size': rng.integers(1, 8, n),
                'cvd': np.cumsum(rng.normal(0.0, 0.7, n)),
                'obi': rng.uniform(-1.0, 1.0, n),
                'inter_event_time': rng.exponential(0.4, n),
                'micro_atr': np.abs(rng.normal(0.2, 0.04, n)),
            }
        )
        split_ctx = {'split_idx': 4000}
        train_idx = np.arange(4000, dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            labels, info = _fit_regime_surface(
                df,
                train_idx=train_idx,
                split_ctx=split_ctx,
                output_dir=tmpdir,
                regime_mode='rules',
                regime_stride=12,
                regime_window=50,
                regime_progress_every=0,
            )

        self.assertEqual(len(labels), len(df))
        self.assertEqual(info['mode'], 'rules')
        self.assertEqual(info['effective_stride'], 12)
        self.assertLess(info['full_sample_rows'], len(df))
        self.assertLess(info['train_sample_rows'], len(train_idx))
        self.assertTrue(np.isin(labels, [0, 1, 2, 3]).all())

    def test_regime_surface_can_be_disabled(self):
        df = pd.DataFrame({'price': [1.0, 1.1, 1.2]})
        labels, info = _fit_regime_surface(
            df,
            train_idx=np.array([0, 1], dtype=np.int32),
            split_ctx={'split_idx': 2},
            output_dir='.',
            regime_mode='off',
            regime_stride=50,
            regime_window=50,
            regime_progress_every=0,
        )
        np.testing.assert_array_equal(labels, np.zeros(len(df), dtype=np.int8))
        self.assertEqual(info['mode'], 'off')

    def test_stage1_cached_meta_surface_rejects_old_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, 'meta_features_oof_v19.npy'), np.zeros((5, 6), dtype=np.float32))
            np.save(os.path.join(tmpdir, 'meta_coverage_v19.npy'), np.ones(5, dtype=np.uint8))
            for name in ('catboost_advisor_v19.cbm', 'catboost_classes_v19.json', 'regime_classifier.pkl'):
                with open(os.path.join(tmpdir, name), 'wb') as f:
                    f.write(b'0')
            with self.assertRaises(ValueError):
                _load_required_stage1_artifacts(tmpdir, n_rows=5)

    def test_feature_factory_rejects_old_schema_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {
                'version': 'v19-event-binary',
                'seq_len': 50,
                'stat_features': ['cvd'],
                'meta_features': ['cb_prob_long', 'cb_prob_short', 'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3'],
                'visual_features': [],
                'input_dim': 7,
            }
            with open(os.path.join(tmpdir, 'feature_schema_v19.json'), 'w') as f:
                import json
                json.dump(schema, f)
            with self.assertRaises(ValueError):
                V19FeatureFactory(tmpdir)

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
