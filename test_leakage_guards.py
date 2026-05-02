import os
import tempfile
import unittest
import json

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QUANTSYSTEM_SKIP_HEAVY_ML", "1")
os.environ.setdefault("QUANTSYSTEM_SKIP_GPU_DETECT", "1")

from backtest_v19 import _load_meta_features, _realized_fill_pricing, _simulate_trade_path
from prepare_training_data import _fit_regime_surface, _frame_integrity_snapshot, _process_mbp10, _require_causal_label_runtime
from modules.dynamic_labels import EventGate, engineer_features
from modules.feature_factory_v19 import V19FeatureFactory, infer_meta_feature_layout, resolve_meta_feature_names
from modules.labels_v19 import _compute_adaptive_horizons, build_causal_event_labels
from modules.dynamic_labels import kalman_trend
from modules.meta_learner import _input_shape_matches
from modules.oof_stacking import run_sequential_oof
from modules.purging_embargo import walk_forward_expanding
from modules.regime_classifier import RegimeClassifier, REGIME_META_SCORE_COLS, REGIME_ONE_HOT_COLS
from train_v19 import _assert_single_contract_df, _load_required_stage1_artifacts, _project_sequence_aux_context, _raw_stat_frame, _time_series, build_inference_scaler_params
from modules.failsafe_v19 import evaluate_system_health, decide_runtime_mode


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

    def test_kalman_trend_is_causal_across_prefixes(self):
        prefix = np.linspace(100.0, 101.0, 120, dtype=np.float64)
        base = np.r_[prefix, np.linspace(101.0, 102.0, 80, dtype=np.float64)]
        spiked = np.r_[prefix, np.linspace(101.0, 120.0, 80, dtype=np.float64)]

        labels_base, strength_base, price_base = kalman_trend(base, slope_threshold=0.05)
        labels_spiked, strength_spiked, price_spiked = kalman_trend(spiked, slope_threshold=0.05)

        np.testing.assert_array_equal(labels_base[:120], labels_spiked[:120])
        np.testing.assert_allclose(strength_base[:120], strength_spiked[:120], atol=1e-8)
        np.testing.assert_allclose(price_base[:120], price_spiked[:120], atol=1e-8)

    def test_spoofing_baseline_does_not_backfill_future_depth(self):
        n = 32
        ts = pd.date_range("2026-01-01", periods=n, freq="s")
        base = pd.DataFrame(
            {
                "ts_event": ts,
                "action": ["A"] * n,
                "bid_px_00": np.full(n, 100.0),
                "ask_px_00": np.full(n, 100.25),
                "bid_sz_00": np.full(n, 10.0),
                "ask_sz_00": np.full(n, 10.0),
            }
        )
        for level in range(1, 10):
            base[f"bid_px_{level:02d}"] = 100.0 - 0.25 * level
            base[f"ask_px_{level:02d}"] = 100.25 + 0.25 * level
            base[f"bid_sz_{level:02d}"] = 10.0
            base[f"ask_sz_{level:02d}"] = 10.0

        future_depth = base.copy()
        future_depth.loc[20:, "bid_sz_00"] = 10_000.0
        future_depth.loc[20:, "ask_sz_00"] = 10_000.0

        base_feat = _process_mbp10(base, tick_size=0.25)
        spiked_feat = _process_mbp10(future_depth, tick_size=0.25)

        np.testing.assert_allclose(
            base_feat.loc[:19, "spoofing_ratio"].values,
            spiked_feat.loc[:19, "spoofing_ratio"].values,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            base_feat.loc[:19, "spoofing_duration"].values,
            spiked_feat.loc[:19, "spoofing_duration"].values,
            atol=1e-8,
        )

    def test_integrity_snapshot_flags_timestamp_and_value_anomalies(self):
        df = pd.DataFrame(
            {
                "ts_event": [
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:01",
                    "2025-12-31 23:59:59",
                    None,
                ],
                "price": [100.0, 0.0, 101.0, -5.0],
                "size": [1.0, -2.0, 0.0, np.nan],
                "symbol": ["ES", "ES", "NQ", "ES"],
            }
        )

        snap = _frame_integrity_snapshot(df)

        self.assertEqual(snap["rows"], 4)
        self.assertEqual(snap["missing_ts_rows"], 1)
        self.assertEqual(snap["duplicate_ts_rows"], 1)
        self.assertEqual(snap["non_monotonic_ts_steps"], 1)
        self.assertEqual(snap["price_nonpositive_rows"], 2)
        self.assertEqual(snap["size_negative_rows"], 1)
        self.assertEqual(snap["size_zero_rows"], 1)
        self.assertEqual(snap["symbol_unique_count"], 2)

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

    def test_cost_floors_do_not_improve_backtest_pnl(self):
        row = {
            'bid_px_00': 100.0,
            'ask_px_00': 101.0,
            'bid_sz_00': 10.0,
            'ask_sz_00': 10.0,
        }
        base = _realized_fill_pricing(
            entry_row=row,
            exit_row=row,
            direction='LONG',
            size=1,
            raw_pnl_pips=10.0,
            tick_size=1.0,
            round_trip_cost_pips=1.0,
            tick_value=10.0,
            commission_per_side=1.0,
            min_spread_ticks=1.0,
            min_slippage_ticks=1.0,
            spread_multiplier=0.5,
        )
        stress = _realized_fill_pricing(
            entry_row=row,
            exit_row=row,
            direction='LONG',
            size=1,
            raw_pnl_pips=10.0,
            tick_size=1.0,
            round_trip_cost_pips=1.0,
            tick_value=10.0,
            commission_per_side=2.0,
            min_spread_ticks=1.0,
            min_slippage_ticks=2.0,
            spread_multiplier=0.5,
        )
        self.assertLessEqual(stress['net_pnl_pips'], base['net_pnl_pips'])

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

    def test_time_series_rejects_missing_or_invalid_required_timestamps(self):
        with self.assertRaises(ValueError):
            _time_series(pd.DataFrame({'x': [1, 2, 3]}), 'ts_event')
        with self.assertRaises(ValueError):
            _time_series(pd.DataFrame({'ts_event': [None, '2025-01-01 00:00:02', None]}), 'ts_event')

    def test_run_sequential_oof_rejects_overlapping_test_rows(self):
        splits = [
            (np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)),
            (np.array([0, 1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)),
        ]

        def _predictor(train_idx, test_idx, fold_no):
            return np.zeros((len(test_idx), 2), dtype=np.float32), {'fold_no': fold_no}

        with self.assertRaises(ValueError):
            run_sequential_oof(5, 2, splits, _predictor)

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

    def test_training_contract_guard_rejects_mixed_symbols(self):
        df = pd.DataFrame(
            {
                'symbol': ['ES', 'NQ', 'ES'],
                'bias_label': [0, 1, 2],
            }
        )
        with self.assertRaises(RuntimeError):
            _assert_single_contract_df(df, context='unit_test')

    def test_rollout_requires_shadow_approval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'feature_schema_v19.json'), 'w') as f:
                json.dump({}, f)
            with open(os.path.join(tmpdir, 'manifest.json'), 'w') as f:
                json.dump({}, f)
            health = evaluate_system_health(
                models_dir=tmpdir,
                engine_status={
                    'catboost_available': True,
                    'meta_available': True,
                    'regime_available': True,
                    'visual_available': True,
                },
                feature_row={'cvd': 1.0, 'obi': 0.1, 'micro_atr': 0.2, 'kyle_lambda': 0.1, 'hawkes_intensity': 0.1, 'vwap_z_score': 0.0},
                policy={'require_shadow_approval_for_rollout': True},
                manifest_path=os.path.join(tmpdir, 'manifest.json'),
            )
            runtime = decide_runtime_mode(health, {'require_shadow_approval_for_rollout': True})
            self.assertFalse(runtime['allow_rollout'])
            self.assertIn('shadow_approval_missing', runtime['reason'])

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

    def test_engineer_features_regime_is_causal_across_prefixes(self):
        prefix = pd.DataFrame(
            {
                'obi': np.linspace(-0.5, 0.5, 120, dtype=np.float64),
                'cvd': np.linspace(0.0, 5.0, 120, dtype=np.float64),
            }
        )
        base = pd.concat([prefix, pd.DataFrame({'obi': np.linspace(0.1, 0.2, 40), 'cvd': np.linspace(5.0, 6.0, 40)})], ignore_index=True)
        spiked = pd.concat([prefix, pd.DataFrame({'obi': np.linspace(0.1, 10.0, 40), 'cvd': np.linspace(5.0, 6.0, 40)})], ignore_index=True)

        base_regime = engineer_features(base, roll_window=20)['regime'].to_numpy(dtype=np.int8)
        spiked_regime = engineer_features(spiked, roll_window=20)['regime'].to_numpy(dtype=np.int8)

        np.testing.assert_array_equal(base_regime[:120], spiked_regime[:120])

    def test_bias_labels_are_causal_across_prefixes(self):
        prefix_prices = np.linspace(100.0, 101.0, 120, dtype=np.float64)
        base_prices = np.r_[prefix_prices, np.linspace(101.0, 102.0, 80, dtype=np.float64)]
        spiked_prices = np.r_[prefix_prices, np.linspace(101.0, 120.0, 80, dtype=np.float64)]

        def _label(price_arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(
                {
                    'ts_event': pd.date_range('2026-01-01', periods=len(price_arr), freq='s'),
                    'price': price_arr,
                    'size': np.ones(len(price_arr), dtype=np.float32),
                    'cvd': np.linspace(0.0, 1.0, len(price_arr), dtype=np.float32),
                }
            )
            out = build_causal_event_labels(
                df,
                horizon=20,
                direction_threshold_ticks=1.0,
                tp_mult=1.2,
                sl_mult=1.0,
                tick_size=0.01,
                trend_strength_min=0.05,
            )
            return out['bias_label'].to_numpy(dtype=np.int8)

        base_bias = _label(base_prices)
        spiked_bias = _label(spiked_prices)
        np.testing.assert_array_equal(base_bias[:100], spiked_bias[:100])

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
        self.assertEqual(info['effective_stride'], 1)
        self.assertEqual(info['full_sample_rows'], len(df))
        self.assertEqual(info['train_sample_rows'], len(train_idx))
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

    def test_meta_feature_layout_supports_catboost_xgboost_surface(self):
        names = resolve_meta_feature_names(include_xgboost=True)
        layout = infer_meta_feature_layout(names)
        self.assertEqual([spec['name'] for spec in layout['base_models']], ['catboost', 'xgboost'])
        self.assertEqual(layout['base_prob_dim'], 4)
        self.assertEqual(len(layout['regime_meta_cols']), 7)

    def test_walk_forward_expanding_produces_disjoint_test_blocks(self):
        splits = list(
            walk_forward_expanding(
                n_samples=3913,
                n_folds=9,
                test_size=0.10,
                embargo_pct=0.02,
                min_train_pct=0.20,
                min_train_rows=200,
            )
        )
        seen = np.array([], dtype=np.int64)
        self.assertGreaterEqual(len(splits), 1)
        for _, test_idx in splits:
            self.assertEqual(len(np.intersect1d(seen, test_idx)), 0)
            seen = np.r_[seen, np.asarray(test_idx, dtype=np.int64)]

    def test_walk_forward_expanding_coverage_comes_from_prefix_holdout_not_tiny_folds(self):
        low_fold_splits = list(
            walk_forward_expanding(
                n_samples=3913,
                n_folds=5,
                test_size=0.10,
                embargo_pct=0.02,
                min_train_pct=0.20,
                min_train_rows=200,
            )
        )
        high_fold_splits = list(
            walk_forward_expanding(
                n_samples=3913,
                n_folds=9,
                test_size=0.10,
                embargo_pct=0.02,
                min_train_pct=0.20,
                min_train_rows=200,
            )
        )

        self.assertEqual(len(low_fold_splits), 9)
        self.assertEqual(len(high_fold_splits), 9)
        self.assertGreaterEqual(min(len(test_idx) for _, test_idx in low_fold_splits), 300)
        self.assertEqual(
            sum(len(test_idx) for _, test_idx in low_fold_splits),
            sum(len(test_idx) for _, test_idx in high_fold_splits),
        )

    def test_feature_factory_accepts_legacy_catboost_only_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {
                'version': 'v19-event-binary',
                'seq_len': 50,
                'stat_features': ['cvd'],
                'meta_features': resolve_meta_feature_names(include_xgboost=False),
                'visual_features': [],
                'input_dim': 10,
            }
            with open(os.path.join(tmpdir, 'feature_schema_v19.json'), 'w') as f:
                json.dump(schema, f)
            factory = V19FeatureFactory(tmpdir)
            self.assertEqual(factory.meta_features, resolve_meta_feature_names(include_xgboost=False))

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
