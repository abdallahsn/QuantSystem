# Graph Report - /Users/abdallah/Downloads/QS_FINAL  (2026-06-01)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2022 nodes · 6238 edges · 75 communities (69 shown, 6 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 713 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `eea03f3d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Data Validation and Features|Data Validation and Features]]
- [[_COMMUNITY_CatBoost Dashboard and Reporting|CatBoost Dashboard and Reporting]]
- [[_COMMUNITY_Market Data Cleaning|Market Data Cleaning]]
- [[_COMMUNITY_Backtest Diagnostics and Utilities|Backtest Diagnostics and Utilities]]
- [[_COMMUNITY_Auto-Calibrator and Autoencoders|Auto-Calibrator and Autoencoders]]
- [[_COMMUNITY_Soft Label Engine|Soft Label Engine]]
- [[_COMMUNITY_Live Prediction Pipeline|Live Prediction Pipeline]]
- [[_COMMUNITY_CatBoost Model Management|CatBoost Model Management]]
- [[_COMMUNITY_Configuration and Logging|Configuration and Logging]]
- [[_COMMUNITY_Causal Event Labeling|Causal Event Labeling]]
- [[_COMMUNITY_Label and Feature Auditing|Label and Feature Auditing]]
- [[_COMMUNITY_Training Dataset Engineering|Training Dataset Engineering]]
- [[_COMMUNITY_LSTM Meta-Learner|LSTM Meta-Learner]]
- [[_COMMUNITY_Dynamic Orderbook Labels|Dynamic Orderbook Labels]]
- [[_COMMUNITY_DeepLOB CNN Models|DeepLOB CNN Models]]
- [[_COMMUNITY_LOB Heatmap Visualization|LOB Heatmap Visualization]]
- [[_COMMUNITY_Experiment Comparison Reports|Experiment Comparison Reports]]
- [[_COMMUNITY_Threshold and ATR Analysis|Threshold and ATR Analysis]]
- [[_COMMUNITY_CatBoost Training Logic|CatBoost Training Logic]]
- [[_COMMUNITY_Contract Data Extraction|Contract Data Extraction]]
- [[_COMMUNITY_Runtime Context and Logging|Runtime Context and Logging]]
- [[_COMMUNITY_Refinery Comparison Reports|Refinery Comparison Reports]]
- [[_COMMUNITY_Orderbook Liquidity Detection|Orderbook Liquidity Detection]]
- [[_COMMUNITY_Feature Scaling and Normalization|Feature Scaling and Normalization]]
- [[_COMMUNITY_V19 Prediction Engine|V19 Prediction Engine]]
- [[_COMMUNITY_Training Workflow Orchestration|Training Workflow Orchestration]]
- [[_COMMUNITY_Training Data Alignment|Training Data Alignment]]
- [[_COMMUNITY_Intrabar Microstructure Aggregation|Intrabar Microstructure Aggregation]]
- [[_COMMUNITY_Release Gates and Walkforward|Release Gates and Walkforward]]
- [[_COMMUNITY_Rolling CatBoost Predictor|Rolling CatBoost Predictor]]
- [[_COMMUNITY_Position Sizing and Slippage|Position Sizing and Slippage]]
- [[_COMMUNITY_Market Data Conversion|Market Data Conversion]]
- [[_COMMUNITY_Decision Policy Evaluation|Decision Policy Evaluation]]
- [[_COMMUNITY_Lean Model Builder|Lean Model Builder]]
- [[_COMMUNITY_Monte Carlo Label Weights|Monte Carlo Label Weights]]
- [[_COMMUNITY_Transformer and LSTM Brains|Transformer and LSTM Brains]]
- [[_COMMUNITY_Unsupervised Learning Layer|Unsupervised Learning Layer]]
- [[_COMMUNITY_LOB Transformer Model|LOB Transformer Model]]
- [[_COMMUNITY_Sharded Feature Artifacts|Sharded Feature Artifacts]]
- [[_COMMUNITY_Interactive Label Visualizer|Interactive Label Visualizer]]
- [[_COMMUNITY_TCN-LSTM Meta-Learner|TCN-LSTM Meta-Learner]]
- [[_COMMUNITY_System Readiness Checks|System Readiness Checks]]
- [[_COMMUNITY_Server Environment Sanity|Server Environment Sanity]]
- [[_COMMUNITY_Soft Label Quality Control|Soft Label Quality Control]]
- [[_COMMUNITY_Feature Intelligence Reporting|Feature Intelligence Reporting]]
- [[_COMMUNITY_Meta-Learner Training Entrypoint|Meta-Learner Training Entrypoint]]
- [[_COMMUNITY_Signal Sanity Checks|Signal Sanity Checks]]
- [[_COMMUNITY_Market Replay Builder|Market Replay Builder]]
- [[_COMMUNITY_Structured Event Logging|Structured Event Logging]]
- [[_COMMUNITY_Walk-Forward Validation|Walk-Forward Validation]]
- [[_COMMUNITY_Cyclical Session Features|Cyclical Session Features]]
- [[_COMMUNITY_Anomaly Reconstruction Scoring|Anomaly Reconstruction Scoring]]
- [[_COMMUNITY_Pip Size and ADR|Pip Size and ADR]]
- [[_COMMUNITY_Structural Range Context|Structural Range Context]]
- [[_COMMUNITY_GPU Resource Management|GPU Resource Management]]
- [[_COMMUNITY_Microstructure and Iceberg Detection|Microstructure and Iceberg Detection]]
- [[_COMMUNITY_Day Trading Hardening Tests|Day Trading Hardening Tests]]
- [[_COMMUNITY_System Failsafe Policies|System Failsafe Policies]]
- [[_COMMUNITY_Fractional Differentiation|Fractional Differentiation]]
- [[_COMMUNITY_Raw Model Backtesting|Raw Model Backtesting]]
- [[_COMMUNITY_HTML Signal Reports|HTML Signal Reports]]
- [[_COMMUNITY_Artifact Manifest Generation|Artifact Manifest Generation]]
- [[_COMMUNITY_Backtest Patching Tools|Backtest Patching Tools]]
- [[_COMMUNITY_Dataset Verification|Dataset Verification]]
- [[_COMMUNITY_Matrix Transformation Utilities|Matrix Transformation Utilities]]
- [[_COMMUNITY_TensorFlow GPU Installation|TensorFlow GPU Installation]]
- [[_COMMUNITY_Artifact Discovery|Artifact Discovery]]
- [[_COMMUNITY_System Audit Planning|System Audit Planning]]
- [[_COMMUNITY_GPU Library Diagnostics|GPU Library Diagnostics]]
- [[_COMMUNITY_Diagnostics Package|Diagnostics Package]]
- [[_COMMUNITY_CUDA Symlink Fixes|CUDA Symlink Fixes]]
- [[_COMMUNITY_Tooling Namespace|Tooling Namespace]]
- [[_COMMUNITY_System Instructions|System Instructions]]

## God Nodes (most connected - your core abstractions)
1. `str` - 80 edges
2. `DataFrame` - 65 edges
3. `int` - 60 edges
4. `V19PredictionEngine` - 59 edges
5. `run_refinery()` - 59 edges
6. `str` - 52 edges
7. `run_training_pipeline()` - 43 edges
8. `MetaLearnerLSTM` - 39 edges
9. `DataFrame` - 39 edges
10. `float` - 38 edges

## Surprising Connections (you probably didn't know these)
- `DataFrame` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py
- `str` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py
- `Series` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py
- `int` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py
- `ndarray` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py

## Import Cycles
- None detected.

## Communities (75 total, 6 thin omitted)

### Community 0 - "Data Validation and Features"
Cohesion: 0.06
Nodes (94): deep_validation.py — 5 Scientific Tests for V16Pro-3, compute_daily_weekly_levels(), str, context_features.py — السياق الكامل للسوق, read_table(), apply_fractional_diff(), DataFrame, يطبق Fractional Differentiation على الـ features التراكمية (فقط).     تم تصحيح ق (+86 more)

### Community 1 - "CatBoost Dashboard and Reporting"
Cohesion: 0.06
Nodes (73): main(), plot_5m_catboost.py - Standalone 5m CatBoost dashboard, _apply_decision_policy_to_bars(), _apply_signal_pipeline_to_bars(), _build_confusion_chart(), _build_dashboard(), _build_price_hover_trace(), _build_turns_table() (+65 more)

### Community 2 - "Market Data Cleaning"
Cohesion: 0.09
Nodes (95): _apply_contract_filters(), _audit_command(), _book_level_columns(), _build_post_audit(), clean_market_data(), clean_mbo_frame(), clean_mbp_frame(), CleanConfig (+87 more)

### Community 3 - "Backtest Diagnostics and Utilities"
Cohesion: 0.06
Nodes (72): _binary_ece(), _build_backtest_window_mask(), _build_visual_diagnostics(), _coerce_row_aligned_array(), _configure_stdio_utf8(), _coverage_counts(), _coverage_sidecar_candidates(), _day_trading_manifest_path() (+64 more)

### Community 4 - "Auto-Calibrator and Autoencoders"
Cohesion: 0.13
Nodes (57): bool, DataFrame, int, str, auto_calibrator.py — معايرة تلقائية من الداتا, autoencoder_extractor.py — مستخرج الميزات بالـ Autoencoder, GARCHVolatilityProxy, LiquiditySweepDetector (+49 more)

### Community 5 - "Soft Label Engine"
Cohesion: 0.08
Nodes (68): Labels V19, soft_label_engine.py — Soft Labels for V19 QuantSystem =========================, محرك Soft Labels لنظام V19.          يُنتج:     - soft_label         : P(ربح | ا, أضف أعمدة Soft Labels إلى DataFrame المُصنَّف.                  يتوقع الأعمدة ال, واجهة بسيطة لإضافة Soft Labels إلى مخرجات build_causal_event_labels()., إعدادات محرك Soft Labels, SoftLabelConfig, add_day_trading_features() (+60 more)

### Community 6 - "Live Prediction Pipeline"
Cohesion: 0.05
Nodes (45): evaluate_on_test_set(), get_regime_features(), predict_live(), bool, DataFrame, float, object, Series (+37 more)

### Community 7 - "CatBoost Model Management"
Cohesion: 0.07
Nodes (26): Enum, CatBoostQuantBrain, bool, float, int, ndarray, str, catboost_brain.py — CatBoost Quantitative Brain ════════════════════════════════ (+18 more)

### Community 8 - "Configuration and Logging"
Cohesion: 0.11
Nodes (46): _deep_merge(), load_v19_config(), config_v19.py - Config loading helpers for QuantSystem V19, EventLogWriter, ExecutionLogger, compute_drift(), _default_alert_rules(), emit_alerts() (+38 more)

### Community 9 - "Causal Event Labeling"
Cohesion: 0.13
Nodes (47): _apply_trend_filter(), _bias_slice_counts(), _build_broad_event_gate(), build_causal_event_labels(), _build_training_event_gate(), _causal_expanding_median(), _coerce_label_timestamp_series(), _compute_adaptive_horizons() (+39 more)

### Community 10 - "Label and Feature Auditing"
Cohesion: 0.15
Nodes (45): binary_auc(), build_label_report(), build_risk_flags(), build_splits(), choose_features(), compute_feature_fold_metrics(), counts(), directional_mask() (+37 more)

### Community 11 - "Training Dataset Engineering"
Cohesion: 0.09
Nodes (33): append_dynamic_orderbook_features(), build_event_filter(), _build_quality_thresholds(), build_training_dataset(), compute_dynamic_levels(), _direction_from_future_return(), engineer_features(), EventGate (+25 more)

### Community 12 - "LSTM Meta-Learner"
Cohesion: 0.10
Nodes (26): _build_tcn_block(), _input_shape_matches(), _late_fusion_block(), _load_keras_model_allow_lambda(), make_volatility_weighted_loss(), MetaLearnerLSTM, _multitask_aux_heads(), _normalize_model_input_shape() (+18 more)

### Community 13 - "Dynamic Orderbook Labels"
Cohesion: 0.10
Nodes (32): dynamic_labels.py — V2: Unified Pipeline (OrderBook → Features → Labels) =======, append_dynamic_orderbook_features(), build_event_filter(), _build_quality_thresholds(), build_training_dataset(), compute_dynamic_levels(), _direction_from_future_return(), engineer_features() (+24 more)

### Community 14 - "DeepLOB CNN Models"
Cohesion: 0.09
Nodes (29): _build_targets(), _load_frame(), main(), _prepare_frame(), build_lob_tensor_dataset(), _build_snapshot_sampling_plan(), estimate_lob_tensor_bytes(), _prepare_sorted_frame() (+21 more)

### Community 15 - "LOB Heatmap Visualization"
Cohesion: 0.12
Nodes (38): DatetimeIndex, _find_best_soft_label_shard(), _load_lob(), _load_price_window(), main(), _parquet_feature_columns(), _plot_price_context(), _tensor_index_merge_asof() (+30 more)

### Community 16 - "Experiment Comparison Reports"
Cohesion: 0.21
Nodes (38): artifact_root_from_path(), as_float(), as_int(), best_run(), build_markdown_report(), candidate_report_roots(), collect_one_run(), collect_values_by_key() (+30 more)

### Community 17 - "Threshold and ATR Analysis"
Cohesion: 0.16
Nodes (36): add_approx_soft_labels(), analyze_atr(), analyze_returns(), _assign_session_utc(), composite_soft_neutral_score(), _first_long_event_bar(), _first_short_event_bar(), main() (+28 more)

### Community 18 - "CatBoost Training Logic"
Cohesion: 0.11
Nodes (17): CatBoostQuantBrain, bool, float, int, ndarray, str, catboost_brain.py — CatBoost Quantitative Brain ════════════════════════════════, FIX-Context: يُضيف rolling mean + std آخر rolling_window صف لكل feature. (+9 more)

### Community 19 - "Contract Data Extraction"
Cohesion: 0.17
Nodes (32): Counter, build_default_output_path(), choose_existing_column(), choose_target_symbol(), detect_month_from_data(), detect_month_from_filename(), detect_year_from_data(), detect_year_from_filename() (+24 more)

### Community 20 - "Runtime Context and Logging"
Cohesion: 0.17
Nodes (26): _directional_metrics(), DailyContextEngine, محرك سياق اليوم: يراقب اختراق أول ساعة (IB) والوقود اليومي المتبقي., EventGate, Online deterministic replica of the offline event filter., BaseRuntimeLogger, DataQualityLogger, PredictionLogger (+18 more)

### Community 21 - "Refinery Comparison Reports"
Cohesion: 0.17
Nodes (31): main(), plot_refinery_compare_5m.py --------------------------- Build daily 5m compariso, _aggregate_raw_bars(), _aggregate_refinery_bars(), _build_candle_hover_trace(), _build_daily_compare_figure(), _combine_ohlc_partials(), _daily_nav() (+23 more)

### Community 22 - "Orderbook Liquidity Detection"
Cohesion: 0.09
Nodes (16): deque, float, int, float, int, str, LiquidityTrapDetector, OrderBookSnapshotEngine (+8 more)

### Community 23 - "Feature Scaling and Normalization"
Cohesion: 0.16
Nodes (22): _clip_numeric_series(), fit_numeric_scaler_param(), infer_meta_feature_layout(), _load_json(), normalize_timestamp_columns(), prepare_feature_frame(), prepare_feature_row(), bool (+14 more)

### Community 24 - "V19 Prediction Engine"
Cohesion: 0.16
Nodes (10): int, ndarray, _pseudo_prob_head(), ndarray, Scalar regression output in (0,1) -> [p, 1-p] aligned with meta stacking., Meta-off: soft regression path uses a lower bar than classifier probs., V19PredictionEngine, IsotonicRegression (+2 more)

### Community 25 - "Training Workflow Orchestration"
Cohesion: 0.10
Nodes (27): V19 Default Config, Meta Learner, resolve_artifact_root(), _configure_stdio_utf8(), _pool_worker_init(), Windows consoles often default to cp1256; emoji/unicode logs then crash on print, Ensure worker processes (spawn) can emit UTF-8 without UnicodeEncodeError., _adaptive_tree_early_stopping_rounds() (+19 more)

### Community 26 - "Training Data Alignment"
Cohesion: 0.21
Nodes (25): _align_lob_to_rows(), _assert_lob_event_alignment(), _assert_no_forbidden_model_inputs(), build_event_training_view(), build_inference_scaler_params(), _build_row_time_mask(), build_time_splits(), _event_gate_train_prefix() (+17 more)

### Community 27 - "Intrabar Microstructure Aggregation"
Cohesion: 0.11
Nodes (24): datetime64, _depth_slope(), _num(), DataFrame, float, int, ndarray, Series (+16 more)

### Community 28 - "Release Gates and Walkforward"
Cohesion: 0.16
Nodes (24): _load_csv(), _compare(), evaluate_release_gates(), bool, float, str, release_gates_v19.py - Release gate evaluation for QuantSystem V19, save_gate_report() (+16 more)

### Community 29 - "Rolling CatBoost Predictor"
Cohesion: 0.13
Nodes (15): CatBoostQuantBrain, bool, float, int, ndarray, str, catboost_brain.py — CatBoost Quantitative Brain ════════════════════════════════, FIX-Context: يُضيف rolling mean + std آخر rolling_window صف لكل feature. (+7 more)

### Community 30 - "Position Sizing and Slippage"
Cohesion: 0.15
Nodes (14): confidence_bet_size(), fractional_kelly_bet_size(), kelly_bet_size(), position_size_from_prediction(), bool, float, int, str (+6 more)

### Community 31 - "Market Data Conversion"
Cohesion: 0.24
Nodes (23): _add_causal_atr(), _bars_from_ticks(), build_threshold_features(), _coerce_market_schema(), _combine_mbo_mbp_bars(), _diagnose_market_df(), _jsonable(), main() (+15 more)

### Community 32 - "Decision Policy Evaluation"
Cohesion: 0.24
Nodes (23): build_decision_policy(), _clip01(), _coerce_side_block(), effective_round_trip_cost_pips(), evaluate_decision_policy(), normalize_regime_probs(), normalized_entropy(), Any (+15 more)

### Community 33 - "Lean Model Builder"
Cohesion: 0.20
Nodes (23): ArgumentParser, build_arg_parser(), _build_requested_columns(), _chronological_split(), _ensure_cyclical_time_features(), _ensure_dir(), _ensure_relative_volatility(), _ensure_timestamp_sorted() (+15 more)

### Community 34 - "Monte Carlo Label Weights"
Cohesion: 0.17
Nodes (23): attach_mc_prior_columns(), compute_label_stability(), compute_mc_soft_weights(), _estimate_local_drift_px(), _gambler_ruin_p_tp(), MCWeightConfig, bool, DataFrame (+15 more)

### Community 35 - "Transformer and LSTM Brains"
Cohesion: 0.14
Nodes (8): make_causal_mask(), int, ndarray, تقوم بدمج التتابعات الزمنية (Sequences) مع الـ Embeddings         تُكرر الـ Embe, تم تسريع الـ MC Dropout عن طريق הـ Vectorization بدل الـ For-Loop, تم تصحيح الخلل الرياضي: حساب التدرجات بناءً على الفئة المتوقعة وليس مصفوفة الـ S, TransformerQuantitativeBrain_V3, WarmupCosineDecay

### Community 36 - "Unsupervised Learning Layer"
Cohesion: 0.17
Nodes (12): _aggregate_sequence(), bool, int, ndarray, str, unsupervised_layer.py — طبقة التعلم غير الخاضع للإشراف (Deep Autoencoder Bridge), تدريب الموديل لو الداتا مجهزة كـ Rolling Window (مسطحة), استخراج البصمة من صف واحد مجهز (+4 more)

### Community 37 - "LOB Transformer Model"
Cohesion: 0.18
Nodes (12): _infer_depth_view(), _load_keras_model_allow_lambda(), bool, float, int, ndarray, str, lob_transformer.py - LOB depth encoder for late-fusion path (+4 more)

### Community 38 - "Sharded Feature Artifacts"
Cohesion: 0.28
Nodes (21): Feature Artifact V19, _abs(), checkpoint_dir(), _ensure_dir(), iter_table_chunks(), load_artifact_manifest(), load_feature_artifact(), parquet_shard_paths() (+13 more)

### Community 39 - "Interactive Label Visualizer"
Cohesion: 0.24
Nodes (19): build_confusion_chart(), build_dashboard(), build_turns_table(), compute_signal_stats(), _dominant_bias(), load_and_prepare(), _load_csv(), main() (+11 more)

### Community 40 - "TCN-LSTM Meta-Learner"
Cohesion: 0.12
Nodes (38): MetaLearnerTCNLSTM, التعديل 3: نسخة محسّنة من MetaLearnerLSTM تضيف TCN block.      الفرق الوحيد عن ا, fill_uncovered_probabilities(), _adaptive_tree_depth(), _apply_scaler_to_stat_frame(), _apply_wall_scales(), _binary_ece(), _build_inner_time_split() (+30 more)

### Community 41 - "System Readiness Checks"
Cohesion: 0.22
Nodes (18): default_release_gates_path_for_profile(), _load_any(), load_release_gates(), str, Return the release-gate file for a V19 profile.      V19 currently ships one con, decide_runtime_mode(), read_market_data(), build_readiness_report() (+10 more)

### Community 42 - "Server Environment Sanity"
Cohesion: 0.25
Nodes (17): check_conda(), check_nvidia(), check_package(), check_python_and_env(), check_tensorflow(), evaluate(), import_name_for(), main() (+9 more)

### Community 43 - "Soft Label Quality Control"
Cohesion: 0.15
Nodes (13): compute_soft_sample_weights(), bool, DataFrame, float, int, ndarray, Monte Carlo: n_scenarios اختبارات مع perturbations على horizon / TP / SL., استرجع dynamic_threshold من الأعمدة المخزونة.         أولوية: label_dynamic_thre (+5 more)

### Community 45 - "Feature Intelligence Reporting"
Cohesion: 0.26
Nodes (15): build_report(), _is_label_artifact(), main(), _mi_score(), _numeric_series(), _spearman_no_scipy(), write_markdown(), Any (+7 more)

### Community 46 - "Meta-Learner Training Entrypoint"
Cohesion: 0.22
Nodes (17): main(), stage3_train.py - Standalone entrypoint for final MetaLearner training stage, _deeplob_aux_env_int(), _infer_event_gate_schema(), _load_lob_inputs(), _load_meta_learner_class(), _load_visual_runtime(), _phase_banner() (+9 more)

### Community 47 - "Signal Sanity Checks"
Cohesion: 0.32
Nodes (15): json_safe(), main(), parse_feature_sets(), robust_scale_train_only(), run_sanity(), _threshold_metrics(), write_csv(), write_markdown() (+7 more)

### Community 48 - "Market Replay Builder"
Cohesion: 0.26
Nodes (14): build_replay_dataset(), _coerce_optional_timestamp(), filter_timerange(), normalize_ts(), bool, DataFrame, float, int (+6 more)

### Community 49 - "Structured Event Logging"
Cohesion: 0.26
Nodes (7): feature_hash_from_dict(), log_event(), Any, float, str, safe_jsonable(), utc_now_iso()

### Community 50 - "Walk-Forward Validation"
Cohesion: 0.19
Nodes (9): float, int, ndarray, str, walk_forward.py — Walk-Forward Validation ══════════════════════════════════════, Walk-Forward Validation للـ Expansion Bias model., يرجع قائمة من (train_idx, test_idx) زمنية., يشغّل Walk-Forward كامل ويرجع النتائج.          Returns:             { (+1 more)

### Community 51 - "Cyclical Session Features"
Cohesion: 0.21
Nodes (12): add_cyclical_session_features(), add_session_features(), _coerce_session_timestamp_scalar(), _coerce_session_timestamp_series(), bool, DataFrame, float, Series (+4 more)

### Community 52 - "Anomaly Reconstruction Scoring"
Cohesion: 0.17
Nodes (6): bool, int, ndarray, str, يعيد نسبة الخطأ في إعادة البناء (Reconstruction Error) كـ Anomaly Score, يستخرج الأرقام السحرية (Latent Features) من طبقة البوتل نيك

### Community 53 - "Pip Size and ADR"
Cohesion: 0.21
Nodes (7): DataFrame, float, int, Series, Timestamp, bids / asks: list of [price, size], حساب batch للداتا الكاملة (للـ prepare_training_data.py).         يعيد DataFrame

### Community 54 - "Structural Range Context"
Cohesion: 0.22
Nodes (10): append_kalman_range_context(), append_wall_forward_deltas(), bool, DataFrame, float, int, ndarray, str (+2 more)

### Community 55 - "GPU Resource Management"
Cohesion: 0.25
Nodes (10): detect_gpu(), get_multiprocessing_workers(), get_optimal_batch_size(), print_gpu_report(), int, _query_nvidia_smi_gpus(), يحسب أفضل batch_size بناءً على الـ VRAM المتاح.     يتجنب التضخم المفرط الذي يدم, يرجع عدد الـ workers المأمون لمنع الـ CPU Thrashing (+2 more)

### Community 56 - "Microstructure and Iceberg Detection"
Cohesion: 0.27
Nodes (5): FastMicrostructureEngine, float, int, str, Hidden Volume Ratio (HVR) — NQ Iceberg Detection     تم تصحيح منطق اكتشاف الأوام

### Community 57 - "Day Trading Hardening Tests"
Cohesion: 0.42
Nodes (9): _base_bars(), main(), test_causal_event_gate_smoke(), test_cvd_session_features(), test_label_artifact_raw_names_are_detectable(), test_lob_depth_source(), test_lob_tensor_near_to_far(), DataFrame (+1 more)

### Community 58 - "System Failsafe Policies"
Cohesion: 0.50
Nodes (7): evaluate_system_health(), _load_shadow_approval(), _manifest_exists(), bool, str, failsafe_v19.py - Explicit degradation and runtime gating policies for V19, _schema_manifest_match()

### Community 59 - "Fractional Differentiation"
Cohesion: 0.29
Nodes (8): frac_diff_series(), _get_weights_ffd(), float, int, ndarray, Series, يطبق Fractional Differentiation بسرعة الضوء (Vectorized), يحسب أوزان الـ fractional diff (Fixed-width Window)     بدون عكس المصفوفة، لتجهي

### Community 60 - "Raw Model Backtesting"
Cohesion: 0.29
Nodes (7): main(), bool, float, int, str, raw_backtest_v19.py - Backtest existing V19 models directly on raw market data, run_raw_backtest()

### Community 61 - "HTML Signal Reports"
Cohesion: 0.57
Nodes (6): _fmt_num(), generate_report(), _kv_rows(), output_report.py — HTML signal report compatible with test_system.py, _reason_cards(), str

### Community 62 - "Artifact Manifest Generation"
Cohesion: 0.48
Nodes (6): collect_artifacts(), int, str, manifest_v19.py - Artifact manifest generation for QuantSystem V19, _sha256_file(), write_manifest()

### Community 63 - "Backtest Patching Tools"
Cohesion: 0.43
Nodes (6): main(), patch_file(), int, Path, str, _repl()

### Community 64 - "Dataset Verification"
Cohesion: 0.33
Nodes (6): _load_manifest(), main(), _series_stats(), int, Series, str

### Community 65 - "Matrix Transformation Utilities"
Cohesion: 0.53
Nodes (3): bool, DataFrame, ndarray

### Community 66 - "TensorFlow GPU Installation"
Cohesion: 0.83
Nodes (3): ensure_venv(), retry(), install_tf_gpu_cu12.sh script

### Community 68 - "System Audit Planning"
Cohesion: 0.67
Nodes (3): Codex Audit Plan, QuantSystem V19, Stage 1 Refinery

## Knowledge Gaps
- **93 isolated node(s):** `int`, `ArgumentParser`, `ndarray`, `Timestamp`, `Namespace` (+88 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `deque` connect `Orderbook Liquidity Detection` to `Auto-Calibrator and Autoencoders`, `CatBoost Model Management`, `Training Dataset Engineering`, `Dynamic Orderbook Labels`, `DeepLOB CNN Models`, `CatBoost Training Logic`, `Runtime Context and Logging`, `Microstructure and Iceberg Detection`, `Rolling CatBoost Predictor`, `Position Sizing and Slippage`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `MetaLearnerLSTM` connect `LSTM Meta-Learner` to `TCN-LSTM Meta-Learner`, `Meta-Learner Training Entrypoint`, `Runtime Context and Logging`, `V19 Prediction Engine`, `Training Workflow Orchestration`, `Training Data Alignment`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `V19PredictionEngine` connect `V19 Prediction Engine` to `CatBoost Dashboard and Reporting`, `Backtest Diagnostics and Utilities`, `LOB Transformer Model`, `Configuration and Logging`, `LSTM Meta-Learner`, `DeepLOB CNN Models`, `Runtime Context and Logging`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `str` (e.g. with `auto_calibrator.py` and `autoencoder_extractor.py`) actually correct?**
  _`str` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `DataFrame` (e.g. with `auto_calibrator.py` and `autoencoder_extractor.py`) actually correct?**
  _`DataFrame` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `int` (e.g. with `auto_calibrator.py` and `autoencoder_extractor.py`) actually correct?**
  _`int` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `V19PredictionEngine` (e.g. with `bool` and `DataFrame`) actually correct?**
  _`V19PredictionEngine` has 29 INFERRED edges - model-reasoned connections that need verification._