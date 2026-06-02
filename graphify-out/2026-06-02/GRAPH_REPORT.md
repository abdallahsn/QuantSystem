# Graph Report - .  (2026-06-02)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2056 nodes · 6269 edges · 84 communities (67 shown, 17 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 724 edges (avg confidence: 0.51)
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
- [[_COMMUNITY_Community 44|Community 44]]
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
- [[_COMMUNITY_CUDA Symlink Fixes|CUDA Symlink Fixes]]
- [[_COMMUNITY_Environment Repair Scripts|Environment Repair Scripts]]
- [[_COMMUNITY_Tooling Namespace|Tooling Namespace]]
- [[_COMMUNITY_Package Initialization|Package Initialization]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]

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
- `FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading` --semantically_similar_to--> `QuantSystem V19`  [INFERRED] [semantically similar]
  raw/arxiv_2603_21330.md → constraints.txt
- `DataFrame` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py
- `str` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py
- `Series` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py
- `int` --uses--> `V19PredictionEngine`  [INFERRED]
  backtest_v19.py → predict_v19.py

## Import Cycles
- None detected.

## Communities (84 total, 17 thin omitted)

### Community 0 - "Data Validation and Features"
Cohesion: 0.06
Nodes (121): V19 Default Config, Meta Learner, resolve_artifact_root(), infer_meta_feature_layout(), resolve_meta_feature_names(), MetaLearnerTCNLSTM, التعديل 3: نسخة محسّنة من MetaLearnerLSTM تضيف TCN block.      الفرق الوحيد عن ا, align_probability_columns() (+113 more)

### Community 1 - "CatBoost Dashboard and Reporting"
Cohesion: 0.05
Nodes (91): Codex Audit Plan, _base_bars(), main(), test_causal_event_gate_smoke(), test_cvd_session_features(), test_label_artifact_raw_names_are_detectable(), test_lob_depth_source(), test_lob_tensor_near_to_far() (+83 more)

### Community 2 - "Market Data Cleaning"
Cohesion: 0.09
Nodes (95): _apply_contract_filters(), _audit_command(), _book_level_columns(), _build_post_audit(), clean_market_data(), clean_mbo_frame(), clean_mbp_frame(), CleanConfig (+87 more)

### Community 3 - "Backtest Diagnostics and Utilities"
Cohesion: 0.06
Nodes (73): _binary_ece(), _build_backtest_window_mask(), _build_visual_diagnostics(), _coerce_row_aligned_array(), _configure_stdio_utf8(), _coverage_counts(), _coverage_sidecar_candidates(), _day_trading_manifest_path() (+65 more)

### Community 4 - "Auto-Calibrator and Autoencoders"
Cohesion: 0.08
Nodes (79): read_table(), التعديل 4: تصنيف الـ Regime باستخدام Wasserstein Distance.      لماذا Wasserstei, WassersteinRegimeClassifier, إعدادات محرك Soft Labels, SoftLabelConfig, _accumulate_integrity_snapshot(), _apply_continuous_contract_root(), _artifact_phase_dir() (+71 more)

### Community 5 - "Soft Label Engine"
Cohesion: 0.07
Nodes (59): main(), plot_5m_catboost.py - Standalone 5m CatBoost dashboard, _apply_decision_policy_to_bars(), _apply_signal_pipeline_to_bars(), _build_confusion_chart(), _build_dashboard(), _build_price_hover_trace(), _build_turns_table() (+51 more)

### Community 6 - "Live Prediction Pipeline"
Cohesion: 0.07
Nodes (57): DatetimeIndex, _find_best_soft_label_shard(), _load_lob(), _load_price_window(), main(), _parquet_feature_columns(), _plot_price_context(), _tensor_index_merge_asof() (+49 more)

### Community 7 - "CatBoost Model Management"
Cohesion: 0.05
Nodes (45): evaluate_on_test_set(), get_regime_features(), predict_live(), bool, DataFrame, float, object, Series (+37 more)

### Community 8 - "Configuration and Logging"
Cohesion: 0.16
Nodes (45): bool, DataFrame, int, str, auto_calibrator.py — معايرة تلقائية من الداتا, LiquiditySweepDetector, MomentumContextEngine, FastFIMDetector (+37 more)

### Community 9 - "Causal Event Labeling"
Cohesion: 0.07
Nodes (26): Enum, CatBoostQuantBrain, bool, float, int, ndarray, str, catboost_brain.py — CatBoost Quantitative Brain ════════════════════════════════ (+18 more)

### Community 10 - "Label and Feature Auditing"
Cohesion: 0.07
Nodes (24): deque, CatBoostQuantBrain, bool, float, int, ndarray, str, catboost_brain.py — CatBoost Quantitative Brain ════════════════════════════════ (+16 more)

### Community 11 - "Training Dataset Engineering"
Cohesion: 0.13
Nodes (47): _apply_trend_filter(), _bias_slice_counts(), _build_broad_event_gate(), build_causal_event_labels(), _build_training_event_gate(), _causal_expanding_median(), _coerce_label_timestamp_series(), _compute_adaptive_horizons() (+39 more)

### Community 12 - "LSTM Meta-Learner"
Cohesion: 0.15
Nodes (45): binary_auc(), build_label_report(), build_risk_flags(), build_splits(), choose_features(), compute_feature_fold_metrics(), counts(), directional_mask() (+37 more)

### Community 13 - "Dynamic Orderbook Labels"
Cohesion: 0.10
Nodes (34): append_dynamic_orderbook_features(), build_event_filter(), _build_quality_thresholds(), build_training_dataset(), compute_dynamic_levels(), _direction_from_future_return(), engineer_features(), EventGate (+26 more)

### Community 14 - "DeepLOB CNN Models"
Cohesion: 0.10
Nodes (26): _build_tcn_block(), _input_shape_matches(), _late_fusion_block(), _load_keras_model_allow_lambda(), make_volatility_weighted_loss(), MetaLearnerLSTM, _multitask_aux_heads(), _normalize_model_input_shape() (+18 more)

### Community 15 - "LOB Heatmap Visualization"
Cohesion: 0.10
Nodes (30): append_dynamic_orderbook_features(), build_event_filter(), _build_quality_thresholds(), build_training_dataset(), compute_dynamic_levels(), _direction_from_future_return(), engineer_features(), kalman_trend() (+22 more)

### Community 16 - "Experiment Comparison Reports"
Cohesion: 0.21
Nodes (38): artifact_root_from_path(), as_float(), as_int(), best_run(), build_markdown_report(), candidate_report_roots(), collect_one_run(), collect_values_by_key() (+30 more)

### Community 17 - "Threshold and ATR Analysis"
Cohesion: 0.16
Nodes (36): add_approx_soft_labels(), analyze_atr(), analyze_returns(), _assign_session_utc(), composite_soft_neutral_score(), _first_long_event_bar(), _first_short_event_bar(), main() (+28 more)

### Community 18 - "CatBoost Training Logic"
Cohesion: 0.17
Nodes (32): Counter, build_default_output_path(), choose_existing_column(), choose_target_symbol(), detect_month_from_data(), detect_month_from_filename(), detect_year_from_data(), detect_year_from_filename() (+24 more)

### Community 19 - "Contract Data Extraction"
Cohesion: 0.19
Nodes (31): EventLogWriter, ExecutionLogger, RiskLogger, emit_alerts(), load_baseline_from_artifacts(), load_jsonl(), MonitoringState, EventLogWriter (+23 more)

### Community 20 - "Runtime Context and Logging"
Cohesion: 0.17
Nodes (31): main(), plot_refinery_compare_5m.py --------------------------- Build daily 5m compariso, _aggregate_raw_bars(), _aggregate_refinery_bars(), _build_candle_hover_trace(), _build_daily_compare_figure(), _combine_ohlc_partials(), _daily_nav() (+23 more)

### Community 21 - "Refinery Comparison Reports"
Cohesion: 0.16
Nodes (10): _directional_metrics(), int, ndarray, main(), _pseudo_prob_head(), DataFrame, ndarray, Scalar regression output in (0,1) -> [p, 1-p] aligned with meta stacking. (+2 more)

### Community 22 - "Orderbook Liquidity Detection"
Cohesion: 0.13
Nodes (19): _infer_depth_view(), _load_keras_model_allow_lambda(), bool, float, int, ndarray, str, lob_transformer.py - LOB depth encoder for late-fusion path (+11 more)

### Community 23 - "Feature Scaling and Normalization"
Cohesion: 0.12
Nodes (19): _build_targets(), _load_frame(), main(), _prepare_frame(), _prepare_sorted_frame(), bool, ndarray, str (+11 more)

### Community 24 - "V19 Prediction Engine"
Cohesion: 0.11
Nodes (24): datetime64, _depth_slope(), _num(), DataFrame, float, int, ndarray, Series (+16 more)

### Community 25 - "Training Workflow Orchestration"
Cohesion: 0.16
Nodes (20): EventGate, Online deterministic replica of the offline event filter., DataQualityLogger, PredictionLogger, preprocessing_v19.py - Shared preprocessing for QuantSystem V19 inference, Loads the V19 feature schema and scaler params, then applies the same     featur, V19FeaturePreprocessor, DailyLossGuard (+12 more)

### Community 26 - "Training Data Alignment"
Cohesion: 0.13
Nodes (15): CatBoostQuantBrain, bool, float, int, ndarray, str, catboost_brain.py — CatBoost Quantitative Brain ════════════════════════════════, FIX-Context: يُضيف rolling mean + std آخر rolling_window صف لكل feature. (+7 more)

### Community 27 - "Intrabar Microstructure Aggregation"
Cohesion: 0.24
Nodes (23): _add_causal_atr(), _bars_from_ticks(), build_threshold_features(), _coerce_market_schema(), _combine_mbo_mbp_bars(), _diagnose_market_df(), _jsonable(), main() (+15 more)

### Community 28 - "Release Gates and Walkforward"
Cohesion: 0.20
Nodes (23): ArgumentParser, build_arg_parser(), _build_requested_columns(), _chronological_split(), _ensure_cyclical_time_features(), _ensure_dir(), _ensure_relative_volatility(), _ensure_timestamp_sorted() (+15 more)

### Community 29 - "Rolling CatBoost Predictor"
Cohesion: 0.16
Nodes (20): _deep_merge(), default_release_gates_path_for_profile(), _load_any(), load_release_gates(), load_v19_config(), str, config_v19.py - Config loading helpers for QuantSystem V19, Return the release-gate file for a V19 profile.      V19 currently ships one con (+12 more)

### Community 30 - "Position Sizing and Slippage"
Cohesion: 0.16
Nodes (21): collect_artifacts(), int, str, manifest_v19.py - Artifact manifest generation for QuantSystem V19, _sha256_file(), write_manifest(), _compare(), evaluate_release_gates() (+13 more)

### Community 31 - "Market Data Conversion"
Cohesion: 0.15
Nodes (22): build_replay_dataset(), _coerce_optional_timestamp(), filter_timerange(), normalize_ts(), bool, DataFrame, float, int (+14 more)

### Community 32 - "Decision Policy Evaluation"
Cohesion: 0.24
Nodes (23): build_decision_policy(), _clip01(), _coerce_side_block(), effective_round_trip_cost_pips(), evaluate_decision_policy(), normalize_regime_probs(), normalized_entropy(), Any (+15 more)

### Community 33 - "Lean Model Builder"
Cohesion: 0.17
Nodes (23): attach_mc_prior_columns(), compute_label_stability(), compute_mc_soft_weights(), _estimate_local_drift_px(), _gambler_ruin_p_tp(), MCWeightConfig, bool, DataFrame (+15 more)

### Community 34 - "Monte Carlo Label Weights"
Cohesion: 0.14
Nodes (16): compute_daily_weekly_levels(), DailyContextEngine, GARCHVolatilityProxy, LiquidityWallsEngine, DataFrame, float, int, Series (+8 more)

### Community 35 - "Transformer and LSTM Brains"
Cohesion: 0.14
Nodes (8): make_causal_mask(), int, ndarray, تقوم بدمج التتابعات الزمنية (Sequences) مع الـ Embeddings         تُكرر الـ Embe, تم تسريع الـ MC Dropout عن طريق הـ Vectorization بدل الـ For-Loop, تم تصحيح الخلل الرياضي: حساب التدرجات بناءً على الفئة المتوقعة وليس مصفوفة الـ S, TransformerQuantitativeBrain_V3, WarmupCosineDecay

### Community 36 - "Unsupervised Learning Layer"
Cohesion: 0.17
Nodes (12): _aggregate_sequence(), bool, int, ndarray, str, unsupervised_layer.py — طبقة التعلم غير الخاضع للإشراف (Deep Autoencoder Bridge), تدريب الموديل لو الداتا مجهزة كـ Rolling Window (مسطحة), استخراج البصمة من صف واحد مجهز (+4 more)

### Community 37 - "LOB Transformer Model"
Cohesion: 0.28
Nodes (21): Feature Artifact V19, _abs(), checkpoint_dir(), _ensure_dir(), iter_table_chunks(), load_artifact_manifest(), load_feature_artifact(), parquet_shard_paths() (+13 more)

### Community 38 - "Sharded Feature Artifacts"
Cohesion: 0.24
Nodes (19): build_confusion_chart(), build_dashboard(), build_turns_table(), compute_signal_stats(), _dominant_bias(), load_and_prepare(), _load_csv(), main() (+11 more)

### Community 39 - "Interactive Label Visualizer"
Cohesion: 0.22
Nodes (18): decide_runtime_mode(), evaluate_system_health(), _load_shadow_approval(), _manifest_exists(), bool, str, failsafe_v19.py - Explicit degradation and runtime gating policies for V19, _schema_manifest_match() (+10 more)

### Community 40 - "TCN-LSTM Meta-Learner"
Cohesion: 0.20
Nodes (12): confidence_bet_size(), fractional_kelly_bet_size(), kelly_bet_size(), position_size_from_prediction(), bool, float, int, str (+4 more)

### Community 41 - "System Readiness Checks"
Cohesion: 0.23
Nodes (18): check_conda(), check_nvidia(), check_package(), check_python_and_env(), check_tensorflow(), evaluate(), import_name_for(), main() (+10 more)

### Community 42 - "Server Environment Sanity"
Cohesion: 0.16
Nodes (13): build_lob_tensor_dataset(), _build_snapshot_sampling_plan(), estimate_lob_tensor_bytes(), LOBTensorBuilder, float, int, نسخة خفيفة من update_mbp تستقبل arrays/scalars مباشرة         لتجنب بناء dict لك, يستلم trade tick ويضيفه للـ footprint الحالي.         يُحدد مستوى السعر بناءً عل (+5 more)

### Community 43 - "Soft Label Quality Control"
Cohesion: 0.24
Nodes (9): BaseRuntimeLogger, feature_hash_from_dict(), log_event(), Any, float, str, logging_v19.py - Structured JSONL logging for QuantSystem V19, safe_jsonable() (+1 more)

### Community 44 - "Community 44"
Cohesion: 0.26
Nodes (15): build_report(), _is_label_artifact(), main(), _mi_score(), _numeric_series(), _spearman_no_scipy(), write_markdown(), Any (+7 more)

### Community 45 - "Feature Intelligence Reporting"
Cohesion: 0.19
Nodes (9): LiquidityTrapDetector, OrderBookSnapshotEngine, float, int, str, Liquidity Trap Score:     يكشف فخاخ السيولة بناءً على علاقة الـ OBI بحركة السعر., يحسب OBI من كل snapshot MBP10.     V16Pro: أضاف OBI Z-Score Dynamic, يكتشف الـ Spoofing من خلال مقارنة الـ Snapshot الحالي بالسابق.         Spoofing (+1 more)

### Community 46 - "Meta-Learner Training Entrypoint"
Cohesion: 0.32
Nodes (15): json_safe(), main(), parse_feature_sets(), robust_scale_train_only(), run_sanity(), _threshold_metrics(), write_csv(), write_markdown() (+7 more)

### Community 47 - "Signal Sanity Checks"
Cohesion: 0.20
Nodes (7): bool, int, ndarray, str, autoencoder_extractor.py — مستخرج الميزات بالـ Autoencoder, يعيد نسبة الخطأ في إعادة البناء (Reconstruction Error) كـ Anomaly Score, يستخرج الأرقام السحرية (Latent Features) من طبقة البوتل نيك

### Community 48 - "Market Replay Builder"
Cohesion: 0.21
Nodes (12): add_cyclical_session_features(), add_session_features(), _coerce_session_timestamp_scalar(), _coerce_session_timestamp_series(), bool, DataFrame, float, Series (+4 more)

### Community 49 - "Structured Event Logging"
Cohesion: 0.19
Nodes (9): float, int, ndarray, str, walk_forward.py — Walk-Forward Validation ══════════════════════════════════════, Walk-Forward Validation للـ Expansion Bias model., يرجع قائمة من (train_idx, test_idx) زمنية., يشغّل Walk-Forward كامل ويرجع النتائج.          Returns:             { (+1 more)

### Community 50 - "Walk-Forward Validation"
Cohesion: 0.21
Nodes (12): deep_validation.py — 5 Scientific Tests for V16Pro-3, _add_rolling_context(), _build_refinery_split_context(), _coerce_naive_timestamp_series(), _merge(), _process_mbo(), _process_mbo_sequential(), _process_mbp10() (+4 more)

### Community 51 - "Cyclical Session Features"
Cohesion: 0.23
Nodes (11): apply_fractional_diff(), frac_diff_series(), _get_weights_ffd(), DataFrame, float, int, ndarray, Series (+3 more)

### Community 52 - "Anomaly Reconstruction Scoring"
Cohesion: 0.33
Nodes (10): compute_drift(), _default_alert_rules(), _extract_feature_vectors(), _percentile(), _psi(), float, monitoring_v19.py - Monitoring, drift, and alerting for QuantSystem V19, _safe_mean() (+2 more)

### Community 53 - "Pip Size and ADR"
Cohesion: 0.25
Nodes (10): detect_gpu(), get_multiprocessing_workers(), get_optimal_batch_size(), print_gpu_report(), int, _query_nvidia_smi_gpus(), يحسب أفضل batch_size بناءً على الـ VRAM المتاح.     يتجنب التضخم المفرط الذي يدم, يرجع عدد الـ workers المأمون لمنع الـ CPU Thrashing (+2 more)

### Community 54 - "Structural Range Context"
Cohesion: 0.22
Nodes (10): append_kalman_range_context(), append_wall_forward_deltas(), bool, DataFrame, float, int, ndarray, str (+2 more)

### Community 55 - "GPU Resource Management"
Cohesion: 0.38
Nodes (10): Universal features of price formation in financial markets: perspectives from Deep Learning, BDLOB: Bayesian Deep Convolutional Neural Networks for Limit Order Books, Multi-Level Order-Flow Imbalance in a Limit Order Book, Deep Attentive Survival Analysis in Limit Order Books: Estimating Fill Probabilities with Convolutional-Transformers, Deep Limit Order Book Forecasting, HLOB -- Information Persistence and Structure in Limit Order Books, Price predictability in limit order book with deep learning model, TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data (+2 more)

### Community 56 - "Microstructure and Iceberg Detection"
Cohesion: 0.40
Nodes (3): OrderWallScanner, Scan MBP10 snapshots and return:       - walls/gaps used for dynamic TP/SL, _RefineryProgressTracker

### Community 57 - "Day Trading Hardening Tests"
Cohesion: 0.32
Nodes (3): float, int, str

### Community 58 - "System Failsafe Policies"
Cohesion: 0.57
Nodes (6): _fmt_num(), generate_report(), _kv_rows(), output_report.py — HTML signal report compatible with test_system.py, _reason_cards(), str

### Community 59 - "Fractional Differentiation"
Cohesion: 0.43
Nodes (6): main(), patch_file(), int, Path, str, _repl()

### Community 60 - "Raw Model Backtesting"
Cohesion: 0.33
Nodes (6): _load_manifest(), main(), _series_stats(), int, Series, str

### Community 61 - "HTML Signal Reports"
Cohesion: 0.33
Nodes (6): FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading, Early Detection of Latent Microstructure Regimes in Limit Order Books, numpy==1.26.4, tensorflow==2.17.1, QuantSystem V19, V19 Release Gates Metrics

### Community 62 - "Artifact Manifest Generation"
Cohesion: 0.83
Nodes (3): ensure_venv(), retry(), install_tf_gpu_cu12.sh script

### Community 63 - "Backtest Patching Tools"
Cohesion: 0.50
Nodes (4): _configure_stdio_utf8(), _pool_worker_init(), Windows consoles often default to cp1256; emoji/unicode logs then crash on print, Ensure worker processes (spawn) can emit UTF-8 without UnicodeEncodeError.

### Community 64 - "Dataset Verification"
Cohesion: 0.67
Nodes (3): ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination, PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Market Trading and Latency Arbitrage, OOM-RL: Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems

## Knowledge Gaps
- **116 isolated node(s):** `int`, `ArgumentParser`, `ndarray`, `Timestamp`, `Namespace` (+111 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `deque` connect `Label and Feature Auditing` to `Monte Carlo Label Weights`, `Configuration and Logging`, `Causal Event Labeling`, `TCN-LSTM Meta-Learner`, `Dynamic Orderbook Labels`, `Feature Intelligence Reporting`, `LOB Heatmap Visualization`, `Feature Scaling and Normalization`, `Training Workflow Orchestration`, `Training Data Alignment`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `V19PredictionEngine` connect `Refinery Comparison Reports` to `Data Validation and Features`, `Monte Carlo Label Weights`, `Backtest Diagnostics and Utilities`, `Soft Label Engine`, `DeepLOB CNN Models`, `Contract Data Extraction`, `Orderbook Liquidity Detection`, `Feature Scaling and Normalization`, `Training Workflow Orchestration`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `Feature Artifact V19` connect `LOB Transformer Model` to `Data Validation and Features`, `Backtest Diagnostics and Utilities`, `Auto-Calibrator and Autoencoders`, `LSTM Meta-Learner`, `Contract Data Extraction`, `Training Workflow Orchestration`, `Release Gates and Walkforward`, `Market Data Conversion`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `str` (e.g. with `auto_calibrator.py` and `autoencoder_extractor.py`) actually correct?**
  _`str` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `DataFrame` (e.g. with `auto_calibrator.py` and `autoencoder_extractor.py`) actually correct?**
  _`DataFrame` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `int` (e.g. with `auto_calibrator.py` and `autoencoder_extractor.py`) actually correct?**
  _`int` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `V19PredictionEngine` (e.g. with `bool` and `DataFrame`) actually correct?**
  _`V19PredictionEngine` has 29 INFERRED edges - model-reasoned connections that need verification._