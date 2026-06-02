# Codex Audit Plan

Repository: `/Users/abdallah/Downloads/QS_FINAL`

Date: `2026-05-21`

Scope: static repository audit plus integration-plan review for the current V19 pipeline and the proposed V19.2 layer. This document does not delete, rename, or change behavior. الهدف هنا تثبيت الخريطة قبل أي تعديل.

## Executive Summary

Most important issue: the current supervision stack is still dominated by hard `bias_label` generation. Soft labels do not independently solve the 98% NEUTRAL problem because `/Users/abdallah/Downloads/QS_FINAL/modules/soft_label_engine.py:135` `SoftLabelEngine.attach_soft_labels()` derives the main `soft_label` from the existing `bias_label`; NEUTRAL rows are kept near `0.5`. Any migration must fix causal hard labels and event selection before expecting soft-label training to improve.

The active training path is:

1. `/Users/abdallah/Downloads/QS_FINAL/stage1_refinery.py:23` `main()`
2. `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3461` `run_refinery()`
3. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1097` `build_causal_event_labels()`
4. `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3121` `_normalize_and_save()`
5. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:4035` `run_training_pipeline()`
6. Optional wrappers:
   - `/Users/abdallah/Downloads/QS_FINAL/stage2_catboost.py:111` `main()`
   - `/Users/abdallah/Downloads/QS_FINAL/stage3_train.py:24` `main()`

The active deployment and replay path is:

1. `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:144` `V19PredictionEngine`
2. `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py:1177` `run_causal_backtest()`
3. `/Users/abdallah/Downloads/QS_FINAL/raw_backtest_v19.py:17` `run_raw_backtest()`
4. `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py:373` `run_walkforward()`
5. `/Users/abdallah/Downloads/QS_FINAL/shadow_v19.py:34` `run_shadow()`
6. `/Users/abdallah/Downloads/QS_FINAL/paper_v19.py:57` `run_paper()`

The V19.2 layer described in `/Users/abdallah/Downloads/QuantSystem_Integration_Report.pdf` is not present as source files in this checkout. Treat it as an external candidate layer to import under a new namespace, not as code already wired into production.

## Current Architecture Map

### Primary Event/Tick Pipeline

| Stage | Entry point | Key functions | Current role |
|---|---|---|---|
| Stage 1 CLI | `/Users/abdallah/Downloads/QS_FINAL/stage1_refinery.py:23` | `main()` | Thin wrapper that loads config and calls `prepare_training_data.run_refinery()`. |
| Event/Tick refinery | `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3461` | `run_refinery()` | Main MBO/MBP ingestion, feature generation, label generation, tensor generation, normalization, and artifact writing. |
| V19 labels | `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1097` | `build_causal_event_labels()` | Active causal event labeler for the event/tick pipeline. |
| Soft labels | `/Users/abdallah/Downloads/QS_FINAL/modules/soft_label_engine.py:118` | `SoftLabelEngine`, `attach_soft_labels()` | Adds `soft_label`, `label_confidence`, and `soft_sample_weight` after hard labels exist. |
| MC priors | `/Users/abdallah/Downloads/QS_FINAL/modules/mc_label_weights.py:298` | `attach_mc_prior_columns()` | Adds Monte Carlo prior and stability sample-weight columns. |
| Feature artifact IO | `/Users/abdallah/Downloads/QS_FINAL/modules/feature_artifact_v19.py` | `load_feature_artifact()` and artifact helpers | Used by training, prediction, paper, shadow, and raw replay paths. |
| Training | `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:4035` | `run_training_pipeline()` | Central CatBoost/XGBoost, visual embedding, and meta-learner trainer. |
| Prediction | `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:144` | `V19PredictionEngine` | Live-like row-by-row inference engine. |

### Day-Trading Hybrid Pipeline

| Stage | Entry point | Key functions | Current role |
|---|---|---|---|
| Day-trading refinery | `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2246` | `run_day_trading_refinery()` | Separate MBO/MBP-to-bars feature and label pipeline. |
| MBO bars | `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:755` | `aggregate_mbo_to_bars()` | Aggregates raw MBO into 5-minute style bars. |
| Day-trading features | `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:970` | `add_day_trading_features()` | Adds technical, intrabar, volatility, session, and microstructure features. |
| Event detection | `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1528` | `detect_microstructure_events()` | Builds causal event gate for the day-trading path. |
| Day-trading labels | `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1711` | `label_by_outcome()` | Current active day-trading barrier/outcome labeler. |
| Day-trading soft labels | `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2209` | `attach_soft_labels_dt()` | Calls V19 soft-label engine, then preserves day-trading base `soft_label`. |

### Backtest, Walk-Forward, Paper, Live

| Mode | Entry point | Key functions | Notes |
|---|---|---|---|
| Causal backtest | `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py:1177` | `run_causal_backtest()` | Replays features through `V19PredictionEngine` one row at a time. |
| Raw replay backtest | `/Users/abdallah/Downloads/QS_FINAL/raw_backtest_v19.py:17` | `run_raw_backtest()` | Builds replay dataset from raw MBO/MBP, then calls causal backtest. |
| Walk-forward | `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py:373` | `run_walkforward()` | Builds train/test raw replay windows, trains, backtests, and applies gates. |
| Shadow | `/Users/abdallah/Downloads/QS_FINAL/shadow_v19.py:34` | `run_shadow()` | Live-like shadow logging using `V19PredictionEngine(run_mode='shadow')`. |
| Paper | `/Users/abdallah/Downloads/QS_FINAL/paper_v19.py:57` | `run_paper()` | Paper/rollout loop using `V19PredictionEngine`. |
| Legacy live | `/Users/abdallah/Downloads/QS_FINAL/live_predictor.py:193` | `run_bar_pipeline()` | Deprecated unsafe path; raises `RuntimeError` telling users to use `predict_v19.V19PredictionEngine`. |

## Exact Files Used by the Current Training Pipeline

### Stage 1 Feature and Label Build

Active files:

- `/Users/abdallah/Downloads/QS_FINAL/stage1_refinery.py`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/config_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/soft_label_engine.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/mc_label_weights.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/structural_context_labels_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/feature_factory_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/feature_artifact_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/deeplob_cnn.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/lob_transformer.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/session_features.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/microstructure.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/micro_volatility.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/market_research_features.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/context_features.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/fractional_diff.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/fim_anomaly.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/fisher_alpha.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/orderbook.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/autoencoder_extractor.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/auto_calibrator.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/purging_embargo.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/slippage_model.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/manifest_v19.py`

Critical functions:

- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:152` `_call_build_causal_event_labels()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:256` `_resolve_soft_label_runtime_config()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:429` `_require_causal_label_runtime()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:1976` `_process_mbo_chunk()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2179` `_process_mbo_sequential()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2285` `_process_mbp10()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2507` `_merge()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2930` `_build_refinery_split_context()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3121` `_normalize_and_save()`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3461` `run_refinery()`

### Stage 2 and Stage 3 Training

Active files:

- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/stage2_catboost.py`
- `/Users/abdallah/Downloads/QS_FINAL/stage3_train.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/meta_learner.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/oof_stacking.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/regime_classifier.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/deeplob_cnn.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/lob_transformer.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/preprocessing_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/decision_policy_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/manifest_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/purging_embargo.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/gpu_config.py`

Critical functions and constants:

- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:164` `SCHEMA_VERSION = 'v19-event-binary'`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:165` `TRAIN_MODE_EVENT_BINARY`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:166` `TRAIN_MODE_DIRECTIONAL_ALL`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:172` `STAGE1_TARGET_BIAS`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:173` `STAGE1_TARGET_SOFT_LABEL`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:214` `FORBIDDEN_MODEL_INPUT_COLS`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:812` `load_training_csv()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:893` `build_event_training_view()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1178` `_quality_sample_weights()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1284` `_stability_mc_sample_weights()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1689` `build_time_splits()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1778` `_build_inner_time_split()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1806` `stage1_oof_meta()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:2838` `_load_lob_inputs()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:2924` `_align_lob_to_rows()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:3025` `stage2_oof_visual_embeddings()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:3584` `stage3_meta_learner_v19()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:4035` `run_training_pipeline()`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:4626` `main()`

## Label Generation Flow

### Event/Tick V19 Label Flow

Current path:

1. `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3461` `run_refinery()` merges MBO/MBP and prepares event/tick rows.
2. `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:152` `_call_build_causal_event_labels()` calls the V19 labeler.
3. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1097` `build_causal_event_labels()` generates causal labels.
4. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:542` `_build_broad_event_gate()` creates broad `event_flag`.
5. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:410` `_build_training_event_gate()` creates rate-controlled `train_event_flag`.
6. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:576` `_compute_adaptive_horizons()` computes per-row horizons when enabled.
7. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:672` `_forward_scan_rows()` scans future path outcomes.
8. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:827` `_forward_scan_per_row()` assigns raw path/bias results.
9. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1008` `_apply_trend_filter()` can mask directional labels back to NEUTRAL.
10. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1753` attaches soft labels if enabled.
11. `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1770` attaches Monte Carlo prior columns if enabled.

Outputs created by this path include:

- `bias_label`
- `bias_label_detail`
- `long_label`
- `short_label`
- `path_outcome`
- `neutral_reason`
- `forward_return`
- `label_end_ts`
- `label_horizon_steps`
- `effective_horizon`
- `label_dynamic_threshold`
- `event_flag`
- `train_event_flag`
- `event_score`
- `event_trigger_count`
- `is_event`
- `soft_label`
- `label_confidence`
- `soft_sample_weight`
- Monte Carlo prior/weight columns when enabled

High-risk label dependency:

```python
# /Users/abdallah/Downloads/QS_FINAL/modules/soft_label_engine.py
bias = df["bias_label"].to_numpy(dtype=np.int8, copy=False)
soft_label[bias == DIR_LONG] = soft_long[bias == DIR_LONG]
soft_label[bias == DIR_SHORT] = soft_short[bias == DIR_SHORT]
soft_label[bias == DIR_NEUTRAL] = 0.5
```

This means soft labels are downstream of hard labels. إذا كانت `bias_label` 98% NEUTRAL، فإن `soft_label` سيحمل نفس المشكلة بدرجة كبيرة، خصوصا عندما يكون `stage1_target=soft_label`.

### Day-Trading Label Flow

Current path:

1. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2246` `run_day_trading_refinery()`
2. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:755` `aggregate_mbo_to_bars()`
3. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:970` `add_day_trading_features()`
4. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1063` `assign_regime_label()`
5. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1131` `add_event_direction()`
6. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1528` `detect_microstructure_events()`
7. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1711` `label_by_outcome()`
8. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2170` `add_soft_labels()`
9. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2209` `attach_soft_labels_dt()`

Important distinction: `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2036` `build_day_trading_labels()` exists, but the current `run_day_trading_refinery()` path uses `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1711` `label_by_outcome()` as the active labeler.

## Feature Generation Flow

### Event/Tick Feature Flow

1. MBO chunk processing:
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:1976` `_process_mbo_chunk()`
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2179` `_process_mbo_sequential()`
   - Generates features such as CVD, absorption, cancel ratio, micro ATR, Kyle lambda, Hawkes intensity, `vnet`, VWAP-related fields, and event/order-flow summaries.

2. MBP10/order-book processing:
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2285` `_process_mbp10()`
   - Generates LOB/microstructure fields including OBI, spoofing flags, liquidity gaps, wall strength, wall distance, liquidity density, and level/depth features.

3. MBO + MBP merge:
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2507` `_merge()`
   - Uses `pd.merge_asof(..., direction='backward', tolerance=...)`.
   - Risk: any V19.2 feature generator must preserve this causality contract. No forward joins.

4. Rolling and context features:
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2276` `_add_rolling_context()`
   - `/Users/abdallah/Downloads/QS_FINAL/modules/session_features.py`
   - `/Users/abdallah/Downloads/QS_FINAL/modules/context_features.py`
   - `/Users/abdallah/Downloads/QS_FINAL/modules/fractional_diff.py`
   - `/Users/abdallah/Downloads/QS_FINAL/modules/micro_volatility.py`

5. Labeling:
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3927` calls `_call_build_causal_event_labels()`.

6. LOB tensor generation:
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:4086` calls `build_lob_tensor_dataset()`.
   - Implementation: `/Users/abdallah/Downloads/QS_FINAL/modules/deeplob_cnn.py`.

7. Normalization and artifact writing:
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:2930` `_build_refinery_split_context()`
   - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:3121` `_normalize_and_save()`
   - This path preserves `raw__*` CatBoost advisor features, fits scalers on train slice only, trains unsupervised components on train slice only, and writes final Parquet/artifact outputs.

### Day-Trading Feature Flow

1. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:755` `aggregate_mbo_to_bars()`
2. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:970` `add_day_trading_features()`
3. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1331` `build_rolling_lob_tensors_from_mbp()`
4. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1466` `build_rolling_lob_tensors_mbo_only()`
5. `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2246` `run_day_trading_refinery()` writes `day_trading_features.parquet` and manifest outputs.

## Training Flow

### Active `train_v19.py` Flow

1. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:812` `load_training_csv()`
   - Loads via `modules.feature_artifact_v19`.
   - Requires `bias_label`, `ts_event`, and `label_end_ts`.
   - Sorts chronologically by `ts_event`.
   - Validates `label_end_ts >= ts_event`.

2. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:893` `build_event_training_view()`
   - `event_binary`: trains on `(train_event_flag == 1)` and directional `bias_label in [LONG, SHORT]`, with fallback logic if the event pool is empty.
   - `directional_all`: trains on all directional LONG/SHORT rows and excludes NEUTRAL.
   - Adds `quality_sample_weight`, `conf_target`, and `event_seq_idx`.

3. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1689` `build_time_splits()`
   - Uses chronological expanding/walk-forward style splits.
   - Uses `label_end_ts` and dynamic embargo based on `label_horizon_steps`.

4. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1778` `_build_inner_time_split()`
   - Applies overlap purge and embargo through `/Users/abdallah/Downloads/QS_FINAL/modules/purging_embargo.py`.

5. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1806` `stage1_oof_meta()`
   - Trains OOF CatBoost/XGBoost and regime meta.
   - If `stage1_target='soft_label'`, uses regression heads and maps scalar predictions to pseudo-probabilities.
   - If `stage1_target='bias'`, uses directional binary classification.

6. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:3025` `stage2_oof_visual_embeddings()`
   - Builds OOF visual/deep LOB embeddings.

7. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:3584` `stage3_meta_learner_v19()`
   - Trains final sequence/meta learner.

8. `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:4035` `run_training_pipeline()`
   - Orchestrates phases and artifact output.

### Training Wrappers

- `/Users/abdallah/Downloads/QS_FINAL/stage2_catboost.py:111` `main()`
  - Calls `run_training_pipeline(..., phase='catboost')`.
  - Important diagnostic at `/Users/abdallah/Downloads/QS_FINAL/stage2_catboost.py:46`: current Stage 2 CatBoost does not use `/Users/abdallah/Downloads/QS_FINAL/modules/catboost_brain.py`.

- `/Users/abdallah/Downloads/QS_FINAL/stage3_train.py:24` `main()`
  - Calls `run_training_pipeline(..., phase='train')`.

- `/Users/abdallah/Downloads/QS_FINAL/build_lean_model.py:306` `train_lean_model()`
  - Separate lean CatBoost path. Treat as alternative/experimental unless explicitly selected.

## Deployment and Backtest Flow

### Prediction Engine

Core file: `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py`

Key functions:

- `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:144` `V19PredictionEngine`
- `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:428` `_base_fallback_min_edge()`
- `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:701` `predict_step()`
- `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:925` `predict_live()`
- `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:934` `run_backtest()`
- `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:1023` `main()`

Important dependencies:

- `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py:496` `EventGate`
- `/Users/abdallah/Downloads/QS_FINAL/modules/preprocessing_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/feature_artifact_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/decision_policy_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/modules/failsafe_v19.py`

### Causal Backtest

Core file: `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py`

Key functions:

- `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py:123` `_realized_fill_pricing()`
- `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py:202` `_simulate_trade_path()`
- `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py:518` `_enforce_oos_backtest_guard()`
- `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py:1177` `run_causal_backtest()`
- `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py:1639` `main()`

High-risk flags in this area:

- `--allow_in_sample_live_meta_override`
- `--allow_in_sample_data_override`
- `--allow_oracle_forward_return`
- `--live_like_runtime_inputs`

These flags are useful for diagnostics, but dangerous for release metrics. لا تعتمد عليها في تقييم production.

### Walk-Forward

Core file: `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py`

Key functions:

- `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py:36` `build_walkforward_windows()`
- `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py:100` `compute_eval_visual_embeddings()`
- `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py:373` `run_walkforward()`
- `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py:632` `main()`

Flow:

1. Build chronological windows.
2. Build train replay dataset from raw MBO/MBP.
3. Train via `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:4035` `run_training_pipeline()`.
4. Build test replay dataset with external scaler.
5. Run base and stress causal backtests.
6. Evaluate release gates via `/Users/abdallah/Downloads/QS_FINAL/modules/release_gates_v19.py`.

### Paper and Shadow

- `/Users/abdallah/Downloads/QS_FINAL/shadow_v19.py:34` `run_shadow()`
- `/Users/abdallah/Downloads/QS_FINAL/paper_v19.py:57` `run_paper()`

Both use `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py:144` `V19PredictionEngine`.

## Duplicate or Unused Module Candidates

Do not delete these yet. هذه قائمة ownership/risk فقط.

| Candidate | Status | Evidence | Risk |
|---|---|---|---|
| `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py` | Active legacy/base module | Imported by `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py`, `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py`, `/Users/abdallah/Downloads/QS_FINAL/train_v19.py`, and `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py`. | Contains legacy `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py:741` `label_with_forward_scan()` and `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py:700` `_direction_from_future_return()`. Keep boundary explicit. |
| `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels2.py` | Unused duplicate candidate | No active imports found. File differs from `dynamic_labels.py`. | High confusion risk; archive later only after import tests and git history review. |
| `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py` | Active current event/tick labeler | Called from `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:152`. | High blast radius. Do not rewrite in place during V19.2 import. |
| `/Users/abdallah/Downloads/QS_FINAL/modules/catboost_brain.py` | Legacy/experimental | File header says active Stage 2 trains via `train_v19.py`, not this module. | Confuses model ownership. Keep but mark legacy. |
| `/Users/abdallah/Downloads/QS_FINAL/modules/catboost_brain2.py` | Legacy/experimental duplicate | Defines `/Users/abdallah/Downloads/QS_FINAL/modules/catboost_brain2.py:43` `CatBoostQuantBrain`; not used by active training path. | Confuses current CatBoost implementation. |
| `/Users/abdallah/Downloads/QS_FINAL/modules/catboost_brain3.py` | Legacy/experimental duplicate | Defines `/Users/abdallah/Downloads/QS_FINAL/modules/catboost_brain3.py:54` `CatBoostQuantBrain`; not used by active training path. | Confuses current CatBoost implementation. |
| `/Users/abdallah/Downloads/QS_FINAL/live_predictor.py` | Deprecated legacy live path | `/Users/abdallah/Downloads/QS_FINAL/live_predictor.py:193` `run_bar_pipeline()` raises and points to `predict_v19.V19PredictionEngine`. | README/docs may still imply it is active. Documentation risk. |
| `/Users/abdallah/Downloads/QS_FINAL/modules/live_predictor.py` | Compatibility shim | Imports root `live_predictor` symbols. | Keeps deprecated API visible. |
| `/Users/abdallah/Downloads/QS_FINAL/online_learning.py` and `/Users/abdallah/Downloads/QS_FINAL/modules/online_learning.py` | Root implementation plus shim | Module shim imports root classes. | Ownership ambiguity. |
| `/Users/abdallah/Downloads/QS_FINAL/regime_config.py` and `/Users/abdallah/Downloads/QS_FINAL/modules/regime_config.py` | Root config plus shim | Module shim imports root constants. | Ownership ambiguity. |
| `/Users/abdallah/Downloads/QS_FINAL/modules/intrabar_microstructure.py` | Compatibility shim | Imports `/Users/abdallah/Downloads/QS_FINAL/modules/tick_intrabar_slices.py`. | Low risk, but mark as shim. |
| Root diagnostic scripts | Utility/diagnostic | Examples now live under `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/` after Stage 1 structural cleanup. | Keep diagnostics out of active trading entry points. |

## V19.2 Integration Layer Status

The PDF references V19.2 modules for statistical discovery, microstructure simulation, liquidity topology, and tensor building. These files were not found in the current repo root or `modules/` directory:

- `fix_ohlc.py`
- `combine_months.py`
- `combine_raw.py`
- `feature_simulators.py`
- `wall_depth_simulator.py`
- `iceberg_simulator.py`
- `session_mapper.py`
- `liquidity_topology_engine.py`
- `edge_scanner.py`
- `cluster_engine.py`
- `statistics_module.py`
- `market_specs.py`
- `tensor_builder.py`
- `run_pipeline.py`
- `show_candidates.py`

Recommendation: import these under a new namespace such as `/Users/abdallah/Downloads/QS_FINAL/modules/v19_2/` or `/Users/abdallah/Downloads/QS_FINAL/research_v19_2/`, with no production wiring until standalone tests pass.

## Risk List

1. Label imbalance and hard-label dependency
   - Location: `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1097` and `/Users/abdallah/Downloads/QS_FINAL/modules/soft_label_engine.py:135`.
   - Risk: 98% NEUTRAL hard labels make soft labels degenerate. Model may learn inactivity or threshold artifacts.
   - Recommendation: first measure event pool, directional count, path outcome count, and neutral reasons before changing models.

2. Soft-label target can hide a broken hard-label process
   - Location: `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:173` and `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:1806`.
   - Risk: `stage1_target='soft_label'` can produce smooth regression outputs even if labels are structurally untradeable.
   - Recommendation: require soft-label distribution, calibration, and per-class forward-return audits.

3. Multiple label engines with different semantics
   - Locations:
     - `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py:1097` `build_causal_event_labels()`
     - `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:1711` `label_by_outcome()`
     - `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py:2036` `build_day_trading_labels()`
     - `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py:741` `label_with_forward_scan()`
   - Risk: metrics across event/tick and day-trading paths are not directly comparable.
   - Recommendation: define a label contract document and regression tests before migration.

4. Legacy `dynamic_labels.py` is still on the active path
   - Location: `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py`.
   - Risk: older utilities remain imported by active V19 labels and live event gate. Refactoring it can break training and inference.
   - Recommendation: isolate new V19.2 logic in new files first.

5. Forward-looking supervision columns must remain forbidden
   - Locations:
     - `/Users/abdallah/Downloads/QS_FINAL/modules/structural_context_labels_v19.py:79` `append_wall_forward_deltas()`
     - `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:214` `FORBIDDEN_MODEL_INPUT_COLS`
   - Risk: forward deltas or `forward_return` can leak into features if artifact contracts drift.
   - Recommendation: add regression tests that fail if any forbidden column reaches model input.

6. Backtest override flags can invalidate release metrics
   - Location: `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py`.
   - Risk: in-sample override or oracle forward-return modes can contaminate reported performance.
   - Recommendation: release scripts must assert these flags are off.

7. LOB/visual coverage can silently become weak
   - Locations:
     - `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py:4086`
     - `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:2838`
     - `/Users/abdallah/Downloads/QS_FINAL/train_v19.py:2924`
   - Risk: model may train with zero-filled visual embeddings or low MBP coverage.
   - Recommendation: gate on LOB coverage, timestamp alignment, and nonzero tensor count.

8. `train_v19.py` has high blast radius
   - Location: `/Users/abdallah/Downloads/QS_FINAL/train_v19.py`.
   - Risk: training phases, feature filtering, OOF logic, artifact writing, and reporting are concentrated in one file.
   - Recommendation: use wrapper tests and small config-gated changes only.

9. Documentation mismatch around live path
   - Location: `/Users/abdallah/Downloads/QS_FINAL/live_predictor.py:193`.
   - Risk: user or automation may call deprecated live path.
   - Recommendation: update docs only after migration plan is accepted.

10. No complete end-to-end regression suite
   - Existing tests are useful, including:
     - `/Users/abdallah/Downloads/QS_FINAL/tests/test_leakage_guards.py`
     - `/Users/abdallah/Downloads/QS_FINAL/tests/test_train_v19_training_modes.py`
     - `/Users/abdallah/Downloads/QS_FINAL/tests/test_backtest_oos_guard.py`
     - `/Users/abdallah/Downloads/QS_FINAL/tests/test_day_trading_labels.py`
   - Gap: no full formal refinery-to-train-to-backtest regression contract covering label distribution, leakage, chronology, and release gating.

## Recommended Execution Order

### Phase 0 - Freeze and Baseline Current Behavior

Risk: low. No behavior change.

Actions:

1. Keep all current files in place.
2. Record current configs, model artifact locations, and sample datasets.
3. Run baseline syntax/tests before any integration:

```bash
python -m py_compile prepare_training_data.py prepare_day_trading.py train_v19.py predict_v19.py backtest_v19.py walkforward_v19.py
pytest tests/test_day_trading_labels.py tests/test_train_v19_training_modes.py tests/test_leakage_guards.py tests/test_backtest_oos_guard.py
```

### Phase 1 - Create an Ownership Registry

Risk: low. Documentation only.

Actions:

1. Mark active files, legacy files, shims, diagnostics, and external V19.2 candidates.
2. Do not delete `dynamic_labels2.py` or `catboost_brain*.py`.
3. Make `predict_v19.py` the documented live path and mark `live_predictor.py` as deprecated.

### Phase 2 - Add Regression Tests Before Code Changes

Risk: low to medium. Tests may expose existing issues.

Required tests:

1. Label distribution contract:
   - counts for LONG, SHORT, NEUTRAL
   - counts by `event_flag` and `train_event_flag`
   - `neutral_reason` distribution
   - `path_outcome` distribution

2. Soft-label dependency test:
   - prove NEUTRAL hard labels map to near `soft_label=0.5`
   - prove directional hard labels drive `soft_label`

3. Leakage tests:
   - no `forward_return`
   - no `label_end_ts`
   - no `path_outcome`
   - no `bias_label`
   - no `*_fwd_*` supervision fields in model input

4. Chronology tests:
   - `ts_event` sorted
   - `label_end_ts >= ts_event`
   - train rows do not overlap validation label windows
   - embargo is applied

5. Backtest safety tests:
   - release mode rejects in-sample override flags
   - oracle forward-return mode cannot be used for release reporting

### Phase 3 - Import V19.2 as Isolated Code

Risk: medium. Import risk only if names collide.

Actions:

1. Place V19.2 modules under a new namespace:
   - Preferred: `/Users/abdallah/Downloads/QS_FINAL/modules/v19_2/`
   - Alternative: `/Users/abdallah/Downloads/QS_FINAL/research_v19_2/`

2. Do not overwrite:
   - `/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py`
   - `/Users/abdallah/Downloads/QS_FINAL/modules/dynamic_labels.py`
   - `/Users/abdallah/Downloads/QS_FINAL/train_v19.py`
   - `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py`

3. Add standalone unit tests for V19.2 modules before wiring:
   - timestamp order
   - backward-only joins
   - no future labels in features
   - deterministic output on a small sample

### Phase 4 - Run V19.2 Statistical Discovery as Read-Only Diagnostics

Risk: low to medium. No production feature change.

Actions:

1. Run edge scanner/statistical discovery on historical artifacts.
2. Use time-based windows only.
3. Require multiple-comparison controls such as permutation tests or FDR-style reporting.
4. Output diagnostic reports only; do not feed discoveries into training yet.

### Phase 5 - Add V19.2 Features Behind Config Flags

Risk: medium to high. Feature changes can introduce leakage or instability.

Actions:

1. Add optional `v192__*` feature columns behind a disabled-by-default config flag.
2. Preserve current V19 artifact schema.
3. Require feature audits:
   - missing rate
   - zero/constant rate
   - timestamp alignment
   - train/test drift
   - correlation with forward returns checked only in validation diagnostics
4. Keep all forward-looking V19.2 fields out of model input.

### Phase 6 - Label Migration as a New Mode, Not a Rewrite

Risk: high. Label generation is critical system logic.

Actions:

1. Add a new label mode, for example `label_mode='v19_2'`, only after tests exist.
2. Keep existing `label_mode='v19'` as default.
3. Compare old vs new labels on the same raw sample:
   - LONG/SHORT/NEUTRAL distribution
   - event rate
   - average horizon
   - TP/SL first-hit distribution
   - `neutral_reason`
   - post-cost expectancy per label
4. Target is not just fewer NEUTRAL rows. Target is causal, executable, positive expected value after spread, slippage, fees, and latency.

### Phase 7 - Redesign Soft Labels After Hard Labels Are Fixed

Risk: high if done before hard-label repair.

Actions:

1. Decouple soft-label probability from hard `bias_label` where possible.
2. Produce separate `P(long_win)`, `P(short_win)`, and calibrated event quality.
3. Do not force all NEUTRAL rows to `0.5` if the row contains asymmetric path evidence.
4. Validate calibration and expected value, not accuracy.

### Phase 8 - Train Old and New Systems Side by Side

Risk: medium.

Actions:

1. Produce two artifacts:
   - baseline: current V19
   - candidate: V19 + gated V19.2 changes
2. Compare:
   - event pool size
   - class balance
   - per-class precision/recall/F1
   - confusion matrix
   - calibration curves
   - walk-forward PnL after costs
   - drawdown
   - Sharpe/Sortino
   - turnover
   - average win/loss
   - hit rate

### Phase 9 - Walk-Forward and Release Gates

Risk: medium.

Actions:

1. Use `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py:373` `run_walkforward()`.
2. Run base and stress-cost backtests.
3. Reject candidates that only work under optimistic slippage/spread assumptions.
4. Require release gate reports from `/Users/abdallah/Downloads/QS_FINAL/modules/release_gates_v19.py`.

### Phase 10 - Cleanup Only After Candidate Passes

Risk: low if delayed until after tests and tagged baseline.

Actions:

1. Move diagnostics into a `tools/diagnostics/` area only after behavior is frozen and tests pass.
2. Mark duplicate modules as legacy in docs.
3. Do not delete duplicate files until:
   - import graph confirms unused
   - tests pass
   - one production artifact is rebuilt
   - one walk-forward run passes
   - git tag or archive exists

## Immediate Next Commands

Use these only after this audit is accepted:

```bash
python -m py_compile prepare_training_data.py prepare_day_trading.py train_v19.py predict_v19.py backtest_v19.py walkforward_v19.py
pytest tests/test_day_trading_labels.py tests/test_train_v19_training_modes.py tests/test_leakage_guards.py tests/test_backtest_oos_guard.py
```

For a specific existing feature artifact:

```bash
python -m tools.diagnostics.audit_v19_labels_features --data <FEATURE_ARTIFACT_OR_PARQUET> --out <AUDIT_OUTPUT_DIR>
```

## Bottom Line

Do not start by rewiring V19.2 into training. البداية الصحيحة: freeze current behavior, test label/leakage invariants, import V19.2 isolated, run it read-only, then migrate labels behind a new mode. The label imbalance problem is a label/event-definition problem first, not a CatBoost or deep-learning architecture problem.
