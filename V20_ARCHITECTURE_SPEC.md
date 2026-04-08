# QuantSystem V20 Architecture Spec

## Objective

V20 replaces the weakest architectural assumptions in V19 while preserving the
parts that still make quantitative sense:

- Keep the causal labeling, purged/embargoed OOF discipline, and CatBoost expert
- Replace the image-like DeepLOB branch with a microstructure-native LOB encoder
- Replace the LSTM meta-learner with a lightweight causal Transformer encoder
- Preserve train/inference parity by saving both OOF and live/final-model stage
  artifacts explicitly

V20 is designed for event-time classification on noisy MBO/MBP10 data, not for
generic multi-horizon forecasting. For that reason, we do **not** use TFT.


## Why V20

### Why not TFT

TFT is suboptimal here because:

- it is specialized for forecasting with known-future covariates and static
  covariates
- it is heavier than necessary for short-horizon event-time classification
- its variable-selection machinery adds variance without addressing the true V19
  bottlenecks
- it does not solve the LOB representation problem upstream

### Why not stay with LSTM

The V19 LSTM in [modules/meta_learner.py](/Users/abdallah/Downloads/QS_FINAL/modules/meta_learner.py)
is still serviceable for short windows, but it is no longer the best sequence
model once we:

- widen context from `50` to `128-256` events
- feed richer stage outputs than just probabilities and one-hot regimes
- want sparse long-range interactions instead of hidden-state compression

### Why not pure 2D DeepLOB

The V19 CNN in [modules/deeplob_cnn.py](/Users/abdallah/Downloads/QS_FINAL/modules/deeplob_cnn.py)
treats the book like a tiny image. That is only partially correct.

- price levels are not translation-invariant pixels
- bid/ask symmetry is structured, not generic spatial symmetry
- microstructure noise is bursty and event-driven, not texture-like
- level identity and distance-to-mid should be explicit embeddings, not implicit
  convolution side effects


## V20 Overview

### Stage 1: Expert Booster

Keep CatBoost as the first expert, but change what is exported downstream.

Inputs:

- handcrafted stat features
- optional regime features

Outputs per row:

- class logits, not probabilities only
- probabilities
- confidence margin
- predictive entropy
- optional leaf embedding projection
- regime posterior or soft assignment

Recommended downstream export vector:

- `3` class logits
- `3` class probs
- `1` margin
- `1` entropy
- `4` regime posteriors
- `8-16` leaf embedding dims

Target downstream dimension:

- `20-28` expert dims per event


### Stage 2: LOB Encoder

Replace the single image-style CNN with a two-part encoder:

1. `Level encoder`
2. `Temporal encoder`

#### 2.1 Level Encoder

For each snapshot, build tokens over price levels instead of a raw image.

Per-level token features:

- log bid/ask size
- size delta vs previous snapshot
- queue imbalance
- buy footprint at that level
- sell footprint at that level
- add/cancel/trade intensity near that level if available
- relative distance to mid in ticks
- side id embedding
- level id embedding

Token shape:

- input: `(T, 2L, F_level)`
- recommended: `L=10` or `20`, `F_level=8-16`

Level encoder options:

- default: 2 blocks of side-aware cross-level attention
- fallback: side-aware 1D convolutions across levels

Snapshot output:

- `snapshot_emb_dim = 32-64`

#### 2.2 Temporal Encoder

Encode the sequence of snapshot embeddings using causal dilated convolutions.

Recommended stack:

- residual dilated Conv1D blocks
- dilations: `1, 2, 4, 8, 16, 32`
- GLU or gated residual activation
- pre-norm + dropout
- optional local causal attention block every 2 residual blocks

Why this is preferred:

- multi-scale receptive field without recurrent state bottleneck
- lower variance and better stability than a full Transformer on noisy LOB data
- strictly causal, latency-friendly, and robust under bursty event arrival

LOB encoder output per event:

- `lob_emb_dim = 24-32`


### Stage 3: Fusion Encoder

Replace the V19 LSTM meta-learner with a small causal Transformer encoder over
event tokens.

Per-event fusion token:

- scaled handcrafted stats
- Stage-1 expert vector
- Stage-2 LOB embedding
- `delta_t` / inter-arrival-time embedding
- optional session metadata embedding if allowed causally

Recommended token width:

- `token_dim = 96`

Recommended model:

- pre-norm causal Transformer encoder
- `2-4` layers
- `4` heads
- FFN multiplier `4x`
- local attention window `32-64`
- relative event-position bias

Outputs:

- bias classification head
- confidence / abstention head
- optional expected-return head
- optional calibrated risk bucket head

Why not a full global Transformer:

- local causal attention is usually enough at this horizon
- lower latency and lower variance
- cheaper walk-forward and OOF training


## Recommended Data Shapes

### Event Context

- stat sequence length: `128`
- optional long context experiment: `256`

### LOB Context

- snapshot sequence length: `128`
- price levels per side: `10` initially
- channels/features per level: `8-16`

### Embedding Sizes

- stage-1 expert vector: `20-28`
- stage-2 LOB vector: `24-32`
- fusion token: `96`


## Train/Inference Contract

V20 must keep strict parity between training and replay/live paths.

Saved artifacts must include:

- `expert_features_oof_v20.npy`
- `expert_features_live_v20.npy`
- `lob_embeddings_oof_v20.npy`
- `lob_embeddings_live_v20.npy`
- `fusion_model_v20.keras`
- `catboost_expert_v20.cbm`
- `regime_classifier_v20.pkl`
- `scaler_params_v20.json`
- `feature_schema_v20.json`
- `manifest_v20.json`

Rule:

- stage-3 validation uses OOF stage artifacts
- backtest may choose either:
  - OOF artifacts for training-sample honesty
  - live/final-model artifacts for deployment-style replay


## Proposed Module Layout

### Keep As-Is Or Lightly Reuse

- [modules/labels_v19.py](/Users/abdallah/Downloads/QS_FINAL/modules/labels_v19.py)
  - reuse logic, rename to `labels_v20.py` only if label semantics change
- [modules/purging_embargo.py](/Users/abdallah/Downloads/QS_FINAL/modules/purging_embargo.py)
  - keep and reuse
- [modules/regime_classifier.py](/Users/abdallah/Downloads/QS_FINAL/modules/regime_classifier.py)
  - keep, but export soft/posterior features where possible
- [modules/manifest_v19.py](/Users/abdallah/Downloads/QS_FINAL/modules/manifest_v19.py)
  - clone to `manifest_v20.py`
- [modules/release_gates_v19.py](/Users/abdallah/Downloads/QS_FINAL/modules/release_gates_v19.py)
  - clone to `release_gates_v20.py`

### Replace Or Split

- [modules/deeplob_cnn.py](/Users/abdallah/Downloads/QS_FINAL/modules/deeplob_cnn.py)
  - split into:
    - `modules/lob_tokenizer_v20.py`
    - `modules/lob_level_encoder_v20.py`
    - `modules/lob_temporal_encoder_v20.py`
    - `modules/lob_encoder_v20.py`

- [modules/meta_learner.py](/Users/abdallah/Downloads/QS_FINAL/modules/meta_learner.py)
  - replace with:
    - `modules/fusion_transformer_v20.py`

- [modules/feature_factory_v19.py](/Users/abdallah/Downloads/QS_FINAL/modules/feature_factory_v19.py)
  - clone and extend into:
    - `modules/feature_factory_v20.py`

### New Training / Inference Entry Points

- [train_v19.py](/Users/abdallah/Downloads/QS_FINAL/train_v19.py)
  - clone into `train_v20.py`
- [predict_v19.py](/Users/abdallah/Downloads/QS_FINAL/predict_v19.py)
  - clone into `predict_v20.py`
- [walkforward_v19.py](/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py)
  - clone into `walkforward_v20.py`


## Direct Mapping From V19 To V20

### V19 `prepare_training_data.py`

Current responsibility:

- build stat features
- labels
- LOB tensors

V20 change:

- continue building causal stat features
- additionally emit a richer LOB event stream artifact for the level encoder
- save per-snapshot level features, not only image tensors

New artifacts:

- `lob_level_tokens_v20.npy`
- `lob_level_timestamps_v20.npy`

### V19 `modules/deeplob_cnn.py`

V20 replacement:

- tokenizer converts MBP/MBO into level tokens with explicit side/level/distance
- level encoder models cross-level interactions within each snapshot
- temporal encoder models event-time dynamics causally

### V19 `train_v19.py`

V20 stages:

1. fit expert booster with purged OOF
2. fit LOB encoder with purged OOF embeddings
3. fit fusion Transformer on OOF stage outputs
4. calibrate decision thresholds on final validation split

### V19 `predict_v19.py`

V20 runtime:

1. build stat row
2. compute CatBoost expert vector
3. compute LOB embedding from live tokenizer + encoder
4. concatenate into fusion token
5. append to causal token buffer
6. run fusion Transformer decision head


## Conceptual Model Graph

```text
MBO / MBP10
  -> LOB tokenizer
  -> level encoder
  -> temporal TCN / local attention
  -> lob_emb_t

Handcrafted features
  -> CatBoost expert
  -> logits / probs / entropy / margin / leaf_emb
  -> expert_emb_t

Fusion token u_t = [stat_t, expert_emb_t, lob_emb_t, dt_emb_t]
  -> causal Transformer encoder
  -> bias head
  -> confidence head
  -> optional return head
```


## Training Recipe

### Stage 1: Expert

- purged expanding-window OOF
- inner time-valid split for early stopping
- save both OOF and live/final expert outputs

### Stage 2: LOB Encoder

Primary training objective:

- supervised auxiliary target aligned strictly to tensor time

Optional auxiliary objectives:

- signed short-horizon return bucket
- imbalance bucket
- next-event volatility bucket

Recommended loss:

- multi-task classification/regression

### Stage 3: Fusion

- consume OOF stage-1 and stage-2 outputs only
- causal split before sequence construction
- no overlap across train/validation sequences
- confidence head trained jointly with bias head


## Recommended Hyperparameters

### First Production Candidate

- `stat_seq_len = 128`
- `lob_seq_len = 128`
- `levels_per_side = 10`
- `snapshot_emb_dim = 48`
- `lob_emb_dim = 32`
- `expert_emb_dim = 24`
- `fusion_token_dim = 96`
- `fusion_layers = 3`
- `fusion_heads = 4`
- `attention_window = 32`
- `dropout = 0.10`

### Ablation Order

1. V19 fixed stack baseline
2. V19 CatBoost + TCN fusion only
3. V20 LOB encoder + LSTM fusion
4. V20 LOB encoder + Transformer fusion
5. Add leaf embeddings
6. Add local attention on top of dilated TCN


## Implementation Plan

### Phase 1

- create `feature_factory_v20.py`
- create `fusion_transformer_v20.py`
- clone `train_v19.py` into `train_v20.py`
- keep Stage 2 temporarily on existing LOB tensors
- replace only stage-3 LSTM with causal Transformer

This isolates whether the fusion model change alone adds value.

### Phase 2

- build `lob_tokenizer_v20.py`
- build `lob_level_encoder_v20.py`
- build `lob_temporal_encoder_v20.py`
- replace DeepLOB branch fully

### Phase 3

- add CatBoost leaf embeddings
- add multi-task auxiliary heads
- add calibration / abstention tuning


## Acceptance Criteria

V20 should only be considered better than V19 if it improves:

- walk-forward trade accuracy
- realized PnL after costs
- calibration of confidence vs realized win rate
- stability across folds
- coverage-adjusted performance

It is **not** enough to improve offline classification accuracy alone.


## Bottom Line

The V20 target architecture is:

- `CatBoost expert`
- `side-aware level encoder + causal dilated temporal LOB encoder`
- `small causal Transformer fusion model`

This is the correct direction for your codebase because it matches the
microstructure geometry more closely than image CNNs and scales better than
LSTM once the event window and upstream representation become richer.
