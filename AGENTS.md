# QuantSystem Instructions

This repository is a production-grade quantitative AI trading system.

## Core rules
- Never introduce data leakage.
- Never use random splits for time-series market data.
- Preserve chronological order in all training and evaluation.
- Prefer walk-forward validation.
- Treat label generation as critical system logic.
- Validate all market data before training.

## Data checks
Before training or feature generation, check:
- timestamp ordering
- duplicate rows
- zero or invalid prices
- crossed bid/ask
- missing bid/ask levels
- abnormal spreads
- symbol and contract consistency
- gaps between sessions

## Modeling
Primary models may include:
- CatBoost
- XGBoost/LightGBM only when justified
- LSTM / DeepLOB-style models
- Autoencoder embeddings
- Meta-learner / ensemble layer

Evaluation must include:
- confusion matrix
- per-class precision/recall/F1
- calibration
- walk-forward performance
- realistic backtest with spread, slippage, fees, and latency assumptions

## Coding
- Use clear Python with type hints where useful.
- Prefer config-driven parameters.
- Add logging.
- Avoid large rewrites.
- Add sanity-check scripts for any core data or labeling change.