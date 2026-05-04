# QuantSystem V19 — Codex Instructions

## Project Goal
This is a quantitative AI trading production system. Your job is to review, improve, and harden the system with extreme care.

The main goal is NOT to make unrealistic profit claims. The goal is:
1. eliminate data leakage,
2. validate labels,
3. improve model generalization,
4. improve backtesting realism,
5. improve production reliability,
6. produce measurable, reproducible improvements.

## Critical Rules
- Never assume the model is correct without evidence.
- Never optimize only on test results.
- Never use future data in features, labels, scaling, filtering, or validation.
- Always check temporal order.
- Always separate train / validation / test chronologically.
- Always report uncertainty and risks.
- Any change must include verification steps.
- Prefer small safe patches over huge rewrites.
- Do not delete existing code without explaining why.

## Trading System Review Priorities

### 1. Data Integrity
Check:
- duplicated rows
- missing timestamps
- non-monotonic timestamps
- timezone issues
- symbol/contract consistency
- price/size abnormal values
- mismatched MBO/MBP merge logic
- incorrect resampling
- feature rows that use future events

### 2. Labeling Review
For event labels / V19 labels, verify:
- horizon logic
- TP / SL / neutral thresholds
- tick size correctness
- no lookahead leakage in features
- labels are computed only from future price path, while features use only current/past data
- class imbalance
- label stability across months/contracts

### 3. Validation
Use:
- chronological split only
- walk-forward validation
- purged / embargoed validation when overlapping horizons exist
- out-of-sample monthly testing
- per-contract and per-month metrics

Do not accept random train_test_split for time-series trading data.

### 4. Model Evaluation
Report:
- accuracy
- precision / recall / F1 per class
- confusion matrix
- ROC-AUC if suitable
- calibration
- profit factor
- max drawdown
- Sharpe-like metric
- average trade expectancy
- number of trades
- win rate
- fees/slippage sensitivity

ML metrics alone are not enough.

### 5. Backtesting Realism
Check:
- transaction fees
- spread
- slippage
- latency assumptions
- order fill assumptions
- no execution at impossible prices
- no using future candles/order book states
- position sizing
- stop loss / take profit implementation
- max daily loss / risk controls

### 6. Production Readiness
Check:
- reproducibility with random seeds
- logging
- config files
- model artifact versioning
- data schema validation
- crash recovery
- memory usage for huge CSV files
- GPU/CPU compatibility
- batch inference performance

## Required Output Format
For every task, respond with:

1. Executive summary
2. Critical issues found
3. Evidence from code/files
4. Risk level: Critical / High / Medium / Low
5. Exact recommended fixes
6. Patch or code changes
7. How to verify
8. Next improvement plan

## Definition of Done
A task is done only when:
- code runs without errors,
- no obvious leakage remains,
- validation is chronological,
- metrics are reproducible,
- changes are explained,
- verification command is provided.