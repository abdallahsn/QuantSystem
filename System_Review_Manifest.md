# QuantSystem V19 - Codebase Audit & Improvement Manifest

## 1. System Context
- **Base Architecture:** Read the provided `README.md` to understand the pipeline.
- **Core Strategy:** High-frequency/Mid-frequency order book dynamics using MBO (Market-by-Order) and MBP10 data.
- **Current Pipeline:** 1. CatBoost + Regime Meta-features.
  2. OOF (Out-of-Fold) DeepLOB Visual Embeddings.
  3. MetaLearner LSTM.

## 2. Unseen Realities & Assumptions (User to fill this part)
- **Asset Class / Market:** [مثال: Crypto Perpetual Futures on Binance / US Equities]
- **Target Holding Period:** [مثال: 5 minutes to 2 hours]
- **Current Bottlenecks/Pains:** [مثال: الـ Drawdown عالي في الأسواق العرضية / هناك بطء في تدريب الـ LSTM / الموديل يربح في الباك تيست ويخسر في الـ Paper trading]
- **Latency & Slippage Assumptions:** [مثال: أفترض 5ms latency و 1 tick slippage في الباك تيست]

## 3. The Objective for Claude
I am not looking for standard code refactoring (like variable naming or PEP8). I am looking for a brutal Quantitative and Machine Learning audit. Your goal is to find fatal flaws in causal modeling, data leakage, and statistical robustness, and to propose cutting-edge ML/Quant improvements.