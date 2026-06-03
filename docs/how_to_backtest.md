# How To Backtest

Current production backtesting remains `backtest_v19.py` until Phase 5.

v20 backtests must include:

- spread
- slippage
- fees
- latency
- fill-probability assumptions
- trade logs
- equity curve
- drawdown and risk metrics

The Phase 1 execution assumptions are defined in
`backtesting.execution_model.ExecutionAssumptions`.
