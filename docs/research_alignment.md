# Research Alignment

## MLOFI

`raw/arxiv_1907_06230.md` supports multi-level order-flow imbalance. v20 adds
`feature_engineering.ofi.compute_mlofi()` to compute per-level OFI from current
and previous MBP snapshots only. `prepare_v20.py` now writes `mlofi_00` through
`mlofi_09`, `mlofi_sum`, and `mlofi_top3` on the MBP feature clock.

## DeepLOB and BDLOB

DeepLOB-style tensors remain useful, but uncertainty from `raw/arxiv_1811_10041.md`
should be added before using neural outputs for position sizing. Phase 1 keeps
model interfaces only.

## Fill Probability

`raw/arxiv_2306_05479.md` shows fills are not guaranteed. v20 adds an explicit
fill-probability approximation in `backtesting.execution_model` as the starting
point for stricter execution realism.

## LOBFrame and Tradability

`raw/arxiv_2403_09267.md` warns that ML metrics do not imply executable trades.
v20 release gates include cost, latency, fill model, and risk requirements.

## TLOB and Spread-Aware Labels

`raw/arxiv_2502_15757.md` supports spread/cost-aware trend definitions. v20
triple-barrier labels include spread and round-trip cost floors.

## Phase 2 Preparation Choices

`prepare_v20.py` uses MBP-10 as the tabular feature clock because the book
snapshot defines the state the model observes. MBO trade-flow state is aligned
with a backward as-of join. This is weaker than full order-by-order queue
reconstruction, but it is causal, auditable, and safe for Phase 2 artifact
generation.

Labels follow a cost-aware triple-barrier design:

- Upper/lower barriers scale with causal volatility.
- Barrier floors include round-trip cost and spread.
- Timeout labels can be directional but non-tradeable.
- `label_end_ts` and `horizon_end_ts` mark the future window used for labeling.

This means ML accuracy alone is not the objective. The artifact exposes
`tradeability_label` and `train_event_flag` so later training/backtesting can
separate directional drift from executable opportunities.

## Known Research Gaps For Phase 3

- Full MBO queue reconstruction and fill probability calibration are not yet
  part of `prepare_v20.py`.
- V19 compatibility columns include explicit zero placeholders for legacy
  heuristic features that are not implemented in V20 Phase 2.
- Transformer/DeepLOB tensor artifacts are not generated in this phase; the
  current output is a tabular CatBoost/XGBoost-compatible artifact.
