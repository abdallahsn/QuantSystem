# v20 Data Preparation Pipeline

Phase 2 makes `prepare_v20.py` a real artifact-producing pipeline for Databento
MBO + MBP-10 data. Phase 3 training/model refactors are intentionally out of
scope.

```text
raw Databento MBO + raw Databento MBP-10
-> schema validation
-> timestamp/duplicate/price/size cleaning with row-count reports
-> optional symbol and date filtering
-> MBP feature clock
-> backward-asof MBO trade-flow alignment
-> LOB, MLOFI, rolling causal, liquidity, and safe session features
-> cost-aware triple-barrier labels
-> label_end_ts / horizon_end_ts
-> features.parquet, labels.parquet, metadata.parquet
-> final/features_*.parquet for train_v19 compatibility
-> manifest.json and validation reports
```

## Outputs

`prepare_v20.py` writes:

- `features.parquet`
- `labels.parquet`
- `metadata.parquet`
- `metadata.json`
- `manifest.json`
- `artifact_manifest.json`
- `final/features_*.parquet`
- `data_validation_report.json`
- `feature_validation_report.json`
- `label_distribution_report.json`
- `leakage_precheck_report.json`
- `train_v19_compatibility_report.json`
- `label_calibration_report.json` and `label_calibration_grid.csv` when
  `label_calibration_v20.py` is run after artifact generation
- `prepare_v20_summary.json`

## Safety Notes

Features are causal by construction:

- MBP rows define the feature clock.
- MBO state is aligned with `merge_asof(direction="backward")`.
- MLOFI uses only current and previous MBP snapshots.
- Rolling features use pandas rolling windows over chronological rows only.
- Future rows are used only for labels, and `label_end_ts` records the final
  future timestamp touched by each label.

Rows dropped or rejected during cleaning are counted in
`data_validation_report.json`. Stale MBO alignment is reported through
`alignment.mbo_state_age_ms` and warnings such as `mbo_state_age_p95_gt_1s`.
Placeholder V19 compatibility columns remain in parquet artifacts for loader
compatibility, but they are marked as `compatibility_only_feature_columns` and
excluded from canonical v20 `feature_columns`.

## Commands

Real Databento readiness checks:

```bash
python3 prepare_v20.py \
  --mbo /workspace/QuantSystem/mbo2.csv \
  --mbp /workspace/QuantSystem/mbp2.csv \
  --output outputs_v20_real_1day \
  --symbol 6BH5 \
  --start 2025-01-15 \
  --end 2025-01-16 \
  --chunk_rows 100000 \
  --max_memory_gb 8 \
  --write_partitions
```

```bash
python3 prepare_v20.py \
  --mbo /workspace/QuantSystem/mbo2.csv \
  --mbp /workspace/QuantSystem/mbp2.csv \
  --output outputs_v20_real_7day \
  --symbol 6BH5 \
  --start 2025-01-15 \
  --end 2025-01-22 \
  --chunk_rows 100000 \
  --max_memory_gb 8 \
  --write_partitions
```

Use `--max_rows` as a guard, not as a sampler. Use `--sample_rows` for smoke
samples.

Dry run:

```bash
python3 prepare_v20.py \
  --mbo /data/6B/mbo.csv \
  --mbp /data/6B/mbp10.csv \
  --output outputs/v20_dry_run \
  --symbol 6BM6 \
  --dry_run \
  --chunk_rows 500000
```

1000-row sample:

```bash
python3 prepare_v20.py \
  --mbo /data/6B/mbo.csv \
  --mbp /data/6B/mbp10.csv \
  --output outputs/v20_sample_1000 \
  --symbol 6BM6 \
  --sample_rows 1000 \
  --horizon 50 \
  --tick_size 0.0001
```

Label calibration grid on the prepared sample:

```bash
python3 label_calibration_v20.py \
  --artifact outputs/v20_sample_1000 \
  --output outputs/v20_sample_1000 \
  --tick_size 0.0001 \
  --horizons 50,100,200 \
  --tp_mults 0.5,0.75,1.0,1.5 \
  --sl_mults 0.5,0.75,1.0 \
  --neutral_mults 0.1,0.2,0.3,0.45
```

One day:

```bash
python3 prepare_v20.py \
  --mbo /data/6B/mbo.csv \
  --mbp /data/6B/mbp10.csv \
  --output outputs/v20_1d \
  --symbol 6BM6 \
  --start 2026-01-05 \
  --end 2026-01-06 \
  --horizon 50 \
  --chunk_rows 500000
```

Seven days:

```bash
python3 prepare_v20.py \
  --mbo /data/6B/mbo.csv \
  --mbp /data/6B/mbp10.csv \
  --output outputs/v20_7d \
  --symbol 6BM6 \
  --start 2026-01-05 \
  --end 2026-01-12 \
  --horizon 50 \
  --chunk_rows 500000
```

Three months:

```bash
python3 prepare_v20.py \
  --mbo /data/6B/mbo.parquet \
  --mbp /data/6B/mbp10.parquet \
  --output outputs/v20_3m \
  --symbol 6BM6 \
  --start 2026-01-01 \
  --end 2026-04-01 \
  --horizon 50 \
  --chunk_rows 500000 \
  --rows_per_shard 250000 \
  --strict
```

Six months:

```bash
python3 prepare_v20.py \
  --mbo /data/6B/mbo.parquet \
  --mbp /data/6B/mbp10.parquet \
  --output outputs/v20_6m \
  --symbol 6BM6 \
  --start 2026-01-01 \
  --end 2026-07-01 \
  --horizon 50 \
  --chunk_rows 500000 \
  --rows_per_shard 250000 \
  --strict
```
