# How To Train

Phase 2 does not refactor training. Production training still uses
`train_v19.py`, but `prepare_v20.py` now writes V19-loadable
`final/features_*.parquet` shards plus `artifact_manifest.json`.

## Prepare Then Train

Check CatBoost before running the V19 sanity train:

```bash
python3 - <<'PY'
try:
    import catboost
    print("CatBoost OK", catboost.__version__)
except Exception as exc:
    print("CatBoost missing:", exc)
    print("Install with: python -m pip install catboost")
PY
```

Prepare one day:

```bash
python3 prepare_v20.py \
  --mbo /data/6B/mbo.csv \
  --mbp /data/6B/mbp10.csv \
  --output outputs/v20_1d \
  --symbol 6BM6 \
  --start 2026-01-05 \
  --end 2026-01-06 \
  --horizon 50 \
  --tick_size 0.0001
```

Sanity train with V19:

```bash
python3 train_v19.py \
  --data outputs/v20_1d \
  --output outputs/train_v19_from_v20_1d
```

V19 compatibility should be checked before training:

```bash
python3 - <<'PY'
import json
with open("outputs/v20_1d/train_v19_compatibility_report.json") as f:
    report = json.load(f)
print(report["passed"], report["loader_ok"], report["placeholder_zero_v19_features"])
PY
```

Calibrate label parameters before training if directional labels are too rare:

```bash
python3 label_calibration_v20.py \
  --artifact outputs/v20_1d \
  --output outputs/v20_1d \
  --tick_size 0.0001 \
  --horizons 50,100,200 \
  --tp_mults 0.5,0.75,1.0,1.5 \
  --sl_mults 0.5,0.75,1.0 \
  --neutral_mults 0.1,0.2,0.3,0.45
```

## Scale Commands

Dry run:

```bash
python3 prepare_v20.py --mbo /data/6B/mbo.csv --mbp /data/6B/mbp10.csv --output outputs/v20_dry --symbol 6BM6 --dry_run
```

1000-row sample:

```bash
python3 prepare_v20.py --mbo /data/6B/mbo.csv --mbp /data/6B/mbp10.csv --output outputs/v20_1000 --symbol 6BM6 --sample_rows 1000
```

Seven days:

```bash
python3 prepare_v20.py --mbo /data/6B/mbo.csv --mbp /data/6B/mbp10.csv --output outputs/v20_7d --symbol 6BM6 --start 2026-01-05 --end 2026-01-12 --horizon 50
```

Three months:

```bash
python3 prepare_v20.py --mbo /data/6B/mbo.parquet --mbp /data/6B/mbp10.parquet --output outputs/v20_3m --symbol 6BM6 --start 2026-01-01 --end 2026-04-01 --horizon 50 --chunk_rows 500000 --rows_per_shard 250000 --strict
```

Six months:

```bash
python3 prepare_v20.py --mbo /data/6B/mbo.parquet --mbp /data/6B/mbp10.parquet --output outputs/v20_6m --symbol 6BM6 --start 2026-01-01 --end 2026-07-01 --horizon 50 --chunk_rows 500000 --rows_per_shard 250000 --strict
```

## Warnings To Inspect

Always inspect these before training:

- `data_validation_report.json`: rejected rows, crossed/locked books, stale MBO alignment.
- `label_distribution_report.json`: collapsed labels, extreme neutral share, long/short imbalance.
- `leakage_precheck_report.json`: `label_end_ts >= ts_event` and purged split precheck.
- `train_v19_compatibility_report.json`: V19 loader result and placeholder feature list.

Placeholder V19 columns are explicit causal zeros for V19 heuristic features not
implemented in V20 Phase 2. They make `train_v19.py` loadable, but they are
excluded from canonical v20 `feature_columns` and listed as
`compatibility_only_feature_columns`.
