# QuantSystem v20 Architecture

QuantSystem v20 separates research logic from orchestration. The v19 system has
valuable code, but Graphify shows large god nodes such as `run_refinery()`,
`run_training_pipeline()`, `run_walkforward()`, and `run_causal_backtest()`.
The v20 design turns those into thin entry points over explicit packages.

## Layers

- `data_ingestion/`: Databento MBO/MBP schema contracts and chunked readers.
- `data_cleaning/`: timestamp, duplicate, price, and contract cleaning reports.
- `feature_engineering/`: LOB features, MLOFI, rolling causal features, tensors.
- `labeling/`: cost-aware triple-barrier labels and `label_end_ts`.
- `datasets/`: parquet shards, manifest metadata, split metadata.
- `models/`: CatBoost, XGBoost, DeepLOB, Transformer, and meta-model interfaces.
- `training/`: purged walk-forward splits, metrics, model registry.
- `backtesting/`: replay, execution assumptions, cost and fill models.
- `validation/`: data, feature, label, split, backtest, and schema checks.

## Rule

No feature may use rows after its decision timestamp. Labels may scan future
paths, but every row must carry `label_end_ts` so training splits can purge
overlapping outcomes.
