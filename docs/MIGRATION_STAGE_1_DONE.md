# Migration Stage 1 Done

Repository: `/Users/abdallah/Downloads/QS_FINAL`

Date: `2026-05-21`

Scope completed: structural cleanup only. No trading logic, label logic, model logic, feature logic, backtest logic, or deployment logic was intentionally changed.

## Summary

Stage 1 moved one-off diagnostics, audits, smoke checks, and visualization/report helpers from the repository root into:

`/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/`

The active root entry points remain in place:

- `/Users/abdallah/Downloads/QS_FINAL/stage1_refinery.py`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_training_data.py`
- `/Users/abdallah/Downloads/QS_FINAL/prepare_day_trading.py`
- `/Users/abdallah/Downloads/QS_FINAL/train_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/stage2_catboost.py`
- `/Users/abdallah/Downloads/QS_FINAL/stage3_train.py`
- `/Users/abdallah/Downloads/QS_FINAL/predict_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/backtest_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/raw_backtest_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/walkforward_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/shadow_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/paper_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/monitor_v19.py`
- `/Users/abdallah/Downloads/QS_FINAL/readiness_v19.py`

## Created Package Directories

Added lightweight `__init__.py` files to the target package namespaces:

- `/Users/abdallah/Downloads/QS_FINAL/core/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/simulators/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/context/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/discovery/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/dl_pipeline/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/training/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/deployment/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/tools/__init__.py`
- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/__init__.py`

## Moved Files

| Old path | New path |
|---|---|
| `/Users/abdallah/Downloads/QS_FINAL/analyze_thresholds.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/analyze_thresholds.py` |
| `/Users/abdallah/Downloads/QS_FINAL/audit_mbo_mbp_data.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/audit_mbo_mbp_data.py` |
| `/Users/abdallah/Downloads/QS_FINAL/audit_v19_labels_features.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/audit_v19_labels_features.py` |
| `/Users/abdallah/Downloads/QS_FINAL/compare_v19_experiments.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/compare_v19_experiments.py` |
| `/Users/abdallah/Downloads/QS_FINAL/deep_validation.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/deep_validation.py` |
| `/Users/abdallah/Downloads/QS_FINAL/deeplob_refinery_smoke.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/deeplob_refinery_smoke.py` |
| `/Users/abdallah/Downloads/QS_FINAL/feature_intelligence_report.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/feature_intelligence_report.py` |
| `/Users/abdallah/Downloads/QS_FINAL/find_training_artifacts.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/find_training_artifacts.py` |
| `/Users/abdallah/Downloads/QS_FINAL/label_visualizer_5m.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/label_visualizer_5m.py` |
| `/Users/abdallah/Downloads/QS_FINAL/output_report.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/output_report.py` |
| `/Users/abdallah/Downloads/QS_FINAL/pipeline_feature_diagnostic.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/pipeline_feature_diagnostic.py` |
| `/Users/abdallah/Downloads/QS_FINAL/plot_5m_catboost.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_5m_catboost.py` |
| `/Users/abdallah/Downloads/QS_FINAL/plot_best_soft_label_lob_heatmap.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_best_soft_label_lob_heatmap.py` |
| `/Users/abdallah/Downloads/QS_FINAL/plot_refinery_compare_5m.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_refinery_compare_5m.py` |
| `/Users/abdallah/Downloads/QS_FINAL/plot_v19_power_dashboard.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_v19_power_dashboard.py` |
| `/Users/abdallah/Downloads/QS_FINAL/server_sanity_check.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/server_sanity_check.py` |
| `/Users/abdallah/Downloads/QS_FINAL/smoke_day_trading_hardening.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/smoke_day_trading_hardening.py` |
| `/Users/abdallah/Downloads/QS_FINAL/stage1_signal_sanity.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/stage1_signal_sanity.py` |
| `/Users/abdallah/Downloads/QS_FINAL/verify_day_trading_dataset.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/verify_day_trading_dataset.py` |

## Import and Reference Repairs

Only structural import/reference repairs were made:

- `/Users/abdallah/Downloads/QS_FINAL/clean_mbo_mbp_data.py`
  - Now imports audit helpers from `tools.diagnostics.audit_mbo_mbp_data`.
  - Generated verification command now uses `python -m tools.diagnostics.audit_mbo_mbp_data`.

- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/stage1_signal_sanity.py`
  - Now imports `audit_v19_labels_features` through `tools.diagnostics`.

- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/smoke_day_trading_hardening.py`
  - Now imports `LABEL_DERIVED_FEATURES` from `tools.diagnostics.verify_day_trading_dataset`.

- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_v19_power_dashboard.py`
  - Now imports heatmap helpers from `tools.diagnostics.plot_best_soft_label_lob_heatmap`.

- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/server_sanity_check.py`
  - Repo root resolution was updated after the file move so `requirements.txt` is still found at the project root.

- `/Users/abdallah/Downloads/QS_FINAL/tests/test_audit_v19_labels_features.py`
- `/Users/abdallah/Downloads/QS_FINAL/tests/test_day_trading_pipeline_features.py`
- `/Users/abdallah/Downloads/QS_FINAL/tests/test_stage1_signal_sanity.py`
  - Test imports now reference `tools.diagnostics`.

- `/Users/abdallah/Downloads/QS_FINAL/install_tf_gpu_cu12.sh`
  - Now calls `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/server_sanity_check.py`.

- `/Users/abdallah/Downloads/QS_FINAL/README.md`, `/Users/abdallah/Downloads/QS_FINAL/commands`, and `/Users/abdallah/Downloads/QS_FINAL/convert_rich_csv_to_parquet.py`
  - Updated direct diagnostic command references to `python -m tools.diagnostics.<tool>`.

## Verification

### Import Check

Command:

```bash
QUANTSYSTEM_SKIP_HEAVY_ML=1 python - <<'PY'
import importlib

mods = [
    "prepare_training_data",
    "prepare_day_trading",
    "train_v19",
    "predict_v19",
    "backtest_v19",
    "raw_backtest_v19",
    "walkforward_v19",
    "shadow_v19",
    "paper_v19",
    "clean_mbo_mbp_data",
    "tools.diagnostics.audit_mbo_mbp_data",
    "tools.diagnostics.audit_v19_labels_features",
    "tools.diagnostics.feature_intelligence_report",
    "tools.diagnostics.stage1_signal_sanity",
    "tools.diagnostics.verify_day_trading_dataset",
    "tools.diagnostics.smoke_day_trading_hardening",
]

for name in mods:
    importlib.import_module(name)
    print(f"OK {name}")
PY
```

Result: passed. TensorFlow-related optional warnings appeared because TensorFlow is not installed in this environment; imports still succeeded.

### Focused Test Check

Command:

```bash
QUANTSYSTEM_SKIP_HEAVY_ML=1 python -m pytest \
  tests/test_audit_v19_labels_features.py \
  tests/test_stage1_signal_sanity.py \
  tests/test_day_trading_pipeline_features.py \
  tests/test_clean_mbo_mbp_data.py
```

Result: `12 passed, 27 warnings`.

### Environment Note

A first attempt with bare `pytest` used `/Library/Frameworks/Python.framework/Versions/3.8/bin/pytest` and failed during collection on pre-existing environment incompatibilities:

- Python 3.8 cannot parse/evaluate some existing `float | None` annotations.
- The installed sklearn/NumPy combination under that launcher hits deprecated `np.float`.

The same focused tests passed under the repository's active `python` interpreter:

- `python`: `/Users/abdallah/anaconda3/bin/python`
- version: `Python 3.10.9`

## Safety Statement

No files were deleted. The root diagnostics were moved into `tools/diagnostics/`. Active trading, labeling, feature generation, model training, prediction, backtest, walk-forward, paper, and shadow logic remain in their existing root/module locations.
