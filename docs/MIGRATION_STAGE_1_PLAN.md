# Migration Stage 1 Plan

Repository: `/Users/abdallah/Downloads/QS_FINAL`

Date: `2026-05-21`

Scope: structural cleanup only. No model logic, label logic, feature logic, training behavior, backtest behavior, or deployment behavior will be changed.

## Objective

Move one-off diagnostics and exploratory/reporting scripts out of the repository root into:

`/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/`

Also create the target package directories requested for later stages:

- `/Users/abdallah/Downloads/QS_FINAL/core/`
- `/Users/abdallah/Downloads/QS_FINAL/simulators/`
- `/Users/abdallah/Downloads/QS_FINAL/context/`
- `/Users/abdallah/Downloads/QS_FINAL/discovery/`
- `/Users/abdallah/Downloads/QS_FINAL/dl_pipeline/`
- `/Users/abdallah/Downloads/QS_FINAL/training/`
- `/Users/abdallah/Downloads/QS_FINAL/deployment/`
- `/Users/abdallah/Downloads/QS_FINAL/tools/`
- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/`

## Files Planned for Move

These files are diagnostics, audits, smoke checks, visualization/report helpers, or exploratory scripts. They are not the active trading/training/deployment entry points.

| Old path | New path | Reason |
|---|---|---|
| `/Users/abdallah/Downloads/QS_FINAL/analyze_thresholds.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/analyze_thresholds.py` | Exploratory threshold diagnostics. |
| `/Users/abdallah/Downloads/QS_FINAL/audit_mbo_mbp_data.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/audit_mbo_mbp_data.py` | Raw market-data audit helper. |
| `/Users/abdallah/Downloads/QS_FINAL/audit_v19_labels_features.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/audit_v19_labels_features.py` | Read-only label/feature audit. |
| `/Users/abdallah/Downloads/QS_FINAL/compare_v19_experiments.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/compare_v19_experiments.py` | Experiment comparison report. |
| `/Users/abdallah/Downloads/QS_FINAL/deep_validation.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/deep_validation.py` | Scientific validation script. |
| `/Users/abdallah/Downloads/QS_FINAL/deeplob_refinery_smoke.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/deeplob_refinery_smoke.py` | DeepLOB refinery smoke check. |
| `/Users/abdallah/Downloads/QS_FINAL/feature_intelligence_report.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/feature_intelligence_report.py` | Feature intelligence diagnostics. |
| `/Users/abdallah/Downloads/QS_FINAL/find_training_artifacts.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/find_training_artifacts.py` | Artifact locator utility. |
| `/Users/abdallah/Downloads/QS_FINAL/label_visualizer_5m.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/label_visualizer_5m.py` | Label visualization report. |
| `/Users/abdallah/Downloads/QS_FINAL/output_report.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/output_report.py` | HTML signal report helper. |
| `/Users/abdallah/Downloads/QS_FINAL/pipeline_feature_diagnostic.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/pipeline_feature_diagnostic.py` | Stage-1 feature diagnostics. |
| `/Users/abdallah/Downloads/QS_FINAL/plot_5m_catboost.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_5m_catboost.py` | Standalone CatBoost dashboard. |
| `/Users/abdallah/Downloads/QS_FINAL/plot_best_soft_label_lob_heatmap.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_best_soft_label_lob_heatmap.py` | LOB/soft-label visualization. |
| `/Users/abdallah/Downloads/QS_FINAL/plot_refinery_compare_5m.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_refinery_compare_5m.py` | Raw-vs-refinery comparison dashboard. |
| `/Users/abdallah/Downloads/QS_FINAL/plot_v19_power_dashboard.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_v19_power_dashboard.py` | Unified diagnostic dashboard. |
| `/Users/abdallah/Downloads/QS_FINAL/server_sanity_check.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/server_sanity_check.py` | Environment/GPU sanity check. |
| `/Users/abdallah/Downloads/QS_FINAL/smoke_day_trading_hardening.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/smoke_day_trading_hardening.py` | Day-trading smoke checks. |
| `/Users/abdallah/Downloads/QS_FINAL/stage1_signal_sanity.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/stage1_signal_sanity.py` | Stage-1 signal diagnostics. |
| `/Users/abdallah/Downloads/QS_FINAL/verify_day_trading_dataset.py` | `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/verify_day_trading_dataset.py` | Day-trading dataset verification. |

## Files Not Planned for Move in Stage 1

These remain in the root because they are active entry points, compatibility shims, or data-preparation utilities that may be called directly by existing workflows:

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
- `/Users/abdallah/Downloads/QS_FINAL/build_lean_model.py`
- `/Users/abdallah/Downloads/QS_FINAL/clean_mbo_mbp_data.py`
- `/Users/abdallah/Downloads/QS_FINAL/convert_rich_csv_to_parquet.py`
- `/Users/abdallah/Downloads/QS_FINAL/extract_quarterly_contract.py`
- `/Users/abdallah/Downloads/QS_FINAL/live_predictor.py`
- `/Users/abdallah/Downloads/QS_FINAL/online_learning.py`
- `/Users/abdallah/Downloads/QS_FINAL/regime_config.py`
- `/Users/abdallah/Downloads/QS_FINAL/patch_backtest_v19_daytrade.py`

## Required Import/Reference Repairs

Only import/reference repairs caused by the file moves will be made:

- `/Users/abdallah/Downloads/QS_FINAL/clean_mbo_mbp_data.py`
  - Import audit helpers from `tools.diagnostics.audit_mbo_mbp_data`.
  - Update generated audit command string to use `python -m tools.diagnostics.audit_mbo_mbp_data`.

- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/stage1_signal_sanity.py`
  - Import `audit_v19_labels_features` from `tools.diagnostics`.

- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/smoke_day_trading_hardening.py`
  - Import `LABEL_DERIVED_FEATURES` from `tools.diagnostics.verify_day_trading_dataset`.

- `/Users/abdallah/Downloads/QS_FINAL/tools/diagnostics/plot_v19_power_dashboard.py`
  - Import heatmap helpers from `tools.diagnostics.plot_best_soft_label_lob_heatmap`.

- Tests that import moved diagnostics:
  - `/Users/abdallah/Downloads/QS_FINAL/tests/test_audit_v19_labels_features.py`
  - `/Users/abdallah/Downloads/QS_FINAL/tests/test_day_trading_pipeline_features.py`
  - `/Users/abdallah/Downloads/QS_FINAL/tests/test_stage1_signal_sanity.py`

- `/Users/abdallah/Downloads/QS_FINAL/install_tf_gpu_cu12.sh`
  - Update the environment sanity check path to `tools/diagnostics/server_sanity_check.py`.

## Safety Checks

After the move, run a basic import check for key active modules and moved diagnostics. If imports fail, stop and repair structural imports before doing anything else.

Planned check:

```bash
python - <<'PY'
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
]

for name in mods:
    importlib.import_module(name)
    print(f"OK {name}")
PY
```

## Non-Goals

- No label changes.
- No feature engineering changes.
- No model changes.
- No backtest changes.
- No deletion of legacy files.
- No broad README rewrite in this stage.
