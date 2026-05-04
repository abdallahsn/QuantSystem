"""
stage2_catboost.py - Standalone entrypoint for CatBoost/XGBoost/regime stage
"""

from __future__ import annotations

import argparse
import json
import os

try:
    import catboost  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "❌ CatBoost غير مثبّت في هذه البيئة.\n"
        "نفّذ أولًا:\n"
        "pip install -r requirements.txt"
    ) from e

try:
    import xgboost  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "❌ XGBoost غير مثبّت في هذه البيئة.\n"
        "نفّذ أولًا:\n"
        "pip install -r requirements.txt"
    ) from e

from modules.config_v19 import load_v19_config
from modules.catboost_5m_report import generate_catboost_5m_report
from train_v19 import run_training_pipeline


def _load_optional_json(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _print_directional_event_diagnostics(output_dir: str) -> None:
    event_view = _load_optional_json(os.path.join(output_dir, 'event_training_view.json')) or {}
    stage1_metrics = _load_optional_json(os.path.join(output_dir, 'stage1_v19_metrics.json')) or {}
    calibration = _load_optional_json(os.path.join(output_dir, 'calibration_report.json')) or {}

    if not event_view and not stage1_metrics and not calibration:
        return

    print("\n🔎 Stage 2 Diagnostics")
    print(
        "  ℹ️ Current Stage 2 CatBoost does NOT use modules/catboost_brain.py. "
        "It trains via train_v19.py on directional event rows only."
    )

    rows_full = int(event_view.get('rows_full', 0) or 0)
    rows_event = int(event_view.get('rows_event_directional', 0) or 0)
    event_rate = float(event_view.get('event_rate_full', 0.0) or 0.0)
    raw_event_rate = float(event_view.get('raw_event_rate_full', 0.0) or 0.0)
    bias_counts = event_view.get('bias_counts', {}) or {}
    fallback_reason = event_view.get('fallback_reason')
    score_fit_rows = int(event_view.get('score_fit_rows', 0) or 0)

    if rows_full > 0:
        print(
            "  Directional Event View: "
            f"{rows_event:,}/{rows_full:,} rows ({event_rate:.1%}) "
            f"| raw_event={raw_event_rate:.1%} "
            f"| bias={bias_counts} "
            f"| score_fit_rows={score_fit_rows:,}"
        )
        if fallback_reason:
            print(f"  ⚠️ Event view fallback: {fallback_reason}")
        if event_rate < 0.05:
            print(
                "  ⚠️ Very low directional event rate. "
                "Few training rows can make CatBoost look weak even if labels are causal."
            )
        long_n = int(bias_counts.get('0', bias_counts.get(0, 0)) or 0)
        short_n = int(bias_counts.get('1', bias_counts.get(1, 0)) or 0)
        directional_total = max(long_n + short_n, 1)
        imbalance = abs(long_n - short_n) / directional_total
        if imbalance > 0.35:
            print(
                "  ⚠️ Directional class imbalance is high: "
                f"LONG={long_n:,} SHORT={short_n:,} | imbalance={imbalance:.1%}"
            )

    cb_cov = float(stage1_metrics.get('catboost_coverage_ratio', 0.0) or 0.0)
    xgb_cov = float(stage1_metrics.get('xgboost_coverage_ratio', 0.0) or 0.0)
    regime_cov = float(stage1_metrics.get('regime_coverage_ratio', 0.0) or 0.0)
    policy_cov = float(stage1_metrics.get('decision_policy_coverage_ratio', 0.0) or 0.0)
    regime_source = stage1_metrics.get('regime_source')
    if stage1_metrics:
        print(
            "  Stage1 Coverage: "
            f"catboost={cb_cov:.1%} | xgboost={xgb_cov:.1%} | regime={regime_cov:.1%} | policy={policy_cov:.1%}"
        )
        if regime_source:
            print(f"  Regime Source: {regime_source}")
        if bool((stage1_metrics.get('coverage_warning', {}) or {}).get('catboost', False)):
            print("  ⚠️ CatBoost OOF coverage is low; priors/backfills may dominate parts of the meta surface.")

    cb_cal = (calibration.get('final_catboost_holdout_calibration', {}) or {})
    xgb_cal = (calibration.get('final_xgboost_holdout_calibration', {}) or {})
    if calibration:
        print(
            "  Holdout Calibration: "
            f"catboost_enabled={bool(cb_cal.get('enabled', False))} | "
            f"xgboost_enabled={bool(xgb_cal.get('enabled', False))}"
        )


def main():
    defaults = load_v19_config().get('training', {})
    p = argparse.ArgumentParser(description='QuantSystem V19 - Stage 2 CatBoost + XGBoost')
    p.add_argument('--csv', required=True, help='training_features_ready.csv from stage 1')
    p.add_argument('--lob', default=None, help='optional lob_tensors.npy')
    p.add_argument('--lob_ts', default=None, help='optional lob_tensor_timestamps.npy')
    p.add_argument('--output', default=defaults.get('output_dir', 'outputs_v19'))
    p.add_argument('--n_folds', type=int, default=int(defaults.get('n_folds', 6)))
    p.add_argument('--test_size', type=float, default=float(defaults.get('test_size', 0.10)))
    p.add_argument('--embargo_pct', type=float, default=float(defaults.get('embargo_pct', 0.02)))
    p.add_argument('--min_train_pct', type=float, default=float(defaults.get('min_train_pct', 0.20)))
    p.add_argument('--train_frac', type=float, default=float(defaults.get('train_frac', 0.80)))
    p.add_argument('--min_seq_coverage', type=float, default=float(defaults.get('min_seq_coverage', 0.80)))
    p.add_argument('--catboost_device', default='auto', choices=['auto', 'cpu', 'gpu'])
    p.add_argument('--training_mode', default=str(defaults.get('mode', 'event_binary')))
    p.add_argument('--quality_weight_strong', type=float, default=float(defaults.get('quality_weight_strong', 2.0)))
    p.add_argument('--quality_weight_weak', type=float, default=float(defaults.get('quality_weight_weak', 1.0)))
    p.add_argument('--split_time', default=None, help='explicit holdout start timestamp (UTC/parsible string)')
    p.add_argument('--train_days', type=float, default=None, help='limit training window to N days immediately before split_time')
    p.add_argument('--backtest_days', type=float, default=None, help='limit holdout/backtest window to the last N days before window_end or dataset end')
    p.add_argument('--window_end', default=None, help='exclusive end timestamp for the train/backtest window')
    p.add_argument('--config', default=None, help='optional config file')
    args = p.parse_args()

    cfg = load_v19_config(args.config)
    run_training_pipeline(
        csv_path=args.csv,
        output_dir=args.output,
        lob_path=args.lob,
        lob_ts_path=args.lob_ts,
        n_folds=args.n_folds,
        test_size=args.test_size,
        embargo_pct=args.embargo_pct,
        min_train_pct=args.min_train_pct,
        train_frac=args.train_frac,
        min_seq_coverage=args.min_seq_coverage,
        catboost_device=args.catboost_device,
        training_mode=args.training_mode,
        quality_weight_strong=args.quality_weight_strong,
        quality_weight_weak=args.quality_weight_weak,
        split_time=args.split_time,
        train_days=args.train_days,
        backtest_days=args.backtest_days,
        window_end=args.window_end,
        phase='catboost',
        config_snapshot=cfg,
    )
    _print_directional_event_diagnostics(args.output)

    try:
        summary = generate_catboost_5m_report(
            csv_path=args.csv,
            models_dir=args.output,
            output_dir=args.output,
            freq='5min',
            report_name='catboost_5m',
        )
        print("\n📊 5m CatBoost dashboard generated")
        for key, path in summary.get('files', {}).items():
            print(f"  {key}: {path}")
        if summary.get("raw_direction_counts"):
            print(f"  raw_direction_counts: {summary.get('raw_direction_counts', {})}")
        if summary.get("policy_available"):
            print(f"  policy_direction_counts: {summary.get('policy_direction_counts', {})}")
            print(f"  pre_rsm_direction_counts: {summary.get('pre_rsm_direction_counts', {})}")
        if summary.get("rsm_action_counts"):
            print(f"  rsm_action_counts: {summary.get('rsm_action_counts', {})}")
        print(f"  direction_counts: {summary.get('direction_counts', {})}")
        if summary.get("all_neutral_after_rsm"):
            print(
                "  ℹ️ All plotted bars became NEUTRAL after RSM filtering. "
                "Check raw_direction_counts vs rsm_action_counts before judging the base model."
            )
        print(f"  transitions: {summary.get('transitions', 0)}")
    except Exception as e:
        print(f"\n⚠️ تعذر توليد Dashboard الـ 5m: {e}")


if __name__ == '__main__':
    main()
