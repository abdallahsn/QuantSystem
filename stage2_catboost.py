"""
stage2_catboost.py - Standalone entrypoint for CatBoost/XGBoost/regime stage
"""

from __future__ import annotations

import argparse

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
