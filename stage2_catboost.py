"""
stage2_catboost.py - Standalone entrypoint for CatBoost/regime stage
"""

from __future__ import annotations

import argparse

try:
    import catboost  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "❌ CatBoost غير مثبّت في هذه البيئة.\n"
        "نفّذ أولًا:\n"
        "pip install catboost"
    ) from e

from modules.config_v19 import load_v19_config
from modules.hourly_catboost_report import generate_hourly_catboost_report
from train_v19 import run_training_pipeline


def main():
    defaults = load_v19_config().get('training', {})
    p = argparse.ArgumentParser(description='QuantSystem V19 - Stage 2 CatBoost only')
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
        phase='catboost',
        config_snapshot=cfg,
    )

    try:
        summary = generate_hourly_catboost_report(
            csv_path=args.csv,
            models_dir=args.output,
            output_dir=args.output,
            freq='1H',
            report_name='hourly_catboost',
        )
        print("\n📊 Hourly CatBoost report generated")
        for key, path in summary.get('files', {}).items():
            print(f"  {key}: {path}")
        print(f"  direction_counts: {summary.get('direction_counts', {})}")
        print(f"  transitions: {summary.get('transitions', 0)}")
    except Exception as e:
        print(f"\n⚠️ تعذر توليد الرسم/التقرير الساعي: {e}")


if __name__ == '__main__':
    main()
