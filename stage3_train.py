"""
stage3_train.py - Standalone entrypoint for final MetaLearner training stage
"""

from __future__ import annotations

import argparse

try:
    import tensorflow  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "❌ TensorFlow غير مثبّت في هذه البيئة.\n"
        "نفّذ أولًا:\n"
        "bash install_tf_gpu_cu12.sh\n"
        "أو CPU فقط:\n"
        "pip install tensorflow"
    ) from e

from modules.config_v19 import load_v19_config
from train_v19 import run_training_pipeline


def main():
    defaults = load_v19_config().get('training', {})
    p = argparse.ArgumentParser(description='QuantSystem V19 - Stage 3 final training only')
    p.add_argument('--csv', required=True, help='training_features_ready.csv from stage 1')
    p.add_argument('--lob', default=None, help='optional lob_tensors.npy')
    p.add_argument('--lob_ts', default=None, help='optional lob_tensor_timestamps.npy')
    p.add_argument('--output', default=defaults.get('output_dir', 'outputs_v19'))
    p.add_argument('--epochs', type=int, default=int(defaults.get('epochs', 100)))
    p.add_argument('--batch', type=int, default=int(defaults.get('batch', 64)))
    p.add_argument('--min_train_pct', type=float, default=float(defaults.get('min_train_pct', 0.20)))
    p.add_argument('--train_frac', type=float, default=float(defaults.get('train_frac', 0.80)))
    p.add_argument('--min_seq_coverage', type=float, default=float(defaults.get('min_seq_coverage', 0.80)))
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
        epochs=args.epochs,
        batch=args.batch,
        min_train_pct=args.min_train_pct,
        train_frac=args.train_frac,
        min_seq_coverage=args.min_seq_coverage,
        training_mode=args.training_mode,
        quality_weight_strong=args.quality_weight_strong,
        quality_weight_weak=args.quality_weight_weak,
        split_time=args.split_time,
        train_days=args.train_days,
        backtest_days=args.backtest_days,
        window_end=args.window_end,
        phase='train',
        config_snapshot=cfg,
    )


if __name__ == '__main__':
    main()
