"""
stage1_refinery.py - Standalone entrypoint for the refinery stage
"""

from __future__ import annotations

import argparse
import os

from modules.config_v19 import load_v19_config


def _configure_stage1_runtime_env() -> None:
    # Multiprocessing-heavy stages perform best when BLAS/OpenMP stay single-threaded
    # per worker. Respect user overrides if already set.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None, help='optional config override file')
    pre_args, _ = pre.parse_known_args()
    full_cfg = load_v19_config(pre_args.config)
    defaults = full_cfg.get('refinery', {}) or {}
    soft_defaults = full_cfg.get('soft_labels', {}) or {}
    _configure_stage1_runtime_env()
    from prepare_training_data import run_refinery
    p = argparse.ArgumentParser(description='QuantSystem V19 - Stage 1 refinery only')
    p.add_argument('--config', default=pre_args.config, help='optional config override file')
    p.add_argument('--mbo', required=True)
    p.add_argument('--mbp', required=True)
    p.add_argument('--symbol', default='')
    p.add_argument('--output', default=defaults.get('output_dir', 'outputs_v19'))
    p.add_argument('--chunk_rows', '--chunksize', dest='chunk_rows', type=int, default=int(defaults.get('chunk_rows', defaults.get('chunksize', 2_000_000))))
    p.add_argument('--label_mode', choices=['v19'], default=defaults.get('label_mode', 'v19'))
    p.add_argument('--n_workers', type=int, default=defaults.get('n_workers'))
    p.add_argument('--mbo_workers', type=int, default=defaults.get('mbo_workers'))
    p.add_argument('--mbp_workers', type=int, default=defaults.get('mbp_workers'))
    p.add_argument('--resume', action='store_true', default=bool(defaults.get('resume', False)))
    p.add_argument('--shard_warmup_rows', type=int, default=int(defaults.get('shard_warmup_rows', 5000)))
    p.add_argument('--target_bars', type=int, default=int(defaults.get('target_bars', 500)))

    # FIX: رُفع من 50 → 150 tick
    # 50 tick ≈ 70 ثانية على بيانات 12 ساعة — قصير جداً لا يسمح للسعر بالوصول لـ TP
    # 150 tick ≈ 3.5 دقيقة — يتوافق مع الشمعة 5 دقيقة ويُعطي حركة كافية
    p.add_argument('--label_horizon', type=int, default=int(defaults.get('label_horizon', 150)))

    p.add_argument('--event_roll_window', type=int, default=int(defaults.get('event_roll_window', 30)))
    p.add_argument('--feature_roll_window', type=int, default=int(defaults.get('feature_roll_window', 150)))

    # FIX: خُفّض من 5.0 → 2.0 tick
    # 5 tick floor كان يرفع TP/SL بشكل مبالغ فيه على بيانات منخفضة التذبذب
    p.add_argument('--direction_threshold_ticks', type=float, default=float(defaults.get('direction_threshold_ticks', 1.0)))
    p.add_argument('--causal_threshold_mode', choices=['expanding', 'fixed'], default=str(defaults.get('causal_threshold_mode', 'fixed')))
    p.add_argument('--raw_event_target_rate', type=float, default=float(defaults.get('raw_event_target_rate', 0.70)))
    p.add_argument('--training_event_target_rate', type=float, default=float(defaults.get('training_event_target_rate', 0.25)))
    p.add_argument('--training_event_score_threshold', type=float, default=defaults.get('training_event_score_threshold', 0.0))

    p.add_argument('--lob_event_sample', type=int, default=int(defaults.get('lob_event_sample', 100000)))

    # معاملات جديدة للمصفاة
    p.add_argument('--tp_mult', type=float, default=float(defaults.get('tp_mult', 1.2)),
                   help='TP = tp_mult × ATR (default: 1.2)')
    p.add_argument('--sl_mult', type=float, default=float(defaults.get('sl_mult', 1.0)),
                   help='SL = sl_mult × ATR (default: 1.0)')
    p.add_argument('--tp_sl_threshold_mode', choices=['fixed', 'atr'],
                   default=str(defaults.get('tp_sl_threshold_mode', 'fixed')),
                   help="label TP/SL threshold mode: 'fixed' or 'atr' (default: fixed)")
    p.set_defaults(
        adaptive_horizon=bool(defaults.get('adaptive_horizon', True)),
        trend_filter=bool(defaults.get('trend_filter', True)),
        trend_filter_strict=bool(defaults.get('trend_filter_strict', False)),
        emit_meta_labels=bool(defaults.get('emit_meta_labels', True)),
    )
    p.add_argument('--adaptive_horizon', dest='adaptive_horizon', action='store_true',
                   help='enable ATR-adaptive label horizon')
    p.add_argument('--no_adaptive_horizon', dest='adaptive_horizon', action='store_false',
                   help='disable ATR-adaptive label horizon')
    p.add_argument('--trend_filter', dest='trend_filter', action='store_true',
                   help='enable Kalman trend filter for labels')
    p.add_argument('--no_trend_filter', dest='trend_filter', action='store_false',
                   help='disable Kalman trend filter for labels')
    p.add_argument('--trend_filter_strict', dest='trend_filter_strict', action='store_true',
                   help='enable stricter Kalman trend filter behaviour')
    p.add_argument('--no_trend_filter_strict', dest='trend_filter_strict', action='store_false',
                   help='disable stricter Kalman trend filter behaviour')
    p.add_argument('--emit_meta_labels', dest='emit_meta_labels', action='store_true',
                   help='emit stable meta-label/context columns (default: on)')
    p.add_argument('--no_emit_meta_labels', dest='emit_meta_labels', action='store_false',
                   help='disable meta-label/context column emission')
    p.add_argument('--use_soft_labels', action=argparse.BooleanOptionalAction, default=bool(soft_defaults.get('enabled', True)),
                   help='enable/disable soft labels in the label refinery')
    p.add_argument('--soft_label_mode', choices=['analytical', 'monte_carlo'],
                   default=str(soft_defaults.get('mode', 'monte_carlo')))
    p.add_argument('--soft_label_n_scenarios', '--soft_label_scenarios', dest='soft_label_n_scenarios', type=int,
                   default=int(soft_defaults.get('n_scenarios', 50)))
    p.add_argument('--soft_label_random_seed', '--soft_label_seed', dest='soft_label_random_seed', type=int,
                   default=int(soft_defaults.get('random_seed', 42)))
    p.add_argument('--soft_label_horizon_std', '--soft_label_horizon_jitter', dest='soft_label_horizon_std', type=float,
                   default=float(soft_defaults.get('horizon_std', 0.15)))
    p.add_argument('--soft_label_tp_std', '--soft_label_tp_jitter', dest='soft_label_tp_std', type=float,
                   default=float(soft_defaults.get('tp_std', 0.10)))
    p.add_argument('--soft_label_sl_std', '--soft_label_sl_jitter', dest='soft_label_sl_std', type=float,
                   default=float(soft_defaults.get('sl_std', 0.10)))
    p.add_argument('--kalman_slope_threshold', type=float,
                   default=float(defaults.get('kalman_slope_threshold', 0.05)),
                   help='حد قوة الميل في Kalman (default: 0.05, القديم: 1e-5)')
    p.add_argument('--trend_strength_min', type=float,
                   default=float(defaults.get('trend_strength_min', 0.05)),
                   help='الحد الأدنى لقوة الترند المعاكس لتفعيل فلتر الحذف (default: 0.05)')
    p.add_argument('--regime_mode', choices=['rules', 'wasserstein', 'off'],
                   default=str(defaults.get('regime_mode', 'rules')))
    p.add_argument('--regime_stride', type=int, default=int(defaults.get('regime_stride', 50)))
    p.add_argument('--regime_window', type=int, default=int(defaults.get('regime_window', 50)))
    p.add_argument('--regime_progress_every', type=int, default=int(defaults.get('regime_progress_every', 25000)))
    p.add_argument('--merge_tolerance_ms', type=int, default=int(defaults.get('merge_tolerance_ms', 500)))
    p.add_argument('--step4_min_parallel_rows', type=int, default=int(defaults.get('step4_min_parallel_rows', 250000)))

    p.add_argument(
        '--enforce-economic-tp-floor',
        dest='enforce_economic_tp_floor',
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get('enforce_economic_tp_floor', True)),
        help='TP floor ≥ SL floor + cost ticks (aligns soft_label / MC weights with execution economics; default on).',
    )

    args = p.parse_args()

    run_refinery(
        mbo_path=args.mbo,
        mbp_path=args.mbp,
        symbol=args.symbol,
        output_dir=args.output,
        chunksize=None if args.chunk_rows == 0 else args.chunk_rows,
        chunk_rows=None if args.chunk_rows == 0 else args.chunk_rows,
        label_mode=args.label_mode,
        n_workers=args.n_workers,
        mbo_workers=args.mbo_workers,
        mbp_workers=args.mbp_workers,
        resume=args.resume,
        target_bars=args.target_bars,
        label_horizon=args.label_horizon,
        event_roll_window=args.event_roll_window,
        feature_roll_window=args.feature_roll_window,
        direction_threshold_ticks=args.direction_threshold_ticks,
        causal_threshold_mode=args.causal_threshold_mode,
        raw_event_target_rate=args.raw_event_target_rate,
        training_event_target_rate=args.training_event_target_rate,
        training_event_score_threshold=args.training_event_score_threshold,
        lob_event_sample=args.lob_event_sample,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        adaptive_horizon=args.adaptive_horizon,
        trend_filter=args.trend_filter,
        trend_filter_strict=args.trend_filter_strict,
        use_soft_labels=args.use_soft_labels,
        soft_label_mode=args.soft_label_mode,
        soft_label_n_scenarios=args.soft_label_n_scenarios,
        soft_label_random_seed=args.soft_label_random_seed,
        soft_label_horizon_std=args.soft_label_horizon_std,
        soft_label_tp_std=args.soft_label_tp_std,
        soft_label_sl_std=args.soft_label_sl_std,
        kalman_slope_threshold=args.kalman_slope_threshold,
        trend_strength_min=args.trend_strength_min,
        regime_mode=args.regime_mode,
        regime_stride=args.regime_stride,
        regime_window=args.regime_window,
        regime_progress_every=args.regime_progress_every,
        shard_warmup_rows=args.shard_warmup_rows,
        merge_tolerance_ms=args.merge_tolerance_ms,
        step4_min_parallel_rows=args.step4_min_parallel_rows,
        config_path=args.config,
        enforce_economic_tp_floor=args.enforce_economic_tp_floor,
    )


if __name__ == '__main__':
    main()
