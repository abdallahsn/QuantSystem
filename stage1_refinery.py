"""
stage1_refinery.py - Standalone entrypoint for the refinery stage
"""

from __future__ import annotations

import argparse

from modules.config_v19 import load_v19_config
from prepare_training_data import run_refinery


def main():
    defaults = load_v19_config().get('refinery', {})
    p = argparse.ArgumentParser(description='QuantSystem V19 - Stage 1 refinery only')
    p.add_argument('--mbo', required=True)
    p.add_argument('--mbp', required=True)
    p.add_argument('--symbol', default='')
    p.add_argument('--output', default='outputs_v19')
    p.add_argument('--chunksize', type=int, default=int(defaults.get('chunksize', 300_000)))
    p.add_argument('--label_mode', choices=['v19'], default=defaults.get('label_mode', 'v19'))
    p.add_argument('--n_workers', type=int, default=None)
    p.add_argument('--target_bars', type=int, default=int(defaults.get('target_bars', 500)))

    # FIX: رُفع من 50 → 150 tick
    # 50 tick ≈ 70 ثانية على بيانات 12 ساعة — قصير جداً لا يسمح للسعر بالوصول لـ TP
    # 150 tick ≈ 3.5 دقيقة — يتوافق مع الشمعة 5 دقيقة ويُعطي حركة كافية
    p.add_argument('--label_horizon', type=int, default=int(defaults.get('label_horizon', 150)))

    p.add_argument('--event_roll_window', type=int, default=int(defaults.get('event_roll_window', 30)))

    # FIX: خُفّض من 5.0 → 2.0 tick
    # 5 tick floor كان يرفع TP/SL بشكل مبالغ فيه على بيانات منخفضة التذبذب
    p.add_argument('--direction_threshold_ticks', type=float, default=float(defaults.get('direction_threshold_ticks', 2.0)))

    p.add_argument('--lob_event_sample', type=int, default=int(defaults.get('lob_event_sample', 100000)))
    p.add_argument('--feature_roll_window', type=int, default=int(defaults.get('feature_roll_window', 150)))

    # معاملات جديدة للمصفاة
    p.add_argument('--tp_mult', type=float, default=float(defaults.get('tp_mult', 1.5)),
                   help='TP = tp_mult × ATR (default: 1.5)')
    p.add_argument('--sl_mult', type=float, default=float(defaults.get('sl_mult', 1.0)),
                   help='SL = sl_mult × ATR (default: 1.0)')
    p.add_argument('--kalman_slope_threshold', type=float,
                   default=float(defaults.get('kalman_slope_threshold', 0.05)),
                   help='حد قوة الميل في Kalman (default: 0.05, القديم: 1e-5)')
    p.add_argument('--trend_strength_min', type=float,
                   default=float(defaults.get('trend_strength_min', 0.20)),
                   help='الحد الأدنى لقوة الترند لتفعيل فلتر الحذف (default: 0.20)')

    args = p.parse_args()

    run_refinery(
        mbo_path=args.mbo,
        mbp_path=args.mbp,
        symbol=args.symbol,
        output_dir=args.output,
        chunksize=None if args.chunksize == 0 else args.chunksize,
        label_mode=args.label_mode,
        n_workers=args.n_workers,
        target_bars=args.target_bars,
        label_horizon=args.label_horizon,
        event_roll_window=args.event_roll_window,
        direction_threshold_ticks=args.direction_threshold_ticks,
        lob_event_sample=args.lob_event_sample,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        kalman_slope_threshold=args.kalman_slope_threshold,
        trend_strength_min=args.trend_strength_min,
    )


if __name__ == '__main__':
    main()
