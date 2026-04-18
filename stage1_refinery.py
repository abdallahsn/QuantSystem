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
    p.add_argument('--label_horizon', type=int, default=int(defaults.get('label_horizon', 50)))
    p.add_argument('--event_roll_window', type=int, default=int(defaults.get('event_roll_window', 50)))
    p.add_argument('--direction_threshold_ticks', type=float, default=float(defaults.get('direction_threshold_ticks', 5.0)))
    p.add_argument('--lob_event_sample', type=int, default=int(defaults.get('lob_event_sample', 100000)))
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
    )


if __name__ == '__main__':
    main()
