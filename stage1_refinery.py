"""
stage1_refinery.py - Standalone entrypoint for the refinery stage
"""

from __future__ import annotations

import argparse

from prepare_training_data import run_refinery


def main():
    p = argparse.ArgumentParser(description='QuantSystem V19 - Stage 1 refinery only')
    p.add_argument('--mbo', required=True)
    p.add_argument('--mbp', required=True)
    p.add_argument('--symbol', default='')
    p.add_argument('--output', default='outputs_v19')
    p.add_argument('--chunksize', type=int, default=300_000)
    p.add_argument('--label_mode', choices=['v19'], default='v19')
    p.add_argument('--n_workers', type=int, default=None)
    p.add_argument('--target_bars', type=int, default=500)
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
    )


if __name__ == '__main__':
    main()
