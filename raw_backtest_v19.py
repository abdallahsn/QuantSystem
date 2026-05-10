"""
raw_backtest_v19.py - Backtest existing V19 models directly on raw market data
"""

from __future__ import annotations

import argparse
import json
import os

from backtest_v19 import _load_csv, run_causal_backtest
from modules.config_v19 import load_v19_config
from modules.raw_replay_v19 import build_replay_dataset
from walkforward_v19 import compute_eval_visual_embeddings


def run_raw_backtest(
    mbo_path: str,
    mbp_path: str,
    models_dir: str,
    output_dir: str,
    config_path: str | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
    tick_size: float | None = None,
    tick_value: float | None = None,
    round_trip_cost_pips: float | None = None,
    max_size: int | None = None,
    starting_equity: float | None = None,
    direction_threshold_ticks: float | None = None,
    tp_mult: float | None = None,
    sl_mult: float | None = None,
    max_horizon_steps: int | None = None,
    allow_oracle_forward_return: bool = False,
    single_position_only: bool = True,
    cooldown_rows: int = 0,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    config = load_v19_config(config_path)
    ref_cfg = config.get('refinery', {})
    bt_cfg = config.get('backtest', {})
    label_horizon = int(ref_cfg.get('label_horizon', 150))
    event_roll_window = int(ref_cfg.get('event_roll_window', 50))
    regime_window = int(ref_cfg.get('regime_window', 50))
    warmup_rows = max(500, event_roll_window * 6, regime_window * 4, 200)
    tail_rows = max(label_horizon * 3, 50)

    scaler_path = os.path.join(models_dir, 'scaler_params.json')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"❌ Missing scaler params in models dir: {scaler_path}\n"
            "Raw backtest needs the training scaler to transform raw features consistently."
        )

    dataset_dir = os.path.join(output_dir, 'dataset')
    replay_build = build_replay_dataset(
        mbo_path=mbo_path,
        mbp_path=mbp_path,
        output_dir=dataset_dir,
        start_ts=start_ts,
        end_ts=end_ts,
        label_mode=str(ref_cfg.get('label_mode', 'v19')),
        chunksize=int(ref_cfg.get('chunksize', 0) or 0),
        n_workers=ref_cfg.get('n_workers'),
        target_bars=int(ref_cfg.get('target_bars', 500)),
        label_horizon=int(ref_cfg.get('label_horizon', 150)),
        event_roll_window=int(ref_cfg.get('event_roll_window', 50)),
        direction_threshold_ticks=float(ref_cfg.get('direction_threshold_ticks', 1.0)),
        tp_mult=float(ref_cfg.get('tp_mult', 1.2)),
        sl_mult=float(ref_cfg.get('sl_mult', 1.0)),
        kalman_slope_threshold=float(ref_cfg.get('kalman_slope_threshold', 0.05)),
        trend_strength_min=float(ref_cfg.get('trend_strength_min', 0.05)),
        regime_mode=str(ref_cfg.get('regime_mode', 'rules')),
        regime_stride=int(ref_cfg.get('regime_stride', 50)),
        regime_window=int(ref_cfg.get('regime_window', 50)),
        regime_progress_every=int(ref_cfg.get('regime_progress_every', 25000)),
        lob_event_sample=int(ref_cfg.get('lob_event_sample', 100000)),
        external_scaler_path=scaler_path,
        fit_aux_models=False,
        warmup_rows=warmup_rows,
        tail_rows=tail_rows,
    )

    df = _load_csv(replay_build['csv'])
    visual_embeddings, visual_diagnostics = compute_eval_visual_embeddings(
        test_csv=replay_build['csv'],
        test_lob=replay_build['lob'],
        test_lob_ts=replay_build['lob_ts'],
        models_dir=models_dir,
        return_diagnostics=True,
    )
    with open(os.path.join(output_dir, 'eval_visual_diagnostics.json'), 'w') as f:
        json.dump(visual_diagnostics, f, indent=2)
    print(
        "  ℹ️ Eval visual coverage: "
        f"{visual_diagnostics.get('rows_with_visual', 0):,}/{visual_diagnostics.get('rows_total', 0):,} "
        f"({visual_diagnostics.get('visual_coverage_ratio', 0.0):.1%}) | "
        f"rows_with_tensor={visual_diagnostics.get('rows_with_tensor', 0):,} | "
        f"used_tensors={visual_diagnostics.get('used_tensor_count', 0):,}"
    )

    _, _, backtest_summary = run_causal_backtest(
        df=df,
        models_dir=models_dir,
        output_dir=output_dir,
        visual_embeddings=visual_embeddings,
        visual_diagnostics=visual_diagnostics,
        meta_features=None,
        input_scaled=True,
        tick_size=float(tick_size if tick_size is not None else bt_cfg.get('tick_size', 0.0001)),
        tick_value=float(tick_value if tick_value is not None else bt_cfg.get('tick_value', 10.0)),
        round_trip_cost_pips=float(
            round_trip_cost_pips if round_trip_cost_pips is not None else bt_cfg.get('round_trip_cost_pips', 1.0)
        ),
        max_size=int(max_size if max_size is not None else bt_cfg.get('max_size', 5)),
        starting_equity=float(starting_equity if starting_equity is not None else bt_cfg.get('starting_equity', 100000.0)),
        direction_threshold_ticks=float(
            direction_threshold_ticks if direction_threshold_ticks is not None else ref_cfg.get('direction_threshold_ticks', 1.0)
        ),
        tp_mult=float(tp_mult if tp_mult is not None else ref_cfg.get('tp_mult', 1.2)),
        sl_mult=float(sl_mult if sl_mult is not None else ref_cfg.get('sl_mult', 1.0)),
        max_horizon_steps=max_horizon_steps,
        allow_oracle_forward_return=allow_oracle_forward_return,
        single_position_only=single_position_only,
        cooldown_rows=int(max(cooldown_rows, 0)),
        score_start_ts=start_ts,
        score_end_ts=end_ts,
    )

    summary = {
        'inputs': {
            'mbo': mbo_path,
            'mbp': mbp_path,
            'models': models_dir,
            'start_ts': start_ts,
            'end_ts': end_ts,
        },
        'dataset': replay_build,
        'visual_diagnostics': visual_diagnostics,
        'backtest': backtest_summary,
    }
    with open(os.path.join(output_dir, 'raw_backtest_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser(description='Run a V19 backtest on raw MBO/MBP data using existing trained models')
    p.add_argument('--mbo', required=True, help='raw MBO/trades file')
    p.add_argument('--mbp', required=True, help='raw MBP/order-book file')
    p.add_argument('--models', required=True, help='trained V19 models directory')
    p.add_argument('--output', default='outputs_v19_raw_backtest', help='output directory for dataset + backtest')
    p.add_argument('--config', default=None, help='optional config override')
    p.add_argument('--start_ts', default=None, help='optional inclusive slice start timestamp')
    p.add_argument('--end_ts', default=None, help='optional exclusive slice end timestamp')
    p.add_argument('--tick_size', type=float, default=None)
    p.add_argument('--tick_value', type=float, default=None)
    p.add_argument('--round_trip_cost_pips', type=float, default=None)
    p.add_argument('--max_size', type=int, default=None)
    p.add_argument('--starting_equity', type=float, default=None)
    p.add_argument('--direction_threshold_ticks', type=float, default=None)
    p.add_argument('--tp_mult', type=float, default=None)
    p.add_argument('--sl_mult', type=float, default=None)
    p.add_argument('--max_horizon_steps', type=int, default=0,
                   help='optional cap on replay horizon in rows; 0 uses label_horizon_steps as-is')
    p.add_argument('--allow_oracle_forward_return', action='store_true',
                   help='dangerous: fall back to stored forward_return when no causal replay window is available')
    p.add_argument('--disable_single_position_only', action='store_true',
                   help='allow overlapping trades; default keeps one active position at a time')
    p.add_argument('--cooldown_rows', type=int, default=0,
                   help='rows to wait after closing a trade before opening a new one')
    args = p.parse_args()

    summary = run_raw_backtest(
        mbo_path=args.mbo,
        mbp_path=args.mbp,
        models_dir=args.models,
        output_dir=args.output,
        config_path=args.config,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        round_trip_cost_pips=args.round_trip_cost_pips,
        max_size=args.max_size,
        starting_equity=args.starting_equity,
        direction_threshold_ticks=args.direction_threshold_ticks,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        max_horizon_steps=(args.max_horizon_steps if args.max_horizon_steps > 0 else None),
        allow_oracle_forward_return=args.allow_oracle_forward_return,
        single_position_only=(not args.disable_single_position_only),
        cooldown_rows=args.cooldown_rows,
    )
    print("\n✅ Raw V19 backtest complete")
    print(json.dumps(summary['backtest'], indent=2))


if __name__ == '__main__':
    main()
