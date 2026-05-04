"""
paper_v19.py - Paper trading and rollout-control runner for QuantSystem V19
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from modules.config_v19 import load_v19_config
from modules.feature_artifact_v19 import load_feature_artifact
from modules.failsafe_v19 import decide_runtime_mode, evaluate_system_health
from modules.logging_v19 import EventLogWriter, ExecutionLogger, RiskLogger, log_event
from modules.monitoring_v19 import MonitoringState, emit_alerts, load_baseline_from_artifacts, load_jsonl, write_monitoring_outputs
from modules.slippage_model import SlippageModel, position_size_from_prediction
from backtest_v19 import _realized_fill_pricing, _simulate_trade_path
from predict_v19 import V19PredictionEngine


def _visual_embeddings(path: str | None, n_rows: int, n_dim: int) -> np.ndarray:
    if not path or not os.path.exists(path):
        return np.zeros((n_rows, n_dim), dtype=np.float32)
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((n_rows, n_dim), dtype=np.float32)
    if len(arr) < n_rows:
        out = np.zeros((n_rows, n_dim), dtype=np.float32)
        out[:len(arr), :min(arr.shape[1], n_dim)] = arr[:, :n_dim]
        return out
    return arr[:n_rows, :n_dim]


def _spread_pips(row: pd.Series, tick_size: float) -> float:
    bid = float(row.get('bid_px_00', 0.0) or 0.0)
    ask = float(row.get('ask_px_00', 0.0) or 0.0)
    if bid > 0 and ask > 0:
        return max((ask - bid) / max(tick_size, 1e-8), 0.0)
    spread = float(row.get('spread', 0.0) or 0.0)
    if spread > 0:
        return max(spread / max(tick_size, 1e-8), 0.0)
    return 0.0


def _session_allowed(ts, allowed_sessions: list[str]) -> bool:
    if not allowed_sessions:
        return True
    ts = pd.Timestamp(ts)
    h = ts.hour
    current = 'asia' if h < 8 else ('london' if h < 13 else ('ny_open' if h < 16 else 'ny_main'))
    return current in allowed_sessions


def run_paper(
    csv_path: str,
    models_dir: str,
    output_dir: str,
    input_scaled: bool = True,
    visual_npy: str | None = None,
    config: dict | None = None,
    run_mode: str = 'paper',
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    cfg = config or load_v19_config()
    paper_cfg = cfg.get('paper', {})
    rollout_cfg = cfg.get('rollout', {})
    failsafe_cfg = cfg.get('failsafe', {})
    log_cfg = cfg.get('logging', {})
    ref_cfg = cfg.get('refinery', {})
    bt_cfg = cfg.get('backtest', {})

    events_path = os.path.join(output_dir, log_cfg.get('events_file', 'paper_events.jsonl'))
    orders_path = os.path.join(output_dir, 'paper_orders.jsonl')
    fills_path = os.path.join(output_dir, 'paper_fills.jsonl')
    trades_path = os.path.join(output_dir, 'paper_trades.jsonl')
    writer = EventLogWriter(events_path)
    orders_writer = EventLogWriter(orders_path)
    fills_writer = EventLogWriter(fills_path)
    trades_writer = EventLogWriter(trades_path)
    exec_logger = ExecutionLogger(writer, run_mode=run_mode, manifest_path=os.path.join(models_dir, 'manifest.json'))
    risk_logger = RiskLogger(writer, run_mode=run_mode, manifest_path=os.path.join(models_dir, 'manifest.json'))

    engine = V19PredictionEngine(
        models_dir=models_dir,
        run_mode=run_mode,
        event_writer=writer,
        manifest_path=os.path.join(models_dir, 'manifest.json'),
        failsafe_policy={**failsafe_cfg, **rollout_cfg},
    )
    df = load_feature_artifact(csv_path)
    canonical_df = engine.factory.prepare_frame(df, already_scaled=input_scaled, include_meta=True)
    visual_embeddings = _visual_embeddings(visual_npy, len(canonical_df), len(engine.visual_features))
    prices = pd.to_numeric(canonical_df.get('price', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    horizons = pd.to_numeric(canonical_df.get('label_horizon_steps', 0), errors='coerce').fillna(0).astype(np.int32).to_numpy()
    micro_atr = pd.to_numeric(canonical_df.get('micro_atr', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    tick_size = float(bt_cfg.get('tick_size', 0.0001))
    tick_value = float(bt_cfg.get('tick_value', 10.0))
    round_trip_cost_pips = float(bt_cfg.get('round_trip_cost_pips', 1.0))
    direction_threshold_ticks = float(ref_cfg.get('direction_threshold_ticks', 1.0))
    tp_mult = float(ref_cfg.get('tp_mult', 1.2))
    sl_mult = float(ref_cfg.get('sl_mult', 1.0))

    slippage = SlippageModel(tick_size=tick_size, tick_value=tick_value, commission=0.0)
    trades = []
    active = None
    cooldown_until = -1
    equity = float(cfg.get('backtest', {}).get('starting_equity', 100000.0))
    engine.loss_guard.update_equity(equity)

    for i, (_, row) in enumerate(canonical_df.iterrows()):
        ts = row.get('ts_event', None)
        spread_pips = _spread_pips(row, tick_size)
        pred = engine.predict_step(
            row.to_dict(),
            visual_embedding=visual_embeddings[i] if len(engine.visual_features) else None,
            ts=ts,
            already_scaled=True,
        )

        health = evaluate_system_health(
            models_dir=models_dir,
            engine_status=engine.get_runtime_status(),
            feature_row=row.to_dict(),
            ts=ts,
            policy={**failsafe_cfg, **rollout_cfg},
            manifest_path=os.path.join(models_dir, 'manifest.json'),
            loss_guard_status=engine.loss_guard.status(),
        )
        runtime = decide_runtime_mode(health, {**failsafe_cfg, **rollout_cfg})

        if active is not None:
            if i >= int(active.get('exit_idx', i + 1)):
                raw_pnl_pips = float(active.get('raw_pnl_pips', 0.0) or 0.0)
                fill_pricing = _realized_fill_pricing(
                    entry_row=active.get('entry_row', {}),
                    exit_row=row.to_dict(),
                    direction=str(active['direction']),
                    size=int(active['size']),
                    raw_pnl_pips=raw_pnl_pips,
                    tick_size=tick_size,
                    round_trip_cost_pips=round_trip_cost_pips,
                )
                pnl = float(fill_pricing['net_pnl_pips']) * tick_value * active['size']
                equity += pnl
                engine.loss_guard.update_equity(equity)
                engine.loss_guard.record_trade(pnl, ts=ts)
                active['exit_ts'] = str(ts)
                active['pnl_dollars'] = round(pnl, 2)
                active['real_pnl_pips'] = float(fill_pricing['net_pnl_pips'])
                active['fill_pnl_pips'] = float(fill_pricing.get('fill_pnl_pips', raw_pnl_pips))
                active['cost_pips'] = float(fill_pricing.get('dynamic_cost_pips', round_trip_cost_pips))
                active['used_dynamic_fill_cost'] = bool(fill_pricing.get('used_dynamic_fill', False))
                active['exit_price'] = float(
                    fill_pricing.get('exit_fill', {}).get('fill_price', row.get('price', 0.0)) or row.get('price', 0.0)
                )
                active['exit_reason'] = active.get('exit_reason', 'path_replay')
                trades.append(active)
                exec_logger.log_order(
                    'paper_trade_closed',
                    {
                        'bias': active['direction'],
                        'confidence': active['confidence'],
                        'position_size': active['size'],
                        'tradeable': True,
                        'extra': {
                            'entry_ts': active['entry_ts'],
                            'exit_ts': active['exit_ts'],
                            'pnl_dollars': active['pnl_dollars'],
                            'real_pnl_pips': active['real_pnl_pips'],
                            'fill_pnl_pips': active['fill_pnl_pips'],
                            'cost_pips': active['cost_pips'],
                            'used_dynamic_fill_cost': active['used_dynamic_fill_cost'],
                            'exit_reason': active['exit_reason'],
                            'equity_after': equity,
                        },
                    },
                    ts=ts,
                )
                log_event(
                    trades_writer,
                    'paper_trade_closed',
                    {
                        'run_mode': run_mode,
                        'bias': active['direction'],
                        'confidence': active['confidence'],
                        'position_size': active['size'],
                        'tradeable': True,
                        'extra': {
                            'entry_ts': active['entry_ts'],
                            'exit_ts': active['exit_ts'],
                            'pnl_dollars': active['pnl_dollars'],
                            'real_pnl_pips': active['real_pnl_pips'],
                            'fill_pnl_pips': active['fill_pnl_pips'],
                            'cost_pips': active['cost_pips'],
                            'used_dynamic_fill_cost': active['used_dynamic_fill_cost'],
                            'exit_reason': active['exit_reason'],
                            'equity_after': equity,
                        },
                    },
                    ts=ts,
                )
                active = None
                cooldown_until = i + int(paper_cfg.get('cooldown_rows', 0))

        allowed = runtime.get('allow_paper', True) if run_mode == 'paper' else runtime.get('allow_rollout', False)
        min_conf_by_regime = rollout_cfg.get('min_confidence_by_regime', {})
        min_conf = float(min_conf_by_regime.get(pred.get('cluster_name', ''), rollout_cfg.get('min_confidence_default', 0.65)))
        size = int(position_size_from_prediction(
            pred,
            base_size=int(paper_cfg.get('base_size', 1)),
            max_size=int(paper_cfg.get('max_size', rollout_cfg.get('max_contracts', 5))),
            fraction=float(paper_cfg.get('fractional_kelly', 0.25) or 0.25),
        ))

        block_reason = ''
        if active is not None and bool(paper_cfg.get('single_position_only', True)):
            block_reason = 'single_position_only'
        elif i < cooldown_until:
            block_reason = 'cooldown_active'
        elif not _session_allowed(ts, list(paper_cfg.get('allowed_sessions', []))):
            block_reason = 'session_disallowed'
        elif spread_pips > float(paper_cfg.get('max_spread_pips', rollout_cfg.get('max_spread_pips', 3.0))):
            block_reason = 'spread_too_wide'
        elif float(pred.get('latency_ms', 0.0) or 0.0) > float(rollout_cfg.get('max_latency_ms', 500.0)):
            block_reason = 'latency_too_high'
        elif float(pred.get('confidence', 0.0) or 0.0) < min_conf:
            block_reason = 'confidence_below_regime_min'
        elif size <= 0:
            block_reason = 'position_size_zero'
        elif size > int(rollout_cfg.get('max_contracts', 5)):
            block_reason = 'size_above_max_contracts'
        elif not allowed:
            block_reason = runtime.get('reason', 'runtime_mode_block')

        if pred.get('tradeable', False) and pred.get('bias') in ('LONG', 'SHORT'):
            exec_logger.log_order(
                'order_submitted',
                {
                    'bias': pred.get('bias'),
                    'confidence': pred.get('confidence'),
                    'position_size': size,
                    'tradeable': not bool(block_reason),
                    'reject_reason': block_reason,
                    'cluster': pred.get('cluster', 0),
                    'cluster_name': pred.get('cluster_name', 'Unknown'),
                    'latency_ms': pred.get('latency_ms', 0.0),
                    'extra': {'spread_pips': spread_pips, 'rollout_mode': run_mode},
                },
                ts=ts,
            )
            log_event(
                orders_writer,
                'order_submitted',
                {
                    'run_mode': run_mode,
                    'bias': pred.get('bias'),
                    'confidence': pred.get('confidence'),
                    'position_size': size,
                    'tradeable': not bool(block_reason),
                    'reject_reason': block_reason,
                    'cluster': pred.get('cluster', 0),
                    'cluster_name': pred.get('cluster_name', 'Unknown'),
                    'latency_ms': pred.get('latency_ms', 0.0),
                    'extra': {'spread_pips': spread_pips, 'rollout_mode': run_mode},
                },
                ts=ts,
            )

        if block_reason:
            if pred.get('tradeable', False):
                risk_logger.log_block(block_reason, ts=ts, extra={'runtime': runtime, 'spread_pips': spread_pips}, event_type='rollout_guard_triggered' if run_mode == 'rollout' else 'risk_blocked')
                exec_logger.log_order(
                    'order_rejected',
                    {
                        'bias': pred.get('bias', 'NEUTRAL'),
                        'confidence': pred.get('confidence', 0.0),
                        'position_size': size,
                        'tradeable': False,
                        'reject_reason': block_reason,
                        'extra': {'runtime': runtime},
                    },
                    ts=ts,
                    level='WARNING',
                )
                log_event(
                    orders_writer,
                    'order_rejected',
                    {
                        'run_mode': run_mode,
                        'bias': pred.get('bias', 'NEUTRAL'),
                        'confidence': pred.get('confidence', 0.0),
                        'position_size': size,
                        'tradeable': False,
                        'reject_reason': block_reason,
                        'extra': {'runtime': runtime},
                    },
                    ts=ts,
                    level='WARNING',
                )
            continue

        if active is None and pred.get('tradeable', False) and pred.get('bias') in ('LONG', 'SHORT'):
            fill = slippage.compute_fill(row.to_dict(), size=size, direction=pred['bias'].lower())
            trade_path = _simulate_trade_path(
                entry_idx=i,
                direction=pred['bias'],
                prices=prices,
                horizons=horizons,
                micro_atr=micro_atr,
                tick_size=tick_size,
                direction_threshold_ticks=direction_threshold_ticks,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                row_data=row.to_dict(),
            )
            if trade_path is None:
                continue
            active = {
                'direction': pred['bias'],
                'confidence': float(pred.get('confidence', 0.0) or 0.0),
                'size': size,
                'entry_ts': str(ts),
                'entry_price': float(fill.get('fill_price', row.get('price', 0.0)) or row.get('price', 0.0)),
                'entry_row': row.to_dict(),
                'exit_idx': int(trade_path['exit_idx']),
                'raw_pnl_pips': float(trade_path['raw_pnl_pips']),
                'exit_reason': str(trade_path['exit_reason']),
                'feature_hash': pred.get('feature_hash', ''),
            }
            exec_logger.log_order(
                'order_filled',
                {
                    'bias': active['direction'],
                    'confidence': active['confidence'],
                    'position_size': active['size'],
                    'tradeable': True,
                    'cluster': pred.get('cluster', 0),
                    'cluster_name': pred.get('cluster_name', 'Unknown'),
                    'extra': {
                        'entry_price': active['entry_price'],
                        'slippage_pips': fill.get('slippage_pips', 0.0),
                        'commission_pips': fill.get('commission_pips', 0.0),
                        'fill_ratio': fill.get('fill_ratio', 0.0),
                    },
                },
                ts=ts,
            )
            log_event(
                fills_writer,
                'order_filled',
                {
                    'run_mode': run_mode,
                    'bias': active['direction'],
                    'confidence': active['confidence'],
                    'position_size': active['size'],
                    'tradeable': True,
                    'cluster': pred.get('cluster', 0),
                    'cluster_name': pred.get('cluster_name', 'Unknown'),
                    'extra': {
                        'entry_price': active['entry_price'],
                        'slippage_pips': fill.get('slippage_pips', 0.0),
                        'commission_pips': fill.get('commission_pips', 0.0),
                        'fill_ratio': fill.get('fill_ratio', 0.0),
                    },
                },
                ts=ts,
            )
            exec_logger.log_order(
                'paper_trade_opened',
                {
                    'bias': active['direction'],
                    'confidence': active['confidence'],
                    'position_size': active['size'],
                    'tradeable': True,
                    'extra': {'entry_ts': active['entry_ts'], 'entry_price': active['entry_price']},
                },
                ts=ts,
            )
            log_event(
                trades_writer,
                'paper_trade_opened',
                {
                    'run_mode': run_mode,
                    'bias': active['direction'],
                    'confidence': active['confidence'],
                    'position_size': active['size'],
                    'tradeable': True,
                    'extra': {'entry_ts': active['entry_ts'], 'entry_price': active['entry_price']},
                },
                ts=ts,
            )

    summary = {
        'rows': int(len(canonical_df)),
        'closed_trades': int(len(trades)),
        'ending_equity': round(equity, 2),
        'total_pnl_dollars': round(sum(float(t.get('pnl_dollars', 0.0) or 0.0) for t in trades), 2),
        'win_rate': round(float(np.mean([float(t.get('pnl_dollars', 0.0) or 0.0) > 0 for t in trades])), 4) if trades else 0.0,
        'run_mode': run_mode,
    }
    with open(os.path.join(output_dir, 'paper_daily_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(trades_path, 'a', encoding='utf-8') as f:
        for trade in trades:
            f.write(json.dumps(trade, ensure_ascii=False) + '\n')

    baseline = load_baseline_from_artifacts(models_dir)
    monitoring = MonitoringState(baseline=baseline).summarize(load_jsonl(events_path))
    alerts = emit_alerts(monitoring, writer=None, config=cfg.get('monitoring', {}))
    monitor_paths = write_monitoring_outputs(output_dir, monitoring, alerts)
    return {
        **summary,
        **monitor_paths,
        'events_file': events_path,
        'paper_orders': orders_path,
        'paper_fills': fills_path,
        'paper_trades': trades_path,
    }


def main():
    p = argparse.ArgumentParser(description='QuantSystem V19 paper/rollout runner')
    p.add_argument('--csv', required=True)
    p.add_argument('--models', default='outputs_v19')
    p.add_argument('--output', default='outputs_v19_paper')
    p.add_argument('--visual_npy', default=None)
    p.add_argument('--input_scaled', action='store_true')
    p.add_argument('--config', default=None)
    p.add_argument('--mode', choices=['paper', 'rollout'], default='paper')
    args = p.parse_args()

    cfg = load_v19_config(args.config)
    summary = run_paper(
        csv_path=args.csv,
        models_dir=args.models,
        output_dir=args.output,
        input_scaled=args.input_scaled,
        visual_npy=args.visual_npy,
        config=cfg,
        run_mode=args.mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
