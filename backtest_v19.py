"""
backtest_v19.py - Causal replay backtester for QuantSystem V19
================================================================
This backtester replays rows one-by-one through V19PredictionEngine using the
same step-by-step path as live inference. It evaluates predictions against the
causal V19 labels and computes trading metrics using forward_return.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.slippage_model import confidence_bet_size
from predict_v19 import V19PredictionEngine

try:
    from modules.html_reporter import generate_backtest_report
    HTML_REPORT_AVAILABLE = True
except ImportError:
    HTML_REPORT_AVAILABLE = False


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ('ts_event', 'label_end_ts'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)
    return df


def _load_visual_embeddings(
    df: pd.DataFrame,
    explicit_path: str | None,
    default_path: str | None,
    expected_dim: int,
) -> np.ndarray:
    n = len(df)
    zero = np.zeros((n, expected_dim), dtype=np.float32)

    path = explicit_path if explicit_path else default_path
    if not path or not os.path.exists(path):
        return zero

    vis = np.load(path)
    vis = np.asarray(vis, dtype=np.float32)
    if vis.ndim != 2:
        raise ValueError(f'❌ visual embeddings must be 2D, got {vis.shape}')

    if explicit_path and len(vis) != n:
        raise ValueError(
            f'❌ visual embeddings rows ({len(vis)}) do not match CSV rows ({n}) for explicit file {path}'
        )

    if len(vis) < n:
        out = zero.copy()
        out[:len(vis), :min(vis.shape[1], expected_dim)] = vis[:, :expected_dim]
        return out

    return vis[:n, :expected_dim]


def _load_meta_features(
    df: pd.DataFrame,
    explicit_path: str | None,
    expected_dim: int,
) -> np.ndarray | None:
    n = len(df)
    if not explicit_path or not os.path.exists(explicit_path):
        return None

    meta = np.load(explicit_path)
    meta = np.asarray(meta, dtype=np.float32)
    if meta.ndim != 2:
        raise ValueError(f'❌ meta features must be 2D, got {meta.shape}')
    if len(meta) != n:
        raise ValueError(
            f'❌ meta features rows ({len(meta)}) do not match CSV rows ({n}) for explicit file {explicit_path}'
        )
    if meta.shape[1] < expected_dim:
        raise ValueError(
            f'❌ meta features columns ({meta.shape[1]}) أقل من المطلوب ({expected_dim})'
        )
    return meta[:, :expected_dim]


def _equity_metrics(equity_curve: list[float]) -> tuple[float, float]:
    if not equity_curve:
        return 0.0, 0.0
    eq = np.asarray(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(eq)
    dd = peaks - eq
    mdd_abs = float(dd.max()) if len(dd) else 0.0
    mdd_pct = float((dd / np.maximum(peaks, 1e-8)).max()) if len(dd) else 0.0
    return mdd_abs, mdd_pct


def _trade_sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    arr = np.asarray(pnls, dtype=np.float64)
    std = float(arr.std())
    if std <= 1e-12:
        return 0.0
    return float(arr.mean() / std * math.sqrt(len(arr)))


def run_causal_backtest(
    df: pd.DataFrame,
    models_dir: str,
    output_dir: str,
    visual_embeddings: np.ndarray,
    meta_features: np.ndarray | None,
    input_scaled: bool,
    tick_size: float,
    tick_value: float,
    round_trip_cost_pips: float,
    max_size: int,
    starting_equity: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    engine = V19PredictionEngine(models_dir)
    engine.reset_state()
    engine.loss_guard.update_equity(starting_equity)
    source_has_labels = 'bias_label' in df.columns
    source_has_fwd = 'forward_return' in df.columns
    replay_df = engine.factory.prepare_frame(df, already_scaled=input_scaled, include_meta=True)

    results = []
    trades = []
    equity_curve = [float(starting_equity)]
    equity = float(starting_equity)

    has_labels = source_has_labels
    has_fwd = source_has_fwd
    blocked_count = 0

    for i, (_, row) in enumerate(replay_df.iterrows()):
        ts = row.get('ts_event', None)
        visual = visual_embeddings[i] if visual_embeddings.size else None
        pred = engine.predict_step(
            row.to_dict(),
            visual_embedding=visual,
            meta_override=meta_features[i] if meta_features is not None else None,
            ts=ts,
            already_scaled=True,
        )

        if pred.get('reason', '').startswith('Warming up'):
            continue

        pred['idx'] = i
        pred['ts_event'] = str(ts) if ts is not None and not pd.isna(ts) else ''
        pred['price'] = _safe_float(row.get('price', 0.0))

        if has_labels:
            true_bias = int(row.get('bias_label', 2))
            pred['true_bias'] = true_bias
            pred['true_label'] = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}.get(true_bias, '?')
            pred['correct'] = (pred.get('bias_idx') == true_bias)

        if pred.get('reason') and 'Warming up' not in pred.get('reason', ''):
            blocked_count += 1

        tradeable = bool(pred.get('tradeable', False))
        direction = pred.get('bias', 'NEUTRAL')
        pred['executed'] = False

        if tradeable and direction in ('LONG', 'SHORT') and has_fwd:
            size = int(pred.get('position_size') or confidence_bet_size(
                float(pred.get('confidence', 0.0)),
                base_size=1,
                max_size=max_size,
            ))
            forward_return = _safe_float(row.get('forward_return', 0.0))
            raw_pnl_pips = (forward_return / tick_size) if direction == 'LONG' else (-forward_return / tick_size)
            net_pnl_pips = raw_pnl_pips - round_trip_cost_pips
            net_pnl_dollars = net_pnl_pips * tick_value * size

            equity += net_pnl_dollars
            engine.loss_guard.update_equity(equity)
            engine.loss_guard.record_trade(net_pnl_dollars, ts=ts)
            equity_curve.append(float(equity))

            result = 'WIN' if net_pnl_pips > 0 else ('LOSE' if net_pnl_pips < 0 else 'FLAT')
            trade = {
                'idx': i,
                'ts_event': pred['ts_event'],
                'dir': direction,
                'confidence': round(_safe_float(pred.get('confidence', 0.0)), 4),
                'size': size,
                'ep': _safe_float(row.get('price', 0.0)),
                'xp': _safe_float(row.get('price', 0.0)) + forward_return,
                'dur_min': _safe_float(row.get('label_horizon_steps', 0.0)),
                'pips': round(net_pnl_pips, 4),
                'raw_pnl_pips': round(raw_pnl_pips, 4),
                'cost_pips': round(round_trip_cost_pips, 4),
                'pnl': round(net_pnl_dollars, 2),
                'result': result,
                'cluster': pred.get('cluster', 0),
                'cluster_name': pred.get('cluster_name', 'Unknown'),
            }
            trades.append(trade)
            pred['executed'] = True
            pred['position_size'] = size
            pred['net_pnl_pips'] = round(net_pnl_pips, 4)
            pred['net_pnl_dollars'] = round(net_pnl_dollars, 2)
            pred['equity_after'] = round(equity, 2)

        results.append(pred)

    results_df = pd.DataFrame(results)
    trades_df = pd.DataFrame(trades)

    mdd_abs, mdd_pct = _equity_metrics(equity_curve)
    trade_pnls = trades_df['pnl'].tolist() if not trades_df.empty else []
    wins = trades_df[trades_df['pnl'] > 0] if not trades_df.empty else trades_df
    losses = trades_df[trades_df['pnl'] < 0] if not trades_df.empty else trades_df

    summary = {
        'predictions': int(len(results_df)),
        'trades': int(len(trades_df)),
        'tradeable_signals': int(results_df['tradeable'].sum()) if 'tradeable' in results_df.columns else 0,
        'accuracy': round(float(results_df['correct'].mean()), 4)
            if 'correct' in results_df.columns and len(results_df) else 0.0,
        'trade_accuracy': round(float(results_df.loc[results_df['executed'] == True, 'correct'].mean()), 4)
            if 'correct' in results_df.columns and 'executed' in results_df.columns
            and (results_df['executed'] == True).any() else 0.0,
        'win_rate': round(float((trades_df['pnl'] > 0).mean()), 4) if len(trades_df) else 0.0,
        'total_pnl_dollars': round(float(trades_df['pnl'].sum()), 2) if len(trades_df) else 0.0,
        'avg_trade_pnl_dollars': round(float(trades_df['pnl'].mean()), 2) if len(trades_df) else 0.0,
        'avg_trade_pips': round(float(trades_df['pips'].mean()), 4) if len(trades_df) else 0.0,
        'profit_factor': round(float(wins['pnl'].sum() / abs(losses['pnl'].sum())), 4)
            if len(losses) and abs(float(losses['pnl'].sum())) > 1e-9 else 0.0,
        'trade_sharpe': round(_trade_sharpe(trade_pnls), 4),
        'max_drawdown_dollars': round(float(mdd_abs), 2),
        'max_drawdown_pct': round(float(mdd_pct), 4),
        'ending_equity': round(float(equity), 2),
        'blocked_predictions': int(blocked_count),
        'visual_coverage': round(float((np.linalg.norm(visual_embeddings, axis=1) > 0).mean()), 4)
            if visual_embeddings.size else 0.0,
    }

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'backtest_v19_results.csv'), index=False)
    trades_df.to_csv(os.path.join(output_dir, 'backtest_v19_trades.csv'), index=False)
    with open(os.path.join(output_dir, 'backtest_v19_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if HTML_REPORT_AVAILABLE and len(trades_df):
        try:
            generate_backtest_report(
                output_dir=output_dir,
                trades=trades,
                equity=equity_curve,
                n_test_bars=len(results_df),
                model_acc=summary.get('accuracy', 0.0),
                n_features=0,
                n_dataset=len(replay_df),
            )
        except Exception as e:
            print(f"  ⚠️ HTML report skipped: {e}")

    return results_df, trades_df, summary


def main():
    p = argparse.ArgumentParser(description='Causal replay backtester for QuantSystem V19')
    p.add_argument('--csv', required=True, help='training_features_ready.csv or compatible V19 feature CSV')
    p.add_argument('--models', default='outputs_v19', help='trained V19 models directory')
    p.add_argument('--output', default='outputs_v19', help='backtest output directory')
    p.add_argument('--visual_npy', default=None, help='optional row-aligned visual embeddings file')
    p.add_argument('--meta_npy', default=None, help='optional row-aligned stage-1 meta features file')
    p.add_argument('--input_scaled', action='store_true',
                   help='set when CSV is already scaled like training_features_ready.csv')
    p.add_argument('--tick_size', type=float, default=0.0001)
    p.add_argument('--tick_value', type=float, default=10.0)
    p.add_argument('--round_trip_cost_pips', type=float, default=1.0)
    p.add_argument('--max_size', type=int, default=5)
    p.add_argument('--starting_equity', type=float, default=100000.0)
    args = p.parse_args()

    df = _load_csv(args.csv)
    engine = V19PredictionEngine(args.models)
    visual_embeddings = _load_visual_embeddings(
        df,
        explicit_path=args.visual_npy,
        default_path=engine.visual_emb_path,
        expected_dim=len(engine.visual_features),
    )
    meta_features = _load_meta_features(
        df,
        explicit_path=args.meta_npy,
        expected_dim=len(engine.meta_features),
    )

    _, _, summary = run_causal_backtest(
        df=df,
        models_dir=args.models,
        output_dir=args.output,
        visual_embeddings=visual_embeddings,
        meta_features=meta_features,
        input_scaled=args.input_scaled,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        round_trip_cost_pips=args.round_trip_cost_pips,
        max_size=args.max_size,
        starting_equity=args.starting_equity,
    )

    print("\n✅ V19 causal backtest complete")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
