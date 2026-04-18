"""
walkforward_v19.py - Raw-data walk-forward evaluation for QuantSystem V19
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from backtest_v19 import _load_csv, run_causal_backtest
from modules.config_v19 import load_release_gates, load_v19_config
from modules.manifest_v19 import write_manifest
from modules.raw_replay_v19 import build_replay_dataset, normalize_ts, read_market_data
from modules.release_gates_v19 import evaluate_release_gates, save_gate_report
from train_v19 import _align_lob_to_rows, _load_lob_inputs, run_training_pipeline

try:
    from modules.deeplob_cnn import DeepLOBCNN, VISUAL_EMB_DIM
    DEEPLOB_AVAILABLE = True
except ImportError:
    VISUAL_EMB_DIM = 8
    DEEPLOB_AVAILABLE = False


def build_walkforward_windows(
    timestamps: pd.Series,
    n_splits: int,
    initial_train_frac: float,
    test_frac: float,
    min_train_rows: int,
) -> list[dict]:
    ts = normalize_ts(pd.DataFrame({'ts_event': timestamps}))['ts_event']
    n = len(ts)
    windows = []
    base_train_end = max(int(n * initial_train_frac), min_train_rows)
    test_rows = max(int(n * test_frac), 1)

    for fold in range(n_splits):
        train_end_idx = base_train_end + fold * test_rows
        test_end_idx = min(train_end_idx + test_rows, n)
        if train_end_idx >= n or (test_end_idx - train_end_idx) <= 10:
            break
        windows.append({
            'fold': fold + 1,
            'train_start': ts.iloc[0],
            'train_end': ts.iloc[train_end_idx - 1],
            'test_start': ts.iloc[train_end_idx],
            'test_end': ts.iloc[test_end_idx - 1] + pd.Timedelta(microseconds=1),
            'train_rows_est': int(train_end_idx),
            'test_rows_est': int(test_end_idx - train_end_idx),
        })
    return windows


def compute_eval_visual_embeddings(test_csv: str, test_lob: str, test_lob_ts: str, models_dir: str) -> np.ndarray:
    df = _load_csv(test_csv)
    if not DEEPLOB_AVAILABLE:
        return np.zeros((len(df), VISUAL_EMB_DIM), dtype=np.float32)

    deeplob_path = os.path.join(models_dir, 'deeplob_cnn_v19.keras')
    if not os.path.exists(deeplob_path):
        return np.zeros((len(df), VISUAL_EMB_DIM), dtype=np.float32)

    lob_tensors, lob_timestamps = _load_lob_inputs(test_lob, test_lob_ts)
    if lob_tensors is None or lob_timestamps is None:
        return np.zeros((len(df), VISUAL_EMB_DIM), dtype=np.float32)

    row_to_tensor, _, _ = _align_lob_to_rows(df, lob_timestamps)
    cnn = DeepLOBCNN(brain_file=deeplob_path)
    if cnn.model is None or not cnn._fitted:
        return np.zeros((len(df), VISUAL_EMB_DIM), dtype=np.float32)

    used_tensor_ids = np.unique(row_to_tensor[row_to_tensor >= 0]).astype(np.int32)
    emb_lookup = {}
    if len(used_tensor_ids):
        X = np.asarray(lob_tensors[used_tensor_ids], dtype=np.float32)
        emb = np.asarray(cnn.get_embeddings(X), dtype=np.float32)
        for i, tid in enumerate(used_tensor_ids):
            emb_lookup[int(tid)] = emb[i]

    out = np.zeros((len(df), VISUAL_EMB_DIM), dtype=np.float32)
    for row_idx, tensor_idx in enumerate(row_to_tensor):
        if int(tensor_idx) in emb_lookup:
            out[row_idx] = emb_lookup[int(tensor_idx)][:VISUAL_EMB_DIM]
    return out


def aggregate_fold_metrics(fold_reports: list[dict]) -> dict:
    if not fold_reports:
        return {
            'n_folds': 0,
            'mean_directional_precision': 0.0,
            'mean_directional_recall': 0.0,
            'mean_directional_f1': 0.0,
            'mean_event_gate_rate': 0.0,
            'mean_win_rate': 0.0,
            'mean_trade_sharpe': 0.0,
            'max_drawdown_pct': 0.0,
            'total_trades': 0,
            'total_pnl_dollars': 0.0,
        }

    return {
        'n_folds': int(len(fold_reports)),
        'mean_directional_precision': float(np.mean([r['backtest'].get('directional_precision_macro', 0.0) for r in fold_reports])),
        'mean_directional_recall': float(np.mean([r['backtest'].get('directional_recall_macro', 0.0) for r in fold_reports])),
        'mean_directional_f1': float(np.mean([r['backtest'].get('directional_f1_macro', 0.0) for r in fold_reports])),
        'mean_event_gate_rate': float(np.mean([r['backtest'].get('event_gate_rate', 0.0) for r in fold_reports])),
        'mean_win_rate': float(np.mean([r['backtest'].get('win_rate', 0.0) for r in fold_reports])),
        'mean_trade_sharpe': float(np.mean([r['backtest'].get('trade_sharpe', 0.0) for r in fold_reports])),
        'max_drawdown_pct': float(np.max([r['backtest'].get('max_drawdown_pct', 0.0) for r in fold_reports])),
        'total_trades': int(np.sum([r['backtest'].get('trades', 0) for r in fold_reports])),
        'total_pnl_dollars': float(np.sum([r['backtest'].get('total_pnl_dollars', 0.0) for r in fold_reports])),
    }


def run_walkforward(
    mbo_path: str,
    mbp_path: str,
    output_dir: str,
    config: dict,
    gates: dict,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    walk_cfg = config.get('walkforward', {})
    train_cfg = config.get('training', {})
    ref_cfg = config.get('refinery', {})
    bt_cfg = config.get('backtest', {})

    mbo_df = normalize_ts(read_market_data(mbo_path))
    windows = build_walkforward_windows(
        timestamps=mbo_df['ts_event'],
        n_splits=int(walk_cfg.get('n_splits', 3)),
        initial_train_frac=float(walk_cfg.get('initial_train_frac', 0.55)),
        test_frac=float(walk_cfg.get('test_frac', 0.15)),
        min_train_rows=int(walk_cfg.get('min_train_rows', 1000)),
    )

    fold_reports = []
    for window in windows:
        fold_name = f"fold_{window['fold']:02d}"
        fold_dir = os.path.join(output_dir, fold_name)
        train_dir = os.path.join(fold_dir, 'train_data')
        test_dir = os.path.join(fold_dir, 'test_data')
        model_dir = os.path.join(fold_dir, 'models')
        eval_dir = os.path.join(fold_dir, 'evaluation')

        print(f"\n{'=' * 65}\n🔁 Walk-forward {fold_name}\n{'=' * 65}")
        train_build = build_replay_dataset(
            mbo_path=mbo_path,
            mbp_path=mbp_path,
            output_dir=train_dir,
            start_ts=None,
            end_ts=window['test_start'],
            label_mode=ref_cfg.get('label_mode', 'v19'),
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
            lob_event_sample=int(ref_cfg.get('lob_event_sample', 100000)),
        )
        train_summary = run_training_pipeline(
            csv_path=train_build['csv'],
            output_dir=model_dir,
            lob_path=train_build['lob'],
            lob_ts_path=train_build['lob_ts'],
            epochs=int(train_cfg.get('epochs', 100)),
            batch=int(train_cfg.get('batch', 64)),
            n_folds=int(train_cfg.get('n_folds', 6)),
            test_size=float(train_cfg.get('test_size', 0.10)),
            embargo_pct=float(train_cfg.get('embargo_pct', 0.02)),
            min_train_pct=float(train_cfg.get('min_train_pct', 0.20)),
            train_frac=float(train_cfg.get('train_frac', 0.80)),
            min_seq_coverage=float(train_cfg.get('min_seq_coverage', 0.80)),
            stage=int(train_cfg.get('stage', 0)),
            training_mode=str(train_cfg.get('mode', 'event_binary')),
            quality_weight_strong=float(train_cfg.get('quality_weight_strong', 2.0)),
            quality_weight_weak=float(train_cfg.get('quality_weight_weak', 1.0)),
            config_snapshot=config,
        )

        test_build = build_replay_dataset(
            mbo_path=mbo_path,
            mbp_path=mbp_path,
            output_dir=test_dir,
            start_ts=window['test_start'],
            end_ts=window['test_end'],
            label_mode=ref_cfg.get('label_mode', 'v19'),
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
            lob_event_sample=int(ref_cfg.get('lob_event_sample', 100000)),
            external_scaler_path=os.path.join(model_dir, 'scaler_params.json'),
            fit_aux_models=False,
        )
        test_df = _load_csv(test_build['csv'])
        test_visual = compute_eval_visual_embeddings(
            test_csv=test_build['csv'],
            test_lob=test_build['lob'],
            test_lob_ts=test_build['lob_ts'],
            models_dir=model_dir,
        )
        _, _, backtest_summary = run_causal_backtest(
            df=test_df,
            models_dir=model_dir,
            output_dir=eval_dir,
            visual_embeddings=test_visual,
            meta_features=None,
            input_scaled=True,
            tick_size=float(bt_cfg.get('tick_size', 0.0001)),
            tick_value=float(bt_cfg.get('tick_value', 10.0)),
            round_trip_cost_pips=float(bt_cfg.get('round_trip_cost_pips', 1.0)),
            max_size=int(bt_cfg.get('max_size', 5)),
            starting_equity=float(bt_cfg.get('starting_equity', 100000.0)),
        )

        fold_report = {
            'fold': window['fold'],
            'window': {
                'train_start': str(window['train_start']),
                'train_end': str(window['train_end']),
                'test_start': str(window['test_start']),
                'test_end': str(window['test_end']),
            },
            'train': train_summary,
            'backtest': backtest_summary,
        }
        fold_reports.append(fold_report)
        with open(os.path.join(fold_dir, 'fold_report.json'), 'w') as f:
            json.dump(fold_report, f, indent=2)

    aggregate = aggregate_fold_metrics(fold_reports)
    gate_report = evaluate_release_gates(aggregate, gates)
    gates_path = save_gate_report(output_dir, gate_report)
    manifest_path = write_manifest(
        output_dir=output_dir,
        kind='walkforward_v19',
        config=config,
        inputs={'mbo': mbo_path, 'mbp': mbp_path},
        metrics=aggregate,
        extra={'release_gates': gate_report, 'windows': windows},
    )

    out = {
        'folds': fold_reports,
        'aggregate': aggregate,
        'release_gates': gate_report,
        'manifest': manifest_path,
        'release_gates_report': gates_path,
    }
    with open(os.path.join(output_dir, 'walkforward_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out


def main():
    p = argparse.ArgumentParser(description='QuantSystem V19 raw-data walk-forward evaluator')
    p.add_argument('--mbo', required=True)
    p.add_argument('--mbp', required=True)
    p.add_argument('--output', default='outputs_v19_walkforward')
    p.add_argument('--config', default=None)
    p.add_argument('--gates', default=None)
    args = p.parse_args()

    config = load_v19_config(args.config)
    gates = load_release_gates(args.gates)
    summary = run_walkforward(
        mbo_path=args.mbo,
        mbp_path=args.mbp,
        output_dir=args.output,
        config=config,
        gates=gates,
    )
    print("\n✅ Walk-forward complete")
    print(json.dumps(summary['aggregate'], indent=2))
    print(json.dumps(summary['release_gates'], indent=2))


if __name__ == '__main__':
    main()
