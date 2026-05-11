"""
walkforward_v19.py - Raw-data walk-forward evaluation for QuantSystem V19
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from backtest_v19 import _load_csv, run_causal_backtest
from modules.config_v19 import load_release_gates, load_v19_config
from modules.manifest_v19 import write_manifest
from modules.raw_replay_v19 import build_replay_dataset, normalize_ts, read_market_data
from modules.release_gates_v19 import evaluate_release_gates, save_gate_report
from train_v19 import _align_lob_to_rows, _load_lob_inputs, _resolve_visual_model_type, run_training_pipeline

try:
    from modules.deeplob_cnn import DeepLOBCNN, VISUAL_EMB_DIM
    DEEPLOB_AVAILABLE = True
except ImportError:
    VISUAL_EMB_DIM = 8
    DEEPLOB_AVAILABLE = False

try:
    from modules.lob_transformer import LOBTransformer
    LOB_TRANSFORMER_AVAILABLE = True
except ImportError:
    LOB_TRANSFORMER_AVAILABLE = False


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
    months = ts.dt.to_period('M')
    unique_months = list(months.dropna().unique())
    if len(unique_months) >= 6:
        fold_no = 1
        for month in unique_months[1:]:
            test_mask = (months == month).to_numpy(dtype=bool)
            train_mask = (months < month).to_numpy(dtype=bool)
            train_rows = int(train_mask.sum())
            test_rows = int(test_mask.sum())
            if train_rows < int(min_train_rows) or test_rows <= 10:
                continue
            test_ts = ts.loc[test_mask]
            train_ts = ts.loc[train_mask]
            windows.append({
                'fold': fold_no,
                'mode': 'monthly_expanding',
                'train_start': train_ts.iloc[0],
                'train_end': train_ts.iloc[-1],
                'test_start': test_ts.iloc[0],
                'test_end': test_ts.iloc[-1] + pd.Timedelta(microseconds=1),
                'train_rows_est': train_rows,
                'test_rows_est': test_rows,
                'test_month': str(month),
            })
            fold_no += 1
        return windows

    base_train_end = max(int(n * initial_train_frac), min_train_rows)
    test_rows = max(int(n * test_frac), 1)
    for fold in range(max(int(n_splits), 3)):
        train_end_idx = base_train_end + fold * test_rows
        test_end_idx = min(train_end_idx + test_rows, n)
        if train_end_idx >= n or (test_end_idx - train_end_idx) <= 10:
            break
        windows.append({
            'fold': fold + 1,
            'mode': 'chronological_expanding',
            'train_start': ts.iloc[0],
            'train_end': ts.iloc[train_end_idx - 1],
            'test_start': ts.iloc[train_end_idx],
            'test_end': ts.iloc[test_end_idx - 1] + pd.Timedelta(microseconds=1),
            'train_rows_est': int(train_end_idx),
            'test_rows_est': int(test_end_idx - train_end_idx),
        })
    return windows


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def compute_eval_visual_embeddings(
    test_csv: str,
    test_lob: str,
    test_lob_ts: str,
    models_dir: str,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    df = _load_csv(test_csv)
    diagnostics = {
        'source': 'zeros',
        'reason': 'init',
        'visual_model_type': None,
        'rows_total': int(len(df)),
        'lob_tensors_available': 0,
        'lob_timestamps_available': 0,
        'rows_with_tensor': 0,
        'rows_with_tensor_ratio': 0.0,
        'used_tensor_count': 0,
        'rows_with_visual': 0,
        'visual_coverage_ratio': 0.0,
    }
    zero = np.zeros((len(df), VISUAL_EMB_DIM), dtype=np.float32)

    def _return(out: np.ndarray, *, source: str, reason: str, extra: dict | None = None):
        diagnostics.update({
            'source': source,
            'reason': reason,
        })
        if extra:
            diagnostics.update(extra)
        if out.size:
            rows_with_visual = int((np.linalg.norm(out, axis=1) > 0).sum())
            diagnostics['rows_with_visual'] = rows_with_visual
            diagnostics['visual_coverage_ratio'] = round(_safe_ratio(rows_with_visual, len(out)), 4)
        if return_diagnostics:
            return out, diagnostics
        return out

    schema_path = os.path.join(models_dir, 'feature_schema_v19.json')
    schema = {}
    if os.path.exists(schema_path):
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
        except Exception:
            schema = {}
    deeplob_cfg = (schema.get('deeplob') or {}) if isinstance(schema, dict) else {}
    visual_model_type = _resolve_visual_model_type(deeplob_cfg.get('model_type'))
    visual_artifact = (
        deeplob_cfg.get('model_artifact')
        or ('lob_transformer_v19.keras' if visual_model_type == 'lob_transformer' else 'deeplob_cnn_v19.keras')
    )
    visual_model_path = os.path.join(models_dir, visual_artifact)
    diagnostics['visual_model_type'] = visual_model_type

    if visual_model_type == 'lob_transformer':
        if not LOB_TRANSFORMER_AVAILABLE:
            return _return(zero, source='zeros', reason='lob_transformer_unavailable')
    else:
        if not DEEPLOB_AVAILABLE:
            return _return(zero, source='zeros', reason='deeplob_unavailable')

    if not os.path.exists(visual_model_path):
        return _return(zero, source='zeros', reason=f'missing_visual_model:{visual_model_type}')

    lob_tensors, lob_timestamps = _load_lob_inputs(test_lob, test_lob_ts)
    if lob_tensors is None or lob_timestamps is None:
        return _return(zero, source='zeros', reason='missing_lob_inputs')

    diagnostics['lob_tensors_available'] = int(len(lob_tensors))
    diagnostics['lob_timestamps_available'] = int(len(lob_timestamps))

    row_to_tensor, _, _ = _align_lob_to_rows(df, lob_timestamps)
    rows_with_tensor = int((row_to_tensor >= 0).sum())
    diagnostics['rows_with_tensor'] = rows_with_tensor
    diagnostics['rows_with_tensor_ratio'] = round(_safe_ratio(rows_with_tensor, len(df)), 4)

    if visual_model_type == 'lob_transformer':
        encoder = LOBTransformer(brain_file=visual_model_path)
    else:
        encoder = DeepLOBCNN(brain_file=visual_model_path)
    if encoder.model is None or not encoder._fitted:
        return _return(zero, source='zeros', reason=f'encoder_not_fitted:{visual_model_type}')

    used_tensor_ids = np.unique(row_to_tensor[row_to_tensor >= 0]).astype(np.int32)
    diagnostics['used_tensor_count'] = int(len(used_tensor_ids))
    emb_lookup = {}
    if len(used_tensor_ids):
        X = np.asarray(lob_tensors[used_tensor_ids], dtype=np.float32)
        emb = np.asarray(encoder.get_embeddings(X), dtype=np.float32)
        for i, tid in enumerate(used_tensor_ids):
            emb_lookup[int(tid)] = emb[i]

    out = zero.copy()
    for row_idx, tensor_idx in enumerate(row_to_tensor):
        if int(tensor_idx) in emb_lookup:
            out[row_idx] = emb_lookup[int(tensor_idx)][:VISUAL_EMB_DIM]
    return _return(out, source=f'{visual_model_type}_eval', reason='ok')


def aggregate_fold_metrics(fold_reports: list[dict]) -> dict:
    if not fold_reports:
        return {
            'n_folds': 0,
            'mean_directional_precision': 0.0,
            'mean_directional_recall': 0.0,
            'mean_directional_f1': 0.0,
            'mean_event_gate_rate': 0.0,
            'mean_brier_score': 0.0,
            'mean_ece': 0.0,
            'mean_win_rate': 0.0,
            'mean_profit_factor': 0.0,
            'mean_trade_sharpe': 0.0,
            'mean_trade_pnl_dollars': 0.0,
            'mean_expectancy_dollars': 0.0,
            'max_drawdown_pct': 0.0,
            'total_trades': 0,
            'total_pnl_dollars': 0.0,
        }

    def _backtest_block(report: dict) -> dict:
        block = report.get('backtest_base')
        if isinstance(block, dict):
            return block
        block = report.get('backtest')
        if isinstance(block, dict):
            return block
        return {}

    blocks = [_backtest_block(report) for report in fold_reports]

    return {
        'n_folds': int(len(fold_reports)),
        'mean_directional_precision': float(np.mean([block.get('directional_precision_macro', 0.0) for block in blocks])),
        'mean_directional_recall': float(np.mean([block.get('directional_recall_macro', 0.0) for block in blocks])),
        'mean_directional_f1': float(np.mean([block.get('directional_f1_macro', 0.0) for block in blocks])),
        'mean_event_gate_rate': float(np.mean([block.get('event_gate_rate', 0.0) for block in blocks])),
        'mean_brier_score': float(np.mean([block.get('brier_score', 0.0) for block in blocks])),
        'mean_ece': float(np.mean([block.get('ece', 0.0) for block in blocks])),
        'mean_win_rate': float(np.mean([block.get('win_rate', 0.0) for block in blocks])),
        'mean_profit_factor': float(np.mean([block.get('profit_factor', 0.0) for block in blocks])),
        'mean_trade_sharpe': float(np.mean([block.get('trade_sharpe', 0.0) for block in blocks])),
        'mean_trade_pnl_dollars': float(np.mean([block.get('avg_trade_pnl_dollars', 0.0) for block in blocks])),
        'mean_expectancy_dollars': float(np.mean([block.get('avg_trade_expectancy_dollars', 0.0) for block in blocks])),
        'max_drawdown_pct': float(np.max([block.get('max_drawdown_pct', 0.0) for block in blocks])),
        'total_trades': int(np.sum([block.get('trades', 0) for block in blocks])),
        'total_pnl_dollars': float(np.sum([block.get('total_pnl_dollars', 0.0) for block in blocks])),
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
    label_horizon = int(ref_cfg.get('label_horizon', 150))
    event_roll_window = int(ref_cfg.get('event_roll_window', 50))
    regime_window = int(ref_cfg.get('regime_window', 50))
    warmup_rows = max(500, event_roll_window * 6, regime_window * 4, 200)
    tail_rows = max(label_horizon * 3, 50)

    mbo_df = normalize_ts(read_market_data(mbo_path))
    windows = build_walkforward_windows(
        timestamps=mbo_df['ts_event'],
        n_splits=int(walk_cfg.get('n_splits', 3)),
        initial_train_frac=float(walk_cfg.get('initial_train_frac', 0.55)),
        test_frac=float(walk_cfg.get('test_frac', 0.15)),
        min_train_rows=int(walk_cfg.get('min_train_rows', 1000)),
    )

    fold_reports = []
    release_blockers: list[dict] = []
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
            causal_threshold_mode=str(ref_cfg.get('causal_threshold_mode', 'expanding')),
            tp_mult=float(ref_cfg.get('tp_mult', 1.2)),
            sl_mult=float(ref_cfg.get('sl_mult', 1.0)),
            kalman_slope_threshold=float(ref_cfg.get('kalman_slope_threshold', 0.05)),
            trend_strength_min=float(ref_cfg.get('trend_strength_min', 0.05)),
            regime_mode=str(ref_cfg.get('regime_mode', 'rules')),
            regime_stride=int(ref_cfg.get('regime_stride', 50)),
            regime_window=int(ref_cfg.get('regime_window', 50)),
            regime_progress_every=int(ref_cfg.get('regime_progress_every', 25000)),
            lob_event_sample=int(ref_cfg.get('lob_event_sample', 100000)),
            merge_tolerance_ms=int(ref_cfg.get('merge_tolerance_ms', 500)),
            tail_rows=tail_rows,
            trim_to_score_window=True,
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
            causal_threshold_mode=str(ref_cfg.get('causal_threshold_mode', 'expanding')),
            tp_mult=float(ref_cfg.get('tp_mult', 1.2)),
            sl_mult=float(ref_cfg.get('sl_mult', 1.0)),
            kalman_slope_threshold=float(ref_cfg.get('kalman_slope_threshold', 0.05)),
            trend_strength_min=float(ref_cfg.get('trend_strength_min', 0.05)),
            regime_mode=str(ref_cfg.get('regime_mode', 'rules')),
            regime_stride=int(ref_cfg.get('regime_stride', 50)),
            regime_window=int(ref_cfg.get('regime_window', 50)),
            regime_progress_every=int(ref_cfg.get('regime_progress_every', 25000)),
            lob_event_sample=int(ref_cfg.get('lob_event_sample', 100000)),
            merge_tolerance_ms=int(ref_cfg.get('merge_tolerance_ms', 500)),
            warmup_rows=warmup_rows,
            tail_rows=tail_rows,
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
        base_bt_cfg = {
            'tick_size': float(bt_cfg.get('tick_size', 0.0001)),
            'tick_value': float(bt_cfg.get('tick_value', 10.0)),
            'round_trip_cost_pips': float(bt_cfg.get('round_trip_cost_pips', 1.0)),
            'commission_per_side': float(bt_cfg.get('commission_per_side', 0.5 * float(bt_cfg.get('tick_value', 10.0)))),
            'min_spread_ticks': float(bt_cfg.get('min_spread_ticks', 1.0)),
            'min_slippage_ticks': float(bt_cfg.get('min_slippage_ticks', 1.0)),
            'spread_multiplier': float(bt_cfg.get('spread_multiplier', 0.5)),
            'max_size': int(bt_cfg.get('max_size', 5)),
            'starting_equity': float(bt_cfg.get('starting_equity', 100000.0)),
            'latency_rows': int(bt_cfg.get('latency_rows', 1)),
            'max_daily_loss_pct': float(bt_cfg.get('max_daily_loss_pct', 0.02)),
            'direction_threshold_ticks': float(ref_cfg.get('direction_threshold_ticks', 1.0)),
            'tp_mult': float(ref_cfg.get('tp_mult', 1.2)),
            'sl_mult': float(ref_cfg.get('sl_mult', 1.0)),
            'score_start_ts': str(window['test_start']),
            'score_end_ts': str(window['test_end']),
        }
        _, _, backtest_base = run_causal_backtest(
            df=test_df,
            models_dir=model_dir,
            output_dir=os.path.join(eval_dir, 'base'),
            visual_embeddings=test_visual,
            meta_features=None,
            input_scaled=True,
            scenario_name='base',
            **base_bt_cfg,
        )
        stress_bt_cfg = {
            **base_bt_cfg,
            'commission_per_side': float(base_bt_cfg['commission_per_side']) * 2.0,
            'min_slippage_ticks': float(base_bt_cfg['min_slippage_ticks']) * 2.0,
            'latency_rows': max(int(base_bt_cfg['latency_rows']), 2),
        }
        _, _, backtest_stress = run_causal_backtest(
            df=test_df,
            models_dir=model_dir,
            output_dir=os.path.join(eval_dir, 'stress'),
            visual_embeddings=test_visual,
            meta_features=None,
            input_scaled=True,
            scenario_name='stress',
            **stress_bt_cfg,
        )

        class_counts = {
            'LONG': int((pd.to_numeric(test_df.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int8) == 0).sum()),
            'SHORT': int((pd.to_numeric(test_df.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int8) == 1).sum()),
            'NEUTRAL': int((pd.to_numeric(test_df.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int8) == 2).sum()),
        }
        blockers = []
        if int(backtest_base.get('trades', 0)) <= 0:
            blockers.append('zero_directional_trades')
        if min(class_counts['LONG'], class_counts['SHORT']) <= 0:
            blockers.append('severe_class_collapse')
        if float(backtest_base.get('avg_trade_expectancy_dollars', 0.0)) < 0 and float(backtest_stress.get('avg_trade_expectancy_dollars', 0.0)) < 0:
            blockers.append('negative_expectancy_base_and_stress')
        if blockers:
            release_blockers.append({'fold': int(window['fold']), 'reasons': blockers})

        fold_report = {
            'fold': window['fold'],
            'window': {
                'train_start': str(window['train_start']),
                'train_end': str(window['train_end']),
                'test_start': str(window['test_start']),
                'test_end': str(window['test_end']),
            },
            'class_balance': class_counts,
            'train': train_summary,
            'backtest_base': backtest_base,
            'backtest_stress': backtest_stress,
            'release_blockers': blockers,
        }
        fold_reports.append(fold_report)
        with open(os.path.join(fold_dir, 'fold_report.json'), 'w') as f:
            json.dump(fold_report, f, indent=2)

    aggregate = aggregate_fold_metrics(fold_reports)
    aggregate['release_blocker_count'] = int(len(release_blockers))
    gate_report = evaluate_release_gates(aggregate, gates)
    gates_path = save_gate_report(output_dir, gate_report)
    walkforward_report = {
        'folds': fold_reports,
        'aggregate': aggregate,
        'release_blockers': release_blockers,
    }
    with open(os.path.join(output_dir, 'walkforward_report.json'), 'w') as f:
        json.dump(walkforward_report, f, indent=2)
    monitor_baseline = {
        'generated_at': pd.Timestamp.utcnow().replace(microsecond=0).isoformat(),
        'prediction_baseline': {
            'confidence_mean': 0.0,
            'event_gate_rate': float(aggregate.get('mean_event_gate_rate', 0.0)),
        },
        'shadow_baseline': {
            'brier_score': float(aggregate.get('mean_brier_score', 0.0)),
            'win_rate': float(aggregate.get('mean_win_rate', 0.0)),
        },
    }
    with open(os.path.join(output_dir, 'monitor_baseline.json'), 'w') as f:
        json.dump(monitor_baseline, f, indent=2)
    manifest_path = write_manifest(
        output_dir=output_dir,
        kind='walkforward_v19',
        config=config,
        inputs={'mbo': mbo_path, 'mbp': mbp_path},
        metrics=aggregate,
        extra={'release_gates': gate_report, 'windows': windows, 'release_blockers': release_blockers},
    )

    out = {
        'folds': fold_reports,
        'aggregate': aggregate,
        'release_gates': gate_report,
        'release_blockers': release_blockers,
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
