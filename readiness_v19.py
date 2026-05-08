"""
readiness_v19.py - Unified production readiness evaluator for QuantSystem V19
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from modules.config_v19 import default_release_gates_path_for_profile, load_release_gates, load_v19_config
from modules.failsafe_v19 import decide_runtime_mode, evaluate_system_health
from modules.raw_replay_v19 import normalize_ts, read_market_data
from walkforward_v19 import run_walkforward


def _read_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _detect_contract_ids(mbo_path: str) -> list[str]:
    df = normalize_ts(read_market_data(mbo_path))
    if 'symbol' in df.columns:
        values = df['symbol']
    elif 'instrument_id' in df.columns:
        values = df['instrument_id']
    else:
        return []
    unique = sorted({str(value) for value in values.dropna().astype(str).tolist() if str(value).strip()})
    return unique


def _summarize_drift(models_dir: str) -> dict:
    report = _read_json(os.path.join(models_dir, 'feature_coverage_drift_report.json'))
    months = report.get('months', []) or []
    max_missing = 0.0
    max_psi = 0.0
    for month in months:
        features = month.get('features', {}) or {}
        for feature_stats in features.values():
            max_missing = max(max_missing, float(feature_stats.get('missing_rate', 0.0) or 0.0))
            max_psi = max(max_psi, float(feature_stats.get('psi', 0.0) or 0.0))
    return {
        'available': bool(report),
        'months_evaluated': int(len(months)),
        'dead_features_3m': list(report.get('dead_features_3m', []) or []),
        'max_missing_ratio': float(max_missing),
        'max_psi': float(max_psi),
        'critical_alert': bool(max_missing > 0.05 or max_psi > 0.20),
        'path': os.path.join(models_dir, 'feature_coverage_drift_report.json'),
    }


def _collect_fold_gate_status(full_walkforward_dir: str, folds: list[dict]) -> dict:
    integrity_reports = []
    training_reports = []
    missing_reports = []
    for fold in folds:
        fold_dir = os.path.join(full_walkforward_dir, f"fold_{int(fold.get('fold', 0)):02d}")
        for rel_path, bucket in (
            ('train_data/data_integrity_gate_report.json', integrity_reports),
            ('test_data/data_integrity_gate_report.json', integrity_reports),
            ('models/event_training_gate_report.json', training_reports),
        ):
            path = os.path.join(fold_dir, rel_path)
            payload = _read_json(path)
            if payload:
                bucket.append({'path': path, **payload})
            else:
                missing_reports.append(path)
    integrity_passed = all(bool(report.get('passed', False)) for report in integrity_reports) if integrity_reports else False
    training_passed = all(bool(report.get('passed', False)) for report in training_reports) if training_reports else False
    return {
        'integrity_reports_found': int(len(integrity_reports)),
        'training_reports_found': int(len(training_reports)),
        'missing_reports': missing_reports,
        'integrity_passed': bool(integrity_passed),
        'training_passed': bool(training_passed),
    }


def _latest_fold_models_dir(full_walkforward_dir: str, folds: list[dict]) -> str | None:
    if not folds:
        return None
    last_fold = max(int(fold.get('fold', 0) or 0) for fold in folds)
    models_dir = os.path.join(full_walkforward_dir, f'fold_{last_fold:02d}', 'models')
    return models_dir if os.path.isdir(models_dir) else None


def build_readiness_report(
    *,
    mbo_path: str,
    mbp_path: str,
    output_dir: str,
    config: dict,
    gates: dict,
    config_path: str | None = None,
    models_dir: str | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    full_walkforward_dir = os.path.join(output_dir, 'full_walkforward')
    full_summary = run_walkforward(
        mbo_path=mbo_path,
        mbp_path=mbp_path,
        output_dir=full_walkforward_dir,
        config=config,
        gates=gates,
        config_path=config_path,
    )
    folds = list(full_summary.get('folds', []) or [])
    aggregate = dict(full_summary.get('aggregate', {}) or {})
    gate_status = _collect_fold_gate_status(full_walkforward_dir, folds)
    selected_models_dir = models_dir or _latest_fold_models_dir(full_walkforward_dir, folds)
    drift_summary = _summarize_drift(selected_models_dir) if selected_models_dir else {'available': False, 'critical_alert': True}
    contracts = _detect_contract_ids(mbo_path)

    monthly_rows = [
        {
            'month': fold.get('test_month'),
            'trades': int(((fold.get('backtest_base') or {}).get('trades', 0) or 0)),
            'total_pnl_dollars': float(((fold.get('backtest_base') or {}).get('total_pnl_dollars', 0.0) or 0.0)),
            'passed': len(fold.get('release_blockers', []) or []) == 0,
        }
        for fold in folds
        if fold.get('test_month')
    ]
    monthly_evaluable = int(len(monthly_rows))
    contract_slices = []
    if len(contracts) == 1:
        contract_slices.append({
            'contract_id': contracts[0],
            'passed': bool(aggregate.get('contract_pass_rate', 0.0) >= 1.0),
            'total_trades': int(aggregate.get('total_trades', 0) or 0),
            'total_pnl_dollars': float(aggregate.get('total_pnl_dollars', 0.0) or 0.0),
        })

    legacy_blockers = []
    shadow_health = {}
    rollout_mode = {}
    if selected_models_dir:
        shadow_health = evaluate_system_health(
            models_dir=selected_models_dir,
            engine_status={
                'catboost_available': True,
                'meta_available': True,
                'regime_available': True,
                'visual_available': True,
            },
            feature_row={},
            policy={**(config.get('failsafe', {}) or {}), **(config.get('rollout', {}) or {})},
            manifest_path=os.path.join(selected_models_dir, 'manifest.json'),
        )
        rollout_mode = decide_runtime_mode(
            shadow_health,
            policy={**(config.get('failsafe', {}) or {}), **(config.get('rollout', {}) or {})},
        )
    else:
        legacy_blockers.append('selected_models_dir_missing')

    release_passed = bool((full_summary.get('release_gates') or {}).get('passed', False))
    shadow_ready = bool(
        gate_status.get('integrity_passed', False)
        and gate_status.get('training_passed', False)
        and release_passed
        and not legacy_blockers
        and bool(selected_models_dir)
    )
    paper_ready = bool(
        shadow_ready
        and bool((shadow_health.get('shadow_approval') or {}).get('available', False))
        and bool((shadow_health.get('shadow_approval') or {}).get('passed', False))
        and not bool(drift_summary.get('critical_alert', True))
    )
    live_ready = bool(
        paper_ready
        and bool(rollout_mode.get('allow_rollout', False))
        and not bool(drift_summary.get('critical_alert', True))
    )

    state = 'research_only'
    if monthly_evaluable >= 3 and len(contract_slices) >= 1:
        if live_ready:
            state = 'live_ready'
        elif paper_ready:
            state = 'paper_ready'
        elif shadow_ready:
            state = 'shadow_ready'

    report = {
        'generated_at': pd.Timestamp.utcnow().replace(microsecond=0).isoformat(),
        'profile': str(config.get('profile', 'research')),
        'full_walkforward_dir': full_walkforward_dir,
        'selected_models_dir': selected_models_dir,
        'state': state,
        'shadow_ready': bool(shadow_ready),
        'paper_ready': bool(paper_ready),
        'live_ready': bool(live_ready),
        'release_gates': full_summary.get('release_gates', {}),
        'aggregate': aggregate,
        'integrity_training_gates': gate_status,
        'monthly_slices': {
            'evaluable_count': monthly_evaluable,
            'rows': monthly_rows,
        },
        'contract_slices': {
            'evaluable_count': int(len(contract_slices)),
            'rows': contract_slices,
            'detected_contracts': contracts,
        },
        'drift_summary': drift_summary,
        'shadow_health': shadow_health,
        'rollout_mode': rollout_mode,
        'legacy_release_dependencies_detected': legacy_blockers,
        'blockers': {
            'insufficient_months': bool(monthly_evaluable < 3),
            'missing_contract_slices': bool(len(contract_slices) < 1),
            'critical_drift_or_calibration': bool(drift_summary.get('critical_alert', True)),
            'release_gates_failed': not release_passed,
            'integrity_gates_failed': not bool(gate_status.get('integrity_passed', False)),
            'training_gates_failed': not bool(gate_status.get('training_passed', False)),
            'shadow_approval_issue': not bool((shadow_health.get('shadow_approval') or {}).get('passed', False)),
            'kill_switch_or_runtime_blocker': not bool(rollout_mode.get('allow_rollout', False)) if rollout_mode else True,
        },
        'source_summary': full_summary,
    }
    path = os.path.join(output_dir, 'readiness_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description='QuantSystem V19 production readiness evaluator')
    p.add_argument('--mbo', required=True)
    p.add_argument('--mbp', required=True)
    p.add_argument('--output', default='outputs_v19_readiness')
    p.add_argument('--config', default=None)
    p.add_argument('--gates', default=None)
    p.add_argument('--models', default=None, help='optional trained model bundle dir for shadow/live checks')
    args = p.parse_args()

    config = load_v19_config(args.config)
    gates_path = args.gates or default_release_gates_path_for_profile(config.get('profile'))
    gates = load_release_gates(gates_path)
    report = build_readiness_report(
        mbo_path=args.mbo,
        mbp_path=args.mbp,
        output_dir=args.output,
        config=config,
        gates=gates,
        config_path=args.config,
        models_dir=args.models,
    )
    print("\n✅ Readiness evaluation complete")
    print(json.dumps({
        'state': report['state'],
        'shadow_ready': report['shadow_ready'],
        'paper_ready': report['paper_ready'],
        'live_ready': report['live_ready'],
    }, indent=2))


if __name__ == '__main__':
    main()
