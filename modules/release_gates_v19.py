"""
release_gates_v19.py - Release gate evaluation for QuantSystem V19
"""

from __future__ import annotations

import json
import os


def _compare(metric_value: float, rule: dict) -> tuple[bool, str]:
    if 'min' in rule and metric_value < rule['min']:
        return False, f"{metric_value} < min {rule['min']}"
    if 'max' in rule and metric_value > rule['max']:
        return False, f"{metric_value} > max {rule['max']}"
    return True, 'ok'


def evaluate_release_gates(metrics: dict, gates: dict) -> dict:
    rules = gates.get('metrics', {}) or {}
    results = []
    passed = True
    for metric_name, rule in rules.items():
        rule = rule or {}
        if not isinstance(rule, dict):
            passed = False
            results.append({
                'metric': metric_name,
                'value': None,
                'rule': rule,
                'passed': False,
                'reason': 'invalid_gate_rule',
            })
            continue

        required = bool(rule.get('required', True))
        if metric_name not in metrics or metrics.get(metric_name) is None:
            ok = not required
            results.append({
                'metric': metric_name,
                'value': None,
                'rule': rule,
                'passed': bool(ok),
                'reason': 'optional_metric_missing' if ok else 'required_metric_missing',
            })
            passed = passed and ok
            continue

        try:
            value = float(metrics.get(metric_name))
        except Exception:
            results.append({
                'metric': metric_name,
                'value': metrics.get(metric_name),
                'rule': rule,
                'passed': False,
                'reason': 'metric_value_not_numeric',
            })
            passed = False
            continue

        ok, reason = _compare(value, rule)
        results.append({
            'metric': metric_name,
            'value': value,
            'rule': rule,
            'passed': bool(ok),
            'reason': reason,
        })
        passed = passed and ok

    blocker_count = int(metrics.get('release_blocker_count', 0) or 0)
    if blocker_count > 0:
        passed = False
        results.append({
            'metric': 'release_blocker_count',
            'value': float(blocker_count),
            'rule': {'max': 0},
            'passed': False,
            'reason': f'{blocker_count} fold-level release blockers present',
        })

    return {
        'passed': bool(passed),
        'results': results,
        'required_checks': list(rules.keys()),
    }


def save_gate_report(output_dir: str, report: dict, filename: str = 'release_gates_report.json') -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path
