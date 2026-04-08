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
    rules = gates.get('metrics', {})
    results = []
    passed = True
    for metric_name, rule in rules.items():
        value = float(metrics.get(metric_name, 0.0))
        ok, reason = _compare(value, rule)
        results.append({
            'metric': metric_name,
            'value': value,
            'rule': rule,
            'passed': bool(ok),
            'reason': reason,
        })
        passed = passed and ok

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
