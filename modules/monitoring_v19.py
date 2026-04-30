"""
monitoring_v19.py - Monitoring, drift, and alerting for QuantSystem V19
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from modules.logging_v19 import EventLogWriter, log_event


def load_jsonl(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _psi(ref_mean: float, ref_std: float, obs_mean: float, obs_std: float) -> float:
    ref_std = max(abs(ref_std), 1e-8)
    obs_std = max(abs(obs_std), 1e-8)
    return float(abs(obs_mean - ref_mean) / ref_std + abs(obs_std - ref_std) / ref_std)


def _extract_feature_vectors(events: list[dict]) -> dict[str, list[float]]:
    bucket = defaultdict(list)
    for event in events:
        feats = (((event or {}).get('extra') or {}).get('stat_features') or {})
        for key, value in feats.items():
            try:
                bucket[key].append(float(value))
            except Exception:
                continue
    return dict(bucket)


def compute_drift(events: list[dict], baseline: dict) -> dict:
    vectors = _extract_feature_vectors(events)
    feature_drift = {}
    for feat, vals in vectors.items():
        base = (((baseline.get('feature_baseline') or {}).get(feat)) or {})
        ref_mean = float(base.get('mean', 0.0))
        ref_std = float(base.get('std', 1.0))
        obs_mean = _safe_mean(vals)
        obs_std = _safe_std(vals)
        feature_drift[feat] = {
            'mean': obs_mean,
            'std': obs_std,
            'psi_proxy': _psi(ref_mean, ref_std, obs_mean, obs_std),
            'baseline_mean': ref_mean,
            'baseline_std': ref_std,
            'count': len(vals),
        }

    pred_biases = [e.get('bias', 'NEUTRAL') for e in events if e.get('event_type') == 'prediction_emitted']
    clusters = [int(e.get('cluster', 0) or 0) for e in events if e.get('event_type') == 'prediction_emitted']
    confidences = [float(e.get('confidence', 0.0) or 0.0) for e in events if e.get('event_type') == 'prediction_emitted']

    prediction_drift = {
        'bias_distribution': dict(Counter(pred_biases)),
        'cluster_distribution': dict(Counter(clusters)),
        'confidence_mean': _safe_mean(confidences),
        'confidence_std': _safe_std(confidences),
        'confidence_mean_delta': _safe_mean(confidences) - float((baseline.get('prediction_baseline') or {}).get('confidence_mean', 0.0)),
        'event_gate_rate': _safe_mean([1.0 if e.get('event_gate_passed', False) else 0.0 for e in events if e.get('event_type') in ('prediction_emitted', 'prediction_blocked')]),
    }
    return {
        'feature_drift': feature_drift,
        'prediction_drift': prediction_drift,
    }


class MonitoringState:
    def __init__(self, baseline: dict | None = None):
        self.baseline = baseline or {}

    def summarize(self, events: list[dict]) -> dict:
        pred_events = [e for e in events if e.get('event_type') in ('prediction_emitted', 'prediction_blocked')]
        emitted = [e for e in pred_events if e.get('event_type') == 'prediction_emitted']
        shadow_outcomes = [e for e in events if e.get('event_type') == 'shadow_outcome_realized']
        blocked = [e for e in events if e.get('event_type') in ('prediction_blocked', 'risk_blocked', 'rollout_guard_triggered')]
        fills = [e for e in events if e.get('event_type') == 'order_filled']
        rejects = [e for e in events if e.get('event_type') == 'order_rejected']
        fallbacks = [e for e in events if e.get('event_type') in ('visual_branch_unavailable', 'regime_fallback_used')]
        gaps = [e for e in events if e.get('event_type') == 'data_gap_detected']
        latencies = [float(e.get('latency_ms', 0.0) or 0.0) for e in pred_events]
        confidences = [float(e.get('confidence', 0.0) or 0.0) for e in emitted]
        missing_counts = [len((((e.get('extra') or {}).get('missing_features')) or [])) for e in pred_events]
        slippages = [float(((e.get('extra') or {}).get('slippage_pips') or 0.0)) for e in fills]
        event_gate_rate = _safe_mean([1.0 if e.get('event_gate_passed', False) else 0.0 for e in pred_events])
        shadow_probs = []
        shadow_true = []
        shadow_correct = []
        for e in shadow_outcomes:
            extra = (e.get('extra') or {})
            probs = extra.get('direction_probs') or {}
            shadow_probs.append(float(probs.get('LONG', 0.0) or 0.0) if isinstance(probs, dict) else 0.0)
            shadow_true.append(int(extra.get('true_bias', 2) or 2))
            shadow_correct.append(bool(extra.get('correct', False)))
        directional_mask = [yb in (0, 1) for yb in shadow_true]
        brier = 0.0
        if shadow_probs and any(directional_mask):
            p = np.asarray([p for p, keep in zip(shadow_probs, directional_mask) if keep], dtype=np.float64)
            y = np.asarray([yb for yb, keep in zip(shadow_true, directional_mask) if keep], dtype=np.int32)
            brier = float(np.mean((p - (y == 0).astype(np.float64)) ** 2)) if len(y) else 0.0
        shadow_win_rate = _safe_mean([1.0 if ok else 0.0 for ok in shadow_correct])
        shadow_baseline = self.baseline.get('shadow_baseline', {}) or {}

        drift = compute_drift(events, self.baseline)
        return {
            'counts': {
                'events': len(events),
                'predictions': len(pred_events),
                'prediction_emitted': len(emitted),
                'blocked': len(blocked),
                'fills': len(fills),
                'rejects': len(rejects),
                'fallbacks': len(fallbacks),
                'data_gaps': len(gaps),
            },
            'data_quality': {
                'mean_missing_features': _safe_mean(missing_counts),
                'missing_ratio_events': float(np.mean([c > 0 for c in missing_counts])) if missing_counts else 0.0,
                'data_gap_events': len(gaps),
            },
            'execution_health': {
                'fill_ratio': len(fills) / max(len(fills) + len(rejects), 1),
                'reject_ratio': len(rejects) / max(len(fills) + len(rejects), 1),
                'realized_slippage_mean': _safe_mean(slippages),
                'latency_p50': _percentile(latencies, 50),
                'latency_p95': _percentile(latencies, 95),
                'latency_p99': _percentile(latencies, 99),
            },
            'risk_health': {
                'blocked_frequency': len(blocked) / max(len(pred_events), 1),
                'fallback_frequency': len(fallbacks) / max(len(events), 1),
            },
            'prediction_health': {
                'confidence_mean': _safe_mean(confidences),
                'confidence_std': _safe_std(confidences),
                'event_gate_rate': event_gate_rate,
                'event_gate_rate_delta': event_gate_rate - float((self.baseline.get('prediction_baseline') or {}).get('event_gate_rate', 0.0)),
            },
            'shadow_health': {
                'realized_outcomes': int(len(shadow_outcomes)),
                'brier_score': float(brier),
                'win_rate': shadow_win_rate,
                'brier_score_delta': float(brier - float(shadow_baseline.get('brier_score', 0.0))),
                'win_rate_delta': float(shadow_win_rate - float(shadow_baseline.get('win_rate', 0.0))),
            },
            'drift': drift,
        }


def _default_alert_rules(config: dict | None = None) -> dict:
    c = config or {}
    return {
        'feature_missing_ratio_max': float(c.get('feature_missing_ratio_max', 0.05)),
        'latency_p95_max_ms': float(c.get('latency_p95_max_ms', 500.0)),
        'confidence_mean_delta_max': float(c.get('confidence_mean_delta_max', 0.15)),
        'fallback_frequency_max': float(c.get('fallback_frequency_max', 0.10)),
        'reject_ratio_max': float(c.get('reject_ratio_max', 0.15)),
        'data_gap_events_max': int(c.get('data_gap_events_max', 0)),
        'psi_proxy_max': float(c.get('psi_proxy_max', 0.25)),
        'psi_warn': float(c.get('psi_warn', c.get('psi_proxy_max', 0.20))),
        'psi_critical': float(c.get('psi_critical', max(c.get('psi_proxy_max', 0.25), 0.35))),
        'event_gate_shift_warn': float(c.get('event_gate_shift_warn', 0.30)),
        'shadow_brier_deterioration_critical': float(c.get('shadow_brier_deterioration_critical', 0.20)),
        'shadow_win_rate_drop_critical': float(c.get('shadow_win_rate_drop_critical', 0.10)),
    }


def emit_alerts(summary: dict, writer: EventLogWriter | None = None, config: dict | None = None) -> list[dict]:
    rules = _default_alert_rules(config)
    alerts = []

    def add_alert(name: str, reason: str, extra: dict | None = None):
        payload = {'reject_reason': reason, 'extra': extra or {}}
        alerts.append(log_event(writer, name, payload, level='WARNING'))

    dq = summary.get('data_quality', {})
    ex = summary.get('execution_health', {})
    rh = summary.get('risk_health', {})
    ph = summary.get('prediction_health', {})
    sh = summary.get('shadow_health', {})
    drift = summary.get('drift', {})

    if float(dq.get('missing_ratio_events', 0.0)) > rules['feature_missing_ratio_max']:
        add_alert('feature_missing', f"missing_ratio {dq.get('missing_ratio_events')} > {rules['feature_missing_ratio_max']}")
    if float(ex.get('latency_p95', 0.0)) > rules['latency_p95_max_ms']:
        add_alert('latency_alert', f"latency_p95 {ex.get('latency_p95')} > {rules['latency_p95_max_ms']}")
    if abs(float((drift.get('prediction_drift') or {}).get('confidence_mean_delta', 0.0))) > rules['confidence_mean_delta_max']:
        add_alert('confidence_drift', f"confidence_mean_delta exceeded threshold")
    if float(rh.get('fallback_frequency', 0.0)) > rules['fallback_frequency_max']:
        add_alert('fallback_frequency_alert', f"fallback_frequency {rh.get('fallback_frequency')} > {rules['fallback_frequency_max']}")
    if float(ex.get('reject_ratio', 0.0)) > rules['reject_ratio_max']:
        add_alert('execution_reject_alert', f"reject_ratio {ex.get('reject_ratio')} > {rules['reject_ratio_max']}")
    if int(dq.get('data_gap_events', 0)) > rules['data_gap_events_max']:
        add_alert('data_gap_detected', f"data_gap_events {dq.get('data_gap_events')} > {rules['data_gap_events_max']}")

    if abs(float(ph.get('event_gate_rate_delta', 0.0))) > rules['event_gate_shift_warn']:
        add_alert('event_gate_shift_alert', 'event_gate_rate shifted materially from baseline')
    if float(sh.get('brier_score_delta', 0.0)) > rules['shadow_brier_deterioration_critical']:
        add_alert('shadow_brier_alert', 'shadow brier score deteriorated beyond baseline tolerance')
    if float(-sh.get('win_rate_delta', 0.0)) > rules['shadow_win_rate_drop_critical']:
        add_alert('shadow_win_rate_alert', 'shadow win rate dropped materially from baseline')
    for feat, stat in (drift.get('feature_drift') or {}).items():
        if float(stat.get('psi_proxy', 0.0)) > rules['psi_critical']:
            add_alert('feature_drift_alert', f"{feat} psi_proxy {stat.get('psi_proxy')} > {rules['psi_critical']}", extra={'feature': feat, 'stats': stat})
        elif float(stat.get('psi_proxy', 0.0)) > rules['psi_warn']:
            add_alert('feature_drift_warning', f"{feat} psi_proxy {stat.get('psi_proxy')} > {rules['psi_warn']}", extra={'feature': feat, 'stats': stat})
    return alerts


def load_baseline_from_artifacts(models_dir: str) -> dict:
    manifest_path = os.path.join(models_dir, 'manifest.json')
    scaler_path = os.path.join(models_dir, 'scaler_params.json')
    monitor_baseline_path = os.path.join(models_dir, 'monitor_baseline.json')
    baseline = {'feature_baseline': {}, 'prediction_baseline': {}}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        baseline['manifest'] = manifest
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            scaler = json.load(f)
        for feat, params in scaler.items():
            if params.get('type') == 'robust':
                baseline['feature_baseline'][feat] = {'mean': float(params.get('median', 0.0)), 'std': float(params.get('iqr', 1.0))}
            elif params.get('type') == 'minmax':
                mn = float(params.get('min', 0.0))
                mx = float(params.get('max', 1.0))
                baseline['feature_baseline'][feat] = {'mean': (mn + mx) / 2.0, 'std': max((mx - mn) / 2.0, 1e-8)}
            else:
                baseline['feature_baseline'][feat] = {'mean': 0.0, 'std': 1.0}
    if os.path.exists(monitor_baseline_path):
        try:
            with open(monitor_baseline_path) as f:
                monitor_baseline = json.load(f)
            baseline['prediction_baseline'].update((monitor_baseline.get('prediction_baseline') or {}))
            baseline['shadow_baseline'] = monitor_baseline.get('shadow_baseline', {})
        except Exception:
            pass
    return baseline


def write_monitoring_outputs(output_dir: str, summary: dict, alerts: list[dict]) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'monitoring_summary.json')
    drift_path = os.path.join(output_dir, 'drift_report.json')
    alerts_path = os.path.join(output_dir, 'alerts.jsonl')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    with open(drift_path, 'w') as f:
        json.dump(summary.get('drift', {}), f, indent=2)
    writer = EventLogWriter(alerts_path)
    for alert in alerts:
        writer.write(alert)
    return {
        'monitoring_summary': summary_path,
        'drift_report': drift_path,
        'alerts': alerts_path,
    }
