"""
failsafe_v19.py - Explicit degradation and runtime gating policies for V19
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def _manifest_exists(path: str | None) -> bool:
    return bool(path) and os.path.exists(path)


def _schema_manifest_match(models_dir: str) -> bool:
    return os.path.exists(os.path.join(models_dir, 'feature_schema_v19.json')) and os.path.exists(os.path.join(models_dir, 'manifest.json'))


def evaluate_system_health(
    *,
    models_dir: str,
    engine_status: dict,
    feature_row: dict | None = None,
    ts=None,
    policy: dict | None = None,
    manifest_path: str | None = None,
    loss_guard_status: dict | None = None,
) -> dict:
    policy = policy or {}
    issues = []
    degraded = []

    if not _schema_manifest_match(models_dir):
        issues.append('manifest_schema_mismatch')

    if manifest_path and not _manifest_exists(manifest_path):
        issues.append('manifest_missing')

    if not engine_status.get('catboost_available', False):
        issues.append('catboost_unavailable')
    if not engine_status.get('meta_available', False):
        degraded.append('meta_unavailable')
    if not engine_status.get('regime_available', False):
        degraded.append('regime_fallback')
    if not engine_status.get('visual_available', False):
        degraded.append('visual_branch_unavailable')

    feature_row = feature_row or {}
    critical_features = set(policy.get('critical_features', ['cvd', 'obi', 'micro_atr', 'kyle_lambda', 'hawkes_intensity', 'vwap_z_score']))
    missing_critical = sorted([f for f in critical_features if feature_row.get(f, None) is None or pd.isna(feature_row.get(f, None))])
    if missing_critical:
        degraded.append('feature_missing')

    stale_seconds = float(policy.get('stale_timestamp_seconds', 120))
    if ts is not None:
        now_ts = pd.Timestamp.utcnow().tz_localize(None)
        event_ts = pd.Timestamp(ts)
        if event_ts.tzinfo is not None:
            event_ts = event_ts.tz_convert(None)
        if (now_ts - event_ts).total_seconds() > stale_seconds:
            issues.append('timestamp_stale')

    lg = loss_guard_status or {}
    if lg.get('blocked', False):
        issues.append('loss_guard_blocked')

    return {
        'healthy': len(issues) == 0,
        'blocking_issues': issues,
        'degraded_components': sorted(set(degraded)),
        'missing_critical_features': missing_critical,
        'engine_status': engine_status,
    }


def decide_runtime_mode(health: dict, policy: dict | None = None) -> dict:
    policy = policy or {}
    degraded = set(health.get('degraded_components', []))
    issues = set(health.get('blocking_issues', []))
    kill_switch_file = policy.get('kill_switch_file')
    manual_override = str(policy.get('manual_override_state', 'auto')).lower()

    allow_shadow = True
    allow_paper = True
    allow_rollout = True
    reasons = []

    if kill_switch_file and os.path.exists(kill_switch_file):
        issues.add('kill_switch_active')
    if manual_override in ('shadow_only', 'shadow'):
        allow_paper = False
        allow_rollout = False
        reasons.append('manual_override_shadow_only')
    elif manual_override in ('paper_only',):
        allow_rollout = False
        reasons.append('manual_override_paper_only')

    if 'manifest_schema_mismatch' in issues or 'manifest_missing' in issues or 'kill_switch_active' in issues:
        allow_paper = False
        allow_rollout = False
        reasons.extend(sorted(issues & {'manifest_schema_mismatch', 'manifest_missing', 'kill_switch_active'}))

    if 'catboost_unavailable' in issues:
        allow_paper = False
        allow_rollout = False
        reasons.append('catboost_required_for_non_shadow')

    if 'timestamp_stale' in issues or 'loss_guard_blocked' in issues:
        allow_paper = False
        allow_rollout = False
        reasons.extend(sorted(issues & {'timestamp_stale', 'loss_guard_blocked'}))

    if 'meta_unavailable' in degraded:
        allow_rollout = False
        reasons.append('meta_unavailable_rollout_block')

    if 'visual_branch_unavailable' in degraded and not bool(policy.get('allow_visual_degraded_in_rollout', False)):
        allow_rollout = False
        reasons.append('visual_degraded_rollout_block')

    if 'feature_missing' in degraded and not bool(policy.get('allow_missing_features_in_paper', True)):
        allow_paper = False
        allow_rollout = False
        reasons.append('critical_feature_missing')
    elif 'feature_missing' in degraded:
        allow_rollout = False
        reasons.append('critical_feature_missing_rollout_block')

    return {
        'allow_shadow': bool(allow_shadow),
        'allow_paper': bool(allow_paper),
        'allow_rollout': bool(allow_rollout),
        'degraded_components': sorted(degraded),
        'reason': '; '.join(reasons) if reasons else 'OK',
        'blocking_issues': sorted(issues),
    }
