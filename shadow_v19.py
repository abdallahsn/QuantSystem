"""
shadow_v19.py - Shadow mode runner for QuantSystem V19
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from modules.config_v19 import load_v19_config
from modules.feature_artifact_v19 import load_feature_artifact
from modules.logging_v19 import EventLogWriter, log_event
from modules.monitoring_v19 import MonitoringState, emit_alerts, load_baseline_from_artifacts, load_jsonl, write_monitoring_outputs
from predict_v19 import V19PredictionEngine


def _load_visual_embeddings(path: str | None, n_rows: int, n_dim: int) -> np.ndarray:
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


def run_shadow(csv_path: str, models_dir: str, output_dir: str, input_scaled: bool = True, visual_npy: str | None = None, config: dict | None = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    cfg = config or load_v19_config()
    log_cfg = cfg.get('logging', {})
    failsafe_cfg = cfg.get('failsafe', {})
    events_path = os.path.join(output_dir, log_cfg.get('events_file', 'shadow_predictions.jsonl'))
    writer = EventLogWriter(events_path)
    outcome_writer = EventLogWriter(os.path.join(output_dir, 'shadow_outcomes.jsonl'))

    engine = V19PredictionEngine(
        models_dir=models_dir,
        run_mode='shadow',
        event_writer=writer,
        manifest_path=os.path.join(models_dir, 'manifest.json'),
        failsafe_policy=failsafe_cfg,
    )
    df = load_feature_artifact(csv_path)
    canonical_df = engine.factory.prepare_frame(df, already_scaled=input_scaled, include_meta=True)
    visual_embeddings = _load_visual_embeddings(visual_npy, len(canonical_df), len(engine.visual_features))

    realized = 0
    emitted = 0
    blocked = 0

    for i, (_, row) in enumerate(canonical_df.iterrows()):
        ts = row.get('ts_event', None)
        pred = engine.predict_step(
            row.to_dict(),
            visual_embedding=visual_embeddings[i] if len(engine.visual_features) else None,
            ts=ts,
            already_scaled=True,
        )
        pred['idx'] = i
        if pred.get('sequence_ready', False):
            emitted += 1
            if 'bias_label' in canonical_df.columns:
                true_bias = int(row.get('bias_label', 2) or 2)
                correct = pred.get('bias_idx', -1) == true_bias
                log_event(
                    outcome_writer,
                    'shadow_outcome_realized',
                    {
                        'run_mode': 'shadow',
                        'bias': pred.get('bias', 'NEUTRAL'),
                        'bias_idx': pred.get('bias_idx', -1),
                        'confidence': float(pred.get('confidence', 0.0) or 0.0),
                        'chosen_threshold': float(pred.get('chosen_threshold', 0.0) or 0.0),
                        'tradeable': bool(pred.get('tradeable', False)),
                        'feature_hash': (((pred.get('feature_hash')) or '') if isinstance(pred, dict) else ''),
                        'extra': {
                            'prediction_idx': i,
                            'true_bias': true_bias,
                            'correct': bool(correct),
                            'resolved_immediately': True,
                            'direction_probs': pred.get('direction_probs', {}),
                            'raw_direction_probs': pred.get('raw_direction_probs', {}),
                            'realized_path_outcome': int(row.get('path_outcome', 4) or 4) if 'path_outcome' in canonical_df.columns else None,
                            'adverse_path_flag': int(row.get('adverse_path_flag', 0) or 0) if 'adverse_path_flag' in canonical_df.columns else 0,
                        },
                    },
                    ts=ts,
                )
                realized += 1
        else:
            blocked += 1

        if (i + 1) % int(log_cfg.get('heartbeat_every_rows', 100) or 100) == 0:
            log_event(writer, 'heartbeat', {'run_mode': 'shadow', 'extra': {'row_idx': i + 1}}, ts=ts)

    summary = {
        'rows': int(len(canonical_df)),
        'emitted_predictions': int(emitted),
        'blocked_predictions': int(blocked),
        'realized_outcomes': int(realized),
        'pending_unresolved': int(max(emitted - realized, 0)),
        'events_file': events_path,
    }
    with open(os.path.join(output_dir, 'shadow_daily_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    baseline = load_baseline_from_artifacts(models_dir)
    monitoring = MonitoringState(baseline=baseline).summarize(load_jsonl(events_path) + load_jsonl(os.path.join(output_dir, 'shadow_outcomes.jsonl')))
    alerts = emit_alerts(monitoring, writer=None, config=cfg.get('monitoring', {}))
    monitor_paths = write_monitoring_outputs(output_dir, monitoring, alerts)
    shadow_cfg = cfg.get('shadow', {})
    readiness = {
        'generated_at': pd.Timestamp.utcnow().replace(microsecond=0).isoformat(),
        'passed': True,
        'requirements': {
            'min_emitted_predictions': int(shadow_cfg.get('min_emitted_predictions', 100)),
            'min_realized_outcomes': int(shadow_cfg.get('min_realized_outcomes', 100)),
            'max_brier_score_delta': float(shadow_cfg.get('max_brier_score_delta', 0.20)),
            'max_win_rate_drop': float(shadow_cfg.get('max_win_rate_drop', 0.10)),
            'max_alerts': int(shadow_cfg.get('max_alerts', 0)),
        },
        'summary': {
            'emitted_predictions': int(emitted),
            'realized_outcomes': int(realized),
            'shadow_health': monitoring.get('shadow_health', {}),
            'alerts': int(len(alerts)),
        },
        'failures': [],
    }
    req = readiness['requirements']
    if emitted < req['min_emitted_predictions']:
        readiness['passed'] = False
        readiness['failures'].append('insufficient_emitted_predictions')
    if realized < req['min_realized_outcomes']:
        readiness['passed'] = False
        readiness['failures'].append('insufficient_realized_outcomes')
    if float((monitoring.get('shadow_health') or {}).get('brier_score_delta', 0.0)) > float(req['max_brier_score_delta']):
        readiness['passed'] = False
        readiness['failures'].append('shadow_brier_delta_too_high')
    if float(-((monitoring.get('shadow_health') or {}).get('win_rate_delta', 0.0))) > float(req['max_win_rate_drop']):
        readiness['passed'] = False
        readiness['failures'].append('shadow_win_rate_drop_too_large')
    if len(alerts) > int(req['max_alerts']):
        readiness['passed'] = False
        readiness['failures'].append('too_many_monitoring_alerts')

    readiness_path = os.path.join(output_dir, 'shadow_readiness_report.json')
    with open(readiness_path, 'w') as f:
        json.dump(readiness, f, indent=2)

    approval_path = os.path.join(models_dir, 'shadow_approval.json')
    with open(approval_path, 'w') as f:
        json.dump(readiness, f, indent=2)

    return {**summary, **monitor_paths, 'shadow_readiness_report': readiness_path, 'shadow_approval': approval_path, 'shadow_ready': bool(readiness['passed'])}


def main():
    p = argparse.ArgumentParser(description='QuantSystem V19 shadow mode runner')
    p.add_argument('--csv', required=True)
    p.add_argument('--models', default='outputs_v19')
    p.add_argument('--output', default='outputs_v19_shadow')
    p.add_argument('--visual_npy', default=None)
    p.add_argument('--input_scaled', action='store_true')
    p.add_argument('--config', default=None)
    args = p.parse_args()

    cfg = load_v19_config(args.config)
    summary = run_shadow(
        csv_path=args.csv,
        models_dir=args.models,
        output_dir=args.output,
        input_scaled=args.input_scaled or bool(cfg.get('shadow', {}).get('input_scaled', True)),
        visual_npy=args.visual_npy,
        config=cfg,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
