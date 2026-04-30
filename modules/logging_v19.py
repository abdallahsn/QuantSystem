"""
logging_v19.py - Structured JSONL logging for QuantSystem V19
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any

import numpy as np


DEFAULT_EVENT_FIELDS = {
    'ts_event': None,
    'ts_logged_utc': None,
    'event_type': '',
    'level': 'INFO',
    'run_mode': 'shadow',
    'dataset_id': '',
    'model_id': '',
    'model_version': 'v19',
    'schema_version': 'v19',
    'symbol': '',
    'sequence_ready': False,
    'event_gate_passed': False,
    'event_gate_reason': '',
    'bias': 'NEUTRAL',
    'bias_idx': -1,
    'confidence': 0.0,
    'chosen_threshold': 0.0,
    'tradeable': False,
    'reject_reason': '',
    'cluster': 0,
    'cluster_name': 'Unknown',
    'position_size': 0,
    'feature_hash': '',
    'visual_norm': 0.0,
    'latency_ms': 0.0,
    'input_source': '',
    'artifacts_manifest': '',
    'extra': {},
}


def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def safe_jsonable(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'isoformat'):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def feature_hash_from_dict(features: dict, feature_order: list[str] | None = None) -> str:
    feature_order = feature_order or sorted(features.keys())
    payload = []
    for key in feature_order:
        val = features.get(key, 0.0)
        try:
            payload.append(f'{key}={float(val):.8f}')
        except Exception:
            payload.append(f'{key}={val}')
    return hashlib.sha256('|'.join(payload).encode('utf-8')).hexdigest()[:16]


class EventLogWriter:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)

    def write(self, event: dict) -> None:
        record = {**DEFAULT_EVENT_FIELDS, **safe_jsonable(event)}
        record['ts_logged_utc'] = record.get('ts_logged_utc') or utc_now_iso()
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def log_event(writer: EventLogWriter | None, event_type: str, payload: dict, ts=None, level: str = 'INFO') -> dict:
    event = {
        **payload,
        'event_type': event_type,
        'level': level,
        'ts_event': safe_jsonable(ts) if ts is not None else payload.get('ts_event'),
    }
    if writer is not None:
        writer.write(event)
    return event


class BaseRuntimeLogger:
    def __init__(
        self,
        writer: EventLogWriter | None,
        run_mode: str,
        manifest_path: str = '',
        model_version: str = 'v19',
        schema_version: str = 'v19',
        symbol: str = '',
    ):
        self.writer = writer
        self.run_mode = run_mode
        self.manifest_path = manifest_path
        self.model_version = model_version
        self.schema_version = schema_version
        self.symbol = symbol
        self.dataset_id = ''
        self.model_id = os.path.basename(os.path.abspath(os.path.dirname(manifest_path))) if manifest_path else ''
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                extra = manifest.get('extra', {}) or {}
                source_contract = extra.get('source_contract', {}) or {}
                self.dataset_id = str(source_contract.get('dataset_id') or extra.get('dataset_id') or '')
                self.model_id = str(extra.get('model_id') or self.model_id)
            except Exception:
                pass

    def _base_payload(self, **kwargs) -> dict:
        return {
            'run_mode': self.run_mode,
            'dataset_id': self.dataset_id,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'schema_version': self.schema_version,
            'symbol': self.symbol,
            'artifacts_manifest': self.manifest_path,
            **kwargs,
        }


class PredictionLogger(BaseRuntimeLogger):
    def log_prediction(self, prediction: dict, *, features: dict, ts=None, input_source: str = '', latency_ms: float = 0.0) -> dict:
        missing = sorted([k for k, v in features.items() if v is None or (isinstance(v, float) and np.isnan(v))])
        payload = self._base_payload(
            sequence_ready=prediction.get('sequence_ready', False),
            event_gate_passed=bool(prediction.get('event_gate_passed', False)),
            event_gate_reason=str(prediction.get('event_gate_reason', '') or ''),
            bias=prediction.get('bias', 'NEUTRAL'),
            bias_idx=int(prediction.get('bias_idx', -1)) if prediction.get('bias_idx') is not None else -1,
            confidence=float(prediction.get('confidence', 0.0) or 0.0),
            chosen_threshold=float(prediction.get('chosen_threshold', 0.0) or 0.0),
            tradeable=bool(prediction.get('tradeable', False)),
            reject_reason=str(prediction.get('reason', '') or ''),
            cluster=int(prediction.get('cluster', 0) or 0),
            cluster_name=prediction.get('cluster_name', 'Unknown'),
            position_size=int(prediction.get('position_size', 0) or 0),
            feature_hash=feature_hash_from_dict(features),
            visual_norm=float(prediction.get('visual_norm', 0.0) or 0.0),
            latency_ms=float(latency_ms),
            input_source=input_source,
            extra={
                'stat_features': safe_jsonable(features),
                'cb_probs': prediction.get('cb_probs', {}),
                'raw_direction_probs': prediction.get('raw_direction_probs', {}),
                'direction_probs': prediction.get('direction_probs', {}),
                'source': prediction.get('source', ''),
                'missing_features': missing,
            },
        )
        event_type = 'prediction_emitted' if prediction.get('sequence_ready', False) else 'prediction_blocked'
        return log_event(self.writer, event_type, payload, ts=ts, level='INFO')


class ExecutionLogger(BaseRuntimeLogger):
    def log_order(self, event_type: str, payload: dict, ts=None, level: str = 'INFO') -> dict:
        return log_event(self.writer, event_type, self._base_payload(**payload), ts=ts, level=level)


class RiskLogger(BaseRuntimeLogger):
    def log_block(self, reason: str, *, ts=None, extra: dict | None = None, event_type: str = 'risk_blocked') -> dict:
        return log_event(
            self.writer,
            event_type,
            self._base_payload(reject_reason=reason, tradeable=False, extra=extra or {}),
            ts=ts,
            level='WARNING',
        )


class DataQualityLogger(BaseRuntimeLogger):
    def log_data_issue(self, event_type: str, *, reason: str, ts=None, extra: dict | None = None, level: str = 'WARNING') -> dict:
        return log_event(
            self.writer,
            event_type,
            self._base_payload(reject_reason=reason, tradeable=False, extra=extra or {}),
            ts=ts,
            level=level,
        )
