"""
manifest_v19.py - Artifact manifest generation for QuantSystem V19
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - keep helper usable in lighter contexts
    pd = None


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts(root_dir: str) -> list[dict]:
    artifacts = []
    if not os.path.exists(root_dir):
        return artifacts
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isfile(path):
            continue
        artifacts.append({
            'name': name,
            'path': path,
            'size_bytes': int(os.path.getsize(path)),
            'sha256': _sha256_file(path),
        })
    return artifacts


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if pd is not None:
        if value is pd.NaT or value is pd.NA:
            return None
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return value.isoformat()
        if isinstance(value, (pd.Series, pd.Index)):
            return [to_jsonable(v) for v in value.tolist()]
        if isinstance(value, pd.DataFrame):
            return [to_jsonable(row) for row in value.to_dict(orient='records')]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
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


def write_manifest(
    output_dir: str,
    kind: str,
    config: dict | None = None,
    inputs: dict | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
    filename: str = 'manifest.json',
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        'kind': kind,
        'created_at_utc': _dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        'output_dir': output_dir,
        'config': config or {},
        'inputs': inputs or {},
        'metrics': metrics or {},
        'artifacts': collect_artifacts(output_dir),
        'extra': extra or {},
    }
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(to_jsonable(manifest), f, indent=2)
    return path
