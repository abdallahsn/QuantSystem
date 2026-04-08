"""
manifest_v19.py - Artifact manifest generation for QuantSystem V19
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os


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
        json.dump(manifest, f, indent=2)
    return path
