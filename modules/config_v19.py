"""
config_v19.py - Config loading helpers for QuantSystem V19
"""

from __future__ import annotations

import copy
import json
import os


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'configs',
    'v19',
    'defaults.yaml',
)

DEFAULT_GATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'configs',
    'v19',
    'release_gates.yaml',
)

DEFAULT_PRODUCTION_GATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'configs',
    'v19',
    'release_gates.production.yaml',
)


def _load_any(path: str) -> dict:
    with open(path) as f:
        raw = f.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(raw)
        return data or {}
    except Exception:
        return json.loads(raw)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_v19_config(path: str | None = None) -> dict:
    cfg = _load_any(DEFAULT_CONFIG_PATH)
    if path:
        cfg = _deep_merge(cfg, _load_any(path))
    return cfg


def load_release_gates(path: str | None = None) -> dict:
    if not path:
        return _load_any(DEFAULT_GATES_PATH)
    resolved = os.path.abspath(path)
    if resolved == os.path.abspath(DEFAULT_PRODUCTION_GATES_PATH):
        return _load_any(path)
    gates = _load_any(DEFAULT_GATES_PATH)
    return _deep_merge(gates, _load_any(path))


def default_release_gates_path_for_profile(profile: str | None = None) -> str:
    name = str(profile or '').strip().lower()
    if name == 'production':
        return DEFAULT_PRODUCTION_GATES_PATH
    return DEFAULT_GATES_PATH
