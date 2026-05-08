"""
reproducibility_v19.py - Deterministic runtime helpers for QuantSystem V19
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import random
import sys
from typing import Any

import numpy as np


def _safe_version(module_name: str) -> str | None:
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(mod, "__version__", None) or "")


def _section(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}
    sec = config.get("reproducibility", {})
    return sec if isinstance(sec, dict) else {}


def apply_reproducibility_config(
    config: dict | None,
    *,
    output_dir: str | None = None,
    component: str = "",
) -> dict:
    cfg = _section(config)
    enabled = bool(cfg.get("enabled", True))
    seed = int(cfg.get("seed", 42))
    write_manifest = bool(cfg.get("write_seed_manifest", True))
    deterministic_tf_requested = bool(cfg.get("enable_deterministic_tf_ops", False))

    manifest: dict[str, Any] = {
        "enabled": bool(enabled),
        "component": str(component or ""),
        "seed": int(seed),
        "deterministic_tf_ops_requested": bool(deterministic_tf_requested),
        "deterministic_tf_ops_enabled": False,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        },
        "versions": {
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "sklearn": _safe_version("sklearn"),
            "tensorflow": _safe_version("tensorflow"),
            "catboost": _safe_version("catboost"),
            "xgboost": _safe_version("xgboost"),
        },
    }

    if enabled:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf  # type: ignore

            tf.keras.utils.set_random_seed(seed)
            if deterministic_tf_requested:
                enable = getattr(tf.config.experimental, "enable_op_determinism", None)
                if callable(enable):
                    enable()
                    manifest["deterministic_tf_ops_enabled"] = True
        except Exception as exc:
            manifest["tensorflow_status"] = f"unavailable:{exc}"

    if output_dir and write_manifest:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "seed_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        manifest["path"] = path
    return manifest
