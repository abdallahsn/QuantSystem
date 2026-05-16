from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

from modules.config_v19 import default_release_gates_path_for_profile
from walkforward_v19 import run_walkforward


ROOT = Path(__file__).resolve().parents[1]


def test_default_release_gates_path_exists():
    path = Path(default_release_gates_path_for_profile("research"))

    assert path.exists()
    assert path.name == "release_gates.yaml"


def test_run_walkforward_accepts_config_path_kwarg():
    sig = inspect.signature(run_walkforward)

    assert "config_path" in sig.parameters
    assert sig.parameters["config_path"].default is None


def test_readiness_help_imports_successfully():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("QUANTSYSTEM_SKIP_HEAVY_ML", "1")

    proc = subprocess.run(
        [sys.executable, "readiness_v19.py", "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "production readiness evaluator" in proc.stdout
