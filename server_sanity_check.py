#!/usr/bin/env python3
"""Quick server sanity check for Conda, TensorFlow, NVIDIA GPU, and core libs."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT / "requirements.txt"

IMPORT_NAME_OVERRIDES = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
}


def run_command(command: list[str], timeout: int = 15) -> dict:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": f"command not found: {command[0]}",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": 124,
            "stdout": "",
            "stderr": f"command timed out after {timeout}s",
        }

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def parse_requirements(path: Path) -> list[dict]:
    if not path.exists():
        return [
            {"distribution": "numpy", "min_version": "1.24"},
            {"distribution": "pandas", "min_version": "2.0"},
            {"distribution": "scikit-learn", "min_version": "1.3"},
            {"distribution": "scipy", "min_version": "1.11"},
            {"distribution": "matplotlib", "min_version": "3.7"},
            {"distribution": "plotly", "min_version": "5.0"},
            {"distribution": "openpyxl", "min_version": "3.1"},
            {"distribution": "pyyaml", "min_version": "6.0"},
            {"distribution": "pyarrow", "min_version": "12.0"},
            {"distribution": "catboost", "min_version": "1.2"},
            {"distribution": "tensorflow", "min_version": "2.13"},
            {"distribution": "tqdm", "min_version": "4.66"},
        ]

    packages = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\s*>=\s*([A-Za-z0-9_.-]+))?", line)
        if match:
            packages.append(
                {
                    "distribution": match.group(1),
                    "min_version": match.group(2) or "",
                }
            )
    return packages


def safe_version(distribution_name: str, module) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except Exception:
        return getattr(module, "__version__", "unknown")


def version_tuple(version_text: str) -> tuple[int, ...]:
    if not version_text:
        return ()
    parts = re.findall(r"\d+", version_text)
    return tuple(int(part) for part in parts[:3])


def import_name_for(distribution_name: str) -> str:
    lowered = distribution_name.lower()
    return IMPORT_NAME_OVERRIDES.get(lowered, distribution_name.replace("-", "_"))


def check_python_and_env() -> dict:
    env_kind = "system"
    if os.environ.get("VIRTUAL_ENV"):
        env_kind = "venv"
    elif os.environ.get("CONDA_PREFIX"):
        env_kind = "conda"

    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "venv": os.environ.get("VIRTUAL_ENV", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "env_kind": env_kind,
    }


def check_conda() -> dict:
    info = {
        "installed": shutil.which("conda") is not None,
        "active": bool(os.environ.get("CONDA_PREFIX")),
        "version": "",
        "base_prefix": "",
        "active_prefix_name": "",
        "active_prefix": os.environ.get("CONDA_PREFIX", ""),
        "env_name": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "error": "",
    }

    if not info["installed"]:
        info["error"] = "conda command not found"
        return info

    version_proc = run_command(["conda", "--version"])
    if version_proc["ok"]:
        info["version"] = version_proc["stdout"].replace("conda ", "").strip()
    elif version_proc["stderr"]:
        info["error"] = version_proc["stderr"]

    json_proc = run_command(["conda", "info", "--json"], timeout=20)
    if json_proc["ok"]:
        try:
            import json

            payload = json.loads(json_proc["stdout"])
            info["base_prefix"] = payload.get("root_prefix", "")
            info["active_prefix_name"] = payload.get("active_prefix_name", "")
            if not info["env_name"]:
                info["env_name"] = payload.get("active_prefix_name", "")
            if not info["active_prefix"]:
                info["active_prefix"] = payload.get("active_prefix", "")
            info["active"] = bool(info["active_prefix"])
        except Exception as exc:
            if not info["error"]:
                info["error"] = f"failed to parse conda info: {exc}"

    return info


def check_nvidia() -> dict:
    info = {
        "nvidia_smi_found": shutil.which("nvidia-smi") is not None,
        "gpu_count": 0,
        "cuda_version": "",
        "driver_version": "",
        "gpus": [],
        "error": "",
        "nvcc_version": "",
    }

    if info["nvidia_smi_found"]:
        query = run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        if query["ok"]:
            for line in query["stdout"].splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 4:
                    gpu = {
                        "index": parts[0],
                        "name": parts[1],
                        "driver_version": parts[2],
                        "memory_total_mb": parts[3],
                    }
                    info["gpus"].append(gpu)
            info["gpu_count"] = len(info["gpus"])
            if info["gpus"]:
                info["driver_version"] = info["gpus"][0]["driver_version"]
        else:
            info["error"] = query["stderr"] or "nvidia-smi query failed"

        full = run_command(["nvidia-smi"], timeout=15)
        if full["ok"]:
            match = re.search(r"CUDA Version:\s*([0-9.]+)", full["stdout"])
            if match:
                info["cuda_version"] = match.group(1)

    else:
        info["error"] = "nvidia-smi command not found"

    if shutil.which("nvcc") is not None:
        nvcc = run_command(["nvcc", "--version"], timeout=10)
        if nvcc["ok"]:
            match = re.search(r"release\s+([0-9.]+)", nvcc["stdout"])
            if match:
                info["nvcc_version"] = match.group(1)

    return info


def check_package(requirement: dict) -> dict:
    distribution_name = requirement["distribution"]
    min_version = requirement.get("min_version", "")
    import_name = import_name_for(distribution_name)
    result = {
        "distribution": distribution_name,
        "import_name": import_name,
        "required_min_version": min_version,
        "ok": False,
        "version": "",
        "version_ok": True,
        "path": "",
        "error": "",
    }

    try:
        module = importlib.import_module(import_name)
        result["ok"] = True
        result["version"] = safe_version(distribution_name, module)
        result["path"] = str(getattr(module, "__file__", "") or "")
        if min_version and version_tuple(result["version"]) and version_tuple(min_version):
            result["version_ok"] = version_tuple(result["version"]) >= version_tuple(min_version)
            if not result["version_ok"]:
                result["error"] = f"installed version {result['version']} is below required >= {min_version}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def check_tensorflow() -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    info = {
        "installed": False,
        "version": "",
        "module_path": "",
        "build_cuda_version": "",
        "build_cudnn_version": "",
        "is_cuda_build": None,
        "gpu_count": 0,
        "gpus": [],
        "smoke_ok": False,
        "smoke_device": "",
        "error": "",
    }

    try:
        import tensorflow as tf

        info["installed"] = True
        info["version"] = getattr(tf, "__version__", "unknown")
        info["module_path"] = str(getattr(tf, "__file__", "") or "")

        try:
            build_info = tf.sysconfig.get_build_info()
            info["build_cuda_version"] = str(build_info.get("cuda_version", "") or "")
            info["build_cudnn_version"] = str(build_info.get("cudnn_version", "") or "")
            info["is_cuda_build"] = bool(build_info.get("is_cuda_build"))
        except Exception:
            try:
                info["is_cuda_build"] = bool(tf.test.is_built_with_cuda())
            except Exception:
                info["is_cuda_build"] = None

        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_count"] = len(gpus)

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
            try:
                details = tf.config.experimental.get_device_details(gpu)
            except Exception:
                details = {}
            info["gpus"].append(
                {
                    "name": details.get("device_name", gpu.name),
                    "device_type": gpu.device_type,
                    "logical_name": gpu.name,
                }
            )

        device_name = "/GPU:0" if gpus else "/CPU:0"
        with tf.device(device_name):
            a = tf.random.uniform((256, 256))
            b = tf.random.uniform((256, 256))
            c = tf.matmul(a, b)
            _ = c.numpy()
            info["smoke_ok"] = True
            info["smoke_device"] = getattr(c, "device", device_name)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    return info


def evaluate(env_info: dict, conda_info: dict, nvidia_info: dict, tf_info: dict, packages: list[dict]) -> dict:
    warns = []
    fails = []

    if conda_info["installed"] and not conda_info["active"] and not env_info.get("venv"):
        warns.append("Conda is installed but no active Conda environment was detected.")

    if env_info.get("env_kind") == "system":
        warns.append("No isolated Python environment detected; prefer a project .venv before training.")

    conda_prefix = conda_info.get("active_prefix", "") or env_info.get("conda_prefix", "")
    venv_prefix = env_info.get("venv", "")
    python_in_conda = bool(conda_prefix and sys.executable.startswith(conda_prefix))
    python_in_venv = bool(venv_prefix and sys.executable.startswith(venv_prefix))

    if conda_prefix and venv_prefix:
        if python_in_venv:
            warns.append(
                "Both Conda and virtualenv are active; current Python is coming from VIRTUAL_ENV, not CONDA_PREFIX."
            )
        elif not python_in_conda:
            fails.append(
                "Both Conda and virtualenv are set, but current Python executable is outside both environment prefixes."
            )
    elif conda_prefix and not python_in_conda:
        fails.append("Active Conda environment detected, but current Python executable is outside CONDA_PREFIX.")
    elif venv_prefix and not python_in_venv:
        fails.append("Active virtualenv detected, but current Python executable is outside VIRTUAL_ENV.")

    missing_packages = [
        pkg["distribution"]
        for pkg in packages
        if not pkg["ok"] and pkg["distribution"].lower() != "tensorflow"
    ]
    if missing_packages:
        fails.append("Missing or broken required packages: " + ", ".join(missing_packages))

    version_mismatches = [
        f"{pkg['distribution']} ({pkg['version']} < {pkg['required_min_version']})"
        for pkg in packages
        if pkg["ok"] and not pkg["version_ok"]
    ]
    if version_mismatches:
        fails.append("Installed package versions are below requirements: " + ", ".join(version_mismatches))

    if not tf_info["installed"]:
        fails.append("TensorFlow is not installed or failed to import.")
    elif tf_info["error"]:
        fails.append("TensorFlow import/runtime check failed: " + tf_info["error"])

    if tf_info["installed"] and conda_prefix and not venv_prefix and tf_info["module_path"]:
        if not tf_info["module_path"].startswith(conda_prefix):
            fails.append("TensorFlow is imported from outside the active Conda environment.")

    if not nvidia_info["nvidia_smi_found"]:
        warns.append("nvidia-smi is not available, so GPU/driver status could not be confirmed from the OS.")
    elif nvidia_info["gpu_count"] == 0:
        fails.append("nvidia-smi is installed but no NVIDIA GPUs were detected.")

    if nvidia_info["gpu_count"] > 0 and tf_info["installed"]:
        if tf_info["gpu_count"] == 0:
            fails.append("NVIDIA GPUs are visible to the OS, but TensorFlow does not see any GPU.")
        if tf_info["is_cuda_build"] is False:
            fails.append("TensorFlow appears to be a CPU-only build while NVIDIA GPUs are present.")

    driver_cuda = nvidia_info.get("cuda_version", "")
    tf_cuda = tf_info.get("build_cuda_version", "")
    if driver_cuda and tf_cuda:
        if version_tuple(driver_cuda) and version_tuple(tf_cuda):
            if version_tuple(driver_cuda) < version_tuple(tf_cuda):
                fails.append(
                    f"Driver max CUDA version ({driver_cuda}) is lower than TensorFlow build CUDA version ({tf_cuda})."
                )

    if tf_info["installed"] and tf_info["gpu_count"] > 0 and not tf_info["smoke_ok"]:
        fails.append("TensorFlow detected GPU(s), but the TensorFlow smoke test did not complete.")

    overall = "PASS"
    if fails:
        overall = "FAIL"
    elif warns:
        overall = "WARN"

    return {
        "overall": overall,
        "warns": warns,
        "fails": fails,
    }


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_kv(key: str, value: str) -> None:
    print(f"{key:<24}: {value}")


def main() -> int:
    env_info = check_python_and_env()
    conda_info = check_conda()
    nvidia_info = check_nvidia()
    package_requirements = parse_requirements(REQUIREMENTS_FILE)
    packages = [check_package(req) for req in package_requirements]
    tf_info = check_tensorflow()
    verdict = evaluate(env_info, conda_info, nvidia_info, tf_info, packages)

    print_header("SERVER SANITY CHECK")
    print_kv("Python", env_info["python_version"])
    print_kv("Executable", env_info["python_executable"])
    print_kv("Platform", env_info["platform"])
    print_kv("Working Dir", env_info["cwd"])
    print_kv("Environment", env_info["env_kind"])
    print_kv("VIRTUAL_ENV", env_info["venv"] or "-")
    print_kv("CONDA_DEFAULT_ENV", env_info["conda_env"] or "-")
    print_kv("CONDA_PREFIX", env_info["conda_prefix"] or "-")
    print_kv("CUDA_VISIBLE_DEVICES", env_info["cuda_visible_devices"] or "-")

    print_header("CONDA")
    print_kv("Installed", "yes" if conda_info["installed"] else "no")
    print_kv("Active Env", "yes" if conda_info["active"] else "no")
    print_kv("Conda Version", conda_info["version"] or "-")
    print_kv("Env Name", conda_info["env_name"] or "-")
    print_kv("Active Prefix", conda_info["active_prefix"] or "-")
    if not conda_info["installed"]:
        print_kv("Conda Note", "optional; this repo works with venv + pip")
    elif conda_info["error"]:
        print_kv("Conda Note", conda_info["error"])

    print_header("NVIDIA / CUDA")
    print_kv("nvidia-smi", "found" if nvidia_info["nvidia_smi_found"] else "missing")
    print_kv("GPU Count", str(nvidia_info["gpu_count"]))
    print_kv("Driver Version", nvidia_info["driver_version"] or "-")
    print_kv("Driver CUDA Max", nvidia_info["cuda_version"] or "-")
    print_kv("nvcc Version", nvidia_info["nvcc_version"] or "-")
    if nvidia_info["error"]:
        print_kv("NVIDIA Note", nvidia_info["error"])
    for gpu in nvidia_info["gpus"]:
        label = f"GPU {gpu['index']}"
        value = f"{gpu['name']} | {gpu['memory_total_mb']} MiB | driver {gpu['driver_version']}"
        print_kv(label, value)

    print_header("CORE PACKAGES")
    for package in packages:
        if package["ok"] and package["version_ok"]:
            where = package["path"] or "-"
            print(f"[OK]   {package['distribution']:<16} version={package['version']:<12} path={where}")
        elif package["ok"]:
            print(
                f"[FAIL] {package['distribution']:<16} version={package['version']:<12} "
                f"requires>={package['required_min_version']}"
            )
        else:
            print(f"[FAIL] {package['distribution']:<16} {package['error']}")

    print_header("TENSORFLOW")
    print_kv("Installed", "yes" if tf_info["installed"] else "no")
    print_kv("Version", tf_info["version"] or "-")
    print_kv("Module Path", tf_info["module_path"] or "-")
    print_kv("CUDA Build", str(tf_info["is_cuda_build"]))
    print_kv("Build CUDA", tf_info["build_cuda_version"] or "-")
    print_kv("Build cuDNN", tf_info["build_cudnn_version"] or "-")
    print_kv("Visible GPUs", str(tf_info["gpu_count"]))
    print_kv("Smoke Test", "passed" if tf_info["smoke_ok"] else "failed")
    print_kv("Smoke Device", tf_info["smoke_device"] or "-")
    if tf_info["error"]:
        print_kv("TensorFlow Note", tf_info["error"])
    for index, gpu in enumerate(tf_info["gpus"]):
        print_kv(f"TF GPU {index}", f"{gpu['name']} | {gpu['logical_name']}")

    print_header("FINAL VERDICT")
    print_kv("Overall", verdict["overall"])
    if verdict["fails"]:
        print("Failures:")
        for item in verdict["fails"]:
            print(f"  - {item}")
    if verdict["warns"]:
        print("Warnings:")
        for item in verdict["warns"]:
            print(f"  - {item}")

    if verdict["overall"] == "PASS":
        print("\nResult: server environment looks ready for TensorFlow + GPU work.")
    elif verdict["overall"] == "WARN":
        print("\nResult: server is usable, but review the warnings above.")
    else:
        print("\nResult: there are blocking issues that should be fixed before training.")

    return 0 if verdict["overall"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
