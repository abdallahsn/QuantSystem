#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=""
ENV_KIND="system"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  ENV_KIND="venv"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  ENV_KIND="conda"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No active python interpreter was found. Activate your .venv or Conda env first."
  exit 1
fi

TF_DIR="$("$PYTHON_BIN" - <<'PY'
import os
import tensorflow as tf
print(os.path.dirname(tf.__file__))
PY
)"

TF_SO="$("$PYTHON_BIN" - <<'PY'
import os
import tensorflow as tf
so_path = os.path.join(os.path.dirname(tf.__file__), "python", "_pywrap_tensorflow_internal.so")
print(so_path)
PY
)"

echo "Python: $PYTHON_BIN"
echo "Environment: $ENV_KIND"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
echo "TensorFlow dir: $TF_DIR"
echo "TensorFlow core .so: $TF_SO"

echo
echo "Installed NVIDIA wheels:"
"$PYTHON_BIN" -m pip list | grep '^nvidia-' || true

echo
echo "Top-level TensorFlow GPU symlinks:"
find "$TF_DIR" -maxdepth 1 -type l \
  \( -name 'libcu*.so*' -o -name 'libnv*.so*' -o -name 'libcudnn*.so*' -o -name 'libnccl*.so*' \) \
  -print | sort || true

echo
echo "Missing shared libraries according to ldd:"
ldd "$TF_SO" | grep 'not found' || echo "No missing libraries reported by ldd."

echo
echo "TensorFlow GPU probe:"
"$PYTHON_BIN" - <<'PY'
import tensorflow as tf
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
PY
