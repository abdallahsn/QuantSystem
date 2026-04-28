#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate your .venv first."
  exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"

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
