#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "VIRTUAL_ENV is not active. Activate your .venv first."
  exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

TF_DIR="$("$PYTHON_BIN" - <<'PY'
import os
import tensorflow as tf
print(os.path.dirname(tf.__file__))
PY
)"

echo "TensorFlow package dir: $TF_DIR"
pushd "$TF_DIR" >/dev/null

shopt -s nullglob
libs=(../nvidia/*/lib/*.so*)
if (( ${#libs[@]} == 0 )); then
  echo "No ../nvidia/*/lib/*.so* libraries were found next to TensorFlow."
  echo "Install tensorflow[and-cuda] first."
  exit 1
fi

for lib in "${libs[@]}"; do
  ln -svf "$lib" .
done
popd >/dev/null

PTXAS_SRC="$("$PYTHON_BIN" - <<'PY'
import os
import nvidia.cuda_nvcc
print(os.path.dirname(os.path.dirname(nvidia.cuda_nvcc.__file__)))
PY
)"

PTXAS_BIN="$(find "$PTXAS_SRC" -path '*/bin/ptxas' -print -quit || true)"
if [[ -n "$PTXAS_BIN" ]]; then
  ln -svf "$PTXAS_BIN" "${VIRTUAL_ENV}/bin/ptxas"
  echo "Linked ptxas: ${VIRTUAL_ENV}/bin/ptxas -> $PTXAS_BIN"
else
  echo "ptxas not found under $PTXAS_SRC"
fi

echo
echo "Verifying TensorFlow GPU visibility..."
"$PYTHON_BIN" - <<'PY'
import tensorflow as tf

print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
PY
