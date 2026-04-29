#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV="$ROOT_DIR/.venv"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -x "$DEFAULT_VENV/bin/python" ]]; then
    export VIRTUAL_ENV="$DEFAULT_VENV"
  else
    echo "VIRTUAL_ENV is not active and $DEFAULT_VENV was not found."
    exit 1
  fi
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

# Clean up stale top-level GPU symlinks from older mismatched installs first.
find . -maxdepth 1 -type l \
  \( -name 'libcu*.so*' -o -name 'libnv*.so*' -o -name 'libcudnn*.so*' -o -name 'libnccl*.so*' \) \
  -print -delete || true

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
try:
    import nvidia.cuda_nvcc
    print(os.path.dirname(os.path.dirname(nvidia.cuda_nvcc.__file__)))
except Exception:
    print("")
PY
)"

PTXAS_BIN=""
if [[ -n "$PTXAS_SRC" ]]; then
  PTXAS_BIN="$(find "$PTXAS_SRC" -path '*/bin/ptxas' -print -quit || true)"
fi
if [[ -z "$PTXAS_BIN" ]]; then
  SITE_PACKAGES_DIR="$(dirname "$TF_DIR")"
  PTXAS_BIN="$(find "$SITE_PACKAGES_DIR"/nvidia -path '*/bin/ptxas' -print -quit 2>/dev/null || true)"
fi

if [[ -n "$PTXAS_BIN" ]]; then
  ln -svf "$PTXAS_BIN" "${VIRTUAL_ENV}/bin/ptxas"
  echo "Linked ptxas: ${VIRTUAL_ENV}/bin/ptxas -> $PTXAS_BIN"
else
  echo "ptxas not found. Continuing without a ptxas symlink."
fi

echo
echo "Verifying TensorFlow GPU visibility..."
"$PYTHON_BIN" - <<'PY'
import tensorflow as tf

print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
PY
