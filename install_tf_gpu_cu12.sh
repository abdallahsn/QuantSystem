#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_VENV="$ROOT_DIR/.venv"

ensure_venv() {
  local base_python
  base_python="$(command -v python3)"

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    return
  fi

  if [[ ! -x "$DEFAULT_VENV/bin/python" ]]; then
    echo "Creating virtual environment at: $DEFAULT_VENV"
    "$base_python" -m venv "$DEFAULT_VENV"
  fi

  PYTHON_BIN="$DEFAULT_VENV/bin/python"
}

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
fi

ensure_venv

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  export VIRTUAL_ENV="$DEFAULT_VENV"
fi

retry() {
  local attempts="$1"
  shift
  local try=1
  until "$@"; do
    if [[ "$try" -ge "$attempts" ]]; then
      return 1
    fi
    local sleep_for=$((try * 5))
    echo
    echo "Retry ${try}/${attempts} failed. Sleeping ${sleep_for}s..."
    sleep "$sleep_for"
    try=$((try + 1))
  done
}

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Virtual env: $VIRTUAL_ENV"

retry 3 "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
retry 3 "$PYTHON_BIN" -m pip install --timeout 300 --retries 20 -r requirements.txt
retry 5 "$PYTHON_BIN" -m pip install --timeout 300 --retries 20 "tensorflow[and-cuda]"

echo
echo "Verifying TensorFlow GPU visibility..."
"$PYTHON_BIN" - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    a = tf.random.uniform((256, 256))
    b = tf.random.uniform((256, 256))
    c = tf.matmul(a, b)
    print("Smoke device:", getattr(c, "device", "unknown"))
    print("Smoke OK:", c.shape)
PY

GPU_COUNT="$("$PYTHON_BIN" - <<'PY'
import tensorflow as tf
print(len(tf.config.list_physical_devices("GPU")))
PY
)"

if [[ "$GPU_COUNT" == "0" ]]; then
  echo
  echo "TensorFlow installed but GPU is not visible yet. Applying symlink repair..."
  bash "$ROOT_DIR/fix_tf_gpu_symlinks.sh"
fi

echo
echo "Running sanity check..."
"$PYTHON_BIN" server_sanity_check.py
