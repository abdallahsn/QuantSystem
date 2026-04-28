#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
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

echo
echo "Running sanity check..."
"$PYTHON_BIN" server_sanity_check.py
