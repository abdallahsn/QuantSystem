#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV="$ROOT_DIR/.venv"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -x "$DEFAULT_VENV/bin/python" ]]; then
    export VIRTUAL_ENV="$DEFAULT_VENV"
  else
    echo "Activate your .venv first, or create $DEFAULT_VENV."
    exit 1
  fi
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
PIP_ARGS=(--no-cache-dir --timeout 300 --retries 20)

retry() {
  local attempts="$1"
  shift
  local try=1
  until "$@"; do
    if [[ "$try" -ge "$attempts" ]]; then
      return 1
    fi
    local sleep_for=$((try * 10))
    echo
    echo "Retry ${try}/${attempts} failed. Sleeping ${sleep_for}s..."
    sleep "$sleep_for"
    try=$((try + 1))
  done
}

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo
echo "[1/4] Removing conflicting TensorFlow / NVIDIA / TensorRT packages..."
"$PYTHON_BIN" -m pip uninstall -y \
  tensorflow tensorflow-cpu tensorflow-estimator keras ml-dtypes tensorboard tensorflow-io-gcs-filesystem \
  tensorrt tensorrt-bindings tensorrt-libs \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 \
  >/dev/null 2>&1 || true

echo
echo "[2/4] Installing TensorFlow 2.15 core..."
retry 3 "$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" \
  "tensorflow==2.15.0"

echo
echo "[3/4] Installing CUDA 12.2 runtime packages that match TensorFlow 2.15..."
CUDA_PACKAGES=(
  "nvidia-cublas-cu12==12.2.5.6"
  "nvidia-cuda-cupti-cu12==12.2.142"
  "nvidia-cuda-nvcc-cu12==12.2.140"
  "nvidia-cuda-nvrtc-cu12==12.2.140"
  "nvidia-cuda-runtime-cu12==12.2.140"
  "nvidia-cudnn-cu12==8.9.4.25"
  "nvidia-cufft-cu12==11.0.8.103"
  "nvidia-curand-cu12==10.3.3.141"
  "nvidia-nvjitlink-cu12==12.2.140"
  "nvidia-cusparse-cu12==12.1.2.141"
  "nvidia-cusolver-cu12==11.5.2.141"
  "nvidia-nccl-cu12==2.16.5"
)

for pkg in "${CUDA_PACKAGES[@]}"; do
  echo "Installing $pkg ..."
  retry 5 "$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" --no-deps "$pkg"
done

echo
echo "[4/4] Linking libraries into the venv and verifying GPU visibility..."
bash "$ROOT_DIR/fix_tf_gpu_symlinks.sh"
