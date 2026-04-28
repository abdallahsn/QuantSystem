#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate your .venv first."
  exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"

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
"$PYTHON_BIN" -m pip install --no-cache-dir \
  "tensorflow==2.15.0"

echo
echo "[3/4] Installing CUDA 12.2 runtime packages that match TensorFlow 2.15..."
"$PYTHON_BIN" -m pip install --no-cache-dir \
  "nvidia-cublas-cu12==12.2.5.6" \
  "nvidia-cuda-cupti-cu12==12.2.142" \
  "nvidia-cuda-nvcc-cu12==12.2.140" \
  "nvidia-cuda-nvrtc-cu12==12.2.140" \
  "nvidia-cuda-runtime-cu12==12.2.140" \
  "nvidia-cudnn-cu12==8.9.4.25" \
  "nvidia-cufft-cu12==11.0.8.103" \
  "nvidia-curand-cu12==10.3.3.141" \
  "nvidia-cusolver-cu12==11.5.2.141" \
  "nvidia-cusparse-cu12==12.1.2.141" \
  "nvidia-nccl-cu12==2.16.5" \
  "nvidia-nvjitlink-cu12==12.2.140"

echo
echo "[4/4] Linking libraries into the venv and verifying GPU visibility..."
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fix_tf_gpu_symlinks.sh"
