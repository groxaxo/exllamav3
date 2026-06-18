#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-huihui_qwen3_4b_exl3_6bpw}"
_BIND_HOST="${HOST:-0.0.0.0}"
_BIND_PORT="${PORT:-8001}"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-exl3-dev}"
EXL3_DIR="/home/op/exllamav3_ampere"
MODEL_PATH="${MODEL_PATH:-/home/op/exllamav3_ampere/models/Huihui-Qwen3-4B-exl3-6bpw}"
MODEL_ID="${MODEL_ID:-Huihui-Qwen3-4B-exl3-6bpw}"
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
CACHE_TOKENS="${CACHE_TOKENS:-32768}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
LOGFILE="${LOGFILE:-/home/op/exllamav3_ampere/logs/huihui-qwen3-4b-exl3-6bpw-current.log}"

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory not found: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

HOST="$_BIND_HOST"
PORT="$_BIND_PORT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
export MODEL_DIR="$MODEL_PATH"
export MODEL_ID
export GPU_ID
export PORT
export HOST
export CACHE_TOKENS
export ENABLE_THINKING

mkdir -p "$(dirname "$LOGFILE")"

{
  printf '[%s] Starting %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SESSION"
  printf 'Model:                 %s\n' "$MODEL_PATH"
  printf 'Model ID:              %s\n' "$MODEL_ID"
  printf 'Host:Port:             %s:%s\n' "$HOST" "$PORT"
  printf 'CUDA_VISIBLE_DEVICES:  %s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'GPU_ID:                %s\n' "$GPU_ID"
  printf 'Cache tokens:          %s\n' "$CACHE_TOKENS"
  printf 'Thinking:              %s\n' "$ENABLE_THINKING"
  printf 'Conda env:             %s\n\n' "$CONDA_ENV"
} | tee "$LOGFILE"

cd "$EXL3_DIR"

exec python server_envdriven.py 2>&1 | stdbuf -oL -eL tee -a "$LOGFILE"
