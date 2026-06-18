#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-qwen27b_3090_layersplit}"
_BIND_HOST="${HOST:-0.0.0.0}"
_BIND_PORT="${PORT:-1234}"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-exl3-dev}"
EXL3_DIR="/home/op/exllamav3_ampere"
MODEL_PATH="${MODEL_PATH:-/home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3}"
MODEL_ID="${MODEL_ID:-Qwen3.5-27B-exl3-3090-layersplit}"
GPU_SPLIT="${GPU_SPLIT:-22.0,22.0}"
CACHE_TOKENS="${CACHE_TOKENS:-32768}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-16000}"
MIN_MAX_TOKENS="${MIN_MAX_TOKENS:-12}"
MAX_MAX_TOKENS="${MAX_MAX_TOKENS:-16000}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
PRESERVE_THINK_OUTPUT="${PRESERVE_THINK_OUTPUT:-true}"
MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-1024}"
EXLLAMA_EMBED_PREFER_CPU="${EXLLAMA_EMBED_PREFER_CPU:-0}"
LOGFILE="${LOGFILE:-/home/op/exllamav3_ampere/logs/qwen27b-3090-layersplit-current.log}"

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

HOST="$_BIND_HOST"
PORT="$_BIND_PORT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
_ENV_LIB="/home/op/miniconda3/envs/${CONDA_ENV}/lib"
_SP="${_ENV_LIB}/python3.11/site-packages"
export LD_LIBRARY_PATH="${_ENV_LIB}:${_SP}/nvidia/cuda_runtime/lib:${_SP}/nvidia/nccl/lib:${_SP}/torch/lib:${LD_LIBRARY_PATH:-}"
unset _SP
unset _ENV_LIB
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MODEL_DIR="$MODEL_PATH"
export MODEL_ID
export GPU_SPLIT
export CACHE_TOKENS
export DEFAULT_MAX_TOKENS
export MIN_MAX_TOKENS
export MAX_MAX_TOKENS
export ENABLE_THINKING
export PRESERVE_THINK_OUTPUT
export MAX_THINKING_TOKENS
export EXLLAMA_EMBED_PREFER_CPU
export PORT
export HOST

mkdir -p "$(dirname "$LOGFILE")"

{
  printf '[%s] Starting %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SESSION"
  printf 'Model:                 %s\n' "$MODEL_PATH"
  printf 'Model ID:              %s\n' "$MODEL_ID"
  printf 'Host:Port:             %s:%s\n' "$HOST" "$PORT"
  printf 'CUDA_VISIBLE_DEVICES:  %s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'GPU split:             %s\n' "$GPU_SPLIT"
  printf 'Cache tokens:          %s\n' "$CACHE_TOKENS"
  printf 'OMP_NUM_THREADS:       %s\n' "$OMP_NUM_THREADS"
  printf 'EXLLAMA_EMBED_PREFER_CPU: %s\n' "$EXLLAMA_EMBED_PREFER_CPU"
  printf 'Conda env:             %s\n\n' "$CONDA_ENV"
} | tee "$LOGFILE"

cd "$EXL3_DIR"

exec python server_27b_layer.py \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --gpu-split "$GPU_SPLIT" \
  --cache-tokens "$CACHE_TOKENS" \
  2>&1 | stdbuf -oL -eL tee -a "$LOGFILE"
