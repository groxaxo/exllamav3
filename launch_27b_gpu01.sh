#!/usr/bin/env bash
# launch_27b_gpu01.sh
# Optimized TP2 launch for Qwen3.5-27B-exl3 on GPUs 0,1 (RTX 3090 pair)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-exl3-dev}"

_BIND_HOST="${HOST:-0.0.0.0}"
_BIND_PORT="${PORT:-1234}"
MODEL_DIR="${MODEL_DIR:-/home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3}"
MODEL_ID="${MODEL_ID:-Qwen3.5-27B-exl3-3090-tp2}"
GPU_SPLIT="${GPU_SPLIT:-22.0,22.0}"
CACHE_TOKENS="${CACHE_TOKENS:-131072}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-16000}"
MIN_MAX_TOKENS="${MIN_MAX_TOKENS:-12}"
MAX_MAX_TOKENS="${MAX_MAX_TOKENS:-16000}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
PRESERVE_THINK_OUTPUT="${PRESERVE_THINK_OUTPUT:-true}"
MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-1024}"
EXLLAMA_EMBED_PREFER_CPU="${EXLLAMA_EMBED_PREFER_CPU:-0}"
EXLLAMA_STARTUP_WARMUP="${EXLLAMA_STARTUP_WARMUP:-1}"
EXLLAMA_FP16_REDUCE_THRESHOLD="${EXLLAMA_FP16_REDUCE_THRESHOLD:-65536}"
# NCCL tuning for PCIe-connected 3090s (no NVLink)
NCCL_ALGO="${NCCL_ALGO:-TREE}"
TP_BACKEND="${TP_BACKEND:-nccl}"

LOG_DIR="${LOG_DIR:-/home/op/exllamav3_ampere/logs}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOG_DIR}/qwen27b-exl3-gpu01-${TIMESTAMP}.log"
CURRENT_LOG="${LOG_DIR}/qwen27b-exl3-gpu01-current.log"

if [[ ! -f "$CONDA_SH" ]]; then
    printf 'Conda activation script not found: %s\n' "$CONDA_SH" >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    printf 'Model directory not found: %s\n' "$MODEL_DIR" >&2
    exit 1
fi

touch "$LOGFILE"
ln -sfn "$LOGFILE" "$CURRENT_LOG"

printf 'Starting Qwen3.5-27B-exl3 on GPUs 0,1 (TP2/NCCL, optimized)\n'
printf '  Model dir   : %s\n' "$MODEL_DIR"
printf '  Host:Port   : %s:%s\n' "$_BIND_HOST" "$_BIND_PORT"
printf '  Model ID    : %s\n' "$MODEL_ID"
printf '  GPU split   : %s\n' "$GPU_SPLIT"
printf '  Cache tokens: %s\n' "$CACHE_TOKENS"
printf '  TP backend  : %s (NCCL_ALGO=%s)\n' "$TP_BACKEND" "$NCCL_ALGO"
printf '  Reduce thr  : %s\n' "$EXLLAMA_FP16_REDUCE_THRESHOLD"
printf '  Max tokens  : %s\n' "$DEFAULT_MAX_TOKENS"
printf '  Thinking    : %s (preserve=%s, budget=%s)\n' "$ENABLE_THINKING" "$PRESERVE_THINK_OUTPUT" "$MAX_THINKING_TOKENS"
printf '  Log file    : %s\n' "$LOGFILE"
printf '  Current log : %s\n' "$CURRENT_LOG"
printf '\n'

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONUNBUFFERED=1
# NCCL 2.28.9 requires CUDA 12.8 libcudart; prepend nvidia/cuda_runtime/lib before torch/lib
_ENV_LIB="/home/op/miniconda3/envs/${CONDA_ENV}/lib"
_SP="${_ENV_LIB}/python3.11/site-packages"
export LD_LIBRARY_PATH="${_ENV_LIB}:${_SP}/nvidia/cuda_runtime/lib:${_SP}/nvidia/nccl/lib:${_SP}/torch/lib:${LD_LIBRARY_PATH:-}"
unset _SP
unset _ENV_LIB
HOST="$_BIND_HOST"
PORT="$_BIND_PORT"
export MODEL_DIR
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
export EXLLAMA_STARTUP_WARMUP
export EXLLAMA_FP16_REDUCE_THRESHOLD
export NCCL_ALGO
export TP_BACKEND
export PORT
export HOST

cd "$SCRIPT_DIR"

nohup python server_27b_tp2.py \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL_DIR" \
    --gpu-split "$GPU_SPLIT" \
    --cache-tokens "$CACHE_TOKENS" \
    >> "$LOGFILE" 2>&1 &

SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "${LOG_DIR}/qwen27b-exl3-gpu01.pid"

printf 'Started server with PID %s\n' "$SERVER_PID"
printf 'Health check: curl http://127.0.0.1:%s/health\n' "$PORT"
printf 'Models endpoint: curl http://127.0.0.1:%s/v1/models\n' "$PORT"
printf '\nMonitor logs: tail -f %s\n' "$CURRENT_LOG"
