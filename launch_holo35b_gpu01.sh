#!/usr/bin/env bash
# Optimized TP2 launch for Holo3-35B-A3B-exl3 on GPUs 0,1.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-exl3-dev}"

_BIND_HOST="${HOST:-0.0.0.0}"
_BIND_PORT="${PORT:-1234}"
MODEL_DIR="${MODEL_DIR:-/home/op/exllamav3_ampere/models/Holo3-35B-A3B-exl3-6bpw}"
MODEL_ID="${MODEL_ID:-Holo3-35B-A3B-exl3-tp2-q8}"
GPU_SPLIT="${GPU_SPLIT:-22.0,22.0}"
CACHE_TOKENS="${CACHE_TOKENS:-262144}"
CACHE_QUANT="${CACHE_QUANT:-8}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-4096}"
MIN_MAX_TOKENS="${MIN_MAX_TOKENS:-12}"
MAX_MAX_TOKENS="${MAX_MAX_TOKENS:-16000}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
PRESERVE_THINK_OUTPUT="${PRESERVE_THINK_OUTPUT:-false}"
MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-512}"
EXLLAMA_EMBED_PREFER_CPU="${EXLLAMA_EMBED_PREFER_CPU:-0}"
EXLLAMA_STARTUP_WARMUP="${EXLLAMA_STARTUP_WARMUP:-1}"
TP_BACKEND="${TP_BACKEND:-nccl}"
NCCL_ALGO="${NCCL_ALGO:-}"
EXLLAMA_FP16_REDUCE_THRESHOLD="${EXLLAMA_FP16_REDUCE_THRESHOLD:-}"

LOG_DIR="${LOG_DIR:-/home/op/exllamav3_ampere/logs}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOG_DIR}/holo35b-exl3-gpu01-${TIMESTAMP}.log"
CURRENT_LOG="${LOG_DIR}/holo35b-exl3-gpu01-current.log"
PIDFILE="${LOG_DIR}/holo35b-exl3-gpu01.pid"

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

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

_ENV_LIB="/home/op/miniconda3/envs/${CONDA_ENV}/lib"
_SP="${_ENV_LIB}/python3.11/site-packages"
export LD_LIBRARY_PATH="${_SP}/nvidia/cuda_runtime/lib:${_SP}/nvidia/nccl/lib:${_SP}/torch/lib:${LD_LIBRARY_PATH:-}"
unset _ENV_LIB

NCCL_VERSION_LINE="$(python - <<'PY'
import os, site, subprocess
base = site.getsitepackages()[0]
lib = os.path.join(base, "nvidia", "nccl", "lib", "libnccl.so.2")
line = subprocess.check_output(["strings", lib], text=True)
for raw in line.splitlines():
    if "NCCL version" in raw:
        print(raw.strip())
        break
PY
)"

if [[ "$NCCL_VERSION_LINE" == *"cuda13"* ]]; then
    printf 'Refusing to launch with incompatible NCCL: %s\n' "$NCCL_VERSION_LINE" >&2
    exit 1
fi

unset _SP

HOST="$_BIND_HOST"
PORT="$_BIND_PORT"
export MODEL_DIR
export MODEL_ID
export GPU_SPLIT
export CACHE_TOKENS
export CACHE_QUANT
export DEFAULT_MAX_TOKENS
export MIN_MAX_TOKENS
export MAX_MAX_TOKENS
export ENABLE_THINKING
export PRESERVE_THINK_OUTPUT
export MAX_THINKING_TOKENS
export EXLLAMA_EMBED_PREFER_CPU
export EXLLAMA_STARTUP_WARMUP
export TP_BACKEND
export HOST
export PORT

if [[ -n "$NCCL_ALGO" ]]; then
    export NCCL_ALGO
fi

if [[ -n "$EXLLAMA_FP16_REDUCE_THRESHOLD" ]]; then
    export EXLLAMA_FP16_REDUCE_THRESHOLD
fi

cd "$SCRIPT_DIR"

printf 'Starting Holo3-35B-A3B-exl3 on GPUs 0,1 (TP2/NCCL)\n'
printf '  Model dir   : %s\n' "$MODEL_DIR"
printf '  Host:Port   : %s:%s\n' "$HOST" "$PORT"
printf '  Model ID    : %s\n' "$MODEL_ID"
printf '  GPU split   : %s\n' "$GPU_SPLIT"
printf '  Cache tokens: %s\n' "$CACHE_TOKENS"
printf '  Cache quant : %s\n' "$CACHE_QUANT"
printf '  TP backend  : %s\n' "$TP_BACKEND"
printf '  Thinking    : %s (preserve=%s, budget=%s)\n' "$ENABLE_THINKING" "$PRESERVE_THINK_OUTPUT" "$MAX_THINKING_TOKENS"
printf '  NCCL line   : %s\n' "$NCCL_VERSION_LINE"
printf '  Log file    : %s\n' "$LOGFILE"
printf '  Current log : %s\n' "$CURRENT_LOG"
printf '\n'

nohup python server_27b_tp2.py \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL_DIR" \
    --gpu-split "$GPU_SPLIT" \
    --cache-tokens "$CACHE_TOKENS" \
    --cache-quant "$CACHE_QUANT" \
    >> "$LOGFILE" 2>&1 &

SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PIDFILE"

printf 'Started server with PID %s\n' "$SERVER_PID"
printf 'Health check: curl http://127.0.0.1:%s/health\n' "$PORT"
printf 'Models endpoint: curl http://127.0.0.1:%s/v1/models\n' "$PORT"
printf '\nMonitor logs: tail -f %s\n' "$CURRENT_LOG"
