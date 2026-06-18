#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"

DEFAULT_MODEL_DIR="/home/op/models/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-heretic-EXL3-6bpw"
DEFAULT_GPU_UUIDS="GPU-828df6fd-3fd0-ed25-0b2b-2b6d9d8dca47,GPU-78996a05-18c5-e153-b621-096273299d41"
DEFAULT_GPU_SPLIT="23.0,23.0"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="1234"
DEFAULT_MAX_MODEL_LEN="130048"
DEFAULT_DEFAULT_MAX_TOKENS="16000"
DEFAULT_MIN_MAX_TOKENS="12"
DEFAULT_MAX_MAX_TOKENS="16000"
DEFAULT_ENABLE_THINKING="true"
DEFAULT_PRESERVE_THINK_OUTPUT="true"
DEFAULT_CONDA_ENV="exl3-dev"

prompt_default() {
    local label="$1"
    local default_value="$2"
    local value
    read -r -p "$label [$default_value]: " value
    if [[ -z "$value" ]]; then
        value="$default_value"
    fi
    printf '%s' "$value"
}

prompt_yes_no() {
    local label="$1"
    local default_value="$2"
    local value
    local normalized_default="$default_value"

    while true; do
        read -r -p "$label [$normalized_default]: " value
        if [[ -z "$value" ]]; then
            value="$normalized_default"
        fi
        value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
        case "$value" in
            y|yes|true|1) printf 'true'; return ;;
            n|no|false|0) printf 'false'; return ;;
        esac
        printf 'Please answer yes or no.\n' >&2
    done
}

round_up_256() {
    local value="$1"
    printf '%s' "$(( ((value + 255) / 256) * 256 ))"
}

port_pid() {
    local port="$1"
    local pid
    pid="$(ss -ltnp "( sport = :${port} )" 2>/dev/null | python3 -c 'import re,sys; text=sys.stdin.read(); m=re.search(r"pid=(\d+)", text); print(m.group(1) if m else "")')"
    printf '%s' "$pid"
}

if [[ ! -f "$CONDA_SH" ]]; then
    printf 'Conda activation script not found at %s\n' "$CONDA_SH" >&2
    exit 1
fi

printf 'Interactive launcher for the Qwen 27B EXL3 server\n\n'

CONDA_ENV="$(prompt_default 'Conda env' "$DEFAULT_CONDA_ENV")"
MODEL_DIR="$(prompt_default 'Model directory' "$DEFAULT_MODEL_DIR")"
HOST="$(prompt_default 'Bind host' "$DEFAULT_HOST")"
PORT="$(prompt_default 'Port' "$DEFAULT_PORT")"
MAX_MODEL_LEN_INPUT="$(prompt_default 'Max model len / cache tokens' "$DEFAULT_MAX_MODEL_LEN")"
GPU_UUIDS="$(prompt_default 'GPU UUIDs (comma-separated)' "$DEFAULT_GPU_UUIDS")"
GPU_SPLIT="$(prompt_default 'GPU split (comma-separated GB values)' "$DEFAULT_GPU_SPLIT")"
DEFAULT_MAX_TOKENS="$(prompt_default 'Default max_tokens when client omits it' "$DEFAULT_DEFAULT_MAX_TOKENS")"
MIN_MAX_TOKENS="$(prompt_default 'Minimum max_tokens' "$DEFAULT_MIN_MAX_TOKENS")"
MAX_MAX_TOKENS="$(prompt_default 'Maximum max_tokens cap' "$DEFAULT_MAX_MAX_TOKENS")"
ENABLE_THINKING="$(prompt_yes_no 'Enable thinking mode' 'yes')"
PRESERVE_THINK_OUTPUT="$(prompt_yes_no 'Preserve raw <think> output in responses' 'yes')"
MODEL_ID="$(basename "$MODEL_DIR")"
LOG_PATH_DEFAULT="/home/op/models/${MODEL_ID}-server-${PORT}.log"
LOG_PATH="$(prompt_default 'Log file path' "$LOG_PATH_DEFAULT")"

if [[ ! -d "$MODEL_DIR" ]]; then
    printf 'Model directory does not exist: %s\n' "$MODEL_DIR" >&2
    exit 1
fi

if ! [[ "$MAX_MODEL_LEN_INPUT" =~ ^[0-9]+$ ]]; then
    printf 'Max model len must be an integer.\n' >&2
    exit 1
fi

MAX_MODEL_LEN="$(round_up_256 "$MAX_MODEL_LEN_INPUT")"
if [[ "$MAX_MODEL_LEN" != "$MAX_MODEL_LEN_INPUT" ]]; then
    printf 'Adjusted max model len to %s so it is a multiple of 256.\n' "$MAX_MODEL_LEN"
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    printf 'Port must be an integer.\n' >&2
    exit 1
fi

EXISTING_PID="$(port_pid "$PORT")"
if [[ -n "$EXISTING_PID" ]]; then
    KILL_EXISTING="$(prompt_yes_no "Port $PORT is already in use by PID $EXISTING_PID. Kill it" 'yes')"
    if [[ "$KILL_EXISTING" == "true" ]]; then
        kill "$EXISTING_PID"
        sleep 2
    else
        printf 'Aborting because port %s is already in use.\n' "$PORT" >&2
        exit 1
    fi
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export LD_LIBRARY_PATH="/home/op/miniconda3/envs/${CONDA_ENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export MODEL_DIR
export MODEL_ID
export PORT
export CACHE_TOKENS="$MAX_MODEL_LEN"
export GPU_SPLIT
export DEFAULT_MAX_TOKENS
export MIN_MAX_TOKENS
export MAX_MAX_TOKENS
export ENABLE_THINKING

cd "$SCRIPT_DIR"

printf '\nLaunching with:\n'
printf '  conda env: %s\n' "$CONDA_ENV"
printf '  model dir: %s\n' "$MODEL_DIR"
printf '  host: %s\n' "$HOST"
printf '  port: %s\n' "$PORT"
printf '  cache tokens: %s\n' "$MAX_MODEL_LEN"
printf '  gpu uuids: %s\n' "$GPU_UUIDS"
printf '  gpu split: %s\n' "$GPU_SPLIT"
printf '  default max_tokens: %s\n' "$DEFAULT_MAX_TOKENS"
printf '  min max_tokens: %s\n' "$MIN_MAX_TOKENS"
printf '  max max_tokens: %s\n' "$MAX_MAX_TOKENS"
printf '  thinking enabled: %s\n' "$ENABLE_THINKING"
printf '  preserve think output: %s\n' "$PRESERVE_THINK_OUTPUT"
printf '  log: %s\n\n' "$LOG_PATH"

if [[ "$PRESERVE_THINK_OUTPUT" == "true" ]]; then
    nohup python -c "import uvicorn, server_mixed; server_mixed.ENABLE_THINKING = ${ENABLE_THINKING^}; server_mixed._strip_thinking = lambda text: text; uvicorn.run(server_mixed.app, host='${HOST}', port=${PORT}, access_log=True)" > "$LOG_PATH" 2>&1 &
else
    nohup python -c "import uvicorn, server_mixed; server_mixed.ENABLE_THINKING = ${ENABLE_THINKING^}; uvicorn.run(server_mixed.app, host='${HOST}', port=${PORT}, access_log=True)" > "$LOG_PATH" 2>&1 &
fi

SERVER_PID="$!"

sleep 2

printf 'Started server with PID %s\n' "$SERVER_PID"
printf 'Health URL: http://127.0.0.1:%s/health\n' "$PORT"
printf 'Log file: %s\n' "$LOG_PATH"
