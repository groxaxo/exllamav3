#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/op/exllamav3_ampere"
PYTHON_BIN="/home/op/miniconda3/envs/exl3-dev/bin/python"
SERVER_SCRIPT="$ROOT/server_holo_3090.py"
LOG_DIR="$ROOT/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/server_holo_3090_port1235.log}"
PID_FILE="$LOG_DIR/server_holo_3090_port1235.pid"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1235}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,4}"
GPU_SPLIT="${GPU_SPLIT:-22.0,22.0}"
CACHE_TOKENS="${CACHE_TOKENS:-80000}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-6000}"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        kill "$old_pid"
        for _ in $(seq 1 30); do
            if ! kill -0 "$old_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
    fi
    rm -f "$PID_FILE"
fi

existing_pids="$(pgrep -f "$SERVER_SCRIPT" || true)"
if [[ -n "$existing_pids" ]]; then
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        kill "$pid" 2>/dev/null || true
    done <<< "$existing_pids"
    sleep 2
fi

nohup env \
    HOST="$HOST" \
    PORT="$PORT" \
    CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    GPU_SPLIT="$GPU_SPLIT" \
    CACHE_TOKENS="$CACHE_TOKENS" \
    MAX_TOKENS="$MAX_TOKENS" \
    MAX_THINKING_TOKENS="$MAX_THINKING_TOKENS" \
    "$PYTHON_BIN" "$SERVER_SCRIPT" \
    > "$LOG_FILE" 2>&1 < /dev/null &

new_pid=$!
echo "$new_pid" > "$PID_FILE"

printf 'Started Holo server\n'
printf 'PID: %s\n' "$new_pid"
printf 'Log: %s\n' "$LOG_FILE"
printf 'Host: %s\n' "$HOST"
printf 'Port: %s\n' "$PORT"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES"
printf 'GPU_SPLIT: %s\n' "$GPU_SPLIT"
printf 'CACHE_TOKENS: %s\n' "$CACHE_TOKENS"
printf 'MAX_TOKENS: %s\n' "$MAX_TOKENS"
printf 'MAX_THINKING_TOKENS: %s\n' "$MAX_THINKING_TOKENS"
