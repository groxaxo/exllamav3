#!/usr/bin/env bash
set -euo pipefail

SESSION="qwen122b_ablit_5gpu_200k"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
MODEL_DIR="/home/op/exllamav3_ampere/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
MODEL_ID="Qwen3.5-122B-A10B-abliterated-exl3-4bpw"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
export GPU_SPLIT="auto"
export CACHE_TOKENS="200000"
export CACHE_QUANT="auto"
export HOST="0.0.0.0"
export PORT="1234"
export DEFAULT_MAX_TOKENS="16000"
export MIN_MAX_TOKENS="12"
export MAX_MAX_TOKENS="16000"
export ENABLE_THINKING="true"
export PRESERVE_THINK_OUTPUT="true"
export MAX_THINKING_TOKENS="1024"
export EXLLAMA_STARTUP_WARMUP="1"
export EXLLAMA_EMBED_PREFER_CPU="0"
export EXLLAMA_DISABLE_HOST_RECURRENT_CACHE="1"
export EXLLAMA_STRICT_GPU_ONLY="1"
export EXLLAMA_GDN_RECURRENT_BACKEND="ext"
export RESERVE_PER_GPU_GB="0.50"
export MAX_VRAM_UTILIZATION="0.97"
export RUNTIME_HEADROOM_GB="1.00"
export OMP_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"
export CUDA_MODULE_LOADING="LAZY"
export CUDA_MANAGED_FORCE_DEVICE_ALLOC="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.80"

SP="/home/op/miniconda3/envs/${CONDA_ENV}/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${SP}/nvidia/cuda_runtime/lib:${SP}/nvidia/nccl/lib:${SP}/torch/lib:${LD_LIBRARY_PATH:-}"
unset SP

LOG_DIR="/home/op/exllamav3_ampere/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOG_DIR}/qwen122b-ablit-5gpu-200k-${TIMESTAMP}.log"
CURRENT_LOG="${LOG_DIR}/qwen122b-ablit-5gpu-200k-current.log"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

touch "$LOGFILE"
ln -sfn "$LOGFILE" "$CURRENT_LOG"

{
  printf '[%s] Starting %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SESSION"
  printf 'Model              : %s\n' "$MODEL_DIR"
  printf 'Model ID           : %s\n' "$MODEL_ID"
  printf 'Host:Port          : %s:%s\n' "$HOST" "$PORT"
  printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'GPU split          : %s\n' "$GPU_SPLIT"
  printf 'Cache tokens       : %s\n' "$CACHE_TOKENS"
  printf 'Cache quant        : %s\n' "$CACHE_QUANT"
  printf 'Reserve per GPU    : %s GiB\n' "$RESERVE_PER_GPU_GB"
  printf 'VRAM utilization   : %s\n' "$MAX_VRAM_UTILIZATION"
  printf 'Thinking           : %s (preserve=%s, budget=%s)\n' "$ENABLE_THINKING" "$PRESERVE_THINK_OUTPUT" "$MAX_THINKING_TOKENS"
  printf 'Conda env          : %s\n' "$CONDA_ENV"
  printf 'Log                : %s\n' "$LOGFILE"
  printf '\n'
} | tee "$LOGFILE"

cd /home/op/exllamav3_ampere

nohup python server_27b_layer_optimized.py \
    --model "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-split "$GPU_SPLIT" \
    --cache-tokens "$CACHE_TOKENS" \
    --cache-quant "$CACHE_QUANT" \
    --reserve-per-gpu-gb "$RESERVE_PER_GPU_GB" \
    --max-vram-utilization "$MAX_VRAM_UTILIZATION" \
    --runtime-headroom-gb "$RUNTIME_HEADROOM_GB" \
    >> "$LOGFILE" 2>&1 &

SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "${LOG_DIR}/qwen122b-ablit-5gpu-200k.pid"
printf 'Started server with PID %s\n' "$SERVER_PID"
printf 'Health check : curl http://127.0.0.1:%s/health\n' "$PORT"
printf 'Models       : curl http://127.0.0.1:%s/v1/models\n' "$PORT"
printf 'Log (follow) : tail -f %s\n' "$CURRENT_LOG"
