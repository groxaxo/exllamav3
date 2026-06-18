#!/usr/bin/env bash
set -euo pipefail

SESSION="qwen122b_ablit"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
MODEL_DIR="/home/op/exllamav3_ampere/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
MODEL_ID="Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
GPU_SPLIT="21,21,18"
CACHE_TOKENS="32768"
CACHE_QUANT="8,8"
PORT="8000"
LOGFILE="/home/op/exllamav3_ampere/logs/qwen122b_ablit_server.log"

# RTX 3090 UUIDs
GPU_UUIDS="GPU-828df6fd-3fd0-ed25-0b2b-2b6d9d8dca47,GPU-78996a05-18c5-e153-b621-096273299d41,GPU-89c6bfdc-6f42-d312-de77-a9fb1ae370d8"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export LD_LIBRARY_PATH="/home/op/miniconda3/envs/${CONDA_ENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export MODEL_DIR
export MODEL_ID
export PORT
export CACHE_TOKENS
export GPU_SPLIT
export CACHE_QUANT

mkdir -p "$(dirname "$LOGFILE")"

printf "Starting 122B Abliterated Server on Port %s...\n" "$PORT"
printf "Using GPUs: %s\n" "$GPU_UUIDS"
printf "Log: %s\n" "$LOGFILE"

# Use the python snippet to launch with 8,8 cache and custom load logic
nohup python -c "
import os, sys, uvicorn
sys.path.insert(0, os.getcwd())
import server_mixed
from exllamav3.cache import CacheLayer_quant

# Override defaults
server_mixed.MODEL_DIR = os.environ['MODEL_DIR']
server_mixed.GPU_SPLIT = [float(x) for x in os.environ['GPU_SPLIT'].split(',')]
server_mixed.CACHE_TOKENS = int(os.environ['CACHE_TOKENS'])
server_mixed.PORT = int(os.environ['PORT'])

# Inject 8-bit cache config
def custom_load_model():
    global model, config, cache, tokenizer, generator
    server_mixed.logger.info(f'Loading 122B with 8-bit cache on {server_mixed.GPU_SPLIT}')
    server_mixed.config = server_mixed.Config.from_directory(server_mixed.MODEL_DIR)
    server_mixed.model = server_mixed.Model.from_config(server_mixed.config)
    server_mixed.cache = server_mixed.Cache(
        server_mixed.model,
        max_num_tokens=server_mixed.CACHE_TOKENS,
        layer_type=CacheLayer_quant,
        k_bits=8,
        v_bits=8
    )
    server_mixed.model.load(use_per_device=server_mixed.GPU_SPLIT, progressbar=True)
    server_mixed.tokenizer = server_mixed.Tokenizer.from_config(server_mixed.config)
    server_mixed.generator = server_mixed.Generator(model=server_mixed.model, cache=server_mixed.cache, tokenizer=server_mixed.tokenizer)
    server_mixed.logger.info('122B Model Loaded with 8-bit cache!')

# Replace the startup handler with our custom one
server_mixed.app.router.on_startup.clear()
server_mixed.app.on_event('startup')(custom_load_model)
uvicorn.run(server_mixed.app, host='0.0.0.0', port=server_mixed.PORT, access_log=True)
" > "$LOGFILE" 2>&1 &

printf "Server PID: %s\n" "$!"
