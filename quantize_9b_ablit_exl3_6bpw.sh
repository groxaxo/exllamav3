#!/bin/bash
# EXL3 6-bit quantization for wangzhang/Qwen3.5-9B-abliterated
set -euo pipefail

cd /home/op/exllamav3_ampere

export PYTHONPATH=/home/op/exllamav3_ampere
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="${HF_TOKEN:-}"

SRC_MODEL="wangzhang/Qwen3.5-9B-abliterated"
WORK_DIR="models/work_9b_ablit_6bpw"
OUT_DIR="models/Qwen3.5-9B-abliterated-exl3-6bpw"
LOG="logs/quantize_9b_ablit_exl3_6bpw.log"

mkdir -p "${WORK_DIR}" "${OUT_DIR}" logs

echo "[INFO] Starting EXL3 6.0 bpw quantization"
echo "[INFO] Source: ${SRC_MODEL}"
echo "[INFO] Work:   ${WORK_DIR}"
echo "[INFO] Output: ${OUT_DIR}"
echo "[INFO] Device: GPU 1 (RTX 3090)"

python3 -m exllamav3.conversion.convert_model \
    -i "${SRC_MODEL}" \
    -w "${WORK_DIR}" \
    -o "${OUT_DIR}" \
    -b 6.0 \
    -hb 6 \
    -d 1 \
    -cr 256 \
    -cc 2048 \
    --codebook mcg \
    -v 2>&1 | tee "${LOG}"

echo "[INFO] Done! Output: ${OUT_DIR}"
