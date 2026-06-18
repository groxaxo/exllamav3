#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
EXL3_DIR="/home/op/exllamav3_ampere"

LOCAL_SRC="/home/op/models/Qwen3.5-122B-A10B-abliterated"
WORK_DIR="${EXL3_DIR}/models/work_122b_ablit_4bpw_3x3090"
OUT_DIR="${EXL3_DIR}/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
LOG="${EXL3_DIR}/logs/quantize_122b_ablit_4bpw_3x3090.log"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${OUT_DIR}" "$(dirname "${LOG}")"

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST="8.6"
export HF_TOKEN="${HF_TOKEN:-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}"

{
  printf '[%s] Starting EXL3 4.0 bpw quantization (3x RTX 3090)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Source:    %s\n' "${LOCAL_SRC}"
  printf 'Work dir:  %s\n' "${WORK_DIR}"
  printf 'Output:    %s\n' "${OUT_DIR}"
  printf 'Devices:   0,1,4  (3x RTX 3090, ratios 1:1:1)\n'
  printf 'ETA:       ~4.8 h\n\n'
} | tee "${LOG}"

cd "${EXL3_DIR}"

python convert.py \
  -i  "${LOCAL_SRC}" \
  -w  "${WORK_DIR}" \
  -o  "${OUT_DIR}" \
  -b  4.0 \
  -hb 8 \
  -cr 250 \
  -cc 2048 \
  -d  0,1,4 \
  -dr 1,1,1 \
  -pm \
  -cpi 9999999 \
  -v \
  2>&1 | tee -a "${LOG}"

printf '\n[%s] Quantization complete → %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${OUT_DIR}" | tee -a "${LOG}"
