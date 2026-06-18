#!/usr/bin/env bash
# Quantize huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated to EXL3 6.0 bpw
set -euo pipefail

CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
EXL3_DIR="/home/op/exllamav3_ampere"

HF_REPO="huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated"
LOCAL_SRC="/home/op/models/Huihui-Qwen3.5-35B-A3B-abliterated"
WORK_DIR="${EXL3_DIR}/models/work_huihui_qwen35_6bpw"
OUT_DIR="${EXL3_DIR}/models/Huihui-Qwen3.5-35B-A3B-abliterated-exl3-6bpw"
LOG="${EXL3_DIR}/logs/quantize_huihui_qwen35_6bpw.log"

mkdir -p "${WORK_DIR}" "${OUT_DIR}" "$(dirname "${LOG}")"

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_TOKEN="${HF_TOKEN:-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}"

# Download source model if not present
if [ ! -f "${LOCAL_SRC}/config.json" ]; then
  printf '[%s] Downloading %s → %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${HF_REPO}" "${LOCAL_SRC}"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    local_dir='${LOCAL_SRC}',
    token='${HF_TOKEN}',
)
print('Download complete.')
"
else
  printf '[%s] Source model already present at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LOCAL_SRC}"
fi

{
  printf '[%s] Starting EXL3 6.0 bpw quantization\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Source:  %s\n' "${LOCAL_SRC}"
  printf 'Work:    %s\n' "${WORK_DIR}"
  printf 'Output:  %s\n' "${OUT_DIR}"
  printf 'Log:     %s\n' "${LOG}"
  printf 'Devices: 0,1,4 (3x RTX 3090)\n\n'
} | tee "${LOG}"

cd "${EXL3_DIR}"

python convert.py \
  -i  "${LOCAL_SRC}" \
  -w  "${WORK_DIR}" \
  -o  "${OUT_DIR}" \
  -b  6.0 \
  -hb 8 \
  -cr 250 \
  -cc 2048 \
  -d  0,1,4 \
  -dr 1,1,1 \
  -pm \
  -cpi 300 \
  -v \
  2>&1 | tee -a "${LOG}"

printf '\n[%s] Quantization complete → %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${OUT_DIR}" | tee -a "${LOG}"
