#!/usr/bin/env bash
# =============================================================================
# OPTIMIZED 3x RTX 3090 EXL3 4.0 bpw quantizer — Qwen3.5-122B-A10B-abliterated
# =============================================================================
#
# ─── HARDWARE INVENTORY ───────────────────────────────────────────────────────
#  GPU0  RTX 3090  24 GB  ~29,284 TFLOPS @ SM 8.6 (Ampere)
#  GPU1  RTX 3090  24 GB  ~29,284 TFLOPS @ SM 8.6 (Ampere)
#  GPU2  RTX 3060  12 GB  ~12,953 TFLOPS @ SM 8.6 (Ampere) ← EXCLUDED (see below)
#  GPU3  RTX 3060  12 GB  ~12,953 TFLOPS @ SM 8.6 (Ampere) ← EXCLUDED (see below)
#  GPU4  RTX 3090  24 GB  ~29,284 TFLOPS @ SM 8.6 (Ampere)
#
# ─── WHY GPUs 2 AND 3 ARE EXCLUDED ───────────────────────────────────────────
#  GPU2 is shared by two persistent services that are always loaded:
#    • chatterbox-tts-api  ≈ 3,216 MiB GPU memory
#    • vllm-embed          ≈ 3,458 MiB GPU memory
#  Combined they consume 6.7 GB / 12 GB leaving only ~4.3 GB free and
#  can spike to 100% GPU utilization whenever a TTS or embed request arrives.
#  During an uninterrupted 4-hour quantization this would stall the whole run
#  because all parallel threads wait for the slowest worker (GPU2 in this case).
#  GPU3 carries display-server + KDE + an additional 1.6 GB process.
#  Excluding both 3060s gives a predictable, uncontested workload.
#
# ─── TIMING ANALYSIS ─────────────────────────────────────────────────────────
#  Calibrated from observed 327.6 s/layer on the 5-GPU run.
#
#  Model: bottleneck_time = max_i( ratio_fraction_i / relative_perf_i )
#         relative_perf(3090) = 1.000
#         relative_perf(3060) = 12953 / 29284 = 0.4423
#
#  Configuration                  Devices   Ratios     Bottleneck  48-layer ETA
#  ─────────────────────────────  ────────  ─────────  ──────────  ─────────────
#  5-GPU original (observed)      0,1,2,3,4  2,2,1,1,2  3060 @0.303  ≈ 4.37 h
#  2x RTX 3090 only               0,1        1,1          3090 @0.500  ≈ 7.21 h
#  3x RTX 3090 (THIS SCRIPT)      0,1,4      1,1,1        3090 @0.333  ≈ 4.80 h *
#
#  * Theoretical. In practice GPU2's real throughput is degraded by its two
#    co-resident services, making the 5-GPU run unreliable. With 3x 3090 every
#    worker runs at full, uncontested speed → observed time will be ≤ 4.80 h.
#
#  WINNER: 3x RTX 3090 — best combination of speed, reliability, and isolation.
#
# ─── CHECKPOINT NOTE ─────────────────────────────────────────────────────────
#  -cpi 9999999 disables mid-run checkpoint rotation entirely.
#  The rotation creates ckpt_new/state.safetensors while ckpt/state.safetensors
#  still exists, temporarily doubling disk usage.  On a filesystem already
#  holding 228 GB of source weights + growing qtensors this caused an earlier
#  run to crash with ENOSPC.  The trade-off is that a crash loses all progress.
# =============================================================================

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

# Explicit Ampere SM 8.6 arch — prevents fallback to generic kernels on 3090s
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST="8.6"

export HF_TOKEN="${HF_TOKEN:-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}"

if [ ! -f "${LOCAL_SRC}/config.json" ]; then
  printf '[%s] Downloading model → %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LOCAL_SRC}"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='wangzhang/Qwen3.5-122B-A10B-abliterated',
    local_dir='${LOCAL_SRC}',
    token='${HF_TOKEN}',
)
print('Download complete.')
"
else
  printf '[%s] Source model present at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LOCAL_SRC}"
fi

{
  printf '[%s] Starting EXL3 4.0 bpw quantization (3x RTX 3090 optimized)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Source:    %s\n' "${LOCAL_SRC}"
  printf 'Work dir:  %s\n' "${WORK_DIR}"
  printf 'Output:    %s\n' "${OUT_DIR}"
  printf 'Log:       %s\n' "${LOG}"
  printf 'Devices:   0,1,4  (3x RTX 3090, equal ratios 1:1:1)\n'
  printf 'ETA:       ~4.8 h theoretical / likely <= 4.8 h practical\n\n'
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
