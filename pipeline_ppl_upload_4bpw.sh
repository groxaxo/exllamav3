#!/usr/bin/env bash
# Autonomous post-quantization pipeline:
#   1. Wait for quantization to complete
#   2. Run perplexity evaluation on the 4bpw model
#   3. Upload to HuggingFace
set -euo pipefail

CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
EXL3_DIR="/home/op/exllamav3_ampere"
MODEL_4B="${EXL3_DIR}/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
QUANT_LOG="${EXL3_DIR}/logs/quantize_122b_ablit_4bpw_3x3090.nohup.log"
PPL_LOG="${EXL3_DIR}/logs/ppl_122b_ablit_4bpw.log"
UPLOAD_LOG="${EXL3_DIR}/logs/upload_122b_ablit_4bpw.log"
PID_FILE="${EXL3_DIR}/logs/quantize_122b_4bpw.pid"

set +u; source "${CONDA_SH}"; conda activate "${CONDA_ENV}"; set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_TOKEN="${HF_TOKEN:-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}"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# ── 1. Wait for quantization process ──────────────────────────────────────────
log "Waiting for quantization process to finish..."
QUANT_PID=$(cat "${PID_FILE}" 2>/dev/null || echo "")
if [ -n "${QUANT_PID}" ]; then
  while kill -0 "${QUANT_PID}" 2>/dev/null; do
    LAYER=$(grep -oP 'layers\.\K[0-9]+(?=\s)' "${QUANT_LOG}" 2>/dev/null | sort -n | tail -1 || echo "?")
    ETA=$(grep "Estimated remaining" "${QUANT_LOG}" 2>/dev/null | tail -1 | sed 's/.*remaining time: //' || echo "unknown")
    log "  Still running (PID ${QUANT_PID}) | latest layer: ${LAYER} | ETA: ${ETA}"
    sleep 300
  done
fi

# Confirm quantization log says "complete"
if ! grep -q "Quantization complete" "${QUANT_LOG}" 2>/dev/null; then
  log "ERROR: Quantization log does not show completion. Check ${QUANT_LOG}"
  exit 1
fi
log "Quantization complete!"

# ── 2. Perplexity evaluation ───────────────────────────────────────────────────
log "Starting perplexity evaluation on ${MODEL_4B}..."
{
  log "=== Perplexity Evaluation: Qwen3.5-122B-A10B-abliterated-exl3-4bpw ==="
  log "Model: ${MODEL_4B}"
  log "Dataset: wikitext-2-raw-v1 (test), 100 rows × 2048 tokens"
  log ""
} | tee "${PPL_LOG}"

cd "${EXL3_DIR}/eval"
python ppl.py \
  -m "${MODEL_4B}" \
  -r 100 \
  -l 2048 \
  -gs "22,22,0,0,22" \
  2>&1 | tee -a "${PPL_LOG}"

log "Perplexity evaluation complete. Log: ${PPL_LOG}"

# ── 3. Upload to HuggingFace ────────────────────────────────────────────────────
log "Uploading to HuggingFace..."
{
  log "=== HuggingFace Upload: groxaxo/Qwen3.5-122B-A10B-abliterated-exl3-4bpw ==="
} | tee "${UPLOAD_LOG}"

python3 - <<'PYEOF' 2>&1 | tee -a "${UPLOAD_LOG}"
import os, sys
from huggingface_hub import HfApi

TOKEN   = os.environ["HF_TOKEN"]
LOCAL   = "/home/op/exllamav3_ampere/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
REPO_ID = "groxaxo/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"

files = [f for f in os.listdir(LOCAL) if os.path.isfile(os.path.join(LOCAL, f))]
total_gb = sum(os.path.getsize(os.path.join(LOCAL, f)) for f in files) / (1024**3)
print(f"Model dir: {LOCAL}", flush=True)
print(f"Files: {len(files)}  Total: {total_gb:.1f} GB", flush=True)

api = HfApi(token=TOKEN)
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
print(f"Repo: https://huggingface.co/{REPO_ID}", flush=True)
print("Uploading...", flush=True)

api.upload_folder(
    folder_path=LOCAL,
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"\n✓ Upload complete: https://huggingface.co/{REPO_ID}", flush=True)
PYEOF

log "Pipeline complete!"
