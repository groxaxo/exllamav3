#!/usr/bin/env bash
# Upload all three quantized 122B-abliterated EXL3 models to HuggingFace.
# Uploads 6bpw, 5bpw, and 4bpw variants to groxaxo/
# Run after quantization is complete.
set -euo pipefail

CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
EXL3_DIR="/home/op/exllamav3_ampere"
LOG_DIR="${EXL3_DIR}/logs"
LOG="${LOG_DIR}/upload_122b_ablit.log"

mkdir -p "${LOG_DIR}"

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export HF_TOKEN="${HF_TOKEN:-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}"

{
  printf '[%s] Uploading Qwen3.5-122B-A10B-abliterated EXL3 quants to HuggingFace\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${LOG}"

python3 - <<'PYEOF'
import os, sys
from huggingface_hub import HfApi

TOKEN   = os.environ["HF_TOKEN"]
EXL3    = "/home/op/exllamav3_ampere/models"

UPLOADS = [
    {
        "local":   f"{EXL3}/Qwen3.5-122B-A10B-abliterated-exl3-6bpw",
        "repo_id": "groxaxo/Qwen3.5-122B-A10B-abliterated-exl3-6bpw",
        "desc":    "6.0 bpw",
    },
    {
        "local":   f"{EXL3}/Qwen3.5-122B-A10B-abliterated-exl3-5bpw",
        "repo_id": "groxaxo/Qwen3.5-122B-A10B-abliterated-exl3-5bpw",
        "desc":    "5.0 bpw",
    },
    {
        "local":   f"{EXL3}/Qwen3.5-122B-A10B-abliterated-exl3-4bpw",
        "repo_id": "groxaxo/Qwen3.5-122B-A10B-abliterated-exl3-4bpw",
        "desc":    "4.0 bpw",
    },
]

api = HfApi(token=TOKEN)

for entry in UPLOADS:
    local   = entry["local"]
    repo_id = entry["repo_id"]
    desc    = entry["desc"]

    if not os.path.isdir(local):
        print(f"[SKIP] {desc} — directory not found: {local}", flush=True)
        continue

    files = [f for f in os.listdir(local) if os.path.isfile(os.path.join(local, f))]
    total_gb = sum(os.path.getsize(os.path.join(local, f)) for f in files) / (1024**3)
    print(f"\n[{desc}] Creating repo: {repo_id}  ({total_gb:.1f} GB, {len(files)} files)", flush=True)

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"[{desc}] Uploading...", flush=True)
    api.upload_folder(
        folder_path=local,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[{desc}] ✓ https://huggingface.co/{repo_id}", flush=True)

print("\nAll uploads complete.", flush=True)
PYEOF

printf '\n[%s] Upload script finished.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
