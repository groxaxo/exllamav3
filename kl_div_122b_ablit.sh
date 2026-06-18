#!/usr/bin/env bash
# Measure KL divergence between the three quantized 122B-abliterated EXL3 models.
# Uses 6bpw as the reference (model A) and compares 5bpw and 4bpw against it.
# Outputs JSON results to logs/kl_div_122b_abliterated.json
set -euo pipefail

CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
EXL3_DIR="/home/op/exllamav3_ampere"
EVAL_DIR="${EXL3_DIR}/eval"
LOG_DIR="${EXL3_DIR}/logs"

MODEL_6B="${EXL3_DIR}/models/Qwen3.5-122B-A10B-abliterated-exl3-6bpw"
MODEL_5B="${EXL3_DIR}/models/Qwen3.5-122B-A10B-abliterated-exl3-5bpw"
MODEL_4B="${EXL3_DIR}/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
RESULTS_JSON="${LOG_DIR}/kl_div_122b_abliterated.json"
LOG="${LOG_DIR}/kl_div_122b_abliterated.log"

mkdir -p "${LOG_DIR}"

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_TOKEN="${HF_TOKEN:-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}"

{
  printf '[%s] KL divergence evaluation — Qwen3.5-122B-A10B-abliterated EXL3\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Reference (A): %s\n' "${MODEL_6B}"
  printf 'Candidate  B:  %s\n' "${MODEL_5B}"
  printf 'Candidate  C:  %s\n' "${MODEL_4B}"
  printf 'Results:       %s\n\n' "${RESULTS_JSON}"
} | tee "${LOG}"

cd "${EVAL_DIR}"

# ── Pass 1: 6bpw (ref) vs 5bpw ────────────────────────────────────────────────
printf '[%s] Pass 1/2 — 6bpw vs 5bpw\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"

python compare_q.py \
  --model_a   "${MODEL_6B}" \
  --model_a_type exllamav3 \
  --model_b   "${MODEL_5B}" \
  --model_b_type exllamav3 \
  --rows 100 \
  --kld \
  --output "${LOG_DIR}/kl_div_6b_vs_5b.json" \
  2>&1 | tee -a "${LOG}"

# ── Pass 2: 6bpw (ref) vs 4bpw ────────────────────────────────────────────────
printf '[%s] Pass 2/2 — 6bpw vs 4bpw\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"

python compare_q.py \
  --model_a   "${MODEL_6B}" \
  --model_a_type exllamav3 \
  --model_b   "${MODEL_4B}" \
  --model_b_type exllamav3 \
  --rows 100 \
  --kld \
  --output "${LOG_DIR}/kl_div_6b_vs_4b.json" \
  2>&1 | tee -a "${LOG}"

# ── Merge results into one summary ────────────────────────────────────────────
python3 - <<'PYEOF'
import json, os, sys
log_dir = os.environ.get("LOG_DIR", "/home/op/exllamav3_ampere/logs")

results = {}
for fname, label in [("kl_div_6b_vs_5b.json", "6bpw_vs_5bpw"), ("kl_div_6b_vs_4b.json", "6bpw_vs_4bpw")]:
    fpath = os.path.join(log_dir, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            results[label] = json.load(f)
    else:
        results[label] = "result file not found"

out = os.path.join(log_dir, "kl_div_122b_abliterated.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Summary written → {out}")
print(json.dumps(results, indent=2))
PYEOF

printf '\n[%s] KL divergence evaluation complete.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
