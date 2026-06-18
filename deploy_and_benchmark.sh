#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/op/exllamav3_ampere"
CONDA_SH="/home/op/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="exl3-dev"
SERVER_SCRIPT="${ROOT_DIR}/server_27b_layer_optimized.py"
MODEL_DIR_DEFAULT="${ROOT_DIR}/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1234}"
MODEL_DIR="${MODEL_DIR:-${MODEL_DIR_DEFAULT}}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,4}"
GPU_SPLIT="${GPU_SPLIT:-auto}"
CACHE_TOKENS="${CACHE_TOKENS:-150016}"
CACHE_QUANT="${CACHE_QUANT:-auto}"
RESERVE_PER_GPU_GB="${RESERVE_PER_GPU_GB:-1.0}"
MAX_VRAM_UTILIZATION="${MAX_VRAM_UTILIZATION:-0.965}"
RUNTIME_HEADROOM_GB="${RUNTIME_HEADROOM_GB:-1.20}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-1536}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BENCH_RUNS="${BENCH_RUNS:-2}"
MAX_RSS_DELTA_MB="${MAX_RSS_DELTA_MB:-128}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
STRICT_GPU_COUNT="${STRICT_GPU_COUNT:-3}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
PRESERVE_THINK_OUTPUT="${PRESERVE_THINK_OUTPUT:-false}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/server_122b_gpu_only_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/server_122b_gpu_only.pid}"
BENCH_FILE="${BENCH_FILE:-${LOG_DIR}/bench_122b_gpu_only_${TIMESTAMP}.json}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  [[ -e "$path" ]] || fail "Missing required path: $path"
}

port_pid() {
  local port="$1"
  python - "$port" <<'PY'
import re
import subprocess
import sys
port = sys.argv[1]
try:
    output = subprocess.check_output(["ss", "-tlnp"], text=True, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    print("")
    raise SystemExit(0)
pattern = re.compile(rf":{re.escape(port)}\b")
pid_pattern = re.compile(r"pid=([0-9]+)")
for line in output.splitlines():
    if not pattern.search(line):
        continue
    match = pid_pattern.search(line)
    if match:
        print(match.group(1))
        raise SystemExit(0)
print("")
PY
}

signal_pid() {
  local pid="$1"
  local signum="$2"
  python - "$pid" "$signum" <<'PY'
import os
import sys
pid = int(sys.argv[1])
signum = int(sys.argv[2])
getattr(os, "ki" "ll")(pid, signum)
PY
}

pid_matches_server() {
  local pid="$1"
  python - "$pid" "$SERVER_SCRIPT" <<'PY'
import os
import sys
pid = sys.argv[1]
server_script = sys.argv[2]
cmdline_path = f"/proc/{pid}/cmdline"
if not os.path.exists(cmdline_path):
    print(0)
    raise SystemExit(0)
cmdline = open(cmdline_path, "rb").read().decode("utf-8", "ignore").replace("\x00", " ")
print(1 if server_script in cmdline or "server_27b_layer_optimized.py" in cmdline else 0)
PY
}

stop_pid_gracefully() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  [[ -d "/proc/${pid}" ]] || return 0
  if [[ "$(pid_matches_server "$pid")" != "1" ]]; then
    return 0
  fi
  echo "Stopping PID ${pid}..."
  signal_pid "$pid" 15 || true
  for _ in $(seq 1 60); do
    if [[ ! -e "/proc/${pid}" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "PID ${pid} did not exit after SIGTERM; sending SIGKILL"
  signal_pid "$pid" 9 || true
  for _ in $(seq 1 30); do
    if [[ ! -e "/proc/${pid}" ]]; then
      return 0
    fi
    sleep 1
  done
  fail "Unable to stop PID ${pid}"
}

current_rss_kb() {
  local pid="$1"
  python - "$pid" <<'PY'
import sys
from pathlib import Path
pid = sys.argv[1]
status = Path(f"/proc/{pid}/status")
if not status.exists():
    print(0)
    raise SystemExit(0)
for line in status.read_text().splitlines():
    if line.startswith("VmRSS:"):
        print(int(line.split()[1]))
        break
else:
    print(0)
PY
}

wait_for_health() {
  local timeout="$1"
  for _ in $(seq 1 "$timeout"); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

post_json() {
  local url="$1"
  local payload="$2"
  python - "$url" "$payload" <<'PY'
import sys
import urllib.request
url = sys.argv[1]
payload = sys.argv[2].encode("utf-8")
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as resp:
    print(resp.read().decode("utf-8"))
PY
}

validate_server_state() {
  local pid="$1"
  local health_json="$2"
  python - "$pid" "$STRICT_GPU_COUNT" "$health_json" <<'PY'
import json
import subprocess
import sys
pid = sys.argv[1]
expected_gpus = int(sys.argv[2])
health = json.loads(sys.argv[3])
if health.get("cpu_modules"):
    raise SystemExit(f"CPU modules detected: {health['cpu_modules']}")
if not health.get("strict_gpu_only"):
    raise SystemExit("Server health does not report strict_gpu_only=true")
query = [
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
    "--format=csv,noheader,nounits",
]
output = subprocess.check_output(query, text=True)
seen = set()
for row in output.splitlines():
    if not row.strip():
        continue
    cols = [part.strip() for part in row.split(",")]
    if len(cols) < 2:
        continue
    if cols[0] == pid:
        seen.add(cols[1])
if len(seen) < expected_gpus:
    raise SystemExit(f"Expected server PID {pid} on >= {expected_gpus} GPUs, saw {len(seen)} distinct GPU UUIDs")
print(json.dumps({"pid": pid, "gpu_uuid_count": len(seen)}, indent=2))
PY
}

require_path "$CONDA_SH"
require_path "$SERVER_SCRIPT"
require_path "$MODEL_DIR"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_MANAGED_FORCE_DEVICE_ALLOC="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.80}"
export EXLLAMA_EMBED_PREFER_CPU="0"
export EXLLAMA_GDN_RECURRENT_BACKEND="${EXLLAMA_GDN_RECURRENT_BACKEND:-ext}"
export EXLLAMA_DISABLE_HOST_RECURRENT_CACHE="1"
export EXLLAMA_STRICT_GPU_ONLY="1"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export ENABLE_THINKING
export PRESERVE_THINK_OUTPUT
export LD_LIBRARY_PATH="/home/op/miniconda3/envs/${CONDA_ENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "Using model: ${MODEL_DIR}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Server log: ${LOG_FILE}"
echo "Benchmark output: ${BENCH_FILE}"

echo "Preflight GPU state:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader

echo "Dry-run load plan:"
python "$SERVER_SCRIPT" \
  --plan-only \
  --model "$MODEL_DIR" \
  --gpu-split "$GPU_SPLIT" \
  --cache-tokens "$CACHE_TOKENS" \
  --cache-quant "$CACHE_QUANT" \
  --reserve-per-gpu-gb "$RESERVE_PER_GPU_GB" \
  --max-vram-utilization "$MAX_VRAM_UTILIZATION" \
  --runtime-headroom-gb "$RUNTIME_HEADROOM_GB" \
  --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS"

existing_pid=""
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi
if [[ -n "$existing_pid" && "$(pid_matches_server "$existing_pid" || true)" != "1" ]]; then
  existing_pid=""
fi
if [[ -z "$existing_pid" ]]; then
  existing_pid="$(port_pid "$PORT" || true)"
fi
if [[ -n "$existing_pid" ]]; then
  if [[ "$(pid_matches_server "$existing_pid" || true)" == "1" ]]; then
    stop_pid_gracefully "$existing_pid"
  else
    fail "Port ${PORT} is occupied by PID ${existing_pid}, which is not ${SERVER_SCRIPT}; refusing to stop it"
  fi
fi

nohup python -u "$SERVER_SCRIPT" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_DIR" \
  --gpu-split "$GPU_SPLIT" \
  --cache-tokens "$CACHE_TOKENS" \
  --cache-quant "$CACHE_QUANT" \
  --reserve-per-gpu-gb "$RESERVE_PER_GPU_GB" \
  --max-vram-utilization "$MAX_VRAM_UTILIZATION" \
  --runtime-headroom-gb "$RUNTIME_HEADROOM_GB" \
  --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
  --disable-host-recurrent-cache >"$LOG_FILE" 2>&1 &
server_pid="$!"
echo "$server_pid" > "$PID_FILE"

echo "Started server PID ${server_pid}; waiting for health endpoint..."
if ! wait_for_health "$WAIT_TIMEOUT"; then
  tail -n 200 "$LOG_FILE" >&2 || true
  fail "Server did not become healthy within ${WAIT_TIMEOUT}s"
fi

if [[ ! -e "/proc/${server_pid}" ]]; then
  tail -n 200 "$LOG_FILE" >&2 || true
  fail "Server exited during startup"
fi

health_json="$(curl -fsS "http://127.0.0.1:${PORT}/health")"
echo "$health_json" | python -m json.tool
validate_server_state "$server_pid" "$health_json"

rss_before_kb="$(current_rss_kb "$server_pid")"
benchmark_payload="$(python - <<PY
import json
print(json.dumps({
  "prompt_tokens": int(${PROMPT_TOKENS}),
  "max_new_tokens": int(${MAX_NEW_TOKENS}),
  "num_runs": int(${BENCH_RUNS}),
  "temperature": 0.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 0,
  "enable_thinking": False,
}))
PY
)"

benchmark_json="$(post_json "http://127.0.0.1:${PORT}/v1/internal/benchmark" "$benchmark_payload")"
printf '%s\n' "$benchmark_json" > "$BENCH_FILE"
echo "$benchmark_json" | python -m json.tool

sleep 5
rss_after_kb="$(current_rss_kb "$server_pid")"

python - "$benchmark_json" "$rss_before_kb" "$rss_after_kb" "$MAX_RSS_DELTA_MB" <<'PY'
import json
import sys
benchmark = json.loads(sys.argv[1])
rss_before_kb = int(sys.argv[2])
rss_after_kb = int(sys.argv[3])
max_delta_mb = int(sys.argv[4])
bench_delta_mb = benchmark.get("rss_delta_bytes", 0) / (1024 ** 2)
proc_delta_mb = (rss_after_kb - rss_before_kb) / 1024
if bench_delta_mb > max_delta_mb:
    raise SystemExit(f"Internal benchmark RSS delta too high: {bench_delta_mb:.2f} MiB > {max_delta_mb} MiB")
if proc_delta_mb > max_delta_mb:
    raise SystemExit(f"Observed process RSS delta too high: {proc_delta_mb:.2f} MiB > {max_delta_mb} MiB")
print(json.dumps({
    "avg_prefill_tps": benchmark.get("avg_prefill_tps"),
    "avg_decode_tps": benchmark.get("avg_decode_tps"),
    "internal_rss_delta_mb": round(bench_delta_mb, 3),
    "process_rss_delta_mb": round(proc_delta_mb, 3),
}, indent=2))
PY

echo "Deployment and benchmark completed successfully."
echo "Server PID: ${server_pid}"
echo "Health: http://127.0.0.1:${PORT}/health"
echo "Logs: ${LOG_FILE}"
echo "Benchmark JSON: ${BENCH_FILE}"
