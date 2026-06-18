#!/usr/bin/env bash
# Quick progress snapshot
LOG="/home/op/exllamav3_ampere/logs/quantize_122b_ablit_4bpw_3x3090.nohup.log"
PID=$(cat /home/op/exllamav3_ampere/logs/quantize_122b_4bpw.pid 2>/dev/null)

printf '=== [%s] Quantization Monitor ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID: RUNNING"
else
  echo "Process $PID: STOPPED"
fi
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "--- Recent Log ---"
grep -E "Quantized:.*\[.*s\]|Estimated remaining|complete" "${LOG}" | tail -10
