#!/usr/bin/env bash
# Validation tests for dual-3060 Qwen3.5-9B deployment
# Run these after starting the server with run_qwen9b_3060_pair.sh

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1235}"
BASE_URL="http://${HOST}:${PORT}"

echo "=== Dual 3060 Deployment Validation ==="
echo "Target: ${BASE_URL}"
echo

# Test 1: Health check
echo "--- Test 1: Health Check ---"
curl -fsS "${BASE_URL}/health" | python3 -m json.tool
echo

# Test 2: Models endpoint
echo "--- Test 2: Models Endpoint ---"
curl -fsS "${BASE_URL}/v1/models" | python3 -m json.tool
echo

# Test 3: Simple completion (no thinking instruction)
echo "--- Test 3: Simple Completion ---"
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3-3060-dual",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 64,
    "temperature": 0
  }' | python3 -m json.tool
echo

# Test 4: Thinking completion with FINAL tag
echo "--- Test 4: Thinking Completion with FINAL ---"
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3-3060-dual",
    "messages": [{"role": "user", "content": "Think step by step, then end with a final line starting with FINAL:. What is 2+2?"}],
    "max_tokens": 2048,
    "temperature": 0
  }' | python3 -m json.tool
echo

# Test 5: GPU verification
echo "--- Test 5: GPU Verification ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
echo

echo "=== All Tests Passed ==="
