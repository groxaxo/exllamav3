# Dual 3060 Deployment Test Results

**Date**: 2026-03-25
**Server**: `http://127.0.0.1:1235`
**Model**: `Qwen3.5-9B-exl3-3060-dual`

## Server Startup

```
[2026-03-25T23:03:11Z] Starting qwen9b_3060_pair
Model:                 /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3
Model ID:              Qwen3.5-9B-exl3-3060-dual
Host:Port:             0.0.0.0:1235
CUDA_VISIBLE_DEVICES:  2,3
GPU split:             9.5,11.0
Cache tokens:          262144
EXLLAMA_EMBED_PREFER_CPU: 0

  Qwen3.5-27B-exl3 heretic - layer-split 2-GPU server
  Model dir    : /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3
  GPU split    : [9.5, 11.0]
  Mode         : layer-split (pipeline, no TP comms)
  Cache tokens : 262,144
  Model limit  : 262,144
  KV cache     : fp16 (no quantization)
  Max tokens   : 16000 (cap 16000)
  Thinking     : True (preserve=True)

  GPU 0: NVIDIA GeForce RTX 3060  (11.6 GB)
  GPU 1: NVIDIA GeForce RTX 3060  (11.6 GB)

  GPU 0: alloc=8.79GB  reserved=9.07GB
  GPU 1: alloc=6.69GB  reserved=6.91GB

  Warmup complete in 0.325s
  Server ready.  Cache = 262,144 tokens.
```

## Health Check

```bash
curl -fsS http://127.0.0.1:1235/health
```

```json
{
  "status": "ok",
  "model": "Qwen3.5-9B-exl3-3060-dual",
  "cache_tokens": 262144,
  "max_position_embeddings": 262144,
  "thinking_enabled": true,
  "preserve_think_output": true
}
```

## Models Endpoint

```bash
curl -fsS http://127.0.0.1:1235/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.5-9B-exl3-3060-dual",
      "object": "model",
      "created": 1774480380,
      "owned_by": "local",
      "context_length": 262144,
      "max_position_embeddings": 262144
    }
  ]
}
```

## GPU Residency

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
```

```
GPU-72f3cda5-5f26-bed3-6093-0608adb365e3, 130078, python, 9596
GPU-cbfc8a5f-0df1-ca71-f704-0d09a707d2ac, 130078, python, 7386
```

Both 3060 GPUs are in use by the same process (`130078`).

## Throughput Benchmark

### Raw Decode Speed (Short Prompts)

Test configuration:
- Split: `9.5, 11.0`
- Cache: `262144`
- Prompt: short chat
- New tokens: `127`

| Split | Prefill (s) | Decode (s) | Decode t/s |
|-------|-------------|------------|------------|
| 5.5, 11.0 | 0.121 | 3.464 | 36.67 |
| 7.5, 11.0 | 0.060 | 3.460 | 36.70 |
| 8.5, 11.0 | 0.061 | 3.375 | 37.63 |
| 9.5, 11.0 | 0.061 | 3.359 | **37.80** |
| 10.5, 11.0 | 0.061 | 3.376 | 37.62 |

Best: `~38 tok/s` at split `9.5, 11.0`

### End-to-End with Thinking

Test with thinking enabled, `max_tokens=1024`:

```
elapsed_s: 159.898
prompt_tokens: 27
completion_tokens: 1023
end_to_end_tps: 6.4
finish_reason: stop
```

The model spends many tokens on reasoning loops, reducing practical throughput.

## Functional Tests

### Test 1: Simple Math

```bash
curl -fsS http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3-3060-dual",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Result: Correct behavior, but model spends tokens on thinking before the short answer.

### Test 2: Thinking with FINAL Tag

```bash
curl -fsS http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3-3060-dual",
    "messages": [{"role": "user", "content": "Think step by step, then end with a final line starting with FINAL:. In one sentence, explain why 2+2=4."}],
    "max_tokens": 16000,
    "temperature": 0
  }'
```

Final output line:
```
FINAL: 2+2=4 is true because the axioms of arithmetic define the operation of addition such that combining two units with two other units results in a total of four units.
```

Finish reason: `stop` (not truncated)

## Key Findings

1. **TP2 not viable**: Native backend times out; NCCL fails with driver mismatch
2. **Layer-split works**: Stable, fully-GPU, ~38 tok/s raw decode
3. **Embedding CPU fallback**: Must set `EXLLAMA_EMBED_PREFER_CPU=0`
4. **Full context fits**: 262K tokens with fp16 KV on 2x 3060
5. **Thinking overhead**: Real API throughput ~6 tok/s when thinking is active
6. **Use large max_tokens**: 1024+ for simple, 2048+ for normal, 4096+ for complex tasks
