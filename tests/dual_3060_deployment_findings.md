# Dual RTX 3060 Deployment Findings

## Target

Deploy `Qwen3.5-9B-exl3` on the two RTX 3060 GPUs (physical `2,3`) with:
- Longest possible context
- Fastest stable inference
- fp16 KV cache
- No CPU placement for embeddings or model weights

## What Was Tried

### 1. Tensor Parallel (TP2) - Native Backend

```bash
CUDA_VISIBLE_DEVICES=2,3 python server_27b_tp2.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 11.0,11.0 \
  --cache-tokens 32768 \
  --tp-backend native
```

**Result: Failed**

Error during warmup/prefill:
```
RuntimeError: Synchronization timeout
CPU reduce wait timeout
```

The native TP2 backend hit synchronization timeouts in the CPU reduce path during the first forward pass. The model loaded, but the Generator prefill never completed.

### 2. Tensor Parallel (TP2) - NCCL Backend

```bash
CUDA_VISIBLE_DEVICES=2,3 python server_27b_tp2.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 11.0,11.0 \
  --cache-tokens 32768 \
  --tp-backend nccl
```

**Result: Failed**

Error at NCCL initialization:
```
torch.distributed.DistBackendError: NCCL error
Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'
```

NCCL TP2 is not viable on this machine due to the CUDA driver/runtime mismatch.

### 3. Layer-Split with CPU Embedding (Default)

```bash
CUDA_VISIBLE_DEVICES=2,3 python server_27b_layer.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 9.5,11.0 \
  --cache-tokens 262144
```

**Result: Partially Working**

- Model loaded successfully
- All transformer layers on `cuda:0` and `cuda:1`
- Embedding module landed on CPU (default behavior)
- This violates the "no CPU placement" constraint

### 4. Layer-Split with GPU-Only Embeddings

```bash
CUDA_VISIBLE_DEVICES=2,3 EXLLAMA_EMBED_PREFER_CPU=0 \
  python server_27b_layer.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 9.5,11.0 \
  --cache-tokens 262144
```

**Result: Success**

- All modules on GPU (no CPU placement)
- Embeddings on `cuda:0`
- Transformer layers split across both GPUs
- Full 262K context allocation succeeded
- Warmup passed
- Server stable

## Final Configuration

| Parameter | Value |
|-----------|-------|
| Model | `models/Qwen3.5-9B-exl3` |
| GPUs | Physical `2,3` (RTX 3060 × 2) |
| Backend | layer-split |
| GPU Split | `9.5, 11.0` |
| Cache Tokens | `262144` |
| KV Cache | fp16 |
| Embedding Placement | GPU (`EXLLAMA_EMBED_PREFER_CPU=0`) |
| Max Tokens | `16000` |
| Host | `0.0.0.0` |
| Port | `1235` |

## Placement Verification

After load:
```
GPU 0: alloc=8.79GB  reserved=9.07GB
GPU 1: alloc=6.69GB  reserved=6.91GB
```

Module distribution:
- `cuda:0`: 20 modules (embedding + transformer layers)
- `cuda:1`: 15 modules (transformer layers)
- `cpu`: 0 modules

## Speed Results

### Raw Decode Benchmark

Short prompt, 128 token generation, greedy sampling:

| GPU Split | Prefill (s) | Decode (s) | Tokens | Decode t/s |
|-----------|-------------|------------|--------|------------|
| 5.5, 11.0 | 0.09 | 3.479 | 127 | 36.51 |
| 6.5, 11.0 | 0.061 | 3.382 | 127 | 37.55 |
| 7.5, 11.0 | 0.062 | 3.38 | 127 | 37.58 |
| 8.5, 11.0 | 0.061 | 3.375 | 127 | 37.63 |
| 9.5, 11.0 | 0.061 | 3.359 | 127 | **37.8** |
| 10.5, 11.0 | 0.061 | 3.376 | 127 | 37.62 |

Best: `~38 tok/s` raw decode at split `9.5, 11.0`

### End-to-End API with Thinking Enabled

Prompt asking for step-by-step reasoning, `max_tokens=1024`:

- Elapsed: `159.9s`
- Completion tokens: `1023`
- End-to-end t/s: `6.4`
- Finish reason: `stop`

The model spends many tokens on visible reasoning loops, so practical throughput is lower than raw decode speed.

## Lessons

1. TP2 is not viable on the 3060 pair on this machine (native timeouts, NCCL driver mismatch)
2. Layer-split is the stable path for dual-3060 inference
3. Default embedding placement is CPU; must set `EXLLAMA_EMBED_PREFER_CPU=0` for fully-GPU deployment
4. Split `9.5, 11.0` gives the best balance of speed and stability
5. Full 262K context fits on the 3060 pair with fp16 KV cache
6. Raw decode is `~38 t/s`, but thinking-enabled workloads are slower due to reasoning overhead

## Files Changed

- `exllamav3/modules/embedding.py`: added `EXLLAMA_EMBED_PREFER_CPU` toggle
- `run_qwen9b_3060_pair.sh`: new runner script
- `doc/qwen9b_dual_3060_fastest.md`: deployment notes
- `README.md`: updated with deployment playbook
