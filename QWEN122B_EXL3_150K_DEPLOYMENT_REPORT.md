# Qwen3.5-122B-A10B-abliterated EXL3 4bpw Deployment Report

## Outcome

Final deployed model:
- Local path: `/home/op/exllamav3_ampere/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw`
- Hugging Face: `https://huggingface.co/groxaxo/Qwen3.5-122B-A10B-abliterated-exl3-4bpw`

Final chosen serving configuration:
- Server: `0.0.0.0:1234`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,4,3`
  - logical order = `3090, 3090, 3090, 3060`
- Split: `22.8,22.8,22.8,10.6`
- KV cache: `fp16`
- Cache tokens: `150016`
- Thinking: `disabled`
- Result: this was the fastest measured configuration that satisfied the user's revised requirement of using `8-bit cache or fp16 cache only` at about `150k` context.

## Benchmarks

### Final comparison at ~150k context

| Layout | Cache | Context | Decode tok/s | Prefill tok/s | Result |
|---|---:|---:|---:|---:|---|
| 3x3090 | q8 | 150016 | 44.28 | 249.59 | slower |
| 3x3090 + 1x3060 | q8 | 150016 | 45.87 | 269.14 | slower |
| 3x3090 + 1x3060 | fp16 | 150016 | **52.91** | **303.74** | **winner** |

Benchmark artifacts:
- `bench_122b_ls3_150k_q8.json`
- `bench_122b_ls4_150k_q8.json`
- `bench_122b_ls4_150k_fp16.json`

### Earlier max-context reference

This was measured earlier before the cache-mode requirement changed:

| Layout | Cache | Context | Decode tok/s | Prefill tok/s |
|---|---:|---:|---:|---:|
| 3x3090 + 1x3060 | q4 | 262144 | 47.53 | 281.30 |

Reference artifact:
- `bench_122b_ls4_262k_q4_3060last.json`

## Perplexity

Perplexity was rerun cleanly on the chosen split without the earlier invalid cache flag.

- Command shape:
  - `CUDA_VISIBLE_DEVICES=0,1,4,3 python eval/ppl.py -m ... -gs 22.8,22.8,22.8,10.6 -r 20 -l 2048`
- Result:
  - `Perplexity: 5.155404`
- Log:
  - `logs/ppl_122b_4bpw_20rows.log`

## Functional validation

### Server health

The final server is healthy and currently serving on `0.0.0.0:1234`.

Health response:
```json
{"status":"ok","model":"Qwen3.5-122B-A10B-abliterated-exl3-4bpw","cache_tokens":150016,"cache_quant":null,"max_position_embeddings":262144,"thinking_enabled":false,"preserve_think_output":false,"max_thinking_tokens":1024}
```

### Exact-output test

Prompt: `Reply with exactly READY and nothing else.`

Result:
```text
READY
```

### Coding/aider validation

A real isolated `aider` test was run against the OpenAI-compatible endpoint and succeeded.

- Env: `aider-env`
- API base: `http://127.0.0.1:1234/v1`
- Result: successful edit, return code `0`

## Final launch command

```bash
source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate exl3-dev
cd /home/op/exllamav3_ampere
PYTHONUNBUFFERED=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,4,3 \
OMP_NUM_THREADS=1 \
ENABLE_THINKING=false \
PRESERVE_THINK_OUTPUT=false \
EXLLAMA_STARTUP_WARMUP=1 \
python -u server_27b_layer.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw \
  --gpu-split 22.8,22.8,22.8,10.6 \
  --cache-tokens 150016 \
  --host 0.0.0.0 \
  --port 1234
```

Live log:
- `logs/server_122b_150k_fp16_port1234_final.log`

## Notes

- `fp16` KV cache outperformed `q8` on this machine at the target context, so `q8` was not chosen.
- The 3060 works best as the last stage in the layer-split ordering for this deployment.
- The Hugging Face repo already contains the full 18-file model folder and is ready for use.
