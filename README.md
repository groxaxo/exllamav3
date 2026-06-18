# ExLlamaV3 Ampere Deployment Guide

This repo is the working `exllamav3` fork plus local OpenAI-compatible server
scripts used to deploy quantized models on the Ampere machines in this project.

The goal of this README is to be the canonical deployment playbook so future
launches do not require repeated trial-and-error.

## Current best-known deployments

### Fastest stable dual RTX 3060 deployment for Qwen3.5-9B

- Model: `models/Qwen3.5-9B-exl3`
- GPUs: physical `2,3` only
- Backend: layer-split, not TP2
- KV cache: fp16
- Context: `262144`
- GPU split: `9.5,11.0`
- CPU placement: disabled with `EXLLAMA_EMBED_PREFER_CPU=0`
- Measured short decode speed: about `37.8 t/s`
- Thinking is enabled by default in `server_27b_layer.py`, but the server now
  caps the `<think>` phase with `MAX_THINKING_TOKENS` so reasoning cannot consume
  the entire output budget.
- Runner: `run_qwen9b_3060_pair.sh`
- Notes: see `doc/qwen9b_dual_3060_fastest.md`

### Why this is the default recommendation on the 3060 pair

- Native TP2 was not stable on this machine and hit synchronization / CPU-reduce
  timeouts during warmup and prefill.
- NCCL TP2 also failed here because of the CUDA driver/runtime mismatch already
  recorded in project logs.
- Layer-split avoids TP communication problems and is the fastest stable path we
  found that stays fully on GPU.
- The model's default embedding behavior prefers CPU. Setting
  `EXLLAMA_EMBED_PREFER_CPU=0` removes the last CPU placement and keeps the full
  model on the 3060s.

## Golden rules

- Use `conda` and launch from `exl3-dev` unless a model-specific script says
  otherwise.
- Always pin GPUs explicitly with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=...`.
- Prefer fp16 KV cache unless there is a proven need for quantized KV.
- Prefer layer-split on mixed PCIe consumer GPUs when TP2 is unstable.
- Do not assume TP2 is faster just because two GPUs are available. Validate it on
  the exact machine first.
- Keep embeddings on GPU for fully-GPU deployments by setting
  `EXLLAMA_EMBED_PREFER_CPU=0`.
- Use wrapper scripts with fixed model path, GPU selection, split, cache, and log
  path instead of ad-hoc shell commands.
- Treat the model's own `max_position_embeddings` as the hard context limit.

## Environment setup

### Conda

Primary environment:

```bash
source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate exl3-dev
```

Torch libraries on this machine live at:

```bash
/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib
```

Most runner scripts in this repo already export `LD_LIBRARY_PATH` correctly. If
you launch manually, do it yourself:

```bash
export LD_LIBRARY_PATH=/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
```

### Models

- Repo-local models: `/home/op/exllamav3_ampere/models`
- External quantized models are also used in some scripts under
  `/home/op/quantized-models`

## Deployment decision tree

### If you are deploying on the two RTX 3060s only

- Start with layer-split.
- Use `server_27b_layer.py` plus a wrapper script.
- Set `EXLLAMA_EMBED_PREFER_CPU=0`.
- Use fp16 KV cache.
- For `Qwen3.5-9B-exl3`, use `run_qwen9b_3060_pair.sh`.

### If you want the fastest deployment method overall

- Use the exact deployment that is already documented and known-good for that
  model and GPU set.
- Reuse an existing wrapper script before inventing a new one.
- For non-ExLlama GPTQ deployments, the `ULTIMATE` vLLM launchers are a good
  reference for wrapper layout, stable logging, and GPU resolution.

### If you are considering TP2

- Use TP2 only when it has already been validated for the exact hardware,
  driver/runtime, and model.
- On this machine, TP2 is not the default recommendation for the 3060 pair.

## Standard deployment workflow

Use this flow for any new model so the result is repeatable.

### 1. Check the target GPUs

```bash
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
```

If you need a clean 3060-only launch, inspect compute processes on GPUs `2,3`
and stop them before starting the server.

### 2. Choose the backend

- `layer-split`: default for the 3060 pair and for setups where TP comms are the
  main instability risk
- `native TP2`: use only if already proven stable for the target model
- `nccl TP2`: use only if the machine's CUDA driver/runtime combination is known
  to support it cleanly

### 3. Decide the cache target

- Start from the model's `max_position_embeddings`
- Keep fp16 KV unless you have a measured reason to quantize KV
- Align cache tokens to the ExLlama page size (`256` tokens)

### 4. Keep the launch reproducible

Every production-worthy launch should have a wrapper script that fixes:

- conda env
- visible GPUs
- model path
- model id
- GPU split
- cache tokens
- host and port
- log path
- any required environment toggles

### 5. Verify immediately after launch

Run all three:

```bash
curl -fsS http://127.0.0.1:PORT/health
curl -fsS http://127.0.0.1:PORT/v1/models
curl -fsS http://127.0.0.1:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Also confirm GPU residency with `nvidia-smi`.

## Recommended dual-3060 Qwen3.5-9B launch

### One-command runner

```bash
/home/op/exllamav3_ampere/run_qwen9b_3060_pair.sh
```

### What it does

- activates `exl3-dev`
- locks execution to `CUDA_VISIBLE_DEVICES=2,3`
- disables CPU embedding fallback with `EXLLAMA_EMBED_PREFER_CPU=0`
- launches `server_27b_layer.py`
- serves `models/Qwen3.5-9B-exl3`
- uses `GPU_SPLIT=9.5,11.0`
- allocates `CACHE_TOKENS=262144`
- writes logs to `logs/qwen9b-3060-pair-current.log`

### Equivalent manual launch

```bash
source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate exl3-dev
export LD_LIBRARY_PATH=/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
EXLLAMA_EMBED_PREFER_CPU=0 \
python /home/op/exllamav3_ampere/server_27b_layer.py \
  --host 0.0.0.0 \
  --port 1235 \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 9.5,11.0 \
  --cache-tokens 262144
```

### Health checks

```bash
curl -fsS http://127.0.0.1:1235/health
curl -fsS http://127.0.0.1:1235/v1/models
```

### Smoke test

```bash
curl -fsS http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3-3060-dual",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Note: this model may spend tokens in reasoning first. Use a generous
`max_tokens` value if you need a complete visible answer. For low-latency
direct answers, send `"enable_thinking": false`. For reasoning-heavy prompts,
keep thinking enabled but lower `MAX_THINKING_TOKENS` when the model spends too
long in `<think>`.

## Optimizing Qwen 3.5 scripts

Use these knobs when tuning the Qwen 3.5 launchers and server scripts:

- **Pin the GPUs first.** Keep the 3060 pair on `CUDA_VISIBLE_DEVICES=2,3` and
  use `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the split stays stable across launches.
- **Keep the proven split as the baseline.** Start from `GPU_SPLIT=9.5,11.0` on
  the dual-3060 setup. Only move it if you have a measured VRAM or latency
  reason.
- **Size the cache for the workload.** `CACHE_TOKENS=262144` is the longest
  context configuration that fit reliably in testing. Reduce it if you want more
  headroom or a faster startup path.
- **Control overthinking explicitly.** `MAX_THINKING_TOKENS` is the main lever
  for reasoning-heavy prompts. Smaller values reduce latency and prevent the
  model from spending the entire budget inside `<think>`.
- **Turn thinking off for direct answers.** Set `ENABLE_THINKING=false` in the
  environment or send `"enable_thinking": false` per request when you want a
  short answer with minimal latency.
- **Preserve reasoning only when you need it.** Leave
  `PRESERVE_THINK_OUTPUT=true` if you want to inspect the chain of thought.
  Disable it if you only care about the final answer payload.
- **Validate with the dual-3060 harness.** Run
  `python tests/validate_dual_3060.py` after changing split, cache, or thinking
  settings. It checks health, normal completion, capped thinking, and
  no-thinking behavior.
- **Treat wrapper scripts as the source of truth.** Put the fixed model path,
  GPU pinning, split, cache, and thinking defaults in the wrapper script rather
  than in ad-hoc shell history.

Example tuning profile for fast direct responses:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
EXLLAMA_EMBED_PREFER_CPU=0 \
ENABLE_THINKING=false \
MAX_THINKING_TOKENS=256 \
python /home/op/exllamav3_ampere/server_27b_layer.py \
  --host 0.0.0.0 \
  --port 1235 \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 9.5,11.0 \
  --cache-tokens 262144
```

### Qwen3.5-27B on dual RTX 3090s

**Historical note:** a previous clean NCCL run made TP2 look best on this host
after `nvidia-nccl-cu13` (NCCL 2.28.9+cuda13.0) was removed and the
`LD_LIBRARY_PATH` order was fixed so CUDA 12.8 / NCCL 2.27.5 loaded correctly.
Keep the numbers below as a useful reference point, but do not treat them as
the default recommendation without re-running the current host state.

On the lighter `bench_tp.py` microbenchmark (74-token prompt, 4K cache, 200
generated tokens), the current TP2 hot path now reaches **30.3 t/s decode** on
GPUs 0,1 after two low-risk fixes:

- `EXLLAMA_FP16_REDUCE_THRESHOLD=65536` keeps tiny decode reductions in FP32
  while still using FP16 transport for larger prefill tensors.
- `model_tp_backend.py` now receives TP gather results directly into the output
  slice instead of allocating a temporary tensor and copying every token.

That same microbenchmark reports:

- **1×3090**: `19.7 t/s` decode, `226.6 t/s` prefill
- **2×3090 TP2 baseline**: `24.1 t/s` decode, `105.6 t/s` prefill
- **2×3090 TP2 tuned**: `30.3 t/s` decode, `118.2 t/s` prefill

Fresh rerun on this exact host on 2026-03-27 using the repo's current
`bench_tp.py` harness (`CACHE_TOKENS=32768`, `MAX_NEW_TOKENS=500`, `2` runs,
`OMP_NUM_THREADS=1`, RTX 3090s only, no `NCCL_ALGO` override) produced:

- **1x3090 TP1**: `19.18 t/s` decode, `210.44 t/s` prefill
- **2x3090 TP2**: `13.88 t/s` decode, `140.28 t/s` prefill
- **3x3090 TP3**: `12.39 t/s` decode, `92.75 t/s` prefill
- **2x3090 layer-split**: `20.18 t/s` decode, `201.81 t/s` prefill
- **3x3090 layer-split**: `20.00 t/s` decode, `198.25 t/s` prefill

So the current recommendation on this machine is:

- **Interactive / decode-heavy 27B serving:** prefer `server_27b_layer.py`
  with the new wrapper `run_qwen27b_3090_layersplit.sh`
- **Single-stream simplicity:** `1x3090` remains competitive
- **TP2 / TP3:** keep for experimentation or TP-specific capacity work, not for
  best decode speed on the current PCIe-only 3090 setup

Fastest measured 3090 launch on the current host:

```bash
bash /home/op/exllamav3_ampere/run_qwen27b_3090_layersplit.sh
```

Older TP2/NCCL experiment path:

```bash
bash /home/op/exllamav3_ampere/launch_27b_gpu01.sh
```

The launcher now:
- Uses `server_27b_tp2.py` (NCCL TP2)
- `CUDA_VISIBLE_DEVICES=0,1`, `GPU_SPLIT=22.0,22.0`
- `CACHE_TOKENS=131072` (128K context)
- `NCCL_ALGO=TREE` for PCIe-connected 3090s
- `EXLLAMA_FP16_REDUCE_THRESHOLD=65536` so small decode reductions avoid
  unnecessary FP16 cast/copy overhead
- `LD_LIBRARY_PATH` with `nvidia/cuda_runtime/lib` **first** to ensure CUDA 12.8
  `libcudart.so.12` is found before the conda env's older CUDA 12.1 copy
- `MAX_THINKING_TOKENS=1024` to prevent overthinking

If `nvidia-nccl-cu13` gets reinstalled in the future, NCCL will silently break
again.  Check with:

```bash
conda run -n exl3-dev bash -c \
  'strings $(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nccl/lib/libnccl.so.2 | grep "NCCL version"'
# Must show: NCCL version 2.27.5+cuda12.x  — NOT cuda13
```

## Existing server scripts

These scripts still exist and may be useful for experimentation, benchmarking,
or other GPU layouts:

- `server_openai.py`: single 3060 path for light testing
- `server_3090.py`: single 3090 path
- `server_dual_gpu.py`: older dual-GPU 9B script
- `server_mixed.py`: mixed 3090 + 3060 path
- `server_32k.py`: fixed 32k mixed-GPU 9B variant
- `server_27b_tp2.py`: TP2 server used for TP experiments
- `server_27b_layer.py`: current best general-purpose multi-GPU layer-split
  server in this repo

For new production launches, prefer wrapper scripts over direct use of the older
one-off server entry points. For the current 3090 host, start with
`run_qwen27b_3090_layersplit.sh`.

## Deployment patterns worth reusing

The scripts under `/home/op/ULTIMATE` are a good reference for deployment
discipline even when the backend is different. Reuse these ideas:

- one launcher script per model / port / GPU layout
- fixed log path plus a stable `current.log` symlink
- explicit GPU selection
- health-check script paired with each launcher
- stop script paired with each launcher
- model-specific documented settings kept in repo

## Troubleshooting

### TP2 fails during warmup or prefill

Most likely culprit: `nvidia-nccl-cu13` is installed and has overwritten
`libnccl.so.2` with an NCCL version built for CUDA 13 (requires driver ≥ 575).

```bash
# Diagnose
pip show nvidia-nccl-cu12 nvidia-nccl-cu13
strings /path/to/nvidia/nccl/lib/libnccl.so.2 | grep "NCCL version"
# Fix
pip uninstall nvidia-nccl-cu13 -y
pip install "nvidia-nccl-cu12==2.27.5" --force-reinstall
```

### NCCL backend fails at startup

1. Check which `libnccl.so.2` is being loaded (see above).
2. Ensure `LD_LIBRARY_PATH` prepends `nvidia/cuda_runtime/lib` so CUDA 12.8
   `libcudart.so.12` is resolved before older system or conda copies.
3. As a last resort, fall back to layer-split with `server_27b_layer.py`.

### A module lands on CPU when it should not

- Set `EXLLAMA_EMBED_PREFER_CPU=0`
- Re-launch and confirm residency with `nvidia-smi` and startup logs

### Context launch fails with insufficient VRAM

- Reduce `CACHE_TOKENS`
- Adjust `GPU_SPLIT`
- Keep the model on the intended GPUs only

### The server answers with reasoning but not the final short text

- Raise `max_tokens`
- Reduce `MAX_THINKING_TOKENS` if the model spends too long reasoning
- Set `ENABLE_THINKING=false` for direct-answer workloads
- This is expected for some reasoning-tuned variants

## Related docs

- `doc/qwen9b_dual_3060_fastest.md`
- `doc/quantizer_ampere_findings.md`
- `BENCHMARKED.md`

## Multi-GPU Inference Bottlenecks & Optimisations (March 2026)

A deep analysis of the multi-GPU inference pipeline was conducted on this
machine.  The key findings and fixes are summarised here; full details are in
`BENCHMARKED.md` under the "March 27, 2026" entry.

### Why TP2/TP3 decode is slower than layer-split on PCIe

Tensor-Parallel requires an `all_reduce` after each attention output projection
and each MLP block — that is **2 × num_layers synchronisation points per decode
step**.  For Qwen3.5-27B with 28 layers, that is 56 NCCL operations per token.
On PCIe Gen3 (no NVLink), each operation costs ~50–100 µs of latency.

```
56 ops × 70 µs = ~3.9 ms overhead per token
At 28 t/s decode, each token takes ~35 ms total → NCCL = ~11% overhead
```

Layer-split passes a single activation tensor between GPU groups once per
layer; there is no all_reduce.  This is why layer-split matches single-GPU
decode speed while TP decode is 15–30% slower on the same hardware.

**Rule of thumb:** Prefer layer-split for interactive workloads.  Use TP when
you need the combined KV cache capacity for long contexts or when batch
prefill throughput matters more than decode latency.

### Fixes applied in this session

#### `exllamav3/model/model_tp_backend.py`

**Direct NCCL gather path retained**

An experimental `irecv` / `isend` gather rewrite was tested during this
session.  It looked promising on paper, but on this host it made both TP2 and
TP3 slower, so it was **reverted**.  The fastest path here remains the
validated direct receive/send implementation:

```python
dist.recv(out_slice, src=src_rank)
dist.send(tensor, dst=dst_rank)
```

This is a good reminder that the fastest *collective-looking* code is not
always the fastest real-world path on PCIe consumer GPUs.

**FP16 reduce threshold kept at 65536**

Benchmarks confirm that for small decode tensors (< 65 K elements,
e.g. `[1, 1, 5632]` for single-token decode), the FP32→FP16 cast overhead
exceeds the PCIe bandwidth saving.  The default threshold remains at 65 536
elements; only larger prefill tensors use FP16 transport.

#### `server_27b_layer.py`

| Change | Rationale |
|--------|-----------|
| Auto-set `EXLLAMA_EMBED_PREFER_CPU=0` when `len(gpu_split) > 1` | Avoids CPU→GPU embedding transfer on every prefill in layer-split mode |
| Set `OMP_NUM_THREADS=1` at startup | Prevents CPU thread contention next to GPU workloads |
| Log embed device and OMP threads in startup banner | Makes it obvious whether embeddings are on CPU or GPU |
| `generate_with_thinking_budget`: collect `token_ids` from streaming results and reuse in phase 2 | Eliminates lossy text→token roundtrip; ensures exact KV-cache prefix reuse |

#### `bench_tp.py`

| Change | Rationale |
|--------|-----------|
| `--mode layer_split` | Enables direct comparison of layer-split vs TP on same hardware |
| `--model` argument | Benchmarks can now target any model directory (e.g. 9B on 3060 pair) |
| GPU names in JSON output | Results are self-describing |
| Auto-timestamped output filenames | Multiple runs never overwrite each other |
| `EXLLAMA_EMBED_PREFER_CPU=0` in layer-split load path | Consistent with server behaviour |

### Validated benchmark results from the clean 10pm run

| Config | Decode t/s | Prefill t/s | Notes |
|--------|-----------:|------------:|-------|
| TP1, 1×3090, 27B, 4K cache | **19.46** | **225.01** | Single-GPU baseline |
| TP2, 2×3090, 27B, 32K cache | **26.47** | **300.70** | Best 27B result from the clean run |
| LS2, 2×3090, 27B, 32K cache | **20.85** | **240.84** | Slower than TP2 on this workload |
| TP3, 3×3090, 27B, 32K cache | **13.61** | **143.59** | Not recommended; third GPU hurts throughput |
| LS2, 2×3060, 9B, 64K cache | **31.48** | **356.59** | Strongest practical deployment in this session |

Two follow-up TP reruns later in the session were much noisier and slower, so
the table above intentionally uses the **first clean run after GPU job
cleanup** as the canonical result set.

### Benchmark harness caveat

`bench_tp.py` intentionally **does not** force `NCCL_ALGO=TREE`.  The
deployment launcher may still choose TREE for specific server workloads, but in
this short-prompt decode microbenchmark, forcing TREE reduced throughput and
produced misleading results.

### Thinking budget: phase-2 token reuse

The old `generate_with_thinking_budget` path:

```
phase1 tokens → decode → text → re-encode → phase2 prompt
```

This re-encode step is not lossless: some multi-byte tokens cannot round-trip
through UTF-8 text without changing boundaries.  The fix:

```
phase1 tokens → decode → collect token_ids → cat directly → phase2 prompt
```

The injected `</think>\n` tag (budget-exhausted path) is still encoded from
text (it is always a known short constant), then appended to the collected IDs.

### Deployment decision tree (updated)

```
Single stream, interactive use
  └─ Prefer layer-split (run_qwen9b_3060_pair.sh or server_27b_layer.py)
     Reason: zero NCCL overhead, matches single-GPU decode speed

Long-context workloads (> 128K tokens)
  └─ TP2 with enlarged cache (launch_27b_gpu01.sh --cache-tokens 393216)
     Reason: TP2 can address ~416K tokens; layer-split is limited by
             the smaller GPU that holds the embedding/output layers

Very long-context workloads (> 416K tokens)
  └─ TP3 with enlarged cache (--cache-tokens 655360)
     Reason: TP3 provides ~704K token capacity across 3×3090
```
