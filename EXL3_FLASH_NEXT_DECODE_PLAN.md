# EXL3 Flash-Next Decode Optimization Plan (v2)

Date: 2026-09-21 (rev 2 after review)
Target: match the 80 tok/s single-stream decode of the vLLM AutoRound build,
using the EXL3 quant (KLD 0.0067 vs BF16, 2.2x better than the AutoRound mixed)
on the same 3x RTX 3090 hardware.

Multi-GPU policy: **pipeline (layer-split) is the reference topology** for every
phase in this plan. Tensor parallel TP3 exists upstream (`supports_tp=True`,
commit 9d18a7f) and is tested only as an optional separate experiment — never
mixed into a phase's A/B comparisons.

Reference deployment (the one to match):
- Model: `/home/op/models/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound` (served by
  container `flash-next`, launch script `/home/op/op/../flash-next/run-flash-next.sh`)
- Measured: 80 tok/s single-stream flat 3.6k-248k ctx, ~155 tok/s @ 4 reqs,
  prefill 3,701 tok/s @ 248k. GPU kernel floor 12.6 ms/step.
- Author's writeup: https://minamism.com/posts/flash-next-vllm/

---

## Why the vLLM build is fast (the mechanism being ported)

The speed does not come from the INT4 quant. Per the author's own decomposition,
a decode step spends 12.6 ms in GPU kernels; everything above that was host
stall. Two patches took single-stream from 45 -> 80 tok/s:

1. **PLE host-gather hoist** (`flash-next-decode-01-ple-host-gather.patch` +
   `flash-next-decode-02-model-state-hook.patch`, ~230 lines total):
   the n-gram table lookup is a host-side gather (it reads row ids on CPU).
   Inside `forward()` it synchronizes host and device every step, destroying
   run-ahead. The patches move the gather to `prepare_inputs()` (runs on host
   right before forward) and reduce `forward()` to reading a fixed-address
   device buffer allocated once. Note: measured alone this changes nothing
   (45.1 tok/s under PIECEWISE); its value is that it makes full-graph capture
   legal.
2. **Full decode CUDA graphs** (`cudagraph_mode: FULL_DECODE_ONLY`): the whole
   decode forward captured as one graph, no per-layer eager islands. This is
   what pays.

Both patch files are readable at:
`/home/op/models/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound/vllm-patch/3x3090/`

## State of the trees (verified 2026-09-21, rev 3)

Stale reference checkout (do NOT benchmark from it): `/home/op/exllamav3`
(v1.5.0+cu128). Env: `/home/op/miniforge3/envs/exl3` (torch 2.11.0).
**Engine baseline: fork branch `qwen38-flash-next-decode-v2`** at
`/home/op/src/exllamav3-fork` = upstream master `6b84a21` (v1.5.1) plus this
branch's own commits (parity server, plan revisions, server hardening). The
server refuses to start when `EXPECTED_EXL3_ROOT` is set and the imported
`exllamav3` does not live under it, so the wrong checkout cannot be
benchmarked silently. (The old `qwen38-gguf-parity-server` branch predates
flash-next engine support and is retired for engine work.)

| Piece | State |
|---|---|
| flash-next arch support | present in v2 branch: `modules/ple.py`, `modules/qsa_indexer.py`, `modules/gated_delta_net.py`, `modules/hyperconnections.py`, arch_specific |
| n-gram table | streaming first-class: `DiskTensorHandles`, trellis codec, pinned staging (`modules/ngram_embedding.py`). Async `prefetch()` exists with ThreadPoolExecutor + CUDA events, but **declines below `PREFETCH_MIN_TOKENS = 256`** (line 45) — decode-sized inputs take the inline gather inside `forward()`. A/B switch: `EXL3_NGRAM_PREFETCH` |
| PLE prefetch plumbing | `ple.py:220` sets `caps["prefetch_ids"] = True`; `model.py:366,391` calls `m.prefetch(x, params)` on layers advertising it — the hook vLLM's patch-02 added already exists here |
| QSA cache | `cache/qsa.py`: indexer planes (raw + pooled keys) alongside K/V, all VRAM-resident. No host offload |
| CUDA graphs | none anywhere in the Python tree (`graph.cu` in ext has no callers) |
| Decode seam | `generator/generator.py` — `iterate_gen()` calls `self.model.forward(...)` |
| Sharding | pipeline layer-split (default) and TP both available; Qwen4Exp `supports_tp=True` |

---

## Phase 0 — Restore the quant (prerequisite)

The EXL3 quants were deleted in the 2026-09-21 disk cleanup (287G freed).
Re-download the exact quant from the author's KLD table:

- Repo: `turboderp/Qwen3.8-Flash-Next-exl3`
- **Revision (git branch): `4.05bpw_h6_ng6`** — the KLD-0.0067 entry
- Full size: **100.1 GiB** (9 model shards ~8.06G each = 64.8G, plus
  `ngram_embedding.safetensors` 39.0G, `vision_k6.safetensors` 0.56G,
  `quantization_config.json` 96M, tokenizers). The ~37G figure previously
  recorded here was the n-gram trellis alone, not the checkpoint.
- Disk has room (564G free at time of writing).

```
hf download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 4.05bpw_h6_ng6 \
  --local-dir /home/op/models/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6
```

Record in this file under Measurements: HF repo, revision, total bytes,
ngram bytes, and the local dir actually used. Do NOT restore
`exl3-unsloth-shaped-h8-mtp5` (98G) unless MTP experiments resume; the author
measured MTP slower than no drafting in real use.

Do NOT trust the quant name from memory: `4.00bpw_hq_h6_ng6` (rev 1 of this
plan) does not exist; the user's own `groxaxo/Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6`
is a different custom quant.

Runtime pins are logged automatically by the server at startup
(`_log_runtime_pins`): python/torch/CUDA/driver/GPU count, exllamav3 version +
`__file__` (editable-install trap), model dir + arch + supports_tp. Verify
`exllamav3.__file__` points at the env's install before load; a pip/site-packages
x git-checkout mismatch produces ghost debugging.

### Phase 0 smoke test (deliberately boring)

```
export CUDA_VISIBLE_DEVICES=0,1,2
export MODEL_DIR=/home/op/models/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6
export SERVED_MODEL_NAME=qwen38-flash-next-exl3-4.05bpw-h6-ng6
export CHAT_TEMPLATE_FILE="$MODEL_DIR/chat_template.jinja"   # quant ships its own
export EXPECTED_EXL3_ROOT=/home/op/src/exllamav3-fork        # refuse wrong checkout
export MULTI_GPU=pipeline          # layer-split, the reference topology
export GPU_SPLIT=22.5,22.5,22.5
export CACHE_SIZE=65536
export CHUNK_SIZE=4096
export MAX_BATCH_SIZE=1            # B1 everywhere for Phases 0-3
export NGRAM_RAM=1
cd qwen38_flash_gguf_parity && python -u exl3_qwen38_server.py
```

Then `curl /health`, then one completion with `temperature=0, seed=12345,
max_tokens=32` asking for exactly `PHASE0_OK`. We do not care yet whether it
does 30, 45 or 70 tok/s.

### Phase 0 acceptance gate
- [ ] Startup ran under `EXPECTED_EXL3_ROOT=/home/op/src/exllamav3-fork`
      (guard active: wrong-checkout import aborts)
- [ ] Exact ExLlamaV3 git SHA + clean/dirty recorded (pin log `git=` / `tree=`)
- [ ] Exact torch/CUDA/driver recorded
- [ ] Exact model repo + revision + total/ngram bytes recorded
- [ ] Qwen4Exp architecture detected in the pin log
- [ ] PLE / QSA / NGramEmbedding present (pin log + module count)
- [ ] 3x RTX 3090 detected
- [ ] Pipeline layer-split load succeeds (log: `multi-gpu mode: layer-split pipeline`)
- [ ] 3 identical deterministic requests complete (`PHASE0_OK` x3)
- [ ] No NaN/Inf/error in the log
- [ ] EOS terminates generation correctly
- [ ] Peak VRAM recorded per GPU
- [ ] NGRAM mode recorded (RAM vs disk)
- [ ] (optional, separate session) `MULTI_GPU=tensor` TP3 load + same smoke test

## Phase 1 — Baseline and profile

Methodology copied from the author's article:

- Single-stream tok/s at contexts 3.6k / 121k / 248k (expect roughly flat if
  the QSA path behaves; the number itself is the baseline to beat)
- `torch.profiler` trace of ~100 decode steps: measure GPU-busy fraction and
  host gap per step. The vLLM floor was 12.6 ms; establish ours (trellis
  kernels, Ampere)
- Record: power draw, PCIe read/token (from `nvidia-smi dmon`), the position
  of every `.synchronize()/.cpu()/.item()` on the decode path

Acceptance: numbers written into this file under "Measurements" below.
No patching before this exists.

## Phase 2 — Decode-path gather via the existing prefetch machinery

Do NOT port the vLLM patch wholesale. Upstream already ships the ingredients:
`NGramEmbedding.prefetch()` (ThreadPoolExecutor, pinned staging, CUDA event),
the `prefetch_ids` cap, and `model.py` calling `m.prefetch(x, params)` before
the module's forward. Decode simply declines it today (`PREFETCH_MIN_TOKENS`).
The change is therefore small and surgical:

1. `exllamav3/modules/ngram_embedding.py`:
   - Add a fixed device output buffer `torch.zeros(max_rows, heads, dim)`,
     allocated once at init. Zeros, not empty: a forward running before the
     first gather must read zeros (matters for graph capture warmup).
   - Add `host_gather(ids)`: disk gather + H2D upload running OUTSIDE
     forward, writing into that buffer; reuses `_acquire_pin`/`_stage`/executor
     — the change is who calls it and when (the generator, not the module's
     own forward).
   - Reduce the inline gather branch of `forward()` to returning a view
     `self._gather_buffer[:n]` of the fixed buffer. Address must never change:
     graph replays re-read the same pointer.
2. `exllamav3/generator/generator.py` (`iterate_gen`, before the
   `self.model.forward` call): invoke `host_gather()` with the final row ids —
   the vLLM `prepare_inputs` equivalent, riding the existing prefetch hook
   rather than a new one.
3. Dummy-input path: during capture/profiling, gather zeros so the PLE
   projection has consumable data (patch-02's `_ple_dummy_input_ids`).
4. Make `PREFETCH_MIN_TOKENS` (or a new env, mirroring `EXL3_NGRAM_PREFETCH`)
   the A/B switch restoring the old in-forward gather on the same build.

Acceptance: generation output identical (same seed, same prompt) with hoist on
and off. Throughput measured; expected ~no change alone (see F2 note above) —
its value is making Phase 3 capture legal.

## Phase 3 — Per-stage decode CUDA graphs (the 45->80 multiplier, 1-2 weeks)

New module `exllamav3/generator/cudagraph.py`. **One graph per pipeline
stage/device, NOT one graph around `Model.forward()`.** `Model.forward()`
runs `prepare_inputs()`, the prefetch loop, `forward_ls()` across several
CUDA devices, and `advance_recurrent_states()` which mutates Python state
(`r.position`, `r.last_history`, `post_advance()`). PyTorch CUDA graph
capture is per-device; whole-function capture over a multi-device layer-split
forward is not supported. The vLLM reference itself is **PP=3, TP=1** with
decode graphs captured per rank; its pipeline round trip costs ~1.1 ms.

Decode step shape after this phase:

```
generator host prep (ids, positions, row ids)
  -> PLE host gather into fixed buffers      (Phase 2)
  -> prepare_inputs / recurrent metadata     (host)
  -> GPU0 stage graph replay                 (fixed-address in/out)
  -> fixed-address GPU0->GPU1 activation copy
  -> GPU1 stage graph replay
  -> fixed-address GPU1->GPU2 activation copy
  -> GPU2 stage graph replay
  -> advance_recurrent_states()              (host, outside graphs)
  -> sampler                                 (outside graphs)
```

- Partition the layer-split `fwd_modules` into contiguous runs by `.device`;
  capture one graph per run at warmup, bucketed by batch size.
- **Batch size 1 first.** The parity server serializes requests through one
  dispatch worker, so real decode batches are 1 until batching is added.
  Bucket {2,4} only after B1 pays off.
- Requirements after Phase 2 (all satisfiable):
  - each stage's graphs contain zero host work (gather hoisted; sampler and
    recurrent-state bookkeeping stay OUTSIDE)
  - per-slot state tensors of fixed shape/address: GDN recurrent state, PLE
    conv_state/id_state, QSA raw/pooled planes — all in-place updated
  - page table as a fixed-address tensor updated in place
  - input ids / positions / inter-stage activations in fixed-address staging
    buffers, batch padded to the bucket
- Prefill stays eager. This replicates `FULL_DECODE_ONLY` on a PP topology.
- MoE routing: gate + top-k computed on device; no host reads of tensor
  values inside a captured stage (verify with the profiler from Phase 1).
- No `torch.compile` shortcut on the primary path: the vLLM reference ran
  compilation mode:0 with manual full decode graphs — Inductor is not what
  produced the win, and silent graph-breaks on ext kernels waste profiling
  time. Manual per-device graphs are the first implementation.

Acceptance: tok/s at 248k ctx >= 70 (target 80); output bit-identical eager
vs graph on fixed seed; no warmup-dependent drift over 10k steps.

## Phase 4 (optional) — QSA K/V host offload via UVA

Only if context beyond what VRAM allows is wanted (author reached 3M ctx).
NOT a speed change.

- New QSA CacheLayer variant: main K/V rows in pinned host memory behind a
  UVA view; indexer selection stays on GPU; per-step read is bounded by
  `indexer_budget` (2048 sel x 2 heads x 256 dim x 2 x 2B x 12 layers =
  48 MiB/token, ~3.9 GB/s at 80 tok/s — a fraction of PCIe 4.0 x16).
- Kernel work in `exllamav3_ext` (C++/CUDA): attention/gather kernels must
  accept host pointers. Heaviest item in this plan.
- `generator/cpu_cache.py` (existing pinned second tier) does NOT fit: it
  copies whole pages; QSA needs zero-copy gather of selected rows.
- Budget guard like `VLLM_QSA_KV_OFFLOAD_MAX_GIB` to refuse start rather
  than OOM the host.

Acceptance: decode tok/s at 1M ctx within 10% of tok/s at 100k ctx.

## Phase 5 — Verification and quality gate

- tok/s flat across 3.6k/121k/248k (and 1M+ if Phase 4 done)
- KLD vs BF16 with the author's measurement script, same trace; expect
  ~0.0067 nats (the quant's number, independent of these patches)
- A/B every phase: hoist on/off, graph on/off, `EXL3_NGRAM_PREFETCH` 0/1
- 4-request aggregate throughput via a **direct Generator batch harness**
  (4 jobs in one `iterate_gen` batch), NOT through the parity server: its
  `_dispatch_worker` serializes requests, so 4 HTTP clients are 4 serial
  generations, not an EXL3 batch. Server-level batching is a later, separate
  phase; never compare the serial path against vLLM's ~155 tok/s batch figure
- Optional single experiment: `MULTI_GPU=tensor` TP3 vs pipeline on the final
  build (topology A/B, never mixed with phase A/Bs)

---

## Measurements (fill in as phases complete)

| Metric | Baseline (P1) | After P2 | After P3 | vLLM ref |
|---|---|---|---|---|
| tok/s single @ 3.6k | | | | 44-80 |
| tok/s single @ 248k | | | | 80 |
| tok/s @ 4 reqs | | | | ~155 |
| GPU-busy ms/step | | | | 12.6 |
| Host gap ms/step | | | | ~0 (FULL graph) |
| Prefill t/s @ 248k | | | | 3,701 |

## Complementary track (kernel side, independent)

The fork's existing "Ampere GPU inference bottleneck analysis" (commit
efa77b9) ranks 12 engine bottlenecks — unoptimized GEMV (20-40% decode),
shared-memory cap, kernel tuning. That track lowers the kernel floor; this
plan removes the gaps above the floor. Doing both is the path to beating 80,
not just matching it.

## Risks

- Phase 3 is where the effort is: capture with hybrid state models is
  fiddly, and the per-device stage boundaries add their own risk — the
  fixed-address inter-stage activation copies must cover every tensor shape
  a stage emits (MoE routing included); verify with the Phase 1 profiler
  that no captured stage reads tensor values on host
- The parity server (`qwen38_flash_gguf_parity/`) must keep working; all
  engine changes stay behind env switches defaulting to old behaviour until
  Phase 3 acceptance passes

## Changelog

- v2 rev 3: per review round 2 — runtime pins now record the git SHA +
  clean/dirty of the imported exllamav3 and `EXPECTED_EXL3_ROOT` aborts on
  wrong-checkout imports; `MULTI_GPU` strictly validated (invalid values and
  tensor-on-unsupported-arch are startup errors, never silent fallbacks);
  smoke test pins SERVED_MODEL_NAME/CHAT_TEMPLATE_FILE/MAX_BATCH_SIZE=1;
  Phase 3 rewritten as per-device pipeline-stage CUDA graphs (vLLM reference
  is PP3/TP1 with per-rank graphs; no torch.compile on the primary path);
  Phase 5 batch test moved to a direct Generator harness; `mdl.load`
  max_batch_size aligned with the Generator's; machine-readable
  `experiment_config=` JSON line at startup.
- v2 rev 2: rebased fork onto upstream `6b84a21`; server multi-GPU mode
  now explicit (`MULTI_GPU=pipeline|tensor`, pipeline default) with runtime
  pin logging; Phase 0 quant corrected to `turboderp/...@4.05bpw_h6_ng6`
  (100.1 GiB, not 37G); acceptance gate hardened; Phase 2 rewritten around
  the existing prefetch machinery; Phase 3 scoped to batch 1 first.
- v1: initial plan (commit 54a8683 on the retired branch).
