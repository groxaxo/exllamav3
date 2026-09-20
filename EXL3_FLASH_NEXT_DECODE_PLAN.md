# EXL3 Flash-Next Decode Optimization Plan

Date: 2026-09-21
Target: match the 80 tok/s single-stream decode of the vLLM AutoRound build,
using the EXL3 quant (KLD 0.0067 vs BF16, 2.2x better than the AutoRound mixed)
on the same 3x RTX 3090 hardware.

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

## State of the local ExLlamaV3 tree (verified 2026-09-21)

Upstream checkout: `/home/op/exllamav3` (v1.5.0+cu128, torch 2.11.0)
Env: `/home/op/miniforge3/envs/exl3`
Fork (server parity layer only, no engine patches):
`/home/op/src/exllamav3-fork` branch `qwen38-gguf-parity-server`

| Piece | State |
|---|---|
| flash-next arch support | present: `modules/ple.py`, `modules/qsa_indexer.py`, `modules/gated_delta_net.py`, `modules/hyperconnections.py`, arch_specific |
| n-gram table | streaming first-class: `DiskTensorHandles`, trellis codec, pinned staging (`modules/ngram_embedding.py`). Gather runs INSIDE `forward()` (line 511) with `pin.event.synchronize()` (line 443). A/B switch exists: `EXL3_NGRAM_PREFETCH` |
| QSA cache | `cache/qsa.py`: indexer planes (raw + pooled keys) alongside K/V, all VRAM-resident. No host offload |
| CUDA graphs | none anywhere in the Python tree (`graph.cu` in ext has no callers) |
| Decode seam | `generator/generator.py:1027` — `iterate_gen()` calls `self.model.forward(...)` |
| Sharding | TP=3 works for this model (upstream commit "MoE: Fix gather for TP-split layers") |

This is exactly the pre-decode-patch state of the vLLM build: gather in
forward, no graphs. The 45 tok/s ceiling class.

---

## Phase 0 — Restore the quant (prerequisite)

The EXL3 quants were deleted in the 2026-09-21 disk cleanup (287G freed).
Re-download `Qwen3.8-Flash-Next-exl3-4.00bpw_hq_h6_ng6` (~37G, ngram in
6-bit trellis = 36.4G inside). It is the quant from the author's KLD table.
Disk has room (564G free at time of writing).

Do NOT restore `exl3-unsloth-shaped-h8-mtp5` (98G) unless MTP experiments
resume; the author measured MTP slower than no drafting in real use.

Acceptance: model loads in the exl3 env, one generation completes.

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

## Phase 2 — Port the gather hoist (~150 lines, days)

Port of decode-01 + decode-02 into exllamav3:

1. `exllamav3/modules/ngram_embedding.py`:
   - Add a fixed device output buffer `torch.zeros(max_rows, heads, dim)`,
     allocated once at init. Zeros, not empty: a forward running before the
     first gather must read zeros (this matters for graph capture warmup).
   - Add `host_gather(ids)`: the disk gather + H2D upload, running OUTSIDE the
     forward, writing into that buffer. The existing pinned staging and
     thread pool can be reused; the change is who calls it and when.
   - Reduce `forward()` (line 511) to returning a view
     `self._prefetch_buffer[:n]` of the fixed buffer. Address must never
     change: graph replays re-read the same pointer.
2. `exllamav3/generator/generator.py` (`iterate_gen`, before the
   `self.model.forward` call at line 1027): invoke `host_gather()` with the
   final row ids — the equivalent of vLLM's `prepare_inputs` hook (patch-02).
   Batch/padding must be resolved before the call so the buffer covers every
   row the forward will read.
3. Dummy-input path: during capture/profiling, gather zeros so the PLE
   projection has consumable data (patch-02's `_ple_dummy_input_ids`).

Keep an A/B switch mirroring `VLLM_PLE_HOST_GATHER=0` semantics: env var to
restore the old in-forward gather on the same build.

Acceptance: generation output identical (same seed, same prompt) with hoist on
and off. Throughput measured; expected ~no change (see F2 note above).

## Phase 3 — Decode CUDA graphs (the 45->80 multiplier, 1-2 weeks)

New module `exllamav3/generator/cudagraph.py`:

- Capture the decode-step model forward at warmup, bucketed by batch size
  {1, 2, 4} (the real forward batch sizes under the parity server; same
  reasoning as the author's `cudagraph_capture_sizes`).
- Requirements after Phase 2 (all satisfiable):
  - forward contains zero host work (gather hoisted; sampler stays OUTSIDE
    the graph)
  - per-slot state tensors of fixed shape/address: GDN recurrent state, PLE
    conv_state/id_state, QSA raw/pooled planes — all in-place updated
  - page table as a fixed-address tensor updated in place
  - input ids / positions in fixed-address staging buffers, batch padded to
    the bucket
- Prefill stays eager. This replicates `FULL_DECODE_ONLY` exactly.
- MoE routing: gate + top-k computed on device; no host reads of tensor
  values inside the captured region (verify with the profiler from Phase 1).
- Shortcut worth trying first: `torch.compile(mode="reduce-overhead")` on the
  model forward after Phase 2. Custom ext kernels capture fine as long as no
  host-side branching reads tensor values. If it cooperates, it replaces most
  of the custom capture code; if it graph-breaks, fall back to manual
  `torch.cuda.CUDAGraph`.

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
- 4-request aggregate throughput through the parity server (batched
  `iterate_gen`); compare against vLLM's ~155 tok/s

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

- Phase 3 is where the effort is; bucket capture with hybrid state models is
  fiddly (the states are the easy part — MoE expert routing inside the graph
  is the part to verify carefully)
- torch.compile shortcut may silently graph-break on ext kernels: verify with
  profiler, not with absence of errors
- The parity server (`qwen38_flash_gguf_parity/`) must keep working; all
  engine changes stay behind env switches defaulting to old behaviour until
  Phase 3 acceptance passes
