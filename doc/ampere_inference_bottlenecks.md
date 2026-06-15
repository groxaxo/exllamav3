# Ampere GPU Inference Bottleneck Analysis for ExLlamaV3

## Context

This repo is a custom ExLlamaV3 fork targeting Ampere GPUs (RTX 3090/3060, SM 8.0/8.6). The upstream documentation explicitly states: **"Quantization kernel is tuned for Ada GPUs at the moment, and especially Ampere performance is likely to improve with more optimization"** (`doc/convert.md:98`). The quantization tables show the RTX 3090 is 2-3x slower than the RTX 4090 for conversion, and inference follows a similar pattern. This analysis identifies the specific bottlenecks and provides actionable fixes.

The primary inference hot path per transformer layer is:
1. Hadamard transform + scale
2. **EXL3 quantized GEMM** (dominant cost for all linear layers)
3. **EXL3 GEMV** (dominant cost for single-token decode, batch=1)
4. Flash Attention (`flash_attn >= 2.7.4`)
5. cuBLAS hgemm (unquantized FP16 ops)

---

## CRITICAL SEVERITY

### B1. GEMV Kernel Unoptimized - Single Largest Decode Bottleneck
**Impact: 20-40% decode latency improvement**

**Files:**
- `exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh:1-6` - hardcoded tile sizes
- `exllamav3/exllamav3_ext/quant/exl3_gemv.cu:18,72-73,90,106` - hardcoded K_SPLIT=1, only 4-bit/cb0

**Problem:** The GEMV kernel (hot path for batch_size=1 autoregressive decode) has multiple deficiencies:
- Only supports 4-bit quantization with codebook 0; other bit widths fall back to the slower GEMM kernel
- No K-split parallelism (`K_SPLIT=1`): for a typical n=4096 layer, only 128 blocks launch (4096/32), giving ~1.5 blocks/SM on the 3090 (82 SMs) - severely underutilizing the GPU
- Fixed tile sizes `M=8, K=512, N=32` not tuned for Ampere
- Only 2-stage async pipeline
- No shared memory bank conflict mitigation (unlike the GEMM kernel)

**Fix:**
1. Make K_SPLIT configurable (minimum K_SPLIT=2 for k >= 2*TILESIZE_K)
2. Template over all bit widths (1-8) and codebook types (0-2)
3. Increase TILESIZE_N to 64 or 128 to improve occupancy
4. Add Ampere-specific shared memory swizzle for B loads

**Verify:** Benchmark `exl3_gemv()` with typical LLM shapes (k=4096, n=4096/14336) before and after.

---

### B2. GEMM Shared Memory Cap at 90KB Penalizes SM 8.0 (A100)
**Impact: 10-20% GEMM throughput on A100**

**Files:**
- `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh:7` - `#define SMEM_MAX (90 * 1024)`
- `exllamav3/exllamav3_ext/quant/exl3_gemm.cu:131,154` - `cudaFuncSetAttribute(..., SMEM_MAX)`

**Problem:** SMEM_MAX is hardcoded to 90KB (SM 8.6 max). But SM 8.0 (A100) supports up to 163KB. The 3090 supports ~100KB. Both are artificially limited. This prevents deeper async copy pipelines (more SH_STAGES) and larger tile sizes.

**Fix:**
1. Make SMEM_MAX runtime-configurable via `cudaDevAttrMaxSharedMemoryPerBlockOptin`
2. Add SM 8.0-specific shapes with deeper pipelines (e.g., SH_STAGES=6+ using ~130KB)
3. Re-run `science/qgemm_pretune.py` on A100 with new shapes

**Verify:** `science/qgemm_benchmark.py` on A100, comparing median latencies across (k, n) shapes.

---

### B3. Kernel Tuning Samples May Be Sub-Optimal for Ampere
**Impact: 5-15% latency reduction**

**Files:**
- `exllamav3/exllamav3_ext/quant/exl3_kernel_map_samples.cuh` - 4368 Ampere samples (cc=2)
- `exllamav3/exllamav3_ext/quant/exl3_kernel_map.cu:37-46` - Ampere heuristic fallback

**Problem:** The tuning samples store pre-benchmarked `(shape_idx, num_sms)` pairs. The Ampere samples were likely tuned on a single GPU variant. The optimal `num_sms` depends on SM count (82 on 3090, 28 on 3060, 108 on A100) and memory bandwidth, which varies significantly across Ampere chips. The heuristic fallback for Ampere (4 branches) is also simpler than Ada's (5 branches) and never selects Shape 1.

**Fix:**
1. Re-run `science/qgemm_pretune.py` on each target Ampere GPU (3090, 3060, A100)
2. Consider SM count as an additional dimension in the sample table
3. Test whether Shape 1 (M=16, K=16, N=128, SH_STAGES=6, FRAG_STAGES=5) benefits small-N cases on Ampere

**Verify:** Compare generated samples vs current hardcoded ones. Measure per-shape latency deltas.

---

## MEDIUM SEVERITY

### B4. MoE Kernel Hardcoded to 12 SMs Per Expert
**Impact: 5-15% MoE layer improvement on consumer Ampere**

**Files:**
- `exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh:12` - `#define MOE_SMS_PER_EXPERT 12`
- `exllamav3/exllamav3_ext/quant/exl3_moe.cu:17,203-204`

**Problem:** Fixed 12 SMs per expert group. On RTX 3060 (28 SMs), only 2 concurrent expert groups with 4 SMs wasted (14% idle). For 8-expert models, the 3060 processes experts in 4 batches vs 1 on the 4090.

**Fix:** Make `MOE_SMS_PER_EXPERT` runtime-configurable. Test with 7 SMs/expert on Ampere (11 groups on 3090, 4 on 3060).

---

### B5. Codebook Decode Compiler Workaround Missing for SM 8.0
**Impact: 2-5% GEMM improvement for codebook 1**

**Files:**
- `exllamav3/exllamav3_ext/quant/codebook.cuh:3-23` - `mul_const_u32<w>()` only targets `__CUDA_ARCH__ == 860`

**Problem:** The forced MAD workaround (replacing IMUL) is only active on SM 8.6. SM 8.0 (A100) doesn't get this optimization and may have the same suboptimal compiler behavior.

**Fix:** Profile with `ncu` on SM 8.0. If beneficial, extend `#if` to `__CUDA_ARCH__ <= 860`.

---

### B6. Split-K Global Barriers Under L2 Cache Pressure
**Impact: 3-10% on RTX 3060 (3MB L2)**

**Files:**
- `exllamav3/exllamav3_ext/ptx.cuh:103-138` - `barrier_acquire`/`barrier_release`
- `exllamav3/exllamav3_ext/quant/exl3_devctx.cuh:7-8` - 4MB lock buffer

**Problem:** Lock buffer is 4MB regardless of actual need. On the RTX 3060 (3MB L2) and 3090 (6MB L2), this causes heavy L2 pressure. The tight spin-loop in `barrier_acquire` generates repeated coherent loads that miss when L2 is thrashed.

**Fix:**
1. Reduce lock buffer to actual need (`MAX(size_n/128, 1)` per call)
2. Add `__nanosleep()` backoff to spin loop (similar to `group_barrier` at ptx.cuh:343)
3. Use `cp_async_stream` for B loads (L2 eviction hint) - the function exists at ptx.cuh:172 but is unused

---

### B7. Shared Memory Bank Conflicts on B-Matrix Loads
**Impact: 3-8% GEMM improvement (needs profiling to confirm)**

**Files:**
- `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh:240-248` - B loads have NO swizzle
- `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh:318` - TODO comment about reduction bank conflicts

**Problem:** A-matrix loads have Ampere-specific swizzle (lines 217-235, 266-286), but B-matrix loads use straight `cp_async` with no bank conflict mitigation. The B reads go through `dq_dispatch()` which reads uint32s at potentially conflicting addresses.

**Fix:** Profile with `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`. If significant, apply swizzle to B shared memory writes and adjust `dq_dispatch` reads.

---

### B8. Dequantization Overhead for Odd Bit-Widths (3, 5, 7-bit)
**Impact: 3-8% GEMM improvement for 3/5-bit models**

**Files:**
- `exllamav3/exllamav3_ext/quant/exl3_dq.cuh:96-161` - generic `dq8()` fallback
- `exllamav3/exllamav3_ext/quant/exl3_dq.cuh:254-292` - `dq_dispatch()` routing

**Problem:** Even bit-widths (1, 2, 4) have specialized aligned extraction routines. Odd widths (3, 5, 7) fall back to generic `dq8` or `dq4` with up to 8 separate funnel shifts. These share the integer ALU pipeline with FP32 on Ampere, and can't be fully hidden by double-buffered fragments.

**Fix:** Write specialized `dq8_aligned_3bits`, `dq8_aligned_5bits` routines that exploit known alignment for shift reuse.

---

## LOW SEVERITY

### B9. L2 Eviction Hints Defined But Unused for Weight Loads
**Impact: 1-5%**

`cp_async_stream()` (ptx.cuh:172) with `L2::evict_first` exists but the B-matrix loads at exl3_gemm_inner.cuh:247 use plain `cp_async`. Switching B loads to the streaming variant would reduce L2 pollution on Ampere's smaller caches.

### B10. Flash Attention 2.x on Ampere
**Impact: 0-5% total inference**

External dependency with limited control. FA2 achieves lower utilization on Ampere due to smaller shared memory and no hardware async barriers. Profile the attention vs GEMM time split to quantify.

### B11. cuBLAS TF32 Not Explicitly Managed
**Impact: 0-2x for hgemm FP32 path if TF32 inadvertently disabled**

hgemm.cu uses `CUBLAS_COMPUTE_32F` without explicit `cublasSetMathMode`. Default is TF32 on Ampere, but user env can override silently.

### B12. Potential B-Matrix Index Bug (Dormant)
**Impact: Correctness-critical for future shapes**

At `exl3_gemm_inner.cuh:115`: `load_b_gl[i] = k * ... * k + n` has `k` squared. Currently dormant because `load_b_iters=1` for all shapes (k=0), but will cause silent corruption if new shapes trigger `load_b_iters > 1`. **Must fix before adding new shapes (B2).**

---

## Implementation Sequence

| Phase | Bottleneck | Expected Impact | Effort |
|-------|-----------|-----------------|--------|
| 1     | B1: GEMV optimization | 20-40% decode | High |
| 1     | B3: Re-tune samples on target GPUs | 5-15% | Medium |
| 2     | B2: SMEM_MAX for SM 8.0 + new shapes | 10-20% on A100 | Medium |
| 2     | B4: Dynamic MOE_SMS_PER_EXPERT | 5-15% MoE | Low |
| 2     | B12: Fix B-matrix index formula | Correctness | Low |
| 3     | B5: Codebook MAD for SM 8.0 | 2-5% | Low |
| 3     | B7: B-matrix bank conflicts | 3-8% | Medium |
| 3     | B6: L2-aware barriers | 3-10% on 3060 | Medium |
| 3     | B9: L2 eviction hints | 1-5% | Low |
| 3     | B8: Odd bit-width dequant | 3-8% for 3/5-bit | Medium |
| 4     | B10: FA2 profiling | 0-5% | Low |
| 4     | B11: TF32 management | 0-2x edge case | Low |

## Verification Strategy

1. **Baseline measurements:** Run `science/qgemm_benchmark.py` on target Ampere GPU(s) for all (K, k, n) combinations at each bit width
2. **Per-change measurement:** After each fix, re-run benchmarks and compare latency distributions
3. **End-to-end validation:** Run the inference server scripts (`server_3090.py`, `server_mixed.py`) and measure tokens/sec for prompt processing and generation
4. **Correctness:** Compare model output (logits) before and after kernel changes to ensure numerical equivalence

## Critical Files

- `exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh` - GEMV kernel (B1)
- `exllamav3/exllamav3_ext/quant/exl3_gemv.cu` - GEMV dispatch (B1)
- `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh` - GEMM inner loop (B2, B7, B9, B12)
- `exllamav3/exllamav3_ext/quant/exl3_kernel_map.cu` - Kernel selection (B3)
- `exllamav3/exllamav3_ext/quant/exl3_kernel_map.cuh` - Shape definitions (B2, B3)
- `exllamav3/exllamav3_ext/quant/exl3_kernel_map_samples.cuh` - Tuning data (B3)
- `exllamav3/exllamav3_ext/quant/exl3_devctx.cuh` - Device context/SMEM config (B2)
- `exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh` - MoE kernel (B4)
- `exllamav3/exllamav3_ext/quant/codebook.cuh` - Codebook decode (B5)
- `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` - Dequantization (B8)
- `exllamav3/exllamav3_ext/ptx.cuh` - PTX intrinsics (B6, B9)
