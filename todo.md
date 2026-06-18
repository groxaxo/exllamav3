# Ampere Kernel Optimizations Tasks

## Done (2026-04-16) — Critical Bug Fixes

### Bug 1: GEMM B shared memory swizzle mismatch (exl3_gemm_inner.cuh)
- B tiles were loaded into shared memory **linearly** via `cp_async_stream`, but read with XOR swizzle.
- This caused `misaligned address` GPU assert on every inference, manifesting as `CUBLAS_STATUS_EXECUTION_FAILED`.
- **Fix**: Removed the B read swizzle block (lines 305-311). A matrix swizzle was correctly paired and left intact.

### Bug 2: GEMV B shared memory swizzle mismatch (exl3_gemv_kernel.cuh)
- Write path swizzled each uint4 independently within a row (`x ^ (y & 7)`), scrambling physical layout.
- Read path only adjusted the block start pointer, then `dq_dispatch` read sequentially — getting wrong data for 28/32 K-blocks.
- **Fix**: Removed B swizzle from both write and read paths. Kept `cp_async_stream` L2 evict-first hint for Ampere.

### Bug 3: dq8_aligned_3bits / dq8_aligned_5bits off-by-one (exl3_dq.cuh)
- Custom routines used `8 * bits` instead of `7 * bits` in bit-range calculations, shifting every extracted word by one position.
- **Fix**: Reverted dq_dispatch to use proven upstream `dq8<3, cb, 4>` for 3-bit and `dq4+dq4` for 5-bit.

### Also fixed
- Removed duplicate `C.zero_()` call in exl3_gemv.cu (was at lines 77 and 93, kept single call after CC detection).
- Fixed `#pragma unroll` placement, removed unused `bit_end` and `frag_col` (prior session).

## Benchmark Results (2026-04-16) — Qwen3.5-27B-abliterated-exl3-3bpw

| Config | Cache | Decode (t/s) | Prefill (t/s) |
|--------|-------|-------------|---------------|
| 1×RTX 3090, 4K fp16 | 4,096 | 21.9 | 270 |
| 2×RTX 3090 layer-split, 32K fp16 | 32,768 | 22.6 | 273 |
| 2×RTX 3090 layer-split, 131K fp16 | 131,072 | **25.0** | **294** |
| 2×RTX 3090 layer-split, 131K fp16, 500 tok | 131,072 | **24.6** | **291** |
| 2×RTX 3090 layer-split, 131K 8-bit KV | 131,072 | 21.5 | 254 |
| 2×RTX 3090 layer-split, 200K 4-bit KV | 199,936 | 21.6 | 265 |

## Notes / Next steps

- The custom `dq8_aligned_3bits`/`dq8_aligned_5bits` functions remain in the codebase but are unused (dead code). They could be removed or fixed and re-enabled.
- The A matrix swizzle in exl3_gemm_inner.cuh is correctly paired (write+read) and provides bank-conflict reduction. B swizzle could be re-added with matching write/read if beneficial.
- 200K context works on 2×RTX 3090 with 4-bit KV cache (27B model at 3bpw).
- Consider TP mode benchmarks once NCCL backend issues are resolved.
