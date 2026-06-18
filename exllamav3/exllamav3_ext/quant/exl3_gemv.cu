#include <cuda_fp16.h>
#include "exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "hadamard.cuh"
//#include <set>

#include <ATen/Tensor.h>

#define K_SPLIT 1

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*bits), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Zero-initialized by dispatch when split_k > 1
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

//std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
)
{
//    had_r_128(A, A_had.value(), suh, c10::nullopt, 1.0);

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, 1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16;

    TORCH_CHECK(size_m <= 8, "size_m must be <= 8");

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int bits = B.size(2) / 16;
    int split_k = (size_k >= 2 * TILESIZE_K) ? 2 : 1;
    int tilesize_n = 32;

    int device;
    cudaGetDevice(&device);
    int cc = DevCtx::instance().get_cc(device);
    if (cc == CC_AMPERE) {
        if (size_n >= 2048) tilesize_n = 64;
    }

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    dim3 threads(32, 1, TILESIZE_K / 16);

    // split_k > 1 uses atomicAdd, so C must start at zero
    if (split_k > 1) C.zero_();

    int smem_max = DevCtx::instance().get_smem_max(device);

    #define DISPATCH_GEMV_KERNEL(BITS, C_FP32, CB, SPLIT_K, TILESIZE_N) \
        { \
            int blocks_n = TILESIZE_N / 16; \
            int blocks_k = 512 / 16; \
            int bsh0 = 2; \
            int bsh1 = 256 * BITS / 8 * blocks_n * blocks_k; \
            int b_sh_size = (bsh0 * bsh1 + 15) & ~15; \
            int c_sh_size = std::max(blocks_k / 2, 1) * blocks_n * 128 * sizeof(float); \
            int smem_size = b_sh_size + c_sh_size; \
            auto kernel = exl3_gemv_kernel<BITS, C_FP32, CB, SPLIT_K, TILESIZE_N>; \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max); \
            kernel<<<blocks, threads, smem_size, stream>>>(A_ptr, B_ptr, C_ptr, size_m, size_k, size_n); \
        }

    #define DISPATCH_GEMV_TILESIZE(BITS, C_FP32, CB, SPLIT_K) \
        if (tilesize_n == 128) { \
            dim3 blocks(1, size_n / 128, SPLIT_K); \
            DISPATCH_GEMV_KERNEL(BITS, C_FP32, CB, SPLIT_K, 128) \
        } else if (tilesize_n == 64) { \
            dim3 blocks(1, size_n / 64, SPLIT_K); \
            DISPATCH_GEMV_KERNEL(BITS, C_FP32, CB, SPLIT_K, 64) \
        } else { \
            dim3 blocks(1, size_n / 32, SPLIT_K); \
            DISPATCH_GEMV_KERNEL(BITS, C_FP32, CB, SPLIT_K, 32) \
        }

    #define DISPATCH_GEMV_SPLITK(BITS, C_FP32, CB) \
        if (split_k == 2) { \
            DISPATCH_GEMV_TILESIZE(BITS, C_FP32, CB, 2) \
        } else { \
            DISPATCH_GEMV_TILESIZE(BITS, C_FP32, CB, 1) \
        }

    #define DISPATCH_GEMV_CB(BITS, C_FP32) \
        if (cb == 2) { \
            DISPATCH_GEMV_SPLITK(BITS, C_FP32, 2) \
        } else if (cb == 1) { \
            DISPATCH_GEMV_SPLITK(BITS, C_FP32, 1) \
        } else { \
            DISPATCH_GEMV_SPLITK(BITS, C_FP32, 0) \
        }

    #define DISPATCH_GEMV_FP32(BITS) \
        if (c_fp32) { \
            DISPATCH_GEMV_CB(BITS, true) \
        } else { \
            DISPATCH_GEMV_CB(BITS, false) \
        }

    if (bits == 1) DISPATCH_GEMV_FP32(1)
    else if (bits == 2) DISPATCH_GEMV_FP32(2)
    else if (bits == 3) DISPATCH_GEMV_FP32(3)
    else if (bits == 4) DISPATCH_GEMV_FP32(4)
    else if (bits == 5) DISPATCH_GEMV_FP32(5)
    else if (bits == 6) DISPATCH_GEMV_FP32(6)
    else if (bits == 7) DISPATCH_GEMV_FP32(7)
    else if (bits == 8) DISPATCH_GEMV_FP32(8)
    else TORCH_CHECK(false, "Unsupported bits for GEMV");

    #undef DISPATCH_GEMV_KERNEL
    #undef DISPATCH_GEMV_TILESIZE
    #undef DISPATCH_GEMV_SPLITK
    #undef DISPATCH_GEMV_CB
    #undef DISPATCH_GEMV_FP32

    cuda_check(cudaPeekAtLastError());
}