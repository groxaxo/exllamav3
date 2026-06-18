#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "exl3_devctx.cuh"
#include "../util.h"
#include "../util.cuh"

// B2: Runtime-configurable shared memory cap, initialized per-device in prepare_ctx
int g_smem_max = 0;

DevCtx& DevCtx::instance()
{
    static DevCtx ctx;
    return ctx;
}

int DevCtx::get_num_sms(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!num_sms[device])
        cuda_check(cudaDeviceGetAttribute(&num_sms[device], cudaDevAttrMultiProcessorCount, device));
    return num_sms[device];
}

int DevCtx::get_cc(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!cc[device])
    {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, device));
        if (prop.major >= 10) cc[device] = CC_BLACKWELL;
        else if (prop.major >= 9) cc[device] = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc[device] = CC_ADA;
        else if (prop.major >= 8) cc[device] = CC_AMPERE;
        else cc[device] = CC_OLD;
    }
    return cc[device];
}

// B2: Query per-device max optin shared memory
int DevCtx::get_smem_max(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!smem_max[device])
    {
        cuda_check(cudaDeviceGetAttribute(&smem_max[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        // Ensure we don't exceed what's actually available
        if (smem_max[device] <= 0) smem_max[device] = 90 * 1024;  // fallback
    }
    return smem_max[device];
}

void* DevCtx::get_ws(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!ws[device])
    {
        cudaSetDevice(device);
        cudaMalloc(&ws[device], WORKSPACE_SIZE);
    }
    return ws[device];
}

int* DevCtx::get_locks(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks[device])
    {
        cudaSetDevice(device);
        size_t size = (MAX_TILES_C + MAX_BARRIERS * 2) * sizeof(int);
        cudaMalloc(&locks[device], size);
        cudaMemset(locks[device], 0, size);
    }
    return (int*) locks[device];
}

int g_get_cc(int device)
{
    return DevCtx::instance().get_cc(device);
}

int g_get_num_sms(int device)
{
    return DevCtx::instance().get_num_sms(device);
}

void prepare_ctx(int device)
{
    DevCtx::instance().get_num_sms(device);
    DevCtx::instance().get_cc(device);
    DevCtx::instance().get_locks(device);

    // B2: Initialize the global SMEM max for the current device
    int device_smem = DevCtx::instance().get_smem_max(device);
    if (!g_smem_max || device_smem < g_smem_max)
        g_smem_max = device_smem;
}

