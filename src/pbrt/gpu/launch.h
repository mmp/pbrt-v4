// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_LAUNCH_H
#define PBRT_GPU_LAUNCH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/log.h>

#include <typeindex>
#include <typeinfo>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace pbrt {

struct GPUKernelStats {
    GPUKernelStats() = default;
    GPUKernelStats(const char *description) : description(description) {
        launchEvents.reserve(256);
    }

    std::string description;
    int blockSize = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> launchEvents;
};

GPUKernelStats &GetGPUKernelStats(std::type_index typeIndex, const char *description);

template <typename T>
inline GPUKernelStats &GetGPUKernelStats(const char *description) {
    return GetGPUKernelStats(std::type_index(typeid(T)), description);
}

template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(tid);
}

template <typename F>
void GPUParallelFor(const char *description, int nItems, F func) {
#ifdef NVTX
    nvtxRangePush(description);
#endif
    auto kernel = &Kernel<F>;

    GPUKernelStats &kernelStats = GetGPUKernelStats<F>(description);
    if (kernelStats.blockSize == 0) {
        int minGridSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &kernelStats.blockSize, kernel, 0, 0));

        LOG_VERBOSE("[%s]: block size %d", description, kernelStats.blockSize);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifndef NDEBUG
    LOG_VERBOSE("Launching %s", description);
#endif
    cudaEventRecord(start);
    int gridSize = (nItems + kernelStats.blockSize - 1) / kernelStats.blockSize;
    kernel<<<gridSize, kernelStats.blockSize>>>(func, nItems);
    cudaEventRecord(stop);

    kernelStats.launchEvents.push_back(std::make_pair(start, stop));

#ifndef NDEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_VERBOSE("Post-sync %s", description);
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

template <typename F>
void GPUDo(const char *description, F func) {
    GPUParallelFor(description, 1, [=] PBRT_GPU(int) { func(); });
}

void ReportKernelStats();

}  // namespace pbrt

#endif  // PBRT_GPU_LAUNCH_H
