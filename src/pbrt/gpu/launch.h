// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_LAUNCH_H
#define PBRT_GPU_LAUNCH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/log.h>

#include <map>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace pbrt {

std::pair<cudaEvent_t, cudaEvent_t> GetProfilerEvents(const char *description);

template <typename F>
inline int GetBlockSize(const char *description, F kernel) {
    // Note: this isn't re-entrant, but that's fine for our purposes...
    static std::map<std::type_index, int> kernelBlockSizes;

    std::type_index index = std::type_index(typeid(F));

    auto iter = kernelBlockSizes.find(index);
    if (iter != kernelBlockSizes.end())
        return iter->second;

    int minGridSize, blockSize;
    CUDA_CHECK(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    kernelBlockSizes[index] = blockSize;
    LOG_VERBOSE("[%s]: block size %d", description, blockSize);

    return blockSize;
}

#ifdef __CUDACC__
template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(tid);
}
#endif  // __CUDACC__

// GPU Launch Function Declarations
template <typename F>
void GPUParallelFor(const char *description, int nItems, F func);

template <typename F>
void GPUDo(const char *description, F func) {
    GPUParallelFor(description, 1, [=] PBRT_GPU(int) mutable { func(); });
}

void GPUWait();

#ifdef __CUDACC__
template <typename F>
void GPUParallelFor(const char *description, int nItems, F func) {
#ifdef NVTX
    nvtxRangePush(description);
#endif
    auto kernel = &Kernel<F>;

    int blockSize = GetBlockSize(description, kernel);
    std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(description);

#ifndef NDEBUG
    LOG_VERBOSE("Launching %s", description);
#endif
    cudaEventRecord(events.first);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(func, nItems);
    cudaEventRecord(events.second);

#ifndef NDEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_VERBOSE("Post-sync %s", description);
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}
#endif  // __CUDACC__

void ReportKernelStats();

}  // namespace pbrt

#endif  // PBRT_GPU_LAUNCH_H
