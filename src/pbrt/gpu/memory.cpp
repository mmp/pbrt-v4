// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/memory.h>

#include <pbrt/gpu/util.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace pbrt {

void *CUDAMemoryResource::do_allocate(size_t size, size_t alignment) {
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    CHECK_EQ(0, intptr_t(ptr) % alignment);
    return ptr;
}

void CUDAMemoryResource::do_deallocate(void *p, size_t bytes, size_t alignment) {
    CUDA_CHECK(cudaFree(p));
}

void *CUDATrackedMemoryResource::do_allocate(size_t size, size_t alignment) {
    if (size == 0)
        return nullptr;

    // GPU cache line alignment to avoid false sharing...
    alignment = std::max<size_t>(128, alignment);

    if (bypassSlab(size))
        return cudaAllocate(size, alignment);

    CHECK_LT(ThreadIndex, maxThreads);
    Slab &slab = threadSlabs[ThreadIndex];
    if ((slab.offset % alignment) != 0)
        slab.offset += alignment - (slab.offset % alignment);

    if (slab.offset + size > slabSize) {
        slab.ptr = (uint8_t *)cudaAllocate(slabSize, 128);
        slab.offset = 0;
    }

    uint8_t *ptr = slab.ptr + slab.offset;
    slab.offset += size;
    return ptr;
}

void *CUDATrackedMemoryResource::cudaAllocate(size_t size, size_t alignment) {
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    DCHECK_EQ(0, intptr_t(ptr) % alignment);

    std::lock_guard<std::mutex> lock(mutex);
    allocations[ptr] = size;
    bytesAllocated += size;

    return ptr;
}

void CUDATrackedMemoryResource::do_deallocate(void *p, size_t size, size_t alignment) {
    if (!p)
        return;

    if (bypassSlab(size)) {
        CUDA_CHECK(cudaFree(p));

        std::lock_guard<std::mutex> lock(mutex);
        auto iter = allocations.find(p);
        DCHECK(iter != allocations.end());
        allocations.erase(iter);
        bytesAllocated -= size;
    }
    // Note: no deallocation is done if it is in a slab...
}

void CUDATrackedMemoryResource::PrefetchToGPU() const {
    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));

    std::lock_guard<std::mutex> lock(mutex);

    LOG_VERBOSE("Prefetching %d allocations to GPU memory", allocations.size());
    size_t bytes = 0;
    for (auto iter : allocations) {
        CUDA_CHECK(
            cudaMemPrefetchAsync(iter.first, iter.second, deviceIndex, 0 /* stream */));
        bytes += iter.second;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_VERBOSE("Done prefetching: %d bytes total", bytes);
}

static CUDATrackedMemoryResource cudaTrackedMemoryResource;
Allocator gpuMemoryAllocator(&cudaTrackedMemoryResource);

} // namespace pbrt
