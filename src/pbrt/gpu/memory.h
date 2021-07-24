// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_MEMORY_H
#define PBRT_GPU_MEMORY_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace pbrt {

#ifdef PBRT_BUILD_GPU_RENDERER

class CUDAMemoryResource : public pstd::pmr::memory_resource {
    void *do_allocate(size_t size, size_t alignment);
    void do_deallocate(void *p, size_t bytes, size_t alignment);

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }
};

class CUDATrackedMemoryResource : public CUDAMemoryResource {
  public:
    void *do_allocate(size_t size, size_t alignment);
    void do_deallocate(void *p, size_t bytes, size_t alignment);

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }

    void PrefetchToGPU() const;
    size_t BytesAllocated() const { return bytesAllocated; }

  private:
    bool bypassSlab(size_t size) const {
#ifdef PBRT_DEBUG_BUILD
        return true;
#else
        return size > slabSize / 4;
#endif
    }

    void *cudaAllocate(size_t size, size_t alignment);

    static constexpr int slabSize = 1024 * 1024;
    struct alignas(64) Slab {
        uint8_t *ptr = nullptr;
        size_t offset = slabSize;
    };
    static constexpr int maxThreads = 256;
    Slab threadSlabs[maxThreads];

    mutable std::mutex mutex;
    std::atomic<size_t> bytesAllocated{};
    std::unordered_map<void *, size_t> allocations;
};

extern Allocator gpuMemoryAllocator;

#endif

} // namespace pbrt

#endif // PBRT_GPU_MEMORY_H
