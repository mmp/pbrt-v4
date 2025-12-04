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
    void Free();


    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }

    void PrefetchToGPU() const;
    size_t BytesAllocated() const { return bytesAllocated; }

    static CUDATrackedMemoryResource singleton;

  private:
    mutable std::mutex mutex;
    std::atomic<size_t> bytesAllocated{};
    std::unordered_map<void *, size_t> allocations;
};

#endif

}  // namespace pbrt

#endif  // PBRT_GPU_MEMORY_H
