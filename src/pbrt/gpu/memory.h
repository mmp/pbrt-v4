
#ifndef PBRT_GPU_MEMORY_H
#define PBRT_GPU_MEMORY_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

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

    size_t bytesAllocated = 0;
    uint8_t *currentSlab = nullptr;
    static constexpr int slabSize = 1024 * 1024;
    size_t slabOffset = slabSize;
    mutable std::mutex mutex;
    std::unordered_map<void *, size_t> allocations;
};

extern Allocator gpuMemoryAllocator;

#endif

} // namespace pbrt

#endif // PBRT_GPU_MEMORY_H
