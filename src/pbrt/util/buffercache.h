// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_BUFFERCACHE_H
#define PBRT_UTIL_BUFFERCACHE_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Redundant vertex and index buffers", redundantBufferBytes);
STAT_PERCENT("Geometry/Buffer cache hits", nBufferCacheHits, nBufferCacheLookups);

// BufferCache Definition
template <typename T>
class BufferCache {
  public:
    // BufferCache Public Methods
    BufferCache(Allocator alloc) : alloc(alloc) {}

    const T *LookupOrAdd(const std::vector<T> &buf) {
        ++nBufferCacheLookups;
        std::lock_guard<std::mutex> lock(mutex);
        // Return pointer to data if _buf_ contents is already in the cache
        Buffer lookupBuffer(buf.data(), buf.size());
        if (auto iter = cache.find(lookupBuffer); iter != cache.end()) {
            DCHECK(std::memcmp(buf.data(), iter->ptr, buf.size() * sizeof(T)) == 0);
            ++nBufferCacheHits;
            redundantBufferBytes += buf.capacity() * sizeof(T);
            return iter->ptr;
        }

        // Add _buf_ contents to cache and return pointer to cached copy
        T *ptr = alloc.allocate_object<T>(buf.size());
        std::copy(buf.begin(), buf.end(), ptr);
        bytesUsed += buf.size() * sizeof(T);
        cache.insert(Buffer(ptr, buf.size()));
        return ptr;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto iter : cache)
            alloc.deallocate_object(const_cast<T *>(iter.ptr), iter.size);
        cache.clear();
    }

    size_t BytesUsed() const { return bytesUsed; }

  private:
    // BufferCache::Buffer Definition
    struct Buffer {
        // BufferCache::Buffer Public Methods
        Buffer() = default;
        Buffer(const T *ptr, size_t size) : ptr(ptr), size(size) {}

        bool operator==(const Buffer &b) const {
            return size == b.size && std::memcmp(ptr, b.ptr, size * sizeof(T)) == 0;
        }

        const T *ptr = nullptr;
        size_t size = 0;
    };

    // BufferCache::BufferHasher Definition
    struct BufferHasher {
        size_t operator()(const Buffer &b) const { return HashBuffer(b.ptr, b.size); }
    };

    // BufferCache Private Members
    Allocator alloc;
    std::mutex mutex;
    std::unordered_set<Buffer, BufferHasher> cache;
    size_t bytesUsed = 0;
};

// BufferCache Global Declarations
extern BufferCache<int> *intBufferCache;
extern BufferCache<Point2f> *point2BufferCache;
extern BufferCache<Point3f> *point3BufferCache;
extern BufferCache<Vector3f> *vector3BufferCache;
extern BufferCache<Normal3f> *normal3BufferCache;

void InitBufferCaches(Allocator alloc);
void FreeBufferCaches();

}  // namespace pbrt

#endif  // PBRT_UTIL_BUFFERCACHE_H
