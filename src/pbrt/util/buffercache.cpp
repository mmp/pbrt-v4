// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/buffercache.h>

#include <pbrt/util/stats.h>

namespace pbrt {

// BufferCache Global Definitions
BufferCache<int> *intBufferCache;
BufferCache<Point2f> *point2BufferCache;
BufferCache<Point3f> *point3BufferCache;
BufferCache<Vector3f> *vector3BufferCache;
BufferCache<Normal3f> *normal3BufferCache;

void InitBufferCaches(Allocator alloc) {
    CHECK(intBufferCache == nullptr);
    intBufferCache = alloc.new_object<BufferCache<int>>(alloc);
    point2BufferCache = alloc.new_object<BufferCache<Point2f>>(alloc);
    point3BufferCache = alloc.new_object<BufferCache<Point3f>>(alloc);
    vector3BufferCache = alloc.new_object<BufferCache<Vector3f>>(alloc);
    normal3BufferCache = alloc.new_object<BufferCache<Normal3f>>(alloc);
}

STAT_MEMORY_COUNTER("Memory/Mesh indices", meshIndexBytes);
STAT_MEMORY_COUNTER("Memory/Mesh vertex positions", meshPositionBytes);
STAT_MEMORY_COUNTER("Memory/Mesh normals", meshNormalBytes);
STAT_MEMORY_COUNTER("Memory/Mesh uvs", meshUVBytes);
STAT_MEMORY_COUNTER("Memory/Mesh tangents", meshTangentBytes);
STAT_MEMORY_COUNTER("Memory/Mesh face indices", meshFaceIndexBytes);

void FreeBufferCaches() {
    LOG_VERBOSE("int buffer bytes: %d", intBufferCache->BytesUsed());
    meshIndexBytes += intBufferCache->BytesUsed();
    intBufferCache->Clear();

    LOG_VERBOSE("p bytes: %d", point3BufferCache->BytesUsed());
    meshPositionBytes += point3BufferCache->BytesUsed();
    point3BufferCache->Clear();

    LOG_VERBOSE("n bytes: %d", normal3BufferCache->BytesUsed());
    meshNormalBytes += normal3BufferCache->BytesUsed();
    normal3BufferCache->Clear();

    LOG_VERBOSE("uv bytes: %d", point2BufferCache->BytesUsed());
    meshUVBytes += point2BufferCache->BytesUsed();
    point2BufferCache->Clear();

    LOG_VERBOSE("s bytes: %d", vector3BufferCache->BytesUsed());
    meshTangentBytes += vector3BufferCache->BytesUsed();
    vector3BufferCache->Clear();
}

}  // namespace pbrt
