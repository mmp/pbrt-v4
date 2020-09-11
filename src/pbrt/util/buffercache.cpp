// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/buffercache.h>

#include <pbrt/util/stats.h>

namespace pbrt {

BufferCache<int> *indexBufferCache;
BufferCache<Point3f> *pBufferCache;
BufferCache<Normal3f> *nBufferCache;
BufferCache<Point2f> *uvBufferCache;
BufferCache<Vector3f> *sBufferCache;
BufferCache<int> *faceIndexBufferCache;

void InitBufferCaches(Allocator alloc) {
    CHECK(indexBufferCache == nullptr);
    indexBufferCache = alloc.new_object<BufferCache<int>>(alloc);
    pBufferCache = alloc.new_object<BufferCache<Point3f>>(alloc);
    nBufferCache = alloc.new_object<BufferCache<Normal3f>>(alloc);
    uvBufferCache = alloc.new_object<BufferCache<Point2f>>(alloc);
    sBufferCache = alloc.new_object<BufferCache<Vector3f>>(alloc);
    faceIndexBufferCache = alloc.new_object<BufferCache<int>>(alloc);
}

STAT_MEMORY_COUNTER("Memory/Mesh indices", meshIndexBytes);
STAT_MEMORY_COUNTER("Memory/Mesh vertex positions", meshPositionBytes);
STAT_MEMORY_COUNTER("Memory/Mesh normals", meshNormalBytes);
STAT_MEMORY_COUNTER("Memory/Mesh uvs", meshUVBytes);
STAT_MEMORY_COUNTER("Memory/Mesh tangents", meshTangentBytes);
STAT_MEMORY_COUNTER("Memory/Mesh face indices", meshFaceIndexBytes);

void FreeBufferCaches() {
    LOG_VERBOSE("index buffer bytes: %d", indexBufferCache->BytesUsed());
    meshIndexBytes += indexBufferCache->BytesUsed();
    indexBufferCache->Clear();

    LOG_VERBOSE("p bytes: %d", pBufferCache->BytesUsed());
    meshPositionBytes += pBufferCache->BytesUsed();
    pBufferCache->Clear();

    LOG_VERBOSE("n bytes: %d", nBufferCache->BytesUsed());
    meshNormalBytes += nBufferCache->BytesUsed();
    nBufferCache->Clear();
    LOG_VERBOSE("uv bytes: %d", uvBufferCache->BytesUsed());
    meshUVBytes += uvBufferCache->BytesUsed();
    uvBufferCache->Clear();

    LOG_VERBOSE("s bytes: %d", sBufferCache->BytesUsed());
    meshTangentBytes += sBufferCache->BytesUsed();
    sBufferCache->Clear();

    LOG_VERBOSE("face index bytes: %d", faceIndexBufferCache->BytesUsed());
    meshFaceIndexBytes += faceIndexBufferCache->BytesUsed();
    faceIndexBufferCache->Clear();
}

}  // namespace pbrt
