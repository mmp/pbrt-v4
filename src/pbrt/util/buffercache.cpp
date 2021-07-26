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

void InitBufferCaches() {
    CHECK(intBufferCache == nullptr);
    intBufferCache = new BufferCache<int>;
    point2BufferCache = new BufferCache<Point2f>;
    point3BufferCache = new BufferCache<Point3f>;
    vector3BufferCache = new BufferCache<Vector3f>;
    normal3BufferCache = new BufferCache<Normal3f>;
}

}  // namespace pbrt
