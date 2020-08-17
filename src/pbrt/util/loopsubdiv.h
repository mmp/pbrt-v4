// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_LOOPSUBDIV_H
#define PBRT_UTIL_LOOPSUBDIV_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

namespace pbrt {

// LoopSubdiv Declarations
TriangleMesh *LoopSubdivide(const Transform *renderFromObject, bool reverseOrientation,
                            int nLevels, pstd::span<const int> vertexIndices,
                            pstd::span<const Point3f> p, Allocator alloc);

}  // namespace pbrt

#endif  // PBRT_UTIL_LOOPSUBDIV_H
