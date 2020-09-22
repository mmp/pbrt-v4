// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_PMJ02TABLES_H
#define PBRT_UTIL_PMJ02TABLES_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>

namespace pbrt {

// PMJ02BN Table Declarations
constexpr int nPMJ02bnSets = 5;
constexpr int nPMJ02bnSamples = 65536;
extern PBRT_CONST uint32_t pmj02bnSamples[nPMJ02bnSets][nPMJ02bnSamples][2];

// PMJ02BN Table Inline Functions
PBRT_CPU_GPU inline Point2f GetPMJ02BNSample(int setIndex, int sampleIndex);

PBRT_CPU_GPU
inline Point2f GetPMJ02BNSample(int setIndex, int sampleIndex) {
    setIndex %= nPMJ02bnSets;
    DCHECK_LT(sampleIndex, nPMJ02bnSamples);
    sampleIndex %= nPMJ02bnSamples;

    // Convert from fixed-point.
#ifdef PBRT_IS_GPU_CODE
    return Point2f(pmj02bnSamples[setIndex][sampleIndex][0] * 0x1p-32f,
                   pmj02bnSamples[setIndex][sampleIndex][1] * 0x1p-32f);
#else
    // Double precision is key here for the pixel sample sorting, but not
    // necessary otherwise.
    return Point2f(pmj02bnSamples[setIndex][sampleIndex][0] * 0x1p-32,
                   pmj02bnSamples[setIndex][sampleIndex][1] * 0x1p-32);
#endif
}

}  // namespace pbrt

#endif  // PBRT_UTIL_PMJ02TABLES_H
