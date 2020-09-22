// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SOBOLMATRICES_H
#define PBRT_UTIL_SOBOLMATRICES_H

#include <pbrt/pbrt.h>

#include <cstdint>

namespace pbrt {

// Sobol Matrix Declarations
static constexpr int NSobolDimensions = 1024;
static constexpr int SobolMatrixSize = 52;
extern PBRT_CONST uint32_t SobolMatrices32[NSobolDimensions * SobolMatrixSize];
extern PBRT_CONST uint64_t SobolMatrices64[NSobolDimensions * SobolMatrixSize];

extern PBRT_CONST uint64_t VdCSobolMatrices[][SobolMatrixSize];
extern PBRT_CONST uint64_t VdCSobolMatricesInv[][SobolMatrixSize];

}  // namespace pbrt

#endif  // PBRT_UTIL_SOBOLMATRICES_H
