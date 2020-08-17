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

// Define 2D Sobol$'$ generator matrices _CSobol[2]_
PBRT_CONST uint32_t CSobol[2][32] = {
    {0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x8000000, 0x4000000, 0x2000000,
     0x1000000,  0x800000,   0x400000,   0x200000,   0x100000,  0x80000,   0x40000,
     0x20000,    0x10000,    0x8000,     0x4000,     0x2000,    0x1000,    0x800,
     0x400,      0x200,      0x100,      0x80,       0x40,      0x20,      0x10,
     0x8,        0x4,        0x2,        0x1},
    {0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000,
     0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000,
     0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800,
     0xcc00cc00, 0xaa00aa00, 0xff00ff00, 0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
     0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff}};

}  // namespace pbrt

#endif  // PBRT_UTIL_SOBOLMATRICES_H
