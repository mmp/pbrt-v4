// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_BITS_H
#define PBRT_UTIL_BITS_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>

#include <cstdint>

#ifdef PBRT_HAS_INTRIN_H
#include <intrin.h>
#endif  // PBRT_HAS_INTRIN_H

namespace pbrt {

// Bit Operation Inline Functions
PBRT_CPU_GPU
inline uint32_t ReverseBits32(uint32_t n) {
#ifdef PBRT_IS_GPU_CODE
    return __brev(n);
#else
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
#endif
}

PBRT_CPU_GPU
inline uint64_t ReverseBits64(uint64_t n) {
#ifdef PBRT_IS_GPU_CODE
    return __brevll(n);
#else
    uint64_t n0 = ReverseBits32((uint32_t)n);
    uint64_t n1 = ReverseBits32((uint32_t)(n >> 32));
    return (n0 << 32) | n1;
#endif
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// updated to 64 bits.
PBRT_CPU_GPU
inline uint64_t LeftShift2(uint64_t x) {
    x &= 0xffffffff;
    x = (x ^ (x << 16)) & 0x0000ffff0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f0f0f0f0f;
    x = (x ^ (x << 2)) & 0x3333333333333333;
    x = (x ^ (x << 1)) & 0x5555555555555555;
    return x;
}

PBRT_CPU_GPU
inline uint64_t EncodeMorton2(uint32_t x, uint32_t y) {
    return (LeftShift2(y) << 1) | LeftShift2(x);
}

PBRT_CPU_GPU
inline uint32_t LeftShift3(uint32_t x) {
    DCHECK_LE(x, (1u << 10));
    if (x == (1 << 10))
        --x;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

PBRT_CPU_GPU
inline uint32_t EncodeMorton3(float x, float y, float z) {
    DCHECK_GE(x, 0);
    DCHECK_GE(y, 0);
    DCHECK_GE(z, 0);
    return (LeftShift3(z) << 2) | (LeftShift3(y) << 1) | LeftShift3(x);
}

PBRT_CPU_GPU
inline uint32_t Compact1By1(uint64_t x) {
    // TODO: as of Haswell, the PEXT instruction could do all this in a
    // single instruction.
    // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x &= 0x5555555555555555;
    // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 1)) & 0x3333333333333333;
    // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 2)) & 0x0f0f0f0f0f0f0f0f;
    // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff00ff00ff;
    // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff0000ffff;
    // ...
    x = (x ^ (x >> 16)) & 0xffffffff;
    return x;
}

PBRT_CPU_GPU
inline void DecodeMorton2(uint64_t v, uint32_t *x, uint32_t *y) {
    *x = Compact1By1(v);
    *y = Compact1By1(v >> 1);
}

PBRT_CPU_GPU
inline uint32_t Compact1By2(uint32_t x) {
    x &= 0x09249249;                   // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3;   // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f;   // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff;   // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff;  // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
PBRT_CPU_GPU inline uint64_t MixBits(uint64_t v);

inline uint64_t MixBits(uint64_t v) {
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}

}  // namespace pbrt

#endif  // PBRT_UTIL_BITS_H
