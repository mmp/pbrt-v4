// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_LOWDISCREPANCY_H
#define PBRT_UTIL_LOWDISCREPANCY_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sobolmatrices.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <memory>
#include <string>

namespace pbrt {

// DigitPermutation Definition
class DigitPermutation {
  public:
    // DigitPermutation Public Methods
    DigitPermutation() = default;
    DigitPermutation(int base, uint32_t seed, Allocator alloc) : base(base) {
        CHECK_LT(base, 65536);  // uint16_t
        // Compute number of digits needed for _base_
        nDigits = 0;
        Float invBase = (Float)1 / (Float)base, invBaseN = 1;
        while (1 - invBaseN < 1) {
            ++nDigits;
            invBaseN *= invBase;
        }

        permutations = alloc.allocate_object<uint16_t>(nDigits * base);
        for (int digitIndex = 0; digitIndex < nDigits; ++digitIndex) {
            // Compute random permutation for _digitIndex_
            uint32_t digitSeed = MixBits(((base << 8) + digitIndex) ^ seed);
            for (int digitValue = 0; digitValue < base; ++digitValue) {
                int index = digitIndex * base + digitValue;
                permutations[index] = PermutationElement(digitValue, base, digitSeed);
            }
        }
    }

    PBRT_CPU_GPU
    int Permute(int digitIndex, int digitValue) const {
        DCHECK_LT(digitIndex, nDigits);
        DCHECK_LT(digitValue, base);
        return permutations[digitIndex * base + digitValue];
    }

    std::string ToString() const;

  private:
    // DigitPermutation Private Members
    int base, nDigits;
    uint16_t *permutations;
};

// Low Discrepancy Declarations
inline PBRT_CPU_GPU uint64_t SobolIntervalToIndex(const uint32_t log2Scale,
                                                  uint64_t sampleIndex, const Point2i &p);

PBRT_CPU_GPU
Float RadicalInverse(int baseIndex, uint64_t a);
pstd::vector<DigitPermutation> *ComputeRadicalInversePermutations(uint32_t seed,
                                                                  Allocator alloc = {});
PBRT_CPU_GPU
Float ScrambledRadicalInverse(int baseIndex, uint64_t a, const DigitPermutation &perm);

// NoRandomizer Definition
struct NoRandomizer {
    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return v; }
};

// Low Discrepancy Inline Functions
PBRT_CPU_GPU inline Float RadicalInverse(int baseIndex, uint64_t a) {
    int base = Primes[baseIndex];
    Float invBase = (Float)1 / (Float)base, invBaseN = 1;
    uint64_t reversedDigits = 0;
    while (a) {
        // Extract least significant digit from _a_ and update _reversedDigits_
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        reversedDigits = reversedDigits * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    return std::min(reversedDigits * invBaseN, OneMinusEpsilon);
}

PBRT_CPU_GPU
inline uint64_t InverseRadicalInverse(uint64_t inverse, int base, int nDigits) {
    uint64_t index = 0;
    for (int i = 0; i < nDigits; ++i) {
        uint64_t digit = inverse % base;
        inverse /= base;
        index = index * base + digit;
    }
    return index;
}

PBRT_CPU_GPU inline Float ScrambledRadicalInverse(int baseIndex, uint64_t a,
                                                  const DigitPermutation &perm) {
    int base = Primes[baseIndex];
    Float invBase = (Float)1 / (Float)base, invBaseN = 1;
    uint64_t reversedDigits = 0;
    int digitIndex = 0;
    while (1 - invBaseN < 1) {
        // Permute least significant digit from _a_ and update _reversedDigits_
        uint64_t next = a / base;
        int digitValue = a - next * base;
        reversedDigits = reversedDigits * base + perm.Permute(digitIndex, digitValue);
        invBaseN *= invBase;
        ++digitIndex;
        a = next;
    }
    return std::min(invBaseN * reversedDigits, OneMinusEpsilon);
}

PBRT_CPU_GPU inline Float OwenScrambledRadicalInverse(int baseIndex, uint64_t a,
                                                      uint32_t hash) {
    int base = Primes[baseIndex];
    Float invBase = (Float)1 / (Float)base, invBaseN = 1;
    uint64_t reversedDigits = 0;
    int digitIndex = 0;
    while (1 - invBaseN < 1) {
        uint64_t next = a / base;
        int digitValue = a - next * base;
        uint32_t digitHash = MixBits(hash ^ reversedDigits);
        digitValue = PermutationElement(digitValue, base, digitHash);
        reversedDigits = reversedDigits * base + digitValue;
        invBaseN *= invBase;
        ++digitIndex;
        a = next;
    }
    return std::min(invBaseN * reversedDigits, OneMinusEpsilon);
}

PBRT_CPU_GPU inline uint32_t MultiplyGenerator(pstd::span<const uint32_t> C, uint32_t a) {
    uint32_t v = 0;
    for (int i = 0; a != 0; ++i, a >>= 1)
        if (a & 1)
            v ^= C[i];
    return v;
}

// Laine et al., Stratified Sampling for Stochastic Transparency, Sec 3.1...
PBRT_CPU_GPU inline uint32_t FastOwenBinaryScramble(uint32_t v, uint32_t hash) {
    v = ReverseBits32(v);
    v += hash;
    v ^= v * 0x6c50b47cu;
    v ^= v * 0xb82f1e52u;
    v ^= v * 0xc7afe638u;
    v ^= v * 0x8d22f6e6u;
    return ReverseBits32(v);
}

PBRT_CPU_GPU inline uint32_t OwenScrambleBinaryFull(uint32_t v, uint32_t hash) {
    if (hash & 1)
        v ^= 1u << 31;

    for (int b = 1; b < 32; ++b) {
        uint32_t mask = (~0u) << (32 - b);
        if (MixBits((v & mask) ^ hash) & (1u << b))
            v ^= 1u << (31 - b);
    }

    return v;
}

template <typename R>
PBRT_CPU_GPU inline Float SobolSample(int64_t index, int dimension, R randomizer) {
#ifdef PBRT_FLOAT_AS_DOUBLE
    return SobolSampleDouble(index, dimension, randomizer);
#else
    return SobolSampleFloat(index, dimension, randomizer);
#endif
}

template <typename R>
PBRT_CPU_GPU inline float SobolSampleFloat(int64_t a, int dimension, R randomizer) {
    DCHECK_LT(dimension, NSobolDimensions);
    DCHECK(a >= 0 && a < (1ull << SobolMatrixSize));
    // Compute initial Sobol sample _v_ using generator matrices
    uint32_t v = 0;
    for (int i = dimension * SobolMatrixSize; a != 0; a >>= 1, i++)
        if (a & 1)
            v ^= SobolMatrices32[i];

    v = randomizer(v);
    return std::min(v * 0x1p-32f, FloatOneMinusEpsilon);
}

// CranleyPattersonRotator Definition
struct CranleyPattersonRotator {
    PBRT_CPU_GPU
    CranleyPattersonRotator(Float v) : delta(v * (1ull << 32)) {}
    PBRT_CPU_GPU
    CranleyPattersonRotator(uint32_t delta) : delta(delta) {}
    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return v + delta; }
    uint32_t delta;
};

// BinaryPermuteScrambler Definition
struct BinaryPermuteScrambler {
    PBRT_CPU_GPU
    BinaryPermuteScrambler(uint32_t permutation) : permutation(permutation) {}
    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return permutation ^ v; }
    uint32_t permutation;
};

// FastOwenScrambler Definition
struct FastOwenScrambler {
    PBRT_CPU_GPU
    FastOwenScrambler(uint32_t seed) : seed(seed) {}
    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return FastOwenBinaryScramble(v, seed); }
    uint32_t seed;
};

// OwenScrambler Definition
struct OwenScrambler {
    PBRT_CPU_GPU
    OwenScrambler(uint32_t seed) : seed(seed) {}
    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return OwenScrambleBinaryFull(v, seed); }
    uint32_t seed;
};

// RandomizeStrategy Definition
enum class RandomizeStrategy { None, CranleyPatterson, PermuteDigits, FastOwen, Owen };

std::string ToString(RandomizeStrategy r);

PBRT_CPU_GPU
inline uint64_t SobolIntervalToIndex(uint32_t m, uint64_t frame, const Point2i &p) {
    if (m == 0)
        return frame;

    const uint32_t m2 = m << 1;
    uint64_t index = uint64_t(frame) << m2;

    uint64_t delta = 0;
    for (int c = 0; frame; frame >>= 1, ++c)
        if (frame & 1)  // Add flipped column m + c + 1.
            delta ^= VdCSobolMatrices[m - 1][c];

    // flipped b
    uint64_t b = (((uint64_t)((uint32_t)p.x) << m) | ((uint32_t)p.y)) ^ delta;

    for (int c = 0; b; b >>= 1, ++c)
        if (b & 1)  // Add column 2 * m - c.
            index ^= VdCSobolMatricesInv[m - 1][c];

    return index;
}

PBRT_CPU_GPU
inline uint64_t SobolSampleBits64(int64_t a, int dimension) {
    CHECK_LT(dimension, NSobolDimensions);
    DCHECK(a >= 0 && a < (1ull << SobolMatrixSize));
    uint64_t v = 0;
    for (int i = dimension * SobolMatrixSize; a != 0; a >>= 1, i++)
        if (a & 1)
            v ^= SobolMatrices64[i];
    return v;
}

template <typename R>
PBRT_CPU_GPU inline double SobolSampleDouble(int64_t a, int dimension, R randomizer) {
    uint64_t v = SobolSampleBits64(a, dimension);
    // FIXME? We just scramble the high bits here...
    uint32_t vs = randomizer(v >> 32);
    v = (uint64_t(vs) << 32) | (v & 0xffffffff);
    return std::min(v * (1.0 / (1ULL << SobolMatrixSize)), DoubleOneMinusEpsilon);
}

}  // namespace pbrt

#endif  // PBRT_UTIL_LOWDISCREPANCY_H
