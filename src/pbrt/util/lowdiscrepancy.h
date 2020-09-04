// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_LOWDISCREPANCY_H
#define PBRT_UTIL_LOWDISCREPANCY_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/shuffle.h>
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
        Float invBase = (Float)1 / (Float)base;
        Float invBaseN = 1;
        while (1 - invBaseN < 1) {
            ++nDigits;
            invBaseN *= invBase;
        }

        permutations = alloc.allocate_object<uint16_t>(nDigits * base);
        for (int digitIndex = 0; digitIndex < nDigits; ++digitIndex) {
            // Compute random permutation for _digitIndex_
            uint32_t digitSeed = (base * 32 + digitIndex) ^ seed;
            for (int digitValue = 0; digitValue < base; ++digitValue)
                Perm(digitIndex, digitValue) =
                    PermutationElement(digitValue, base, digitSeed);
        }
    }

    PBRT_CPU_GPU
    int Permute(int digitIndex, int digitValue) const {
        DCHECK_LT(digitIndex, nDigits);
        DCHECK_LT(digitValue, base);
        return permutations[digitIndex * base + digitValue];
    }

    std::string ToString() const;

    int base;

  private:
    // DigitPermutation Private Methods
    PBRT_CPU_GPU
    uint16_t &Perm(int digitIndex, int digitValue) {
        return permutations[digitIndex * base + digitValue];
    }

    int nDigits;
    // indexed by [digitIndex * base + digitValue]
    uint16_t *permutations;
};

// Low Discrepancy Declarations
inline PBRT_CPU_GPU uint64_t SobolIntervalToIndex(const uint32_t log2Resolution,
                                                  uint64_t sampleNum, const Point2i &p);

PBRT_CPU_GPU
Float RadicalInverse(int baseIndex, uint64_t a);
pstd::vector<DigitPermutation> *ComputeRadicalInversePermutations(uint32_t seed,
                                                                  Allocator alloc = {});
PBRT_CPU_GPU
Float ScrambledRadicalInverse(int baseIndex, uint64_t a, const DigitPermutation &perm);
#if 0
PBRT_CPU_GPU
Float ScrambledRadicalInverse(int baseIndex, uint64_t a, uint32_t seed);
#endif

// NoRandomizer Definition
struct NoRandomizer {
    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return v; }
};

// Low Discrepancy Inline Functions
PBRT_CPU_GPU inline Float RadicalInverse(int baseIndex, uint64_t a) {
    int base = Primes[baseIndex];
    const Float invBase = (Float)1 / (Float)base;
    uint64_t reversedDigits = 0;
    Float invBaseN = 1;
    while (a) {
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        reversedDigits = reversedDigits * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    return std::min(reversedDigits * invBaseN, OneMinusEpsilon);
}

template <int base>
PBRT_CPU_GPU inline uint64_t InverseRadicalInverse(uint64_t inverse, int nDigits) {
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
    const Float invBase = (Float)1 / (Float)base;
    uint64_t reversedDigits = 0;
    Float invBaseN = 1;
    int digitIndex = 0;
    while (1 - invBaseN < 1) {
        uint64_t next = a / base;
        int digitValue = a - next * base;
        reversedDigits = reversedDigits * base + perm.Permute(digitIndex, digitValue);
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
PBRT_CPU_GPU inline uint32_t OwenScramble(uint32_t v, uint32_t hash) {
    // Expect already reversed?
    v = ReverseBits32(v);
    v += hash;
    v ^= v * 0x6c50b47cu;
    v ^= hash;
    v ^= v * 0xb82f1e52u;
    return ReverseBits32(v);
}

template <typename R>
PBRT_CPU_GPU inline Float SampleGeneratorMatrix(pstd::span<const uint32_t> C, uint32_t a,
                                                R randomizer) {
    return std::min(randomizer(MultiplyGenerator(C, a)) * Float(0x1p-32),
                    OneMinusEpsilon);
}

PBRT_CPU_GPU inline Float SampleGeneratorMatrix(pstd::span<const uint32_t> C,
                                                uint32_t a) {
    return SampleGeneratorMatrix(C, a, NoRandomizer());
}

template <typename R>
PBRT_CPU_GPU inline Float SobolSample(int64_t index, int dimension, R randomizer) {
#ifdef PBRT_FLOAT_AS_DOUBLE
    return SobolSampleDouble(index, dimension, randomizer);
#else
    return SobolSampleFloat(index, dimension, randomizer);
#endif
}

PBRT_CPU_GPU inline uint32_t SobolSampleBits32(int64_t a, int dimension);
template <typename R>
PBRT_CPU_GPU inline float SobolSampleFloat(int64_t a, int dimension, R randomizer) {
    CHECK_LT(dimension, NSobolDimensions);
    uint32_t v = SobolSampleBits32(a, dimension);
    v = randomizer(v);
    return std::min(v * 0x1p-32f /* 1/2^32 */, FloatOneMinusEpsilon);
}

PBRT_CPU_GPU
inline uint32_t SobolSampleBits32(int64_t a, int dimension) {
    uint32_t v = 0;
    for (int i = dimension * SobolMatrixSize; a != 0; a >>= 1, i++)
        if (a & 1)
            v ^= SobolMatrices32[i];
    return v;
}

// CranleyPattersonRotator Definition
struct CranleyPattersonRotator {
    PBRT_CPU_GPU
    CranleyPattersonRotator(Float v) : offset(v * (1ull << 32)) {}
    PBRT_CPU_GPU
    CranleyPattersonRotator(uint32_t offset) : offset(offset) {}

    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return v + offset; }

    uint32_t offset;
};

// XORScrambler Definition
struct XORScrambler {
    PBRT_CPU_GPU
    XORScrambler(uint32_t s) : s(s) {}

    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return s ^ v; }

    uint32_t s;
};

// OwenScrambler Definition
struct OwenScrambler {
    PBRT_CPU_GPU
    OwenScrambler(uint32_t seed) : seed(seed) {}

    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return OwenScramble(v, seed); }

    uint32_t seed;
};

// HaltonPixelIndexer Defintion
class HaltonPixelIndexer {
  public:
    // HaltonPixelIndexer Public Methods
    HaltonPixelIndexer(const Point2i &fullResolution);

    PBRT_CPU_GPU
    void SetPixel(const Point2i &p) {
        pixelSampleForIndex = 0;

        int sampleStride = baseScales[0] * baseScales[1];
        if (sampleStride > 1) {
            Point2i pm(Mod(p[0], MaxHaltonResolution), Mod(p[1], MaxHaltonResolution));
            for (int i = 0; i < 2; ++i) {
                uint64_t dimOffset =
                    (i == 0) ? InverseRadicalInverse<2>(pm[i], baseExponents[i])
                             : InverseRadicalInverse<3>(pm[i], baseExponents[i]);
                pixelSampleForIndex +=
                    dimOffset * (sampleStride / baseScales[i]) * multInverse[i];
            }
            pixelSampleForIndex %= sampleStride;
        }
    }

    PBRT_CPU_GPU
    void SetPixelSample(int pixelSample) {
        int sampleStride = baseScales[0] * baseScales[1];
        sampleIndex = pixelSampleForIndex + pixelSample * sampleStride;
    }

    PBRT_CPU_GPU
    Point2f SampleFirst2D() const {
        return {RadicalInverse(0, sampleIndex >> baseExponents[0]),
                RadicalInverse(1, sampleIndex / baseScales[1])};
    }

    PBRT_CPU_GPU
    int64_t SampleIndex() const { return sampleIndex; }

    std::string ToString() const {
        return StringPrintf("[ HaltonPixelIndexer pixelSampleForIndex: %d "
                            "sampleIndex: %d baseScales: %s baseExponents: %s "
                            "multInverse[0]: %d multInverse[1]: %d ]",
                            pixelSampleForIndex, sampleIndex, baseScales, baseExponents,
                            multInverse[0], multInverse[1]);
    }

  private:
    // HaltonPixelIndexer Private Methods
    static uint64_t multiplicativeInverse(int64_t a, int64_t n);
    static void extendedGCD(uint64_t a, uint64_t b, int64_t *x, int64_t *y);

    // HaltonPixelIndexer Private Members
    static constexpr int MaxHaltonResolution = 128;
    Point2i baseScales, baseExponents;
    int multInverse[2];
    int64_t pixelSampleForIndex;
    int64_t sampleIndex;
};

enum class RandomizeStrategy { None, CranleyPatterson, Xor, Owen };

std::string ToString(RandomizeStrategy r);

// Define _CVanDerCorput_ Generator Matrix
PBRT_CONST uint32_t CVanDerCorput[32] = {
    // clang-format off
    0b10000000000000000000000000000000,
    0b1000000000000000000000000000000,
    0b100000000000000000000000000000,
    0b10000000000000000000000000000,
    // Remainder of Van Der Corput generator matrix entries
0b1000000000000000000000000000,
  0b100000000000000000000000000,
  0b10000000000000000000000000,
  0b1000000000000000000000000,
  0b100000000000000000000000,
  0b10000000000000000000000,
  0b1000000000000000000000,
  0b100000000000000000000,
  0b10000000000000000000,
  0b1000000000000000000,
  0b100000000000000000,
  0b10000000000000000,
  0b1000000000000000,
  0b100000000000000,
  0b10000000000000,
  0b1000000000000,
  0b100000000000,
  0b10000000000,
  0b1000000000,
  0b100000000,
  0b10000000,
  0b1000000,
  0b100000,
  0b10000,
  0b1000,
  0b100,
  0b10,
  0b1,

    // clang-format on
};

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
