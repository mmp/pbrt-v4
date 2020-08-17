// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/lowdiscrepancy.h>

#include <pbrt/util/bits.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/print.h>
#include <pbrt/util/shuffle.h>
#include <pbrt/util/stats.h>

namespace pbrt {

std::string DigitPermutation::ToString() const {
    std::string s = StringPrintf(
        "[ DigitPermitation base: %d nDigits: %d permutations: ", base, nDigits);
    for (int digitIndex = 0; digitIndex < nDigits; ++digitIndex) {
        s += StringPrintf("[%d] ( ", digitIndex);
        for (int digitValue = 0; digitValue < base; ++digitValue) {
            s += StringPrintf("%d", permutations[digitIndex * base + digitValue]);
            if (digitValue != base - 1)
                s += ", ";
        }
        s += ") ";
    }

    return s + " ]";
}

// HaltonIndexer Local Constants
constexpr int HaltonPixelIndexer::MaxHaltonResolution;

// HaltonIndexer Method Definitions
HaltonPixelIndexer::HaltonPixelIndexer(const Point2i &fullResolution) {
    // Find radical inverse base scales and exponents that cover sampling area
    for (int i = 0; i < 2; ++i) {
        int base = (i == 0) ? 2 : 3;
        int scale = 1, exp = 0;
        while (scale < std::min(fullResolution[i], MaxHaltonResolution)) {
            scale *= base;
            ++exp;
        }
        baseScales[i] = scale;
        baseExponents[i] = exp;
    }

    // Compute multiplicative inverses for _baseScales_
    multInverse[0] = multiplicativeInverse(baseScales[1], baseScales[0]);
    multInverse[1] = multiplicativeInverse(baseScales[0], baseScales[1]);
}

uint64_t HaltonPixelIndexer::multiplicativeInverse(int64_t a, int64_t n) {
    int64_t x, y;
    extendedGCD(a, n, &x, &y);
    return Mod(x, n);
}

void HaltonPixelIndexer::extendedGCD(uint64_t a, uint64_t b, int64_t *x, int64_t *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return;
    }
    int64_t d = a / b, xp, yp;
    extendedGCD(b, a % b, &xp, &yp);
    *x = yp;
    *y = xp - (d * yp);
}

std::string ToString(RandomizeStrategy r) {
    switch (r) {
    case RandomizeStrategy::None:
        return "None";
    case RandomizeStrategy::CranleyPatterson:
        return "CranleyPatterson";
    case RandomizeStrategy::Xor:
        return "Xor";
    case RandomizeStrategy::Owen:
        return "Owen";
    default:
        LOG_FATAL("Unhandled RandomizeStrategy");
        return "";
    }
}

// Low Discrepancy Static Functions
template <int base>
PBRT_NOINLINE PBRT_CPU_GPU static Float RadicalInverseSpecialized(uint64_t a) {
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
    DCHECK_LT(reversedDigits * invBaseN, 1.00001);
    return std::min(reversedDigits * invBaseN, OneMinusEpsilon);
}

template <int base>
PBRT_NOINLINE PBRT_CPU_GPU static Float ScrambledRadicalInverseSpecialized(
    uint64_t a, const DigitPermutation &perm) {
    CHECK_EQ(perm.base, base);
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

// Low Discrepancy Function Definitions
pstd::vector<DigitPermutation> *ComputeRadicalInversePermutations(uint32_t seed,
                                                                  Allocator alloc) {
    pstd::vector<DigitPermutation> *perms =
        alloc.new_object<pstd::vector<DigitPermutation>>(alloc);
    perms->resize(PrimeTableSize);
    ParallelFor(0, PrimeTableSize, [&perms, &alloc, seed](int64_t i) {
        (*perms)[i] = DigitPermutation(Primes[i], seed, alloc);
    });
    return perms;
}

}  // namespace pbrt
