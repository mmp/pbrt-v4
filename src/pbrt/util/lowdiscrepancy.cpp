// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/lowdiscrepancy.h>

#include <pbrt/util/bits.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/print.h>
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

std::string ToString(RandomizeStrategy r) {
    switch (r) {
    case RandomizeStrategy::None:
        return "None";
    case RandomizeStrategy::PermuteDigits:
        return "PermuteDigits";
    case RandomizeStrategy::FastOwen:
        return "FastOwen";
    case RandomizeStrategy::Owen:
        return "Owen";
    default:
        LOG_FATAL("Unhandled RandomizeStrategy");
        return "";
    }
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
