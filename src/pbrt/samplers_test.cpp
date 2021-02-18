// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/samplers.h>

#include <set>

using namespace pbrt;

// Make sure all samplers give the same sample values if we go back to the
// same pixel / sample index.
TEST(Sampler, ConsistentValues) {
    constexpr int rootSpp = 4;
    constexpr int spp = rootSpp * rootSpp;
    Point2i resolution(100, 101);

    std::vector<Sampler> samplers;
    samplers.push_back(new HaltonSampler(spp, resolution));
    samplers.push_back(new IndependentSampler(spp));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::None));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::PermuteDigits));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::FastOwen));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::Owen));
    samplers.push_back(new ZSobolSampler(spp, resolution, RandomizeStrategy::None));
    samplers.push_back(new ZSobolSampler(spp, resolution, RandomizeStrategy::PermuteDigits));
    samplers.push_back(new ZSobolSampler(spp, resolution, RandomizeStrategy::FastOwen));
    samplers.push_back(new ZSobolSampler(spp, resolution, RandomizeStrategy::Owen));
    samplers.push_back(new PMJ02BNSampler(spp));
    samplers.push_back(new StratifiedSampler(rootSpp, rootSpp, true));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::None));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::PermuteDigits));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::Owen));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::FastOwen));

    for (auto &sampler : samplers) {
        std::vector<Float> s1d[spp];
        std::vector<Point2f> s2d[spp];

        for (int s = 0; s < spp; ++s) {
            sampler.StartPixelSample({1, 5}, s);
            for (int i = 0; i < 10; ++i) {
                s2d[s].push_back(sampler.Get2D());
                s1d[s].push_back(sampler.Get1D());
            }
        }

        // Go somewhere else and generate some samples, just to make sure
        // things are shaken up.
        sampler.StartPixelSample({0, 6}, 10);
        sampler.Get2D();
        sampler.Get2D();
        sampler.Get1D();

        // Now go back and generate samples again, but enumerate them in a
        // different order to make sure the sampler is doing the right
        // thing.
        for (int s = spp - 1; s >= 0; --s) {
            sampler.StartPixelSample({1, 5}, s);
            for (int i = 0; i < s2d[s].size(); ++i) {
                EXPECT_EQ(s2d[s][i], sampler.Get2D());
                EXPECT_EQ(s1d[s][i], sampler.Get1D());
            }
        }
    }
}

static void checkElementary(const char *name, std::vector<Point2f> samples,
                            int logSamples) {
    for (int i = 0; i <= logSamples; ++i) {
        // Check one set of elementary intervals: number of intervals
        // in each dimension.
        int nx = 1 << i, ny = 1 << (logSamples - i);

        std::vector<int> count(1 << logSamples, 0);
        for (const Point2f &s : samples) {
            // Map the sample to an interval
            Float x = nx * s.x, y = ny * s.y;
            EXPECT_GE(x, 0);
            EXPECT_LT(x, nx);
            EXPECT_GE(y, 0);
            EXPECT_LT(y, ny);
            int index = (int)std::floor(y) * nx + (int)std::floor(x);
            EXPECT_GE(index, 0);
            EXPECT_LT(index, count.size());

            // This should be the first time a sample has landed in its
            // interval.
            EXPECT_EQ(0, count[index])
                << "Sampler " << name << " with interval " << nx << " x " << ny;
            ++count[index];
        }
    }
}

static void checkElementarySampler(const char *name, Sampler sampler,
                                   int logSamples, int res = 1) {
    // Get all of the samples for a pixel.
    int spp = sampler.SamplesPerPixel();
    std::vector<Point2f> samples;
    for (Point2i p : Bounds2i(Point2i(0, 0), Point2i(res, res))) {
        samples.clear();
        for (int i = 0; i < spp; ++i) {
            sampler.StartPixelSample(p, i);
            samples.push_back(sampler.GetPixel2D());
        }

        checkElementary(name, samples, logSamples);
    }
}

// TODO: check Halton (where the elementary intervals are (2^i, 3^j)).

TEST(PaddedSobolSampler, ElementaryIntervals) {
    for (auto rand :
         {RandomizeStrategy::None, RandomizeStrategy::PermuteDigits})
        for (int logSamples = 2; logSamples <= 10; ++logSamples)
            checkElementarySampler("PaddedSobolSampler",
                                   new PaddedSobolSampler(1 << logSamples, rand),
                                   logSamples);
}

TEST(ZSobolSampler, ElementaryIntervals) {
    for (int seed : {0, 1, 5, 6, 10, 15})
        for (auto rand :
                 {RandomizeStrategy::None, RandomizeStrategy::PermuteDigits})
            for (int logSamples = 2; logSamples <= 8; ++logSamples)
                checkElementarySampler(StringPrintf("ZSobolSampler - %s - %d", rand, seed).c_str(),
                                       new ZSobolSampler(1 << logSamples, Point2i(10, 10), rand, seed),
                                       logSamples, 10);
}

TEST(SobolUnscrambledSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; ++logSamples)
        checkElementarySampler(
            "Sobol Unscrambled",
            new SobolSampler(1 << logSamples, Point2i(1, 1), RandomizeStrategy::None),
            logSamples);
}

TEST(SobolXORScrambledSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; ++logSamples)
        checkElementarySampler(
            "Sobol XOR Scrambled",
            new SobolSampler(1 << logSamples, Point2i(1, 1), RandomizeStrategy::PermuteDigits),
            logSamples);
}

TEST(SobolOwenScrambledSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; ++logSamples)
        checkElementarySampler(
            "Sobol Owen Scrambled",
            new SobolSampler(1 << logSamples, Point2i(1, 1), RandomizeStrategy::Owen),
            logSamples);
}

TEST(PMJ02BNSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; logSamples += 2)
        checkElementarySampler("PMJ02BNSampler", new PMJ02BNSampler(1 << logSamples),
                               logSamples);
}

TEST(ZSobolSampler, ValidIndices) {
    Point2i res(16, 9);
    for (int logSamples = 0; logSamples <= 10; ++logSamples) {
        int spp = 1 << logSamples;
        ZSobolSampler sampler(spp, res, RandomizeStrategy::PermuteDigits);

        for (int dim = 0; dim < 7; dim += 3) {
            std::set<uint64_t> returnedIndices;
            for (Point2i p : Bounds2i(Point2i(0, 0), res)) {
                uint64_t pow2Base;
                for (int i = 0; i < spp; ++i) {
                    sampler.StartPixelSample(p, i, dim);
                    uint64_t index = sampler.GetSampleIndex();

                    // Make sure no index is repeated across multiple pixels
                    EXPECT_TRUE(returnedIndices.find(index) == returnedIndices.end());
                    returnedIndices.insert(index);

                    // Make sure that all samples for this pixel are within the
                    // same pow2 aligned and sized range of the sample indices.
                    if (i == 0)
                        pow2Base = index / spp;
                    else
                        EXPECT_EQ(index / spp, pow2Base);
                }
            }
        }
    }
}
