// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/samplers.h>

using namespace pbrt;

// Make sure all samplers give the same sample values if we go back to the
// same pixel / sample index.
TEST(Sampler, ConsistentValues) {
    constexpr int rootSpp = 4;
    constexpr int spp = rootSpp * rootSpp;
    Point2i resolution(100, 101);

    std::vector<SamplerHandle> samplers;
    samplers.push_back(new HaltonSampler(spp, resolution));
    samplers.push_back(new RandomSampler(spp));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::None));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::CranleyPatterson));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::PermuteDigits));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::Owen));
    samplers.push_back(new PMJ02BNSampler(spp));
    samplers.push_back(new StratifiedSampler(rootSpp, rootSpp, true));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::None));
    samplers.push_back(
        new SobolSampler(spp, resolution, RandomizeStrategy::CranleyPatterson));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::PermuteDigits));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::Owen));

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

static void checkElementarySampler(const char *name, SamplerHandle sampler,
                                   int logSamples) {
    // Get all of the samples for a pixel.
    int spp = sampler.SamplesPerPixel();
    std::vector<Point2f> samples;
    for (int i = 0; i < spp; ++i) {
        sampler.StartPixelSample(Point2i(0, 0), i);
        samples.push_back(sampler.Get2D());
    }

    checkElementary(name, samples, logSamples);
}

// TODO: check Halton (where the elementary intervals are (2^i, 3^j)).

TEST(PaddedSobolSampler, ElementaryIntervals) {
    for (auto rand :
         {RandomizeStrategy::None, RandomizeStrategy::Owen, RandomizeStrategy::PermuteDigits})
        for (int logSamples = 2; logSamples <= 10; ++logSamples)
            checkElementarySampler("PaddedSobolSampler",
                                   new PaddedSobolSampler(1 << logSamples, rand),
                                   logSamples);
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
