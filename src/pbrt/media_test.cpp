// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/media.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>

using namespace pbrt;

TEST(HenyeyGreenstein, SamplingMatch) {
    RNG rng;
    for (float g = -.75; g <= 0.75; g += 0.25) {
        HGPhaseFunction hg(g);
        for (int i = 0; i < 100; ++i) {
            Vector3f wo =
                SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
            Point2f u{rng.Uniform<Float>(), rng.Uniform<Float>()};
            auto ps = hg.Sample_p(wo, u);
            EXPECT_TRUE(ps.has_value());
            // Phase function is normalized, and the sampling method should be
            // exact.
            EXPECT_EQ(ps->p, ps->pdf);
            EXPECT_NEAR(ps->p, hg.p(wo, ps->wi), 1e-4f) << "Failure with g = " << g;
        }
    }
}

TEST(HenyeyGreenstein, SamplingOrientationForward) {
    HGPhaseFunction hg(0.95);
    Vector3f wo(-1, 0, 0);
    int nForward = 0, nBackward = 0;
    for (Point2f u : Uniform2D(100)) {
        auto ps = hg.Sample_p(wo, u);
        EXPECT_TRUE(ps.has_value());
        if (ps->wi.x > 0)
            ++nForward;
        else
            ++nBackward;
    }
    // With g = 0.95, almost all of the samples should have wi.x > 0.
    EXPECT_GE(nForward, 10 * nBackward);
}

TEST(HenyeyGreenstein, SamplingOrientationBackward) {
    HGPhaseFunction hg(-0.95);
    Vector3f wo(-1, 0, 0);
    int nForward = 0, nBackward = 0;
    for (Point2f u : Uniform2D(100)) {
        auto ps = hg.Sample_p(wo, u);
        EXPECT_TRUE(ps.has_value());
        if (ps->wi.x > 0)
            ++nForward;
        else
            ++nBackward;
    }
    // With g = -0.95, almost all of the samples should have wi.x < 0.
    EXPECT_GE(nBackward, 10 * nForward);
}

TEST(HenyeyGreenstein, Normalized) {
    RNG rng;
    for (float g = -.75; g <= 0.75; g += 0.25) {
        HGPhaseFunction hg(g);
        Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        Float sum = 0;
        int sqrtSamples = 64;
        int nSamples = sqrtSamples * sqrtSamples;
        for (Point2f u : Stratified2D(sqrtSamples, sqrtSamples)) {
            Vector3f wi = SampleUniformSphere(u);
            sum += hg.p(wo, wi);
        }
        // Phase function should integrate to 1/4pi.
        EXPECT_NEAR(sum / nSamples, 1. / (4. * Pi), 1e-3f);
    }
}

TEST(HenyeyGreenstein, g) {
    RNG rng;
    for (float g = -.75; g <= 0.75; g += 0.25) {
        HGPhaseFunction hg(g);
        Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        Float sum = 0;
        int sqrtSamples = 64;
        int nSamples = sqrtSamples * sqrtSamples;
        for (Point2f u : Stratified2D(sqrtSamples, sqrtSamples)) {
            Vector3f wi = SampleUniformSphere(u);
            // Negate dot to match direction convention
            sum += hg.p(wo, wi) * -Dot(wo, wi);
        }
        Float gEst = sum / (nSamples * UniformSpherePDF());
        EXPECT_NEAR(g, gEst, .01);
    }
}
