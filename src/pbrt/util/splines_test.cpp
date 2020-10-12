// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/splines.h>
#include <pbrt/util/vecmath.h>

#include <array>

using namespace pbrt;

TEST(Spline, BezierBounds) {
    // Simple
    std::array<Point3f, 4> cp;
    Bounds3f b = BoundCubicBezier(pstd::MakeConstSpan(cp), 0.f, 1.f);
    b = Expand(b, 1e-3 * Length(b.Diagonal()));
    for (Float u = 0; u <= 1.f; u += 1.f / 1024.f) {
        Point3f p = EvaluateCubicBezier(pstd::MakeConstSpan(cp), u);
        EXPECT_TRUE(Inside(p, b)) << p << " @ u = " << u << " not in " << b;
    }

    // Randoms...
    RNG rng;
    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 3; ++c)
                cp[j][c] = -5.f + 10.f * rng.Uniform<Float>();

        Bounds3f b = BoundCubicBezier(pstd::MakeConstSpan(cp), 0.f, 1.f);
        b = Expand(b, 1e-3 * Length(b.Diagonal()));
        for (Float u = 0; u <= 1.f; u += 1.f / 1024.f) {
            Point3f p = EvaluateCubicBezier(pstd::MakeConstSpan(cp), u);
            EXPECT_TRUE(Inside(p, b)) << p << " @ u = " << u << " not in " << b;
        }
    }
}
