// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/filters.h>
#include <pbrt/pbrt.h>
#include <pbrt/util/math.h>
#include <pbrt/util/sampling.h>

#include <algorithm>
#include <vector>

using namespace pbrt;

TEST(Sinc, ZeroHandling) {
    Float x = 0;
    Float prev = 1;
    for (int i = 0; i < 10000; ++i) {
        Float cur = Sinc(x);
        EXPECT_LE(cur, prev);
        x = NextFloatUp(x);
        prev = cur;
    }

    x = -0;
    prev = 1;
    for (int i = 0; i < 10000; ++i) {
        Float cur = Sinc(x);
        EXPECT_LE(cur, prev);
        x = NextFloatDown(x);
        prev = cur;
    }
}

TEST(Filter, ZeroPastRadius) {
    auto makeFilters = [](const Vector2f &radius) -> std::vector<Filter> {
        return {new BoxFilter(radius), new GaussianFilter(radius),
                new MitchellFilter(radius), new LanczosSincFilter(radius),
                new TriangleFilter(radius)};
    };

    for (Vector2f r : {Vector2f(1, 1), Vector2f(1.5, .25), Vector2f(.33, 5.2),
                       Vector2f(.1, .1), Vector2f(3, 3)}) {
        for (Filter f : makeFilters(r)) {
            EXPECT_EQ(0, f.Evaluate(Point2f(0, r.y + 1e-3)));
            EXPECT_EQ(0, f.Evaluate(Point2f(r.x, r.y + 1e-3)));
            EXPECT_EQ(0, f.Evaluate(Point2f(-r.x, r.y + 1e-3)));
            EXPECT_EQ(0, f.Evaluate(Point2f(0, -r.y - 1e-3)));
            EXPECT_EQ(0, f.Evaluate(Point2f(r.x, -r.y - 1e-3)));
            EXPECT_EQ(0, f.Evaluate(Point2f(-r.x, -r.y - 1e-3)));
            EXPECT_EQ(0, f.Evaluate(Point2f(r.x + 1e-3, 0)));
            EXPECT_EQ(0, f.Evaluate(Point2f(r.x + 1e-3, r.y)));
            EXPECT_EQ(0, f.Evaluate(Point2f(r.x + 1e-3, -r.y)));
            EXPECT_EQ(0, f.Evaluate(Point2f(-r.x - 1e-3, 0)));
            EXPECT_EQ(0, f.Evaluate(Point2f(-r.x - 1e-3, r.y)));
            EXPECT_EQ(0, f.Evaluate(Point2f(-r.x - 1e-3, -r.y)));
        }
    }
}

static Float integrateFilter(Filter f) {
    Float sum = 0;
    int sqrtSamples = 256;
    int nSamples = sqrtSamples * sqrtSamples;
    Float area = 2 * f.Radius().x * 2 * f.Radius().y;
    for (Point2f u : Stratified2D(sqrtSamples, sqrtSamples)) {
        Point2f p(Lerp(u.x, -f.Radius().x, f.Radius().x),
                  Lerp(u.y, -f.Radius().y, f.Radius().y));
        sum += f.Evaluate(p);
    }
    return sum / nSamples * area;
}

TEST(Filter, Integral) {
    auto approxEqual = [](Float a, Float b) {
        if (std::max(std::abs(a), std::abs(b)) < 1e-3)
            return std::abs(a - b) < 1e-5;
        else
            return 2 * std::abs(a - b) / std::abs(a + b) < 1e-2;
    };
    auto makeFilters = [](const Vector2f &radius) -> std::vector<Filter> {
        return {new BoxFilter(radius), new GaussianFilter(radius),
                new MitchellFilter(radius), new LanczosSincFilter(radius),
                new TriangleFilter(radius)};
    };

    for (Filter f : makeFilters(Vector2f(1, 1)))
        EXPECT_TRUE(approxEqual(f.Integral(), integrateFilter(f))) << f;

    for (Filter f : makeFilters(Vector2f(2.5, 1)))
        EXPECT_TRUE(approxEqual(f.Integral(), integrateFilter(f))) << f;

    for (Filter f : makeFilters(Vector2f(1, 2.5)))
        EXPECT_TRUE(approxEqual(f.Integral(), integrateFilter(f))) << f;

    for (Filter f : makeFilters(Vector2f(3.4, 2.5)))
        EXPECT_TRUE(approxEqual(f.Integral(), integrateFilter(f))) << f;
}
