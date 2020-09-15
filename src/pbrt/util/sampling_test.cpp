// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/util/float.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/shuffle.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <numeric>

using namespace pbrt;

TEST(SampleDiscrete, Basics) {
    Float pdf;

    EXPECT_EQ(0, SampleDiscrete({Float(5)}, 0.251, &pdf));
    EXPECT_EQ(1, pdf);

    EXPECT_EQ(0, SampleDiscrete({Float(0.5), Float(0.5)}, 0., &pdf));
    EXPECT_EQ(0.5f, pdf);

    EXPECT_EQ(0, SampleDiscrete({Float(0.5), Float(0.5)}, 0.499, &pdf));
    EXPECT_EQ(0.5f, pdf);

    Float uRemapped;
    EXPECT_EQ(1, SampleDiscrete({Float(0.5), Float(0.5)}, 0.5f, &pdf, &uRemapped));
    EXPECT_EQ(0.5f, pdf);
    EXPECT_EQ(0, uRemapped);
}

TEST(SampleDiscrete, VsPiecewiseConstant1D) {
    std::vector<Float> values;
    RNG rng;
    for (int i = 0; i < 15; ++i)
        values.push_back(std::max<Float>(0, rng.Uniform<Float>() - .2f));

    PiecewiseConstant1D dist(values);

    for (int i = 0; i < 100; ++i) {
        Float u = rng.Uniform<Float>();
        Float sdPDF;
        int sdIndex = SampleDiscrete(values, u, &sdPDF);

        Float pcPDF;
        Float pcOffset = dist.Sample(u, &pcPDF);
        int pcIndex = int(pcOffset * values.size());
        pcPDF /= values.size();

        EXPECT_EQ(sdIndex, pcIndex) << u;
        EXPECT_LT(std::abs(sdPDF - pcPDF), 1e-3) << u;
    }
}

TEST(Sampling, InvertUniformHemisphere) {
    for (Point2f u : Uniform2D(1000)) {
        Vector3f v = SampleUniformHemisphere(u);
        Point2f up = InvertUniformHemisphereSample(v);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertCosineHemisphere) {
    for (Point2f u : Uniform2D(1000)) {
        Vector3f v = SampleCosineHemisphere(u);
        Point2f up = InvertCosineHemisphereSample(v);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertUniformSphere) {
    for (Point2f u : Uniform2D(1000)) {
        Vector3f v = SampleUniformSphere(u);
        Point2f up = InvertUniformSphereSample(v);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertUniformTriangle) {
    for (Point2f u : Uniform2D(1000)) {
        pstd::array<Float, 3> b = SampleUniformTriangle(u);
        Point2f up = InvertUniformTriangleSample(b);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << b[0] << ", "
                                              << b[1] << ", " << b[2] << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << b[0] << ", "
                                              << b[1] << ", " << b[2] << " -> " << up;
    }
}

TEST(Sampling, InvertUniformCone) {
    RNG rng;
    for (Point2f u : Uniform2D(1000)) {
        Float cosThetaMax = rng.Uniform<Float>();
        Vector3f v = SampleUniformCone(u, cosThetaMax);
        Point2f up = InvertUniformConeSample(v, cosThetaMax);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertUniformDiskPolar) {
    for (Point2f u : Uniform2D(1000)) {
        Point2f p = SampleUniformDiskPolar(u);
        Point2f up = InvertUniformDiskPolarSample(p);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << p << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << p << " -> " << up;
    }
}

TEST(Sampling, InvertUniformDiskConcentric) {
    for (Point2f u : Uniform2D(1000)) {
        Point2f p = SampleUniformDiskConcentric(u);
        Point2f up = InvertUniformDiskConcentricSample(p);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << p << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << p << " -> " << up;
    }
}

TEST(LowDiscrepancy, RadicalInverse) {
    for (int a = 0; a < 1024; ++a) {
        EXPECT_EQ(ReverseBits32(a) * 2.3283064365386963e-10f, RadicalInverse(0, a));
    }
}

TEST(LowDiscrepancy, GeneratorMatrix) {
    uint32_t C[32];
    uint32_t Crev[32];
    // Identity matrix, column-wise
    for (int i = 0; i < 32; ++i) {
        C[i] = 1 << i;
        Crev[i] = ReverseBits32(C[i]);
    }

    for (int a = 0; a < 128; ++a) {
        // Make sure identity generator matrix matches van der Corput
        EXPECT_EQ(a, MultiplyGenerator(C, a));
        EXPECT_EQ(RadicalInverse(0, a),
                  ReverseBits32(MultiplyGenerator(C, a)) * 2.3283064365386963e-10f);
        EXPECT_EQ(RadicalInverse(0, a), SampleGeneratorMatrix(Crev, a));
    }

    // Random / goofball generator matrix
    RNG rng;
    for (int i = 0; i < 32; ++i) {
        C[i] = rng.Uniform<uint32_t>();
        Crev[i] = ReverseBits32(C[i]);
    }
    for (int a = 0; a < 1024; ++a) {
        EXPECT_EQ(ReverseBits32(MultiplyGenerator(C, a)), MultiplyGenerator(Crev, a));
    }
}

TEST(LowDiscrepancy, Sobol) {
    // Check that float and double variants match (as float values).
    for (int i = 0; i < 256; ++i) {
        for (int dim = 0; dim < 100; ++dim) {
            EXPECT_EQ(SobolSampleFloat(i, dim, NoRandomizer()),
                      (float)SobolSampleDouble(i, dim, NoRandomizer()));
        }
    }

    // Make sure first dimension is the regular base 2 radical inverse
    for (int i = 0; i < 8192; ++i) {
        EXPECT_EQ(SobolSampleFloat(i, 0, NoRandomizer()),
                  ReverseBits32(i) * 2.3283064365386963e-10f);
    }
}

TEST(CranleyPattersonRotator, Basics) {
    auto toFixed = [](Float v) { return uint32_t(v * 0x1p+32); };
    auto fromFixed = [](uint32_t v) { return Float(v) * 0x1p-32; };
    EXPECT_EQ(0, toFixed(0));
    EXPECT_EQ(0x80000000, toFixed(0.5f));
    EXPECT_EQ(0x40000000, toFixed(0.25f));
    EXPECT_EQ(fromFixed(0x80000000), 0.5f);
    EXPECT_EQ(fromFixed(0xc0000000), 0.75f);
    for (int i = 1; i < 31; ++i) {
        Float v = 1.f / (1 << i);
        EXPECT_EQ(toFixed(v), 1u << (32 - i));
        EXPECT_EQ(fromFixed(1u << (32 - i)), v);
    }

    EXPECT_EQ(toFixed(0.5), CranleyPattersonRotator(0.5f)(0));
    EXPECT_EQ(toFixed(0.5), CranleyPattersonRotator(0.25f)(toFixed(0.25)));
    EXPECT_EQ(toFixed(0.5), CranleyPattersonRotator(toFixed(0.25f))(toFixed(0.25)));
    EXPECT_EQ(toFixed(0.75), CranleyPattersonRotator(toFixed(0.5f))(toFixed(0.25)));
    EXPECT_EQ(toFixed(0.375f), CranleyPattersonRotator(toFixed(0.25f))(toFixed(0.125)));
}

TEST(Sobol, IntervalToIndex) {
    for (int logRes = 0; logRes < 8; ++logRes) {
        int res = 1 << logRes;
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                // For each pixel sample.
                bool sawCorner = false;
                for (int s = 0; s < 16; ++s) {
                    uint64_t seqIndex = SobolIntervalToIndex(logRes, s, {x, y});
                    Point2f samp(SobolSample(seqIndex, 0, NoRandomizer()),
                                 SobolSample(seqIndex, 1, NoRandomizer()));
                    Point2f ss(samp[0] * res, samp[1] * res);
                    // Make sure all samples aren't landing at the
                    // lower-left corner.
                    if (ss == Point2f(x, y)) {
                        EXPECT_FALSE(sawCorner)
                            << "Multi corner " << Point2i(x, y) << ", res " << res
                            << ", samp " << s << ", seq index " << seqIndex << ", got "
                            << ss << ", from samp " << samp;
                        sawCorner = true;
                    }
                    // Most of the time, the Sobol sample should be within the
                    // pixel's extent. Due to rounding, it may rarely be at the
                    // upper end of the extent; the check here is written
                    // carefully to only accept points just at the upper limit
                    // but not into the next pixel.
                    EXPECT_TRUE(Point2i(x, y) == Point2i(ss) ||
                                (x == int(ss.x) && Float(y + 1) == ss.y ||
                                 Float(x + 1) == ss.x && y == int(ss.y)))
                        << "Pixel " << Point2i(x, y) << ", sample " << s << ", got "
                        << ss;
                }
            }
        }
    }
}

TEST(Sobol, IntervalToIndexRandoms) {
    RNG rng;
    for (int i = 0; i < 100000; ++i) {
        int logRes = rng.Uniform<int>(16);
        int res = 1 << logRes;
        int x = rng.Uniform<int>(res), y = rng.Uniform<int>(res);
        int s = rng.Uniform<int>(8192);

        uint64_t seqIndex = SobolIntervalToIndex(logRes, s, {x, y});
        Point2f samp(SobolSample(seqIndex, 0, NoRandomizer()),
                     SobolSample(seqIndex, 1, NoRandomizer()));
        Point2f ss(int(samp[0] * res), int(samp[1] * res));
        // Most of the time, the Sobol sample should be within the
        // pixel's extent. Due to rounding, it may rarely be at the
        // upper end of the extent; the check here is written carefully
        // to only accept points just at the upper limit but not into
        // the next pixel.
        EXPECT_TRUE(Point2i(x, y) == Point2i(ss) ||
                    (x == int(ss.x) && Float(y + 1) == ss.y ||
                     Float(x + 1) == ss.x && y == int(ss.y)))
            << "Pixel " << Point2i(x, y) << ", sample " << s << ", got " << ss;
    }
}

TEST(PiecewiseConstant1D, Continuous) {
    PiecewiseConstant1D dist({1.f, 1.f, 2.f, 4.f, 8.f});
    EXPECT_EQ(5, dist.size());

    Float pdf;
    int offset;
    EXPECT_EQ(0., dist.Sample(0., &pdf, &offset));
    EXPECT_FLOAT_EQ(dist.size() * 1. / 16., pdf);
    EXPECT_EQ(0, offset);

    // Right at the bounary between the 4 and the 8 segments.
    EXPECT_FLOAT_EQ(.8, dist.Sample(0.5, &pdf, &offset));

    // Middle of the 8 segment
    EXPECT_FLOAT_EQ(.9, dist.Sample(0.75, &pdf, &offset));
    EXPECT_FLOAT_EQ(dist.size() * 8. / 16., pdf);
    EXPECT_EQ(4, offset);

    EXPECT_FLOAT_EQ(0., dist.Sample(0., &pdf));
    EXPECT_FLOAT_EQ(1., dist.Sample(1., &pdf));
}

TEST(PiecewiseConstant1D, Range) {
    auto values = Sample1DFunction([](Float x) { return 1 + x; }, 65536, 4, -1.f, 3.f);
    PiecewiseConstant1D dist(values, -1.f, 3.f);
    // p(x) = (1+x) / 8
    // xi = int_{-1}^x p(x) ->
    // xi = 1/16 x^2 + x/8 + 1/16 ->
    // Solve 0 = 1/16 x^2 + x/8 + 1/16 - xi to sample

    for (Float u : Uniform1D(100)) {
        Float pd;
        Float xd = dist.Sample(u, &pd);

        Float t0, t1;
        ASSERT_TRUE(Quadratic(1. / 16., 1. / 8., 1. / 16 - u, &t0, &t1));
        Float xa = (t0 >= -1 && t0 <= 3) ? t0 : t1;
        Float pa = (1 + xa) / 8;

        EXPECT_LT(std::abs(xd - xa) / xa, 2e-3) << xd << " vs " << xa;
        EXPECT_LT(std::abs(pd - pa) / pa, 2e-3) << pd << " vs " << pa;
    }
}

TEST(PiecewiseConstant1D, InverseUniform) {
    std::vector<Float> values = {Float(1), Float(1), Float(1)};

    PiecewiseConstant1D dist(values);
    EXPECT_EQ(0, *dist.Invert(0));
    EXPECT_EQ(0.5, *dist.Invert(0.5));
    EXPECT_EQ(0.75, *dist.Invert(0.75));

    PiecewiseConstant1D dist2(values, -1, 3);
    EXPECT_EQ(0, *dist2.Invert(-1));
    EXPECT_EQ(0.25, *dist2.Invert(0));
    EXPECT_EQ(0.5, *dist2.Invert(1));
    EXPECT_EQ(0.75, *dist2.Invert(2));
    EXPECT_EQ(1, *dist2.Invert(3));
}

TEST(PiecewiseConstant1D, InverseGeneral) {
    std::vector<Float> values = {Float(0), Float(0.25), Float(0.5), Float(0.25)};

    PiecewiseConstant1D dist(values);
    EXPECT_EQ(0, *dist.Invert(0));
    EXPECT_EQ(1, *dist.Invert(1));
    EXPECT_EQ(0.25, *dist.Invert(0.5));
    EXPECT_EQ(0.5, *dist.Invert(0.625));
    EXPECT_EQ(0.75, *dist.Invert(0.75));
    EXPECT_FLOAT_EQ(0.825, *dist.Invert(0.825));

    PiecewiseConstant1D dist2(values, -1, 3);
    EXPECT_EQ(0, *dist2.Invert(-1));
    EXPECT_EQ(1, *dist2.Invert(3));
    EXPECT_EQ(0.25, *dist2.Invert(Lerp(0.5, -1, 3)));
    EXPECT_EQ(0.5, *dist2.Invert(Lerp(0.625, -1, 3)));
    EXPECT_EQ(0.75, *dist2.Invert(Lerp(0.75, -1, 3)));
    EXPECT_FLOAT_EQ(0.825, *dist2.Invert(Lerp(0.825, -1, 3)));
}

TEST(PiecewiseConstant1D, InverseRandoms) {
    std::vector<Float> values = {Float(0), Float(1.25), Float(0.5), Float(0.25),
                                 Float(3.7)};

    PiecewiseConstant1D dist(values);
    for (Float u : Uniform1D(100)) {
        Float v = dist.Sample(u);
        auto inv = dist.Invert(v);
        ASSERT_TRUE(inv.has_value());
        Float err = std::min(std::abs(*inv - u), std::abs(*inv - u) / u);
        EXPECT_LT(err, 1e-4) << "u " << u << " vs inv " << *inv;
    }

    PiecewiseConstant1D dist2(values, -1, 3);
    for (Float u : Uniform1D(100)) {
        Float v = dist.Sample(u);
        auto inv = dist.Invert(v);
        ASSERT_TRUE(inv.has_value());
        Float err = std::min(std::abs(*inv - u), std::abs(*inv - u) / u);
        EXPECT_LT(err, 1e-4) << "u " << u << " vs inv " << *inv;
    }
}

TEST(PiecewiseConstant2D, InverseUniform) {
    std::vector<Float> values = {Float(1), Float(1), Float(1),
                                 Float(1), Float(1), Float(1)};

    PiecewiseConstant2D dist(values, 3, 2);
    EXPECT_EQ(Point2f(0, 0), *dist.Invert(Point2f(0, 0)));
    EXPECT_EQ(Point2f(1, 1), *dist.Invert(Point2f(1, 1)));
    EXPECT_EQ(Point2f(0.5, 0.5), *dist.Invert(Point2f(0.5, 0.5)));
    EXPECT_EQ(Point2f(0.25, 0.75), *dist.Invert(Point2f(0.25, 0.75)));

    Bounds2f domain(Point2f(-1, -0.5), Point2f(3, 1.5));
    PiecewiseConstant2D dist2(values, 3, 2, domain);
    EXPECT_EQ(Point2f(0, 0), *dist2.Invert(domain.Lerp(Point2f(0, 0))));
    EXPECT_EQ(Point2f(1, 1), *dist2.Invert(domain.Lerp(Point2f(1, 1))));
    EXPECT_EQ(Point2f(0.5, 0.5), *dist2.Invert(domain.Lerp(Point2f(0.5, 0.5))));
    EXPECT_EQ(Point2f(0.25, 0.75), *dist2.Invert(domain.Lerp(Point2f(0.25, 0.75))));
}

TEST(PiecewiseConstant2D, InverseRandoms) {
    int nx = 4, ny = 5;
    std::vector<Float> values;
    RNG rng;
    for (int i = 0; i < nx * ny; ++i)
        values.push_back(rng.Uniform<Float>());

    PiecewiseConstant2D dist(values, nx, ny);
    for (Point2f u : Uniform2D(100)) {
        Point2f v = dist.Sample(u);
        auto inv = dist.Invert(v);
        ASSERT_TRUE(inv.has_value());
        Point2f err(
            std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
            std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
        EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
        EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
    }

    Bounds2f domain(Point2f(-1, -0.5), Point2f(3, 1.5));
    PiecewiseConstant2D dist2(values, nx, ny, domain);
    for (Point2f u : Uniform2D(100, 235351)) {
        Point2f v = dist2.Sample(u);
        auto inv = dist2.Invert(v);
        ASSERT_TRUE(inv.has_value());
        Point2f err(
            std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
            std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
        EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
        EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
    }
}

TEST(PiecewiseConstant2D, FromFuncLInfinity) {
    auto f = [](Float x, Float y) { return x * x * y; };
    auto values = Sample2DFunction(f, 4, 2, 1, Bounds2f(Point2f(0, 0), Point2f(1, 1)));
    PiecewiseConstant2D dSampled(values, 4, 2);

    std::vector<Float> exact = {
        Float(Sqr(0.25) * Float(0.5)), Float(Sqr(0.5) * Float(0.5)),
        Float(Sqr(0.75) * Float(0.5)), Float(Sqr(1) * Float(0.5)),
        Float(Sqr(0.25) * Float(1)),   Float(Sqr(0.5) * Float(1)),
        Float(Sqr(0.75) * Float(1)),   Float(Sqr(1) * Float(1))};
    PiecewiseConstant2D dExact(exact, 4, 2);
    PiecewiseConstant2D::TestCompareDistributions(dSampled, dExact);
}

TEST(Sampling, SphericalTriangle) {
    int count = 1024 * 1024;
    pstd::array<Point3f, 3> v = {Point3f(4, 1, 1), Point3f(-10, 3, 3),
                                 Point3f(-2, -8, 10)};
    Float A = 0.5 * Length(Cross(v[1] - v[0], v[2] - v[0]));
    Vector3f N = Normalize(Cross(v[1] - v[0], v[2] - v[0]));
    Point3f p(.5, -.4, .7);

    // Integrate this function over the projection of the triangle given by
    // |v| at the unit sphere surrounding |p|.
    auto f = [](Point3f p) { return p.x * p.y * p.z; };

    Float sphSum = 0, areaSum = 0;
    for (int i = 0; i < count; ++i) {
        Point2f u(RadicalInverse(0, i), RadicalInverse(1, i));

        Float pdf;
        pstd::array<Float, 3> bs = SampleSphericalTriangle(v, p, u, &pdf);
        Point3f pTri = bs[0] * v[0] + bs[1] * v[1] + bs[2] * v[2];
        sphSum += f(pTri) / pdf;

        pstd::array<Float, 3> ba = SampleUniformTriangle(u);
        pdf = 1 / A;
        pTri = ba[0] * v[0] + ba[1] * v[1] + (1 - ba[0] - ba[1]) * v[2];
        areaSum +=
            f(pTri) * AbsDot(N, Normalize(pTri - p)) / (pdf * DistanceSquared(p, pTri));
    }
    Float sphInt = sphSum / count;
    Float areaInt = areaSum / count;

    EXPECT_LT(std::abs(areaInt - sphInt), 1e-3);
}

TEST(Sampling, SphericalTriangleInverse) {
    RNG rng;
    auto rp = [&rng](Float low = -10, Float high = 10) {
        return Point3f(Lerp(rng.Uniform<Float>(), low, high),
                       Lerp(rng.Uniform<Float>(), low, high),
                       Lerp(rng.Uniform<Float>(), low, high));
    };

    for (int i = 0; i < 10; ++i) {
        pstd::array<Point3f, 3> v = {rp(), rp(), rp()};
        Point3f p = rp(-1, 1);
        for (int j = 0; j < 10; ++j) {
            Float pdf;
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
            pstd::array<Float, 3> bs = SampleSphericalTriangle(v, p, u, &pdf);

            Point3f pTri = bs[0] * v[0] + bs[1] * v[1] + bs[2] * v[2];

            Point2f ui = InvertSphericalTriangleSample(v, p, Normalize(pTri - p));

            auto err = [](Float a, Float ref) {
                if (ref < 1e-3)
                    return std::abs(a - ref);
                else
                    return std::abs((a - ref) / ref);
            };
            // The tolerance has to be fiarly high, unfortunately...
            EXPECT_LT(err(ui[0], u[0]), 0.025) << u << " vs inverse " << ui;
            EXPECT_LT(err(ui[1], u[1]), 0.025) << u << " vs inverse " << ui;
        }
    }
}

TEST(Sampling, SphericalQuad) {
    int count = 1024 * 1024;
    pstd::array<Point3f, 4> v = {Point3f(4, 1, 1), Point3f(6, 1, -2), Point3f(4, 4, 1),
                                 Point3f(6, 4, -2)};
    Float A = Length(v[0] - v[1]) * Length(v[0] - v[2]);
    Vector3f N = Normalize(Cross(v[1] - v[0], v[2] - v[0]));
    Point3f p(.5, -.4, .7);

    // Integrate this function over the projection of the quad given by
    // |v| at the unit sphere surrounding |p|.
    auto f = [](Point3f p) { return p.x * p.y * p.z; };

    Float sphSum = 0, areaSum = 0;
    for (Point2f u : Hammersley2D(count)) {
        Float pdf;
        Point3f pq = SampleSphericalRectangle(p, v[0], v[1] - v[0], v[2] - v[0], u, &pdf);
        sphSum += f(pq) / pdf;

        pq = Lerp(u[1], Lerp(u[0], v[0], v[1]), Lerp(u[0], v[2], v[3]));
        pdf = 1 / A;
        areaSum += f(pq) * AbsDot(N, Normalize(pq - p)) / (pdf * DistanceSquared(p, pq));
    }
    Float sphInt = sphSum / count;
    Float areaInt = areaSum / count;

    EXPECT_LT(std::abs(areaInt - sphInt), 1e-3)
        << "area " << areaInt << " sph " << sphInt;
}

TEST(Sampling, SphericalQuadInverse) {
#if 0
    LOG(WARNING) << "bits " << int(FloatToBits(0.00026721583f) - FloatToBits(0.00026713056f));
    {
    Point3f p( 5.5154743, -6.8645816, -1.2982006), s(6, 1, -2);
    Vector3f ex( 0, 3, 0), ey( -2, 0, 3);
    Point2f u(0.031906128, 0.82836914);

    Point3f pq = SampleSphericalRectangle(p, s, ex, ey, u);
    Point2f ui = InvertSphericalRectangleSample(p, s, ex, ey, pq);
    EXPECT_EQ(u, ui);
    }

    {
        Point3f p(-1.8413692, 3.8777208, 9.158957), s( 6, 4, -2);
        Vector3f ex(0, -3, 0), ey( -2, 0, 3);
        Point2f u (0.11288452, 0.40319824 );
        Point3f pq = SampleSphericalRectangle(p, s, ex, ey, u);
        Point2f ui = InvertSphericalRectangleSample(p, s, ex, ey, pq);
        EXPECT_EQ(u, ui);
    }
//CO    return;
#endif

    int count = 64 * 1024;
    Point3f v[2][2] = {{Point3f(4, 1, 1), Point3f(6, 1, -2)},
                       {Point3f(4, 4, 1), Point3f(6, 4, -2)}};

    RNG rng;
    int nTested = 0;
    for (Point2f u : Hammersley2D(count)) {
        int a = rng.Uniform<int>() & 1, b = rng.Uniform<int>() & 1;

        Point3f p(Lerp(rng.Uniform<Float>(), -10, 10),
                  Lerp(rng.Uniform<Float>(), -10, 10),
                  Lerp(rng.Uniform<Float>(), -10, 10));
        Float pdf;
        Point3f pq = SampleSphericalRectangle(p, v[a][b], v[!a][b] - v[a][b],
                                         v[a][!b] - v[a][b], u, &pdf);

        Float solidAngle = 1 / pdf;
        if (solidAngle < .01)
            continue;
        ++nTested;
        Point2f ui = InvertSphericalRectangleSample(p, v[a][b], v[!a][b] - v[a][b],
                                               v[a][!b] - v[a][b], pq);

        auto err = [](Float a, Float ref) {
            if (ref < 1e-2)
                return std::abs(a - ref);
            else
                return std::abs((a - ref) / ref);
        };
        // The tolerance has to be fairly high, unfortunately...
        // FIXME: super high for now to find the really bad cases...
        EXPECT_LT(err(ui[0], u[0]), 0.01)
            << u << " vs inverse " << ui << ", solid angle " << 1 / pdf;
        EXPECT_LT(err(ui[1], u[1]), 0.01)
            << u << " vs inverse " << ui << ", solid angle " << 1 / pdf;
    }
    EXPECT_GT(nTested, count / 2);
}

TEST(Sampling, SmoothStep) {
    Float start = SampleSmoothStep(0, 10, 20);
    // Fairly high slop since lots of values close to the start are close
    // to zero.
    EXPECT_LT(std::abs(start - 10), .1) << start;

    Float end = SampleSmoothStep(1, -10, -5);
    EXPECT_LT(std::abs(end - -5), 1e-5) << end;

    Float mid = SampleSmoothStep(0.5, 0, 1);
    // Solved this numericalla in Mathematica.
    EXPECT_LT(std::abs(mid - 0.733615), 1e-5) << mid;

    for (Float u : Uniform1D(1000)) {
        Float x = SampleSmoothStep(u, -3, 5);
        Float ratio = SmoothStep(x, -3, 5) / SmoothStepPDF(x, -3, 5);
        // SmoothStep over [-3,5] integrates to 4.
        EXPECT_LT(std::abs(ratio - 4), 1e-5) << ratio;

        auto checkErr = [](Float a, Float b) {
            Float err;
            if (std::min(std::abs(a), std::abs(b)) < 1e-2)
                err = std::abs(a - b);
            else
                err = std::abs(2 * (a - b) / (a + b));
            return err > 1e-2;
        };
        EXPECT_FALSE(checkErr(u, InvertSmoothStepSample(x, -3, 5)));
    }

    auto ss = [](Float v) { return SmoothStep(v, 0, 1); };
    auto values = Sample1DFunction(ss, 1024, 64 * 1024, 0.f, 1.f);
    PiecewiseConstant1D distrib(values);
    for (Float u : Uniform1D(100, 62351)) {
        Float cx = SampleSmoothStep(u, 0, 1);
        Float cp = SmoothStepPDF(cx, 0, 1);

        Float dp;
        Float dx = distrib.Sample(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3)
            << "Closed form = " << cx << ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3)
            << "Closed form PDF = " << cp << ", distrib PDF = " << dp;
    }
}

TEST(Sampling, Linear) {
    int nBuckets = 32;
    std::vector<int> buckets(nBuckets, 0);

    int ranges[][2] = {{0, 1}, {1, 2}, {5, 50}, {100, 0}, {75, 50}};
    for (const auto r : ranges) {
        Float f0 = r[0], f1 = r[1];
        int nSamples = 1000000;
        for (int i = 0; i < nSamples; ++i) {
            Float u = (i + .5) / nSamples;
            Float t = SampleLinear(u, f0, f1);
            ++buckets[std::min<int>(t * nBuckets, nBuckets - 1)];
        }

        for (int i = 0; i < nBuckets; ++i) {
            int expected =
                Lerp(Float(i) / (nBuckets - 1), buckets[0], buckets[nBuckets - 1]);
            EXPECT_GE(buckets[i], .99 * expected);
            EXPECT_LE(buckets[i], 1.01 * expected);
        }
    }

    auto lin = [](Float v) { return 1 + 3 * v; };
    auto values = Sample1DFunction(lin, 1024, 64 * 1024, 0.f, 1.f);
    PiecewiseConstant1D distrib(values);
    for (Float u : Uniform1D(100)) {
        Float cx = SampleLinear(u, 1, 4);
        Float cp = LinearPDF(cx, 1, 4);

        Float dp;
        Float dx = distrib.Sample(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3)
            << "Closed form = " << cx << ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3)
            << "Closed form PDF = " << cp << ", distrib PDF = " << dp;
    }

    RNG rng;
    for (Float u : Uniform1D(100)) {
        Float low = rng.Uniform<Float>() * 10;
        Float high = rng.Uniform<Float>() * 10;
        if (low < high)
            pstd::swap(low, high);
        Float x = SampleLinear(u, low, high);

        auto checkErr = [](Float a, Float b) {
            Float err;
            if (std::min(std::abs(a), std::abs(b)) < 1e-2)
                err = std::abs(a - b);
            else
                err = std::abs(2 * (a - b) / (a + b));
            return err > 1e-2;
        };

        EXPECT_FALSE(checkErr(u, InvertLinearSample(x, low, high)))
            << " u = " << u << " -> x " << x << " -> " << InvertLinearSample(x, low, high)
            << " (over " << low << " - " << high;
    }
}

TEST(Sampling, Tent) {
    // Make sure stratification is preserved at the midpoint of the
    // sampling domain.
    Float dist = std::abs(SampleTent(.501, 1) - SampleTent(.499, 1));
    EXPECT_LT(dist, .01);

    Float rad[] = {Float(1), Float(2.5), Float(.125)};
    RNG rng;
    for (Float radius : rad) {
        auto tent = [&](Float x) { return std::max<Float>(0, 1 - std::abs(x) / radius); };

        auto values = Sample1DFunction(tent, 8192, 64, -radius, radius);
        PiecewiseConstant1D distrib(values, -radius, radius);
        for (int i = 0; i < 100; ++i) {
            Float u = rng.Uniform<Float>();
            Float tx = SampleTent(u, radius);
            Float tp = TentPDF(tx, radius);

            Float dp;
            Float dx = distrib.Sample(u, &dp);
            EXPECT_LT(std::abs(tx - dx), 3e-3)
                << "Closed form = " << tx << ", distrib = " << dx;
            EXPECT_LT(std::abs(tp - dp), 3e-3)
                << "Closed form PDF = " << tp << ", distrib PDF = " << dp;

            auto checkErr = [](Float a, Float b) {
                Float err;
                if (std::min(std::abs(a), std::abs(b)) < 1e-2)
                    err = std::abs(a - b);
                else
                    err = std::abs(2 * (a - b) / (a + b));
                return err > 1e-2;
            };
            EXPECT_FALSE(checkErr(u, InvertTentSample(tx, radius)))
                << "u " << u << " radius " << radius << " x " << tx << " inverse "
                << InvertTentSample(tx, radius);
        }
    }
}

TEST(Sampling, CatmullRom) {
    std::vector<Float> nodes = {Float(0), Float(.1), Float(.4), Float(.88), Float(1)};
    std::vector<Float> values = {Float(0), Float(5), Float(2), Float(10), Float(5)};
    std::vector<Float> cdf(values.size());

    IntegrateCatmullRom(nodes, values, pstd::span<Float>(cdf));

    auto cr = [&](Float v) { return CatmullRom(nodes, values, v); };
    PiecewiseConstant1D distrib(Sample1DFunction(cr, 8192, 1024, 0.f, 1.f));
    for (Float u : Uniform1D(100)) {
        Float cp;
        Float cx = SampleCatmullRom(nodes, values, cdf, u, nullptr, &cp);

        Float dp;
        Float dx = distrib.Sample(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3)
            << "Closed form = " << cx << ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3)
            << "Closed form PDF = " << cp << ", distrib PDF = " << dp;
    }
}

TEST(Sampling, Bilinear) {
    RNG rng;
    Float quads[][4] = {{0, .5, 1.3, 4.7}, {1, 1, 1, 1}, {11, .25, 1, 20}};
    for (const auto v : quads) {
        auto bilerp = [&](Float x, Float y) {
            return ((1 - x) * (1 - y) * v[0] + x * (1 - y) * v[1] + (1 - x) * y * v[2] +
                    x * y * v[3]);
        };

        auto values = Sample2DFunction(bilerp, 1024, 1024, 16,
                                       Bounds2f(Point2f(0, 0), Point2f(1, 1)));
        PiecewiseConstant2D distrib(values, 1024, 1024);
        for (Point2f u : Uniform2D(100)) {
            Point2f pb = SampleBilinear(u, {v, 4});
            Float bp = BilinearPDF(pb, {v, 4});

            Float dp;
            Point2f pd = distrib.Sample(u, &dp);
            EXPECT_LT(std::abs(pb[0] - pd[0]), 3e-3)
                << "X: Closed form = " << pb[0] << ", distrib = " << pd[0];
            EXPECT_LT(std::abs(pb[1] - pd[1]), 3e-3)
                << "Y: Closed form = " << pb[1] << ", distrib = " << pd[1];
            EXPECT_LT(std::abs(bp - dp), 3e-3)
                << "Closed form PDF = " << bp << ", distrib PDF = " << dp;

            // Check InvertBilinear...
            Point2f up = InvertBilinearSample(pb, {v, 4});
            EXPECT_LT(std::abs(up[0] - u[0]), 3e-3)
                << "Invert failure: u[0] = " << u[0] << ", p = " << pb[0]
                << ", up = " << up[0];
            EXPECT_LT(std::abs(up[1] - u[1]), 3e-3)
                << "Invert failure: u[1] = " << u[1] << ", p = " << pb[1]
                << ", up = " << up[1];
        }
    }
}

TEST(Sampling, Logistic) {
    Float params[][3] = {{1., -Pi, Pi}, {5, 0, 3}, {.25, -5, -3}};
    for (const auto p : params) {
        Float s = p[0], a = p[1], b = p[2];
        auto logistic = [&](Float v) { return TrimmedLogistic(v, s, a, b); };

        auto values = Sample1DFunction(logistic, 8192, 16, a, b);
        PiecewiseConstant1D distrib(values, a, b);
        for (Float u : Uniform1D(100)) {
            Float cx = SampleTrimmedLogistic(u, s, a, b);
            Float cp = TrimmedLogisticPDF(cx, s, a, b);

            Float dp;
            Float dx = distrib.Sample(u, &dp);
            EXPECT_LT(std::abs(cx - dx), 3e-3)
                << "Closed form = " << cx << ", distrib = " << dx;
            EXPECT_LT(std::abs(cp - dp), 3e-3)
                << "Closed form PDF = " << cp << ", distrib PDF = " << dp;

            auto checkErr = [](Float a, Float b) {
                Float err;
                if (std::min(std::abs(a), std::abs(b)) < 1e-2)
                    err = std::abs(a - b);
                else
                    err = std::abs(2 * (a - b) / (a + b));
                return err > 1e-2;
            };
            EXPECT_FALSE(checkErr(u, InvertTrimmedLogisticSample(cx, s, a, b)))
                << "u = " << u << " -> x = " << cx << " -> ... "
                << InvertTrimmedLogisticSample(cx, s, a, b);
        }
    }
}

TEST(Sampling, TrimmedExponential) {
    Float params[][3] = {{1., 2.}, {5, 10}, {.25, 20}};
    for (const auto p : params) {
        Float c = p[0], xMax = p[1];
        auto exp = [&](Float x) { return std::exp(-c * x); };

        auto values = Sample1DFunction(exp, 32768, 16, 0, xMax);
        PiecewiseConstant1D distrib(values, 0, xMax);
        for (Float u : Uniform1D(100)) {
            Float cx = SampleTrimmedExponential(u, c, xMax);
            Float cp = TrimmedExponentialPDF(cx, c, xMax);

            Float dp;
            Float dx = distrib.Sample(u, &dp);
            EXPECT_LT(std::abs(cx - dx), 1e-2)
                << "Closed form = " << cx << ", distrib = " << dx;
            EXPECT_LT(std::abs(cp - dp), 1e-2)
                << "Closed form PDF = " << cp << ", distrib PDF = " << dp;

            auto checkErr = [](Float a, Float b) {
                Float err;
                if (std::min(std::abs(a), std::abs(b)) < 1e-2)
                    err = std::abs(a - b);
                else
                    err = std::abs(2 * (a - b) / (a + b));
                return err > 1e-2;
            };
            EXPECT_FALSE(checkErr(u, InvertTrimmedExponentialSample(cx, c, xMax)))
                << "u = " << u << " -> x = " << cx << " -> ... "
                << InvertTrimmedExponentialSample(cx, c, xMax);
        }
    }
}

TEST(Sampling, Normal) {
    Float params[][2] = {{0., 1.}, {-.5, .8}, {.25, .005}, {3.6, 1.6}};
    for (const auto p : params) {
        Float mu = p[0], sigma = p[1];
        auto normal = [&](Float x) {
            return 1 / std::sqrt(2 * Pi * sigma * sigma) *
                   std::exp(-Sqr(x - mu) / (2 * sigma * sigma));
        };
        auto values = Sample1DFunction(normal, 8192, 16, mu - 7 * sigma, mu + 7 * sigma);
        PiecewiseConstant1D distrib(values, mu - 7 * sigma, mu + 7 * sigma);

        for (Float u : Uniform1D(100)) {
            Float cx = SampleNormal(u, mu, sigma);
            Float cp = NormalPDF(cx, mu, sigma);

            Float dp;
            Float dx = distrib.Sample(u, &dp);
            EXPECT_LT(std::abs(cx - dx), 3e-3)
                << "Closed form = " << cx << ", distrib = " << dx;
            EXPECT_LT(std::abs(cp - dp) / dp, .025)
                << "Closed form PDF = " << cp << ", distrib PDF = " << dp;

            auto checkErr = [](Float a, Float b) {
                Float err;
                if (std::min(std::abs(a), std::abs(b)) < 1e-2)
                    err = std::abs(a - b);
                else
                    err = std::abs(2 * (a - b) / (a + b));
                return err > 1e-2;
            };
            EXPECT_FALSE(checkErr(u, InvertNormalSample(cx, mu, sigma)))
                << " u " << u << " -> x = " << cx << " -> "
                << InvertNormalSample(cx, mu, sigma) << " with mu " << mu << " and sigma "
                << sigma;
        }
    }
}

TEST(VarianceEstimator, Zero) {
    VarianceEstimator<Float> ve;
    for (int i = 0; i < 100; ++i)
        ve.Add(10.);
    EXPECT_EQ(ve.Variance(), 0);
}

TEST(VarianceEstimator, VsClosedForm) {
    VarianceEstimator<double> ve;
    int count = 10000;
    double sum = 0;
    for (Float u : Stratified1D(count)) {
        Float v = Lerp(u, -1, 1);
        ve.Add(v);
        sum += v;
    }

    // f(x) = 0, random variables x_i uniform in [-1,1] ->
    // variance is E[x^2] on [-1,1] == 1/3
    Float err = std::abs(ve.Variance() - 1. / 3.);
    EXPECT_LT(err, 1e-3) << ve.Variance();

    err = std::abs((sum / count - ve.Mean()) / (sum / count));
    EXPECT_LT(err, 1e-5);
}

TEST(VarianceEstimator, Merge) {
    int n = 16;
    std::vector<VarianceEstimator<double>> ve(n);

    RNG rng;
    int count = 10000;
    double sum = 0;
    for (Float u : Stratified1D(count)) {
        Float v = Lerp(u, -1, 1);
        int index = rng.Uniform<int>(ve.size());
        ve[index].Add(v);
        sum += v;
    }

    VarianceEstimator<double> veFinal;
    for (const auto &v : ve)
        veFinal.Merge(v);

    // f(x) = 0, random variables x_i uniform in [-1,1] ->
    // variance is E[x^2] on [-1,1] == 1/3
    Float err = std::abs(veFinal.Variance() - 1. / 3.);
    EXPECT_LT(err, 1e-3) << veFinal.Variance();

    err = std::abs((sum / count - veFinal.Mean()) / (sum / count));
    EXPECT_LT(err, 1e-5);
}

TEST(VarianceEstimator, MergeTwo) {
    VarianceEstimator<double> ve[2], veBoth;

    int index = 0;
    for (Float u : Stratified1D(1000)) {
        if (index++ < 400)
            ve[0].Add(u * u * u);
        else
            ve[1].Add(u * u * u);
        veBoth.Add(u * u * u);
    }

    ve[0].Merge(ve[1]);

    Float meanErr = std::abs(ve[0].Mean() - veBoth.Mean()) / veBoth.Mean();
    EXPECT_LT(meanErr, 1e-5);

    Float varError = std::abs(ve[0].Variance() - veBoth.Variance()) / veBoth.Variance();
    EXPECT_LT(varError, 1e-5);
}

// Make sure that the permute function is in fact a valid permutation.
TEST(Sampling, PermutationElement) {
    for (int len = 2; len < 1024; ++len) {
        for (int iter = 0; iter < 10; ++iter) {
            std::vector<bool> seen(len, false);

            for (int i = 0; i < len; ++i) {
                int offset = PermutationElement(i, len, iter);
                ASSERT_TRUE(offset >= 0 && offset < seen.size()) << offset;
                EXPECT_FALSE(seen[offset]);
                seen[offset] = true;
            }
        }
    }
}

TEST(WeightedReservoir, Basic) {
    RNG rng;
    constexpr int n = 16;
    float weights[n];
    for (int i = 0; i < n; ++i)
        weights[i] = .05 + Sqr(rng.Uniform<Float>());

    std::atomic<int> count[n] = {};
    int64_t nTrials = 1000000;
    ParallelFor(0, nTrials, [&](int64_t start, int64_t end) {
#ifdef PBRT_IS_MSVC
        // MSVC2019 doesn't seem to capture this as constexpr...
        constexpr int n = 16;
#endif
        RNG rng(3 * start);
        int localCount[n] = {};

        for (int64_t i = start; i < end; ++i) {
            WeightedReservoirSampler<int> wrs(i);
            int perm[n];
            for (int j = 0; j < n; ++j)
                perm[j] = j;

            for (int j = 0; j < n; ++j) {
                int index = perm[j];
                wrs.Add(index, weights[index]);
            }

            int index = wrs.GetSample();
            ASSERT_TRUE(index >= 0 && index < n);
            ++localCount[index];
        }

        for (int i = 0; i < n; ++i)
            count[i] += localCount[i];
    });

    Float sumW = std::accumulate(std::begin(weights), std::end(weights), Float(0));
    for (int i = 0; i < n; ++i) {
        EXPECT_LE(.97 * count[i] / double(nTrials), weights[i] / sumW);
        EXPECT_GE(1.03 * count[i] / double(nTrials), weights[i] / sumW);
    }
}

TEST(WeightedReservoir, MergeReservoirs) {
    RNG rng(6502);
    constexpr int n = 8;
    float weights[n];
    for (int i = 0; i < n; ++i)
        weights[i] = .01 + rng.Uniform<Float>();

    std::atomic<int> count[n] = {};
    int64_t nTrials = 1000000;
    ParallelFor(0, nTrials, [&](int64_t start, int64_t end) {
#ifdef PBRT_IS_MSVC
        // MSVC2019 doesn't seem to capture this as constexpr...
        constexpr int n = 8;
#endif
        int localCount[n] = {};

        for (int64_t i = start; i < end; ++i) {
            WeightedReservoirSampler<int> wrs0(i);
            WeightedReservoirSampler<int> wrs1(i + 1);

            for (int j = 0; j < n; ++j) {
                if (j & 1)
                    wrs0.Add(j, weights[j]);
                else
                    wrs1.Add(j, weights[j]);
            }

            wrs0.Merge(wrs1);
            ++localCount[wrs0.GetSample()];
        }

        for (int i = 0; i < n; ++i)
            count[i] += localCount[i];
    });

    Float sumW = std::accumulate(std::begin(weights), std::end(weights), Float(0));
    for (int i = 0; i < n; ++i) {
        EXPECT_LE(.98 * count[i] / double(nTrials), weights[i] / sumW);
        EXPECT_GE(1.02 * count[i] / double(nTrials), weights[i] / sumW);
    }
}

TEST(Generators, Uniform1D) {
    int count = 0;
    for (Float u : Uniform1D(120)) {
        EXPECT_TRUE(u >= 0 && u < 1);
        ++count;
    }
    EXPECT_EQ(120, count);
}

TEST(Generators, Uniform1DSeed) {
    std::vector<Float> samples;
    for (Float u : Uniform1D(1250))
        samples.push_back(u);

    // Different seed
    int i = 0;
    for (Float u : Uniform1D(samples.size(), 1)) {
        EXPECT_NE(u, samples[i]);
        ++i;
    }
}

TEST(Generators, Uniform2D) {
    int count = 0;
    for (Point2f u : Uniform2D(32)) {
        EXPECT_TRUE(u[0] >= 0 && u[0] < 1 && u[1] >= 0 && u[1] < 1);
        ++count;
    }
    EXPECT_EQ(32, count);
}

TEST(Generators, Uniform2DSeed) {
    std::vector<Point2f> samples;
    for (Point2f u : Uniform2D(83))
        samples.push_back(u);

    // Different seed
    int i = 0;
    for (Point2f u : Uniform2D(samples.size(), 1)) {
        EXPECT_NE(u, samples[i]);
        ++i;
    }
}

TEST(Generators, Uniform3D) {
    int count = 0;
    for (Point3f u : Uniform3D(32)) {
        EXPECT_TRUE(u[0] >= 0 && u[0] < 1 && u[1] >= 0 && u[1] < 1 && u[2] >= 0 &&
                    u[2] < 1);
        ++count;
    }
    EXPECT_EQ(32, count);
}

TEST(Generators, Stratified1D) {
    int count = 0, n = 128;  // power of 2
    for (Float u : Stratified1D(n)) {
        EXPECT_TRUE(u >= Float(count) / Float(n) && u < Float(count + 1) / Float(n));
        ++count;
    }
    EXPECT_EQ(n, count);
}

TEST(Generators, Stratified2D) {
    int count = 0, nx = 16, ny = 4;  // power of 2
    for (Point2f u : Stratified2D(nx, ny)) {
        int ix = count % nx;
        int iy = count / nx;
        EXPECT_TRUE(u[0] >= Float(ix) / Float(nx) && u[0] < Float(ix + 1) / Float(nx));
        EXPECT_TRUE(u[1] >= Float(iy) / Float(ny) && u[1] < Float(iy + 1) / Float(ny));
        ++count;
    }
    EXPECT_EQ(nx * ny, count);
}

TEST(Generators, Stratified3D) {
    int count = 0, nx = 4, ny = 32, nz = 8;  // power of 2
    for (Point3f u : Stratified3D(nx, ny, nz)) {
        int ix = count % nx;
        int iy = (count / nx) % ny;
        int iz = count / (nx * ny);
        EXPECT_TRUE(u[0] >= Float(ix) / Float(nx) && u[0] < Float(ix + 1) / Float(nx));
        EXPECT_TRUE(u[1] >= Float(iy) / Float(ny) && u[1] < Float(iy + 1) / Float(ny));
        EXPECT_TRUE(u[2] >= Float(iz) / Float(nz) && u[2] < Float(iz + 1) / Float(nz));
        ++count;
    }
    EXPECT_EQ(nx * ny * nz, count);
}

TEST(Generators, Hammersley2D) {
    int count = 0;
    for (Point2f u : Hammersley2D(32)) {
        EXPECT_EQ((Float)count / 32.f, u[0]);
        EXPECT_EQ(RadicalInverse(0, count), u[1]);
        ++count;
    }
    EXPECT_EQ(32, count);
}

TEST(Generators, Hammersley3D) {
    int count = 0;
    for (Point3f u : Hammersley3D(128)) {
        EXPECT_EQ((Float)count / 128.f, u[0]);
        EXPECT_EQ(RadicalInverse(0, count), u[1]);
        EXPECT_EQ(RadicalInverse(1, count), u[2]);
        ++count;
    }
    EXPECT_EQ(128, count);
}

TEST(AliasTable, Uniform) {
    Float values[5] = {2.f, 2.f, 2.f, 2.f, 2.f};

    AliasTable table(values);
    EXPECT_EQ(5, table.size());

    int count[5] = {0};
    int iters = 10000;
    for (Float u : Stratified1D(iters)) {
        Float pdf;
        int offset = table.Sample(u, &pdf);
        ASSERT_TRUE(offset >= 0 && offset < 5);
        ++count[offset];
        EXPECT_GT(pdf, .99 * 1.f / 5.f);
        EXPECT_LT(pdf, 1.01 * 1.f / 5.f);
    }

    for (int i = 0; i < 5; ++i) {
        EXPECT_GT(Float(count[i]) / iters, .99f * 1.f / 5.f);
        EXPECT_LT(Float(count[i]) / iters, 1.01f * 1.f / 5.f);
    }
}

TEST(AliasTable, Varying) {
    RNG rng;
    std::vector<Float> values;
    int n = 103;
    for (int i = 0; i < n; ++i)
        values.push_back(.25f * 2 * rng.Uniform<Float>());

    Float sum = std::accumulate(values.begin(), values.end(), Float(0));

    AliasTable table(values);
    EXPECT_EQ(n, values.size());

    std::vector<int> count(n, 0);
    int iters = 1000000;
    for (Float u : Stratified1D(iters)) {
        Float pdf;
        int offset = table.Sample(u, &pdf);
        ASSERT_TRUE(offset >= 0 && offset < n);
        ++count[offset];

        Float refPdf = values[offset] / sum;
        EXPECT_GT(pdf, .99 * refPdf);
        EXPECT_LT(pdf, 1.01 * refPdf);
    }

    for (int i = 0; i < n; ++i) {
        Float pdf = values[i] / sum;
        EXPECT_GT(Float(count[i]) / iters, .99f * pdf);
        EXPECT_LT(Float(count[i]) / iters, 1.01f * pdf);
    }
}

TEST(SummedArea, Constant) {
    Array2D<Float> v(4, 4);

    for (int y = 0; y < v.ySize(); ++y)
        for (int x = 0; x < v.xSize(); ++x)
            v(x, y) = 1;

    SummedAreaTable sat(v);

    EXPECT_EQ(1, sat.Sum(Bounds2f(Point2f(0, 0), Point2f(1, 1))));
    EXPECT_EQ(0.5, sat.Sum(Bounds2f(Point2f(0, 0), Point2f(1, 0.5))));
    EXPECT_EQ(0.5, sat.Sum(Bounds2f(Point2f(0, 0), Point2f(0.5, 1))));
    EXPECT_EQ(3. / 16., sat.Sum(Bounds2f(Point2f(0, 0), Point2f(.25, .75))));
    EXPECT_EQ(3. / 16., sat.Sum(Bounds2f(Point2f(0.5, 0.25), Point2f(0.75, 1))));
}

TEST(SummedArea, Rect) {
    Array2D<Float> v(8, 4);

    for (int y = 0; y < v.ySize(); ++y)
        for (int x = 0; x < v.xSize(); ++x)
            v(x, y) = x + y;

    SummedAreaTable sat(v);

    // All boxes that line up with boundaries exactly
    for (int y0 = 0; y0 < v.ySize(); ++y0)
        for (int x0 = 0; x0 < v.xSize(); ++x0)
            for (int y1 = y0; y1 < v.ySize(); ++y1)
                for (int x1 = x0; x1 < v.xSize(); ++x1) {
                    Float mySum = 0;
                    for (int y = y0; y < y1; ++y)
                        for (int x = x0; x < x1; ++x)
                            mySum += v(x, y);

                    Bounds2f b(Point2f(Float(x0) / v.xSize(), Float(y0) / v.ySize()),
                               Point2f(Float(x1) / v.xSize(), Float(y1) / v.ySize()));
                    EXPECT_EQ(mySum / (v.xSize() * v.ySize()), sat.Sum(b));
                }
}

TEST(SummedArea, Randoms) {
    std::array<int, 2> dims[] = {{1, 6}, {6, 1}, {12, 19}, {16, 16}, {100, 300}, {49, 2}};
    RNG rng;
    for (const auto d : dims) {
        Array2D<Float> v(d[0], d[1]);

        for (int y = 0; y < v.ySize(); ++y)
            for (int x = 0; x < v.xSize(); ++x)
                v(x, y) = rng.Uniform<int>(32);

        SummedAreaTable sat(v);

        for (int i = 0; i < 100; ++i) {
            Bounds2i bi({rng.Uniform<int>(v.xSize()), rng.Uniform<int>(v.ySize())},
                        {rng.Uniform<int>(v.xSize()), rng.Uniform<int>(v.ySize())});
            Bounds2f bf(Point2f(Float(bi.pMin.x) / Float(v.xSize()),
                                Float(bi.pMin.y) / Float(v.ySize())),
                        Point2f(Float(bi.pMax.x) / Float(v.xSize()),
                                Float(bi.pMax.y) / Float(v.ySize())));
            double ref = 0;

            for (Point2i p : bi)
                ref += v[p];
            ref /= v.xSize() * v.ySize();

            double s = sat.Sum(bf);
            if (ref != s)
                EXPECT_LT(std::abs((ref - s) / ref), 1e-3f)
                    << StringPrintf("ref %f s %f", ref, s);
        }
    }
}

TEST(SummedArea, NonCellAligned) {
    std::array<int, 2> dims[] = {{1, 6}, {6, 1}, {12, 19}, {16, 16}, {100, 300}, {49, 2}};
    RNG rng;
    for (const auto d : dims) {
        Array2D<Float> v(d[0], d[1]);

        for (int y = 0; y < v.ySize(); ++y)
            for (int x = 0; x < v.xSize(); ++x)
                v(x, y) = rng.Uniform<int>(32);

        SummedAreaTable sat(v);

        Bounds2f b({rng.Uniform<Float>(), rng.Uniform<Float>()},
                   {rng.Uniform<Float>(), rng.Uniform<Float>()});

        Float sampledSum = 0;
        int nSamples = 100000;
        for (Point2f u : Hammersley2D(nSamples)) {
            Point2f p = b.Lerp(u);
            Point2i pi(p.x * v.xSize(), p.y * v.ySize());
            sampledSum += v[pi];
        }
        Float sampledResult = sampledSum * b.Area() / nSamples;

        double s = sat.Sum(b);
        if (sampledResult != s)
            EXPECT_LT(std::abs((sampledResult - s) / sampledResult), 1e-3f)
                << StringPrintf("sampled %f s %f", sampledResult, s);
    }
}
