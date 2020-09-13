// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/shapes.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <cmath>

using namespace pbrt;

TEST(Vector2, Basics) {
    Vector2f vf(-1, 10);
    EXPECT_EQ(vf, Vector2f(Vector2i(-1, 10)));
    EXPECT_NE(vf, Vector2f(-1, 100));
    EXPECT_EQ(Vector2f(-2, 20), vf + vf);
    EXPECT_EQ(Vector2f(0, 0), vf - vf);
    EXPECT_EQ(Vector2f(-2, 20), vf * 2);
    EXPECT_EQ(Vector2f(-2, 20), 2 * vf);
    EXPECT_EQ(Vector2f(-0.5, 5), vf / 2);
    EXPECT_EQ(Vector2f(1, 10), Abs(vf));
    EXPECT_EQ(vf, Ceil(Vector2f(-1.5, 9.9)));
    EXPECT_EQ(vf, Floor(Vector2f(-.5, 10.01)));
    EXPECT_EQ(Vector2f(-20, 10), Min(vf, Vector2f(-20, 20)));
    EXPECT_EQ(Vector2f(-1, 20), Max(vf, Vector2f(-20, 20)));
    EXPECT_EQ(-1, MinComponentValue(vf));
    EXPECT_EQ(-10, MinComponentValue(-vf));
    EXPECT_EQ(10, MaxComponentValue(vf));
    EXPECT_EQ(1, MaxComponentValue(-vf));
    EXPECT_EQ(1, MaxComponentIndex(vf));
    EXPECT_EQ(0, MaxComponentIndex(-vf));
    EXPECT_EQ(0, MinComponentIndex(vf));
    EXPECT_EQ(1, MinComponentIndex(-vf));
    EXPECT_EQ(vf, Permute(vf, {0, 1}));
    EXPECT_EQ(Vector2f(10, -1), Permute(vf, {1, 0}));
    EXPECT_EQ(Vector2f(10, 10), Permute(vf, {1, 1}));
}

TEST(Vector3, Basics) {
    Vector3f vf(-1, 10, 2);
    EXPECT_EQ(vf, Vector3f(Vector3i(-1, 10, 2)));
    EXPECT_NE(vf, Vector3f(-1, 100, 2));
    EXPECT_EQ(Vector3f(-2, 20, 4), vf + vf);
    EXPECT_EQ(Vector3f(0, 0, 0), vf - vf);
    EXPECT_EQ(Vector3f(-2, 20, 4), vf * 2);
    EXPECT_EQ(Vector3f(-2, 20, 4), 2 * vf);
    EXPECT_EQ(Vector3f(-0.5, 5, 1), vf / 2);
    EXPECT_EQ(Vector3f(1, 10, 2), Abs(vf));
    EXPECT_EQ(vf, Ceil(Vector3f(-1.5, 9.9, 1.01)));
    EXPECT_EQ(vf, Floor(Vector3f(-.5, 10.01, 2.99)));
    EXPECT_EQ(Vector3f(-20, 10, 1.5), Min(vf, Vector3f(-20, 20, 1.5)));
    EXPECT_EQ(Vector3f(-1, 20, 2), Max(vf, Vector3f(-20, 20, 0)));
    EXPECT_EQ(-1, MinComponentValue(vf));
    EXPECT_EQ(-10, MinComponentValue(-vf));
    EXPECT_EQ(10, MaxComponentValue(vf));
    EXPECT_EQ(1, MaxComponentValue(-vf));
    EXPECT_EQ(1, MaxComponentIndex(vf));
    EXPECT_EQ(0, MaxComponentIndex(-vf));
    EXPECT_EQ(0, MinComponentIndex(vf));
    EXPECT_EQ(1, MinComponentIndex(-vf));
    EXPECT_EQ(vf, Permute(vf, {0, 1, 2}));
    EXPECT_EQ(Vector3f(10, -1, 2), Permute(vf, {1, 0, 2}));
    EXPECT_EQ(Vector3f(2, -1, 10), Permute(vf, {2, 0, 1}));
    EXPECT_EQ(Vector3f(10, 10, -1), Permute(vf, {1, 1, 0}));
}

TEST(Vector, AngleBetween) {
    EXPECT_EQ(0, AngleBetween(Vector3f(1, 0, 0), Vector3f(1, 0, 0)));

    EXPECT_LT(std::abs(AngleBetween(Vector3f(0, 0, 1), Vector3f(0, 0, -1))) - Pi, 1e-7);
    EXPECT_LT(std::abs(AngleBetween(Vector3f(1, 0, 0), Vector3f(0, 1, 0))) - Pi / 2,
              1e-7);

    Vector3f x = Normalize(Vector3f(1, -3, 10));
    EXPECT_EQ(0, AngleBetween(x, x));
    EXPECT_LT(std::abs(AngleBetween(x, -x) - Pi), 3e-7);

    Float maxErr = 0, sumErr = 0;
    for (int i = 0; i < 100000; ++i) {
        RNG rng(i);
        Vector3f a = Normalize(Vector3f(-1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>()));
        Vector3f b = Normalize(Vector3f(-1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>()));

        Vector3<double> ad(a), bd(b);
        ad = Normalize(ad);
        bd = Normalize(bd);

        Float v[2] = {Float(std::acos(Dot(ad, bd))), AngleBetween(a, b)};
        Float err = std::abs(v[0] - v[1]) / v[0];
        maxErr = std::max(err, maxErr);
        sumErr += err;
        EXPECT_LT(err, 5e-6) << v[0] << "vs " << v[1] << ", a: " << a << ", b: " << b;
    }
    // CO    LOG(WARNING) << "MAXERR " << maxErr << ", sum " << sumErr;
    maxErr = 0;
    sumErr = 0;

    for (int i = 0; i < 100000; ++i) {
        RNG rng(i + 10000000);
        Vector3f a = Normalize(Vector3f(-1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>()));
        Vector3f b = Normalize(Vector3f(-1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>(),
                                        -1 + 2 * rng.Uniform<Float>()));
        // Make them face in opposite-ish directions
        a.x = std::copysign(a.x, -b.x);
        a.y = std::copysign(a.y, -b.y);
        a.z = std::copysign(a.z, -b.z);

        Vector3<double> ad(a), bd(b);
        ad = Normalize(ad);
        bd = Normalize(bd);

        Float v[2] = {Float(std::acos(Dot(ad, bd))), AngleBetween(a, b)};
        Float err = std::abs(v[0] - v[1]) / v[0];
        maxErr = std::max(err, maxErr);
        sumErr += err;
        EXPECT_LT(err, 5e-6) << v[0] << "vs " << v[1] << ", a: " << a << ", b: " << b;
    }
    // CO    LOG(WARNING) << "MAXERR " << maxErr << ", sum " << sumErr;

    Vector3f a(1, 1, 1), b(-1, -1.0001, -1);
    a = Normalize(a);
    b = Normalize(b);
    Vector3<long double> ad(1, 1, 1), bd(-1, -1.0001, -1);
    ad = Normalize(ad);
    bd = Normalize(bd);

    Float naive = SafeACos(Dot(a, b));
    Float precise = std::acos(Clamp(Dot(ad, bd), -1, 1));
    Float abet = AngleBetween(a, b);
    Float old = Pi - 2 * SafeASin(Length(a + b) / 2);
    EXPECT_EQ(abet, precise) << StringPrintf("vs naive %f", naive);
    // CO    LOG(WARNING) << StringPrintf("naive %f (err %f), abet %f (err %f)
    // old %f (err %f)", CO                                 naive,
    // std::abs(naive
    // - precise) / precise, CO                                 abet,
    // std::abs(abet - precise) / precise, CO old, std::abs(old - precise) /
    // precise);
}

TEST(Vector, CoordinateSystem) {
    // Duff et al 2017 footnote 1
    auto error = [](Vector3f a, Vector3f b, Vector3f c) {
        return (Sqr(Length(a) - 1) + Sqr(Length(b) - 1) + Sqr(Length(c) - 1) +
                Sqr(Dot(a, b)) + Sqr(Dot(b, c)) + Sqr(Dot(c, a))) /
               6;
    };

    // Coordinate axes.
    Vector3f a, b;
    for (Vector3f v : {Vector3f(1, 0, 0), Vector3f(-1, 0, 0), Vector3f(0, 1, 0),
                       Vector3f(0, -1, 0), Vector3f(0, 0, 1), Vector3f(0, 0, -1)}) {
        CoordinateSystem(v, &a, &b);
        for (int c = 0; c < 3; ++c) {
            if (v[c] != 0) {
                EXPECT_EQ(0, a[c]);
                EXPECT_EQ(0, b[c]);
            }
        }
    }

    // Bad vectors from Duff et al
    for (Vector3f v : {Vector3f(0.00038527316, 0.00038460016, -0.99999988079),
                       Vector3f(-0.00019813581, -0.00008946839, -0.99999988079)}) {
        CoordinateSystem(v, &a, &b);
        EXPECT_LT(error(v, a, b), 1e-10);
    }

    // Random vectors
    RNG rng;
    for (int i = 0; i < 1000; ++i) {
        Point2f u = {rng.Uniform<Float>(), rng.Uniform<Float>()};
        Vector3f v = SampleUniformSphere(u);
        CoordinateSystem(v, &a, &b);
        EXPECT_LT(error(v, a, b), 1e-10);
    }
}

TEST(Bounds2, IteratorBasic) {
    Bounds2i b{{0, 1}, {2, 3}};
    Point2i e[] = {{0, 1}, {1, 1}, {0, 2}, {1, 2}};
    int offset = 0;
    for (auto p : b) {
        EXPECT_LT(offset, PBRT_ARRAYSIZE(e));
        EXPECT_EQ(e[offset], p) << "offset = " << offset;
        ++offset;
    }
}

TEST(Bounds2, IteratorDegenerate) {
    Bounds2i b{{0, 0}, {0, 10}};
    for (auto p : b) {
        // This loop should never run.
        bool reached = true;
        EXPECT_FALSE(reached) << "p = " << p;
        break;
    }

    Bounds2i b2{{0, 0}, {4, 0}};
    for (auto p : b2) {
        // This loop should never run.
        bool reached = true;
        EXPECT_FALSE(reached) << "p = " << p;
        break;
    }

    Bounds2i b3;
    for (auto p : b3) {
        // This loop should never run.
        bool reached = true;
        EXPECT_FALSE(reached) << "p = " << p;
        break;
    }
}

TEST(Bounds3, PointDistance) {
    {
        Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));

        // Points inside the bounding box or on faces
        EXPECT_EQ(0., Distance(Point3f(.5, .5, .5), b));
        EXPECT_EQ(0., Distance(Point3f(0, 1, 1), b));
        EXPECT_EQ(0., Distance(Point3f(.25, .8, 1), b));
        EXPECT_EQ(0., Distance(Point3f(0, .25, .8), b));
        EXPECT_EQ(0., Distance(Point3f(.7, 0, .8), b));

        // Aligned with the plane of one of the faces
        EXPECT_EQ(5., Distance(Point3f(6, 1, 1), b));
        EXPECT_EQ(10., Distance(Point3f(0, -10, 1), b));

        // 2 of the dimensions inside the box's extent
        EXPECT_EQ(2., Distance(Point3f(0.5, 0.5, 3), b));
        EXPECT_EQ(3., Distance(Point3f(0.5, 0.5, -3), b));
        EXPECT_EQ(2., Distance(Point3f(0.5, 3, 0.5), b));
        EXPECT_EQ(3., Distance(Point3f(0.5, -3, 0.5), b));
        EXPECT_EQ(2., Distance(Point3f(3, 0.5, 0.5), b));
        EXPECT_EQ(3., Distance(Point3f(-3, 0.5, 0.5), b));

        // General points
        EXPECT_EQ(3 * 3 + 7 * 7 + 10 * 10, DistanceSquared(Point3f(4, 8, -10), b));
        EXPECT_EQ(6 * 6 + 10 * 10 + 7 * 7, DistanceSquared(Point3f(-6, -10, 8), b));
    }

    {
        // A few with a more irregular box, just to be sure
        Bounds3f b(Point3f(-1, -3, 5), Point3f(2, -2, 18));
        EXPECT_EQ(0., Distance(Point3f(-.99, -2, 5), b));
        EXPECT_EQ(2 * 2 + 6 * 6 + 4 * 4, DistanceSquared(Point3f(-3, -9, 22), b));
    }
}

TEST(Bounds2, Union) {
    Bounds2f a(Point2f(-10, -10), Point2f(0, 20));
    Bounds2f b;  // degenerate
    Bounds2f c = Union(a, b);
    EXPECT_EQ(a, c);

    EXPECT_EQ(b, Union(b, b));

    Bounds2f d(Point2f(-15, 10));
    Bounds2f e = Union(a, d);
    EXPECT_EQ(Bounds2f(Point2f(-15, -10), Point2f(0, 20)), e);
}

TEST(Bounds3, Union) {
    Bounds3f a(Point3f(-10, -10, 5), Point3f(0, 20, 10));
    Bounds3f b;  // degenerate
    Bounds3f c = Union(a, b);
    EXPECT_EQ(a, c);

    EXPECT_EQ(b, Union(b, b));

    Bounds3f d(Point3f(-15, 10, 30));
    Bounds3f e = Union(a, d);
    EXPECT_EQ(Bounds3f(Point3f(-15, -10, 5), Point3f(0, 20, 30)), e);
}

TEST(EqualArea, Randoms) {
    for (Point2f u : Uniform2D(100)) {
        Vector3f v = SampleUniformSphere(u);
        Point2f c = EqualAreaSphereToSquare(v);
        Vector3f vp = EqualAreaSquareToSphere(c);
        EXPECT_TRUE(Length(vp) > 0.9999 && Length(vp) < 1.0001) << Length(vp);
        EXPECT_GT(Dot(v, vp), 0.9999) << v;
    }
}

TEST(EqualArea, RemapEdges) {
    auto checkClose = [&](Point2f a, Point2f b) {
        Vector3f av = EqualAreaSquareToSphere(a);
        b = WrapEqualAreaSquare(b);
        Vector3f bv = EqualAreaSquareToSphere(b);
        EXPECT_GT(Dot(av, bv), .99);
    };

    checkClose(Point2f(.25, .01), Point2f(.25, -.01));
    checkClose(Point2f(.89, .01), Point2f(.89, -.01));

    checkClose(Point2f(.25, .99), Point2f(.25, 1.01));
    checkClose(Point2f(.89, .99), Point2f(.89, 1.01));

    checkClose(Point2f(.01, .66), Point2f(-.01, .66));
    checkClose(Point2f(.01, .15), Point2f(-.01, .15));

    checkClose(Point2f(.99, .66), Point2f(1.01, .66));
    checkClose(Point2f(.99, .15), Point2f(1.01, .15));

    checkClose(Point2f(.01, .01), Point2f(-.01, -.01));
    checkClose(Point2f(.99, .01), Point2f(1.01, -.01));
    checkClose(Point2f(.01, .99), Point2f(-.01, 1.01));
    checkClose(Point2f(.99, .99), Point2f(1.01, 1.01));
}

DirectionCone RandomCone(RNG &rng) {
    Vector3f w = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
    return DirectionCone(w, -1 + 2 * rng.Uniform<Float>());
}

TEST(DirectionCone, UnionBasics) {
    // First encloses second
    DirectionCone c = Union(DirectionCone(Vector3f(0, 0, 1), std::cos(Pi / 2)),
                            DirectionCone(Vector3f(.1, .1, 1), std::cos(.1)));
    EXPECT_EQ(c.w, Vector3f(0, 0, 1));
    EXPECT_EQ(c.cosTheta, std::cos(Pi / 2));

    // Second encloses first
    c = Union(DirectionCone(Vector3f(.1, .1, 1), std::cos(.1)),
              DirectionCone(Vector3f(0, 0, 1), std::cos(Pi / 2)));
    EXPECT_EQ(c.w, Vector3f(0, 0, 1));
    EXPECT_EQ(c.cosTheta, std::cos(Pi / 2));

    // Same direction, first wider
    Vector3f w(1, .5, -.25);
    c = Union(DirectionCone(w, std::cos(.12)), DirectionCone(w, std::cos(.03)));
    EXPECT_EQ(Normalize(w), c.w);
    EXPECT_FLOAT_EQ(std::cos(.12), c.cosTheta);

    // Same direction, second wider
    c = Union(DirectionCone(w, std::cos(.1)), DirectionCone(w, std::cos(.2)));
    EXPECT_EQ(Normalize(w), c.w);
    EXPECT_FLOAT_EQ(std::cos(.2), c.cosTheta);

    // Exactly pointing in opposite directions and covering the sphere when
    // it's all said and done.
    c = Union(DirectionCone(Vector3f(-1, -1, -1), std::cos(Pi / 2)),
              DirectionCone(Vector3f(1, 1, 1), std::cos(Pi / 2)));
    EXPECT_EQ(c.cosTheta, std::cos(Pi));

    // Basically opposite and a bit more than pi/2: should also be the
    // whole sphere.
    c = Union(DirectionCone(Vector3f(-1, -1, -1), std::cos(1.01 * Pi / 2)),
              DirectionCone(Vector3f(1.001, 1, 1), std::cos(1.01 * Pi / 2)));
    EXPECT_EQ(c.cosTheta, std::cos(Pi));

    // Narrow and at right angles; angle should be their midpoint
    c = Union(DirectionCone(Vector3f(1, 0, 0), std::cos(1e-3)),
              DirectionCone(Vector3f(0, 1, 0), std::cos(1e-3)));
    EXPECT_FLOAT_EQ(1, Dot(c.w, Normalize(Vector3f(1, 1, 0))));
    EXPECT_LT(std::abs(std::cos((Pi / 2 + 2e-3) / 2) - c.cosTheta), 1e-3);
}

TEST(DirectionCone, UnionRandoms) {
    RNG rng(16);

    for (int i = 0; i < 100; ++i) {
        DirectionCone a = RandomCone(rng), b = RandomCone(rng);
        DirectionCone c = Union(a, b);

        for (int j = 0; j < 100; ++j) {
            Vector3f w =
                SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
            if (Inside(a, w) || Inside(b, w))
                EXPECT_TRUE(Inside(c, w))
                    << "a: " << a << ", b: " << b << ", union: " << c << ", w: " << w;
        }
    }
}

TEST(DirectionCone, BoundBounds) {
    Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));

    // Point inside the bbox
    DirectionCone c = BoundSubtendedDirections(b, Point3f(.1, .2, .3));
    EXPECT_EQ(std::cos(Pi), c.cosTheta);

    // Outside, .5 units away in the middle (so using the direction to the
    // center gives the best bound).
    //
    // tan theta = (sqrt(.5^2 + .5^2)) / .5
    c = BoundSubtendedDirections(b, Point3f(-.5, .5, .5));
    Float theta = std::acos(c.cosTheta);
    Float precise = std::atan(std::sqrt(.5 * .5 + .5 * .5) / .5);
    // Make sure the returned bound isn't too small.
    EXPECT_GE(theta, 1.0001 * precise);
    // It's fine for it to be a bit big (as it is in practice due to
    // approximations for performance), but it shouldn't be too big.
    EXPECT_LT(theta, 1.1 * precise);

    RNG rng(512);
    for (int i = 0; i < 1000; ++i) {
        Bounds3f b(
            Point3f(Lerp(rng.Uniform<Float>(), -1, 1), Lerp(rng.Uniform<Float>(), -1, 1),
                    Lerp(rng.Uniform<Float>(), -1, 1)),
            Point3f(Lerp(rng.Uniform<Float>(), -1, 1), Lerp(rng.Uniform<Float>(), -1, 1),
                    Lerp(rng.Uniform<Float>(), -1, 1)));

        Point3f p(Lerp(rng.Uniform<Float>(), -4, 4), Lerp(rng.Uniform<Float>(), -4, 4),
                  Lerp(rng.Uniform<Float>(), -4, 4));

        c = BoundSubtendedDirections(b, p);
        if (Inside(p, b))
            EXPECT_EQ(std::cos(Pi), c.cosTheta);
        else {
            Vector3f wx, wy;
            CoordinateSystem(c.w, &wx, &wy);
            for (int j = 0; j < 1000; ++j) {
                Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
                Vector3f w = SampleUniformSphere(u);
                Ray r(p, w);
                bool hit = b.IntersectP(r.o, r.d);
                if (hit)
                    EXPECT_TRUE(Inside(c, w));
                if (!Inside(c, w))
                    EXPECT_FALSE(hit);
            }
        }
    }
}

TEST(DirectionCone, VectorInCone) {
    RNG rng;
    for (int i = 0; i < 100; ++i) {
        DirectionCone dc = RandomCone(rng);

        for (int j = 0; j < 100; ++j) {
            Vector3f wRandom =
                SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
            Vector3f wClosest = dc.ClosestVectorInCone(wRandom);

            if (Inside(dc, wRandom))
                EXPECT_GT(Dot(wClosest, wRandom), .999);
            else {
                // Uniformly sample the circle at the cone's boundary and
                // keep the vector that's closest to wRandom.
                Float sinTheta = SafeSqrt(1 - dc.cosTheta * dc.cosTheta);
                Frame f = Frame::FromZ(dc.w);

                Vector3f wBest;
                Float bestDot = -1;
                const int nk = 1000;
                for (int k = 0; k < nk; ++k) {
                    Float phi = (k + .5) / nk * 2 * Pi;
                    Vector3f w = SphericalDirection(sinTheta, dc.cosTheta, phi);
                    w = f.FromLocal(w);
                    if (Dot(w, wRandom) > bestDot) {
                        wBest = w;
                        bestDot = Dot(w, wRandom);
                    }
                }
                EXPECT_GT(Dot(wBest, wClosest), .999)
                    << wBest << " vs " << wClosest << ", dot " << Dot(wBest, wClosest);
            }
        }
    }
}

TEST(SphericalTriangleArea, Basics) {
    {
        Float a = SphericalTriangleArea(Vector3f(1, 0, 0), Vector3f(0, 1, 0),
                                        Vector3f(0, 0, 1));
        EXPECT_TRUE(a >= .9999 * Pi / 2 && a <= 1.00001 * Pi / 2) << a;
    }

    {
        Float a = SphericalTriangleArea(Vector3f(1, 0, 0), Normalize(Vector3f(1, 1, 0)),
                                        Vector3f(0, 0, 1));
        EXPECT_TRUE(a >= .9999 * Pi / 4 && a <= 1.00001 * Pi / 4) << a;
    }

    // Random rotations
    RNG rng;
    for (int i = 0; i < 100; ++i) {
        Vector3f axis = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        Float theta = 2 * Pi * rng.Uniform<Float>();
        Transform t = Rotate(theta, axis);
        Vector3f va = t(Vector3f(1, 0, 0));
        Vector3f vb = t(Vector3f(0, 1, 0));
        Vector3f vc = t(Vector3f(0, 0, 1));
        Float a = SphericalTriangleArea(va, vb, vc);
        EXPECT_TRUE(a >= .9999 * Pi / 2 && a <= 1.0001 * Pi / 2) << a;
    }

    for (int i = 0; i < 100; ++i) {
        Vector3f axis = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        Float theta = 2 * Pi * rng.Uniform<Float>();
        Transform t = Rotate(theta, axis);
        Vector3f va = t(Vector3f(1, 0, 0));
        Vector3f vb = t(Normalize(Vector3f(1, 1, 0)));
        Vector3f vc = t(Vector3f(0, 0, 1));
        Float a = SphericalTriangleArea(va, vb, vc);
        EXPECT_TRUE(a >= .9999 * Pi / 4 && a <= 1.0001 * Pi / 4) << a;
    }
}

TEST(SphericalTriangleArea, RandomSampling) {
    for (int i = 0; i < 100; ++i) {
        RNG rng(i);
        Vector3f a = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        Vector3f b = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        Vector3f c = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});

        Vector3f axis = Normalize(a + b + c);
        Frame frame = Frame::FromZ(axis);
        Float cosTheta = std::min({Dot(a, axis), Dot(b, axis), Dot(c, axis)});

        Float area = SphericalTriangleArea(a, b, c);
        bool sampleSphere = area > Pi;
        int sqrtN = 200;
        int count = 0;
        for (Point2f u : Hammersley2D(sqrtN * sqrtN)) {
            Vector3f v;
            if (sampleSphere)
                v = SampleUniformSphere(u);
            else {
                v = SampleUniformCone(u, cosTheta);
                v = frame.FromLocal(v);
            }

            if (IntersectTriangle(Ray(Point3f(0, 0, 0), v), Infinity, Point3f(a),
                                  Point3f(b), Point3f(c)))
                ++count;
        }

        Float pdf = sampleSphere ? UniformSpherePDF() : UniformConePDF(cosTheta);
        Float estA = Float(count) / (sqrtN * sqrtN * pdf);

        Float error = std::abs((estA - area) / area);
        EXPECT_LT(error, 0.035f) << "Area " << area << ", estimate " << estA
                                 << ", va = " << a << ", vb = " << b << ", vc = " << c;
    }
}

TEST(PointVector, Interval) {
    // This is really just to make sure that various expected things
    // compile in the first place when using the interval variants of
    // these...
    Point3fi p(1, 2, 3), q(6, 9, 1);
    Vector3fi v(4, 5, 6);

    p += v;
    p = (p - v);
    p = p + 4 * v;
    FloatInterval d = Dot(v, v);
    d = DistanceSquared(p, q);
    d = Distance(p, q);

#if 0
    v = Floor(v);
    p = Ceil(p);

    // These require a little more work since Min(Interval<T>, Interval<T>)
    // ends up returning T...
    v = Min(v, v);
    p = Max(p, q);
#endif

    Vector3fi vv = Cross(v, v);
}
