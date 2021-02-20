// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <pbrt/pbrt.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/vecmath.h>

#include <vector>

using namespace pbrt;

TEST(Pow2, Basics) {
    for (int i = 0; i < 32; ++i) {
        uint32_t ui = 1u << i;
        EXPECT_EQ(true, IsPowerOf2(ui));
        if (ui > 1) {
            EXPECT_EQ(false, IsPowerOf2(ui + 1));
        }
        if (ui > 2) {
            EXPECT_EQ(false, IsPowerOf2(ui - 1));
        }
    }
}

TEST(RoundUpPow2, Basics) {
    EXPECT_EQ(RoundUpPow2(7), 8);
    for (int i = 1; i < (1 << 24); ++i)
        if (IsPowerOf2(i))
            EXPECT_EQ(RoundUpPow2(i), i);
        else
            EXPECT_EQ(RoundUpPow2(i), 1 << (Log2Int(i) + 1));

    for (int64_t i = 1; i < (1 << 24); ++i)
        if (IsPowerOf2(i))
            EXPECT_EQ(RoundUpPow2(i), i);
        else
            EXPECT_EQ(RoundUpPow2(i), 1 << (Log2Int(i) + 1));

    for (int i = 0; i < 30; ++i) {
        int v = 1 << i;
        EXPECT_EQ(RoundUpPow2(v), v);
        if (v > 2)
            EXPECT_EQ(RoundUpPow2(v - 1), v);
        EXPECT_EQ(RoundUpPow2(v + 1), 2 * v);
    }

    for (int i = 0; i < 62; ++i) {
        int64_t v = 1ll << i;
        EXPECT_EQ(RoundUpPow2(v), v);
        if (v > 2)
            EXPECT_EQ(RoundUpPow2(v - 1), v);
        EXPECT_EQ(RoundUpPow2(v + 1), 2 * v);
    }
}

TEST(Morton2, Basics) {
    uint16_t x = 0b01010111, y = 0b11000101;
    uint32_t m = EncodeMorton2(x, y);
    EXPECT_EQ(m, 0b1011000100110111);

#if 0
    for (int x = 0; x <= 65535; ++x)
        for (int y = 0; y <= 65535; ++y) {
            uint32_t m = EncodeMorton2(x, y);
            uint16_t xp, yp;
            DecodeMorton2(m, &xp, &yp);

            EXPECT_EQ(x, xp);
            EXPECT_EQ(y, yp);
        }
#endif

    RNG rng(12351);
    for (int i = 0; i < 100000; ++i) {
        uint32_t x = rng.Uniform<uint32_t>();
        uint32_t y = rng.Uniform<uint32_t>();
        uint64_t m = EncodeMorton2(x, y);

        uint32_t xp, yp;
        DecodeMorton2(m, &xp, &yp);
        EXPECT_EQ(x, xp);
        EXPECT_EQ(y, yp);
    }
}

TEST(Math, Pow) {
    EXPECT_EQ(Pow<0>(2.f), 1 << 0);
    EXPECT_EQ(Pow<1>(2.f), 1 << 1);
    EXPECT_EQ(Pow<2>(2.f), 1 << 2);
    // Test remainder of pow template powers to 29
    EXPECT_EQ(Pow<3>(2.f), 1 << 3);
    EXPECT_EQ(Pow<4>(2.f), 1 << 4);
    EXPECT_EQ(Pow<5>(2.f), 1 << 5);
    EXPECT_EQ(Pow<6>(2.f), 1 << 6);
    EXPECT_EQ(Pow<7>(2.f), 1 << 7);
    EXPECT_EQ(Pow<8>(2.f), 1 << 8);
    EXPECT_EQ(Pow<9>(2.f), 1 << 9);
    EXPECT_EQ(Pow<10>(2.f), 1 << 10);
    EXPECT_EQ(Pow<11>(2.f), 1 << 11);
    EXPECT_EQ(Pow<12>(2.f), 1 << 12);
    EXPECT_EQ(Pow<13>(2.f), 1 << 13);
    EXPECT_EQ(Pow<14>(2.f), 1 << 14);
    EXPECT_EQ(Pow<15>(2.f), 1 << 15);
    EXPECT_EQ(Pow<16>(2.f), 1 << 16);
    EXPECT_EQ(Pow<17>(2.f), 1 << 17);
    EXPECT_EQ(Pow<18>(2.f), 1 << 18);
    EXPECT_EQ(Pow<19>(2.f), 1 << 19);
    EXPECT_EQ(Pow<20>(2.f), 1 << 20);
    EXPECT_EQ(Pow<21>(2.f), 1 << 21);
    EXPECT_EQ(Pow<22>(2.f), 1 << 22);
    EXPECT_EQ(Pow<23>(2.f), 1 << 23);
    EXPECT_EQ(Pow<24>(2.f), 1 << 24);
    EXPECT_EQ(Pow<25>(2.f), 1 << 25);
    EXPECT_EQ(Pow<26>(2.f), 1 << 26);
    EXPECT_EQ(Pow<27>(2.f), 1 << 27);
    EXPECT_EQ(Pow<28>(2.f), 1 << 28);
    EXPECT_EQ(Pow<29>(2.f), 1 << 29);
}

TEST(Math, NewtonBisection) {
    EXPECT_FLOAT_EQ(1, NewtonBisection(0, 10, [](Float x) -> std::pair<Float, Float> {
                        return {Float(-1 + x), Float(1)};
                    }));
    EXPECT_FLOAT_EQ(Pi / 2, NewtonBisection(0, 2, [](Float x) -> std::pair<Float, Float> {
                        return {std::cos(x), -std::sin(x)};
                    }));

    // The derivative is a lie--pointing in the wrong direction, even--but
    // it should still work.
    Float bad = NewtonBisection(0, 2, [](Float x) -> std::pair<Float, Float> {
        return {std::cos(x), 10 * std::sin(x)};
    });
    EXPECT_LT(std::abs(Pi / 2 - bad), 1e-5);

    // Multiple zeros in the domain; make sure we find one.
    Float zero = NewtonBisection(.1, 10.1, [](Float x) -> std::pair<Float, Float> {
        return {std::sin(x), std::cos(x)};
    });
    EXPECT_LT(std::abs(std::sin(zero)), 1e-6);

    // Ill-behaved function with derivatives that go to infinity (and also
    // multiple zeros).
    auto f = [](Float x) -> std::pair<Float, Float> {
        return {std::pow(Sqr(std::sin(x)), .05) - 0.3,
                0.1 * std::cos(x) * std::sin(x) / std::pow(Sqr(std::sin(x)), 0.95)};
    };
    zero = NewtonBisection(.01, 9.42477798, f);
    // Extra slop for a messy function.
    EXPECT_LT(std::abs(f(zero).first), 1e-2);
}

TEST(Math, EvaluatePolynomial) {
    EXPECT_EQ(4, EvaluatePolynomial(100, 4));
    EXPECT_EQ(10, EvaluatePolynomial(2, 4, 3));

    EXPECT_EQ(1.5 + 2.75 * .5 - 4.25 * Pow<2>(.5) + 15.125 * Pow<3>(.5),
              EvaluatePolynomial(.5, 1.5, 2.75, -4.25, 15.125));
}

TEST(Math, CompensatedSum) {
    // In order of decreasing accuracy...
    CompensatedSum<double> kahanSumD;
    long double ldSum = 0;  // note: is plain old double with MSVC
    double doubleSum = 0;
    CompensatedSum<float> kahanSumF;
    float floatSum = 0;

    RNG rng;
    for (int i = 0; i < 16 * 1024 * 1024; ++i) {
        // Hard to sum accurately since the values span many magnitudes.
        float v = std::exp(Lerp(rng.Uniform<Float>(), -5, 20));
        ldSum += v;
        kahanSumD += v;
        doubleSum += v;
        kahanSumF += v;
        floatSum += v;
    }

    int64_t kahanDBits = FloatToBits(double(kahanSumD));
    int64_t ldBits = FloatToBits(double(ldSum));
    int64_t doubleBits = FloatToBits(doubleSum);
    int64_t kahanFBits = FloatToBits(double(float(kahanSumF)));
    int64_t floatBits = FloatToBits(double(floatSum));

    int64_t ldErrorUlps = std::abs(ldBits - kahanDBits);
    int64_t doubleErrorUlps = std::abs(doubleBits - kahanDBits);
    int64_t kahanFErrorUlps = std::abs(kahanFBits - kahanDBits);
    int64_t floatErrorUlps = std::abs(floatBits - kahanDBits);

    // Expect each to be much more accurate than the one before it.
    if (sizeof(long double) > sizeof(double))
        EXPECT_LT(ldErrorUlps * 10000, doubleErrorUlps);
    // Less slop between double and Kahan with floats.
    EXPECT_LT(doubleErrorUlps * 1000, kahanFErrorUlps);
    EXPECT_LT(kahanFErrorUlps * 10000, floatErrorUlps);
}

TEST(Log2Int, Basics) {
    for (int i = 0; i < 32; ++i) {
        uint32_t ui = 1u << i;
        EXPECT_EQ(i, Log2Int(ui));
        EXPECT_EQ(i, Log2Int((uint64_t)ui));
    }

    for (int i = 1; i < 31; ++i) {
        uint32_t ui = 1u << i;
        EXPECT_EQ(i, Log2Int(ui + 1));
        EXPECT_EQ(i, Log2Int((uint64_t)(ui + 1)));
    }

    for (int i = 0; i < 64; ++i) {
        uint64_t ui = 1ull << i;
        EXPECT_EQ(i, Log2Int(ui));
    }

    for (int i = 1; i < 64; ++i) {
        uint64_t ui = 1ull << i;
        EXPECT_EQ(i, Log2Int(ui + 1));
    }

    // Exact powers of two
    for (int i = 120; i >= -120; --i)
        EXPECT_EQ(i, Log2Int(float(std::pow(2.f, i))));
    for (int i = 120; i >= -120; --i)
        EXPECT_EQ(i, Log2Int(double(std::pow(2., i))));

    // Specifically exercise round to nearest even stuff for the Float version
    for (int i = -31; i < 31; ++i) {
        float vlow = std::pow(2, i + 0.499f);
        EXPECT_EQ(i, Log2Int(vlow));
        float vhigh = std::pow(2, i + 0.501f);
        EXPECT_EQ(i + 1, Log2Int(vhigh));
    }
    for (int i = -31; i < 31; ++i) {
        double vlow = pow(2., i + 0.4999999);
        EXPECT_EQ(i, Log2Int(vlow));
        double vhigh = pow(2., i + 0.5000001);
        EXPECT_EQ(i + 1, Log2Int(vhigh));
    }
}

TEST(Log4Int, Basics) {
    EXPECT_EQ(0, Log4Int(1));
    EXPECT_EQ(1, Log4Int(4));
    EXPECT_EQ(2, Log4Int(16));
    EXPECT_EQ(3, Log4Int(64));

    int v = 1;
    int log4 = 0, next = 4;
    for (int v = 1; v < 16385; ++v) {
        if (v == next) {
            ++log4;
            next *= 4;
        }
        EXPECT_EQ(log4, Log4Int(v));
    }
}

TEST(Pow4, Basics) {
    for (int i = 0; i < 12; ++i) {
        int p4 = 1 << (2 * i);
        EXPECT_TRUE(IsPowerOf4(p4));
        EXPECT_EQ(p4, RoundUpPow4(p4));
        if (i > 0)
            EXPECT_EQ(p4, RoundUpPow4(p4 - 1));
        EXPECT_EQ(p4 * 4, RoundUpPow4(p4 + 1));
    }
}

TEST(NextPrime, Basics) {
    EXPECT_EQ(3, NextPrime(2));
    EXPECT_EQ(11, NextPrime(10));
    EXPECT_EQ(37, NextPrime(32));
    EXPECT_EQ(37, NextPrime(37));

    auto isPrime = [](int x) {
        if ((x & 1) == 0)
            return false;
        for (int i = 3; i < std::sqrt(x) + 1; i += 3) {
            if ((x % i) == 0)
                return false;
        }
        return true;
    };

    for (int n = 3; n < 8000; n += 3) {
        if (isPrime(n))
            EXPECT_EQ(n, NextPrime(n));
        else {
            int np = NextPrime(n);
            EXPECT_TRUE(isPrime(np));
        }
    }
}

TEST(Math, ErfInv) {
    float xvl[] = {0., 0.1125, 0.25, .753, 1.521, 2.5115};
    for (float x : xvl) {
        Float e = std::erf(x);
        if (e < 1) {
            Float ei = ErfInv(e);
            if (x == 0)
                EXPECT_EQ(0, ei);
            else {
                Float err = std::abs(ei - x) / x;
                EXPECT_LT(err, 1e-4) << x << " erf " << e << " inv " << ei;
            }
        }
    }
}

#ifndef PBRT_FLOAT_AS_DOUBLE
// The next two expect a higher-precision option to verify with.
TEST(Math, DifferenceOfProducts) {
    for (int i = 0; i < 100000; ++i) {
        RNG rng(i);
        auto r = [&rng]() {
            Float logu = Lerp(rng.Uniform<Float>(), -8, 8);
            return std::pow(10, logu);
        };
        Float a = r(), b = r(), c = r(), d = r();
        Float sign = rng.Uniform<Float>() < -0.5 ? -1 : 1;
        b *= sign;
        c *= sign;
        Float dp = DifferenceOfProducts(a, b, c, d);
        Float dp2 = FMA(double(a), double(b), -double(c) * double(d));
        Float err = std::abs(dp - dp2);
        Float ulp = NextFloatUp(std::abs(dp2)) - std::abs(dp2);
        EXPECT_LT(err, 2 * ulp);
    }
}

TEST(Math, SumOfProducts) {
    for (int i = 0; i < 100000; ++i) {
        RNG rng(i);
        auto r = [&rng]() {
            Float logu = Lerp(rng.Uniform<Float>(), -8, 8);
            return std::pow(10, logu);
        };
        // Make sure mixed signs...
        Float a = r(), b = r(), c = r(), d = -r();
        Float sign = rng.Uniform<Float>() < -0.5 ? -1 : 1;
        b *= sign;
        c *= sign;
        Float sp = SumOfProducts(a, b, c, d);
        Float sp2 = FMA(double(a), double(b), double(c) * double(d));
        Float err = std::abs(sp - sp2);
        Float ulp = NextFloatUp(std::abs(sp2)) - std::abs(sp2);
        EXPECT_LT(err, 2 * ulp);
    }
}
#endif // !PBRT_FLOAT_AS_DOUBLE

TEST(FastExp, Accuracy) {
    EXPECT_EQ(1, FastExp(0));

    Float maxErr = 0;
    RNG rng(6502);
    for (int i = 0; i < 100; ++i) {
        Float v = Lerp(rng.Uniform<Float>(), -20.f, 20.f);
        Float f = FastExp(v);
        Float e = std::exp(v);
        Float err = std::abs((f - e) / e);
        maxErr = std::max(err, maxErr);
        EXPECT_LE(err, 0.0003f) << "At " << v << ", fast = " << f << ", accurate = " << e
                                << " -> relative error = " << err;
    }
#if 0
    fprintf(stderr, "max error %f\n", maxErr);

    // performance
    Float sum = 0;
    std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();
    for (int i = 0; i < 10000000; ++i) {
        Float v = Lerp(rng.Uniform<Float>(), -20.f, 20.f);
        sum += std::exp(v);
    }
    std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();
    Float elapsedMS =
        std::chrono::duration_cast<std::chrono::microseconds>(now - start)
        .count() / 1000.;
    fprintf(stderr, "%.3f ms\n", (Float)elapsedMS);

    EXPECT_NE(sum, 0); // use it
#endif
}

TEST(Math, GaussianIntegral) {
    Float muSigma[][2] = {{0, 1}, {0, 2}, {0, .1}, {1, 2}, {-2, 1}};
    for (int i = 0; i < sizeof(muSigma) / sizeof(muSigma[0]); ++i) {
        RNG rng;
        for (int j = 0; j < 5; ++j) {
            Float x0 = -5 + 10 + rng.Uniform<Float>();
            Float x1 = -5 + 10 + rng.Uniform<Float>();
            if (x0 > x1)
                pstd::swap(x0, x1);

            Float mu = muSigma[i][0], sigma = muSigma[i][1];
            Float sum = 0;
            int n = 8192;
            for (int k = 0; k < n; ++k) {
                Float u = (k + rng.Uniform<Float>()) / n;
                Float x = Lerp(u, x0, x1);
                sum += Gaussian(x, mu, sigma);
            }
            Float est = (x1 - x0) * sum / n;

            auto compareFloats = [](Float ref, Float v) {
                if (std::abs(ref) < 1e-4)
                    return std::abs(ref - v) < 1e-5;
                return std::abs((ref - v) / ref) < 1e-3;
            };
            Float in = GaussianIntegral(x0, x1, mu, sigma);
            EXPECT_TRUE(compareFloats(est, in)) << est << " vs " << in;
        }
    }
}

TEST(SquareMatrix, Basics2) {
    SquareMatrix<2> m2;

    EXPECT_TRUE(m2.IsIdentity());

    EXPECT_EQ(m2, SquareMatrix<2>(1, 0, 0, 1));
    EXPECT_NE(m2, SquareMatrix<2>(0, 1, 1, 0));

    EXPECT_EQ(SquareMatrix<2>(2, 0, 0, -1), SquareMatrix<2>::Diag(2, -1));

    SquareMatrix<2> m(1, 2, 3, 4);
    EXPECT_FALSE(m.IsIdentity());
    SquareMatrix<2> mt(1, 3, 2, 4);
    EXPECT_EQ(Transpose(m), mt);

    pstd::array<Float, 2> v{1, -2};
    pstd::array<Float, 2> vt = m * v;
    EXPECT_EQ(1 - 2 * 2, vt[0]);
    EXPECT_EQ(3 - 4 * 2, vt[1]);

    Vector2f v2(1, -2);
    Vector2f v2t = m * v2;
    EXPECT_EQ(1 - 2 * 2, v2t[0]);
    EXPECT_EQ(3 - 4 * 2, v2t[1]);

    pstd::optional<SquareMatrix<2>> inv = Inverse(m2);
    EXPECT_TRUE(inv.has_value());
    EXPECT_EQ(m2, *inv);

    SquareMatrix<2> ms(2, 4, -4, 8);
    inv = Inverse(ms);
    EXPECT_TRUE(inv.has_value());
    EXPECT_EQ(SquareMatrix<2>(1. / 4., -1. / 8., 1. / 8., 1. / 16.), *inv);

    SquareMatrix<2> degen(0, 0, 2, 0);
    inv = Inverse(degen);
    EXPECT_FALSE(inv.has_value());
}

TEST(SquareMatrix, Basics3) {
    SquareMatrix<3> m3;
    EXPECT_TRUE(m3.IsIdentity());

    EXPECT_EQ(m3, SquareMatrix<3>(1, 0, 0, 0, 1, 0, 0, 0, 1));
    EXPECT_NE(m3, SquareMatrix<3>(0, 1, 0, 0, 1, 0, 0, 0, 1));

    EXPECT_EQ(SquareMatrix<3>(2, 0, 0, 0, -1, 0, 0, 0, 3),
              SquareMatrix<3>::Diag(2, -1, 3));

    SquareMatrix<3> m(1, 2, 3, 4, 5, 6, 7, 8, 9);
    EXPECT_FALSE(m.IsIdentity());
    SquareMatrix<3> mt(1, 4, 7, 2, 5, 8, 3, 6, 9);
    EXPECT_EQ(Transpose(m), mt);

    pstd::array<Float, 3> v{1, -2, 4};
    pstd::array<Float, 3> vt = m * v;
    EXPECT_EQ(1 - 4 + 12, vt[0]);
    EXPECT_EQ(4 - 10 + 24, vt[1]);
    EXPECT_EQ(7 - 16 + 36, vt[2]);

    Vector3f v3(1, -2, 4);
    Vector3f v3t = m * v3;
    EXPECT_EQ(1 - 4 + 12, v3t[0]);
    EXPECT_EQ(4 - 10 + 24, v3t[1]);
    EXPECT_EQ(7 - 16 + 36, v3t[2]);

    pstd::optional<SquareMatrix<3>> inv = Inverse(m3);
    EXPECT_TRUE(inv.has_value());
    EXPECT_EQ(m3, *inv);

    SquareMatrix<3> ms(2, 0, 0, 0, 4, 0, 0, 0, -1);
    inv = Inverse(ms);
    EXPECT_TRUE(inv.has_value());
    EXPECT_EQ(SquareMatrix<3>(0.5, 0, 0, 0, .25, 0, 0, 0, -1), *inv);

    SquareMatrix<3> degen(0, 0, 2, 0, 0, 0, 1, 1, 1);
    inv = Inverse(degen);
    EXPECT_FALSE(inv.has_value());
}

TEST(SquareMatrix, Basics4) {
    SquareMatrix<4> m4;
    EXPECT_TRUE(m4.IsIdentity());

    EXPECT_EQ(m4, SquareMatrix<4>(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1));
    EXPECT_NE(m4, SquareMatrix<4>(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0));

    SquareMatrix<4> diag(8, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, .5);
    EXPECT_EQ(diag, SquareMatrix<4>::Diag(8, 2, 1, .5));

    SquareMatrix<4> m(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    EXPECT_FALSE(m.IsIdentity());
    SquareMatrix<4> mt(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16);
    EXPECT_EQ(Transpose(m), mt);

    pstd::optional<SquareMatrix<4>> inv = Inverse(m4);
    EXPECT_TRUE(inv.has_value());
    EXPECT_EQ(m4, *inv);

    inv = Inverse(diag);
    EXPECT_TRUE(inv.has_value());
    EXPECT_EQ(SquareMatrix<4>::Diag(.125, .5, 1, 2), *inv);

    SquareMatrix<4> degen(2, 0, 0, 0, 0, 4, 0, 0, 0, -3, 0, 1, 0, 0, 0, 0);
    inv = Inverse(degen);
    EXPECT_FALSE(inv.has_value());
}

template <int N>
static SquareMatrix<N> randomMatrix(RNG &rng) {
    SquareMatrix<N> m;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            m[i][j] = -10 + 20 * rng.Uniform<Float>();
    return m;
}

TEST(SquareMatrix, Inverse) {
    auto equal = [](Float a, Float b, Float tol = 1e-4) {
        if (std::abs(a) < 1e-5 || std::abs(b) < 1e-5)
            return std::abs(a) - std::abs(b) < tol;
        return (std::abs(a) - std::abs(b)) / ((std::abs(a) + std::abs(b)) / 2) < tol;
    };

    int nFail = 0;
    int nIters = 1000;
    {
        constexpr int N = 2;
        for (int i = 0; i < nIters; ++i) {
            RNG rng(i);
            SquareMatrix<N> m = randomMatrix<N>(rng);
            pstd::optional<SquareMatrix<N>> inv = Inverse(m);
            if (!inv) {
                ++nFail;
                continue;
            }
            SquareMatrix<N> id = m * *inv;

            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k) {
                    if (j == k)
                        EXPECT_TRUE(equal(id[j][k], 1))
                            << m << ", inv " << *inv << " prod " << id;
                    else
                        EXPECT_LT(std::abs(id[j][k]), 1e-4)
                            << m << ", inv " << *inv << " prod " << id;
                }
        }
    }
    {
        constexpr int N = 3;
        for (int i = 0; i < nIters; ++i) {
            RNG rng(i);
            SquareMatrix<N> m = randomMatrix<N>(rng);
            pstd::optional<SquareMatrix<N>> inv = Inverse(m);
            if (!inv) {
                ++nFail;
                continue;
            }
            SquareMatrix<N> id = m * *inv;

            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k) {
                    if (j == k)
                        EXPECT_TRUE(equal(id[j][k], 1))
                            << m << ", inv " << *inv << " prod " << id;
                    else
                        EXPECT_LT(std::abs(id[j][k]), 1e-4)
                            << m << ", inv " << *inv << " prod " << id;
                }
        }
    }
    {
        constexpr int N = 4;
        for (int i = 0; i < nIters; ++i) {
            RNG rng(i);
            SquareMatrix<N> m = randomMatrix<N>(rng);
            pstd::optional<SquareMatrix<N>> inv = Inverse(m);
            if (!inv) {
                ++nFail;
                continue;
            }
            SquareMatrix<N> id = m * *inv;

            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k) {
                    if (j == k)
                        EXPECT_TRUE(equal(id[j][k], 1))
                            << m << ", inv " << *inv << " prod " << id;
                    else
                        EXPECT_LT(std::abs(id[j][k]), 1e-4)
                            << m << ", inv " << *inv << " prod " << id;
                }
        }
    }

    EXPECT_LT(nFail, 3);
}

TEST(FindInterval, Basics) {
    std::vector<float> a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Check clamping for out of range
    EXPECT_EQ(0, FindInterval(a.size(), [&](int index) { return a[index] <= -1; }));
    EXPECT_EQ(a.size() - 2,
              FindInterval(a.size(), [&](int index) { return a[index] <= 100; }));

    for (size_t i = 0; i < a.size() - 1; ++i) {
        EXPECT_EQ(i, FindInterval(a.size(), [&](int index) { return a[index] <= i; }));
        EXPECT_EQ(i,
                  FindInterval(a.size(), [&](int index) { return a[index] <= i + 0.5; }));
        if (i > 0)
            EXPECT_EQ(i - 1, FindInterval(a.size(), [&](int index) {
                          return a[index] <= i - 0.5;
                      }));
    }
}

///////////////////////////////////////////////////////////////////////////
// Interval tests

// Return an exponentially-distributed floating-point value.
static Interval getFloat(RNG &rng, Float minExp = -6., Float maxExp = 6.) {
    Float logu = Lerp(rng.Uniform<Float>(), minExp, maxExp);
    Float val = std::pow(10, logu);

    // Choose a random error bound.
    Float err = 0;
    switch (rng.Uniform<uint32_t>(4)) {
    case 0:
        // no error
        break;
    case 1: {
        // small typical/reasonable error
        uint32_t ulpError = rng.Uniform<uint32_t>(1024);
        Float offset = BitsToFloat(FloatToBits(val) + ulpError);
        err = std::abs(offset - val);
        break;
    }
    case 2: {
        // bigger ~reasonable error
        uint32_t ulpError = rng.Uniform<uint32_t>(1024 * 1024);
        Float offset = BitsToFloat(FloatToBits(val) + ulpError);
        err = std::abs(offset - val);
        break;
    }
    case 3: {
        err = (4 * rng.Uniform<Float>()) * std::abs(val);
    }
    }
    Float sign = rng.Uniform<Float>() < .5 ? -1. : 1.;
    return Interval::FromValueAndError(sign * val, err);
}

// Given an Interval covering some range, choose a double-precision
// "precise" value that is in the Interval's range.
static double getPrecise(const Interval &ef, RNG &rng) {
    switch (rng.Uniform<uint32_t>(3)) {
    // 2/3 of the time, pick a value that is right at the end of the range;
    // this is a maximally difficult / adversarial choice, so should help
    // ferret out any bugs.
    case 0:
        return ef.LowerBound();
    case 1:
        return ef.UpperBound();
    case 2: {
        // Otherwise choose a value uniformly inside the Interval's range.
        Float t = rng.Uniform<Float>();
        double p = (1 - t) * ef.LowerBound() + t * ef.UpperBound();
        if (p > ef.UpperBound())
            p = ef.UpperBound();
        if (p < ef.LowerBound())
            p = ef.LowerBound();
        return p;
    }
    }
    return (Float)ef;  // NOTREACHED
}

static const int kFloatIntervalIters = 1000000;

TEST(FloatInterval, Abs) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        Interval ef = getFloat(rng);
        double precise = getPrecise(ef, rng);

        Interval efResult = Abs(ef);
        double preciseResult = std::abs(precise);

        EXPECT_GE(preciseResult, efResult.LowerBound());
        EXPECT_LE(preciseResult, efResult.UpperBound());
    }
}

TEST(FloatInterval, Sqrt) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        Interval ef = getFloat(rng);
        double precise = getPrecise(ef, rng);

        Interval efResult = Sqrt(Abs(ef));
        double preciseResult = std::sqrt(std::abs(precise));

        EXPECT_GE(preciseResult, efResult.LowerBound());
        EXPECT_LE(preciseResult, efResult.UpperBound());
    }
}

TEST(FloatInterval, Add) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        Interval ef[2] = {getFloat(rng), getFloat(rng)};
        double precise[2] = {getPrecise(ef[0], rng), getPrecise(ef[1], rng)};

        Interval efResult = ef[0] + ef[1];
        float preciseResult = precise[0] + precise[1];

        EXPECT_GE(preciseResult, efResult.LowerBound());
        EXPECT_LE(preciseResult, efResult.UpperBound());
    }
}

TEST(FloatInterval, Sub) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        Interval ef[2] = {getFloat(rng), getFloat(rng)};
        double precise[2] = {getPrecise(ef[0], rng), getPrecise(ef[1], rng)};

        Interval efResult = ef[0] - ef[1];
        float preciseResult = precise[0] - precise[1];

        EXPECT_GE(preciseResult, efResult.LowerBound());
        EXPECT_LE(preciseResult, efResult.UpperBound());
    }
}

TEST(FloatInterval, Mul) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        Interval ef[2] = {getFloat(rng), getFloat(rng)};
        double precise[2] = {getPrecise(ef[0], rng), getPrecise(ef[1], rng)};

        Interval efResult = ef[0] * ef[1];
        float preciseResult = precise[0] * precise[1];

        EXPECT_GE(preciseResult, efResult.LowerBound());
        EXPECT_LE(preciseResult, efResult.UpperBound());
    }
}

TEST(FloatInterval, Div) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        Interval ef[2] = {getFloat(rng), getFloat(rng)};
        double precise[2] = {getPrecise(ef[0], rng), getPrecise(ef[1], rng)};

        // Things get messy if the denominator's interval straddles zero...
        if (ef[1].LowerBound() * ef[1].UpperBound() < 0.)
            continue;

        Interval efResult = ef[0] / ef[1];
        float preciseResult = precise[0] / precise[1];

        EXPECT_GE(preciseResult, efResult.LowerBound());
        EXPECT_LE(preciseResult, efResult.UpperBound());
    }
}

TEST(FloatInterval, FMA) {
    int nTrials = 10000, nIters = 400;
    float sumRatio = 0;
    int ratioCount = 0;
    int nBetter = 0;
    for (int i = 0; i < nTrials; ++i) {
        RNG rng(i);
        Interval v = Abs(getFloat(rng));
        for (int j = 0; j < nIters; ++j) {
            Interval a = v;
            Interval b = getFloat(rng);
            Interval c = getFloat(rng);

            v = FMA(a, b, c);

            if (std::isinf(v.LowerBound()) || std::isinf(v.UpperBound()))
                break;

            double pa = getPrecise(a, rng);
            double pb = getPrecise(b, rng);
            double pc = getPrecise(c, rng);
            double pv = getPrecise(v, rng);
            float preciseResult = FMA(pa, pb, pc);
            EXPECT_GE(preciseResult, v.LowerBound()) << v;
            EXPECT_LE(preciseResult, v.UpperBound()) << v;

            Interval vp = a * b + c;
            EXPECT_GE(v.LowerBound(), vp.LowerBound()) << v << " vs " << vp;
            EXPECT_LE(v.UpperBound(), vp.UpperBound()) << v << " vs " << vp;

            nBetter +=
                (v.LowerBound() > vp.LowerBound() || v.UpperBound() < vp.UpperBound());
        }
    }

    EXPECT_GT(nBetter, .85 * ratioCount);
}

TEST(FloatInterval, Sqr) {
    Interval a = Interval(1.75, 2.25);
    Interval as = Sqr(a), at = a * a;
    EXPECT_EQ(as.UpperBound(), at.UpperBound());
    EXPECT_EQ(as.LowerBound(), at.LowerBound());

    // Straddle 0
    Interval b = Interval(-.75, 1.25);
    Interval bs = Sqr(b), b2 = b * b;
    EXPECT_EQ(bs.UpperBound(), b2.UpperBound());
    EXPECT_EQ(0, bs.LowerBound());
    EXPECT_LT(b2.LowerBound(), 0);
}

TEST(FloatInterval, SumSquares) {
    {
        Interval a(1), b(2), c(3);
        EXPECT_EQ(1, Float(SumSquares(a)));
        EXPECT_EQ(4, Float(SumSquares(b)));
        EXPECT_EQ(5, Float(SumSquares(a, b)));
        EXPECT_EQ(14, Float(SumSquares(a, b, c)));
    }
}

TEST(FloatInterval, DifferenceOfProducts) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        Interval a = Abs(getFloat(rng));
        Interval b = Abs(getFloat(rng));
        Interval c = Abs(getFloat(rng));
        Interval d = Abs(getFloat(rng));

        Float sign = rng.Uniform<Float>() < -0.5 ? -1 : 1;
        b *= sign;
        c *= sign;

        double pa = getPrecise(a, rng);
        double pb = getPrecise(b, rng);
        double pc = getPrecise(c, rng);
        double pd = getPrecise(d, rng);

        Interval r = DifferenceOfProducts(a, b, c, d);
        double pr = DifferenceOfProducts(pa, pb, pc, pd);

        EXPECT_GE(pr, r.LowerBound()) << trial;
        EXPECT_LE(pr, r.UpperBound()) << trial;
    }
}

TEST(FloatInterval, SumOfProducts) {
    for (int trial = 0; trial < kFloatIntervalIters; ++trial) {
        RNG rng(trial);

        // Make sure signs are mixed
        Interval a = Abs(getFloat(rng));
        Interval b = Abs(getFloat(rng));
        Interval c = Abs(getFloat(rng));
        Interval d = -Abs(getFloat(rng));

        Float sign = rng.Uniform<Float>() < -0.5 ? -1 : 1;
        b *= sign;
        c *= sign;

        double pa = getPrecise(a, rng);
        double pb = getPrecise(b, rng);
        double pc = getPrecise(c, rng);
        double pd = getPrecise(d, rng);

        Interval r = SumOfProducts(a, b, c, d);
        double pr = SumOfProducts(pa, pb, pc, pd);

        EXPECT_GE(pr, r.LowerBound()) << trial;
        EXPECT_LE(pr, r.UpperBound()) << trial;
    }
}

TEST(Math, TwoProd) {
    for (int i = 0; i < 100000; ++i) {
        RNG rng(i);
        auto r = [&rng](int minExp = -10, int maxExp = 10) {
            Float logu = Lerp(rng.Uniform<Float>(), minExp, maxExp);
            Float val = std::pow(10, logu);
            Float sign = rng.Uniform<Float>() < .5 ? -1. : 1.;
            return val * sign;
        };

        Float a = r(), b = r();
        CompensatedFloat tp = TwoProd(a, b);
        EXPECT_EQ((Float)tp, a * b);
        EXPECT_EQ((double)tp, (double)a * (double)b);
    }
}

TEST(Math, TwoSum) {
    for (int i = 0; i < 100000; ++i) {
        RNG rng(i);
        auto r = [&rng](int minExp = -10, int maxExp = 10) {
            Float logu = Lerp(rng.Uniform<Float>(), minExp, maxExp);
            Float val = std::pow(10, logu);
            Float sign = rng.Uniform<Float>() < .5 ? -1. : 1.;
            return val * sign;
        };

        Float a = r(), b = r();
        CompensatedFloat tp = TwoSum(a, b);
        EXPECT_EQ((Float)tp, a + b);
        EXPECT_EQ((double)tp, (double)a + (double)b);
    }
}

// This depends on having a higher precision option to compare to.
#ifndef PBRT_FLOAT_AS_DOUBLE
TEST(Math, InnerProduct) {
    for (int i = 0; i < 100000; ++i) {
        RNG rng(i);
        auto r = [&rng](int minExp = -10, int maxExp = 10) {
            Float logu = Lerp(rng.Uniform<Float>(), minExp, maxExp);
            Float val = std::pow(10, logu);
            Float sign = rng.Uniform<Float>() < .5 ? -1. : 1.;
            return val * sign;
        };

        Float a[4] = {r(), r(), r(), r()};
        Float b[4] = {r(), r(), r(), r()};
        Float ab = (Float)InnerProduct(a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]);
        Float dab = double(a[0]) * double(b[0]) + double(a[1]) * double(b[1]) +
                    double(a[2]) * double(b[2]) + double(a[3]) * double(b[3]);
        EXPECT_EQ(ab, dab);
    }
}
#endif  // !PBRT_FLOAT_AS_DOUBLE

// Make sure that the permute function is in fact a valid permutation.
TEST(PermutationElement, Valid) {
    for (int len = 2; len < 1024; ++len) {
        for (int iter = 0; iter < 10; ++iter) {
            std::vector<bool> seen(len, false);

            for (int i = 0; i < len; ++i) {
                int offset = PermutationElement(i, len, MixBits(1+iter));
                ASSERT_TRUE(offset >= 0 && offset < seen.size()) << offset;
                EXPECT_FALSE(seen[offset]) << StringPrintf("len %d index %d", len, i);
                seen[offset] = true;
            }
        }
    }
}

TEST(PermutationElement, Uniform) {
    for (int n : { 2, 3, 4, 5, 9, 14, 16, 22, 27, 36 }) {
        std::vector<int> count(n * n);

        int numIters = 60000 * n;
        for (int seed = 0; seed < numIters; ++seed) {
            for (int i = 0; i < n; ++i) {
                int ip = PermutationElement(i, n, MixBits(seed));
                int offset = ip * n + i;
                ++count[offset];
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Float tol = 0.03f;
                int offset = j * n + i;
                EXPECT_TRUE(count[offset] >= (1 - tol) * numIters / n &&
                            count[offset] <=(1 + tol) * numIters / n) <<
                StringPrintf("Got count %d for %d -> %d (perm size %d). Expected +/- %d.\n",
                                 count[offset], i, j, n, numIters / n);
            }
        }
    }
}

TEST(PermutationElement, UniformDelta) {
    for (int n : { 2, 3, 4, 5, 9, 14, 16, 22, 27, 36 }) {
        std::vector<int> count(n * n);

        int numIters = 60000 * n;
        for (int seed = 0; seed < numIters; ++seed) {
            for (int i = 0; i < n; ++i) {
                int ip = PermutationElement(i, n, MixBits(seed));
                int delta = ip - i;
                if (delta < 0) delta += n;
                CHECK_LT(delta, n);
                int offset = delta * n + i;
                ++count[offset];
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Float tol = 0.03f;
                int offset = j * n + i;
                EXPECT_TRUE(count[offset] >= (1 - tol) * numIters / n &&
                            count[offset] <=(1 + tol) * numIters / n) <<
                StringPrintf("Got count %d for %d -> %d (perm size %d). Expected +/- %d.\n",
                                 count[offset], i, j, n, numIters / n);
            }
        }
    }
}
