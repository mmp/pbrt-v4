// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>

#include <pbrt/pbrt.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/rng.h>

using namespace pbrt;

TEST(FloatingPoint, Pieces) {
    EXPECT_EQ(1, Exponent(2.f));
    EXPECT_EQ(-1, Exponent(0.5f));
    EXPECT_EQ(0b10000000000000000000000, Significand(3.f));
    EXPECT_EQ(0b11000000000000000000000, Significand(7.f));

    EXPECT_EQ(1, Exponent(2.));
    EXPECT_EQ(-1, Exponent(0.5));
    EXPECT_EQ(0b1000000000000000000000000000000000000000000000000000, Significand(3.));
    EXPECT_EQ(0b1100000000000000000000000000000000000000000000000000, Significand(7.));
}

static float GetFloat(RNG &rng) {
    float f;
    do {
        f = BitsToFloat(rng.Uniform<uint32_t>());
    } while (std::isnan(f));
    return f;
}

static double GetDouble(RNG &rng) {
    double d;
    do {
        d = BitsToFloat(uint64_t(rng.Uniform<uint32_t>()) |
                        (uint64_t(rng.Uniform<uint32_t>()) << 32));
    } while (std::isnan(d));
    return d;
}

TEST(FloatingPoint, NextUpDownFloat) {
    EXPECT_GT(NextFloatUp(-0.f), 0.f);
    EXPECT_LT(NextFloatDown(0.f), 0.f);

    EXPECT_EQ(NextFloatUp((float)Infinity), (float)Infinity);
    EXPECT_LT(NextFloatDown((float)Infinity), (float)Infinity);

    EXPECT_EQ(NextFloatDown(-(float)Infinity), -(float)Infinity);
    EXPECT_GT(NextFloatUp(-(float)Infinity), -(float)Infinity);

    RNG rng;
    for (int i = 0; i < 100000; ++i) {
        float f = GetFloat(rng);
        if (std::isinf(f))
            continue;

        EXPECT_EQ(std::nextafter(f, (float)Infinity), NextFloatUp(f));
        EXPECT_EQ(std::nextafter(f, -(float)Infinity), NextFloatDown(f));
    }
}

TEST(FloatingPoint, NextUpDownDouble) {
    EXPECT_GT(NextFloatUp(-0.), 0.);
    EXPECT_LT(NextFloatDown(0.), 0.);

    EXPECT_EQ(NextFloatUp((double)Infinity), (double)Infinity);
    EXPECT_LT(NextFloatDown((double)Infinity), (double)Infinity);

    EXPECT_EQ(NextFloatDown(-(double)Infinity), -(double)Infinity);
    EXPECT_GT(NextFloatUp(-(double)Infinity), -(double)Infinity);

    RNG rng(3);
    for (int i = 0; i < 100000; ++i) {
        double d = GetDouble(rng);
        if (std::isinf(d))
            continue;

        EXPECT_EQ(std::nextafter(d, (double)Infinity), NextFloatUp(d));
        EXPECT_EQ(std::nextafter(d, -(double)Infinity), NextFloatDown(d));
    }
}

TEST(FloatingPoint, FloatBits) {
    RNG rng(1);
    for (int i = 0; i < 100000; ++i) {
        uint32_t ui = rng.Uniform<uint32_t>();
        float f = BitsToFloat(ui);
        if (std::isnan(f))
            continue;

        EXPECT_EQ(ui, FloatToBits(f));
    }
}

TEST(FloatingPoint, DoubleBits) {
    RNG rng(2);
    for (int i = 0; i < 100000; ++i) {
        uint64_t ui = (uint64_t(rng.Uniform<uint32_t>()) |
                       (uint64_t(rng.Uniform<uint32_t>()) << 32));
        double f = BitsToFloat(ui);

        if (std::isnan(f))
            continue;

        EXPECT_EQ(ui, FloatToBits(f));
    }
}

TEST(FloatingPoint, AtomicFloat) {
    AtomicFloat af(0);
    Float f = 0.;
    EXPECT_EQ(f, af);
    af.Add(1.0251);
    f += 1.0251;
    EXPECT_EQ(f, af);
    af.Add(2.);
    f += 2.;
    EXPECT_EQ(f, af);
}

TEST(Half, Basics) {
    EXPECT_EQ(Half(0.f).Bits(), HalfPositiveZero);
    EXPECT_EQ(Half(-0.f).Bits(), HalfNegativeZero);
    EXPECT_EQ(Half(Infinity).Bits(), HalfPositiveInfinity);
    EXPECT_EQ(Half(-Infinity).Bits(), HalfNegativeInfinity);

    EXPECT_TRUE(Half(std::numeric_limits<Float>::quiet_NaN()).IsNaN());
    EXPECT_TRUE(Half(std::numeric_limits<Float>::quiet_NaN()).IsNaN());
    EXPECT_FALSE(Half::FromBits(HalfPositiveInfinity).IsNaN());

    for (uint32_t bits = 0; bits < 65536; ++bits) {
        Half h = Half::FromBits(bits);
        if (h.IsInf() || h.IsNaN())
            continue;
        EXPECT_EQ(h, -(-h));
        EXPECT_EQ(-h, Half(-float(h)));
    }
}

TEST(Half, ExactConversions) {
    // Test round-trip conversion of integers that are perfectly
    // representable.
    for (Float i = -2048; i <= 2048; ++i) {
        EXPECT_EQ(i, Float(Half(i)));
    }

    // Similarly for some well-behaved floats
    float limit = 1024, delta = 0.5;
    for (int i = 0; i < 10; ++i) {
        for (float f = -limit; f <= limit; f += delta)
            EXPECT_EQ(f, Float(Half(f)));
        limit /= 2;
        delta /= 2;
    }
}

TEST(Half, Randoms) {
    RNG rng;
    // Choose a bunch of random positive floats and make sure that they
    // convert to reasonable values.
    for (int i = 0; i < 1024; ++i) {
        float f = rng.Uniform<Float>() * 512;
        uint16_t h = Half(f).Bits();
        float fh = Float(Half::FromBits(h));
        if (fh == f) {
            // Very unlikely, but we happened to pick a value exactly
            // representable as a half.
            continue;
        } else {
            // The other half value that brackets the float.
            uint16_t hother;
            if (fh > f) {
                // The closest half was a bit bigger; therefore, the half before
                // it s the other one.
                hother = h - 1;
                if (hother > h) {
                    // test for wrapping around zero
                    continue;
                }
            } else {
                hother = h + 1;
                if (hother < h) {
                    // test for wrapping around zero
                    continue;
                }
            }

            // Make sure the two half values bracket the float.
            float fother = Float(Half::FromBits(hother));
            float dh = std::abs(fh - f);
            float dother = std::abs(fother - f);
            if (fh > f)
                EXPECT_LT(fother, f);
            else
                EXPECT_GT(fother, f);

            // Make sure rounding to the other one of them wouldn't have given a
            // closer half.
            EXPECT_LE(dh, dother);
        }
    }
}

TEST(Half, NextUp) {
    Half h = Half::FromBits(HalfNegativeInfinity);
    int iters = 0;
    while (h.Bits() != HalfPositiveInfinity) {
        ASSERT_LT(iters, 65536);
        ++iters;

        Half hup = h.NextUp();
        EXPECT_GT((float)hup, (float)h);
        h = hup;
    }
    // NaNs use the maximum exponent and then the sign bit and have a
    // non-zero significand.
    EXPECT_EQ(65536 - (1 << 11), iters);
}

TEST(Half, NextDown) {
    Half h = Half::FromBits(HalfPositiveInfinity);
    int iters = 0;
    while (h.Bits() != HalfNegativeInfinity) {
        ASSERT_LT(iters, 65536);
        ++iters;

        Half hdown = h.NextDown();
        EXPECT_LT((float)hdown, (float)h) << hdown.Bits() << " " << h.Bits();
        h = hdown;
    }
    // NaNs use the maximum exponent and then the sign bit and have a
    // non-zero significand.
    EXPECT_EQ(65536 - (1 << 11), iters);
}

TEST(Half, Equal) {
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        Half h = Half::FromBits(bits);
        if (h.IsInf() || h.IsNaN())
            continue;
        EXPECT_EQ(h, h);
    }

    // Check that +/- zero behave sensibly.
    Half hpz(0.f), hnz(-0.f);
    EXPECT_NE(hpz.Bits(), hnz.Bits());
    EXPECT_EQ(hpz, hnz);
    // Smallest representable non-zero half value
    Half hmin(5.9605e-08f);
    EXPECT_NE(hpz, hmin);
    EXPECT_NE(hnz, -hmin);
}

TEST(Half, RoundToNearestEven) {
    // For all floats halfway between two half values, make sure that they
    // round to the "even" half (i.e., zero low bit).
    Half h0(0.f), h1 = h0.NextUp();
    while (!h1.IsInf()) {
        float mid = (float(h0) + float(h1)) / 2;
        if ((h0.Bits() & 1) == 0)
            EXPECT_EQ(h0, Half(mid));
        else
            EXPECT_EQ(h1, Half(mid));

        h0 = h1;
        h1 = h0.NextUp();
    }

    h0 = Half(-0.f);
    h1 = h0.NextDown();
    while (!h1.IsInf()) {
        float mid = (float(h0) + float(h1)) / 2;
        if ((h0.Bits() & 1) == 0)
            EXPECT_EQ(h0, Half(mid));
        else
            EXPECT_EQ(h1, Half(mid));

        h0 = h1;
        h1 = h0.NextDown();
    }
}
