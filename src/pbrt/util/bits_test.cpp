// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/bits.h>
#include <pbrt/util/math.h>
#include <pbrt/util/rng.h>

#include <cstdint>

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
