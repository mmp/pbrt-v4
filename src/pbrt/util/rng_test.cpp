// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/image.h>
#include <pbrt/util/rng.h>

#include <array>
#include <vector>

using namespace pbrt;

TEST(RNG, Reseed) {
    RNG rng(1234);
    std::vector<uint32_t> values;
    for (int i = 0; i < 100; ++i)
        values.push_back(rng.Uniform<uint32_t>());

    rng.SetSequence(1234);
    for (int i = 0; i < values.size(); ++i)
        EXPECT_EQ(values[i], rng.Uniform<uint32_t>());
}

TEST(RNG, Advance) {
    // Note: must use float not Float, since doubles consume two 32-bit
    // values from the stream, so the advance tests (as written)
    // consequently fail.
    RNG rng;
    rng.SetSequence(1234, 6502);
    std::vector<float> v;
    for (int i = 0; i < 1000; ++i)
        v.push_back(rng.Uniform<float>());

    rng.SetSequence(1234, 6502);
    rng.Advance(16);
    EXPECT_EQ(rng.Uniform<float>(), v[16]);

    for (int i = v.size() - 1; i >= 0; --i) {
        rng.SetSequence(1234, 6502);
        rng.Advance(i);
        EXPECT_EQ(rng.Uniform<float>(), v[i]);
    }

    // Switch to another sequence
    rng.SetSequence(32);
    rng.Uniform<float>();

    // Go back and check one last time
    for (int i : {5, 998, 552, 37, 16}) {
        rng.SetSequence(1234, 6502);
        rng.Advance(i);
        EXPECT_EQ(rng.Uniform<float>(), v[i]);
    }
}

TEST(RNG, OperatorMinus) {
    RNG ra(1337), rb(1337);
    RNG rng;
    for (int i = 0; i < 10; ++i) {
        int step = 1 + rng.Uniform<uint32_t>(1000);
        for (int j = 0; j < step; ++j)
            (void)ra.Uniform<uint32_t>();
        EXPECT_EQ(step, ra - rb);
        EXPECT_EQ(-step, rb - ra);

        // Reysnchronize them
        if ((rng.Uniform<uint32_t>() & 1) != 0u)
            rb.Advance(step);
        else
            ra.Advance(-step);
        EXPECT_EQ(0, ra - rb);
        EXPECT_EQ(0, rb - ra);
    }
}

TEST(RNG, Int) {
    RNG rng;
    int positive = 0, negative = 0, zero = 0;
    int count = 10000;
    for (int i = 0; i < count; ++i) {
        int v = rng.Uniform<int>();
        if (v < 0)
            ++negative;
        else if (v == 0)
            ++zero;
        else
            ++positive;
    }

    EXPECT_GT(positive, .48 * count);
    EXPECT_LT(positive, .52 * count);
    EXPECT_GT(negative, .48 * count);
    EXPECT_LT(negative, .52 * count);
    EXPECT_LT(zero, .001 * count);
}

TEST(RNG, Uint64) {
    RNG rng;
    std::array<int, 64> bitCounts = {0};
    int count = 10000;
    for (int i = 0; i < count; ++i) {
        uint64_t v = rng.Uniform<uint64_t>();
        for (int b = 0; b < 64; ++b)
            if ((v & (1ull << b)) != 0u)
                ++bitCounts[b];
    }

    for (int b = 0; b < 64; ++b) {
        EXPECT_GT(bitCounts[b], .48 * count);
        EXPECT_LT(bitCounts[b], .52 * count);
    }
}

TEST(RNG, Double) {
    RNG rng;
    for (int i = 0; i < 10; ++i) {
        double v = rng.Uniform<double>();
        EXPECT_NE(v, float(v));
    }
}

#if 0
TEST(RNG, ImageVis) {
    constexpr int nseeds = 256, ndims = 512;
//CO    std::array<int, n> histogram = { };
    // badness (if always using default seed
    uint64_t base = 127*756023296ull << 8, step = 1;
    // more bad: (LB,S): (16,1), (20, 65536), (0,65536)
//CO    uint64_t base = 1ull << atoi(getenv("LB")), step = atoi(getenv("S"));
    printf("step %d\n", step);
    Image im(PixelFormat::Float, {ndims, nseeds}, {"Y"});
    for (int i = 0; i < nseeds; ++i) {
        RNG rng(base + i * step);
        for (int j = 0; j < ndims; ++j) {
            Float u = rng.Uniform<Float>();
//CO            ++histogram[std::min<int>(u * n, n - 1)];
            im.SetChannel({j, i}, 0, u);
        }
    }

//CO    for (int i = 0; i < n; ++i)
//CO        printf("%.2f\t", Float(n * histogram[i]) / Float(n * n));
//CO    printf("\n");

    im.Write("rng.exr");
}
#endif
