// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/util/hash.h>

#include <set>

using namespace pbrt;

TEST(Hash, VarArgs) {
    int64_t buf[] = {1, -12511, 31415821, 37};
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(HashBuffer(buf + i, sizeof(int64_t)), Hash(buf[i]));
}

TEST(Hash, Collisions) {
    std::set<uint32_t> low, high;
    std::set<uint64_t> full;

    int lowCollisions = 0, highCollisions = 0;
    int fullCollisions = 0;
    int same = 0;
    for (int i = 0; i < 10000000; ++i) {
        uint64_t h = Hash(i);

        if (h == i)
            ++same;

        if (low.find(h) != low.end())
            ++lowCollisions;
        if (high.find(h >> 32) != high.end())
            ++highCollisions;
        if (full.find(h >> 32) != full.end())
            ++fullCollisions;
    }

    // It's actually potentially legit if any of these hit; it should
    // shouldn't happen a lot.
    EXPECT_EQ(0, same);
    EXPECT_EQ(0, lowCollisions);
    EXPECT_EQ(0, highCollisions);
    EXPECT_EQ(0, fullCollisions);
}

TEST(Hash, Unaligned) {
    uint64_t buf[] = { 0xfacebeef, 0x65028088, 0x13372048 };
    char cbuf[sizeof(buf) + 8];
    for (int delta = 0; delta < 8; ++ delta) {
        memcpy(cbuf + delta, buf, sizeof(buf));
        EXPECT_EQ(HashBuffer(buf, sizeof(buf)), HashBuffer(cbuf + delta, sizeof(buf)));
    }
}
