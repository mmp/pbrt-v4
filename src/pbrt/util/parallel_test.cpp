// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>
#include <pbrt/pbrt.h>
#include <pbrt/util/parallel.h>
#include <atomic>

using namespace pbrt;

TEST(Parallel, Basics) {
    std::atomic<int> counter{0};
    ParallelFor(0, 1000, [&](int64_t) { ++counter; });
    EXPECT_EQ(1000, counter);

    counter = 0;
    ParallelFor(10, 1010, [&](int64_t start, int64_t end) {
        EXPECT_GT(end, start);
        EXPECT_TRUE(start >= 10 && start < 1010);
        EXPECT_TRUE(end > 10 && end <= 1010);
        for (int64_t i = start; i < end; ++i)
            ++counter;
    });
    EXPECT_EQ(1000, counter);

    counter = 0;
    ParallelFor2D(Bounds2i{{0, 0}, {15, 14}}, [&](Point2i p) { ++counter; });
    EXPECT_EQ(15 * 14, counter);
}

TEST(Parallel, DoNothing) {
    std::atomic<int> counter{0};
    ParallelFor(0, 0, [&](int64_t) { ++counter; });
    EXPECT_EQ(0, counter);

    counter = 0;
    ParallelFor2D(Bounds2i{{0, 0}, {0, 0}}, [&](Bounds2i b) { ++counter; });
    EXPECT_EQ(0, counter);
}

TEST(Parallel, ForEachThread) {
    std::atomic<int> count{MaxThreadIndex()};
    ForEachThread([&count] { --count; });
    EXPECT_EQ(0, count);
}
