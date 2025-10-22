// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>
#include <pbrt/pbrt.h>
#include <pbrt/util/parallel.h>

#include <atomic>
#include <cmath>

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
    std::atomic<int> count{RunningThreads()};
    ForEachThread([&count] { --count; });
    EXPECT_EQ(0, count);
}

TEST(ThreadLocal, Consistency) {
    ThreadLocal<std::thread::id> tids([]() { return std::this_thread::get_id(); });

    std::atomic<int64_t> dummy{0};
    auto busywork = [&dummy](int64_t index) {
        // Do some busy work to burn some time
        // Use the index to do a varying amount of computation
        int64_t result = 0;
        for (int i = 0; i <= index; ++i)
            for (int j = 0; j <= index; ++j)
                result += i * j;
        // Store result in atomic to prevent optimization
        dummy.fetch_add(result, std::memory_order_relaxed);
    };

    ParallelFor(0, 1000, [&](int64_t index) {
        EXPECT_EQ(std::this_thread::get_id(), tids.Get());
        busywork(index);
    });

    std::vector<AsyncJob<int> *> jobs;
    for (int i = 0; i < 100; ++i) {
        jobs.push_back(RunAsync([&]() {
            EXPECT_EQ(std::this_thread::get_id(), tids.Get());
            busywork(i);
            return i;
        }));
    }
    for (int i = 0; i < 100; ++i)
        (void)jobs[i]->GetResult();

    ParallelFor(0, 1000, [&](int64_t index) {
        EXPECT_EQ(std::this_thread::get_id(), tids.Get());
        busywork(index);
    });
}
