// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>

#include <map>
#include <set>
#include <string>

using namespace pbrt;

TEST(Optional, Basics) {
    pstd::optional<int> opt;

    EXPECT_FALSE(opt.has_value());

    opt = 1;
    EXPECT_TRUE(opt.has_value());
    EXPECT_TRUE((bool)opt);
    EXPECT_EQ(1, opt.value());
    EXPECT_EQ(1, *opt);

    opt.reset();
    EXPECT_FALSE(opt.has_value());
    EXPECT_FALSE((bool)opt);
    EXPECT_EQ(2, opt.value_or(2));
    opt = 1;
    EXPECT_EQ(1, opt.value_or(2));

    int x = 3;
    opt.reset();
    opt = std::move(x);
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(3, opt.value());
}

struct AliveCounter {
    AliveCounter() { ++nAlive; }
    AliveCounter(const AliveCounter &c) { ++nAlive; }
    AliveCounter(AliveCounter &&c) { ++nAlive; }
    AliveCounter &operator=(const AliveCounter &c) {
        ++nAlive;
        return *this;
    }
    AliveCounter &operator=(AliveCounter &&c) {
        ++nAlive;
        return *this;
    }
    ~AliveCounter() { --nAlive; }

    static int nAlive;
};

int AliveCounter::nAlive;

TEST(Optional, RunDestructors) {
    AliveCounter::nAlive = 0;

    pstd::optional<AliveCounter> opt;
    EXPECT_EQ(0, AliveCounter::nAlive);

    opt = AliveCounter();
    EXPECT_EQ(1, AliveCounter::nAlive);

    opt = AliveCounter();
    EXPECT_EQ(1, AliveCounter::nAlive);

    opt.reset();
    EXPECT_EQ(0, AliveCounter::nAlive);

    {
        pstd::optional<AliveCounter> opt2 = AliveCounter();
        EXPECT_EQ(1, AliveCounter::nAlive);
    }
    EXPECT_EQ(0, AliveCounter::nAlive);

    {
        AliveCounter ac2;
        EXPECT_EQ(1, AliveCounter::nAlive);

        opt.reset();
        EXPECT_EQ(1, AliveCounter::nAlive);

        opt = std::move(ac2);
        EXPECT_EQ(2, AliveCounter::nAlive);
    }
    EXPECT_EQ(1, AliveCounter::nAlive);

    opt.reset();
    EXPECT_EQ(0, AliveCounter::nAlive);

    {
        AliveCounter ac2;
        EXPECT_EQ(1, AliveCounter::nAlive);
        pstd::optional<AliveCounter> opt2(std::move(ac2));
        EXPECT_EQ(2, AliveCounter::nAlive);

        opt2.reset();
        EXPECT_EQ(1, AliveCounter::nAlive);
    }
    EXPECT_EQ(0, AliveCounter::nAlive);
}

class TrackingResource : public pstd::pmr::memory_resource {
  public:
    void *do_allocate(size_t bytes, size_t alignment) {
        void *ptr = new char[bytes];
        allocs[ptr] = bytes;
        return ptr;
    }

    void do_deallocate(void *ptr, size_t bytes, size_t alignment) {
        auto iter = allocs.find(ptr);
        ASSERT_TRUE(iter != allocs.end());
        ASSERT_EQ(iter->second, bytes);
        allocs.erase(iter);
        delete[] (char *)ptr;
    }

    bool do_is_equal(const memory_resource &other) const noexcept {
        return &other == this;
    }

    std::map<void *, size_t> allocs;
};

TEST(MonotonicBufferResource, NoOverlap) {
    TrackingResource tr;
    pstd::pmr::monotonic_buffer_resource mb(1024, &tr);
    Allocator alloc(&mb);
    RNG rng;
    struct Span {
        char *ptr;
        size_t size;
    };
    std::vector<Span> spans;

    for (int i = 0; i < 10000; ++i) {
        size_t size;
        if (rng.Uniform<Float>() < .5f)
            size = 1 << rng.Uniform<int>(12);
        else
            size = rng.Uniform<int>(2048);

        char *p = (char *)alloc.allocate_bytes(size);
        // O(n^2)...
        for (const Span &s : spans)
            EXPECT_TRUE(p >= s.ptr + s.size || p + size <= s.ptr);
        spans.push_back(Span{p, size});
    }
}
