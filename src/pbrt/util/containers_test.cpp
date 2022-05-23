// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/util/containers.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>

#include <set>
#include <string>

using namespace pbrt;

TEST(Array2D, Basics) {
    const int nx = 5, ny = 9;
    Array2D<Float> a(nx, ny);

    EXPECT_EQ(nx, a.XSize());
    EXPECT_EQ(ny, a.YSize());
    EXPECT_EQ(nx * ny, a.size());

    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
            a(x, y) = 1000 * x + y;

    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
            EXPECT_EQ(1000 * x + y, a(x, y));
}

TEST(Array2D, Bounds) {
    Bounds2i b(Point2i(-4, 3), Point2i(10, 7));
    Array2D<Point2f> a(b);

    EXPECT_EQ(b.pMax.x - b.pMin.x, a.XSize());
    EXPECT_EQ(b.pMax.y - b.pMin.y, a.YSize());

    for (Point2i p : b)
        a[p] = Point2f(p.y, p.x);

    for (Point2i p : b)
        EXPECT_EQ(Point2f(p.y, p.x), a[p]);
}

TEST(HashMap, Basics) {
    Allocator alloc;
    HashMap<int, std::string, std::hash<int>> map(alloc);

    map.Insert(1, std::string("yolo"));
    map.Insert(10, std::string("hello"));
    map.Insert(42, std::string("test"));

    EXPECT_EQ(3, map.size());
    EXPECT_GE(map.capacity(), 3);
    EXPECT_TRUE(map.HasKey(1));
    EXPECT_TRUE(map.HasKey(10));
    EXPECT_TRUE(map.HasKey(42));
    EXPECT_FALSE(map.HasKey(0));
    EXPECT_FALSE(map.HasKey(1240));
    EXPECT_EQ("yolo", map[1]);
    EXPECT_EQ("hello", map[10]);
    EXPECT_EQ("test", map[42]);

    map.Insert(10, std::string("hai"));
    EXPECT_EQ(3, map.size());
    EXPECT_GE(map.capacity(), 3);
    EXPECT_EQ("hai", map[10]);
}

TEST(HashMap, Randoms) {
    Allocator alloc;
    HashMap<int, int, std::hash<int>> map(alloc);
    std::set<int> values;
    RNG rng(1234);

    for (int i = 0; i < 10000; ++i) {
        int v = rng.Uniform<int>();
        values.insert(v);
        map.Insert(v, -v);
    }

    // Could have a collision so thus less...
    EXPECT_EQ(map.size(), values.size());

    // Reset
    rng.SetSequence(1234);
    for (int i = 0; i < 10000; ++i) {
        int v = rng.Uniform<int>();
        ASSERT_TRUE(map.HasKey(v));
        EXPECT_EQ(-v, map[v]);
    }

    int nVisited = 0;
    for (auto iter = map.begin(); iter != map.end(); ++iter) {
        ++nVisited;

        EXPECT_EQ(iter->first, -iter->second);

        int v = iter->first;
        auto siter = values.find(v);
        ASSERT_NE(siter, values.end());
        values.erase(siter);
    }

    EXPECT_EQ(nVisited, 10000);
    EXPECT_EQ(0, values.size());
}

TEST(TypePack, Index) {
    using Pack = TypePack<int, float, double>;

    // Extra parens so EXPECT_TRUE doesn't get confused by the comma
    EXPECT_EQ(0, (IndexOf<int, Pack>::count));
    EXPECT_EQ(1, (IndexOf<float, Pack>::count));
    EXPECT_EQ(2, (IndexOf<double, Pack>::count));
}

TEST(TypePack, HasType) {
    using Pack = TypePack<signed int, float, double>;

    // Extra parens so EXPECT_TRUE doesn't get confused by the comma
    EXPECT_TRUE((HasType<int, Pack>::value));
    EXPECT_TRUE((HasType<float, Pack>::value));
    EXPECT_TRUE((HasType<double, Pack>::value));

    EXPECT_FALSE((HasType<char, Pack>::value));
    EXPECT_FALSE((HasType<unsigned int, Pack>::value));
}

TEST(TypePack, TakeRemove) {
    using Pack = TypePack<signed int, float, double>;

    static_assert(
        std::is_same_v<TypePack<signed int>, typename TakeFirstN<1, Pack>::type>);
    static_assert(std::is_same_v<
                  TypePack<float>,
                  typename TakeFirstN<1, typename RemoveFirstN<1, Pack>::type>::type>);
    static_assert(std::is_same_v<
                  TypePack<double>,
                  typename TakeFirstN<1, typename RemoveFirstN<2, Pack>::type>::type>);
}

template <typename T>
struct Set {};

TEST(TypePack, Map) {
    using SetPack = typename MapType<Set, TypePack<signed int, float, double>>::type;

    static_assert(
        std::is_same_v<TypePack<Set<signed int>>, typename TakeFirstN<1, SetPack>::type>);
    static_assert(std::is_same_v<
                  TypePack<Set<float>>,
                  typename TakeFirstN<1, typename RemoveFirstN<1, SetPack>::type>::type>);
    static_assert(std::is_same_v<
                  TypePack<Set<double>>,
                  typename TakeFirstN<1, typename RemoveFirstN<2, SetPack>::type>::type>);
}

TEST(InternCache, BasicString) {
    InternCache<std::string> cache;

    const std::string *one = cache.Lookup(std::string("one"));
    EXPECT_EQ(1, cache.size());
    EXPECT_EQ("one", *one);

    const std::string *two = cache.Lookup(std::string("two"));
    EXPECT_NE(one, two);
    EXPECT_EQ("two", *two);

    const std::string *anotherOne = cache.Lookup(std::string("one"));
    EXPECT_EQ(2, cache.size());
    EXPECT_EQ("one", *anotherOne);
    EXPECT_EQ(one, anotherOne);
}

TEST(InternCache, BadHash) {
    struct BadIntHash {
        size_t operator()(int i) { return i % 7; }
    };

    InternCache<int, BadIntHash> cache;

    int n = 10000;
    std::vector<const int *> firstPass(n);
    for (int i = 0; i < n; ++i) {
        int ii = PermutationElement(i, n, 0xfeed00f5);
        const int *p = cache.Lookup(ii);
        EXPECT_EQ(*p, ii);
        firstPass[ii] = p;
        EXPECT_EQ(i + 1, cache.size());
    }
    for (int i = 0; i < n; ++i) {
        int ii = PermutationElement(i, n, 0xfeed00f5);
        const int *p = cache.Lookup(ii);
        EXPECT_EQ(*p, ii);
        EXPECT_EQ(p, firstPass[ii]);
        EXPECT_EQ(n, cache.size());
    }
}
