// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/buffercache.h>

#include <vector>

using namespace pbrt;

TEST(BufferCache, Basics) {
    ASSERT_FALSE(intBufferCache == nullptr);

    std::vector<int> v{1, 2, 3, 4, 5};

    size_t baseMem = intBufferCache->BytesUsed();

    Allocator alloc;
    const int *ptr = intBufferCache->LookupOrAdd(v, alloc);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_EQ(ptr[i], v[i]);

    EXPECT_EQ(5 * sizeof(int), intBufferCache->BytesUsed() - baseMem);

    EXPECT_EQ(ptr, intBufferCache->LookupOrAdd(v, alloc));

    EXPECT_EQ(5 * sizeof(int), intBufferCache->BytesUsed() - baseMem);

    std::vector<int> v2{1, 2, 3, 4};
    const int *ptr2 = intBufferCache->LookupOrAdd(v2, alloc);
    EXPECT_NE(ptr, ptr2);
    for (size_t i = 0; i < v2.size(); ++i)
        EXPECT_EQ(ptr2[i], v2[i]);

    EXPECT_EQ(9 * sizeof(int), intBufferCache->BytesUsed() - baseMem);
}
