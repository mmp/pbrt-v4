// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

using namespace pbrt;

template <int n>
struct IntType {
    PBRT_CPU_GPU int func() { return n; }
    PBRT_CPU_GPU int cfunc() const { return n; }
};

struct Handle : public TaggedPointer<IntType<0>, IntType<1>, IntType<2>, IntType<3>,
                                     IntType<4>, IntType<5>, IntType<6>, IntType<7>,
                                     IntType<8>, IntType<9>, IntType<10>, IntType<11>,
                                     IntType<12>, IntType<13>, IntType<14>, IntType<15>> {
    using TaggedPointer::TaggedPointer;

    int func() {
        auto f = [&](auto ptr) { return ptr->func(); };
        return DispatchCPU(f);
    }
    int cfunc() const {
        auto f = [&](auto ptr) { return ptr->cfunc(); };
        return DispatchCPU(f);
    }
};

struct HandleWithEightConcreteTypes
    : public TaggedPointer<IntType<0>, IntType<1>, IntType<2>, IntType<3>, IntType<4>,
                           IntType<5>, IntType<6>, IntType<7>> {
    using TaggedPointer::TaggedPointer;

    int func() {
        auto f = [&](auto ptr) { return ptr->func(); };
        return DispatchCPU(f);
    }
    int cfunc() const {
        auto f = [&](auto ptr) { return ptr->cfunc(); };
        return DispatchCPU(f);
    }

    PBRT_CPU_GPU int funcGPU() {
        auto f = [&](auto ptr) { return ptr->func(); };
        return Dispatch(f);
    }

    PBRT_CPU_GPU int cfuncGPU() {
        auto f = [&](auto ptr) { return ptr->cfunc(); };
        return Dispatch(f);
    }
};

TEST(TaggedPointer, Basics) {
    EXPECT_EQ(nullptr, Handle().ptr());

    EXPECT_EQ(16, Handle::MaxTag());
    EXPECT_EQ(17, Handle::NumTags());

    IntType<0> it0;
    IntType<1> it1;
    IntType<2> it2;
    IntType<3> it3;
    IntType<4> it4;
    IntType<5> it5;
    IntType<6> it6;
    IntType<7> it7;
    IntType<8> it8;
    IntType<9> it9;
    IntType<10> it10;
    IntType<11> it11;
    IntType<12> it12;
    IntType<13> it13;
    IntType<14> it14;
    IntType<15> it15;

    EXPECT_TRUE(Handle(&it0).Is<IntType<0>>());
    EXPECT_TRUE(Handle(&it1).Is<IntType<1>>());
    EXPECT_TRUE(Handle(&it2).Is<IntType<2>>());
    EXPECT_TRUE(Handle(&it3).Is<IntType<3>>());
    EXPECT_TRUE(Handle(&it4).Is<IntType<4>>());
    EXPECT_TRUE(Handle(&it5).Is<IntType<5>>());
    EXPECT_TRUE(Handle(&it6).Is<IntType<6>>());
    EXPECT_TRUE(Handle(&it7).Is<IntType<7>>());
    EXPECT_TRUE(Handle(&it8).Is<IntType<8>>());
    EXPECT_TRUE(Handle(&it9).Is<IntType<9>>());
    EXPECT_TRUE(Handle(&it10).Is<IntType<10>>());
    EXPECT_TRUE(Handle(&it11).Is<IntType<11>>());
    EXPECT_TRUE(Handle(&it12).Is<IntType<12>>());
    EXPECT_TRUE(Handle(&it13).Is<IntType<13>>());
    EXPECT_TRUE(Handle(&it14).Is<IntType<14>>());
    EXPECT_TRUE(Handle(&it15).Is<IntType<15>>());

    EXPECT_FALSE(Handle(&it0).Is<IntType<1>>());
    EXPECT_FALSE(Handle(&it1).Is<IntType<0>>());
    EXPECT_FALSE(Handle(&it2).Is<IntType<5>>());
    EXPECT_FALSE(Handle(&it3).Is<IntType<9>>());
    EXPECT_FALSE(Handle(&it4).Is<IntType<2>>());
    EXPECT_FALSE(Handle(&it5).Is<IntType<3>>());
    EXPECT_FALSE(Handle(&it6).Is<IntType<4>>());
    EXPECT_FALSE(Handle(&it7).Is<IntType<10>>());
    EXPECT_FALSE(Handle(&it8).Is<IntType<1>>());
    EXPECT_FALSE(Handle(&it9).Is<IntType<4>>());
    EXPECT_FALSE(Handle(&it10).Is<IntType<11>>());
    EXPECT_FALSE(Handle(&it11).Is<IntType<10>>());
    EXPECT_FALSE(Handle(&it12).Is<IntType<13>>());
    EXPECT_FALSE(Handle(&it13).Is<IntType<12>>());
    EXPECT_FALSE(Handle(&it14).Is<IntType<10>>());
    EXPECT_FALSE(Handle(&it15).Is<IntType<11>>());

    EXPECT_EQ(0, Handle(nullptr).Tag());
    EXPECT_EQ(1, Handle(&it0).Tag());
    EXPECT_EQ(2, Handle(&it1).Tag());
    EXPECT_EQ(3, Handle(&it2).Tag());
    EXPECT_EQ(4, Handle(&it3).Tag());
    EXPECT_EQ(5, Handle(&it4).Tag());
    EXPECT_EQ(6, Handle(&it5).Tag());
    EXPECT_EQ(7, Handle(&it6).Tag());
    EXPECT_EQ(8, Handle(&it7).Tag());
    EXPECT_EQ(9, Handle(&it8).Tag());
    EXPECT_EQ(10, Handle(&it9).Tag());
    EXPECT_EQ(11, Handle(&it10).Tag());
    EXPECT_EQ(12, Handle(&it11).Tag());
    EXPECT_EQ(13, Handle(&it12).Tag());
    EXPECT_EQ(14, Handle(&it13).Tag());
    EXPECT_EQ(15, Handle(&it14).Tag());
    EXPECT_EQ(16, Handle(&it15).Tag());

    EXPECT_EQ(1, Handle::TypeIndex<decltype(it0)>());
    EXPECT_EQ(2, Handle::TypeIndex<decltype(it1)>());
    EXPECT_EQ(3, Handle::TypeIndex<decltype(it2)>());
    EXPECT_EQ(4, Handle::TypeIndex<decltype(it3)>());
    EXPECT_EQ(5, Handle::TypeIndex<decltype(it4)>());
    EXPECT_EQ(6, Handle::TypeIndex<decltype(it5)>());
    EXPECT_EQ(7, Handle::TypeIndex<decltype(it6)>());
    EXPECT_EQ(8, Handle::TypeIndex<decltype(it7)>());
    EXPECT_EQ(9, Handle::TypeIndex<decltype(it8)>());
    EXPECT_EQ(10, Handle::TypeIndex<decltype(it9)>());
    EXPECT_EQ(11, Handle::TypeIndex<decltype(it10)>());
    EXPECT_EQ(12, Handle::TypeIndex<decltype(it11)>());
    EXPECT_EQ(13, Handle::TypeIndex<decltype(it12)>());
    EXPECT_EQ(14, Handle::TypeIndex<decltype(it13)>());
    EXPECT_EQ(15, Handle::TypeIndex<decltype(it14)>());
    EXPECT_EQ(16, Handle::TypeIndex<decltype(it15)>());

    EXPECT_EQ(&it0, Handle(&it0).CastOrNullptr<IntType<0>>());
    EXPECT_EQ(nullptr, Handle(&it0).CastOrNullptr<IntType<1>>());
    EXPECT_EQ(&it7, Handle(&it7).CastOrNullptr<IntType<7>>());
    EXPECT_EQ(nullptr, Handle(&it7).CastOrNullptr<IntType<0>>());
    EXPECT_EQ(nullptr, Handle(&it7).CastOrNullptr<IntType<8>>());
}

TEST(TaggedPointer, Dispatch) {
    IntType<0> it0;
    Handle h0(&it0);
    ASSERT_EQ(0, it0.func());
    EXPECT_EQ(0, h0.func());
    ASSERT_EQ(0, it0.cfunc());
    EXPECT_EQ(0, h0.cfunc());

    IntType<1> it1;
    Handle h1(&it1);
    ASSERT_EQ(1, it1.func());
    EXPECT_EQ(1, h1.func());
    ASSERT_EQ(1, it1.cfunc());
    EXPECT_EQ(1, h1.cfunc());

    IntType<2> it2;
    Handle h2(&it2);
    ASSERT_EQ(2, it2.func());
    EXPECT_EQ(2, h2.func());
    ASSERT_EQ(2, it2.cfunc());
    EXPECT_EQ(2, h2.cfunc());

    IntType<3> it3;
    Handle h3(&it3);
    ASSERT_EQ(3, it3.func());
    EXPECT_EQ(3, h3.func());
    ASSERT_EQ(3, it3.cfunc());
    EXPECT_EQ(3, h3.cfunc());

    IntType<4> it4;
    Handle h4(&it4);
    ASSERT_EQ(4, it4.func());
    EXPECT_EQ(4, h4.func());
    ASSERT_EQ(4, it4.cfunc());
    EXPECT_EQ(4, h4.cfunc());

    IntType<5> it5;
    Handle h5(&it5);
    ASSERT_EQ(5, it5.func());
    EXPECT_EQ(5, h5.func());
    ASSERT_EQ(5, it5.cfunc());
    EXPECT_EQ(5, h5.cfunc());

    IntType<6> it6;
    Handle h6(&it6);
    ASSERT_EQ(6, it6.func());
    EXPECT_EQ(6, h6.func());
    ASSERT_EQ(6, it6.cfunc());
    EXPECT_EQ(6, h6.cfunc());

    IntType<7> it7;
    Handle h7(&it7);
    ASSERT_EQ(7, it7.func());
    EXPECT_EQ(7, h7.func());
    ASSERT_EQ(7, it7.cfunc());
    EXPECT_EQ(7, h7.cfunc());

    IntType<8> it8;
    Handle h8(&it8);
    ASSERT_EQ(8, it8.func());
    EXPECT_EQ(8, h8.func());
    ASSERT_EQ(8, it8.cfunc());
    EXPECT_EQ(8, h8.cfunc());

    IntType<9> it9;
    Handle h9(&it9);
    ASSERT_EQ(9, it9.func());
    EXPECT_EQ(9, h9.func());
    ASSERT_EQ(9, it9.cfunc());
    EXPECT_EQ(9, h9.cfunc());

    IntType<10> it10;
    Handle h10(&it10);
    ASSERT_EQ(10, it10.func());
    EXPECT_EQ(10, h10.func());
    ASSERT_EQ(10, it10.cfunc());
    EXPECT_EQ(10, h10.cfunc());

    IntType<11> it11;
    Handle h11(&it11);
    ASSERT_EQ(11, it11.func());
    EXPECT_EQ(11, h11.func());
    ASSERT_EQ(11, it11.cfunc());
    EXPECT_EQ(11, h11.cfunc());

    IntType<12> it12;
    Handle h12(&it12);
    ASSERT_EQ(12, it12.func());
    EXPECT_EQ(12, h12.func());
    ASSERT_EQ(12, it12.cfunc());
    EXPECT_EQ(12, h12.cfunc());

    IntType<13> it13;
    Handle h13(&it13);
    ASSERT_EQ(13, it13.func());
    EXPECT_EQ(13, h13.func());
    ASSERT_EQ(13, it13.cfunc());
    EXPECT_EQ(13, h13.cfunc());

    IntType<14> it14;
    Handle h14(&it14);
    ASSERT_EQ(14, it14.func());
    EXPECT_EQ(14, h14.func());
    ASSERT_EQ(14, it14.cfunc());
    EXPECT_EQ(14, h14.cfunc());

    IntType<15> it15;
    Handle h15(&it15);
    ASSERT_EQ(15, it15.func());
    EXPECT_EQ(15, h15.func());
    ASSERT_EQ(15, it15.cfunc());
    EXPECT_EQ(15, h15.cfunc());
}
