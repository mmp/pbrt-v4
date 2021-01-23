// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <double-conversion/double-conversion.h>
#include <array>
#include <sstream>
#include <typeinfo>
#include <vector>

using namespace pbrt;

TEST(StringPrintf, Basics) {
    EXPECT_EQ(StringPrintf("Hello, world"), "Hello, world");
    EXPECT_EQ(StringPrintf("x = %d", 5), "x = 5");
    EXPECT_EQ(StringPrintf("%f, %f, %f", 1., 1.5, -8.125), "1, 1.5, -8.125");
#ifndef PBRT_IS_WINDOWS
    EXPECT_DEATH(StringPrintf("not enough %s"), "Not enough optional values");
    EXPECT_DEATH(StringPrintf("not enough %f yolo"), "Not enough optional values");
    EXPECT_DEATH(StringPrintf("too many %f yolo", 1, 2), "Excess values passed");
    EXPECT_DEATH(StringPrintf("too many", 1), "Excess values passed");
#endif
}

TEST(StringPrintf, FancyPctS) {
    EXPECT_EQ(StringPrintf("%s", false), "false");
    EXPECT_EQ(StringPrintf("%s", true), "true");
    EXPECT_EQ(StringPrintf("%s", Vector3f(Pi, -2, 3.1)), "[ 3.1415927, -2, 3.1 ]");

    std::string s = "string";
    EXPECT_EQ(StringPrintf("%d %s", 1, s), "1 string");

    std::vector<int> v = {1, 2, 3, 4};
    EXPECT_EQ(StringPrintf("%s", v), "[ 1, 2, 3, 4 ]");

    std::array<std::string, 3> a = {"foo", "bar", "bat"};
    EXPECT_EQ(StringPrintf("%s", a), "[ foo, bar, bat ]");

    pstd::span<std::string> span(a);
    EXPECT_EQ(StringPrintf("%s", span), "[ foo, bar, bat ]");

    pstd::optional<float> pi = 3.125;
    EXPECT_EQ(StringPrintf("[ pstd::optional<%s> set: true value: 3.125 ]",
                           typeid(float).name()),
              StringPrintf("%s", pi));

    pstd::optional<float *> oe;
    EXPECT_EQ(StringPrintf("[ pstd::optional<%s> set: false value: n/a ]",
                           typeid(float *).name()),
              StringPrintf("%s", oe));
}

TEST(StringPrintf, optional) {
    pstd::optional<float> pi = 3.125;
    StringPrintf("%s", pi);
}

TEST(StringPrintf, FancyPctD) {
    EXPECT_EQ("1", StringPrintf("%d", 1u));
    EXPECT_EQ("4294967295", StringPrintf("%d", 0xffffffffu));

    // > 4G
    EXPECT_EQ("100000000000", StringPrintf("%d", size_t(100000000000)));

    uint64_t ubig = ~0ull;
    EXPECT_EQ("18446744073709551615", StringPrintf("%d", ubig));

    int64_t ibig = ubig >> 1;
    EXPECT_EQ("9223372036854775807", StringPrintf("%d", ibig));
}

TEST(StringPrintf, Precision) {
    double_conversion::StringToDoubleConverter floatParser(
        double_conversion::StringToDoubleConverter::ALLOW_HEX,
        0. /* empty string value */, 0. /* junk string value */,
        nullptr /* infinity symbol */, nullptr /* NaN symbol */);

    std::string pi = StringPrintf("%f", float(Pi));
    int length;
    float val = floatParser.StringToFloat(pi.data(), pi.size(), &length);
    EXPECT_NE(0, length);
    EXPECT_EQ(val, Pi);

    std::string e = StringPrintf("%f", std::exp(1.));
    double dval = floatParser.StringToDouble(e.data(), e.size(), &length);
    EXPECT_NE(0, length);
    EXPECT_EQ(dval, std::exp(1.));
}

TEST(OperatorLeftShiftPrint, Basics) {
    {
        std::ostringstream os;
        os << Point2f(105.5, -12.75);
        EXPECT_EQ("[ 105.5, -12.75 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Point2i(-9, 5);
        EXPECT_EQ("[ -9, 5 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Point3f(0., 1.25, -9.25);
        EXPECT_EQ("[ 0, 1.25, -9.25 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Point3i(7, -10, 4);
        EXPECT_EQ("[ 7, -10, 4 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Vector2f(105.5, -12.75);
        EXPECT_EQ("[ 105.5, -12.75 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Vector2i(-9, 5);
        EXPECT_EQ("[ -9, 5 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Vector3f(0., 1.25, -9.25);
        EXPECT_EQ("[ 0, 1.25, -9.25 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Vector3i(7, -10, 4);
        EXPECT_EQ("[ 7, -10, 4 ]", os.str());
    }
    {
        std::ostringstream os;
        os << Normal3f(0., 1.25, -9.25);
        EXPECT_EQ("[ 0, 1.25, -9.25 ]", os.str());
    }
    {
        std::ostringstream os;
        Quaternion q;
        q.v = Vector3f(1.25, -8.25, 14.75);
        q.w = -0.5;
        os << q;
        EXPECT_EQ("[ 1.25, -8.25, 14.75, -0.5 ]", os.str());
    }
    {
        std::ostringstream os;
        Ray r(Point3f(-5.5, 2.75, 0.), Vector3f(1.0, -8.75, 2.25), 0.25);
        os << r;
        EXPECT_EQ("[ o: [ -5.5, 2.75, 0 ] d: [ 1, -8.75, 2.25 ] time: 0.25, "
                  "medium: (nullptr) ]",
                  os.str());
    }
    {
        std::ostringstream os;
        Bounds2f b(Point2f(2, -5), Point2f(-8, 3));
        os << b;
        EXPECT_EQ("[ [ -8, -5 ] - [ 2, 3 ] ]", os.str());
    }
    {
        std::ostringstream os;
        Bounds3f b(Point3f(2, -5, .125), Point3f(-8, 3, -128.5));
        os << b;
        EXPECT_EQ("[ [ -8, -5, -128.5 ] - [ 2, 3, 0.125 ] ]", os.str());
    }
    {
        std::ostringstream os;
        SquareMatrix<4> m(0., -1., 2., -3.5, 4.5, 5.5, -6.5, -7.5, 8., 9.25, 10.75, -11,
                          12, 13.25, 14.5, -15.875);
        os << m;
        EXPECT_EQ("[ [ 0, -1, 2, -3.5 ], "
                  "[ 4.5, 5.5, -6.5, -7.5 ], "
                  "[ 8, 9.25, 10.75, -11 ], "
                  "[ 12, 13.25, 14.5, -15.875 ] "
                  "]",
                  os.str());
    }
    {
        std::ostringstream os;
        Transform t = Translate(Vector3f(-1.25, 3.5, 7.875)) * Scale(2., -8., 1);
        os << t;
        EXPECT_EQ("[ m: [ [ 2, 0, 0, -1.25 ], [ 0, -8, 0, 3.5 ], [ 0, 0, 1, "
                  "7.875 ], [ 0, 0, 0, 1 ] ] "
                  "mInv: [ [ 0.5, 0, 0, 0.625 ], [ 0, -0.125, 0, 0.4375 ], [ "
                  "0, 0, 1, -7.875 ], [ 0, 0, 0, 1 ] ] ]",
                  os.str());
    }
}
