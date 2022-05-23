// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/string.h>

#include <string>

using namespace pbrt;

TEST(Unicode, BasicNormalization) {
    // "Amélie" two ways, via https://en.wikipedia.org/wiki/Unicode_equivalence
    std::u16string nfc16(u"\u0041\u006d\u00e9\u006c\u0069\u0065");
    std::u16string nfd16(u"\u0041\u006d\u0065\u0301\u006c\u0069\u0065");
    EXPECT_NE(nfc16, nfd16);

    std::string nfc8 = UTF8FromUTF16(nfc16);
    std::string nfd8 = UTF8FromUTF16(nfd16);
    EXPECT_NE(nfc8, nfd8);

    EXPECT_EQ(nfc8, NormalizeUTF8(nfc8));  // nfc is already normalized
    EXPECT_EQ(nfc8, NormalizeUTF8(nfd8));  // normalizing nfd should make it equal nfc
}
