// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/file.h>

using namespace pbrt;

static std::string inTestDir(const std::string &path) {
    return path;
}

TEST(File, HasExtension) {
    EXPECT_TRUE(HasExtension("foo.exr", "exr"));
    EXPECT_TRUE(HasExtension("foo.Exr", "exr"));
    EXPECT_TRUE(HasExtension("foo.Exr", "exR"));
    EXPECT_TRUE(HasExtension("foo.EXR", "exr"));
    EXPECT_FALSE(HasExtension("foo.xr", "exr"));
    EXPECT_FALSE(HasExtension("/foo/png", "ppm"));
}

TEST(File, RemoveExtension) {
    EXPECT_EQ(RemoveExtension("foo.exr"), "foo");
    EXPECT_EQ(RemoveExtension("fooexr"), "fooexr");
    EXPECT_EQ(RemoveExtension("foo.exr.png"), "foo.exr");
}

TEST(File, ReadWriteFile) {
    std::string fn = inTestDir("readwrite.txt");
    std::string str = "this is a test.";
    EXPECT_TRUE(WriteFileContents(fn, str));
    std::string contents = ReadFileContents(fn);
    EXPECT_FALSE(contents.empty());
    EXPECT_EQ(str, contents);
    EXPECT_EQ(0, remove(fn.c_str()));
}

TEST(File, Success) {
    std::string fn = inTestDir("floatfile_good.txt");
    EXPECT_TRUE(WriteFileContents(fn, R"(1
# comment 6632
-2.5
#6502
3e2       -4.75E-1       5.25




6
)"));

    std::vector<Float> floats = ReadFloatFile(fn);
    EXPECT_EQ(6, floats.size());
    const Float expected[] = {1.f, -2.5f, 300.f, -.475f, 5.25f, 6.f};
    for (int i = 0; i < PBRT_ARRAYSIZE(expected); ++i)
        EXPECT_EQ(expected[i], floats[i]) << StringPrintf("%f %f", expected[i], floats[i]);

    EXPECT_EQ(0, remove(fn.c_str()));
}

TEST(File, Failures) {
    std::vector<Float> floats = ReadFloatFile("NO_SUCH_FILE_64622");
    EXPECT_EQ(0, floats.size());

    std::string fn = inTestDir("malformed.txt");
    EXPECT_TRUE(WriteFileContents(fn, R"(1
2 3 4
l5l
6


)"));
    floats = ReadFloatFile(fn);
    EXPECT_TRUE(floats.empty());

    remove(fn.c_str());
}
