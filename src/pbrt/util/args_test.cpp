// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/args.h>
#include <pbrt/util/string.h>

#include <algorithm>
#include <memory>

using namespace pbrt;

TEST(Args, Simple) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    {
        int nthreads = 0;
        std::vector<std::string> args = SplitStringsFromWhitespace("--nthreads 4");
        auto iter = args.begin();
        EXPECT_TRUE(ParseArg(&iter, args.end(), "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(++iter == args.end());
    }
    {
        int nthreads = 0;
        std::vector<std::string> args = SplitStringsFromWhitespace("--nthreads=4");
        auto iter = args.begin();
        EXPECT_TRUE(ParseArg(&iter, args.end(), "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(++iter == args.end());
    }
}

TEST(Args, Multiple) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    {
        int nthreads = 0;
        bool log = false;
        std::vector<std::string> args = SplitStringsFromWhitespace("--log --nthreads 4 yolo");
        auto iter = args.begin();
        EXPECT_TRUE(ParseArg(&iter, args.end(), "log", &log, expectNoError));
        ++iter;
        EXPECT_TRUE(ParseArg(&iter, args.end(), "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(log);
        EXPECT_FALSE(++iter == args.end());
        EXPECT_EQ(*iter, "yolo");
    }
    {
        int nthreads = 0;
        std::vector<std::string> args = SplitStringsFromWhitespace("yolo --nthreads=4");
        auto iter = args.begin();
        EXPECT_FALSE(ParseArg(&iter, args.end(), "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(*iter, "yolo");
        ++iter;
        EXPECT_TRUE(ParseArg(&iter, args.end(), "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(++iter == args.end());
    }
}

TEST(Args, Bool) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    bool log = true;
    bool benchmark = false;
    bool debug = false;
    std::vector<std::string> args = SplitStringsFromWhitespace("--log=false --benchmark --debug=true");
    auto iter = args.begin();
    EXPECT_TRUE(ParseArg(&iter, args.end(), "log", &log, expectNoError));
    EXPECT_FALSE(log);
    ++iter;
    EXPECT_TRUE(ParseArg(&iter, args.end(), "benchmark", &benchmark, expectNoError));
    EXPECT_TRUE(benchmark);
    ++iter;
    EXPECT_TRUE(ParseArg(&iter, args.end(), "debug", &debug, expectNoError));
    EXPECT_TRUE(debug);
    EXPECT_TRUE(++iter == args.end());
}

TEST(Args, ErrorMissingValue) {
    bool errCalled = false;
    auto callback = [&errCalled](const std::string &s) { errCalled = true; };

    int nthreads = 2;
    std::vector<std::string> args = SplitStringsFromWhitespace("--nthreads");
    auto iter = args.begin();
    EXPECT_FALSE(ParseArg(&iter, args.end(), "nthreads", &nthreads, callback));
    EXPECT_EQ(nthreads, 2);
    EXPECT_TRUE(errCalled);
}

TEST(Args, ErrorMissingValueEqual) {
    bool errCalled = false;
    auto callback = [&errCalled](const std::string &s) { errCalled = true; };

    int nthreads = 2;
    std::vector<std::string> args = SplitStringsFromWhitespace("--nthreads=");
    auto iter = args.begin();
    EXPECT_FALSE(ParseArg(&iter, args.end(), "nthreads", &nthreads, callback));
    EXPECT_EQ(nthreads, 2);
    EXPECT_TRUE(errCalled);
}

TEST(Args, ErrorBogusBool) {
    bool errCalled = false;
    auto callback = [&errCalled](const std::string &s) { errCalled = true; };

    bool log = false;
    std::vector<std::string> args = SplitStringsFromWhitespace("--log=tru3");
    auto iter = args.begin();
    EXPECT_FALSE(ParseArg(&iter, args.end(), "log", &log, callback));
    EXPECT_FALSE(log);
    EXPECT_TRUE(errCalled);
}

TEST(Args, Normalization) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    {
        int nthreads = 0;
        std::vector<std::string> args = SplitStringsFromWhitespace("--n_threads 4");
        auto iter = args.begin();
        EXPECT_TRUE(ParseArg(&iter, args.end(), "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(++iter == args.end());
    }
    {
        int nthreads = 0;
        std::vector<std::string> args = SplitStringsFromWhitespace("--nThreads=4");
        auto iter = args.begin();
        EXPECT_TRUE(ParseArg(&iter, args.end(), "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(++iter == args.end());
    }
}
