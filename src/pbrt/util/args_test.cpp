// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/args.h>

#include <algorithm>
#include <memory>

using namespace pbrt;

// Splits into words at each space character. Assumes just one space
// between words.
static char **makeArgs(std::string args) {
    size_t n = std::count(args.begin(), args.end(), ' ') + 1;
    char **argv = new char *[n + 1];
    for (int i = 0;; ++i) {
        size_t pos = args.find(' ');
        std::string arg = args.substr(0, pos);
        argv[i] = new char[arg.length() + 1];
        strcpy(argv[i], arg.c_str());

        if (pos == std::string::npos) {
            EXPECT_EQ(i + 1, n);
            break;
        }
        args = args.substr(pos + 1);
    }
    argv[n] = nullptr;
    return argv;
}

TEST(Args, Simple) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    {
        int nthreads = 0;
        auto argv = makeArgs("--nthreads 4");
        EXPECT_TRUE(ParseArg(&argv, "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(*argv == nullptr);
    }
    {
        int nthreads = 0;
        auto argv = makeArgs("--nthreads=4");
        EXPECT_TRUE(ParseArg(&argv, "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(*argv == nullptr);
    }
}

TEST(Args, Multiple) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    {
        int nthreads = 0;
        bool log = false;
        auto argv = makeArgs("--log --nthreads 4 yolo");
        EXPECT_TRUE(ParseArg(&argv, "log", &log, expectNoError));
        EXPECT_TRUE(ParseArg(&argv, "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(log);
        EXPECT_EQ(std::string(*argv), "yolo");
    }
    {
        int nthreads = 0;
        auto argv = makeArgs("yolo --nthreads=4");
        EXPECT_FALSE(ParseArg(&argv, "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(std::string(*argv), "yolo");
        ++argv;
        EXPECT_TRUE(ParseArg(&argv, "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(*argv == nullptr);
    }
}

TEST(Args, Bool) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    bool log = true;
    bool benchmark = false;
    bool debug = false;
    auto argv = makeArgs("--log=false --benchmark --debug=true");
    EXPECT_TRUE(ParseArg(&argv, "log", &log, expectNoError));
    EXPECT_FALSE(log);
    EXPECT_TRUE(ParseArg(&argv, "benchmark", &benchmark, expectNoError));
    EXPECT_TRUE(benchmark);
    EXPECT_TRUE(ParseArg(&argv, "debug", &debug, expectNoError));
    EXPECT_TRUE(debug);
    EXPECT_TRUE(*argv == nullptr);
}

TEST(Args, ErrorMissingValue) {
    bool errCalled = false;
    auto callback = [&errCalled](const std::string &s) { errCalled = true; };

    int nthreads = 2;
    auto argv = makeArgs("--nthreads");
    EXPECT_FALSE(ParseArg(&argv, "nthreads", &nthreads, callback));
    EXPECT_EQ(nthreads, 2);
    EXPECT_TRUE(errCalled);
}

TEST(Args, ErrorMissingValueEqual) {
    bool errCalled = false;
    auto callback = [&errCalled](const std::string &s) { errCalled = true; };

    int nthreads = 2;
    auto argv = makeArgs("--nthreads=");
    EXPECT_FALSE(ParseArg(&argv, "nthreads", &nthreads, callback));
    EXPECT_EQ(nthreads, 2);
    EXPECT_TRUE(errCalled);
}

TEST(Args, ErrorBogusBool) {
    bool errCalled = false;
    auto callback = [&errCalled](const std::string &s) { errCalled = true; };

    bool log = false;
    auto argv = makeArgs("--log=tru3");
    EXPECT_FALSE(ParseArg(&argv, "log", &log, callback));
    EXPECT_FALSE(log);
    EXPECT_TRUE(errCalled);
}

TEST(Args, Normalization) {
    auto expectNoError = [](const std::string &s) { ASSERT_FALSE(true); };

    {
        int nthreads = 0;
        auto argv = makeArgs("--n_threads 4");
        EXPECT_TRUE(ParseArg(&argv, "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(*argv == nullptr);
    }
    {
        int nthreads = 0;
        auto argv = makeArgs("--nThreads=4");
        EXPECT_TRUE(ParseArg(&argv, "nthreads", &nthreads, expectNoError));
        EXPECT_EQ(nthreads, 4);
        EXPECT_TRUE(*argv == nullptr);
    }
}
