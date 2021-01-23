// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/options.h>
#include <pbrt/util/args.h>
#include <pbrt/util/error.h>
#include <pbrt/util/print.h>

#include <gtest/gtest.h>
#include <string>

using namespace pbrt;

void usage(const std::string &msg = "") {
    if (!msg.empty())
        fprintf(stderr, "pbrt_test: %s\n\n", msg.c_str());

    fprintf(stderr, R"(pbrt_test arguments:
  --list-tests                List all tests.
  --log-level <level>         Log messages at or above this level, where <level>
                              is "verbose", "error", or "fatal". Default: "error".
  --nthreads <num>            Use specified number of threads for rendering.
  --test-filter <regexp>      Regular expression of test names to run.
  --vlog-level <n>            Set VLOG verbosity. (Default: 0, disabled.)
)");

    exit(msg.empty() ? 0 : 1);
}

int main(int argc, char **argv) {
    PBRTOptions opt;
    opt.quiet = true;
    std::string logLevel = "error";
    std::string testFilter;
    bool listTests = false;

    char **origArgv = argv;
    // Process command-line arguments
    ++argv;
    while (*argv != nullptr) {
        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        if (ParseArg(&argv, "list-tests", &listTests, onError) ||
            ParseArg(&argv, "log-level", &logLevel, onError) ||
            ParseArg(&argv, "nthreads", &opt.nThreads, onError) ||
            ParseArg(&argv, "gtest-filter", &testFilter, onError) ||
            ParseArg(&argv, "test-filter", &testFilter, onError)) {
            // success
        } else if ((strcmp(*argv, "--help") == 0) || (strcmp(*argv, "-h") == 0)) {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *argv));
            return 1;
        }
    }

    opt.logLevel = LogLevelFromString(logLevel);

    InitPBRT(opt);

    int googleArgc = 1;
    const char *googleArgv[4] = {};
    googleArgv[0] = argv[0];
    std::string filter;
    if (!testFilter.empty()) {
        filter = StringPrintf("--gtest_filter=%s", testFilter);
        googleArgv[googleArgc++] = filter.c_str();
    }
    if (listTests)
        googleArgv[googleArgc++] = "--gtest_list_tests";

    testing::InitGoogleTest(&googleArgc, (char **)googleArgv);

    int ret = RUN_ALL_TESTS();

    CleanupPBRT();

    return ret;
}
