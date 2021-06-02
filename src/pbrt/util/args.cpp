// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/args.h>

#ifdef PBRT_IS_WINDOWS
#include <Windows.h>
#endif

namespace pbrt {

std::vector<std::string> GetCommandLineArguments(char *argv[]) {
    std::vector<std::string> argStrings;
#ifdef PBRT_IS_WINDOWS
    // Handle UTF16-encoded arguments on Windows
    int argc;
    LPWSTR *argvw = CommandLineToArgvW(GetCommandLineW(), &argc);
    CHECK(argv != nullptr);
    for (int i = 1; i < argc; ++i) {
        std::u16string su16 = WStringToU16String(argvw[i]);
        argStrings.push_back(UTF16ToUTF8(su16));
    }
#else
    ++argv;  // skip executable name
    while (*argv) {
        argStrings.push_back(*argv);
        ++argv;
    }
#endif  // PBRT_IS_WINDOWS
    return argStrings;
}

}  // namespace pbrt
