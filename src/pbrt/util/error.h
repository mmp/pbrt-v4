// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_ERROR_H
#define PBRT_UTIL_ERROR_H

#include <pbrt/pbrt.h>

#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>

#include <string>
#include <string_view>

namespace pbrt {

// FileLoc Definition
struct FileLoc {
    FileLoc() = default;
    FileLoc(std::string_view filename) : filename(filename) {}
    std::string ToString() const;

    std::string_view filename;
    int line = 1, column = 0;
};

void SuppressErrorMessages();

// Error Reporting Function Declarations
void Warning(const FileLoc *loc, const char *message);
void Error(const FileLoc *loc, const char *message);
[[noreturn]] void ErrorExit(const FileLoc *loc, const char *message);

// Error Reporting Inline Functions
template <typename... Args>
inline void Warning(const FileLoc *loc, const char *fmt, Args &&... args) {
    Warning(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Warning(const char *fmt, Args &&... args) {
    Warning(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Error(const char *fmt, Args &&... args) {
    Error(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Error(const FileLoc *loc, const char *fmt, Args &&... args) {
    Error(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
[[noreturn]] inline void ErrorExit(const char *fmt, Args &&... args) {
    ErrorExit(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
[[noreturn]] inline void ErrorExit(const FileLoc *loc, const char *fmt, Args &&... args) {
    ErrorExit(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

int LastError();
std::string ErrorString(int errorId = LastError());

}  // namespace pbrt

#endif  // PBRT_UTIL_ERROR_H
