// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/print.h>

#include <pbrt/util/check.h>

#include <charconv>

namespace pbrt {

namespace detail {

std::string FloatToString(float v) {
    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, ptr);
}

std::string DoubleToString(double v) {
    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, ptr);
}

void stringPrintfRecursive(std::string *s, const char *fmt) {
    const char *c = fmt;
    // No args left; make sure there aren't any extra formatting
    // specifiers.
    while (*c) {
        if (*c == '%') {
            if (c[1] != '%')
                LOG_FATAL("Not enough optional values passed to Printf.");
            ++c;
        }
        *s += *c++;
    }
}

// 1. Copy from fmt to *s, up to the next formatting directive.
// 2. Advance fmt past the next formatting directive and return the
//    formatting directive as a string.
std::string copyToFormatString(const char **fmt_ptr, std::string *s) {
    const char *&fmt = *fmt_ptr;
    while (*fmt) {
        if (*fmt != '%') {
            *s += *fmt;
            ++fmt;
        } else if (fmt[1] == '%') {
            // "%%"; let it pass through
            *s += '%';
            *s += '%';
            fmt += 2;
        } else
            // fmt is at the start of a formatting directive.
            break;
    }

    std::string nextFmt;
    while (*fmt) {
        char c = *fmt;
        nextFmt += c;
        ++fmt;
        // Is it a conversion specifier?
        if (c == 'd' || c == 'i' || c == 'o' || c == 'u' || c == 'x' || c == 'e' ||
            c == 'E' || c == 'f' || c == 'F' || c == 'g' || c == 'G' || c == 'a' ||
            c == 'A' || c == 'c' || c == 'C' || c == 's' || c == 'S' || c == 'p')
            break;
    }

    return nextFmt;
}

}  // namespace detail

}  // namespace pbrt
