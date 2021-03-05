// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_PRINT_H
#define PBRT_UTIL_PRINT_H

#include <pbrt/pbrt.h>

#include <string>

// Hack: make util/log.h happy
namespace pbrt {
template <typename... Args>
inline std::string StringPrintf(const char *fmt, Args &&...args);
}

#include <pbrt/util/log.h>

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <typeinfo>

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif  // __GNUG__

namespace pbrt {

// helpers, fwiw
template <typename T>
static auto operator<<(std::ostream &os, const T &v) -> decltype(v.ToString(), os) {
    return os << v.ToString();
}
template <typename T>
static auto operator<<(std::ostream &os, const T &v) -> decltype(ToString(v), os) {
    return os << ToString(v);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::shared_ptr<T> &p) {
    if (p)
        return os << p->ToString();
    else
        return os << "(nullptr)";
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::unique_ptr<T> &p) {
    if (p)
        return os << p->ToString();
    else
        return os << "(nullptr)";
}

namespace detail {

std::string FloatToString(float v);
std::string DoubleToString(double v);

template <typename T>
struct IntegerFormatTrait;

template <>
struct IntegerFormatTrait<bool> {
    static constexpr const char *fmt() { return "d"; }
};
template <>
struct IntegerFormatTrait<char> {
    static constexpr const char *fmt() { return "d"; }
};
template <>
struct IntegerFormatTrait<unsigned char> {
    static constexpr const char *fmt() { return "d"; }
};
template <>
struct IntegerFormatTrait<int> {
    static constexpr const char *fmt() { return "d"; }
};
template <>
struct IntegerFormatTrait<unsigned int> {
    static constexpr const char *fmt() { return "u"; }
};
template <>
struct IntegerFormatTrait<short> {
    static constexpr const char *fmt() { return "d"; }
};
template <>
struct IntegerFormatTrait<unsigned short> {
    static constexpr const char *fmt() { return "u"; }
};
template <>
struct IntegerFormatTrait<long> {
    static constexpr const char *fmt() { return "ld"; }
};
template <>
struct IntegerFormatTrait<unsigned long> {
    static constexpr const char *fmt() { return "lu"; }
};
template <>
struct IntegerFormatTrait<long long> {
    static constexpr const char *fmt() { return "lld"; }
};
template <>
struct IntegerFormatTrait<unsigned long long> {
    static constexpr const char *fmt() { return "llu"; }
};
#ifdef PBRT_INT64_IS_OWN_TYPE
template <>
struct IntegerFormatTrait<int64_t> {
    static constexpr const char *fmt() { return PRId64; }
};
template <>
struct IntegerFormatTrait<uint64_t> {
    static constexpr const char *fmt() { return PRIu64; }
};
#endif

template <typename T>
using HasSize =
    std::is_integral<typename std::decay_t<decltype(std::declval<T &>().size())>>;

template <typename T>
using HasData =
    std::is_pointer<typename std::decay_t<decltype(std::declval<T &>().data())>>;

// Don't use size()/data()-based operator<< for std::string...
inline std::ostream &operator<<(std::ostream &os, const std::string &str) {
    return std::operator<<(os, str);
}

template <typename T>
inline std::enable_if_t<HasSize<T>::value && HasData<T>::value, std::ostream &>
operator<<(std::ostream &os, const T &v) {
    os << "[ ";
    auto ptr = v.data();
    for (size_t i = 0; i < v.size(); ++i) {
        os << ptr[i];
        if (i < v.size() - 1)
            os << ", ";
    }
    return os << " ]";
}

// base case
void stringPrintfRecursive(std::string *s, const char *fmt);

// 1. Copy from fmt to *s, up to the next formatting directive.
// 2. Advance fmt past the next formatting directive and return the
//    formatting directive as a string.
std::string copyToFormatString(const char **fmt_ptr, std::string *s);

template <typename T>
inline typename std::enable_if_t<!std::is_class<typename std::decay_t<T>>::value,
                                 std::string>
formatOne(const char *fmt, T &&v) {
    // Figure out how much space we need to allocate; add an extra
    // character for the '\0'.
    size_t size = snprintf(nullptr, 0, fmt, v) + 1;
    std::string str;
    str.resize(size);
    snprintf(&str[0], size, fmt, v);
    str.pop_back();  // remove trailing NUL
    return str;
}

template <typename T>
inline
    typename std::enable_if_t<std::is_class<typename std::decay_t<T>>::value, std::string>
    formatOne(const char *fmt, T &&v) {
    LOG_FATAL("Printf: Non-basic type %s passed for format string %s", typeid(v).name(),
              fmt);
    return "";
}

template <typename T, typename... Args>
inline void stringPrintfRecursive(std::string *s, const char *fmt, T &&v, Args &&...args);

template <typename T, typename... Args>
inline void stringPrintfRecursiveWithPrecision(std::string *s, const char *fmt,
                                               const std::string &nextFmt, T &&v,
                                               Args &&...args) {
    LOG_FATAL("MEH");
}

template <typename T, typename... Args>
inline typename std::enable_if_t<!std::is_class<typename std::decay_t<T>>::value, void>
stringPrintfRecursiveWithPrecision(std::string *s, const char *fmt,
                                   const std::string &nextFmt, int precision, T &&v,
                                   Args &&...args) {
    size_t size = snprintf(nullptr, 0, nextFmt.c_str(), precision, v) + 1;
    std::string str;
    str.resize(size);
    snprintf(&str[0], size, nextFmt.c_str(), precision, v);
    str.pop_back();  // remove trailing NUL
    *s += str;

    stringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
}

#ifdef PBRT_IS_MSVC
#pragma warning(push)
#pragma warning(disable : 4102)  // bogus "unreferenced label" warning for done: below
#endif

// General-purpose version of stringPrintfRecursive; add the formatted
// output for a single StringPrintf() argument to the final result string
// in *s.
template <typename T, typename... Args>
inline void stringPrintfRecursive(std::string *s, const char *fmt, T &&v,
                                  Args &&...args) {
    std::string nextFmt = copyToFormatString(&fmt, s);
    bool precisionViaArg = nextFmt.find('*') != std::string::npos;

    bool isSFmt = nextFmt.find('s') != std::string::npos;
    bool isDFmt = nextFmt.find('d') != std::string::npos;

    if constexpr (std::is_integral<std::decay_t<T>>::value) {
        if (precisionViaArg) {
            stringPrintfRecursiveWithPrecision(s, fmt, nextFmt, v,
                                               std::forward<Args>(args)...);
            return;
        }
    } else if (precisionViaArg)
        LOG_FATAL("Non-integral type provided for %* format.");

    if constexpr (std::is_same<std::decay_t<T>, float>::value)
        if (nextFmt == "%f" || nextFmt == "%s") {
            *s += detail::FloatToString(v);
            goto done;
        }

    if constexpr (std::is_same<std::decay_t<T>, double>::value)
        if (nextFmt == "%f" || nextFmt == "%s") {
            *s += detail::DoubleToString(v);
            goto done;
        }

    if constexpr (std::is_same<std::decay_t<T>, bool>::value)  // FIXME: %-10s with bool
        if (isSFmt) {
            *s += bool(v) ? "true" : "false";
            goto done;
        }

    if constexpr (std::is_integral<std::decay_t<T>>::value) {
        if (isDFmt) {
            nextFmt.replace(nextFmt.find('d'), 1,
                            detail::IntegerFormatTrait<std::decay_t<T>>::fmt());
            *s += formatOne(nextFmt.c_str(), std::forward<T>(v));
            goto done;
        }
    } else if (isDFmt)
        LOG_FATAL("Non-integral type passed to %d format.");

    if (isSFmt) {
        std::stringstream ss;
        ss << v;
        *s += formatOne(nextFmt.c_str(), ss.str().c_str());
    } else if (!nextFmt.empty())
        *s += formatOne(nextFmt.c_str(), std::forward<T>(v));
    else
        LOG_FATAL("Excess values passed to Printf.");

done:
    stringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
}

#ifdef PBRT_IS_MSVC
#pragma warning(pop)
#endif

}  // namespace detail

// Printing Function Declarations
template <typename... Args>
void Printf(const char *fmt, Args &&...args);
template <typename... Args>
inline std::string StringPrintf(const char *fmt, Args &&...args);

template <typename... Args>
inline std::string StringPrintf(const char *fmt, Args &&...args) {
    std::string ret;
    detail::stringPrintfRecursive(&ret, fmt, std::forward<Args>(args)...);
    return ret;
}

template <typename... Args>
void Printf(const char *fmt, Args &&...args) {
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    fputs(s.c_str(), stdout);
}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif  // __GNUG__

// https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
inline std::string Red(const std::string &s) {
    const char *red = "\033[1m\033[31m";  // bold red
    const char *reset = "\033[0m";
    return std::string(red) + s + std::string(reset);
}

inline std::string Yellow(const std::string &s) {
    // https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    const char *yellow = "\033[1m\033[38;5;100m";
    const char *reset = "\033[0m";
    return std::string(yellow) + s + std::string(reset);
}

inline std::string Green(const std::string &s) {
    // https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    const char *green = "\033[1m\033[38;5;22m";
    const char *reset = "\033[0m";
    return std::string(green) + s + std::string(reset);
}

}  // namespace pbrt

#endif  // PBRT_UTIL_PRINT_H
