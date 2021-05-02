// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_CHECK_H
#define PBRT_UTIL_CHECK_H

#include <pbrt/pbrt.h>

#include <pbrt/util/log.h>
#include <pbrt/util/stats.h>

#include <functional>
#include <string>
#include <vector>

namespace pbrt {

void PrintStackTrace();

#ifdef PBRT_IS_GPU_CODE

#define CHECK(x) assert(x)
#define CHECK_IMPL(a, b, op) assert((a)op(b))

#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

#else

// CHECK Macro Definitions
#define CHECK(x) (!(!(x) && (LOG_FATAL("Check failed: %s", #x), true)))

#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

// CHECK\_IMPL Macro Definition
#define CHECK_IMPL(a, b, op)                                                           \
    do {                                                                               \
        auto va = a;                                                                   \
        auto vb = b;                                                                   \
        if (!(va op vb))                                                               \
            LOG_FATAL("Check failed: %s " #op " %s with %s = %s, %s = %s", #a, #b, #a, \
                      va, #b, vb);                                                     \
    } while (false) /* swallow semicolon */

#endif  // PBRT_IS_GPU_CODE

#ifdef PBRT_DEBUG_BUILD

#define DCHECK(x) (CHECK(x))
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)

#else

#define EMPTY_CHECK \
    do {            \
    } while (false) /* swallow semicolon */

// Use an empty check (rather than expanding the macros to nothing) to swallow the
// semicolon at the end, and avoid empty if-statements.
#define DCHECK(x) EMPTY_CHECK

#define DCHECK_EQ(a, b) EMPTY_CHECK
#define DCHECK_NE(a, b) EMPTY_CHECK
#define DCHECK_GT(a, b) EMPTY_CHECK
#define DCHECK_GE(a, b) EMPTY_CHECK
#define DCHECK_LT(a, b) EMPTY_CHECK
#define DCHECK_LE(a, b) EMPTY_CHECK

#endif

#define CHECK_RARE_TO_STRING(x) #x
#define CHECK_RARE_EXPAND_AND_TO_STRING(x) CHECK_RARE_TO_STRING(x)

#ifdef PBRT_IS_GPU_CODE

#define CHECK_RARE(freq, condition)
#define DCHECK_RARE(freq, condition)

#else

#define CHECK_RARE(freq, condition)                                                     \
    static_assert(std::is_floating_point<decltype(freq)>::value,                        \
                  "Expected floating-point frequency as first argument to CHECK_RARE"); \
    static_assert(std::is_integral<decltype(condition)>::value,                         \
                  "Expected Boolean condition as second argument to CHECK_RARE");       \
    do {                                                                                \
        static thread_local int64_t numTrue, total;                                     \
        static StatRegisterer reg([](StatsAccumulator &accum) {                         \
            accum.ReportRareCheck(__FILE__ " " CHECK_RARE_EXPAND_AND_TO_STRING(         \
                                      __LINE__) ": CHECK_RARE failed: " #condition,     \
                                  freq, numTrue, total);                                \
            numTrue = total = 0;                                                        \
        });                                                                             \
        ++total;                                                                        \
        if (condition)                                                                  \
            ++numTrue;                                                                  \
    } while (0)

#ifdef PBRT_DEBUG_BUILD
#define DCHECK_RARE(freq, condition) CHECK_RARE(freq, condition)
#else
#define DCHECK_RARE(freq, condition)
#endif  // NDEBUG

#endif  // PBRT_IS_GPU_CODE

// CheckCallbackScope Definition
class CheckCallbackScope {
  public:
    // CheckCallbackScope Public Methods
    CheckCallbackScope(std::function<std::string(void)> callback);

    ~CheckCallbackScope();

    CheckCallbackScope(const CheckCallbackScope &) = delete;
    CheckCallbackScope &operator=(const CheckCallbackScope &) = delete;

    static void Fail();

  private:
    // CheckCallbackScope Private Members
    static std::vector<std::function<std::string(void)>> callbacks;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_CHECK_H
