// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_FLOAT_H
#define PBRT_UTIL_FLOAT_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#if defined(PBRT_BUILD_GPU_RENDERER) && defined(PBRT_IS_GPU_CODE)
#include <cuda_fp16.h>
#endif

namespace pbrt {

#ifdef PBRT_IS_GPU_CODE

#define DoubleOneMinusEpsilon 0x1.fffffffffffffp-1
#define FloatOneMinusEpsilon float(0x1.fffffep-1)

#ifdef PBRT_FLOAT_AS_DOUBLE
#define OneMinusEpsilon DoubleOneMinusEpsilon
#else
#define OneMinusEpsilon FloatOneMinusEpsilon
#endif

#define Infinity std::numeric_limits<Float>::infinity()
#define MachineEpsilon std::numeric_limits<Float>::epsilon() * 0.5f

#else

// Floating-point Constants
static constexpr Float Infinity = std::numeric_limits<Float>::infinity();

static constexpr Float MachineEpsilon = std::numeric_limits<Float>::epsilon() * 0.5;

static constexpr double DoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
static constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;
#ifdef PBRT_FLOAT_AS_DOUBLE
static constexpr double OneMinusEpsilon = DoubleOneMinusEpsilon;
#else
static constexpr float OneMinusEpsilon = FloatOneMinusEpsilon;
#endif

#endif  // PBRT_IS_GPU_CODE

// Floating-point Inline Functions
template <typename T>
inline PBRT_CPU_GPU typename std::enable_if_t<std::is_floating_point<T>::value, bool>
IsNaN(T v) {
#ifdef PBRT_IS_GPU_CODE
    return isnan(v);
#else
    return std::isnan(v);
#endif
}

template <typename T>
inline PBRT_CPU_GPU typename std::enable_if_t<std::is_integral<T>::value, bool> IsNaN(
    T v) {
    return false;
}

template <typename T>
inline PBRT_CPU_GPU typename std::enable_if_t<std::is_floating_point<T>::value, bool>
IsInf(T v) {
#ifdef PBRT_IS_GPU_CODE
    return isinf(v);
#else
    return std::isinf(v);
#endif
}

template <typename T>
inline PBRT_CPU_GPU typename std::enable_if_t<std::is_integral<T>::value, bool> IsInf(
    T v) {
    return false;
}

PBRT_CPU_GPU
inline float FMA(float a, float b, float c) {
    return std::fma(a, b, c);
}

PBRT_CPU_GPU
inline double FMA(double a, double b, double c) {
    return std::fma(a, b, c);
}
inline long double FMA(long double a, long double b, long double c) {
    return std::fma(a, b, c);
}

PBRT_CPU_GPU
inline uint32_t FloatToBits(float f) {
#ifdef PBRT_IS_GPU_CODE
    return __float_as_uint(f);
#else
    return pstd::bit_cast<uint32_t>(f);
#endif
}

PBRT_CPU_GPU
inline float BitsToFloat(uint32_t ui) {
#ifdef PBRT_IS_GPU_CODE
    return __uint_as_float(ui);
#else
    return pstd::bit_cast<float>(ui);
#endif
}

PBRT_CPU_GPU
inline int Exponent(float v) {
    return (FloatToBits(v) >> 23) - 127;
}

PBRT_CPU_GPU
inline int Significand(float v) {
    return FloatToBits(v) & ((1 << 23) - 1);
}

PBRT_CPU_GPU
inline uint32_t SignBit(float v) {
    return FloatToBits(v) & 0x80000000;
}

PBRT_CPU_GPU
inline uint64_t FloatToBits(double f) {
#ifdef PBRT_IS_GPU_CODE
    return __double_as_longlong(f);
#else
    return pstd::bit_cast<uint64_t>(f);
#endif
}

PBRT_CPU_GPU
inline double BitsToFloat(uint64_t ui) {
#ifdef PBRT_IS_GPU_CODE
    return __longlong_as_double(ui);
#else
    return pstd::bit_cast<double>(ui);
#endif
}

PBRT_CPU_GPU
inline float NextFloatUp(float v) {
    // Handle infinity and negative zero for _NextFloatUp()_
    if (IsInf(v) && v > 0.f)
        return v;
    if (v == -0.f)
        v = 0.f;

    // Advance _v_ to next higher float
    uint32_t ui = FloatToBits(v);
    if (v >= 0)
        ++ui;
    else
        --ui;
    return BitsToFloat(ui);
}

PBRT_CPU_GPU
inline float NextFloatDown(float v) {
    // Handle infinity and positive zero for _NextFloatDown()_
    if (IsInf(v) && v < 0.)
        return v;
    if (v == 0.f)
        v = -0.f;
    uint32_t ui = FloatToBits(v);
    if (v > 0)
        --ui;
    else
        ++ui;
    return BitsToFloat(ui);
}

inline constexpr Float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

inline PBRT_CPU_GPU Float AddRoundUp(Float a, Float b) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __dadd_ru(a, b);
#else
    return __fadd_ru(a, b);
#endif
#else  // GPU
    return NextFloatUp(a + b);
#endif
}
inline PBRT_CPU_GPU Float AddRoundDown(Float a, Float b) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __dadd_rd(a, b);
#else
    return __fadd_rd(a, b);
#endif
#else  // GPU
    return NextFloatDown(a + b);
#endif
}

inline PBRT_CPU_GPU Float SubRoundUp(Float a, Float b) {
    return AddRoundUp(a, -b);
}
inline PBRT_CPU_GPU Float SubRoundDown(Float a, Float b) {
    return AddRoundDown(a, -b);
}

inline PBRT_CPU_GPU Float MulRoundUp(Float a, Float b) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __dmul_ru(a, b);
#else
    return __fmul_ru(a, b);
#endif
#else  // GPU
    return NextFloatUp(a * b);
#endif
}

inline PBRT_CPU_GPU Float MulRoundDown(Float a, Float b) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __dmul_rd(a, b);
#else
    return __fmul_rd(a, b);
#endif
#else  // GPU
    return NextFloatDown(a * b);
#endif
}

inline PBRT_CPU_GPU Float DivRoundUp(Float a, Float b) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __ddiv_ru(a, b);
#else
    return __fdiv_ru(a, b);
#endif
#else  // GPU
    return NextFloatUp(a / b);
#endif
}

inline PBRT_CPU_GPU Float DivRoundDown(Float a, Float b) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __ddiv_rd(a, b);
#else
    return __fdiv_rd(a, b);
#endif
#else  // GPU
    return NextFloatDown(a / b);
#endif
}

inline PBRT_CPU_GPU Float SqrtRoundUp(Float a) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __dsqrt_ru(a);
#else
    return __fsqrt_ru(a);
#endif
#else  // GPU
    return NextFloatUp(std::sqrt(a));
#endif
}

inline PBRT_CPU_GPU Float SqrtRoundDown(Float a) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __dsqrt_rd(a);
#else
    return __fsqrt_rd(a);
#endif
#else  // GPU
    return std::max<Float>(0, NextFloatDown(std::sqrt(a)));
#endif
}

inline PBRT_CPU_GPU Float FMARoundUp(Float a, Float b, Float c) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __fma_ru(a, b, c);  // FIXME: what to do here?
#else
    return __fma_ru(a, b, c);
#endif
#else  // GPU
    return NextFloatUp(FMA(a, b, c));
#endif
}

inline PBRT_CPU_GPU Float FMARoundDown(Float a, Float b, Float c) {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
    return __fma_rd(a, b, c);  // FIXME: what to do here?
#else
    return __fma_rd(a, b, c);
#endif
#else  // GPU
    return NextFloatDown(FMA(a, b, c));
#endif
}

PBRT_CPU_GPU
inline double NextFloatUp(double v) {
    if (IsInf(v) && v > 0.)
        return v;
    if (v == -0.f)
        v = 0.f;
    uint64_t ui = FloatToBits(v);
    if (v >= 0.)
        ++ui;
    else
        --ui;
    return BitsToFloat(ui);
}

PBRT_CPU_GPU
inline double NextFloatDown(double v) {
    if (IsInf(v) && v < 0.)
        return v;
    if (v == 0.f)
        v = -0.f;
    uint64_t ui = FloatToBits(v);
    if (v > 0.)
        --ui;
    else
        ++ui;
    return BitsToFloat(ui);
}

PBRT_CPU_GPU
inline int Exponent(double d) {
    return (FloatToBits(d) >> 52) - 1023;
}

PBRT_CPU_GPU
inline uint64_t Significand(double d) {
    return FloatToBits(d) & ((1ull << 52) - 1);
}

PBRT_CPU_GPU
inline uint64_t SignBit(double v) {
    return FloatToBits(v) & 0x8000000000000000;
}

PBRT_CPU_GPU
inline double FlipSign(double a, double b) {
    return BitsToFloat(FloatToBits(a) ^ SignBit(b));
}

static const int HalfExponentMask = 0b0111110000000000;
static const int HalfSignificandMask = 0b1111111111;
static const int HalfNegativeZero = 0b1000000000000000;
static const int HalfPositiveZero = 0;
// Exponent all 1s, significand zero
static const int HalfNegativeInfinity = 0b1111110000000000;
static const int HalfPositiveInfinity = 0b0111110000000000;

namespace {

// TODO: support for non-AVX systems, check CPUID stuff, etc..

// https://gist.github.com/rygorous/2156668
union FP32 {
    uint32_t u;
    float f;
    struct {
        unsigned int Mantissa : 23;
        unsigned int Exponent : 8;
        unsigned int Sign : 1;
    };
};

union FP16 {
    uint16_t u;
    struct {
        unsigned int Mantissa : 10;
        unsigned int Exponent : 5;
        unsigned int Sign : 1;
    };
};

}  // namespace

class Half {
  public:
    Half() = default;
    Half(const Half &) = default;
    Half &operator=(const Half &) = default;

    PBRT_CPU_GPU
    static Half FromBits(uint16_t v) { return Half(v); }

    PBRT_CPU_GPU
    explicit Half(float ff) {
#ifdef PBRT_IS_GPU_CODE
        h = __half_as_ushort(__float2half(ff));
#else
        // Rounding ties to nearest even instead of towards +inf
        FP32 f;
        f.f = ff;
        FP32 f32infty = {255 << 23};
        FP32 f16max = {(127 + 16) << 23};
        FP32 denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
        unsigned int sign_mask = 0x80000000u;
        FP16 o = {0};

        unsigned int sign = f.u & sign_mask;
        f.u ^= sign;

        // NOTE all the integer compares in this function can be safely
        // compiled into signed compares since all operands are below
        // 0x80000000. Important if you want fast straight SSE2 code
        // (since there's no unsigned PCMPGTD).

        if (f.u >= f16max.u)  // result is Inf or NaN (all exponent bits set)
            o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
        else {                                           // (De)normalized number or zero
            if (f.u < (113 << 23)) {  // resulting FP16 is subnormal or zero
                // use a magic value to align our 10 mantissa bits at the bottom
                // of the float. as long as FP addition is round-to-nearest-even
                // this just works.
                f.f += denorm_magic.f;

                // and one integer subtract of the bias later, we have our final
                // float!
                o.u = f.u - denorm_magic.u;
            } else {
                unsigned int mant_odd = (f.u >> 13) & 1;  // resulting mantissa is odd

                // update exponent, rounding bias part 1
                f.u += (uint32_t(15 - 127) << 23) + 0xfff;
                // rounding bias part 2
                f.u += mant_odd;
                // take the bits!
                o.u = f.u >> 13;
            }
        }

        o.u |= sign >> 16;
        h = o.u;
#endif
    }
    PBRT_CPU_GPU
    explicit Half(double d) : Half(float(d)) {}

    PBRT_CPU_GPU
    explicit operator float() const {
#ifdef PBRT_IS_GPU_CODE
        return __half2float(__ushort_as_half(h));
#else
        FP16 h;
        h.u = this->h;
        static const FP32 magic = {113 << 23};
        static const unsigned int shifted_exp = 0x7c00
                                                << 13;  // exponent mask after shift
        FP32 o;

        o.u = (h.u & 0x7fff) << 13;            // exponent/mantissa bits
        unsigned int exp = shifted_exp & o.u;  // just the exponent
        o.u += (127 - 15) << 23;               // exponent adjust

        // handle exponent special cases
        if (exp == shifted_exp)       // Inf/NaN?
            o.u += (128 - 16) << 23;  // extra exp adjust
        else if (exp == 0) {          // Zero/Denormal?
            o.u += 1 << 23;           // extra exp adjust
            o.f -= magic.f;           // renormalize
        }

        o.u |= (h.u & 0x8000) << 16;  // sign bit
        return o.f;
#endif
    }
    PBRT_CPU_GPU
    explicit operator double() const { return (float)(*this); }

    PBRT_CPU_GPU
    bool operator==(const Half &v) const {
#ifdef PBRT_IS_GPU_CODE
        return __ushort_as_half(h) == __ushort_as_half(v.h);
#else
        if (Bits() == v.Bits())
            return true;
        return ((Bits() == HalfNegativeZero && v.Bits() == HalfPositiveZero) ||
                (Bits() == HalfPositiveZero && v.Bits() == HalfNegativeZero));
#endif
    }
    PBRT_CPU_GPU
    bool operator!=(const Half &v) const { return !(*this == v); }

    PBRT_CPU_GPU
    Half operator-() const { return FromBits(h ^ (1 << 15)); }

    PBRT_CPU_GPU
    uint16_t Bits() const { return h; }

    PBRT_CPU_GPU
    int Sign() { return (h >> 15) ? -1 : 1; }

    PBRT_CPU_GPU
    bool IsInf() { return h == HalfPositiveInfinity || h == HalfNegativeInfinity; }

    PBRT_CPU_GPU
    bool IsNaN() {
        return ((h & HalfExponentMask) == HalfExponentMask &&
                (h & HalfSignificandMask) != 0);
    }

    PBRT_CPU_GPU
    Half NextUp() {
        if (IsInf() && Sign() == 1)
            return *this;

        Half up = *this;
        if (up.h == HalfNegativeZero)
            up.h = HalfPositiveZero;
        // Advance _v_ to next higher float
        if (up.Sign() >= 0)
            ++up.h;
        else
            --up.h;
        return up;
    }

    PBRT_CPU_GPU
    Half NextDown() {
        if (IsInf() && Sign() == -1)
            return *this;

        Half down = *this;
        if (down.h == HalfPositiveZero)
            down.h = HalfNegativeZero;
        if (down.Sign() >= 0)
            --down.h;
        else
            ++down.h;
        return down;
    }

    std::string ToString() const;

  private:
    PBRT_CPU_GPU
    explicit Half(uint16_t h) : h(h) {}

    uint16_t h;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_FLOAT_H
