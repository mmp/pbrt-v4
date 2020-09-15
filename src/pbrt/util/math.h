// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_MATH_H
#define PBRT_UTIL_MATH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/bits.h>
#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

#ifdef PBRT_HAS_INTRIN_H
#include <intrin.h>
#endif  // PBRT_HAS_INTRIN_H

namespace pbrt {

#ifdef PBRT_IS_GPU_CODE

#define ShadowEpsilon 0.0001f
#define Pi Float(3.14159265358979323846)
#define InvPi Float(0.31830988618379067154)
#define Inv2Pi Float(0.15915494309189533577)
#define Inv4Pi Float(0.07957747154594766788)
#define PiOver2 Float(1.57079632679489661923)
#define PiOver4 Float(0.78539816339744830961)
#define Sqrt2 Float(1.41421356237309504880)

#else

// Mathematical Constants
constexpr Float ShadowEpsilon = 0.0001f;

constexpr Float Pi = 3.14159265358979323846;
constexpr Float InvPi = 0.31830988618379067154;
constexpr Float Inv2Pi = 0.15915494309189533577;
constexpr Float Inv4Pi = 0.07957747154594766788;
constexpr Float PiOver2 = 1.57079632679489661923;
constexpr Float PiOver4 = 0.78539816339744830961;
constexpr Float Sqrt2 = 1.41421356237309504880;

#endif

// CompensatedSum Definition
template <typename Float>
class CompensatedSum {
  public:
    CompensatedSum() = default;
    PBRT_CPU_GPU
    explicit CompensatedSum(Float v) : sum(v) {}

    PBRT_CPU_GPU
    CompensatedSum &operator=(Float v) {
        sum = v;
        c = 0;
        return *this;
    }

    PBRT_CPU_GPU
    CompensatedSum &operator+=(Float v) {
        Float delta = v - c;
        Float newSum = sum + delta;
        c = (newSum - sum) - delta;
        sum = newSum;
        return *this;
    }

    PBRT_CPU_GPU
    explicit operator Float() const { return sum; }

    std::string ToString() const;

  private:
    Float sum = 0., c = 0.;
};

// CompensatedFloat Definition
struct CompensatedFloat {
  public:
    PBRT_CPU_GPU
    CompensatedFloat(Float v, Float err) : v(v), err(err) {}

    PBRT_CPU_GPU
    explicit operator float() const { return v + err; }
    PBRT_CPU_GPU
    explicit operator double() const { return double(v) + double(err); }

    Float v, err;

    std::string ToString() const;
};

template <int N>
class SquareMatrix;

// Math Inline Functions
PBRT_CPU_GPU
inline Float Lerp(Float t, Float a, Float b) {
    return (1 - t) * a + t * b;
}

// http://www.plunk.org/~hatch/rightway.php
PBRT_CPU_GPU
inline Float SinXOverX(Float x) {
    if (1 + x * x == 1)
        return 1;
    return std::sin(x) / x;
}

PBRT_CPU_GPU
inline Float Sinc(Float x) {
    return SinXOverX(Pi * x);
}

PBRT_CPU_GPU
inline Float WindowedSinc(Float x, Float radius, Float tau) {
    if (std::abs(x) > radius)
        return 0;
    Float lanczos = Sinc(x / tau);
    return Sinc(x) * lanczos;
}

#ifdef PBRT_IS_MSVC
#pragma warning(push)
#pragma warning(disable : 4018)  // signed/unsigned mismatch
#endif

template <typename T, typename U, typename V>
PBRT_CPU_GPU inline constexpr T Clamp(T val, U low, V high) {
    if (val < low)
        return low;
    else if (val > high)
        return high;
    else
        return val;
}

#ifdef PBRT_IS_MSVC
#pragma warning(pop)
#endif

template <typename T>
PBRT_CPU_GPU inline T Mod(T a, T b) {
    T result = a - (a / b) * b;
    return (T)((result < 0) ? result + b : result);
}

template <>
PBRT_CPU_GPU inline Float Mod(Float a, Float b) {
    return std::fmod(a, b);
}

// (0,0): v[0], (1, 0): v[1], (0, 1): v[2], (1, 1): v[3]
PBRT_CPU_GPU
inline Float Bilerp(pstd::array<Float, 2> p, pstd::span<const Float> v) {
    return ((1 - p[0]) * (1 - p[1]) * v[0] + p[0] * (1 - p[1]) * v[1] +
            (1 - p[0]) * p[1] * v[2] + p[0] * p[1] * v[3]);
}

PBRT_CPU_GPU
inline Float Radians(Float deg) {
    return (Pi / 180) * deg;
}
PBRT_CPU_GPU
inline Float Degrees(Float rad) {
    return (180 / Pi) * rad;
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
// Needed so can use e.g. EvaluatePolynomial() with ints
template <typename T>
PBRT_CPU_GPU inline typename std::enable_if_t<std::is_integral<T>::value, T> FMA(T a, T b,
                                                                                 T c) {
    return a * b + c;
}

PBRT_CPU_GPU
inline float SafeSqrt(float x) {
    DCHECK_GE(x, -1e-3f);  // not too negative
    return std::sqrt(std::max(0.f, x));
}

PBRT_CPU_GPU
inline double SafeSqrt(double x) {
    DCHECK_GE(x, -1e-3);  // not too negative
    return std::sqrt(std::max(0., x));
}

template <typename T>
PBRT_CPU_GPU inline constexpr T Sqr(T v) {
    return v * v;
}

// Would be nice to allow Float to be a template type here, but it's tricky:
// https://stackoverflow.com/questions/5101516/why-function-template-cannot-be-partially-specialized
template <int n>
PBRT_CPU_GPU inline constexpr float Pow(float v) {
    static_assert(n > 0, "Power can't be negative");
    float n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template <>
PBRT_CPU_GPU inline constexpr float Pow<1>(float v) {
    return v;
}

template <>
PBRT_CPU_GPU inline constexpr float Pow<0>(float v) {
    return 1;
}

template <int n>
PBRT_CPU_GPU inline constexpr double Pow(double v) {
    static_assert(n > 0, "Power can't be negative");
    double n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template <>
PBRT_CPU_GPU inline constexpr double Pow<1>(double v) {
    return v;
}

template <>
PBRT_CPU_GPU inline constexpr double Pow<0>(double v) {
    return 1;
}

template <typename Float, typename C>
PBRT_CPU_GPU inline constexpr Float EvaluatePolynomial(Float t, C c) {
    return c;
}

template <typename Float, typename C, typename... Args>
PBRT_CPU_GPU inline constexpr Float EvaluatePolynomial(Float t, C c, Args... cRemaining) {
    using FMAT = typename std::common_type<Float, C>::type;
    return FMA(FMAT(t), FMAT(EvaluatePolynomial(t, cRemaining...)), FMAT(c));
}

PBRT_CPU_GPU
inline Float ApproxSin(Float x) {
#ifdef PBRT_IS_GPU_CODE
    return __sinf(x * (Pi / 4));
#else
    DCHECK_RARE(1e-5, x < 0 || x > 2);
    x = Clamp(x, 0, 2);

    // Coefficients for minimax approximation of sin(x*pi/4), x=[0,2].
    const float s1 = 0.7853975892066955566406250000000000f;
    const float s2 = -0.0807407423853874206542968750000000f;
    const float s3 = 0.0024843954015523195266723632812500f;
    const float s4 = -0.0000341485538228880614042282104492f;
    return EvaluatePolynomial(x * x, s1, s2, s3, s4) * x;
#endif
}

PBRT_CPU_GPU
inline Float ApproxCos(Float x) {
#ifdef PBRT_IS_GPU_CODE
    return __cosf(x * (Pi / 4));
#else
    DCHECK_RARE(1e-5, x < 0 || x > 2);
    x = Clamp(x, 0, 2);

    // Coefficients for minimax approximation of cos(x*pi/4), x=[0,2].
    const float c1 = 0.9999932952821962577665326692990000f;
    const float c2 = -0.3083711259464511647371969120320000f;
    const float c3 = 0.0157862649459062213825197189573000f;
    const float c4 = -0.0002983708648233575495551227373110f;
    return EvaluatePolynomial(x * x, c1, c2, c3, c4);
#endif
}

PBRT_CPU_GPU
inline float SafeASin(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}

PBRT_CPU_GPU
inline double SafeASin(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}

PBRT_CPU_GPU
inline float SafeACos(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

PBRT_CPU_GPU
inline double SafeACos(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

PBRT_CPU_GPU
inline Float Log2(Float x) {
    const Float invLog2 = 1.442695040888963387004650940071;
    return std::log(x) * invLog2;
}

PBRT_CPU_GPU
inline int Log2Int(float v) {
    DCHECK_GT(v, 0);
    if (v < 1)
        return -Log2Int(1 / v);
    // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    // (With an additional check of the significant to get round-to-nearest
    // rather than round down.)
    // midsignif = Significand(std::pow(2., 1.5))
    // i.e. grab the significand of a value halfway between two exponents,
    // in log space.
    const uint32_t midsignif = 0b00000000001101010000010011110011;
    return Exponent(v) + ((Significand(v) >= midsignif) ? 1 : 0);
}

PBRT_CPU_GPU
inline int Log2Int(double v) {
    DCHECK_GT(v, 0);
    if (v < 1)
        return -Log2Int(1 / v);
    // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    // (With an additional check of the significant to get round-to-nearest
    // rather than round down.)
    // midsignif = Significand(std::pow(2., 1.5))
    // i.e. grab the significand of a value halfway between two exponents,
    // in log space.
    const uint64_t midsignif = 0b110101000001001111001100110011111110011101111001101;
    return Exponent(v) + ((Significand(v) >= midsignif) ? 1 : 0);
}

PBRT_CPU_GPU
inline int Log2Int(uint32_t v) {
#ifdef PBRT_IS_GPU_CODE
    return 31 - __clz(v);
#elif defined(PBRT_HAS_INTRIN_H)
    unsigned long lz = 0;
    if (_BitScanReverse(&lz, v))
        return lz;
    return 0;
#else
    return 31 - __builtin_clz(v);
#endif
}

PBRT_CPU_GPU
inline int Log2Int(int32_t v) {
    return Log2Int((uint32_t)v);
}

PBRT_CPU_GPU
inline int Log2Int(uint64_t v) {
#ifdef PBRT_IS_GPU_CODE
    return 64 - __clzll(v);
#elif defined(PBRT_HAS_INTRIN_H)
    unsigned long lz = 0;
#if defined(_WIN64)
    _BitScanReverse64(&lz, v);
#else
    if (_BitScanReverse(&lz, v >> 32))
        lz += 32;
    else
        _BitScanReverse(&lz, v & 0xffffffff);
#endif  // _WIN64
    return lz;
#else   // PBRT_HAS_INTRIN_H
    return 63 - __builtin_clzll(v);
#endif
}

PBRT_CPU_GPU
inline int Log2Int(int64_t v) {
    return Log2Int((uint64_t)v);
}

// log4(x) = log2(x) / log2(4) = 1/2 log2(x) = log2(x) >> 1
template <typename T>
PBRT_CPU_GPU inline int Log4Int(T v) {
    return Log2Int(v) >> 1;
}

// https://stackoverflow.com/a/10792321
PBRT_CPU_GPU
inline float FastExp(float x) {
#ifdef PBRT_IS_GPU_CODE
    return __expf(x);
#else
    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    float t = x * 1.442695041f;
    float fi = std::floor(t);
    float f = t - fi;
    int i = (int)fi;

    // TODO: figure out what these should be.
    if (i < -30)
        return 0;
    if (i > 30)
        return Infinity;

    // built-in exp on OSX is about 69ms, so 1.55x speedup.
    // approximations to 2^f over [0,1]...

    // quadratic:
    // quadratic, from stack exchange: max error: 0.001725, 40.3ms
    // float twoToF = EvaluatePolynomial(f, 1.00172476f, 0.657636276f,
    // 0.3371894346f);

    // mathematica quadratic polynomial via Fit: max error: 0.003699, 40.3ms
    // float twoToF = EvaluatePolynomial(f, 1.00375f, 0.649445f, 0.342662f);

    // mathematica quadratic polynomial via FindFit, Linfinity norm
    // max error: 0.002431, 40.5ms
    // float twoToF = EvaluatePolynomial(f, 1.00248f, 0.651047f, 0.344001f);

    // cubic polynomial via Fit: max error: 0.000183, 44.5ms
    float twoToF = EvaluatePolynomial(f, 0.999813f, 0.696834f, 0.224131f, 0.0790209f);

    // mathematcia rational polynomial via FindFit: max err 0.00059, 47.291ms
    // float twoToF = EvaluatePolynomial(f, 0.263371f, 0.128038f, 0.0268998f) /
    //(0.263355f - 0.0542072f * f);

    // scale by 2^i, including the case of i being negative....
    uint32_t bits = FloatToBits(twoToF);
    bits += (i << 23);
    return BitsToFloat(bits);
#endif
}

PBRT_CPU_GPU
inline Float Gaussian(Float x, Float mu = 0, Float sigma = 1) {
    return 1 / std::sqrt(2 * Pi * sigma * sigma) *
           FastExp(-Sqr(x - mu) / (2 * sigma * sigma));
}

PBRT_CPU_GPU
inline Float GaussianIntegral(Float x0, Float x1, Float mu = 0, Float sigma = 1) {
    DCHECK_GT(sigma, 0);
    Float sigmaRoot2 = sigma * Float(1.414213562373095);
    return 0.5f * (std::erf((mu - x0) / sigmaRoot2) - std::erf((mu - x1) / sigmaRoot2));
}

PBRT_CPU_GPU
inline Float Logistic(Float x, Float s) {
    x = std::abs(x);
    return std::exp(-x / s) / (s * Sqr(1 + std::exp(-x / s)));
}

PBRT_CPU_GPU
inline Float LogisticCDF(Float x, Float s) {
    return 1 / (1 + std::exp(-x / s));
}

PBRT_CPU_GPU
inline Float TrimmedLogistic(Float x, Float s, Float a, Float b) {
    DCHECK_LT(a, b);
    return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

PBRT_CPU_GPU
inline Float ErfInv(Float a);
PBRT_CPU_GPU
inline Float I0(Float x);
PBRT_CPU_GPU
inline Float LogI0(Float x);

/**
 * \brief Find an interval in an ordered set
 *
 * This function is very similar to \c std::upper_bound, but it uses a functor
 * rather than an actual array to permit working with procedurally defined
 * data. It returns the index \c i such that pred(i) is \c true and pred(i+1)
 * is \c false. See below for special cases.
 *
 * This function is primarily used to locate an interval (i, i+1) for linear
 * interpolation, hence its name. To avoid issues out of bounds accesses, and
 * to deal with predicates that evaluate to \c true or \c false on the entire
 * domain, the returned left interval index is clamped to the range <tt>[left,
 * right-2]</tt>.
 * In particular:
 * If there is no index such that pred(i) is true, we return (left).
 * If there is no index such that pred(i+1) is false, we return (right-2).
 */
template <typename Predicate>
PBRT_CPU_GPU inline size_t FindInterval(size_t size_, const Predicate &pred) {
    using ssize_t = std::make_signed_t<size_t>;  // Not all platforms have ssize_t
    ssize_t size = (ssize_t)size_ - 2, first = 1;

    while (size > 0) {
        size_t half = (size_t)size >> 1, middle = first + half;

        // .. and recurse into the left or right side
        bool predResult = pred(middle);
        first = predResult ? middle + 1 : first;
        size = predResult ? size - (half + 1) : half;
    }

    return (size_t)Clamp((ssize_t)first - 1, 0, size_ - 2);
}

template <typename T>
PBRT_CPU_GPU inline constexpr bool IsPowerOf2(T v) {
    return v && !(v & (v - 1));
}

template <typename T>
PBRT_CPU_GPU inline bool IsPowerOf4(T v) {
    return v == 1 << (2 * Log4Int(v));
}

PBRT_CPU_GPU
inline constexpr int32_t RoundUpPow2(int32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

PBRT_CPU_GPU
inline constexpr int64_t RoundUpPow2(int64_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

template <typename T>
PBRT_CPU_GPU inline T RoundUpPow4(T v) {
    return IsPowerOf4(v) ? v : (1 << (2 * (1 + Log4Int(v))));
}

template <typename Float>
PBRT_CPU_GPU inline CompensatedFloat TwoProd(Float a, Float b) {
    Float ab = a * b;
    return {ab, FMA(a, b, -ab)};
}

template <typename Float>
PBRT_CPU_GPU inline CompensatedFloat TwoSum(Float a, Float b) {
    Float s = a + b;
    Float ap = s - b;
    Float bp = s - ap;
    Float da = a - ap;
    Float db = b - bp;
    return {s, da + db};
}

// Returns ab-cd with <1.5 ulps of error
// Claude-Pierre Jeannerod, Nicolas Louvet, and Jean-Michel Muller,
//  "Further Analysis of Kahan's Algorithm for the Accurate Computation
//  of 2x2 Determinants". Mathematics of Computation, Vol. 82, No. 284,
//  Oct. 2013, pp. 2245-2264
template <typename Ta, typename Tb, typename Tc, typename Td>
PBRT_CPU_GPU inline auto DifferenceOfProducts(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto err = FMA(-c, d, cd);  // Error (exact)
    auto dop = FMA(a, b, -cd);
    return dop + err;
}

template <typename Ta, typename Tb, typename Tc, typename Td>
PBRT_CPU_GPU inline auto SumOfProducts(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto err = FMA(c, d, -cd);  // Error (exact)
    auto sop = FMA(a, b, cd);
    return sop + err;
}

namespace internal {

template <typename Float>
PBRT_CPU_GPU inline CompensatedFloat InnerProduct(Float a, Float b) {
    return TwoProd(a, b);
}

// Accurate dot products with FMA: Graillat et al.,
// http://rnc7.loria.fr/louvet_poster.pdf
//
// Accurate summation, dot product and polynomial evaluation in complex
// floating point arithmetic, Graillat and Menissier-Morain.
//
// Precision same as if in working with doubles. Unfortunately is about
// 4.6x slower than plain old float dot product. Going to doubles is just
// ~2x slower...
template <typename Float, typename... T>
PBRT_CPU_GPU inline CompensatedFloat InnerProduct(Float a, Float b, T... terms) {
    CompensatedFloat ab = TwoProd(a, b);
    CompensatedFloat tp = InnerProduct(terms...);
    CompensatedFloat sum = TwoSum(ab.v, tp.v);
    return {sum.v, ab.err + (tp.err + sum.err)};
}

}  // namespace internal

template <typename... T>
PBRT_CPU_GPU inline std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>...>, Float>
InnerProduct(T... terms) {
    CompensatedFloat ip = internal::InnerProduct(terms...);
    return Float(ip);
}

PBRT_CPU_GPU
inline bool Quadratic(float a, float b, float c, float *t0, float *t1) {
    // Find quadratic discriminant
    float discrim = DifferenceOfProducts(b, b, 4 * a, c);
    if (discrim < 0)
        return false;
    float rootDiscrim = std::sqrt(discrim);

    if (a == 0) {
        *t0 = *t1 = -c / b;
        return true;
    }

    // Compute quadratic _t_ values
    float q = -0.5f * (b + std::copysign(rootDiscrim, b));
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1)
        pstd::swap(*t0, *t1);
    return true;
}

PBRT_CPU_GPU
inline bool Quadratic(double a, double b, double c, double *t0, double *t1) {
    // Find quadratic discriminant
    double discrim = DifferenceOfProducts(b, b, 4 * a, c);
    if (discrim < 0)
        return false;
    double rootDiscrim = std::sqrt(discrim);

    if (a == 0) {
        *t0 = *t1 = -c / b;
        return true;
    }

    // Compute quadratic _t_ values
    double q = -0.5 * (b + std::copysign(rootDiscrim, b));
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1)
        pstd::swap(*t0, *t1);
    return true;
}

// Solve f[x] == 0 over [x0, x1]. f should return a std::pair<FloatType,
// FloatType>[f(x), f'(x)]. Only enabled for float and double.
template <typename FloatType, typename Func>
PBRT_CPU_GPU inline FloatType NewtonBisection(
    FloatType x0, FloatType x1, Func f, Float xEps = 1e-6f, Float fEps = 1e-6f,
    typename std::enable_if_t<std::is_floating_point<FloatType>::value> * = nullptr) {
    DCHECK_LT(x0, x1);
    Float fx0 = f(x0).first, fx1 = f(x1).first;
    if (std::abs(fx0) < fEps)
        return x0;
    if (std::abs(fx1) < fEps)
        return x1;
    bool startIsNegative = fx0 < 0;
    // Implicit line equation: (y-y0)/(y1-y0) = (x-x0)/(x1-x0).
    // Solve for y = 0 to find x for starting point.
    FloatType xMid = x0 + (x1 - x0) * -fx0 / (fx1 - fx0);

    while (true) {
        if (!(x0 < xMid && xMid < x1))
            // Fall back to bisection.
            xMid = (x0 + x1) / 2;

        std::pair<FloatType, FloatType> fxMid = f(xMid);
        DCHECK(!IsNaN(fxMid.first));

        if (startIsNegative == fxMid.first < 0)
            x0 = xMid;
        else
            x1 = xMid;

        if ((x1 - x0) < xEps || std::abs(fxMid.first) < fEps)
            return xMid;

        // Try a Newton step.
        xMid -= fxMid.first / fxMid.second;
    }
}

// If an integral type is used for x0 and x1, assume Float.
template <typename NotFloatType, typename Func>
PBRT_CPU_GPU inline Float NewtonBisection(
    NotFloatType x0, NotFloatType x1, Func f, Float xEps = 1e-6f, Float fEps = 1e-6f,
    typename std::enable_if_t<std::is_integral<NotFloatType>::value> * = nullptr) {
    return NewtonBisection(Float(x0), Float(x1), f, xEps, fEps);
}

PBRT_CPU_GPU
inline Float SmoothStep(Float x, Float a, Float b) {
    if (a == b)
        return (x < a) ? 0 : 1;
    DCHECK_LT(a, b);
    Float t = Clamp((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

// Math Function Declarations
int NextPrime(int x);

pstd::optional<SquareMatrix<3>> LinearLeastSquares(const Float A[][3], const Float B[][3],
                                                   int rows);

PBRT_CPU_GPU
inline Float ErfInv(Float a) {
#ifdef PBRT_IS_GPU_CODE
    return erfinv(a);
#else
    // https://stackoverflow.com/a/49743348
    float p;
    float t = std::log(std::max(FMA(a, -a, 1), std::numeric_limits<Float>::min()));
    CHECK(!IsNaN(t) && !std::isinf(t));
    if (std::abs(t) > 6.125f) {          // maximum ulp error = 2.35793
        p = 3.03697567e-10f;             //  0x1.4deb44p-32
        p = FMA(p, t, 2.93243101e-8f);   //  0x1.f7c9aep-26
        p = FMA(p, t, 1.22150334e-6f);   //  0x1.47e512p-20
        p = FMA(p, t, 2.84108955e-5f);   //  0x1.dca7dep-16
        p = FMA(p, t, 3.93552968e-4f);   //  0x1.9cab92p-12
        p = FMA(p, t, 3.02698812e-3f);   //  0x1.8cc0dep-9
        p = FMA(p, t, 4.83185798e-3f);   //  0x1.3ca920p-8
        p = FMA(p, t, -2.64646143e-1f);  // -0x1.0eff66p-2
        p = FMA(p, t, 8.40016484e-1f);   //  0x1.ae16a4p-1
    } else {                             // maximum ulp error = 2.35456
        p = 5.43877832e-9f;              //  0x1.75c000p-28
        p = FMA(p, t, 1.43286059e-7f);   //  0x1.33b458p-23
        p = FMA(p, t, 1.22775396e-6f);   //  0x1.49929cp-20
        p = FMA(p, t, 1.12962631e-7f);   //  0x1.e52bbap-24
        p = FMA(p, t, -5.61531961e-5f);  // -0x1.d70c12p-15
        p = FMA(p, t, -1.47697705e-4f);  // -0x1.35be9ap-13
        p = FMA(p, t, 2.31468701e-3f);   //  0x1.2f6402p-9
        p = FMA(p, t, 1.15392562e-2f);   //  0x1.7a1e4cp-7
        p = FMA(p, t, -2.32015476e-1f);  // -0x1.db2aeep-3
        p = FMA(p, t, 8.86226892e-1f);   //  0x1.c5bf88p-1
    }
    return a * p;
#endif  // PBRT_IS_GPU_CODE
}

PBRT_CPU_GPU
inline Float I0(Float x) {
    Float val = 0;
    Float x2i = 1;
    int64_t ifact = 1;
    int i4 = 1;
    // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
    for (int i = 0; i < 10; ++i) {
        if (i > 1)
            ifact *= i;
        val += x2i / (i4 * Sqr(ifact));
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}

PBRT_CPU_GPU
inline Float LogI0(Float x) {
    if (x > 12)
        return x + 0.5f * (-std::log(2 * Pi) + std::log(1 / x) + 1 / (8 * x));
    else
        return std::log(I0(x));
}

// Interval Definition
template <typename Float>
class Interval {
  public:
    // Interval Public Methods
    Interval() = default;
    PBRT_CPU_GPU
    constexpr Interval(Float low, Float high)
        : low(std::min(low, high)), high(std::max(low, high)) {}

    PBRT_CPU_GPU
    Interval &operator=(Float v) {
        low = high = v;
        return *this;
    }

    PBRT_CPU_GPU
    static Interval FromValueAndError(Float v, Float err) {
        Interval i;
        if (err == 0)
            i.low = i.high = v;
        else {
            // Compute conservative bounds by rounding the endpoints away
            // from the middle. Note that this will be over-conservative in
            // cases where v-err or v+err are exactly representable in
            // floating-point, but it's probably not worth the trouble of
            // checking this case.
            i.low = NextFloatDown(v - err);
            i.high = NextFloatUp(v + err);
        }
        return i;
    }

    PBRT_CPU_GPU
    Float UpperBound() const { return high; }
    PBRT_CPU_GPU
    Float LowerBound() const { return low; }
    PBRT_CPU_GPU
    Float Midpoint() const { return (low + high) / 2; }
    PBRT_CPU_GPU
    Float Width() const { return high - low; }

    PBRT_CPU_GPU
    Float operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? low : high;
    }

    PBRT_CPU_GPU
    explicit Interval(Float v) : low(v), high(v) {}

    PBRT_CPU_GPU
    explicit operator Float() const { return Midpoint(); }

    PBRT_CPU_GPU
    bool Exactly(Float v) const { return low == v && high == v; }

    template <typename T>
    PBRT_CPU_GPU explicit Interval(const Interval<T> &i) {
        *this = i;
    }

    template <typename T>
    PBRT_CPU_GPU Interval &operator=(const Interval<T> &i) {
        low = i.LowerBound();
        high = i.UpperBound();
        if (sizeof(T) > sizeof(Float)) {
            // Assume that if Float is bigger than T, then it's more
            // precise, which seems not unreasonable...
            low = NextFloatDown(low);
            high = NextFloatUp(high);
        }
        return *this;
    }

    PBRT_CPU_GPU
    bool operator==(Float v) const { return Exactly(v); }

    PBRT_CPU_GPU
    Interval operator-() const { return {-high, -low}; }

    template <typename F>
    PBRT_CPU_GPU auto operator+(Interval<F> i) const
        -> Interval<decltype(Float() + F())> {
        using FR = decltype(Float() + F());
        if (Exactly(0))
            return Interval<FR>(i);
        else if (i.Exactly(0))
            return Interval<FR>(*this);

        return Interval<FR>(NextFloatDown(LowerBound() + i.LowerBound()),
                            NextFloatUp(UpperBound() + i.UpperBound()));
    }

    template <typename F>
    PBRT_CPU_GPU auto operator-(Interval<F> i) const
        -> Interval<decltype(Float() - F())> {
        using FR = decltype(Float() - F());
        if (Exactly(0))
            return Interval<FR>(-i);
        else if (i.Exactly(0))
            return Interval<FR>(*this);

        return Interval<FR>(NextFloatDown(LowerBound() - i.UpperBound()),
                            NextFloatUp(UpperBound() - i.LowerBound()));
    }

    template <typename F>
    PBRT_CPU_GPU auto operator*(Interval<F> i) const
        -> Interval<decltype(Float() * F())> {
        using FR = decltype(Float() * F());
        if (Exactly(0) || i.Exactly(0))
            return Interval<FR>(0);
        if (Exactly(1))
            return Interval<FR>(i);
        if (i.Exactly(1))
            return Interval<FR>(*this);

        FR prod[4] = {LowerBound() * i.LowerBound(), UpperBound() * i.LowerBound(),
                      LowerBound() * i.UpperBound(), UpperBound() * i.UpperBound()};
        return Interval<FR>(NextFloatDown(std::min({prod[0], prod[1], prod[2], prod[3]})),
                            NextFloatUp(std::max({prod[0], prod[1], prod[2], prod[3]})));
    }

    template <typename F>
    PBRT_CPU_GPU auto operator/(Interval<F> i) const
        -> Interval<decltype(Float() / F())> {
        using FR = decltype(Float() / F());
        if (Exactly(0))
            // Not going to worry about NaN...
            return Interval<FR>(0);
        if (i.Exactly(1))
            return Interval<FR>(*this);

        if (InRange(0, i))
            // The interval we're dividing by straddles zero, so just
            // return an interval of everything.
            return Interval<FR>(-Infinity, Infinity);

        FR div[4] = {LowerBound() / i.LowerBound(), UpperBound() / i.LowerBound(),
                     LowerBound() / i.UpperBound(), UpperBound() / i.UpperBound()};
        return Interval<FR>(NextFloatDown(std::min({div[0], div[1], div[2], div[3]})),
                            NextFloatUp(std::max({div[0], div[1], div[2], div[3]})));
    }

    template <typename F>
    PBRT_CPU_GPU bool operator==(Interval<F> i) const {
        return low == i.low && high == i.high;
    }

    PBRT_CPU_GPU
    bool operator!=(Float f) const { return f < low || f > high; }

    std::string ToString() const;

    template <typename F>
    PBRT_CPU_GPU Interval &operator+=(Interval<F> i) {
        *this = Interval(*this + i);
        return *this;
    }
    template <typename F>
    PBRT_CPU_GPU Interval &operator-=(Interval<F> i) {
        *this = Interval(*this - i);
        return *this;
    }
    template <typename F>
    PBRT_CPU_GPU Interval &operator*=(Interval<F> i) {
        *this = Interval(*this * i);
        return *this;
    }
    template <typename F>
    PBRT_CPU_GPU Interval &operator/=(Interval<F> i) {
        *this = Interval(*this / i);
        return *this;
    }
    PBRT_CPU_GPU
    Interval &operator+=(Float f) { return *this += Interval<Float>(f); }
    PBRT_CPU_GPU
    Interval &operator-=(Float f) { return *this -= Interval<Float>(f); }
    PBRT_CPU_GPU
    Interval &operator*=(Float f) { return *this *= Interval<Float>(f); }
    PBRT_CPU_GPU
    Interval &operator/=(Float f) { return *this /= Interval<Float>(f); }

#ifndef PBRT_IS_GPU_CODE
    static const Interval Pi;
#endif

  private:
    friend class SOA<Interval<Float>>;
    // Interval Private Members
    Float low, high;
};

using FloatInterval = Interval<Float>;

// Interval Inline Functions
template <typename T, typename Float>
PBRT_CPU_GPU inline bool InRange(T v, Interval<Float> i) {
    return v >= i.LowerBound() && v <= i.UpperBound();
}

template <typename FloatA, typename FloatB>
PBRT_CPU_GPU inline bool InRange(Interval<FloatA> a, Interval<FloatB> b) {
    return a.LowerBound() <= b.UpperBound() && a.UpperBound() >= b.LowerBound();
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator+(T f, Interval<Float> i) {
    return Interval<Float>(f) + i;
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator-(T f, Interval<Float> i) {
    return Interval<Float>(f) - i;
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator*(T f, Interval<Float> i) {
    return Interval<Float>(f) * i;
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator/(T f, Interval<Float> i) {
    return Interval<Float>(f) / i;
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator+(Interval<Float> i, T f) {
    return i + Interval<Float>(f);
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator-(Interval<Float> i, T f) {
    return i - Interval<Float>(f);
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator*(Interval<Float> i, T f) {
    return i * Interval<Float>(f);
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    operator/(Interval<Float> i, T f) {
    return i / Interval<Float>(f);
}

template <typename Float>
PBRT_CPU_GPU inline Float Floor(Interval<Float> i) {
    return std::floor(i.LowerBound());
}

template <typename Float>
PBRT_CPU_GPU inline Float Ceil(Interval<Float> i) {
    return std::ceil(i.UpperBound());
}

template <typename Float>
PBRT_CPU_GPU inline Float floor(Interval<Float> i) {
    return Floor(i);
}

template <typename Float>
PBRT_CPU_GPU inline Float ceil(Interval<Float> i) {
    return Ceil(i);
}

template <typename Float>
PBRT_CPU_GPU inline Float Min(Interval<Float> a, Interval<Float> b) {
    return std::min(a.LowerBound(), b.LowerBound());
}

template <typename Float>
PBRT_CPU_GPU inline Float Max(Interval<Float> a, Interval<Float> b) {
    return std::max(a.UpperBound(), b.UpperBound());
}

template <typename Float>
PBRT_CPU_GPU inline Float min(Interval<Float> a, Interval<Float> b) {
    return Min(a, b);
}

template <typename Float>
PBRT_CPU_GPU inline Float max(Interval<Float> a, Interval<Float> b) {
    return Max(a, b);
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> Sqrt(Interval<Float> i) {
    return Interval<Float>(std::max<Float>(0, NextFloatDown(std::sqrt(i.LowerBound()))),
                           NextFloatUp(std::sqrt(i.UpperBound())));
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> sqrt(Interval<Float> i) {
    return Sqrt(i);
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> Sqr(Interval<Float> i) {
    Float slow = Sqr(i.LowerBound()), shigh = Sqr(i.UpperBound());
    if (slow > shigh)
        pstd::swap(slow, shigh);
    if (InRange(0, i))
        return Interval<Float>(0, NextFloatUp(shigh));
    return Interval<Float>(NextFloatDown(slow), NextFloatUp(shigh));
}

template <typename FloatA, typename FloatB, typename FloatC>
PBRT_CPU_GPU inline auto FMA(Interval<FloatA> a, Interval<FloatB> b, Interval<FloatC> c)
    -> Interval<decltype(FloatA() * FloatB() + FloatC())> {
    using FT = decltype(FloatA() * FloatB() + FloatC());
    Float low =
        NextFloatDown(std::min({FMA(a.LowerBound(), b.LowerBound(), c.LowerBound()),
                                FMA(a.UpperBound(), b.LowerBound(), c.LowerBound()),
                                FMA(a.LowerBound(), b.UpperBound(), c.LowerBound()),
                                FMA(a.UpperBound(), b.UpperBound(), c.LowerBound())}));
    Float high =
        NextFloatUp(std::max({FMA(a.LowerBound(), b.LowerBound(), c.UpperBound()),
                              FMA(a.UpperBound(), b.LowerBound(), c.UpperBound()),
                              FMA(a.LowerBound(), b.UpperBound(), c.UpperBound()),
                              FMA(a.UpperBound(), b.UpperBound(), c.UpperBound())}));
    return Interval<FT>(low, high);
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> DifferenceOfProducts(Interval<Float> a,
                                                         Interval<Float> b,
                                                         Interval<Float> c,
                                                         Interval<Float> d) {
    Float ab[4] = {a.LowerBound() * b.LowerBound(), a.UpperBound() * b.LowerBound(),
                   a.LowerBound() * b.UpperBound(), a.UpperBound() * b.UpperBound()};
    Float abLow = std::min({ab[0], ab[1], ab[2], ab[3]});
    Float abHigh = std::max({ab[0], ab[1], ab[2], ab[3]});
    int abLowIndex = abLow == ab[0] ? 0 : (abLow == ab[1] ? 1 : (abLow == ab[2] ? 2 : 3));
    int abHighIndex =
        abHigh == ab[0] ? 0 : (abHigh == ab[1] ? 1 : (abHigh == ab[2] ? 2 : 3));

    Float cd[4] = {c.LowerBound() * d.LowerBound(), c.UpperBound() * d.LowerBound(),
                   c.LowerBound() * d.UpperBound(), c.UpperBound() * d.UpperBound()};
    Float cdLow = std::min({cd[0], cd[1], cd[2], cd[3]});
    Float cdHigh = std::max({cd[0], cd[1], cd[2], cd[3]});
    int cdLowIndex = cdLow == cd[0] ? 0 : (cdLow == cd[1] ? 1 : (cdLow == cd[2] ? 2 : 3));
    int cdHighIndex =
        cdHigh == cd[0] ? 0 : (cdHigh == cd[1] ? 1 : (cdHigh == cd[2] ? 2 : 3));

    // Invert cd Indices since it's subtracted...
    Float low = DifferenceOfProducts(a[abLowIndex & 1], b[abLowIndex >> 1],
                                     c[cdHighIndex & 1], d[cdHighIndex >> 1]);
    Float high = DifferenceOfProducts(a[abHighIndex & 1], b[abHighIndex >> 1],
                                      c[cdLowIndex & 1], d[cdLowIndex >> 1]);
    DCHECK_LE(low, high);

    return {NextFloatDown(low, 2), NextFloatUp(high, 2)};
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> SumOfProducts(Interval<Float> a, Interval<Float> b,
                                                  Interval<Float> c, Interval<Float> d) {
    return DifferenceOfProducts(a, b, -c, d);
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    MulPow2(T s, Interval<Float> i) {
    return MulPow2(i, s);
}

template <typename T, typename Float>
PBRT_CPU_GPU inline
    typename std::enable_if_t<std::is_arithmetic<T>::value, Interval<Float>>
    MulPow2(Interval<Float> i, T s) {
    T as = std::abs(s);
    if (as < 1)
        DCHECK_EQ(1 / as, 1ull << Log2Int(1 / as));
    else
        DCHECK_EQ(as, 1ull << Log2Int(as));

    // Multiplication by powers of 2 is exaact
    return Interval<Float>(std::min(i.LowerBound() * s, i.UpperBound() * s),
                           std::max(i.LowerBound() * s, i.UpperBound() * s));
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> Abs(Interval<Float> i) {
    if (i.LowerBound() >= 0)
        // The entire interval is greater than zero, so we're all set.
        return i;
    else if (i.UpperBound() <= 0)
        // The entire interval is less than zero.
        return Interval<Float>(-i.UpperBound(), -i.LowerBound());
    else
        // The interval straddles zero.
        return Interval<Float>(0, std::max(-i.LowerBound(), i.UpperBound()));
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> abs(Interval<Float> i) {
    return Abs(i);
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> ACos(Interval<Float> i) {
    Float low = std::acos(std::min<Float>(1, i.UpperBound()));
    Float high = std::acos(std::max<Float>(-1, i.LowerBound()));

    return Interval<Float>(std::max<Float>(0, NextFloatDown(low)), NextFloatUp(high));
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> Sin(Interval<Float> i) {
    CHECK_GE(i.LowerBound(), -1e-16);
    CHECK_LE(i.UpperBound(), 2.0001 * Pi);
    Float low = std::sin(std::max<Float>(0, i.LowerBound()));
    Float high = std::sin(i.UpperBound());
    if (low > high)
        pstd::swap(low, high);
    low = std::max<Float>(-1, NextFloatDown(low));
    high = std::min<Float>(1, NextFloatUp(high));
    if (InRange(Pi / 2, i))
        high = 1;
    if (InRange((3.f / 2.f) * Pi, i))
        low = -1;

    return Interval<Float>(low, high);
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> Cos(Interval<Float> i) {
    CHECK_GE(i.LowerBound(), -1e-16);
    CHECK_LE(i.UpperBound(), 2.0001 * Pi);
    Float low = std::cos(std::max<Float>(0, i.LowerBound()));
    Float high = std::cos(i.UpperBound());
    if (low > high)
        pstd::swap(low, high);
    low = std::max<Float>(-1, NextFloatDown(low));
    high = std::min<Float>(1, NextFloatUp(high));
    if (InRange(Pi, i))
        low = -1;

    return Interval<Float>(low, high);
}

template <typename Float>
PBRT_CPU_GPU inline bool Quadratic(Interval<Float> a, Interval<Float> b,
                                   Interval<Float> c, Interval<Float> *t0,
                                   Interval<Float> *t1) {
    // Find quadratic discriminant
    Interval<Float> discrim = DifferenceOfProducts(b, b, MulPow2(4, a), c);
    if (discrim.LowerBound() < 0)
        return false;
    Interval<Float> floatRootDiscrim = Sqrt(discrim);

    // Compute quadratic _t_ values
    Interval<Float> q;
    if ((Float)b < 0)
        q = MulPow2(-.5, b - floatRootDiscrim);
    else
        q = MulPow2(-.5, b + floatRootDiscrim);
    *t0 = q / a;
    *t1 = c / q;
    if (t0->LowerBound() > t1->LowerBound())
        pstd::swap(*t0, *t1);
    return true;
}

template <typename Float>
PBRT_CPU_GPU inline Interval<Float> SumSquares(Interval<Float> i) {
    return Sqr(i);
}

template <typename Float, typename... Args>
PBRT_CPU_GPU inline Interval<Float> SumSquares(Interval<Float> i, Args... args) {
    Interval<Float> ss = FMA(i, i, SumSquares(args...));
    return Interval<Float>(std::max<Float>(0, ss.LowerBound()), ss.UpperBound());
}

PBRT_CPU_GPU
Vector3f EqualAreaSquareToSphere(const Point2f &p);
PBRT_CPU_GPU
Point2f EqualAreaSphereToSquare(const Vector3f &v);
PBRT_CPU_GPU
Point2f WrapEqualAreaSquare(Point2f p);

// Spline Interpolation Declarations
PBRT_CPU_GPU
Float CatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> values, Float x);
PBRT_CPU_GPU
bool CatmullRomWeights(pstd::span<const Float> nodes, Float x, int *offset,
                       pstd::span<Float> weights);
PBRT_CPU_GPU
Float IntegrateCatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> values,
                          pstd::span<Float> cdf);
PBRT_CPU_GPU
Float InvertCatmullRom(pstd::span<const Float> x, pstd::span<const Float> values,
                       Float u);

namespace {

template <int N>
PBRT_CPU_GPU inline void init(Float m[N][N], int i, int j) {}

template <int N, typename... Args>
PBRT_CPU_GPU inline void init(Float m[N][N], int i, int j, Float v, Args... args) {
    m[i][j] = v;
    if (++j == N) {
        ++i;
        j = 0;
    }
    init<N>(m, i, j, args...);
}

template <int N>
PBRT_CPU_GPU inline void initDiag(Float m[N][N], int i) {}

template <int N, typename... Args>
PBRT_CPU_GPU inline void initDiag(Float m[N][N], int i, Float v, Args... args) {
    m[i][i] = v;
    initDiag<N>(m, i + 1, args...);
}

}  // namespace

// SquareMatrix Definition
template <int N>
class SquareMatrix {
  public:
    PBRT_CPU_GPU
    static SquareMatrix Zero() {
        SquareMatrix m;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m.m[i][j] = 0;
        return m;
    }

    PBRT_CPU_GPU
    SquareMatrix() {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] = (i == j) ? 1 : 0;
    }
    PBRT_CPU_GPU
    SquareMatrix(const Float mat[N][N]) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] = mat[i][j];
    }
    PBRT_CPU_GPU
    SquareMatrix(pstd::span<const Float> t);
    template <typename... Args>
    PBRT_CPU_GPU SquareMatrix(Float v, Args... args) {
        static_assert(1 + sizeof...(Args) == N * N,
                      "Incorrect number of values provided to SquareMatrix constructor");
        init<N>(m, 0, 0, v, args...);
    }
    template <typename... Args>
    PBRT_CPU_GPU static SquareMatrix Diag(Float v, Args... args) {
        static_assert(1 + sizeof...(Args) == N,
                      "Incorrect number of values provided to SquareMatrix::Diag");
        SquareMatrix m;
        initDiag<N>(m.m, 0, v, args...);
        return m;
    }

    PBRT_CPU_GPU
    SquareMatrix operator+(const SquareMatrix &m) const {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] += m.m[i][j];
        return r;
    }
    PBRT_CPU_GPU
    SquareMatrix operator*(Float s) const {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] *= s;
        return r;
    }
    PBRT_CPU_GPU
    SquareMatrix operator/(Float s) const {
        DCHECK_NE(s, 0);
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] /= s;
        return r;
    }

    PBRT_CPU_GPU
    bool operator==(const SquareMatrix<N> &m2) const {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return false;
        return true;
    }

    PBRT_CPU_GPU
    bool operator!=(const SquareMatrix<N> &m2) const {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return true;
        return false;
    }

    PBRT_CPU_GPU
    bool operator<(const SquareMatrix<N> &m2) const {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (m[i][j] < m2.m[i][j])
                    return true;
                if (m[i][j] > m2.m[i][j])
                    return false;
            }
        return false;
    }

    PBRT_CPU_GPU
    Float Determinant() const;

    PBRT_CPU_GPU
    bool IsIdentity() const {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    if (m[i][j] != 1)
                        return false;
                } else if (m[i][j] != 0)
                    return false;
            }
        return true;
    }
    std::string ToString() const;

    PBRT_CPU_GPU
    pstd::span<const Float> operator[](int i) const { return m[i]; }

    PBRT_CPU_GPU
    pstd::span<Float> operator[](int i) { return pstd::span<Float>(m[i]); }

  private:
    Float m[N][N];
};

// SquareMatrix Inline Methods
template <int N>
PBRT_CPU_GPU inline SquareMatrix<N> operator*(Float s, const SquareMatrix<N> &m) {
    return m * s;
}

template <typename Tresult, int N, typename T>
PBRT_CPU_GPU inline Tresult Mul(const SquareMatrix<N> &m, const T &v) {
    Tresult result;
    for (int i = 0; i < N; ++i) {
        result[i] = 0;
        for (int j = 0; j < N; ++j)
            result[i] += m[i][j] * v[j];
    }
    return result;
}

template <>
PBRT_CPU_GPU inline Float SquareMatrix<3>::Determinant() const {
    Float minor12 = DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    Float minor02 = DifferenceOfProducts(m[1][0], m[2][2], m[1][2], m[2][0]);
    Float minor01 = DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    return FMA(m[0][2], minor01,
               DifferenceOfProducts(m[0][0], minor12, m[0][1], minor02));
}

template <int N>
PBRT_CPU_GPU inline SquareMatrix<N> Transpose(const SquareMatrix<N> &m) {
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r[i][j] = m[j][i];
    return r;
}

template <int N>
PBRT_CPU_GPU pstd::optional<SquareMatrix<N>> Inverse(const SquareMatrix<N> &);

template <>
PBRT_CPU_GPU inline pstd::optional<SquareMatrix<3>> Inverse(const SquareMatrix<3> &m) {
    Float det = m.Determinant();
    if (det == 0)
        return {};
    Float invDet = 1 / det;

    SquareMatrix<3> r;

    r[0][0] = invDet * DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    r[1][0] = invDet * DifferenceOfProducts(m[1][2], m[2][0], m[1][0], m[2][2]);
    r[2][0] = invDet * DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    r[0][1] = invDet * DifferenceOfProducts(m[0][2], m[2][1], m[0][1], m[2][2]);
    r[1][1] = invDet * DifferenceOfProducts(m[0][0], m[2][2], m[0][2], m[2][0]);
    r[2][1] = invDet * DifferenceOfProducts(m[0][1], m[2][0], m[0][0], m[2][1]);
    r[0][2] = invDet * DifferenceOfProducts(m[0][1], m[1][2], m[0][2], m[1][1]);
    r[1][2] = invDet * DifferenceOfProducts(m[0][2], m[1][0], m[0][0], m[1][2]);
    r[2][2] = invDet * DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);

    return r;
}

template <int N, typename T>
PBRT_CPU_GPU inline T operator*(const SquareMatrix<N> &m, const T &v) {
    return Mul<T>(m, v);
}

template <>
PBRT_CPU_GPU inline SquareMatrix<4> operator*(const SquareMatrix<4> &m1,
                                              const SquareMatrix<4> &m2) {
    SquareMatrix<4> r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r[i][j] = InnerProduct(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2],
                                   m2[2][j], m1[i][3], m2[3][j]);
    return r;
}

template <>
PBRT_CPU_GPU inline SquareMatrix<3> operator*(const SquareMatrix<3> &m1,
                                              const SquareMatrix<3> &m2) {
    SquareMatrix<3> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i][j] =
                InnerProduct(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2], m2[2][j]);
    return r;
}

template <int N>
PBRT_CPU_GPU inline SquareMatrix<N> operator*(const SquareMatrix<N> &m1,
                                              const SquareMatrix<N> &m2) {
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            r[i][j] = 0;
            for (int k = 0; k < N; ++k)
                r[i][j] = FMA(m1[i][k], m2[k][j], r[i][j]);
        }
    return r;
}

template <int N>
PBRT_CPU_GPU inline SquareMatrix<N>::SquareMatrix(pstd::span<const Float> t) {
    CHECK_EQ(N * N, t.size());
    for (int i = 0; i < N * N; ++i)
        m[i / N][i % N] = t[i];
}

template <int N>
PBRT_CPU_GPU SquareMatrix<N> operator*(const SquareMatrix<N> &m1,
                                       const SquareMatrix<N> &m2);

template <>
PBRT_CPU_GPU inline Float SquareMatrix<1>::Determinant() const {
    return m[0][0];
}

template <>
PBRT_CPU_GPU inline Float SquareMatrix<2>::Determinant() const {
    return DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);
}

template <>
PBRT_CPU_GPU inline Float SquareMatrix<4>::Determinant() const {
    Float s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
    Float s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
    Float s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);

    Float s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
    Float s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
    Float s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

    Float c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
    Float c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
    Float c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);

    Float c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
    Float c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
    Float c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

    return (DifferenceOfProducts(s0, c5, s1, c4) + DifferenceOfProducts(s2, c3, -s3, c2) +
            DifferenceOfProducts(s5, c0, s4, c1));
}

template <int N>
PBRT_CPU_GPU inline Float SquareMatrix<N>::Determinant() const {
    SquareMatrix<N - 1> sub;
    Float det = 0;
    // Inefficient, but we don't currently use N>4 anyway..
    for (int i = 0; i < N; ++i) {
        // Sub-matrix without row 0 and column i
        for (int j = 0; j < N - 1; ++j)
            for (int k = 0; k < N - 1; ++k)
                sub[j][k] = m[j + 1][k < i ? k : k + 1];

        Float sign = (i & 1) ? -1 : 1;
        det += sign * m[0][i] * sub.Determinant();
    }
    return det;
}

template <>
PBRT_CPU_GPU inline pstd::optional<SquareMatrix<4>> Inverse(const SquareMatrix<4> &m) {
    // Via: https://github.com/google/ion/blob/master/ion/math/matrixutils.cc,
    // (c) Google, Apache license.

    // For 4x4 do not compute the adjugate as the transpose of the cofactor
    // matrix, because this results in extra work. Several calculations can be
    // shared across the sub-determinants.
    //
    // This approach is explained in David Eberly's Geometric Tools book,
    // excerpted here:
    //   http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    Float s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
    Float s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
    Float s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);

    Float s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
    Float s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
    Float s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

    Float c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
    Float c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
    Float c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);

    Float c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
    Float c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
    Float c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

    Float determinant = InnerProduct(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (determinant == 0)
        return {};
    Float s = 1 / determinant;

    Float inv[4][4] = {s * InnerProduct(m[1][1], c5, m[1][3], c3, -m[1][2], c4),
                       s * InnerProduct(-m[0][1], c5, m[0][2], c4, -m[0][3], c3),
                       s * InnerProduct(m[3][1], s5, m[3][3], s3, -m[3][2], s4),
                       s * InnerProduct(-m[2][1], s5, m[2][2], s4, -m[2][3], s3),

                       s * InnerProduct(-m[1][0], c5, m[1][2], c2, -m[1][3], c1),
                       s * InnerProduct(m[0][0], c5, m[0][3], c1, -m[0][2], c2),
                       s * InnerProduct(-m[3][0], s5, m[3][2], s2, -m[3][3], s1),
                       s * InnerProduct(m[2][0], s5, m[2][3], s1, -m[2][2], s2),

                       s * InnerProduct(m[1][0], c4, m[1][3], c0, -m[1][1], c2),
                       s * InnerProduct(-m[0][0], c4, m[0][1], c2, -m[0][3], c0),
                       s * InnerProduct(m[3][0], s4, m[3][3], s0, -m[3][1], s2),
                       s * InnerProduct(-m[2][0], s4, m[2][1], s2, -m[2][3], s0),

                       s * InnerProduct(-m[1][0], c3, m[1][1], c1, -m[1][2], c0),
                       s * InnerProduct(m[0][0], c3, m[0][2], c0, -m[0][1], c1),
                       s * InnerProduct(-m[3][0], s3, m[3][1], s1, -m[3][2], s0),
                       s * InnerProduct(m[2][0], s3, m[2][2], s0, -m[2][1], s1)};

    return SquareMatrix<4>(inv);
}

extern template class SquareMatrix<2>;
extern template class SquareMatrix<3>;
extern template class SquareMatrix<4>;

}  // namespace pbrt

#endif  // PBRT_UTIL_MATH_H
