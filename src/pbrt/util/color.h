// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_COLOR_H
#define PBRT_UTIL_COLOR_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>

// A special present from windgi.h on Windows...
#ifdef RGB
#undef RGB
#endif  // RGB

namespace pbrt {

// RGB Definition
class RGB {
  public:
    // RGB Public Methods
    RGB() = default;
    PBRT_CPU_GPU
    RGB(Float r, Float g, Float b) : r(r), g(g), b(b) {}

    PBRT_CPU_GPU
    RGB &operator+=(const RGB &s) {
        r += s.r;
        g += s.g;
        b += s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator+(const RGB &s) const {
        RGB ret = *this;
        return ret += s;
    }

    PBRT_CPU_GPU
    RGB &operator-=(const RGB &s) {
        r -= s.r;
        g -= s.g;
        b -= s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator-(const RGB &s) const {
        RGB ret = *this;
        return ret -= s;
    }
    PBRT_CPU_GPU
    friend RGB operator-(Float a, const RGB &s) { return {a - s.r, a - s.g, a - s.b}; }

    PBRT_CPU_GPU
    RGB &operator*=(const RGB &s) {
        r *= s.r;
        g *= s.g;
        b *= s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator*(const RGB &s) const {
        RGB ret = *this;
        return ret *= s;
    }
    PBRT_CPU_GPU
    RGB operator*(Float a) const {
        DCHECK(!IsNaN(a));
        return {a * r, a * g, a * b};
    }
    PBRT_CPU_GPU
    RGB &operator*=(Float a) {
        DCHECK(!IsNaN(a));
        r *= a;
        g *= a;
        b *= a;
        return *this;
    }
    PBRT_CPU_GPU
    friend RGB operator*(Float a, const RGB &s) { return s * a; }

    PBRT_CPU_GPU
    RGB &operator/=(const RGB &s) {
        r /= s.r;
        g /= s.g;
        b /= s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator/(const RGB &s) const {
        RGB ret = *this;
        return ret /= s;
    }
    PBRT_CPU_GPU
    RGB &operator/=(Float a) {
        DCHECK(!IsNaN(a));
        DCHECK_NE(a, 0);
        r /= a;
        g /= a;
        b /= a;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator/(Float a) const {
        RGB ret = *this;
        return ret /= a;
    }

    PBRT_CPU_GPU
    RGB operator-() const { return {-r, -g, -b}; }

    PBRT_CPU_GPU
    Float Average() const { return (r + g + b) / 3; }

    PBRT_CPU_GPU
    bool operator==(const RGB &s) const { return r == s.r && g == s.g && b == s.b; }
    PBRT_CPU_GPU
    bool operator!=(const RGB &s) const { return r != s.r || g != s.g || b != s.b; }
    PBRT_CPU_GPU
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return r;
        else if (c == 1)
            return g;
        return b;
    }
    PBRT_CPU_GPU
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return r;
        else if (c == 1)
            return g;
        return b;
    }

    std::string ToString() const;

    // RGB Public Members
    Float r = 0, g = 0, b = 0;
};

PBRT_CPU_GPU
inline RGB max(const RGB &a, const RGB &b) {
    return RGB(std::max(a.r, b.r), std::max(a.g, b.g), std::max(a.b, b.b));
}

template <typename U, typename V>
PBRT_CPU_GPU inline RGB Clamp(const RGB &rgb, U min, V max) {
    return RGB(pbrt::Clamp(rgb.r, min, max), pbrt::Clamp(rgb.g, min, max),
               pbrt::Clamp(rgb.b, min, max));
}

PBRT_CPU_GPU
inline RGB ClampZero(const RGB &rgb) {
    return RGB(std::max<Float>(0, rgb.r), std::max<Float>(0, rgb.g),
               std::max<Float>(0, rgb.b));
}

PBRT_CPU_GPU
inline RGB Lerp(Float t, const RGB &s1, const RGB &s2) {
    return (1 - t) * s1 + t * s2;
}

// XYZ Definition
class XYZ {
  public:
    // XYZ Public Methods
    XYZ() = default;
    PBRT_CPU_GPU
    XYZ(Float X, Float Y, Float Z) : X(X), Y(Y), Z(Z) {}

    PBRT_CPU_GPU
    Float Average() const { return (X + Y + Z) / 3; }

    PBRT_CPU_GPU
    Point2f xy() const { return Point2f(X / (X + Y + Z), Y / (X + Y + Z)); }

    PBRT_CPU_GPU
    static XYZ FromxyY(Point2f xy, Float Y = 1) {
        if (xy.y == 0)
            return XYZ(0, 0, 0);
        return XYZ(xy.x * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y);
    }

    PBRT_CPU_GPU
    XYZ &operator+=(const XYZ &s) {
        X += s.X;
        Y += s.Y;
        Z += s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator+(const XYZ &s) const {
        XYZ ret = *this;
        return ret += s;
    }

    PBRT_CPU_GPU
    XYZ &operator-=(const XYZ &s) {
        X -= s.X;
        Y -= s.Y;
        Z -= s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator-(const XYZ &s) const {
        XYZ ret = *this;
        return ret -= s;
    }
    PBRT_CPU_GPU
    friend XYZ operator-(Float a, const XYZ &s) { return {a - s.X, a - s.Y, a - s.Z}; }

    PBRT_CPU_GPU
    XYZ &operator*=(const XYZ &s) {
        X *= s.X;
        Y *= s.Y;
        Z *= s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator*(const XYZ &s) const {
        XYZ ret = *this;
        return ret *= s;
    }
    PBRT_CPU_GPU
    XYZ operator*(Float a) const {
        DCHECK(!IsNaN(a));
        return {a * X, a * Y, a * Z};
    }
    PBRT_CPU_GPU
    XYZ &operator*=(Float a) {
        DCHECK(!IsNaN(a));
        X *= a;
        Y *= a;
        Z *= a;
        return *this;
    }

    PBRT_CPU_GPU
    XYZ &operator/=(const XYZ &s) {
        X /= s.X;
        Y /= s.Y;
        Z /= s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator/(const XYZ &s) const {
        XYZ ret = *this;
        return ret /= s;
    }
    PBRT_CPU_GPU
    XYZ &operator/=(Float a) {
        DCHECK(!IsNaN(a));
        DCHECK_NE(a, 0);
        X /= a;
        Y /= a;
        Z /= a;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator/(Float a) const {
        XYZ ret = *this;
        return ret /= a;
    }

    PBRT_CPU_GPU
    XYZ operator-() const { return {-X, -Y, -Z}; }

    PBRT_CPU_GPU
    bool operator==(const XYZ &s) const { return X == s.X && Y == s.Y && Z == s.Z; }
    PBRT_CPU_GPU
    bool operator!=(const XYZ &s) const { return X != s.X || Y != s.Y || Z != s.Z; }
    PBRT_CPU_GPU
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return X;
        else if (c == 1)
            return Y;
        return Z;
    }
    PBRT_CPU_GPU
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return X;
        else if (c == 1)
            return Y;
        return Z;
    }

    std::string ToString() const;

    // XYZ Public Members
    Float X = 0, Y = 0, Z = 0;
};

PBRT_CPU_GPU
inline XYZ operator*(Float a, const XYZ &s) {
    return s * a;
}

template <typename U, typename V>
PBRT_CPU_GPU inline XYZ Clamp(const XYZ &xyz, U min, V max) {
    return XYZ(pbrt::Clamp(xyz.X, min, max), pbrt::Clamp(xyz.Y, min, max),
               pbrt::Clamp(xyz.Z, min, max));
}

PBRT_CPU_GPU
inline XYZ ClampZero(const XYZ &xyz) {
    return XYZ(std::max<Float>(0, xyz.X), std::max<Float>(0, xyz.Y),
               std::max<Float>(0, xyz.Z));
}

PBRT_CPU_GPU
inline XYZ Lerp(Float t, const XYZ &s1, const XYZ &s2) {
    return (1 - t) * s1 + t * s2;
}

// RGBSigmoidPolynomial Definition
class RGBSigmoidPolynomial {
  public:
    // RGBSigmoidPolynomial Public Methods
    RGBSigmoidPolynomial() = default;
    PBRT_CPU_GPU
    RGBSigmoidPolynomial(Float c0, Float c1, Float c2) : c0(c0), c1(c1), c2(c2) {}
    std::string ToString() const;

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        Float v = EvaluatePolynomial(lambda, c2, c1, c0);
        if (IsInf(v))
            return v > 0 ? 1 : 0;
        return s(v);
    }

    PBRT_CPU_GPU
    Float MaxValue() const {
        if (c0 < 0) {
            Float lambda = -c1 / (2 * c0);
            if (lambda >= 360 && lambda <= 830)
                return std::max({(*this)(lambda), (*this)(360), (*this)(830)});
        }
        return std::max((*this)(360), (*this)(830));
    }

  private:
    // RGBSigmoidPolynomial Private Methods
    PBRT_CPU_GPU
    static Float s(Float x) { return .5f + x / (2 * std::sqrt(1 + x * x)); };

    // RGBSigmoidPolynomial Private Members
    Float c0, c1, c2;
};

// RGBToSpectrumTable Definition
class RGBToSpectrumTable {
  public:
    // RGBToSpectrumTable Public Methods
    RGBToSpectrumTable(int res, const float *scale, const float *data)
        : res(res), scale(scale), data(data) {}

    PBRT_CPU_GPU
    RGBSigmoidPolynomial operator()(const RGB &rgb) const;

    static void Init(Allocator alloc);

    static const RGBToSpectrumTable *sRGB;
    static const RGBToSpectrumTable *DCI_P3;
    static const RGBToSpectrumTable *Rec2020;
    static const RGBToSpectrumTable *ACES2065_1;

    std::string ToString() const;

  private:
    // RGBToSpectrumTable Private Members
    int res = 0;
    const float *scale = nullptr, *data = nullptr;
};

// ColorEncoding Definitions
class LinearColorEncoding;
class sRGBColorEncoding;
class GammaColorEncoding;

class ColorEncodingHandle
    : public TaggedPointer<LinearColorEncoding, sRGBColorEncoding, GammaColorEncoding> {
  public:
    using TaggedPointer::TaggedPointer;
    // ColorEncoding Interface
    PBRT_CPU_GPU inline void ToLinear(pstd::span<const uint8_t> vin,
                                      pstd::span<Float> vout) const;
    PBRT_CPU_GPU inline void FromLinear(pstd::span<const Float> vin,
                                        pstd::span<uint8_t> vout) const;

    PBRT_CPU_GPU inline Float ToFloatLinear(Float v) const;

    std::string ToString() const;

    static const ColorEncodingHandle Get(const std::string &name, Allocator alloc);

    static ColorEncodingHandle Linear;
    static ColorEncodingHandle sRGB;

    static void Init(Allocator alloc);
};

class LinearColorEncoding {
  public:
    PBRT_CPU_GPU
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = vin[i] / 255.f;
    }

    PBRT_CPU_GPU
    Float ToFloatLinear(Float v) const { return v; }

    PBRT_CPU_GPU
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = uint8_t(Clamp(vin[i] * 255.f + 0.5f, 0, 255));
    }

    std::string ToString() const { return "[ LinearColorEncoding ]"; }
};

class sRGBColorEncoding {
  public:
    // sRGBColorEncoding Public Methods
    PBRT_CPU_GPU
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const;
    PBRT_CPU_GPU
    Float ToFloatLinear(Float v) const;
    PBRT_CPU_GPU
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const;

    std::string ToString() const { return "[ sRGBColorEncoding ]"; }
};

class GammaColorEncoding {
  public:
    PBRT_CPU_GPU
    GammaColorEncoding(Float gamma);

    PBRT_CPU_GPU
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const;
    PBRT_CPU_GPU
    Float ToFloatLinear(Float v) const;
    PBRT_CPU_GPU
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const;

    std::string ToString() const;

  private:
    Float gamma;
    pstd::array<Float, 256> applyLUT;
    pstd::array<Float, 1024> inverseLUT;
};

inline void ColorEncodingHandle::ToLinear(pstd::span<const uint8_t> vin,
                                          pstd::span<Float> vout) const {
    auto tolin = [&](auto ptr) { return ptr->ToLinear(vin, vout); };
    Dispatch(tolin);
}

inline Float ColorEncodingHandle::ToFloatLinear(Float v) const {
    auto tfl = [&](auto ptr) { return ptr->ToFloatLinear(v); };
    return Dispatch(tfl);
}

inline void ColorEncodingHandle::FromLinear(pstd::span<const Float> vin,
                                            pstd::span<uint8_t> vout) const {
    auto fl = [&](auto ptr) { return ptr->FromLinear(vin, vout); };
    Dispatch(fl);
}

PBRT_CPU_GPU
inline Float LinearToSRGB(Float value) {
    if (value <= 0.0031308f)
        return 12.92f * value;
    // Minimax polynomial approximation from enoki's color.h.
    Float sqrtValue = SafeSqrt(value);
    Float p = EvaluatePolynomial(sqrtValue, -0.0016829072605308378f, 0.03453868659826638f,
                                 0.7642611304733891f, 2.0041169284241644f,
                                 0.7551545191665577f, -0.016202083165206348f);
    Float q = EvaluatePolynomial(sqrtValue, 4.178892964897981e-7f,
                                 -0.00004375359692957097f, 0.03467195408529984f,
                                 0.6085338522168684f, 1.8970238036421054f, 1.f);
    return p / q * value;
}

PBRT_CPU_GPU
inline uint8_t LinearToSRGB8(Float value, Float dither = 0) {
    if (value <= 0)
        return 0;
    if (value >= 1)
        return 255;
    return Clamp(std::round(255.f * LinearToSRGB(value) + dither), 0, 255);
}

PBRT_CPU_GPU
inline Float SRGBToLinear(Float value) {
    if (value <= 0.04045f)
        return value * (1 / 12.92f);
    // Minimax polynomial approximation from enoki's color.h.
    Float p = EvaluatePolynomial(value, -0.0163933279112946f, -0.7386328024653209f,
                                 -11.199318357635072f, -47.46726633009393f,
                                 -36.04572663838034f);
    Float q = EvaluatePolynomial(value, -0.004261480793199332f, -19.140923959601675f,
                                 -59.096406619244426f, -18.225745396846637f, 1.f);
    return p / q * value;
}

extern PBRT_CONST Float SRGBToLinearLUT[256];

PBRT_CPU_GPU
inline Float SRGB8ToLinear(uint8_t value) {
    return SRGBToLinearLUT[value];
}

// White Balance Definitions
// clang-format off
// These are the Bradford transformation matrices.
const SquareMatrix<3> LMSFromXYZ( 0.8951,  0.2664, -0.1614,
                                 -0.7502,  1.7135,  0.0367,
                                  0.0389, -0.0685,  1.0296);
const SquareMatrix<3> XYZFromLMS( 0.986993,   -0.147054,  0.159963,
                                  0.432305,    0.51836,   0.0492912,
                                 -0.00852866,  0.0400428, 0.968487);
// clang-format on

inline SquareMatrix<3> WhiteBalance(Point2f srcWhite, Point2f targetWhite) {
    XYZ srcXYZ = XYZ::FromxyY(srcWhite), dstXYZ = XYZ::FromxyY(targetWhite);

    auto srcLMS = LMSFromXYZ * srcXYZ, dstLMS = LMSFromXYZ * dstXYZ;
    SquareMatrix<3> LMScorrect = SquareMatrix<3>::Diag(
        dstLMS[0] / srcLMS[0], dstLMS[1] / srcLMS[1], dstLMS[2] / srcLMS[2]);

    return XYZFromLMS * LMScorrect * LMSFromXYZ;
}

}  // namespace pbrt

#endif  // PBRT_UTIL_COLOR_H
