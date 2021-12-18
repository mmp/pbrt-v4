// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SPECTRUM_H
#define PBRT_UTIL_SPECTRUM_H

// PhysLight code contributed by Anders Langlands and Luca Fascione
// Copyright (c) 2020, Weta Digital, Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/float.hpp>
#include <pbrt/util/hash.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/taggedptr.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace pbrt {

// Spectrum Constants
constexpr Float Lambda_min = 360, Lambda_max = 830;

static constexpr int NSpectrumSamples = 4;

static constexpr Float CIE_Y_integral = 106.856895;

// Spectrum Definition
class BlackbodySpectrum;
class ConstantSpectrum;
class PiecewiseLinearSpectrum;
class DenselySampledSpectrum;
class RGBAlbedoSpectrum;
class RGBUnboundedSpectrum;
class RGBIlluminantSpectrum;

class Spectrum : public TaggedPointer<ConstantSpectrum, DenselySampledSpectrum,
                                      PiecewiseLinearSpectrum, RGBAlbedoSpectrum,
                                      RGBUnboundedSpectrum, RGBIlluminantSpectrum,
                                      BlackbodySpectrum> {
  public:
    // Spectrum Interface
    using TaggedPointer::TaggedPointer;
    std::string ToString() const;

    PBRT_CPU_GPU
    Float operator()(Float lambda) const;

    PBRT_CPU_GPU
    Float MaxValue() const;

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;
};

// Spectrum Function Declarations
PBRT_CPU_GPU inline Float Blackbody(Float lambda, Float T) {
    if (T <= 0)
        return 0;
    const Float c = 299792458.f;
    const Float h = 6.62606957e-34f;
    const Float kb = 1.3806488e-23f;
    // Return emitted radiance for blackbody at wavelength _lambda_
    Float l = lambda * 1e-9f;
    Float Le = (2 * h * c * c) / (Pow<5>(l) * (FastExp((h * c) / (l * kb * T)) - 1));
    CHECK(!IsNaN(Le));
    return Le;
}

namespace Spectra {
DenselySampledSpectrum D(Float T, Allocator alloc);
}  // namespace Spectra

Float SpectrumToPhotometric(Spectrum s);

Float SpectrumToPhotometric(Spectrum s);
XYZ SpectrumToXYZ(Spectrum s);

// SampledSpectrum Definition
class SampledSpectrum {
  public:
    // SampledSpectrum Public Methods
    PBRT_CPU_GPU
    SampledSpectrum operator+(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret += s;
    }

    PBRT_CPU_GPU
    SampledSpectrum &operator-=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] -= s.values[i];
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator-(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret -= s;
    }
    PBRT_CPU_GPU
    friend SampledSpectrum operator-(Float a, const SampledSpectrum &s) {
        DCHECK(!IsNaN(a));
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.values[i] = a - s.values[i];
        return ret;
    }

    PBRT_CPU_GPU
    SampledSpectrum &operator*=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] *= s.values[i];
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator*(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret *= s;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator*(Float a) const {
        DCHECK(!IsNaN(a));
        SampledSpectrum ret = *this;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.values[i] *= a;
        return ret;
    }
    PBRT_CPU_GPU
    SampledSpectrum &operator*=(Float a) {
        DCHECK(!IsNaN(a));
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] *= a;
        return *this;
    }
    PBRT_CPU_GPU
    friend SampledSpectrum operator*(Float a, const SampledSpectrum &s) { return s * a; }

    PBRT_CPU_GPU
    SampledSpectrum &operator/=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i) {
            DCHECK_NE(0, s.values[i]);
            values[i] /= s.values[i];
        }
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator/(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret /= s;
    }
    PBRT_CPU_GPU
    SampledSpectrum &operator/=(Float a) {
        DCHECK_NE(a, 0);
        DCHECK(!IsNaN(a));
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] /= a;
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator/(Float a) const {
        SampledSpectrum ret = *this;
        return ret /= a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator-() const {
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.values[i] = -values[i];
        return ret;
    }
    PBRT_CPU_GPU
    bool operator==(const SampledSpectrum &s) const { return values == s.values; }
    PBRT_CPU_GPU
    bool operator!=(const SampledSpectrum &s) const { return values != s.values; }

    std::string ToString() const;

    PBRT_CPU_GPU
    bool HasNaNs() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (IsNaN(values[i]))
                return true;
        return false;
    }

    PBRT_CPU_GPU
    XYZ ToXYZ(const SampledWavelengths &lambda) const;
    PBRT_CPU_GPU
    RGB ToRGB(const SampledWavelengths &lambda, const RGBColorSpace &cs) const;
    PBRT_CPU_GPU
    Float y(const SampledWavelengths &lambda) const;

    SampledSpectrum() = default;
    PBRT_CPU_GPU
    explicit SampledSpectrum(Float c) { values.fill(c); }
    PBRT_CPU_GPU
    SampledSpectrum(pstd::span<const Float> v) {
        DCHECK_EQ(NSpectrumSamples, v.size());
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] = v[i];
    }

    PBRT_CPU_GPU
    Float operator[](int i) const {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return values[i];
    }
    PBRT_CPU_GPU
    Float &operator[](int i) {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return values[i];
    }

    PBRT_CPU_GPU
    explicit operator bool() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (values[i] != 0)
                return true;
        return false;
    }

    PBRT_CPU_GPU
    SampledSpectrum &operator+=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            values[i] += s.values[i];
        return *this;
    }

    PBRT_CPU_GPU
    Float MinComponentValue() const {
        Float m = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::min(m, values[i]);
        return m;
    }
    PBRT_CPU_GPU
    Float MaxComponentValue() const {
        Float m = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::max(m, values[i]);
        return m;
    }
    PBRT_CPU_GPU
    Float Average() const {
        Float sum = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            sum += values[i];
        return sum / NSpectrumSamples;
    }

  private:
    friend struct SOA<SampledSpectrum>;
    pstd::array<Float, NSpectrumSamples> values;
};

// SampledWavelengths Definitions
class SampledWavelengths {
  public:
    // SampledWavelengths Public Methods
    PBRT_CPU_GPU
    bool operator==(const SampledWavelengths &swl) const {
        return lambda == swl.lambda && pdf == swl.pdf;
    }
    PBRT_CPU_GPU
    bool operator!=(const SampledWavelengths &swl) const {
        return lambda != swl.lambda || pdf != swl.pdf;
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    static SampledWavelengths SampleUniform(Float u, Float lambda_min = Lambda_min,
                                            Float lambda_max = Lambda_max) {
        SampledWavelengths swl;
        // Sample first wavelength using _u_
        swl.lambda[0] = Lerp(u, lambda_min, lambda_max);

        // Initialize _lambda_ for remaining wavelengths
        Float delta = (lambda_max - lambda_min) / NSpectrumSamples;
        for (int i = 1; i < NSpectrumSamples; ++i) {
            swl.lambda[i] = swl.lambda[i - 1] + delta;
            if (swl.lambda[i] > lambda_max)
                swl.lambda[i] = lambda_min + (swl.lambda[i] - lambda_max);
        }

        // Compute PDF for sampled wavelengths
        for (int i = 0; i < NSpectrumSamples; ++i)
            swl.pdf[i] = 1 / (lambda_max - lambda_min);

        return swl;
    }

    PBRT_CPU_GPU
    Float operator[](int i) const { return lambda[i]; }
    PBRT_CPU_GPU
    Float &operator[](int i) { return lambda[i]; }
    PBRT_CPU_GPU
    SampledSpectrum PDF() const { return SampledSpectrum(pdf); }

    PBRT_CPU_GPU
    void TerminateSecondary() {
        if (SecondaryTerminated())
            return;
        // Update wavelength probabilities for termination
        for (int i = 1; i < NSpectrumSamples; ++i)
            pdf[i] = 0;
        pdf[0] /= NSpectrumSamples;
    }

    PBRT_CPU_GPU
    bool SecondaryTerminated() const {
        for (int i = 1; i < NSpectrumSamples; ++i)
            if (pdf[i] != 0)
                return false;
        return true;
    }

    PBRT_CPU_GPU
    static SampledWavelengths SampleVisible(Float u) {
        SampledWavelengths swl;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            // Compute _up_ for $i$th wavelength sample
            Float up = u + Float(i) / NSpectrumSamples;
            if (up > 1)
                up -= 1;

            swl.lambda[i] = SampleVisibleWavelengths(up);
            swl.pdf[i] = VisibleWavelengthsPDF(swl.lambda[i]);
        }
        return swl;
    }

  private:
    // SampledWavelengths Private Members
    friend struct SOA<SampledWavelengths>;
    pstd::array<Float, NSpectrumSamples> lambda, pdf;
};

// Spectrum Definitions
class ConstantSpectrum {
  public:
    PBRT_CPU_GPU
    ConstantSpectrum(Float c) : c(c) {}
    PBRT_CPU_GPU
    Float operator()(Float lambda) const { return c; }
    // ConstantSpectrum Public Methods
    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &) const;

    PBRT_CPU_GPU
    Float MaxValue() const { return c; }

    std::string ToString() const;

  private:
    Float c;
};

class DenselySampledSpectrum {
  public:
    // DenselySampledSpectrum Public Methods
    DenselySampledSpectrum(int lambda_min = Lambda_min, int lambda_max = Lambda_max,
                           Allocator alloc = {})
        : lambda_min(lambda_min),
          lambda_max(lambda_max),
          values(lambda_max - lambda_min + 1, alloc) {}
    DenselySampledSpectrum(Spectrum s, Allocator alloc)
        : DenselySampledSpectrum(s, Lambda_min, Lambda_max, alloc) {}
    DenselySampledSpectrum(const DenselySampledSpectrum &s, Allocator alloc)
        : lambda_min(s.lambda_min),
          lambda_max(s.lambda_max),
          values(s.values.begin(), s.values.end(), alloc) {}

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            int offset = std::lround(lambda[i]) - lambda_min;
            if (offset < 0 || offset >= values.size())
                s[i] = 0;
            else
                s[i] = values[offset];
        }
        return s;
    }

    PBRT_CPU_GPU
    void Scale(Float s) {
        for (Float &v : values)
            v *= s;
    }

    PBRT_CPU_GPU
    Float MaxValue() const { return *std::max_element(values.begin(), values.end()); }

    std::string ToString() const;

    DenselySampledSpectrum(Spectrum spec, int lambda_min = Lambda_min,
                           int lambda_max = Lambda_max, Allocator alloc = {})
        : lambda_min(lambda_min),
          lambda_max(lambda_max),
          values(lambda_max - lambda_min + 1, alloc) {
        CHECK_GE(lambda_max, lambda_min);
        if (spec)
            for (int lambda = lambda_min; lambda <= lambda_max; ++lambda)
                values[lambda - lambda_min] = spec(lambda);
    }

    template <typename F>
    static DenselySampledSpectrum SampleFunction(F func, int lambda_min = Lambda_min,
                                                 int lambda_max = Lambda_max,
                                                 Allocator alloc = {}) {
        DenselySampledSpectrum s(lambda_min, lambda_max, alloc);
        for (int lambda = lambda_min; lambda <= lambda_max; ++lambda)
            s.values[lambda - lambda_min] = func(lambda);
        return s;
    }

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        DCHECK_GT(lambda, 0);
        int offset = std::lround(lambda) - lambda_min;
        if (offset < 0 || offset >= values.size())
            return 0;
        return values[offset];
    }

    PBRT_CPU_GPU
    bool operator==(const DenselySampledSpectrum &d) const {
        if (lambda_min != d.lambda_min || lambda_max != d.lambda_max ||
            values.size() != d.values.size())
            return false;
        for (size_t i = 0; i < values.size(); ++i)
            if (values[i] != d.values[i])
                return false;
        return true;
    }

  private:
    friend struct std::hash<pbrt::DenselySampledSpectrum>;
    // DenselySampledSpectrum Private Members
    int lambda_min, lambda_max;
    pstd::vector<Float> values;
};

class PiecewiseLinearSpectrum {
  public:
    // PiecewiseLinearSpectrum Public Methods
    PiecewiseLinearSpectrum() = default;

    PBRT_CPU_GPU
    void Scale(Float s) {
        for (Float &v : values)
            v *= s;
    }

    PBRT_CPU_GPU
    Float MaxValue() const;

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = (*this)(lambda[i]);
        return s;
    }
    PBRT_CPU_GPU
    Float operator()(Float lambda) const;

    std::string ToString() const;

    PiecewiseLinearSpectrum(pstd::span<const Float> lambdas,
                            pstd::span<const Float> values, Allocator alloc = {});

    static pstd::optional<Spectrum> Read(const std::string &filename, Allocator alloc);

    static PiecewiseLinearSpectrum *FromInterleaved(pstd::span<const Float> samples,
                                                    bool normalize, Allocator alloc);

  private:
    // PiecewiseLinearSpectrum Private Members
    pstd::vector<Float> lambdas, values;
};

class BlackbodySpectrum {
  public:
    // BlackbodySpectrum Public Methods
    PBRT_CPU_GPU
    BlackbodySpectrum(Float T) : T(T) {
        // Compute blackbody normalization constant for given temperature
        Float lambdaMax = 2.8977721e-3f / T;
        normalizationFactor = 1 / Blackbody(lambdaMax * 1e9f, T);
    }

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        return Blackbody(lambda, T) * normalizationFactor;
    }

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = Blackbody(lambda[i], T) * normalizationFactor;
        return s;
    }

    PBRT_CPU_GPU
    Float MaxValue() const { return 1.f; }

    std::string ToString() const;

  private:
    // BlackbodySpectrum Private Members
    Float T;
    Float normalizationFactor;
};

class RGBAlbedoSpectrum {
  public:
    // RGBAlbedoSpectrum Public Methods
    PBRT_CPU_GPU
    Float operator()(Float lambda) const { return rsp(lambda); }
    PBRT_CPU_GPU
    Float MaxValue() const { return rsp.MaxValue(); }

    PBRT_CPU_GPU
    RGBAlbedoSpectrum(const RGBColorSpace &cs, RGB rgb);

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = rsp(lambda[i]);
        return s;
    }

    std::string ToString() const;

  private:
    // RGBAlbedoSpectrum Private Members
    RGBSigmoidPolynomial rsp;
};

class RGBUnboundedSpectrum {
  public:
    // RGBUnboundedSpectrum Public Methods
    PBRT_CPU_GPU
    Float operator()(Float lambda) const { return scale * rsp(lambda); }
    PBRT_CPU_GPU
    Float MaxValue() const { return scale * rsp.MaxValue(); }

    PBRT_CPU_GPU
    RGBUnboundedSpectrum(const RGBColorSpace &cs, RGB rgb);

    PBRT_CPU_GPU
    RGBUnboundedSpectrum() : rsp(0, 0, 0), scale(0) {}

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = scale * rsp(lambda[i]);
        return s;
    }

    std::string ToString() const;

  private:
    // RGBUnboundedSpectrum Private Members
    Float scale = 1;
    RGBSigmoidPolynomial rsp;
};

class RGBIlluminantSpectrum {
  public:
    // RGBIlluminantSpectrum Public Methods
    RGBIlluminantSpectrum() = default;
    PBRT_CPU_GPU
    RGBIlluminantSpectrum(const RGBColorSpace &cs, RGB rgb);

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        if (!illuminant)
            return 0;
        return scale * rsp(lambda) * (*illuminant)(lambda);
    }

    PBRT_CPU_GPU
    Float MaxValue() const {
        if (!illuminant)
            return 0;
        return scale * rsp.MaxValue() * illuminant->MaxValue();
    }

    PBRT_CPU_GPU
    const DenselySampledSpectrum *Illuminant() const { return illuminant; }

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        if (!illuminant)
            return SampledSpectrum(0);
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = scale * rsp(lambda[i]);
        return s * illuminant->Sample(lambda);
    }

    std::string ToString() const;

  private:
    // RGBIlluminantSpectrum Private Members
    Float scale;
    RGBSigmoidPolynomial rsp;
    const DenselySampledSpectrum *illuminant;
};

// SampledSpectrum Inline Functions
PBRT_CPU_GPU inline SampledSpectrum SafeDiv(SampledSpectrum a, SampledSpectrum b) {
    SampledSpectrum r;
    for (int i = 0; i < NSpectrumSamples; ++i)
        r[i] = (b[i] != 0) ? a[i] / b[i] : 0.;
    return r;
}

template <typename U, typename V>
PBRT_CPU_GPU inline SampledSpectrum Clamp(const SampledSpectrum &s, U low, V high) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = pbrt::Clamp(s[i], low, high);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum ClampZero(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::max<Float>(0, s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum Sqrt(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::sqrt(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum SafeSqrt(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = SafeSqrt(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum Pow(const SampledSpectrum &s, Float e) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::pow(s[i], e);
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum Exp(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::exp(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum FastExp(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = FastExp(s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum Bilerp(pstd::array<Float, 2> p,
                              pstd::span<const SampledSpectrum> v) {
    return ((1 - p[0]) * (1 - p[1]) * v[0] + p[0] * (1 - p[1]) * v[1] +
            (1 - p[0]) * p[1] * v[2] + p[0] * p[1] * v[3]);
}

PBRT_CPU_GPU
inline SampledSpectrum Lerp(Float t, const SampledSpectrum &s1,
                            const SampledSpectrum &s2) {
    return (1 - t) * s1 + t * s2;
}

// Spectral Data Declarations
namespace Spectra {

void Init(Allocator alloc);

PBRT_CPU_GPU
inline const DenselySampledSpectrum &X() {
#ifdef PBRT_IS_GPU_CODE
    extern PBRT_GPU DenselySampledSpectrum *xGPU;
    return *xGPU;
#else
    extern DenselySampledSpectrum *x;
    return *x;
#endif
}

PBRT_CPU_GPU
inline const DenselySampledSpectrum &Y() {
#ifdef PBRT_IS_GPU_CODE
    extern PBRT_GPU DenselySampledSpectrum *yGPU;
    return *yGPU;
#else
    extern DenselySampledSpectrum *y;
    return *y;
#endif
}

PBRT_CPU_GPU
inline const DenselySampledSpectrum &Z() {
#ifdef PBRT_IS_GPU_CODE
    extern PBRT_GPU DenselySampledSpectrum *zGPU;
    return *zGPU;
#else
    extern DenselySampledSpectrum *z;
    return *z;
#endif
}

}  // namespace Spectra

// Spectral Function Declarations
Spectrum GetNamedSpectrum(std::string name);

std::string FindMatchingNamedSpectrum(Spectrum s);

namespace Spectra {
inline const DenselySampledSpectrum &X();
inline const DenselySampledSpectrum &Y();
inline const DenselySampledSpectrum &Z();
}  // namespace Spectra

// Spectrum Inline Functions
inline Float InnerProduct(Spectrum f, Spectrum g) {
    Float integral = 0;
    for (Float lambda = Lambda_min; lambda <= Lambda_max; ++lambda)
        integral += f(lambda) * g(lambda);
    return integral;
}

// Spectrum Inline Method Definitions
inline Float Spectrum::operator()(Float lambda) const {
    auto op = [&](auto ptr) { return (*ptr)(lambda); };
    return Dispatch(op);
}

inline SampledSpectrum Spectrum::Sample(const SampledWavelengths &lambda) const {
    auto samp = [&](auto ptr) { return ptr->Sample(lambda); };
    return Dispatch(samp);
}

inline Float Spectrum::MaxValue() const {
    auto max = [&](auto ptr) { return ptr->MaxValue(); };
    return Dispatch(max);
}

}  // namespace pbrt

namespace std {

template <>
struct hash<pbrt::DenselySampledSpectrum> {
    PBRT_CPU_GPU
    size_t operator()(const pbrt::DenselySampledSpectrum &s) const {
        return pbrt::HashBuffer(s.values.data(), s.values.size());
    }
};

}  // namespace std

#endif  // PBRT_UTIL_SPECTRUM_H
