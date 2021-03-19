// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SCATTERING_H
#define PBRT_UTIL_SCATTERING_H

#include <pbrt/pbrt.h>

#include <pbrt/util/math.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

// Scattering Inline Functions
PBRT_CPU_GPU
inline Vector3f Reflect(const Vector3f &wo, const Vector3f &n) {
    return -wo + 2 * Dot(wo, n) * n;
}

PBRT_CPU_GPU inline bool Refract(Vector3f wi, Normal3f n, Float eta, Vector3f *wt) {
    // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    Float cosTheta_i = Dot(n, wi);
    Float sin2Theta_i = std::max<Float>(0, 1 - cosTheta_i * cosTheta_i);
    Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    // Handle total internal reflection for transmission
    if (sin2Theta_t >= 1)
        return false;

    Float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * Vector3f(n);
    return true;
}

PBRT_CPU_GPU inline Float HenyeyGreenstein(Float cosTheta, Float g) {
    Float denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

// Fresnel Inline Functions
PBRT_CPU_GPU
inline Float FrConductor2(Float cosTheta_i, pstd::complex<Float> eta) {
    cosTheta_i = Clamp(cosTheta_i, 0, 1);
    using Complex = pstd::complex<Float>;

    // Compute _cosTheta_t_ using Snell's law
    Complex sinTheta_i = pstd::sqrt(1.f - cosTheta_i * cosTheta_i);
    Complex sinTheta_t = sinTheta_i / eta;
    Complex cosTheta_t = pstd::sqrt(1.f - sinTheta_t * sinTheta_t);
    Complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Complex r_perp = (eta * cosTheta_t - cosTheta_i) / (eta * cosTheta_t + cosTheta_i);
    return (pstd::norm(r_parl) + pstd::norm(r_perp)) * .5f;
}

PBRT_CPU_GPU
inline SampledSpectrum FrConductor(Float cosTheta_i, SampledSpectrum eta,
                                   SampledSpectrum k) {
    SampledSpectrum result;
    for (int i = 0; i < NSpectrumSamples; ++i)
        result[i] = FrConductor2(cosTheta_i, pstd::complex<Float>(eta[i], k[i]));
    return result;
}

PBRT_CPU_GPU
inline Float FrDielectric(Float cosTheta_i, Float eta) {
    cosTheta_i = Clamp(cosTheta_i, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosTheta_i > 0;
    if (!entering) {
        eta = 1 / eta;
        cosTheta_i = std::abs(cosTheta_i);
    }

    // Compute _cosTheta_t_ using Snell's law
    Float sinTheta_i = SafeSqrt(1 - cosTheta_i * cosTheta_i);
    Float sinTheta_t = sinTheta_i / eta;
    // Handle total internal reflection
    if (sinTheta_t >= 1)
        return 1;

    Float cosTheta_t = SafeSqrt(1 - sinTheta_t * sinTheta_t);

    Float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (r_parl * r_parl + r_perp * r_perp) / 2;
}

// BSSRDF Utility Declarations
PBRT_CPU_GPU
Float FresnelMoment1(Float invEta);
PBRT_CPU_GPU
Float FresnelMoment2(Float invEta);

// TrowbridgeReitzDistribution Definition
class TrowbridgeReitzDistribution {
  public:
    // TrowbridgeReitzDistribution Public Methods
    TrowbridgeReitzDistribution() = default;
    PBRT_CPU_GPU
    TrowbridgeReitzDistribution(Float alpha_x, Float alpha_y)
        : alpha_x(alpha_x), alpha_y(alpha_y) {}

    PBRT_CPU_GPU
    static Float RoughnessToAlpha(Float roughness) { return std::sqrt(roughness); }

    PBRT_CPU_GPU inline Float D(const Vector3f &wm) const {
        Float tan2Theta = Tan2Theta(wm);
        if (IsInf(tan2Theta))
            return 0;
        Float cos4Theta = Cos2Theta(wm) * Cos2Theta(wm);
        if (cos4Theta < 1e-16f)
            return 0;
        Float e =
            tan2Theta * (Sqr(CosPhi(wm)) / Sqr(alpha_x) + Sqr(SinPhi(wm)) / Sqr(alpha_y));
        return 1 / (Pi * alpha_x * alpha_y * cos4Theta * Sqr(1 + e));
    }

    PBRT_CPU_GPU
    bool EffectivelySmooth() const { return std::min(alpha_x, alpha_y) < 1e-3f; }

    PBRT_CPU_GPU
    Float G1(const Vector3f &w) const { return 1 / (1 + Lambda(w)); }

    PBRT_CPU_GPU
    Float Lambda(const Vector3f &w) const {
        Float tan2Theta = Tan2Theta(w);
        if (IsInf(tan2Theta))
            return 0.;
        // Compute _alpha2_ for direction _w_
        Float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);

        return (-1 + std::sqrt(1 + alpha2 * tan2Theta)) / 2;
    }

    PBRT_CPU_GPU
    Float G(const Vector3f &wo, const Vector3f &wi) const {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }

    PBRT_CPU_GPU
    Vector3f Sample_wm(Point2f u) const {
        return SampleTrowbridgeReitz(alpha_x, alpha_y, u);
    }

    PBRT_CPU_GPU
    Float D(const Vector3f &w, const Vector3f &wm) const {
        return D(wm) * G1(w) * std::max<Float>(0, Dot(w, wm)) / AbsCosTheta(w);
    }

    PBRT_CPU_GPU
    Vector3f Sample_wm(Vector3f wo, Point2f u) const {
        bool flip = wo.z < 0;
        Vector3f wm =
            SampleTrowbridgeReitzVisibleArea(flip ? -wo : wo, alpha_x, alpha_y, u);
        return flip ? -wm : wm;
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wm) const {
        return D(wm) * G1(wo) * AbsDot(wo, wm) / AbsCosTheta(wo);
    }

    PBRT_CPU_GPU
    void Regularize() {
        if (alpha_x < 0.3f)
            alpha_x = Clamp(2 * alpha_x, 0.1f, 0.3f);
        if (alpha_y < 0.3f)
            alpha_y = Clamp(2 * alpha_y, 0.1f, 0.3f);
    }

  private:
    // TrowbridgeReitzDistribution Private Members
    Float alpha_x, alpha_y;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_SCATTERING_H
