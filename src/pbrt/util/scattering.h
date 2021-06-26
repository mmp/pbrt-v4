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

PBRT_CPU_GPU inline bool Refract(Vector3f wi, Normal3f n, Float eta, Float *etap,
                                 Vector3f *wt) {
    Float cosTheta_i = Dot(n, wi);
    // Potentially flip interface orientation for Snell's law
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }

    // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    Float sin2Theta_i = std::max<Float>(0, 1 - Sqr(cosTheta_i));
    Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    // Handle total internal reflection case
    if (sin2Theta_t >= 1)
        return false;

    Float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * Vector3f(n);
    // Provide relative IOR along ray to caller
    if (etap)
        *etap = eta;

    return true;
}

PBRT_CPU_GPU inline Float HenyeyGreenstein(Float cosTheta, Float g) {
    Float denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

// Fresnel Inline Functions
PBRT_CPU_GPU
inline Float FrDielectric(Float cosTheta_i, Float eta) {
    cosTheta_i = Clamp(cosTheta_i, -1, 1);
    // Potentially flip interface orientation for Fresnel equations
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }

    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Float sin2Theta_i = 1 - Sqr(cosTheta_i);
    Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1)
        return 1.f;
    Float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    Float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (r_parl * r_parl + r_perp * r_perp) / 2;
}

PBRT_CPU_GPU
inline Float FrComplex(Float cosTheta_i, pstd::complex<Float> eta) {
    using Complex = pstd::complex<Float>;
    cosTheta_i = Clamp(cosTheta_i, 0, 1);
    // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Float sin2Theta_i = 1 - Sqr(cosTheta_i);
    Complex sin2Theta_t = sin2Theta_i / Sqr(eta);
    Complex cosTheta_t = pstd::sqrt(1.f - sin2Theta_t);

    Complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Complex r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (pstd::norm(r_parl) + pstd::norm(r_perp)) * .5f;
}

PBRT_CPU_GPU
inline SampledSpectrum FrComplex(Float cosTheta_i, SampledSpectrum eta,
                                 SampledSpectrum k) {
    SampledSpectrum result;
    for (int i = 0; i < NSpectrumSamples; ++i)
        result[i] = FrComplex(cosTheta_i, pstd::complex<Float>(eta[i], k[i]));
    return result;
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

    PBRT_CPU_GPU inline Float D(const Vector3f &wm) const {
        Float tan2Theta = Tan2Theta(wm);
        if (IsInf(tan2Theta))
            return 0;
        Float cos4Theta = Sqr(Cos2Theta(wm));
        if (cos4Theta < 1e-16f)
            return 0;
        Float e = tan2Theta * (Sqr(CosPhi(wm) / alpha_x) + Sqr(SinPhi(wm) / alpha_y));
        return 1 / (Pi * alpha_x * alpha_y * cos4Theta * Sqr(1 + e));
    }

    PBRT_CPU_GPU
    bool EffectivelySmooth() const { return std::max(alpha_x, alpha_y) < 1e-3f; }

    PBRT_CPU_GPU
    Float G1(const Vector3f &w) const { return 1 / (1 + Lambda(w)); }

    PBRT_CPU_GPU
    Float Lambda(const Vector3f &w) const {
        Float tan2Theta = Tan2Theta(w);
        if (IsInf(tan2Theta))
            return 0;
        Float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);
        return .5f * (std::sqrt(1 + alpha2 * tan2Theta) - 1);
    }

    PBRT_CPU_GPU
    Float G(const Vector3f &wo, const Vector3f &wi) const {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }

    PBRT_CPU_GPU
    Float D(const Vector3f &w, const Vector3f &wm) const {
        return D(wm) * G1(w) * std::abs(Dot(w, wm) / CosTheta(w));
    }

    PBRT_CPU_GPU
    Vector3f Sample_wm(Vector3f w, Point2f u) const {
        // Transform _w_ to hemispherical configuration for visible area sampling
        Vector3f wh = Normalize(Vector3f(alpha_x * w.x, alpha_y * w.y, w.z));
        if (w.z < 0)
            wh = -wh;

        // Find orthonormal basis for visible area microfacet sampling
        Vector3f T1 = (wh.z < 0.99999f) ? Normalize(Cross(Vector3f(0, 0, 1), wh))
                                        : Vector3f(1, 0, 0);
        Vector3f T2 = Cross(wh, T1);

        // Sample parameterization of projected microfacet area
        Float r = std::sqrt(u[0]);
        Float phi = 2 * Pi * u[1];
        Float t1 = r * std::cos(phi), t2 = r * std::sin(phi);
        Float s = .5f * (1 + wh.z);
        t2 = (1 - s) * std::sqrt(1 - Sqr(t1)) + s * t2;

        // Reproject to hemisphere and transform normal to ellipsoid configuration
        Vector3f nh =
            t1 * T1 + t2 * T2 + std::sqrt(std::max<Float>(0, 1 - Sqr(t1) - Sqr(t2))) * wh;
        CHECK_RARE(1e-5f, nh.z == 0);
        return Normalize(
            Vector3f(alpha_x * nh.x, alpha_y * nh.y, std::max<Float>(1e-6f, nh.z)));
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wm) const {
        return D(wm) * G1(wo) * AbsDot(wo, wm) / AbsCosTheta(wo);
    }

    PBRT_CPU_GPU
    static Float RoughnessToAlpha(Float roughness) { return std::sqrt(roughness); }

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
