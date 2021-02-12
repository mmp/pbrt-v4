// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/bxdfs.h>

#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/media.h>
#include <pbrt/options.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/float.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/stats.h>

#include <unordered_map>

namespace pbrt {

std::string ToString(BxDFReflTransFlags flags) {
    if (flags == BxDFReflTransFlags::Unset)
        return "Unset";
    std::string s;
    if (flags & BxDFReflTransFlags::Reflection)
        s += "Reflection,";
    if (flags & BxDFReflTransFlags::Transmission)
        s += "Transmission,";
    return s;
}

std::string ToString(BxDFFlags flags) {
    if (flags == BxDFFlags::Unset)
        return "Unset";
    std::string s;
    if (flags & BxDFFlags::Reflection)
        s += "Reflection,";
    if (flags & BxDFFlags::Transmission)
        s += "Transmission,";
    if (flags & BxDFFlags::Diffuse)
        s += "Diffuse,";
    if (flags & BxDFFlags::Glossy)
        s += "Glossy,";
    if (flags & BxDFFlags::Specular)
        s += "Specular,";
    return s;
}

std::string ToString(TransportMode mode) {
    return mode == TransportMode::Radiance ? "Radiance" : "Importance";
}

// BxDF Method Definitions
std::string IdealDiffuseBxDF::ToString() const {
    return StringPrintf("[ IdealDiffuseBxDF R: %s ]", R);
}
std::string DiffuseBxDF::ToString() const {
    return StringPrintf("[ DiffuseBxDF R: %s T: %s A: %f B: %f ]", R, T, A, B);
}

template <typename TopBxDF, typename BottomBxDF>
std::string LayeredBxDF<TopBxDF, BottomBxDF>::ToString() const {
    return StringPrintf(
        "[ LayeredBxDF top: %s bottom: %s thickness: %f albedo: %s g: %f ]", top, bottom,
        thickness, albedo, g);
}

// DielectricInterfaceBxDF Method Definitions
SampledSpectrum DielectricInterfaceBxDF::f(Vector3f wo, Vector3f wi,
                                           TransportMode mode) const {
    if (mfDistrib.EffectivelySmooth())
        return SampledSpectrum(0);
    if (SameHemisphere(wo, wi)) {
        // Compute reflection at non-delta dielectric interface
        Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        Vector3f wh = wi + wo;
        // Handle degenerate cases for microfacet reflection
        if (cosTheta_i == 0 || cosTheta_o == 0)
            return SampledSpectrum(0.);
        if (wh.x == 0 && wh.y == 0 && wh.z == 0)
            return SampledSpectrum(0.);
        wh = Normalize(wh);
        Float F = FrDielectric(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))), eta);
        return tint * SampledSpectrum(mfDistrib.D(wh) * mfDistrib.G(wo, wi) * F /
                                      (4 * cosTheta_i * cosTheta_o));

    } else {
        // Compute transmission at non-delta dielectric interface
        Float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
        if (cosTheta_i == 0 || cosTheta_o == 0)
            return {};
        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
        Vector3f wh = wo + wi * etap;
        CHECK_RARE(1e-5f, LengthSquared(wh) == 0);
        if (LengthSquared(wh) == 0)
            return {};
        wh = FaceForward(Normalize(wh), Normal3f(0, 0, 1));

        // Return no transmission if _wi_ and _wo_ are on the same side of _wh_
        if (Dot(wi, wh) * Dot(wo, wh) > 0)
            return {};

        // Evaluate BTDF for transmission through microfacet interface
        Float F = FrDielectric(Dot(wo, wh), eta);
        Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
        Float factor = (mode == TransportMode::Radiance) ? Sqr(1 / etap) : 1;
        return tint *
               SampledSpectrum((1 - F) * factor *
                               std::abs(mfDistrib.D(wh) * mfDistrib.G(wo, wi) *
                                        AbsDot(wi, wh) * AbsDot(wo, wh) /
                                        (cosTheta_i * cosTheta_o * Sqr(sqrtDenom))));
    }
}

pstd::optional<BSDFSample> DielectricInterfaceBxDF::Sample_f(
    Vector3f wo, Float uc, Point2f u, TransportMode mode,
    BxDFReflTransFlags sampleFlags) const {
    if (mfDistrib.EffectivelySmooth()) {
        // Sample delta dielectric interface
        Float R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        Float pr = R, pt = T;
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        if (uc < pr / (pr + pt)) {
            // Sample perfect specular reflection at interface
            Vector3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum fr(tint * R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);

        } else {
            // Sample perfect specular transmission at interface
            // Figure out which $\eta$ is incident and which is transmitted
            bool entering = CosTheta(wo) > 0;
            Float etap = entering ? eta : (1 / eta);

            // Compute ray direction for specular transmission
            Vector3f wi;
            bool tir = !Refract(wo, FaceForward(Normal3f(0, 0, 1), wo), etap, &wi);
            CHECK_RARE(1e-5f, tir);
            if (tir)
                return {};

            SampledSpectrum ft(tint * T / AbsCosTheta(wi));
            // Account for non-symmetry with transmission to different medium
            if (mode == TransportMode::Radiance)
                ft /= Sqr(etap);

            return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission,
                              etap);
        }

    } else {
        // Sample non-delta dielectric interface
        // Sample half-angle vector for outgoing direction and compute Frensel factor
        Vector3f wh = mfDistrib.Sample_wm(wo, u);
        Float F =
            FrDielectric(Dot(Reflect(wo, wh), FaceForward(wh, Vector3f(0, 0, 1))), eta);
        Float R = F, T = 1 - R;

        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        Float pr = R, pt = T;
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        if (uc < pr / (pr + pt)) {
            // Sample reflection at non-delta dielectric interface
            Vector3f wi = Reflect(wo, wh);
            CHECK_RARE(1e-5f, Dot(wo, wh) <= 0);
            if (!SameHemisphere(wo, wi) || Dot(wo, wh) <= 0)
                return {};
            // Compute PDF of direction $\wi$ for microfacet reflection
            Float pdf = mfDistrib.PDF(wo, wh) / (4 * Dot(wo, wh)) * pr / (pr + pt);
            CHECK(!IsNaN(pdf));

            // Evaluate BRDF and return _BSDFSample_ for dielectric microfacet reflection
            Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
            if (cosTheta_i == 0 || cosTheta_o == 0)
                return {};
            SampledSpectrum f(mfDistrib.D(wh) * mfDistrib.G(wo, wi) * F /
                              (4 * cosTheta_i * cosTheta_o));
            return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);

        } else {
            // Sample transmission at non-delta dielectric interface
            Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
            Vector3f wi;
            bool tir = !Refract(wo, (Normal3f)wh, etap, &wi);
            CHECK_RARE(1e-5f, tir);
            if (SameHemisphere(wo, wi))
                return {};
            if (tir || wi.z == 0)
                return {};

            // Evaluate BSDF
            wh = FaceForward(wh, Normal3f(0, 0, 1));

            Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
            Float factor = (mode == TransportMode::Radiance) ? Sqr(1 / etap) : 1;

            SampledSpectrum f(
                (1 - F) * factor *
                std::abs(mfDistrib.D(wh) * mfDistrib.G(wo, wi) * AbsDot(wi, wh) *
                         AbsDot(wo, wh) /
                         (AbsCosTheta(wi) * AbsCosTheta(wo) * Sqr(sqrtDenom))));

            Float dwh_dwi =
                /*Sqr(etap) * */ AbsDot(wi, wh) / Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
            Float pdf = mfDistrib.PDF(wo, wh) * dwh_dwi * pt / (pr + pt);
            CHECK(!IsNaN(pdf));

            return BSDFSample(f, wi, pdf, BxDFFlags::GlossyTransmission, etap);
        }
    }
}

Float DielectricInterfaceBxDF::PDF(Vector3f wo, Vector3f wi, TransportMode mode,
                                   BxDFReflTransFlags sampleFlags) const {
    if (mfDistrib.EffectivelySmooth())
        return 0;
    // Return PDF for non-delta dielectric interface
    if (SameHemisphere(wo, wi)) {
        // Return PDF for non-delta dielectric reflection
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return 0;
        // Compute half-angle vector _wh_ for dielectric reflection PDF
        Vector3f wh = wo + wi;
        CHECK_RARE(1e-5f, LengthSquared(wh) == 0);
        CHECK_RARE(1e-5f, Dot(wo, wh) < 0);
        if (LengthSquared(wh) == 0 || Dot(wo, wh) <= 0)
            return 0;
        wh = Normalize(wh);

        // Compute Fresnel factor and probabilities for dielectric reflection PDF
        Float F = FrDielectric(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))), eta);
        CHECK_RARE(1e-5f, F == 0);
        Float pr = F, pt = 1 - F;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;

        return mfDistrib.PDF(wo, wh) / (4 * Dot(wo, wh)) * pr / (pr + pt);

    } else {
        // Return PDF for non-delta dielectric transmission
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            return 0;
        // Compute $\wh$ for dielectric transmission PDF
        Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
        Vector3f wh = wo + wi * etap;
        CHECK_RARE(1e-5f, LengthSquared(wh) == 0);
        if (LengthSquared(wh) == 0)
            return 0;
        wh = Normalize(wh);
        if (Dot(wi, wh) * Dot(wo, wh) > 0)
            return 0.;

        // Compute Fresnel factor and probabilities for dielectric transmission PDF
        Float F = FrDielectric(Dot(wo, FaceForward(wh, Normal3f(0, 0, 1))), eta);
        Float pr = F, pt = 1 - F;
        if (pt == 0)
            return 0;
        CHECK_RARE(1e-5f, (1 - F) == 0);
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;

        Float dwh_dwi = AbsDot(wi, wh) / Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
        return mfDistrib.PDF(wo, wh) * dwh_dwi * pt / (pr + pt);
    }
}

std::string DielectricInterfaceBxDF::ToString() const {
    return StringPrintf("[ DielectricInterfaceBxDF eta: %f mfDistrib: %s tint: %s ]", eta,
                        mfDistrib.ToString(), tint);
}

std::string ThinDielectricBxDF::ToString() const {
    return StringPrintf("[ ThinDielectricBxDF eta: %f ]", eta);
}

std::string ConductorBxDF::ToString() const {
    return StringPrintf("[ ConductorBxDF mfDistrib: %s eta: %s k: %s ]", mfDistrib, eta,
                        k);
}

// HairBxDF Method Definitions
HairBxDF::HairBxDF(Float h, Float eta, const SampledSpectrum &sigma_a, Float beta_m,
                   Float beta_n, Float alpha)
    : h(h),
      gamma_o(SafeASin(h)),
      eta(eta),
      sigma_a(sigma_a),
      beta_m(beta_m),
      beta_n(beta_n) {
    CHECK(h >= -1 && h <= 1);
    CHECK(beta_m >= 0 && beta_m <= 1);
    CHECK(beta_n >= 0 && beta_n <= 1);
    // Compute longitudinal variance from $\beta_m$
    static_assert(pMax >= 3,
                  "Longitudinal variance code must be updated to handle low pMax");
    v[0] = Sqr(0.726f * beta_m + 0.812f * Sqr(beta_m) + 3.7f * Pow<20>(beta_m));
    v[1] = .25 * v[0];
    v[2] = 4 * v[0];
    for (int p = 3; p <= pMax; ++p)
        // TODO: is there anything better here?
        v[p] = v[2];

    // Compute azimuthal logistic scale factor from $\beta_n$
    static const Float SqrtPiOver8 = 0.626657069f;
    s = SqrtPiOver8 * (0.265f * beta_n + 1.194f * Sqr(beta_n) + 5.372f * Pow<22>(beta_n));
    CHECK(!IsNaN(s));

    // Compute $\alpha$ terms for hair scales
    sin2kAlpha[0] = std::sin(Radians(alpha));
    cos2kAlpha[0] = SafeSqrt(1 - Sqr(sin2kAlpha[0]));
    for (int i = 1; i < 3; ++i) {
        sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
        cos2kAlpha[i] = Sqr(cos2kAlpha[i - 1]) - Sqr(sin2kAlpha[i - 1]);
    }
}

SampledSpectrum HairBxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    // Compute hair coordinate system terms related to _wo_
    Float sinTheta_o = wo.x;
    Float cosTheta_o = SafeSqrt(1 - Sqr(sinTheta_o));
    Float phi_o = std::atan2(wo.z, wo.y);

    // Compute hair coordinate system terms related to _wi_
    Float sinTheta_i = wi.x;
    Float cosTheta_i = SafeSqrt(1 - Sqr(sinTheta_i));
    Float phi_i = std::atan2(wi.z, wi.y);

    // Compute $\cos \thetat$ for refracted ray
    Float sinTheta_t = sinTheta_o / eta;
    Float cosTheta_t = SafeSqrt(1 - Sqr(sinTheta_t));

    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float cosGamma_t = SafeSqrt(1 - Sqr(sinGamma_t));
    Float gamma_t = SafeASin(sinGamma_t);

    // Compute the transmittance _T_ of a single path through the cylinder
    SampledSpectrum T = Exp(-sigma_a * (2 * cosGamma_t / cosTheta_t));

    // Evaluate hair BSDF
    Float phi = phi_i - phi_o;
    pstd::array<SampledSpectrum, pMax + 1> ap = Ap(cosTheta_o, eta, h, T);
    SampledSpectrum fsum(0.);
    for (int p = 0; p < pMax; ++p) {
        // Compute $\sin \thetai$ and $\cos \thetai$ terms accounting for scales
        Float sinThetap_o, cosThetap_o;
        if (p == 0) {
            sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
            cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
        }
        // Handle remainder of $p$ values for hair scale tilt
        else if (p == 1) {
            sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
            cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
            cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
        } else {
            sinThetap_o = sinTheta_o;
            cosThetap_o = cosTheta_o;
        }

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cosThetap_o = std::abs(cosThetap_o);

        fsum += Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]) * ap[p] *
                Np(phi, p, s, gamma_o, gamma_t);
    }
    // Compute contribution of remaining terms after _pMax_
    fsum += Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]) * ap[pMax] /
            (2.f * Pi);

    if (AbsCosTheta(wi) > 0)
        fsum /= AbsCosTheta(wi);
    CHECK(!IsInf(fsum.Average()) && !IsNaN(fsum.Average()));
    return fsum;
}

pstd::array<Float, HairBxDF::pMax + 1> HairBxDF::ComputeApPDF(Float cosTheta_o) const {
    // Compute array of $A_p$ values for _cosThetaO_
    Float sinTheta_o = SafeSqrt(1 - cosTheta_o * cosTheta_o);
    // Compute $\cos \thetat$ for refracted ray
    Float sinTheta_t = sinTheta_o / eta;
    Float cosTheta_t = SafeSqrt(1 - Sqr(sinTheta_t));

    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float cosGamma_t = SafeSqrt(1 - Sqr(sinGamma_t));
    Float gamma_t = SafeASin(sinGamma_t);

    // Compute the transmittance _T_ of a single path through the cylinder
    SampledSpectrum T = Exp(-sigma_a * (2 * cosGamma_t / cosTheta_t));

    pstd::array<SampledSpectrum, pMax + 1> ap = Ap(cosTheta_o, eta, h, T);

    // Compute $A_p$ PDF from individual $A_p$ terms
    pstd::array<Float, pMax + 1> apPDF;
    Float sumY = 0;
    for (const SampledSpectrum &as : ap)
        sumY += as.Average();
    for (int i = 0; i <= pMax; ++i)
        apPDF[i] = ap[i].Average() / sumY;

    return apPDF;
}

pstd::optional<BSDFSample> HairBxDF::Sample_f(Vector3f wo, Float uc, Point2f u,
                                              TransportMode mode,
                                              BxDFReflTransFlags sampleFlags) const {
    // Compute hair coordinate system terms related to _wo_
    Float sinTheta_o = wo.x;
    Float cosTheta_o = SafeSqrt(1 - Sqr(sinTheta_o));
    Float phi_o = std::atan2(wo.z, wo.y);

    // Determine which term $p$ to sample for hair scattering
    pstd::array<Float, pMax + 1> apPDF = ComputeApPDF(cosTheta_o);
    int p = SampleDiscrete(apPDF, uc, nullptr, &uc);

    // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
    Float sinThetap_o, cosThetap_o;
    if (p == 0) {
        sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
        cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
    } else if (p == 1) {
        sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
        cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
    } else if (p == 2) {
        sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
        cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
    } else {
        sinThetap_o = sinTheta_o;
        cosThetap_o = cosTheta_o;
    }

    // Sample $M_p$ to compute $\thetai$
    Float cosTheta = 1 + v[p] * std::log(std::max<Float>(u[0], 1e-5) +
                                         (1 - u[0]) * FastExp(-2 / v[p]));
    Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    Float cosPhi = std::cos(2 * Pi * u[1]);
    Float sinTheta_i = -cosTheta * sinThetap_o + sinTheta * cosPhi * cosThetap_o;
    Float cosTheta_i = SafeSqrt(1 - Sqr(sinTheta_i));

    // Sample $N_p$ to compute $\Delta\phi$
    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float cosGamma_t = SafeSqrt(1 - Sqr(sinGamma_t));
    Float gamma_t = SafeASin(sinGamma_t);

    Float dphi;
    if (p < pMax)
        dphi = Phi(p, gamma_o, gamma_t) + SampleTrimmedLogistic(uc, s, -Pi, Pi);
    else
        dphi = 2 * Pi * uc;

    // Compute _wi_ from sampled hair scattering angles
    Float phi_i = phi_o + dphi;
    Vector3f wi(sinTheta_i, cosTheta_i * std::cos(phi_i), cosTheta_i * std::sin(phi_i));

    // Compute PDF for sampled hair scattering direction _wi_
    Float pdf = 0;
    for (int p = 0; p < pMax; ++p) {
        // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
        Float sinThetap_o, cosThetap_o;
        if (p == 0) {
            sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
            cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
        } else if (p == 1) {
            sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
            cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
            cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
        } else {
            sinThetap_o = sinTheta_o;
            cosThetap_o = cosTheta_o;
        }

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cosThetap_o = std::abs(cosThetap_o);

        pdf += Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]) * apPDF[p] *
               Np(dphi, p, s, gamma_o, gamma_t);
    }
    pdf += Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]) * apPDF[pMax] *
           (1 / (2 * Pi));
    // if (std::abs(wi->x) < .9999) CHECK_NEAR(*pdf, PDF(wo, *wi), .01);

    return BSDFSample(f(wo, wi, mode), wi, pdf, Flags());
}

Float HairBxDF::PDF(Vector3f wo, Vector3f wi, TransportMode mode,
                    BxDFReflTransFlags sampleFlags) const {
    // TODO? flags...

    // Compute hair coordinate system terms related to _wo_
    Float sinTheta_o = wo.x;
    Float cosTheta_o = SafeSqrt(1 - Sqr(sinTheta_o));
    Float phi_o = std::atan2(wo.z, wo.y);

    // Compute hair coordinate system terms related to _wi_
    Float sinTheta_i = wi.x;
    Float cosTheta_i = SafeSqrt(1 - Sqr(sinTheta_i));
    Float phi_i = std::atan2(wi.z, wi.y);

    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float gamma_t = SafeASin(sinGamma_t);

    // Compute PDF for $A_p$ terms
    pstd::array<Float, pMax + 1> apPDF = ComputeApPDF(cosTheta_o);

    // Compute PDF sum for hair scattering events
    Float phi = phi_i - phi_o;
    Float pdf = 0;
    for (int p = 0; p < pMax; ++p) {
        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        Float sinThetap_o, cosThetap_o;
        if (p == 0) {
            sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
            cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
        }

        // Handle remainder of $p$ values for hair scale tilt
        else if (p == 1) {
            sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
            cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
            cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
        } else {
            sinThetap_o = sinTheta_o;
            cosThetap_o = cosTheta_o;
        }

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cosThetap_o = std::abs(cosThetap_o);
        pdf += Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]) * apPDF[p] *
               Np(phi, p, s, gamma_o, gamma_t);
    }
    pdf += Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]) * apPDF[pMax] *
           (1 / (2 * Pi));
    return pdf;
}

RGBUnboundedSpectrum HairBxDF::SigmaAFromConcentration(Float ce, Float cp) {
    RGB eumelaninSigmaA(0.419f, 0.697f, 1.37f);
    RGB pheomelaninSigmaA(0.187f, 0.4f, 1.05f);
    RGB sigma_a = ce * eumelaninSigmaA + cp * pheomelaninSigmaA;
#ifdef PBRT_IS_GPU_CODE
    return RGBUnboundedSpectrum(*RGBColorSpace_sRGB, sigma_a);
#else
    return RGBUnboundedSpectrum(*RGBColorSpace::sRGB, sigma_a);
#endif
}

SampledSpectrum HairBxDF::SigmaAFromReflectance(const SampledSpectrum &c, Float beta_n,
                                                const SampledWavelengths &lambda) {
    SampledSpectrum sigma_a;
    for (int i = 0; i < NSpectrumSamples; ++i)
        sigma_a[i] =
            Sqr(std::log(c[i]) / (5.969f - 0.215f * beta_n + 2.532f * Sqr(beta_n) -
                                  10.73f * Pow<3>(beta_n) + 5.574f * Pow<4>(beta_n) +
                                  0.245f * Pow<5>(beta_n)));
    return sigma_a;
}

std::string HairBxDF::ToString() const {
    return StringPrintf("[ HairBxDF h: %f gamma_o: %f eta: %f beta_m: %f beta_n: %f "
                        "v[0]: %f s: %f sigma_a: %s ]",
                        h, gamma_o, eta, beta_m, beta_n, v[0], s, sigma_a);
}

std::string LayeredBxDFConfig::ToString() const {
    return StringPrintf("[ LayeredBxDFConfig maxDepth: %d nSamples: %d twoSided: %d",
                        maxDepth, nSamples, twoSided);
}

// *****************************************************************************
// Tensor file I/O
// *****************************************************************************

class Tensor {
  public:
    // Data type of the tensor's fields
    enum Type {
        /* Invalid/unspecified */
        Invalid = 0,

        /* Signed and unsigned integer values */
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        UInt64,
        Int64,

        /* Floating point values */
        Float16,
        Float32,
        Float64,
    };

    struct Field {
        // Data type of the tensor's fields
        Type dtype;

        // Offset in the file
        size_t offset;

        /// Specifies both rank and size along each dimension
        std::vector<size_t> shape;

        /// Pointer to the start of the tensor
        std::unique_ptr<uint8_t[]> data;
    };

    /// Load a tensor file into memory
    Tensor(const std::string &filename);

    /// Does the file contain a field of the specified name?
    bool has_field(const std::string &name) const;

    /// Return a data structure with information about the specified field
    const Field &field(const std::string &name) const;

    /// Return a human-readable summary
    std::string ToString() const;

    /// Return the total size of the tensor's data
    size_t size() const { return m_size; }

    std::string filename() const { return m_filename; }

  private:
    std::unordered_map<std::string, Field> m_fields;
    std::string m_filename;
    size_t m_size;
};

static std::ostream &operator<<(std::ostream &os, Tensor::Type value) {
    switch (value) {
    case Tensor::Invalid:
        os << "invalid";
        break;
    case Tensor::UInt8:
        os << "uint8_t";
        break;
    case Tensor::Int8:
        os << "int8_t";
        break;
    case Tensor::UInt16:
        os << "uint16_t";
        break;
    case Tensor::Int16:
        os << "int16_t";
        break;
    case Tensor::UInt32:
        os << "uint32_t";
        break;
    case Tensor::Int32:
        os << "int8_t";
        break;
    case Tensor::UInt64:
        os << "uint64_t";
        break;
    case Tensor::Int64:
        os << "int64_t";
        break;
    case Tensor::Float16:
        os << "float16_t";
        break;
    case Tensor::Float32:
        os << "float32_t";
        break;
    case Tensor::Float64:
        os << "float64_t";
        break;
    default:
        os << "unkown";
        break;
    }
    return os;
}

static size_t type_size(Tensor::Type value) {
    switch (value) {
    case Tensor::Invalid:
        return 0;
        break;
    case Tensor::UInt8:
        return 1;
        break;
    case Tensor::Int8:
        return 1;
        break;
    case Tensor::UInt16:
        return 2;
        break;
    case Tensor::Int16:
        return 2;
        break;
    case Tensor::UInt32:
        return 4;
        break;
    case Tensor::Int32:
        return 4;
        break;
    case Tensor::UInt64:
        return 8;
        break;
    case Tensor::Int64:
        return 8;
        break;
    case Tensor::Float16:
        return 2;
        break;
    case Tensor::Float32:
        return 4;
        break;
    case Tensor::Float64:
        return 8;
        break;
    default:
        return 0;
        break;
    }
}

Tensor::Tensor(const std::string &filename) : m_filename(filename) {
    // Helpful macros to limit error-handling code duplication
#ifdef ASSERT
#undef ASSERT
#endif  // ASSERT

#define ASSERT(cond, msg)                            \
    do {                                             \
        if (!(cond)) {                               \
            fclose(file);                            \
            ErrorExit("%s: Tensor: " msg, filename); \
        }                                            \
    } while (0)

#define SAFE_READ(vars, size, count) \
    ASSERT(fread(vars, size, count, file) == (count), "Unable to read " #vars ".")

    FILE *file = fopen(filename.c_str(), "rb");
    if (file == NULL)
        ErrorExit("%s: unable to open file", filename);

    ASSERT(!fseek(file, 0, SEEK_END), "Unable to seek to end of file.");

    long size = ftell(file);
    ASSERT(size != -1, "Unable to tell file cursor position.");
    m_size = static_cast<size_t>(size);
    rewind(file);

    ASSERT(m_size >= 12 + 2 + 4, "Invalid tensor file: too small, truncated?");

    uint8_t header[12], version[2];
    uint32_t n_fields;
    SAFE_READ(header, sizeof(*header), 12);
    SAFE_READ(version, sizeof(*version), 2);
    SAFE_READ(&n_fields, sizeof(n_fields), 1);

    ASSERT(memcmp(header, "tensor_file", 12) == 0,
           "Invalid tensor file: invalid header.");
    ASSERT(version[0] == 1 && version[1] == 0,
           "Invalid tensor file: unknown file version.");

    for (uint32_t i = 0; i < n_fields; ++i) {
        uint8_t dtype;
        uint16_t name_length, ndim;
        uint64_t offset;

        SAFE_READ(&name_length, sizeof(name_length), 1);
        std::string name(name_length, '\0');
        SAFE_READ((char *)name.data(), 1, name_length);
        SAFE_READ(&ndim, sizeof(ndim), 1);
        SAFE_READ(&dtype, sizeof(dtype), 1);
        SAFE_READ(&offset, sizeof(offset), 1);
        ASSERT(dtype != Invalid && dtype <= Float64,
               "Invalid tensor file: unknown type.");

        std::vector<size_t> shape(ndim);
        size_t total_size = type_size((Type)dtype);  // no need to check here, line 43
                                                     // already removes invalid types
        for (size_t j = 0; j < (size_t)ndim; ++j) {
            uint64_t size_value;
            SAFE_READ(&size_value, sizeof(size_value), 1);
            shape[j] = (size_t)size_value;
            total_size *= shape[j];
        }

        auto data = std::unique_ptr<uint8_t[]>(new uint8_t[total_size]);

        long cur_pos = ftell(file);
        ASSERT(cur_pos != -1, "Unable to tell current cursor position.");
        ASSERT(fseek(file, offset, SEEK_SET) != -1, "Unable to seek to tensor offset.");
        SAFE_READ(data.get(), 1, total_size);
        ASSERT(fseek(file, cur_pos, SEEK_SET) != -1,
               "Unable to seek back to current position");

        m_fields[name] =
            Field{(Type)dtype, static_cast<size_t>(offset), shape, std::move(data)};
    }

    fclose(file);

#undef SAFE_READ
#undef ASSERT
}

/// Does the file contain a field of the specified name?
bool Tensor::has_field(const std::string &name) const {
    return m_fields.find(name) != m_fields.end();
}

/// Return a data structure with information about the specified field
const Tensor::Field &Tensor::field(const std::string &name) const {
    auto it = m_fields.find(name);
    CHECK(it != m_fields.end());
    return it->second;
}

/// Return a human-readable summary
std::string Tensor::ToString() const {
    std::ostringstream oss;
    oss << "Tensor[" << std::endl
        << "  filename = \"" << m_filename << "\"," << std::endl
        << "  size = " << size() << "," << std::endl
        << "  fields = {" << std::endl;

    size_t ctr = 0;
    for (const auto &it : m_fields) {
        oss << "    \"" << it.first << "\""
            << " => [" << std::endl
            << "      dtype = " << it.second.dtype << "," << std::endl
            << "      offset = " << it.second.offset << "," << std::endl
            << "      shape = [";
        const auto &shape = it.second.shape;
        for (size_t j = 0; j < shape.size(); ++j) {
            oss << shape[j];
            if (j + 1 < shape.size())
                oss << ", ";
        }

        oss << "]" << std::endl;

        oss << "    ]";
        if (++ctr < m_fields.size())
            oss << ",";
        oss << std::endl;
    }

    oss << "  }" << std::endl << "]";

    return oss.str();
}

// MeasuredBRDF Definition
class MeasuredBRDF {
  public:
    MeasuredBRDF(Allocator alloc)
        : ndf(alloc),
          sigma(alloc),
          vndf(alloc),
          luminance(alloc),
          spectra(alloc),
          wavelengths(alloc) {}

    static MeasuredBRDF *Create(const std::string &filename, Allocator alloc);

    std::string ToString() const {
        return StringPrintf("[ MeasuredBRDF filename: %s ]", filename);
    }

    using Warp2D0 = PiecewiseLinear2D<0>;
    using Warp2D2 = PiecewiseLinear2D<2>;
    using Warp2D3 = PiecewiseLinear2D<3>;

    Warp2D0 ndf;
    Warp2D0 sigma;
    Warp2D2 vndf;
    Warp2D2 luminance;
    Warp2D3 spectra;
    pstd::vector<float> wavelengths;
    bool isotropic;
    bool jacobian;
    std::string filename;
};

STAT_MEMORY_COUNTER("Memory/Measured BRDF data", measuredBRDFBytes);

MeasuredBRDF *MeasuredBRDF::Create(const std::string &filename, Allocator alloc) {
    Tensor tf = Tensor(filename);
    auto &theta_i = tf.field("theta_i");
    auto &phi_i = tf.field("phi_i");
    auto &ndf = tf.field("ndf");
    auto &sigma = tf.field("sigma");
    auto &vndf = tf.field("vndf");
    auto &spectra = tf.field("spectra");
    auto &luminance = tf.field("luminance");
    auto &wavelengths = tf.field("wavelengths");
    auto &description = tf.field("description");
    auto &jacobian = tf.field("jacobian");

    if (!(description.shape.size() == 1 && description.dtype == Tensor::UInt8 &&

          theta_i.shape.size() == 1 && theta_i.dtype == Tensor::Float32 &&

          phi_i.shape.size() == 1 && phi_i.dtype == Tensor::Float32 &&

          wavelengths.shape.size() == 1 && wavelengths.dtype == Tensor::Float32 &&

          ndf.shape.size() == 2 && ndf.dtype == Tensor::Float32 &&

          sigma.shape.size() == 2 && sigma.dtype == Tensor::Float32 &&

          vndf.shape.size() == 4 && vndf.dtype == Tensor::Float32 &&
          vndf.shape[0] == phi_i.shape[0] && vndf.shape[1] == theta_i.shape[0] &&

          luminance.shape.size() == 4 && luminance.dtype == Tensor::Float32 &&
          luminance.shape[0] == phi_i.shape[0] &&
          luminance.shape[1] == theta_i.shape[0] &&
          luminance.shape[2] == luminance.shape[3] &&

          spectra.dtype == Tensor::Float32 && spectra.shape.size() == 5 &&
          spectra.shape[0] == phi_i.shape[0] && spectra.shape[1] == theta_i.shape[0] &&
          spectra.shape[2] == wavelengths.shape[0] &&
          spectra.shape[3] == spectra.shape[4] &&

          luminance.shape[2] == spectra.shape[3] &&
          luminance.shape[3] == spectra.shape[4] &&

          jacobian.shape.size() == 1 && jacobian.shape[0] == 1 &&
          jacobian.dtype == Tensor::UInt8)) {
        Error("%s: invalid BRDF file structure: %s", filename, tf);
        return nullptr;
    }

    MeasuredBRDF *brdf = alloc.new_object<MeasuredBRDF>(alloc);
    brdf->filename = filename;
    brdf->isotropic = phi_i.shape[0] <= 2;
    brdf->jacobian = ((uint8_t *)jacobian.data.get())[0];

    if (!brdf->isotropic) {
        float *phi_i_data = (float *)phi_i.data.get();
        int reduction =
            (int)std::rint((2 * Pi) / (phi_i_data[phi_i.shape[0] - 1] - phi_i_data[0]));
        if (reduction != 1)
            ErrorExit("%s: reduction %d (!= 1) not supported", filename, reduction);
    }

    /* Construct NDF interpolant data structure */
    brdf->ndf = Warp2D0(alloc, (float *)ndf.data.get(), ndf.shape[1], ndf.shape[0], {},
                        {}, false, false);

    /* Construct projected surface area interpolant data structure */
    brdf->sigma = Warp2D0(alloc, (float *)sigma.data.get(), sigma.shape[1],
                          sigma.shape[0], {}, {}, false, false);

    /* Construct VNDF warp data structure */
    brdf->vndf =
        Warp2D2(alloc, (float *)vndf.data.get(), vndf.shape[3], vndf.shape[2],
                {{(int)phi_i.shape[0], (int)theta_i.shape[0]}},
                {{(const float *)phi_i.data.get(), (const float *)theta_i.data.get()}});

    /* Construct Luminance warp data structure */
    brdf->luminance =
        Warp2D2(alloc, (float *)luminance.data.get(), luminance.shape[3],
                luminance.shape[2], {{(int)phi_i.shape[0], (int)theta_i.shape[0]}},
                {{(const float *)phi_i.data.get(), (const float *)theta_i.data.get()}});

    /* Copy wavelength information */
    size_t size = wavelengths.shape[0];
    brdf->wavelengths.resize(size);
    for (size_t i = 0; i < size; ++i)
        brdf->wavelengths[i] = ((const float *)wavelengths.data.get())[i];

    /* Construct spectral interpolant */
    brdf->spectra =
        Warp2D3(alloc, (float *)spectra.data.get(), spectra.shape[4], spectra.shape[3],
                {{(int)phi_i.shape[0], (int)theta_i.shape[0], (int)wavelengths.shape[0]}},
                {{(const float *)phi_i.data.get(), (const float *)theta_i.data.get(),
                  (const float *)wavelengths.data.get()}},
                false, false);

    measuredBRDFBytes += sizeof(MeasuredBRDF) + 4 * brdf->wavelengths.size() +
                         brdf->ndf.BytesUsed() + brdf->sigma.BytesUsed() +
                         brdf->vndf.BytesUsed() + brdf->luminance.BytesUsed() +
                         brdf->spectra.BytesUsed();

    return brdf;
}

MeasuredBRDF *MeasuredBxDF::BRDFDataFromFile(const std::string &filename,
                                             Allocator alloc) {
    static std::map<std::string, MeasuredBRDF *> loadedData;
    if (loadedData.find(filename) == loadedData.end())
        loadedData[filename] = MeasuredBRDF::Create(filename, alloc);
    return loadedData[filename];
}

// MeasuredBxDF Method Definitions
SampledSpectrum MeasuredBxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    if (!SameHemisphere(wo, wi))
        return SampledSpectrum(0.);
    if (wo.z < 0) {
        wo = -wo;
        wi = -wi;
    }

    Vector3f wm = wi + wo;
    if (LengthSquared(wm) == 0)
        return SampledSpectrum(0);
    wm = Normalize(wm);

    /* Cartesian -> spherical coordinates */
    Float theta_i = SphericalTheta(wi), phi_i = std::atan2(wi.y, wi.x);
    Float theta_m = SphericalTheta(wm), phi_m = std::atan2(wm.y, wm.x);

    /* Spherical coordinates -> unit coordinate system */
    Vector2f u_wi(theta2u(theta_i), phi2u(phi_i));
    Vector2f u_wm(theta2u(theta_m), phi2u(brdf->isotropic ? (phi_m - phi_i) : phi_m));
    u_wm.y = u_wm.y - std::floor(u_wm.y);

    Float params[2] = {phi_i, theta_i};
    auto ui = brdf->vndf.Invert(u_wm, params);
    Vector2f sample = ui.p;
    Float vndfPDF = ui.pdf;

    SampledSpectrum fr(0);
    for (int i = 0; i < pbrt::NSpectrumSamples; ++i) {
        Float params_fr[3] = {phi_i, theta_i, lambda[i]};
        fr[i] = brdf->spectra.Evaluate(sample, params_fr);
        CHECK_RARE(1e-5f, fr[i] < 0);
        fr[i] = std::max<Float>(0, fr[i]);
    }

    return fr * brdf->ndf.Evaluate(u_wm, params) /
           (4 * brdf->sigma.Evaluate(u_wi, params) * AbsCosTheta(wi));
}

pstd::optional<BSDFSample> MeasuredBxDF::Sample_f(Vector3f wo, Float uc, Point2f u,
                                                  TransportMode mode,
                                                  BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection))
        return {};

    bool flipWi = false;
    if (wo.z <= 0) {
        wo = -wo;
        flipWi = true;
    }

    Float theta_i = SphericalTheta(wo), phi_i = std::atan2(wo.y, wo.x);

    Vector2f sample = Vector2f(u.y, u.x);
    Float params[2] = {phi_i, theta_i};
    auto s = brdf->luminance.Sample(sample, params);
    sample = s.p;
    Float lumPDF = s.pdf;

    s = brdf->vndf.Sample(sample, params);
    Vector2f u_wm = s.p;
    Float ndfPDF = s.pdf;

    Float phi_m = u2phi(u_wm.y), theta_m = u2theta(u_wm.x);
    if (brdf->isotropic)
        phi_m += phi_i;

    /* Spherical -> Cartesian coordinates */
    Float sinTheta_m = std::sin(theta_m), cosTheta_m = std::cos(theta_m);
    Vector3f wm = SphericalDirection(sinTheta_m, cosTheta_m, phi_m);

    Vector3f wi = Reflect(wo, wm);
    if (wi.z <= 0)
        return {};

    SampledSpectrum fr(0);
    for (int i = 0; i < pbrt::NSpectrumSamples; ++i) {
        Float params_fr[3] = {phi_i, theta_i, lambda[i]};
        fr[i] = brdf->spectra.Evaluate(sample, params_fr);
        CHECK_RARE(1e-5f, fr[i] < 0);
        fr[i] = std::max<Float>(0, fr[i]);
    }

    Vector2f u_wo = Vector2f(theta2u(theta_i), phi2u(phi_i));
    fr *= brdf->ndf.Evaluate(u_wm, params) /
          (4 * brdf->sigma.Evaluate(u_wo, params) * AbsCosTheta(wi));

    Float jacobian =
        4 * Dot(wo, wm) * std::max<Float>(2 * Sqr(Pi) * u_wm.x * sinTheta_m, 1e-6f);
    Float pdf = ndfPDF * lumPDF / jacobian;

    if (flipWi)
        wi = -wi;
    return BSDFSample(fr, wi, pdf, BxDFFlags::GlossyReflection);
}

Float MeasuredBxDF::PDF(Vector3f wo, Vector3f wi, TransportMode mode,
                        BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection))
        return 0;
    if (!SameHemisphere(wo, wi))
        return 0;
    if (wo.z < 0) {
        wo = -wo;
        wi = -wi;
    }

    Vector3f wm = wi + wo;
    if (LengthSquared(wm) == 0)
        return 0;
    wm = Normalize(wm);

    /* Cartesian -> spherical coordinates */
    Float theta_i = SphericalTheta(wi), phi_i = std::atan2(wi.y, wi.x);
    Float theta_m = SphericalTheta(wm), phi_m = std::atan2(wm.y, wm.x);

    /* Spherical coordinates -> unit coordinate system */
    Vector2f u_wm(theta2u(theta_m), phi2u(brdf->isotropic ? (phi_m - phi_i) : phi_m));
    u_wm.y = u_wm.y - std::floor(u_wm.y);

    Float params[2] = {phi_i, theta_i};
    auto ui = brdf->vndf.Invert(u_wm, params);
    Vector2f sample = ui.p;
    Float vndfPDF = ui.pdf;

    Float pdf = brdf->luminance.Evaluate(sample, params);
    Float sinTheta_m = std::sqrt(Sqr(wm.x) + Sqr(wm.y));
    Float jacobian =
        4.f * Dot(wi, wm) * std::max<Float>(2 * Sqr(Pi) * u_wm.x * sinTheta_m, 1e-6f);
    return vndfPDF * pdf / jacobian;
}

std::string MeasuredBxDF::ToString() const {
    return StringPrintf("[ MeasuredBxDF brdf: %s ]", *brdf);
}

std::string NormalizedFresnelBxDF::ToString() const {
    return StringPrintf("[ NormalizedFresnelBxDF eta: %f ]", eta);
}

// BxDF Method Definitions
SampledSpectrum BxDF::rho(Vector3f wo, pstd::span<const Float> uc,
                          pstd::span<const Point2f> u2) const {
    if (wo.z == 0)
        return {};
    SampledSpectrum r(0.);
    DCHECK_EQ(uc.size(), u2.size());
    for (size_t i = 0; i < uc.size(); ++i) {
        // Compute estimate of $\rho_\roman{hd}$
        pstd::optional<BSDFSample> bs = Sample_f(wo, uc[i], u2[i]);
        if (bs)
            r += bs->f * AbsCosTheta(bs->wi) / bs->pdf;
    }
    return r / uc.size();
}

SampledSpectrum BxDF::rho(pstd::span<const Point2f> u1, pstd::span<const Float> uc,
                          pstd::span<const Point2f> u2) const {
    DCHECK_EQ(uc.size(), u1.size());
    DCHECK_EQ(u1.size(), u2.size());
    SampledSpectrum r(0.f);
    for (size_t i = 0; i < uc.size(); ++i) {
        // Compute estimate of of $\rho_\roman{hh}$
        Vector3f wo = SampleUniformHemisphere(u1[i]);
        if (wo.z == 0)
            continue;
        Float pdfo = UniformHemispherePDF();
        pstd::optional<BSDFSample> bs = Sample_f(wo, uc[i], u2[i]);
        if (bs)
            r += bs->f * AbsCosTheta(bs->wi) * AbsCosTheta(wo) / (pdfo * bs->pdf);
    }
    return r / (Pi * uc.size());
}

std::string BxDF::ToString() const {
    auto toStr = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(toStr);
}

template class LayeredBxDF<DielectricInterfaceBxDF, IdealDiffuseBxDF>;
template class LayeredBxDF<DielectricInterfaceBxDF, ConductorBxDF>;

}  // namespace pbrt
