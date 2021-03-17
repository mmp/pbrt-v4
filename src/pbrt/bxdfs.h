// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BXDFS_H
#define PBRT_BXDFS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/interaction.h>
#include <pbrt/media.h>
#include <pbrt/options.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace pbrt {

// IdealDiffuseBxDF Definition
class IdealDiffuseBxDF {
  public:
    // IdealDiffuseBxDF Public Methods
    IdealDiffuseBxDF() = default;
    PBRT_CPU_GPU
    IdealDiffuseBxDF(const SampledSpectrum &R) : R(R) {}

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        if (!SameHemisphere(wo, wi))
            return SampledSpectrum(0.f);
        return R * InvPi;
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        Vector3f wo, Float uc, Point2f u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};
        // Sample cosine-weighted hemisphere to compute _wi_ and _pdf_
        Vector3f wi = SampleCosineHemisphere(u);
        if (wo.z < 0)
            wi.z *= -1;
        Float pdf = CosineHemispherePDF(AbsCosTheta(wi));

        return BSDFSample(R * InvPi, wi, pdf, BxDFFlags::DiffuseReflection);
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection) || !SameHemisphere(wo, wi))
            return 0;
        return CosineHemispherePDF(AbsCosTheta(wi));
    }

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "IdealDiffuseBxDF"; }

    std::string ToString() const;

    PBRT_CPU_GPU
    void Regularize() {}

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return R ? BxDFFlags::DiffuseReflection : BxDFFlags::Unset;
    }

  private:
    SampledSpectrum R;
};

// DiffuseBxDF Definition
class DiffuseBxDF {
  public:
    // DiffuseBxDF Public Methods
    DiffuseBxDF() = default;
    PBRT_CPU_GPU
    DiffuseBxDF(SampledSpectrum R, SampledSpectrum T, Float sigma) : R(R), T(T) {
        Float sigma2 = Sqr(Radians(sigma));
        A = 1 - sigma2 / (2 * (sigma2 + 0.33f));
        B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        // Return Lambertian BRDF for zero-roughness Oren--Nayar BRDF
        if (B == 0)
            return SameHemisphere(wo, wi) ? (R * InvPi) : (T * InvPi);

        if ((SameHemisphere(wo, wi) && !R) || (!SameHemisphere(wo, wi) && !T))
            return SampledSpectrum(0.);
        // Evaluate Oren--Nayar BRDF for given directions
        Float sinTheta_i = SinTheta(wi), sinTheta_o = SinTheta(wo);
        Float maxCos = std::max<Float>(0, CosDPhi(wi, wo));
        // Compute $\sin \alpha$ and $\tan \beta$ terms of Oren--Nayar model
        Float sinAlpha, tanBeta;
        if (AbsCosTheta(wi) > AbsCosTheta(wo)) {
            sinAlpha = sinTheta_o;
            tanBeta = sinTheta_i / AbsCosTheta(wi);
        } else {
            sinAlpha = sinTheta_i;
            tanBeta = sinTheta_o / AbsCosTheta(wo);
        }

        // Return final Oren--Nayar BSDF value
        if (SameHemisphere(wo, wi))
            return R * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
        else
            return T * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        Vector3f wo, Float uc, Point2f u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        // Compute reflection and transmission probabilities for diffuse BSDF
        Float pr = R.MaxComponentValue(), pt = T.MaxComponentValue();
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        // Randomly sample diffuse BSDF reflection or transmission
        if (uc < pr / (pr + pt)) {
            // Sample diffuse BSDF reflection
            Vector3f wi = SampleCosineHemisphere(u);
            if (wo.z < 0)
                wi.z *= -1;
            Float pdf = CosineHemispherePDF(AbsCosTheta(wi)) * pr / (pr + pt);
            return BSDFSample(f(wo, wi, mode), wi, pdf, BxDFFlags::DiffuseReflection);

        } else {
            // Sample diffuse BSDF transmission
            Vector3f wi = SampleCosineHemisphere(u);
            if (wo.z > 0)
                wi.z *= -1;
            Float pdf = CosineHemispherePDF(AbsCosTheta(wi)) * pt / (pr + pt);
            return BSDFSample(f(wo, wi, mode), wi, pdf, BxDFFlags::DiffuseTransmission);
        }
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        // Compute reflection and transmission probabilities for diffuse BSDF
        Float pr = R.MaxComponentValue(), pt = T.MaxComponentValue();
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        if (SameHemisphere(wo, wi))
            return pr / (pr + pt) * CosineHemispherePDF(AbsCosTheta(wi));
        else
            return pt / (pr + pt) * CosineHemispherePDF(AbsCosTheta(wi));
    }

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "DiffuseBxDF"; }

    std::string ToString() const;

    PBRT_CPU_GPU
    void Regularize() {}

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return ((R ? BxDFFlags::DiffuseReflection : BxDFFlags::Unset) |
                (T ? BxDFFlags::DiffuseTransmission : BxDFFlags::Unset));
    }

  private:
    // DiffuseBxDF Private Members
    SampledSpectrum R, T;
    Float A, B;
};

// DielectricInterfaceBxDF Definition
class DielectricInterfaceBxDF {
  public:
    // DielectricInterfaceBxDF Public Methods
    DielectricInterfaceBxDF() = default;
    PBRT_CPU_GPU
    DielectricInterfaceBxDF(Float eta, SampledSpectrum tint,
                            const TrowbridgeReitzDistribution &mfDistrib)
        : eta(eta == 1 ? 1.001 : eta), tint(tint), mfDistrib(mfDistrib) {}

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return BxDFFlags::Reflection | BxDFFlags::Transmission |
               (mfDistrib.EffectivelySmooth() ? BxDFFlags::Specular : BxDFFlags::Glossy);
    }

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        Vector3f wo, Float uc, Point2f u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "DielectricInterfaceBxDF"; }

    std::string ToString() const;

    PBRT_CPU_GPU
    void Regularize() { mfDistrib.Regularize(); }

  private:
    // DielectricInterfaceBxDF Private Members
    Float eta;
    SampledSpectrum tint;
    TrowbridgeReitzDistribution mfDistrib;
};

// ThinDielectricBxDF Definition
class ThinDielectricBxDF {
  public:
    // ThinDielectricBxDF Public Methods
    ThinDielectricBxDF() = default;
    PBRT_CPU_GPU
    ThinDielectricBxDF(Float eta) : eta(eta) {}

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        return SampledSpectrum(0);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(Vector3f wo, Float uc, Point2f u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const {
        Float R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
        // Compute _R_ and _T_ accounting for scattering between interfaces
        if (R < 1) {
            R += T * T * R / (1 - R * R);
            T = 1 - R;
        }

        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        Float pr = R, pt = T;
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        if (uc < pr / (pr + pt)) {
            // Sample perfect specular reflection at thin dielectric interface
            Vector3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum fr(R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);

        } else {
            // Sample perfect specular transmission at thin dielectric interface
            Vector3f wi = -wo;
            SampledSpectrum ft(T / AbsCosTheta(wi));
            return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission);
        }
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const {
        return 0;
    }

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "ThinDielectricBxDF"; }

    std::string ToString() const;

    PBRT_CPU_GPU
    void Regularize() { /* TODO */
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return (BxDFFlags::Reflection | BxDFFlags::Transmission | BxDFFlags::Specular);
    }

  private:
    Float eta;
};

// ConductorBxDF Definition
class ConductorBxDF {
  public:
    // ConductorBxDF Public Methods
    ConductorBxDF() = default;
    PBRT_CPU_GPU
    ConductorBxDF(const TrowbridgeReitzDistribution &mfDistrib,
                  const SampledSpectrum &eta, const SampledSpectrum &k)
        : mfDistrib(mfDistrib), eta(eta), k(k) {}

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return mfDistrib.EffectivelySmooth() ? BxDFFlags::SpecularReflection
                                             : BxDFFlags::GlossyReflection;
    }

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "ConductorBxDF"; }
    std::string ToString() const;

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        if (!SameHemisphere(wo, wi))
            return {};
        if (mfDistrib.EffectivelySmooth())
            return {};
        // Evaluate Torrance--Sparrow model for conductor BRDF
        // Compute cosines and $\wh$ for conductor BRDF
        Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        if (cosTheta_i == 0 || cosTheta_o == 0)
            return {};
        Vector3f wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0)
            return {};
        wh = Normalize(wh);

        // Evaluate Fresnel factor _F_ for conductor BRDF
        Float frCosTheta_i = AbsDot(wi, wh);
        SampledSpectrum F = FrConductor(frCosTheta_i, eta, k);

        return mfDistrib.D(wh) * mfDistrib.G(wo, wi) * F / (4 * cosTheta_i * cosTheta_o);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        Vector3f wo, Float uc, Point2f u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};
        if (mfDistrib.EffectivelySmooth()) {
            // Sample perfectly specular conductor BRDF
            Vector3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum f = FrConductor(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
            return BSDFSample(f, wi, 1, BxDFFlags::SpecularReflection);
        }
        // Sample Torrance--Sparow model for conductor BRDF
        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        if (wo.z == 0)
            return {};
        Vector3f wh = mfDistrib.Sample_wm(wo, u);
        Vector3f wi = Reflect(wo, wh);
        CHECK_RARE(1e-5f, Dot(wo, wh) <= 0);
        if (!SameHemisphere(wo, wi) || Dot(wo, wh) <= 0)
            return {};

        // Compute PDF of _wi_ for microfacet reflection
        Float pdf = mfDistrib.PDF(wo, wh) / (4 * Dot(wo, wh));

        Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        if (cosTheta_i == 0 || cosTheta_o == 0)
            return {};
        // Evaluate Fresnel factor _F_ for conductor BRDF
        Float frCosTheta_i = AbsDot(wi, wh);
        SampledSpectrum F = FrConductor(frCosTheta_i, eta, k);

        SampledSpectrum f =
            mfDistrib.D(wh) * mfDistrib.G(wo, wi) * F / (4 * cosTheta_i * cosTheta_o);
        return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return 0;
        if (!SameHemisphere(wo, wi))
            return 0;
        if (mfDistrib.EffectivelySmooth())
            return 0;
        // Return PDF for sampling Torrance--Sparrow conductor BRDF
        Vector3f wh = wo + wi;
        CHECK_RARE(1e-5f, LengthSquared(wh) == 0);
        CHECK_RARE(1e-5f, Dot(wo, wh) < 0);
        if (LengthSquared(wh) == 0 || Dot(wo, wh) <= 0)
            return 0;
        wh = Normalize(wh);
        return mfDistrib.PDF(wo, wh) / (4 * Dot(wo, wh));
    }

    PBRT_CPU_GPU
    void Regularize() { mfDistrib.Regularize(); }

  private:
    // ConductorBxDF Private Members
    TrowbridgeReitzDistribution mfDistrib;
    SampledSpectrum eta, k;
};

// TopOrBottomBxDF Definition
template <typename TopBxDF, typename BottomBxDF>
class TopOrBottomBxDF {
  public:
    // TopOrBottomBxDF Public Methods
    TopOrBottomBxDF() = default;
    PBRT_CPU_GPU
    TopOrBottomBxDF &operator=(const TopBxDF *t) {
        top = t;
        bottom = nullptr;
        return *this;
    }
    PBRT_CPU_GPU
    TopOrBottomBxDF &operator=(const BottomBxDF *b) {
        bottom = b;
        top = nullptr;
        return *this;
    }

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        return top ? top->f(wo, wi, mode) : bottom->f(wo, wi, mode);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        Vector3f wo, Float uc, Point2f u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return top ? top->Sample_f(wo, uc, u, mode, sampleFlags)
                   : bottom->Sample_f(wo, uc, u, mode, sampleFlags);
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return top ? top->PDF(wo, wi, mode, sampleFlags)
                   : bottom->PDF(wo, wi, mode, sampleFlags);
    }

    PBRT_CPU_GPU
    bool IsNonSpecular() const {
        BxDFFlags flags = top ? top->Flags() : bottom->Flags();
        return (flags & (BxDFFlags::Diffuse | BxDFFlags::Glossy));
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return top ? top->Flags() : bottom->Flags(); }

  private:
    const TopBxDF *top = nullptr;
    const BottomBxDF *bottom = nullptr;
};

// LayeredBxDF Definition
template <typename TopBxDF, typename BottomBxDF, bool twoSided>
class LayeredBxDF {
  public:
    // LayeredBxDF Public Methods
    LayeredBxDF() = default;
    PBRT_CPU_GPU
    LayeredBxDF(TopBxDF top, BottomBxDF bottom, Float thickness,
                const SampledSpectrum &albedo, Float g, int maxDepth, int nSamples)
        : top(top),
          bottom(bottom),
          thickness(std::max(thickness, std::numeric_limits<Float>::min())),
          g(g),
          albedo(albedo),
          maxDepth(maxDepth),
          nSamples(nSamples) {}

    std::string ToString() const;

    PBRT_CPU_GPU
    void Regularize() {
        top.Regularize();
        bottom.Regularize();
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        BxDFFlags topFlags = top.Flags(), bottomFlags = bottom.Flags();
        CHECK(IsTransmissive(topFlags) ||
              IsTransmissive(bottomFlags));  // otherwise, why bother?

        BxDFFlags flags = BxDFFlags::Reflection;
        if (IsSpecular(topFlags))
            flags = flags | BxDFFlags::Specular;

        if (IsDiffuse(topFlags) || IsDiffuse(bottomFlags) || albedo)
            flags = flags | BxDFFlags::Diffuse;
        else if (IsGlossy(topFlags) || IsGlossy(bottomFlags))
            flags = flags | BxDFFlags::Glossy;

        if (IsTransmissive(topFlags) && IsTransmissive(bottomFlags))
            flags = flags | BxDFFlags::Transmission;

        return flags;
    }

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        SampledSpectrum f(0.);
        // Estimate _LayeredBxDF_ value _f_ using random sampling
        // Set _wi_ and _wi_ for layered BSDF evaluation
        if (twoSided && wo.z < 0) {
            wo = -wo;
            wi = -wi;
        }

        // Determine entrance interface for layered BSDF
        TopOrBottomBxDF<TopBxDF, BottomBxDF> enterInterface;
        bool enteredTop = twoSided || wo.z > 0;
        if (enteredTop)
            enterInterface = &top;
        else
            enterInterface = &bottom;

        // Determine exit interface and exit $z$ for layered BSDF
        TopOrBottomBxDF<TopBxDF, BottomBxDF> exitInterface, nonExitInterface;
        if (SameHemisphere(wo, wi) ^ enteredTop) {
            exitInterface = &bottom;
            nonExitInterface = &top;
        } else {
            exitInterface = &top;
            nonExitInterface = &bottom;
        }
        Float exitZ = (SameHemisphere(wo, wi) ^ enteredTop) ? 0 : thickness;

        // Account for reflection at the entrance interface
        if (SameHemisphere(wo, wi))
            f = nSamples * enterInterface.f(wo, wi, mode);

        // Declare _RNG_ for layered BSDF evaluation
        RNG rng(Hash(GetOptions().seed, wo), Hash(wi));
        auto r = [&rng]() {
            return std::min<Float>(rng.Uniform<Float>(), OneMinusEpsilon);
        };

        for (int s = 0; s < nSamples; ++s) {
            // Sample random walk through layers to estimate BSDF value
            // Sample transmission direction through entrance interface
            Float uc = r();
            pstd::optional<BSDFSample> wos = enterInterface.Sample_f(
                wo, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Transmission);
            if (!wos || !wos->f || wos->pdf == 0 || wos->wi.z == 0)
                continue;

            // Sample BSDF for virtual light from _wi_
            uc = r();
            pstd::optional<BSDFSample> wis = exitInterface.Sample_f(
                wi, uc, Point2f(r(), r()), !mode, BxDFReflTransFlags::Transmission);
            if (!wis || !wis->f || wis->pdf == 0 || wis->wi.z == 0)
                continue;

            // Declare state for random walk through BSDF layers
            SampledSpectrum beta = wos->f * AbsCosTheta(wos->wi) / wos->pdf;
            Float z = enteredTop ? thickness : 0;
            Vector3f w = wos->wi;
            HGPhaseFunction phase(g);

            for (int depth = 0; depth < maxDepth; ++depth) {
                // Sample next event for layered BSDF evaluation random walk
                PBRT_DBG("beta: %f %f %f %f, w: %f %f %f, f: %f %f %f %f\n", beta[0],
                         beta[1], beta[2], beta[3], w.x, w.y, w.z, f[0], f[1], f[2],
                         f[3]);
                // Possibly terminate layered BSDF random walk with Russian Roulette
                if (depth > 3 && beta.MaxComponentValue() < 0.25f) {
                    Float q = std::max<Float>(0, 1 - beta.MaxComponentValue());
                    if (r() < q)
                        break;
                    beta /= 1 - q;
                    PBRT_DBG("After RR with q = %f, beta: %f %f %f %f\n", q, beta[0],
                             beta[1], beta[2], beta[3]);
                }

                // Account for media between layers and possibly scatter
                if (!albedo) {
                    // Advance to next layer boundary and update _beta_ for transmittance
                    z = (z == thickness) ? 0 : thickness;
                    beta *= Tr(thickness, w);

                } else {
                    // Sample medium scattering for layered BSDF evaluation
                    Float sigma_t = 1;
                    Float dz = SampleExponential(r(), sigma_t / std::abs(w.z));
                    Float zp = w.z > 0 ? (z + dz) : (z - dz);
                    DCHECK_RARE(1e-5, z == zp);
                    if (z == zp)
                        continue;
                    if (0 < zp && zp < thickness) {
                        // Handle scattering event in layered BSDF medium
                        // Account for scattering through _exitInterface_ using _wis_
                        Float wt = 1;
                        if (!IsSpecular(exitInterface.Flags()))
                            wt = PowerHeuristic(1, wis->pdf, 1, phase.PDF(-w, -wis->wi));
                        f += beta * albedo * phase.p(-w, -wis->wi) * wt *
                             Tr(zp - exitZ, wis->wi) * wis->f / wis->pdf;

                        // Sample phase function and update layered path state
                        Point2f u{r(), r()};
                        pstd::optional<PhaseFunctionSample> ps = phase.Sample_p(-w, u);
                        if (!ps || ps->pdf == 0 || ps->wi.z == 0)
                            continue;
                        beta *= albedo * ps->p / ps->pdf;
                        w = ps->wi;
                        z = zp;

                        // Possibly account for scattering through _exitInterface_
                        if (((z < exitZ && w.z > 0) || (z > exitZ && w.z < 0)) &&
                            !IsSpecular(exitInterface.Flags())) {
                            // Account for scattering through _exitInterface_
                            SampledSpectrum fExit = exitInterface.f(-w, wi, mode);
                            if (fExit) {
                                Float exitPDF = exitInterface.PDF(
                                    -w, wi, mode, BxDFReflTransFlags::Transmission);
                                Float wt = PowerHeuristic(1, ps->pdf, 1, exitPDF);
                                f += beta * Tr(zp - exitZ, ps->wi) * fExit * wt;
                            }
                        }

                        continue;
                    }
                    z = Clamp(zp, 0, thickness);
                }

                // Account for scattering at appropriate interface
                if (z == exitZ) {
                    // Account for reflection at _exitInterface_
                    Float uc = r();
                    pstd::optional<BSDFSample> bs = exitInterface.Sample_f(
                        -w, uc, Point2f(r(), r()), mode, BxDFReflTransFlags::Reflection);
                    if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
                        break;
                    beta *= bs->f * AbsCosTheta(bs->wi) / bs->pdf;
                    w = bs->wi;

                } else {
                    // Account for scattering at _nonExitInterface_
                    if (!IsSpecular(nonExitInterface.Flags())) {
                        // Add NEE contribution along presampled _wis_ direction
                        Float wt = 1;
                        if (!IsSpecular(exitInterface.Flags()))
                            wt = PowerHeuristic(1, wis->pdf, 1,
                                                nonExitInterface.PDF(-w, -wis->wi, mode));
                        f += beta * nonExitInterface.f(-w, -wis->wi, mode) *
                             AbsCosTheta(wis->wi) * wt * Tr(thickness, wis->wi) * wis->f /
                             wis->pdf;
                    }
                    // Sample new direction using BSDF at _nonExitInterface_
                    Float uc = r();
                    Point2f u(r(), r());
                    pstd::optional<BSDFSample> bs = nonExitInterface.Sample_f(
                        -w, uc, u, mode, BxDFReflTransFlags::Reflection);
                    if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
                        break;
                    beta *= bs->f * AbsCosTheta(bs->wi) / bs->pdf;
                    w = bs->wi;

                    if (!IsSpecular(exitInterface.Flags())) {
                        // Add NEE contribution along direction from BSDF sample
                        SampledSpectrum fExit = exitInterface.f(-w, wi, mode);
                        if (fExit) {
                            Float wt = 1;
                            if (!IsSpecular(nonExitInterface.Flags())) {
                                Float exitPDF = exitInterface.PDF(
                                    -w, wi, mode, BxDFReflTransFlags::Transmission);
                                wt = PowerHeuristic(1, bs->pdf, 1, exitPDF);
                            }
                            f += beta * Tr(thickness, bs->wi) * fExit * wt;
                        }
                    }
                }
            }
        }

        return f / nSamples;
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        Vector3f wo, Float uc, const Point2f &u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        CHECK(sampleFlags == BxDFReflTransFlags::All);  // for now
        // Set _wo_ for layered BSDF sampling
        bool flipWi = false;
        if (twoSided && wo.z < 0) {
            wo = -wo;
            flipWi = true;
        }

        // Sample BSDF at entrance interface to get initial direction _w_
        bool enteredTop = twoSided || wo.z > 0;
        pstd::optional<BSDFSample> bs =
            enteredTop ? top.Sample_f(wo, uc, u, mode) : bottom.Sample_f(wo, uc, u, mode);
        if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
            return {};
        if (bs->IsReflection()) {
            if (flipWi)
                bs->wi = -bs->wi;
            return bs;
        }
        Vector3f w = bs->wi;

        // Declare _RNG_ for layered BSDF sampling
        RNG rng(Hash(GetOptions().seed, wo), Hash(uc, u));
        auto r = [&rng]() {
            return std::min<Float>(rng.Uniform<Float>(), OneMinusEpsilon);
        };

        // Declare common variables for layered BSDF sampling
        SampledSpectrum f = bs->f * AbsCosTheta(bs->wi);
        Float pdf = bs->pdf;
        Float z = enteredTop ? thickness : 0;
        HGPhaseFunction phase(g);

        for (int depth = 0; depth < maxDepth; ++depth) {
            // Follow random walk through layers to sample layered BSDF
            // Possibly terminate layered BSDF sampling with Russian Roulette
            Float rrBeta = f.MaxComponentValue() / pdf;
            if (depth > 3 && rrBeta < 0.25f) {
                Float q = std::max<Float>(0, 1 - rrBeta);
                if (r() < q)
                    return {};
                pdf *= 1 - q;
            }
            if (w.z == 0)
                return {};

            if (albedo) {
                // Sample potential scattering event in layered medium
                Float sigma_t = 1;
                Float dz = SampleExponential(r(), sigma_t / AbsCosTheta(w));
                Float zp = w.z > 0 ? (z + dz) : (z - dz);
                CHECK_RARE(1e-5, zp == z);
                if (zp == z)
                    return {};
                if (0 < zp && zp < thickness) {
                    // Update path state for valid scattering event between interfaces
                    pstd::optional<PhaseFunctionSample> ps =
                        phase.Sample_p(-w, Point2f(r(), r()));
                    if (!ps || ps->pdf == 0 || ps->wi.z == 0)
                        return {};
                    f *= albedo * ps->p;
                    pdf *= ps->pdf;
                    w = ps->wi;
                    z = zp;

                    continue;
                }
                z = Clamp(zp, 0, thickness);
                if (z == 0)
                    DCHECK_LT(w.z, 0);
                else
                    DCHECK_GT(w.z, 0);

            } else {
                // Advance to the other layer interface
                z = (z == thickness) ? 0 : thickness;
                f *= Tr(thickness, w);
            }
            // Initialize _interface_ for current interface surface
            TopOrBottomBxDF<TopBxDF, BottomBxDF> interface;
            if (z == 0)
                interface = &bottom;
            else
                interface = &top;

            // Sample interface BSDF to determine new path direction
            Float uc = r();
            Point2f u(r(), r());
            pstd::optional<BSDFSample> bs = interface.Sample_f(-w, uc, u, mode);
            if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
                return {};
            f *= bs->f;
            pdf *= bs->pdf;
            w = bs->wi;

            // Return _BSDFSample_ if path has left the layers
            if (bs->IsTransmission()) {
                BxDFFlags flags = SameHemisphere(wo, w) ? BxDFFlags::GlossyReflection
                                                        : BxDFFlags::GlossyTransmission;
                if (flipWi)
                    w = -w;
                return BSDFSample(f, w, pdf, flags, 1.f, true);
            }

            // Scale _f_ by cosine term after scattering at the interface
            f *= AbsCosTheta(bs->wi);
        }
        return {};
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        CHECK(sampleFlags == BxDFReflTransFlags::All);  // for now
        // Set _wi_ and _wi_ for layered BSDF evaluation
        if (twoSided && wo.z < 0) {
            wo = -wo;
            wi = -wi;
        }

        // Declare _RNG_ for layered PDF evaluation
        RNG rng(Hash(GetOptions().seed, wi), Hash(wo));
        auto r = [&rng]() {
            return std::min<Float>(rng.Uniform<Float>(), OneMinusEpsilon);
        };

        // Update _pdfSum_ for reflection at the entrance layer
        bool enteredTop = twoSided || wo.z > 0;
        Float pdfSum = 0;
        if (SameHemisphere(wo, wi)) {
            auto reflFlag = BxDFReflTransFlags::Reflection;
            pdfSum += enteredTop ? nSamples * top.PDF(wo, wi, mode, reflFlag)
                                 : nSamples * bottom.PDF(wo, wi, mode, reflFlag);
        }

        for (int s = 0; s < nSamples; ++s) {
            // Evaluate layered BSDF PDF sample
            if (SameHemisphere(wo, wi)) {
                // Evaluate TRT term for PDF estimate
                TopOrBottomBxDF<TopBxDF, BottomBxDF> rInterface, tInterface;
                if (enteredTop) {
                    rInterface = &bottom;
                    tInterface = &top;
                } else {
                    rInterface = &top;
                    tInterface = &bottom;
                }
                // Sample _tInterface_ to get direction into the layers
                auto trans = BxDFReflTransFlags::Transmission;
                pstd::optional<BSDFSample> wos, wis;
                wos = tInterface.Sample_f(wo, r(), {r(), r()}, mode, trans);
                wis = tInterface.Sample_f(wi, r(), {r(), r()}, !mode, trans);

                // Update _pdfSum_ accounting for TRT scattering events
                if (wos && wos->f && wos->pdf > 0 && wis && wis->f && wis->pdf > 0) {
                    if (!tInterface.IsNonSpecular())
                        pdfSum += rInterface.PDF(-wos->wi, -wis->wi, mode);
                    else {
                        // Use multiple importance sampling to estimate PDF product
                        pstd::optional<BSDFSample> rs =
                            rInterface.Sample_f(-wos->wi, r(), {r(), r()}, mode);
                        if (rs && rs->f && rs->pdf > 0) {
                            if (!rInterface.IsNonSpecular())
                                pdfSum += tInterface.PDF(-rs->wi, wi, mode);
                            else {
                                // Compute MIS-weighted estimate of Equation
                                // (\ref{eq:pdf-triple-canceled-one})
                                Float rPDF = rInterface.PDF(-wos->wi, -wis->wi, mode);
                                Float wt = PowerHeuristic(1, wis->pdf, 1, rPDF);
                                pdfSum += wt * rPDF;

                                Float tPDF = tInterface.PDF(-rs->wi, wi, mode);
                                wt = PowerHeuristic(1, rs->pdf, 1, tPDF);
                                pdfSum += wt * tPDF;
                            }
                        }
                    }
                }

            } else {
                // Evaluate TT term for PDF estimate
                TopOrBottomBxDF<TopBxDF, BottomBxDF> toInterface, tiInterface;
                if (enteredTop) {
                    toInterface = &top;
                    tiInterface = &bottom;
                } else {
                    toInterface = &bottom;
                    tiInterface = &top;
                }

                Float uc = r();
                Point2f u(r(), r());
                pstd::optional<BSDFSample> wos = toInterface.Sample_f(wo, uc, u, mode);
                if (!wos || !wos->f || wos->pdf == 0 || wos->wi.z == 0 ||
                    wos->IsReflection())
                    continue;

                uc = r();
                u = Point2f(r(), r());
                pstd::optional<BSDFSample> wis = tiInterface.Sample_f(wi, uc, u, !mode);
                if (!wis || !wos->f || wos->pdf == 0 || wos->wi.z == 0 ||
                    wis->IsReflection())
                    continue;

                if (IsSpecular(toInterface.Flags()))
                    pdfSum += tiInterface.PDF(-wos->wi, wi, mode);
                else if (IsSpecular(tiInterface.Flags()))
                    pdfSum += toInterface.PDF(wo, -wis->wi, mode);
                else
                    pdfSum += (toInterface.PDF(wo, -wis->wi, mode) +
                               tiInterface.PDF(-wos->wi, wi, mode)) /
                              2;
            }
        }
        // Return mixture of PDF estimate and constant PDF
        return Lerp(0.9f, 1 / (4 * Pi), pdfSum / nSamples);
    }

  private:
    // LayeredBxDF Private Methods
    PBRT_CPU_GPU
    static Float Tr(Float dz, Vector3f w) {
        if (std::abs(dz) <= std::numeric_limits<Float>::min())
            return 1;
        return FastExp(-std::abs(dz / w.z));
    }

    // LayeredBxDF Private Members
    TopBxDF top;
    BottomBxDF bottom;
    Float thickness, g;
    SampledSpectrum albedo;
    int maxDepth, nSamples;
};

// CoatedDiffuseBxDF Definition
class CoatedDiffuseBxDF
    : public LayeredBxDF<DielectricInterfaceBxDF, IdealDiffuseBxDF, true> {
  public:
    // CoatedDiffuseBxDF Public Methods
    using LayeredBxDF::LayeredBxDF;
    PBRT_CPU_GPU
    static constexpr const char *Name() { return "CoatedDiffuseBxDF"; }
};

// CoatedConductorBxDF Definition
class CoatedConductorBxDF
    : public LayeredBxDF<DielectricInterfaceBxDF, ConductorBxDF, true> {
  public:
    // CoatedConductorBxDF Public Methods
    PBRT_CPU_GPU
    static constexpr const char *Name() { return "CoatedConductorBxDF"; }
    using LayeredBxDF::LayeredBxDF;
};

// HairBxDF Definition
class HairBxDF {
  public:
    // HairBxDF Public Methods
    HairBxDF() = default;
    PBRT_CPU_GPU
    HairBxDF(Float h, Float eta, const SampledSpectrum &sigma_a, Float beta_m,
             Float beta_n, Float alpha);
    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;
    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(Vector3f wo, Float uc, Point2f u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const;
    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const;

    PBRT_CPU_GPU
    void Regularize() {}

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "HairBxDF"; }
    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return BxDFFlags::GlossyReflection; }

    PBRT_CPU_GPU
    static RGBUnboundedSpectrum SigmaAFromConcentration(Float ce, Float cp);
    PBRT_CPU_GPU
    static SampledSpectrum SigmaAFromReflectance(const SampledSpectrum &c, Float beta_n,
                                                 const SampledWavelengths &lambda);

  private:
    // HairBxDF Constants
    static constexpr int pMax = 3;

    // HairBxDF Private Methods
    PBRT_CPU_GPU
    static Float Mp(Float cosTheta_i, Float cosTheta_o, Float sinTheta_i,
                    Float sinTheta_o, Float v) {
        Float a = cosTheta_i * cosTheta_o / v;
        Float b = sinTheta_i * sinTheta_o / v;
        Float mp = (v <= .1)
                       ? (FastExp(LogI0(a) - b - 1 / v + 0.6931f + std::log(1 / (2 * v))))
                       : (FastExp(-b) * I0(a)) / (std::sinh(1 / v) * 2 * v);
        CHECK(!IsInf(mp) && !IsNaN(mp));
        return mp;
    }

    PBRT_CPU_GPU
    static pstd::array<SampledSpectrum, pMax + 1> Ap(Float cosTheta_o, Float eta, Float h,
                                                     const SampledSpectrum &T) {
        pstd::array<SampledSpectrum, pMax + 1> ap;
        // Compute $p=0$ attenuation at initial cylinder intersection
        Float cosGamma_o = SafeSqrt(1 - h * h);
        Float cosTheta = cosTheta_o * cosGamma_o;
        Float f = FrDielectric(cosTheta, eta);
        ap[0] = SampledSpectrum(f);

        // Compute $p=1$ attenuation term
        ap[1] = Sqr(1 - f) * T;

        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p)
            ap[p] = ap[p - 1] * T * f;

        // Compute attenuation term accounting for remaining orders of scattering
        if (1.f - T * f)
            ap[pMax] = ap[pMax - 1] * f * T / (1.f - T * f);

        return ap;
    }

    PBRT_CPU_GPU
    static inline Float Phi(int p, Float gamma_o, Float gamma_t) {
        return 2 * p * gamma_t - 2 * gamma_o + p * Pi;
    }

    PBRT_CPU_GPU
    static inline Float Np(Float phi, int p, Float s, Float gamma_o, Float gamma_t) {
        Float dphi = phi - Phi(p, gamma_o, gamma_t);
        // Remap _dphi_ to $[-\pi,\pi]$
        while (dphi > Pi)
            dphi -= 2 * Pi;
        while (dphi < -Pi)
            dphi += 2 * Pi;

        return TrimmedLogistic(dphi, s, -Pi, Pi);
    }

    PBRT_CPU_GPU
    pstd::array<Float, pMax + 1> ComputeApPDF(Float cosThetaO) const;

    // HairBxDF Private Members
    Float h, gamma_o, eta;
    SampledSpectrum sigma_a;
    Float beta_m, beta_n;
    Float v[pMax + 1];
    Float s;
    Float sin2kAlpha[3], cos2kAlpha[3];
};

// MeasuredBxDF Definition
class MeasuredBxDF {
  public:
    // MeasuredBxDF Public Methods
    MeasuredBxDF() = default;
    PBRT_CPU_GPU
    MeasuredBxDF(const MeasuredBRDF *brdf, const SampledWavelengths &lambda)
        : brdf(brdf), lambda(lambda) {}

    static MeasuredBRDF *BRDFDataFromFile(const std::string &filename, Allocator alloc);

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(Vector3f wo, Float uc, Point2f u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const;
    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const;

    PBRT_CPU_GPU
    void Regularize() {}

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "MeasuredBxDF"; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Glossy); }

  private:
    // MeasuredBxDF Private Methods
    PBRT_CPU_GPU
    static Float u2theta(Float u) { return Sqr(u) * (Pi / 2.f); }
    PBRT_CPU_GPU
    static Float u2phi(Float u) { return (2.f * u - 1.f) * Pi; }
    PBRT_CPU_GPU
    static Float theta2u(Float theta) { return std::sqrt(theta * (2.f / Pi)); }
    PBRT_CPU_GPU
    static Float phi2u(Float phi) { return (phi + Pi) / (2.f * Pi); }

    // MeasuredBxDF Private Members
    const MeasuredBRDF *brdf;
    SampledWavelengths lambda;
};

// NormalizedFresnelBxDF Definition
class NormalizedFresnelBxDF {
  public:
    // NormalizedFresnelBxDF Public Methods
    NormalizedFresnelBxDF() = default;
    PBRT_CPU_GPU
    NormalizedFresnelBxDF(Float eta) : eta(eta) {}

    PBRT_CPU_GPU
    BSDFSample Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                        TransportMode mode, BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};

        // Cosine-sample the hemisphere, flipping the direction if necessary
        Vector3f wi = SampleCosineHemisphere(u);
        if (wo.z < 0)
            wi.z *= -1;
        return BSDFSample(f(wo, wi, mode), wi, PDF(wo, wi, mode, sampleFlags),
                          BxDFFlags::DiffuseReflection);
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return 0;
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
    }

    PBRT_CPU_GPU
    void Regularize() {}

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "NormalizedFresnelBxDF"; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return BxDFFlags(BxDFFlags::Reflection | BxDFFlags::Diffuse);
    }

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        if (!SameHemisphere(wo, wi))
            return SampledSpectrum(0.f);
        // Compute $\Sw$ factor for BSSRDF value
        Float c = 1 - 2 * FresnelMoment1(1 / eta);
        SampledSpectrum f((1 - FrDielectric(CosTheta(wi), eta)) / (c * Pi));

        // Update BSSRDF transmission term to account for adjoint light transport
        if (mode == TransportMode::Radiance)
            f *= Sqr(eta);

        return f;
    }

  private:
    friend class SOA<NormalizedFresnelBxDF>;
    Float eta;
};

inline SampledSpectrum BxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    auto f = [&](auto ptr) -> SampledSpectrum { return ptr->f(wo, wi, mode); };
    return Dispatch(f);
}

inline pstd::optional<BSDFSample> BxDF::Sample_f(Vector3f wo, Float uc, Point2f u,
                                                 TransportMode mode,
                                                 BxDFReflTransFlags sampleFlags) const {
    auto sample_f = [&](auto ptr) -> pstd::optional<BSDFSample> {
        return ptr->Sample_f(wo, uc, u, mode, sampleFlags);
    };
    return Dispatch(sample_f);
}

inline Float BxDF::PDF(Vector3f wo, Vector3f wi, TransportMode mode,
                       BxDFReflTransFlags sampleFlags) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(wo, wi, mode, sampleFlags); };
    return Dispatch(pdf);
}

inline BxDFFlags BxDF::Flags() const {
    auto flags = [&](auto ptr) { return ptr->Flags(); };
    return Dispatch(flags);
}

inline void BxDF::Regularize() {
    auto regularize = [&](auto ptr) { ptr->Regularize(); };
    return Dispatch(regularize);
}

extern template class LayeredBxDF<DielectricInterfaceBxDF, IdealDiffuseBxDF, true>;
extern template class LayeredBxDF<DielectricInterfaceBxDF, ConductorBxDF, true>;

}  // namespace pbrt

#endif  // PBRT_BXDFS_H
