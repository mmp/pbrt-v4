// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BSSRDF_H
#define PBRT_BSSRDF_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bssrdf.h>
#include <pbrt/bsdf.h>
#include <pbrt/interaction.h>
#include <pbrt/util/check.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// BSSRDFSample Definition
struct BSSRDFSample {
    SampledSpectrum Sp;
    Float pdf;
    BSDF Sw;
    Vector3f wo;
};

// SubsurfaceInteraction Definition
struct SubsurfaceInteraction {
    // SubsurfaceInteraction Public Methods
    SubsurfaceInteraction() = default;
    PBRT_CPU_GPU
    SubsurfaceInteraction(const SurfaceInteraction &si)
        : pi(si.pi),
          n(si.n),
          dpdu(si.dpdu),
          dpdv(si.dpdv),
          ns(si.shading.n),
          dpdus(si.shading.dpdu),
          dpdvs(si.shading.dpdv) {}

    PBRT_CPU_GPU
    operator SurfaceInteraction() const {
        SurfaceInteraction si;
        si.pi = pi;
        si.n = n;
        si.dpdu = dpdu;
        si.dpdv = dpdv;
        si.shading.n = ns;
        si.shading.dpdu = dpdus;
        si.shading.dpdv = dpdvs;
        return si;
    }

    PBRT_CPU_GPU
    Point3f p() const { return Point3f(pi); }

    // SubsurfaceInteraction Public Members
    Point3fi pi;
    Normal3f n, ns;
    Vector3f dpdu, dpdv, dpdus, dpdvs;
};

// BSSRDF Function Declarations
Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r);
Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r);

void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t);

// BSSRDFTable Definition
struct BSSRDFTable {
    // BSSRDFTable Public Members
    pstd::vector<Float> rhoSamples, radiusSamples;
    pstd::vector<Float> profile;
    pstd::vector<Float> rhoEff;
    pstd::vector<Float> profileCDF;

    // BSSRDFTable Public Methods
    BSSRDFTable(int nRhoSamples, int nRadiusSamples, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    Float EvalProfile(int rhoIndex, int radiusIndex) const {
        CHECK(rhoIndex >= 0 && rhoIndex < rhoSamples.size());
        CHECK(radiusIndex >= 0 && radiusIndex < radiusSamples.size());
        return profile[rhoIndex * radiusSamples.size() + radiusIndex];
    }
};

// BSSRDFProbeSegment Definition
struct BSSRDFProbeSegment {
    // BSSRDFProbeSegment Public Methods
    BSSRDFProbeSegment() = default;
    PBRT_CPU_GPU
    BSSRDFProbeSegment(const Point3f &p0, const Point3f &p1) : p0(p0), p1(p1) {}

    Point3f p0, p1;
};

// TabulatedBSSRDF Definition
class TabulatedBSSRDF {
  public:
    // TabulatedBSSRDF Type Definitions
    using BxDF = NormalizedFresnelBxDF;

    // TabulatedBSSRDF Public Methods
    TabulatedBSSRDF() = default;
    PBRT_CPU_GPU
    TabulatedBSSRDF(const Point3f &po, const Normal3f &ns, const Vector3f &wo, Float eta,
                    const SampledSpectrum &sigma_a, const SampledSpectrum &sigma_s,
                    const BSSRDFTable *table)
        : po(po), wo(wo), eta(eta), ns(ns), table(table) {
        sigma_t = sigma_a + sigma_s;
        rho = SafeDiv(sigma_s, sigma_t);
    }

    PBRT_CPU_GPU
    SampledSpectrum Sp(const Point3f &pi) const { return Sr(Distance(po, pi)); }

    PBRT_CPU_GPU
    SampledSpectrum Sr(Float r) const {
        SampledSpectrum Sr(0.f);
        for (int ch = 0; ch < NSpectrumSamples; ++ch) {
            // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
            Float rOptical = r * sigma_t[ch];

            // Compute spline weights to interpolate BSSRDF on channel _ch_
            int rhoOffset, radiusOffset;
            Float rhoWeights[4], radiusWeights[4];
            if (!CatmullRomWeights(table->rhoSamples, rho[ch], &rhoOffset, rhoWeights) ||
                !CatmullRomWeights(table->radiusSamples, rOptical, &radiusOffset,
                                   radiusWeights))
                continue;

            // Set BSSRDF value _Sr[ch]_ using tensor spline interpolation
            Float sr = 0;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    // Accumulate contribution of $(i,j)$ table sample
                    if (Float weight = rhoWeights[i] * radiusWeights[j]; weight != 0)
                        sr +=
                            weight * table->EvalProfile(rhoOffset + i, radiusOffset + j);
                }
            // Cancel marginal PDF factor from tabulated BSSRDF profile
            if (rOptical != 0)
                sr /= 2 * Pi * rOptical;

            Sr[ch] = sr;
        }
        // Transform BSSRDF value into world space units
        Sr *= sigma_t * sigma_t;

        return ClampZero(Sr);
    }

    PBRT_CPU_GPU
    Float SampleSr(Float u) const {
        if (sigma_t[0] == 0)
            return -1;
        return SampleCatmullRom2D(table->rhoSamples, table->radiusSamples, table->profile,
                                  table->profileCDF, rho[0], u) /
               sigma_t[0];
    }

    PBRT_CPU_GPU
    Float PDF_Sr(int ch, Float r) const {
        // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
        Float rOptical = r * sigma_t[ch];

        // Compute spline weights to interpolate BSSRDF density on channel _ch_
        int rhoOffset, radiusOffset;
        Float rhoWeights[4], radiusWeights[4];
        if (!CatmullRomWeights(table->rhoSamples, rho[ch], &rhoOffset, rhoWeights) ||
            !CatmullRomWeights(table->radiusSamples, rOptical, &radiusOffset,
                               radiusWeights))
            return 0;

        // Return BSSRDF profile density for channel _ch_
        Float sr = 0, rhoEff = 0;
        for (int i = 0; i < 4; ++i)
            if (rhoWeights[i] != 0) {
                // Update _rhoEff_ and _sr_ for channel _ch_
                rhoEff += table->rhoEff[rhoOffset + i] * rhoWeights[i];
                for (int j = 0; j < 4; ++j)
                    if (radiusWeights[j] != 0)
                        sr += table->EvalProfile(rhoOffset + i, radiusOffset + j) *
                              rhoWeights[i] * radiusWeights[j];
            }
        // Cancel marginal PDF factor from tabulated BSSRDF profile
        if (rOptical != 0)
            sr /= 2 * Pi * rOptical;

        return std::max<Float>(0, sr * sigma_t[ch] * sigma_t[ch] / rhoEff);
    }

    PBRT_CPU_GPU
    pstd::optional<BSSRDFProbeSegment> SampleSp(Float u1, Point2f u2) const {
        // Choose projection axis for BSSRDF sampling
        Frame f;
        if (u1 < 0.25f)
            f = Frame::FromX(ns);
        else if (u1 < 0.5f)
            f = Frame::FromY(ns);
        else
            f = Frame::FromZ(ns);

        // Sample BSSRDF profile in polar coordinates
        Float r = SampleSr(u2[0]);
        if (r < 0)
            return {};
        Float phi = 2 * Pi * u2[1];

        // Compute BSSRDF profile bounds and intersection height
        Float r_max = SampleSr(0.999f);
        if (r >= r_max)
            return {};
        Float l = 2 * std::sqrt(Sqr(r_max) - Sqr(r));

        // Return BSSRDF sampling ray segment
        Point3f pStart =
            po + r * (f.x * std::cos(phi) + f.y * std::sin(phi)) - l * f.z * 0.5f;
        Point3f pTarget = pStart + l * f.z;
        return BSSRDFProbeSegment{pStart, pTarget};
    }

    PBRT_CPU_GPU
    Float PDF_Sp(const Point3f &pi, const Normal3f &ni) const {
        // Express $\pti-\pto$ and $\bold{n}_i$ with respect to local coordinates at
        // $\pto$
        Vector3f d = pi - po;
        Frame f = Frame::FromZ(ns);
        Vector3f dLocal = f.ToLocal(d);
        Normal3f nLocal = f.ToLocal(ni);

        // Compute BSSRDF profile radius under projection along each axis
        Float rProj[3] = {std::sqrt(Sqr(dLocal.y) + Sqr(dLocal.z)),
                          std::sqrt(Sqr(dLocal.z) + Sqr(dLocal.x)),
                          std::sqrt(Sqr(dLocal.x) + Sqr(dLocal.y))};

        // Return combined probability from all BSSRDF sampling strategies
        Float pdf = 0, axisProb[3] = {.25f, .25f, .5f};
        Float chProb = 1 / (Float)NSpectrumSamples;
        for (int axis = 0; axis < 3; ++axis)
            for (int ch = 0; ch < NSpectrumSamples; ++ch)
                pdf += PDF_Sr(ch, rProj[axis]) * std::abs(nLocal[axis]) * chProb *
                       axisProb[axis];
        return pdf;
    }

    PBRT_CPU_GPU
    BSSRDFSample ProbeIntersectionToSample(const SubsurfaceInteraction &si,
                                           NormalizedFresnelBxDF *bxdf) const {
        *bxdf = NormalizedFresnelBxDF(eta);
        Vector3f wo = Vector3f(si.ns);
        BSDF bsdf(wo, si.n, si.ns, si.dpdus, bxdf, eta);
        return BSSRDFSample{Sp(si.p()), PDF_Sp(si.p(), si.n), bsdf, wo};
    }

    std::string ToString() const;

  private:
    friend class SOA<TabulatedBSSRDF>;
    // TabulatedBSSRDF Private Members
    Point3f po;
    Vector3f wo;
    Normal3f ns;
    Float eta;
    SampledSpectrum sigma_t, rho;
    const BSSRDFTable *table;
};

// BSSRDF Inline Functions
PBRT_CPU_GPU inline void SubsurfaceFromDiffuse(const BSSRDFTable &t,
                                               const SampledSpectrum &rhoEff,
                                               const SampledSpectrum &mfp,
                                               SampledSpectrum *sigma_a,
                                               SampledSpectrum *sigma_s) {
    for (int c = 0; c < NSpectrumSamples; ++c) {
        Float rho = InvertCatmullRom(t.rhoSamples, t.rhoEff, rhoEff[c]);
        (*sigma_s)[c] = rho / mfp[c];
        (*sigma_a)[c] = (1 - rho) / mfp[c];
    }
}

inline pstd::optional<BSSRDFProbeSegment> BSSRDFHandle::SampleSp(Float u1,
                                                                 Point2f u2) const {
    auto sample = [&](auto ptr) { return ptr->SampleSp(u1, u2); };
    return Dispatch(sample);
}

inline BSSRDFSample BSSRDFHandle::ProbeIntersectionToSample(
    const SubsurfaceInteraction &si, ScratchBuffer &scratchBuffer) const {
    auto pits = [&](auto ptr) {
        using BxDF = typename std::remove_reference<decltype(*ptr)>::type::BxDF;
        BxDF *bxdf = (BxDF *)scratchBuffer.Alloc(sizeof(BxDF), alignof(BxDF));
        return ptr->ProbeIntersectionToSample(si, bxdf);
    };
    return Dispatch(pits);
}

}  // namespace pbrt

#endif  // PBRT_BSSRDF_H
