// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BSDF_H
#define PBRT_BSDF_H

#include <pbrt/pbrt.h>

#include <pbrt/bxdfs.h>
#include <pbrt/interaction.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

// BSDF Definition
class BSDF {
  public:
    // BSDF Public Methods
    BSDF() = default;
    PBRT_CPU_GPU
    BSDF(const Vector3f &wo, const Normal3f &n, const Normal3f &ns, const Vector3f &dpdus,
         BxDFHandle bxdf, Float eta = 1)
        : eta(Dot(wo, n) < 0 ? 1 / eta : eta),
          bxdf(bxdf),
          ng(n),
          shadingFrame(Frame::FromXZ(Normalize(dpdus), Vector3f(ns))) {}

    PBRT_CPU_GPU
    operator bool() const { return (bool)bxdf; }

    PBRT_CPU_GPU
    Vector3f RenderToLocal(const Vector3f &v) const { return shadingFrame.ToLocal(v); }
    PBRT_CPU_GPU
    Vector3f LocalToRender(const Vector3f &v) const { return shadingFrame.FromLocal(v); }

    PBRT_CPU_GPU
    BxDFHandle GetBxDF() const { return bxdf; }
    PBRT_CPU_GPU
    void SetBxDF(BxDFHandle b) { bxdf = b; }

    PBRT_CPU_GPU
    bool IsNonSpecular() const {
        return (bxdf.Flags() & (BxDFFlags::Diffuse | BxDFFlags::Glossy));
    }
    PBRT_CPU_GPU
    bool IsDiffuse() const { return (bxdf.Flags() & BxDFFlags::Diffuse); }
    PBRT_CPU_GPU
    bool IsGlossy() const { return (bxdf.Flags() & BxDFFlags::Glossy); }
    PBRT_CPU_GPU
    bool IsSpecular() const { return (bxdf.Flags() & BxDFFlags::Specular); }
    PBRT_CPU_GPU
    bool HasReflection() const { return (bxdf.Flags() & BxDFFlags::Reflection); }
    PBRT_CPU_GPU
    bool HasTransmission() const { return (bxdf.Flags() & BxDFFlags::Transmission); }

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f woRender, Vector3f wiRender,
                      TransportMode mode = TransportMode::Radiance) const {
        Vector3f wi = RenderToLocal(wiRender), wo = RenderToLocal(woRender);
        if (wo.z == 0)
            return {};
        return bxdf.f(wo, wi, mode) * GBump(woRender, wiRender, mode);
    }

    template <typename BxDF>
    PBRT_CPU_GPU SampledSpectrum f(Vector3f woW, Vector3f wiW,
                                   TransportMode mode = TransportMode::Radiance) const {
        Vector3f wi = RenderToLocal(wiW), wo = RenderToLocal(woW);
        if (wo.z == 0)
            return {};
        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        return specificBxDF->f(wo, wi, mode) * GBump(woW, wiW, mode);
    }

    PBRT_CPU_GPU
    SampledSpectrum rho(pstd::span<const Float> uc1, pstd::span<const Point2f> u1,
                        pstd::span<const Float> uc2, pstd::span<const Point2f> u2) const {
        return bxdf.rho(uc1, u1, uc2, u2);
    }
    PBRT_CPU_GPU
    SampledSpectrum rho(const Vector3f &woRender, pstd::span<const Float> uc,
                        pstd::span<const Point2f> u) const {
        Vector3f wo = RenderToLocal(woRender);
        return bxdf.rho(wo, uc, u);
    }

    PBRT_CPU_GPU
    BSDFSample Sample_f(Vector3f woRender, Float u, const Point2f &u2,
                        TransportMode mode = TransportMode::Radiance,
                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = RenderToLocal(woRender);
        if (wo.z == 0 || !(bxdf.Flags() & sampleFlags))
            return {};
        BSDFSample bs = bxdf.Sample_f(wo, u, u2, mode, sampleFlags);
        if (!bs || !bs.f)
            return {};
        DCHECK_GT(bs.pdf, 0);

        PBRT_DBG("For wo = (%f, %f, %f), ng %f %f %f ns %f %f %f "
                 "sampled f = %f %f %f %f, pdf = %f, ratio[0] = %f "
                 "wi = (%f, %f, %f)\n",
                 wo.x, wo.y, wo.z, ng.x, ng.y, ng.z,
                 shadingFrame.z.x, shadingFrame.z.y, shadingFrame.z.z,
                 bs.f[0], bs.f[1], bs.f[2], bs.f[3], bs.pdf,
                 (bs.pdf > 0) ? (bs.f[0] / bs.pdf) : 0, bs.wi.x, bs.wi.y, bs.wi.z);

        bs.wi = LocalToRender(bs.wi);
        bs.f *= GBump(woRender, bs.wi, mode);
        return bs;
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f woRender, Vector3f wiRender,
              TransportMode mode = TransportMode::Radiance,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = RenderToLocal(woRender), wi = RenderToLocal(wiRender);
        if (wo.z == 0)
            return 0;
        return bxdf.PDF(wo, wi, mode, sampleFlags);
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return bxdf.SampledPDFIsProportional(); }

    template <typename BxDF>
    PBRT_CPU_GPU BSDFSample
    Sample_f(Vector3f woRender, Float u, const Point2f &u2,
             TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = RenderToLocal(woRender);
        if (wo.z == 0)
            return {};

        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        if (!(specificBxDF->Flags() & sampleFlags))
            return {};

        BSDFSample bs = specificBxDF->Sample_f(wo, u, u2, mode, sampleFlags);
        if (!bs || !bs.f)
            return {};
        CHECK_GT(bs.pdf, 0);

        PBRT_DBG("For wo = (%f, %f, %f), ng %f %f %f ns %f %f %f "
                 "sampled f = %f %f %f %f, pdf = %f, ratio[0] = %f "
                 "wi = (%f, %f, %f)\n",
                 wo.x, wo.y, wo.z, ng.x, ng.y, ng.z,
                 shadingFrame.z.x, shadingFrame.z.y, shadingFrame.z.z,
                 bs.f[0], bs.f[1], bs.f[2], bs.f[3], bs.pdf,
                 (bs.pdf > 0) ? (bs.f[0] / bs.pdf) : 0, bs.wi.x, bs.wi.y, bs.wi.z);

        bs.wi = LocalToRender(bs.wi);
        bs.f *= GBump(woRender, bs.wi, mode);

        return bs;
    }

    template <typename BxDF>
    PBRT_CPU_GPU Float
    PDF(Vector3f woRender, Vector3f wiRender,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = RenderToLocal(woRender), wi = RenderToLocal(wiRender);
        if (wo.z == 0)
            return 0.;
        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        return specificBxDF->PDF(wo, wi, mode, sampleFlags);
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    void Regularize() { bxdf.Regularize(); }

    // BSDF Public Members
    Float eta;

  private:
    friend class SOA<BSDF>;
    // BSDF Private Methods
    PBRT_CPU_GPU
    Float GBump(Vector3f wo, Vector3f wi, TransportMode mode) const {
        return 1;  // disable for now...

        Vector3f w = (mode == TransportMode::Radiance) ? wi : wo;
        Normal3f ngf = FaceForward(ng, w);
        Normal3f nsf = FaceForward(Normal3f(shadingFrame.z), ngf);
        Float cosThetaIs = std::max<Float>(0, Dot(nsf, w)), cosThetaIg = Dot(ngf, w);
        Float cosThetaN = Dot(ngf, nsf);
        CHECK_GE(cosThetaIs, 0);
        CHECK_GE(cosThetaIg, 0);
        CHECK_GE(cosThetaN, 0);

        if (cosThetaIs == 0 || cosThetaIg == 0 || cosThetaN == 0)
            return 0;
        Float G = cosThetaIg / (cosThetaIs * cosThetaN);
        if (G >= 1)
            return 1;

        return -G * G * G + G * G + G;
    }

    // BSDF Private Members
    BxDFHandle bxdf;
    Frame shadingFrame;
    Normal3f ng;
};

}  // namespace pbrt

#endif  // PBRT_BSDF_H
