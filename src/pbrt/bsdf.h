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
         BxDF bxdf)
        : bxdf(bxdf),
          ng(n),
          shadingFrame(Frame::FromXZ(Normalize(dpdus), Vector3f(ns))) {}

    PBRT_CPU_GPU
    operator bool() const { return (bool)bxdf; }

    PBRT_CPU_GPU
    Vector3f RenderToLocal(const Vector3f &v) const { return shadingFrame.ToLocal(v); }
    PBRT_CPU_GPU
    Vector3f LocalToRender(const Vector3f &v) const { return shadingFrame.FromLocal(v); }

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
        return bxdf.f(wo, wi, mode);
    }

    template <typename BxDF>
    PBRT_CPU_GPU SampledSpectrum f(Vector3f woRender, Vector3f wiRender,
                                   TransportMode mode = TransportMode::Radiance) const {
        Vector3f wi = RenderToLocal(wiRender), wo = RenderToLocal(woRender);
        if (wo.z == 0)
            return {};
        const BxDF *specificBxDF = bxdf.CastOrNullptr<BxDF>();
        return specificBxDF->f(wo, wi, mode);
    }

    PBRT_CPU_GPU
    SampledSpectrum rho(pstd::span<const Point2f> u1, pstd::span<const Float> uc,
                        pstd::span<const Point2f> u2) const {
        return bxdf.rho(u1, uc, u2);
    }
    PBRT_CPU_GPU
    SampledSpectrum rho(const Vector3f &woRender, pstd::span<const Float> uc,
                        pstd::span<const Point2f> u) const {
        Vector3f wo = RenderToLocal(woRender);
        return bxdf.rho(wo, uc, u);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        Vector3f woRender, Float u, Point2f u2,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = RenderToLocal(woRender);
        if (wo.z == 0 || !(bxdf.Flags() & sampleFlags))
            return {};
        // Sample _bxdf_ and return _BSDFSample_
        pstd::optional<BSDFSample> bs = bxdf.Sample_f(wo, u, u2, mode, sampleFlags);
        if (bs)
            DCHECK_GE(bs->pdf, 0);
        if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
            return {};
        PBRT_DBG("For wo = (%f, %f, %f), ng %f %f %f ns %f %f %f "
                 "sampled f = %f %f %f %f, pdf = %f, ratio[0] = %f "
                 "wi = (%f, %f, %f)\n",
                 wo.x, wo.y, wo.z, ng.x, ng.y, ng.z, shadingFrame.z.x, shadingFrame.z.y,
                 shadingFrame.z.z, bs->f[0], bs->f[1], bs->f[2], bs->f[3], bs->pdf,
                 (bs->pdf > 0) ? (bs->f[0] / bs->pdf) : 0, bs->wi.x, bs->wi.y, bs->wi.z);
        bs->wi = LocalToRender(bs->wi);
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

    template <typename BxDF>
    PBRT_CPU_GPU pstd::optional<BSDFSample> Sample_f(
        Vector3f woRender, Float u, Point2f u2,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = RenderToLocal(woRender);
        if (wo.z == 0)
            return {};

        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        if (!(specificBxDF->Flags() & sampleFlags))
            return {};

        pstd::optional<BSDFSample> bs =
            specificBxDF->Sample_f(wo, u, u2, mode, sampleFlags);
        if (!bs || !bs->f || bs->pdf == 0 || bs->wi.z == 0)
            return {};
        DCHECK_GT(bs->pdf, 0);

        PBRT_DBG("For wo = (%f, %f, %f), ng %f %f %f ns %f %f %f "
                 "sampled f = %f %f %f %f, pdf = %f, ratio[0] = %f "
                 "wi = (%f, %f, %f)\n",
                 wo.x, wo.y, wo.z, ng.x, ng.y, ng.z, shadingFrame.z.x, shadingFrame.z.y,
                 shadingFrame.z.z, bs->f[0], bs->f[1], bs->f[2], bs->f[3], bs->pdf,
                 (bs->pdf > 0) ? (bs->f[0] / bs->pdf) : 0, bs->wi.x, bs->wi.y, bs->wi.z);

        bs->wi = LocalToRender(bs->wi);

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

  private:
    // BSDF Private Members
    BxDF bxdf;
    Normal3f ng;
    Frame shadingFrame;
};

}  // namespace pbrt

#endif  // PBRT_BSDF_H
