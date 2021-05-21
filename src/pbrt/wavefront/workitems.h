// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_WORKITEMS_H
#define PBRT_WAVEFRONT_WORKITEMS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/sampler.h>
#include <pbrt/film.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/materials.h>
#include <pbrt/ray.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/soa.h>
#include <pbrt/wavefront/workqueue.h>

namespace pbrt {

// RaySamples Definition
struct RaySamples {
    // RaySamples Public Members
    struct {
        Point2f u;
        Float uc;
    } direct;
    struct {
        Float uc, rr;
        Point2f u;
    } indirect;
    bool haveSubsurface;
    struct {
        Float uc;
        Point2f u;
    } subsurface;
    bool haveMedia;
    struct {
        Float uDist, uMode;
    } media;
};

template <>
struct SOA<RaySamples> {
  public:
    SOA() = default;

    SOA(int size, Allocator alloc) {
        direct = alloc.allocate_object<Float4>(size);
        indirect = alloc.allocate_object<Float4>(size);
        subsurface = alloc.allocate_object<Float4>(size);
        mediaDist = alloc.allocate_object<Float>(size);
        mediaMode = alloc.allocate_object<Float>(size);
    }

    PBRT_CPU_GPU
    RaySamples operator[](int i) const {
        RaySamples rs;
        Float4 dir = Load4(direct + i);
        rs.direct.u = Point2f(dir.v[0], dir.v[1]);
        rs.direct.uc = dir.v[2];

        rs.haveSubsurface = int(dir.v[3]) & 1;
        rs.haveMedia = int(dir.v[3]) & 2;

        Float4 ind = Load4(indirect + i);
        rs.indirect.uc = ind.v[0];
        rs.indirect.rr = ind.v[1];
        rs.indirect.u = Point2f(ind.v[2], ind.v[3]);

        if (rs.haveSubsurface) {
            Float4 ss = Load4(subsurface + i);
            rs.subsurface.uc = ss.v[0];
            rs.subsurface.u = Point2f(ss.v[1], ss.v[2]);
        }

        if (rs.haveMedia) {
            rs.media.uDist = mediaDist[i];
            rs.media.uMode = mediaMode[i];
        }

        return rs;
    }

    struct GetSetIndirector {
        PBRT_CPU_GPU
        operator RaySamples() const { return (*(const SOA *)soa)[index]; }

        PBRT_CPU_GPU
        void operator=(RaySamples rs) {
            int flags = (rs.haveSubsurface ? 1 : 0) | (rs.haveMedia ? 2 : 0);
            soa->direct[index] =
                Float4{rs.direct.u[0], rs.direct.u[1], rs.direct.uc, Float(flags)};
            soa->indirect[index] = Float4{rs.indirect.uc, rs.indirect.rr,
                                          rs.indirect.u[0], rs.indirect.u[1]};
            if (rs.haveSubsurface)
                soa->subsurface[index] =
                    Float4{rs.subsurface.uc, rs.subsurface.u.x, rs.subsurface.u.y, 0.f};
            if (rs.haveMedia) {
                soa->mediaDist[index] = rs.media.uDist;
                soa->mediaMode[index] = rs.media.uMode;
            }
        }

        SOA *soa;
        int index;
    };

    PBRT_CPU_GPU
    GetSetIndirector operator[](int i) { return GetSetIndirector{this, i}; }

  private:
    Float4 *PBRT_RESTRICT direct;
    Float4 *PBRT_RESTRICT indirect;
    Float4 *PBRT_RESTRICT subsurface;
    Float *PBRT_RESTRICT mediaDist, *PBRT_RESTRICT mediaMode;
};

// PixelSampleState Definition
struct PixelSampleState {
    // PixelSampleState Public Members
    Point2i pPixel;
    SampledSpectrum L;
    SampledWavelengths lambda;
    Float filterWeight;
    VisibleSurface visibleSurface;
    SampledSpectrum cameraRayWeight;
    RaySamples samples;
};

// RayWorkItem Definition
struct RayWorkItem {
    // RayWorkItem Public Members
    Ray ray;
    int depth;
    SampledWavelengths lambda;
    int pixelIndex;
    SampledSpectrum T_hat, uniPathPDF, lightPathPDF;
    LightSampleContext prevIntrCtx;
    Float etaScale;
    int isSpecularBounce;
    int anyNonSpecularBounces;
};

// EscapedRayWorkItem Definition
struct EscapedRayWorkItem {
    // EscapedRayWorkItem Public Members
    Point3f rayo;
    Vector3f rayd;
    int depth;
    SampledWavelengths lambda;
    int pixelIndex;
    SampledSpectrum T_hat;
    int specularBounce;
    SampledSpectrum uniPathPDF, lightPathPDF;
    LightSampleContext prevIntrCtx;
};

// HitAreaLightWorkItem Definition
struct HitAreaLightWorkItem {
    // HitAreaLightWorkItem Public Members
    Light areaLight;
    Point3f p;
    Normal3f n;
    Point2f uv;
    Vector3f wo;
    SampledWavelengths lambda;
    int depth;
    SampledSpectrum T_hat, uniPathPDF, lightPathPDF;
    LightSampleContext prevIntrCtx;
    int isSpecularBounce;
    int pixelIndex;
};

// HitAreaLightQueue Definition
using HitAreaLightQueue = WorkQueue<HitAreaLightWorkItem>;

// ShadowRayWorkItem Definition
struct ShadowRayWorkItem {
    Ray ray;
    Float tMax;
    SampledWavelengths lambda;
    SampledSpectrum Ld, uniPathPDF, lightPathPDF;
    int pixelIndex;
};

// GetBSSRDFAndProbeRayWorkItem Definition
struct GetBSSRDFAndProbeRayWorkItem {
    PBRT_CPU_GPU
    MaterialEvalContext GetMaterialEvalContext() const {
        MaterialEvalContext ctx;
        ctx.wo = wo;
        ctx.n = n;
        ctx.ns = ns;
        ctx.dpdus = dpdus;
        ctx.p = p;
        ctx.uv = uv;
        return ctx;
    }

    Material material;
    SampledWavelengths lambda;
    SampledSpectrum T_hat, uniPathPDF;
    Point3f p;
    Vector3f wo;
    Normal3f n, ns;
    Vector3f dpdus;
    Point2f uv;
    int depth;
    MediumInterface mediumInterface;
    Float etaScale;
    int pixelIndex;
};

// SubsurfaceScatterWorkItem Definition
struct SubsurfaceScatterWorkItem {
    Point3f p0, p1;
    int depth;
    Material material;
    TabulatedBSSRDF bssrdf;
    SampledWavelengths lambda;
    SampledSpectrum T_hat, uniPathPDF;
    Float reservoirPDF;
    Float uLight;
    SubsurfaceInteraction ssi;
    MediumInterface mediumInterface;
    Float etaScale;
    int pixelIndex;
};

// MediumSampleWorkItem Definition
struct MediumSampleWorkItem {
    // Both enqueue types (have mtl and no hit)
    Ray ray;
    int depth;
    Float tMax;
    SampledWavelengths lambda;
    SampledSpectrum T_hat;
    SampledSpectrum uniPathPDF;
    SampledSpectrum lightPathPDF;
    int pixelIndex;
    LightSampleContext prevIntrCtx;
    int isSpecularBounce;
    int anyNonSpecularBounces;
    Float etaScale;

    // Have a hit material as well
    Light areaLight;
    Point3fi pi;
    Normal3f n;
    Vector3f dpdu, dpdv;
    Vector3f wo;
    Point2f uv;
    Material material;
    Normal3f ns;
    Vector3f dpdus, dpdvs;
    Normal3f dndus, dndvs;
    int faceIndex;
    MediumInterface mediumInterface;
};

// MediumScatterWorkItem Definition
template <typename PhaseFunction>
struct MediumScatterWorkItem {
    Point3f p;
    int depth;
    SampledWavelengths lambda;
    SampledSpectrum T_hat, uniPathPDF;
    const PhaseFunction *phase;
    Vector3f wo;
    Float time;
    Float etaScale;
    Medium medium;
    int pixelIndex;
};

// MaterialEvalWorkItem Definition
template <typename ConcreteMaterial>
struct MaterialEvalWorkItem {
    // MaterialEvalWorkItem Public Methods
    PBRT_CPU_GPU
    BumpEvalContext GetBumpEvalContext(Float dudx, Float dudy, Float dvdx,
                                       Float dvdy) const {
        BumpEvalContext ctx;
        ctx.p = Point3f(pi);
        ctx.uv = uv;
        ctx.dudx = dudx;
        ctx.dudy = dudy;
        ctx.dvdx = dvdx;
        ctx.dvdy = dvdy;
        ctx.shading.n = ns;
        ctx.shading.dpdu = dpdus;
        ctx.shading.dpdv = dpdvs;
        ctx.shading.dndu = dndus;
        ctx.shading.dndv = dndvs;
        ctx.faceIndex = faceIndex;
        return ctx;
    }

    PBRT_CPU_GPU
    MaterialEvalContext GetMaterialEvalContext(Float dudx, Float dudy, Float dvdx,
                                               Float dvdy, Normal3f ns,
                                               Vector3f dpdus) const {
        MaterialEvalContext ctx;
        ctx.wo = wo;
        ctx.n = n;
        ctx.ns = ns;
        ctx.dpdus = dpdus;
        ctx.p = Point3f(pi);
        ctx.uv = uv;
        ctx.dudx = dudx;
        ctx.dudy = dudy;
        ctx.dvdx = dvdx;
        ctx.dvdy = dvdy;
        ctx.faceIndex = faceIndex;
        return ctx;
    }

    // MaterialEvalWorkItem Public Members
    const ConcreteMaterial *material;
    Point3fi pi;
    Normal3f n;
    Vector3f dpdu, dpdv;
    Float time;
    int depth;
    Normal3f ns;
    Vector3f dpdus, dpdvs;
    Normal3f dndus, dndvs;
    Point2f uv;
    int faceIndex;
    SampledWavelengths lambda;
    int pixelIndex;
    int anyNonSpecularBounces;
    Vector3f wo;
    SampledSpectrum T_hat, uniPathPDF;
    Float etaScale;
    MediumInterface mediumInterface;
};

#include "wavefront_workitems_soa.h"

// RayQueue Definition
class RayQueue : public WorkQueue<RayWorkItem> {
  public:
    using WorkQueue::WorkQueue;
    // RayQueue Public Methods
    PBRT_CPU_GPU
    int PushCameraRay(const Ray &ray, const SampledWavelengths &lambda, int pixelIndex);

    PBRT_CPU_GPU
    int PushIndirectRay(const Ray &ray, int depth, const LightSampleContext &prevIntrCtx,
                        const SampledSpectrum &T_hat, const SampledSpectrum &uniPathPDF,
                        const SampledSpectrum &lightPathPDF,
                        const SampledWavelengths &lambda, Float etaScale,
                        bool isSpecularBounce, bool anyNonSpecularBounces,
                        int pixelIndex);
};

// RayQueue Inline Methods
inline int RayQueue::PushCameraRay(const Ray &ray, const SampledWavelengths &lambda,
                                   int pixelIndex) {
    int index = AllocateEntry();
    DCHECK(!ray.HasNaN());
    this->ray[index] = ray;
    this->depth[index] = 0;
    this->pixelIndex[index] = pixelIndex;
    this->lambda[index] = lambda;
    this->T_hat[index] = SampledSpectrum(1.f);
    this->etaScale[index] = 1.f;
    this->anyNonSpecularBounces[index] = false;
    this->uniPathPDF[index] = SampledSpectrum(1.f);
    this->lightPathPDF[index] = SampledSpectrum(1.f);
    this->isSpecularBounce[index] = false;
    return index;
}

PBRT_CPU_GPU
inline int RayQueue::PushIndirectRay(
    const Ray &ray, int depth, const LightSampleContext &prevIntrCtx,
    const SampledSpectrum &T_hat, const SampledSpectrum &uniPathPDF,
    const SampledSpectrum &lightPathPDF, const SampledWavelengths &lambda, Float etaScale,
    bool isSpecularBounce, bool anyNonSpecularBounces, int pixelIndex) {
    int index = AllocateEntry();
    DCHECK(!ray.HasNaN());
    this->ray[index] = ray;
    this->depth[index] = depth;
    this->pixelIndex[index] = pixelIndex;
    this->prevIntrCtx[index] = prevIntrCtx;
    this->T_hat[index] = T_hat;
    this->uniPathPDF[index] = uniPathPDF;
    this->lightPathPDF[index] = lightPathPDF;
    this->lambda[index] = lambda;
    this->anyNonSpecularBounces[index] = anyNonSpecularBounces;
    this->isSpecularBounce[index] = isSpecularBounce;
    this->etaScale[index] = etaScale;
    return index;
}

// ShadowRayQueue Definition
using ShadowRayQueue = WorkQueue<ShadowRayWorkItem>;

// EscapedRayQueue Definition
class EscapedRayQueue : public WorkQueue<EscapedRayWorkItem> {
  public:
    // EscapedRayQueue Public Methods
    PBRT_CPU_GPU
    int Push(RayWorkItem r);

    using WorkQueue::WorkQueue;

    using WorkQueue::Push;
};

inline int EscapedRayQueue::Push(RayWorkItem r) {
    return Push(EscapedRayWorkItem{r.ray.o, r.ray.d, r.depth, r.lambda, r.pixelIndex,
                                   r.T_hat, (int)r.isSpecularBounce, r.uniPathPDF,
                                   r.lightPathPDF, r.prevIntrCtx});
}

// GetBSSRDFAndProbeRayQueue Definition
class GetBSSRDFAndProbeRayQueue : public WorkQueue<GetBSSRDFAndProbeRayWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    PBRT_CPU_GPU
    int Push(Material material, SampledWavelengths lambda, SampledSpectrum T_hat,
             SampledSpectrum uniPathPDF, Point3f p, Vector3f wo, Normal3f n, Normal3f ns,
             Vector3f dpdus, Point2f uv, int depth, MediumInterface mediumInterface,
             Float etaScale, int pixelIndex) {
        int index = AllocateEntry();
        this->material[index] = material;
        this->lambda[index] = lambda;
        this->T_hat[index] = T_hat;
        this->uniPathPDF[index] = uniPathPDF;
        this->p[index] = p;
        this->wo[index] = wo;
        this->n[index] = n;
        this->ns[index] = ns;
        this->dpdus[index] = dpdus;
        this->uv[index] = uv;
        this->depth[index] = depth;
        this->mediumInterface[index] = mediumInterface;
        this->etaScale[index] = etaScale;
        this->pixelIndex[index] = pixelIndex;
        return index;
    }
};

// SubsurfaceScatterQueue Definition
class SubsurfaceScatterQueue : public WorkQueue<SubsurfaceScatterWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    PBRT_CPU_GPU
    int Push(Point3f p0, Point3f p1, int depth, Material material, TabulatedBSSRDF bssrdf,
             SampledWavelengths lambda, SampledSpectrum T_hat, SampledSpectrum uniPathPDF,
             MediumInterface mediumInterface, Float etaScale, int pixelIndex) {
        int index = AllocateEntry();
        this->p0[index] = p0;
        this->p1[index] = p1;
        this->depth[index] = depth;
        this->material[index] = material;
        this->bssrdf[index] = bssrdf;
        this->lambda[index] = lambda;
        this->T_hat[index] = T_hat;
        this->uniPathPDF[index] = uniPathPDF;
        this->mediumInterface[index] = mediumInterface;
        this->etaScale[index] = etaScale;
        this->pixelIndex[index] = pixelIndex;
        return index;
    }
};

// MediumSampleQueue Definition
class MediumSampleQueue : public WorkQueue<MediumSampleWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    using WorkQueue::Push;

    PBRT_CPU_GPU
    int Push(Ray ray, Float tMax, SampledWavelengths lambda, SampledSpectrum T_hat,
             SampledSpectrum uniPathPDF, SampledSpectrum lightPathPDF, int pixelIndex,
             LightSampleContext prevIntrCtx, int isSpecularBounce,
             int anyNonSpecularBounces, Float etaScale) {
        int index = AllocateEntry();
        this->ray[index] = ray;
        this->tMax[index] = tMax;
        this->lambda[index] = lambda;
        this->T_hat[index] = T_hat;
        this->uniPathPDF[index] = uniPathPDF;
        this->lightPathPDF[index] = lightPathPDF;
        this->pixelIndex[index] = pixelIndex;
        this->prevIntrCtx[index] = prevIntrCtx;
        this->isSpecularBounce[index] = isSpecularBounce;
        this->anyNonSpecularBounces[index] = anyNonSpecularBounces;
        this->etaScale[index] = etaScale;
        return index;
    }

    PBRT_CPU_GPU
    int Push(RayWorkItem r, Float tMax) {
        return Push(r.ray, tMax, r.lambda, r.T_hat, r.uniPathPDF, r.lightPathPDF,
                    r.pixelIndex, r.prevIntrCtx, r.isSpecularBounce,
                    r.anyNonSpecularBounces, r.etaScale);
    }
};

// MediumScatterQueue Definition
using MediumScatterQueue = MultiWorkQueue<
    typename MapType<MediumScatterWorkItem, typename PhaseFunction::Types>::type>;

// MaterialEvalQueue Definition
using MaterialEvalQueue = MultiWorkQueue<
    typename MapType<MaterialEvalWorkItem, typename Material::Types>::type>;

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_WORKITEMS_H
