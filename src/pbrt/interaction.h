// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_INTERACTION_H
#define PBRT_INTERACTION_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bssrdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/sampler.h>
#include <pbrt/ray.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <limits>

namespace pbrt {

// Interaction Definition
class Interaction {
  public:
    // Interaction Public Methods
    Interaction() = default;

    PBRT_CPU_GPU
    Interaction(const Point3fi &pi, const Normal3f &n, const Point2f &uv,
                const Vector3f &wo, Float time)
        : pi(pi), n(n), uv(uv), wo(Normalize(wo)), time(time) {}

    PBRT_CPU_GPU
    Point3f p() const { return Point3f(pi); }

    PBRT_CPU_GPU
    bool IsSurfaceInteraction() const { return n != Normal3f(0, 0, 0); }
    PBRT_CPU_GPU
    bool IsMediumInteraction() const { return !IsSurfaceInteraction(); }

    PBRT_CPU_GPU
    const SurfaceInteraction &AsSurface() const {
        CHECK(IsSurfaceInteraction());
        return (const SurfaceInteraction &)*this;
    }
    PBRT_CPU_GPU
    SurfaceInteraction &AsSurface() {
        CHECK(IsSurfaceInteraction());
        return (SurfaceInteraction &)*this;
    }

    // used by medium ctor
    PBRT_CPU_GPU
    Interaction(const Point3f &p, const Vector3f &wo, Float time, MediumHandle medium)
        : pi(p), time(time), wo(wo), medium(medium) {}
    PBRT_CPU_GPU
    Interaction(const Point3f &p, const Normal3f &n, Float time, MediumHandle medium)
        : pi(p), n(n), time(time), medium(medium) {}
    PBRT_CPU_GPU
    Interaction(const Point3fi &pi, const Normal3f &n, Float time = 0,
                const Point2f &uv = {})
        : pi(pi), n(n), uv(uv), time(time) {}
    PBRT_CPU_GPU
    Interaction(const Point3fi &pi, const Normal3f &n, const Point2f &uv)
        : pi(pi), n(n), uv(uv) {}
    PBRT_CPU_GPU
    Interaction(const Point3f &p, Float time, MediumHandle medium)
        : pi(p), time(time), medium(medium) {}
    PBRT_CPU_GPU
    Interaction(const Point3f &p, const MediumInterface *mediumInterface)
        : pi(p), mediumInterface(mediumInterface) {}
    PBRT_CPU_GPU
    Interaction(const Point3f &p, Float time, const MediumInterface *mediumInterface)
        : pi(p), time(time), mediumInterface(mediumInterface) {}
    PBRT_CPU_GPU
    const MediumInteraction &AsMedium() const {
        CHECK(IsMediumInteraction());
        return (const MediumInteraction &)*this;
    }
    PBRT_CPU_GPU
    MediumInteraction &AsMedium() {
        CHECK(IsMediumInteraction());
        return (MediumInteraction &)*this;
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    Point3f OffsetRayOrigin(const Vector3f &w) const {
        return pbrt::OffsetRayOrigin(pi, n, w);
    }

    PBRT_CPU_GPU
    Point3f OffsetRayOrigin(const Point3f &pt) const { return OffsetRayOrigin(pt - p()); }

    PBRT_CPU_GPU
    RayDifferential SpawnRay(const Vector3f &d) const {
        return RayDifferential(OffsetRayOrigin(d), d, time, GetMedium(d));
    }

    PBRT_CPU_GPU
    Ray SpawnRayTo(const Point3f &p2) const {
        Ray r = pbrt::SpawnRayTo(pi, n, time, p2);
        r.medium = GetMedium(r.d);
        return r;
    }

    PBRT_CPU_GPU
    Ray SpawnRayTo(const Interaction &it) const {
        Ray r = pbrt::SpawnRayTo(pi, n, time, it.pi, it.n);
        r.medium = GetMedium(r.d);
        return r;
    }

    PBRT_CPU_GPU
    MediumHandle GetMedium(const Vector3f &w) const {
        if (mediumInterface != nullptr)
            return Dot(w, n) > 0 ? mediumInterface->outside : mediumInterface->inside;
        return medium;
    }

    PBRT_CPU_GPU
    MediumHandle GetMedium() const {
        if (mediumInterface)
            DCHECK_EQ(mediumInterface->inside, mediumInterface->outside);
        return mediumInterface ? mediumInterface->inside : medium;
    }

    // Interaction Public Members
    Point3fi pi;
    Float time = 0;
    Vector3f wo;
    Normal3f n;
    Point2f uv;
    const MediumInterface *mediumInterface = nullptr;
    MediumHandle medium = nullptr;
};

// MediumInteraction Definition
class MediumInteraction : public Interaction {
  public:
    // MediumInteraction Public Methods
    PBRT_CPU_GPU
    MediumInteraction() : phase(nullptr) {}
    PBRT_CPU_GPU
    MediumInteraction(Point3f p, Vector3f wo, Float time, SampledSpectrum sigma_a,
                      SampledSpectrum sigma_s, SampledSpectrum sigma_maj,
                      SampledSpectrum Le, MediumHandle medium, PhaseFunctionHandle phase)
        : Interaction(p, wo, time, medium),
          phase(phase),
          sigma_a(sigma_a),
          sigma_s(sigma_s),
          sigma_maj(sigma_maj),
          Le(Le) {}

    PBRT_CPU_GPU
    SampledSpectrum sigma_n() const {
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        SampledSpectrum sigma_n = sigma_maj - sigma_t;
        CHECK_RARE(1e-5, sigma_n.MinComponentValue() < 0);
        return ClampZero(sigma_n);
    }

    std::string ToString() const;

    // MediumInteraction Public Members
    PhaseFunctionHandle phase;
    SampledSpectrum sigma_a, sigma_s, sigma_maj, Le;
};

// SurfaceInteraction Definition
class SurfaceInteraction : public Interaction {
  public:
    // SurfaceInteraction Public Methods
    SurfaceInteraction() = default;

    PBRT_CPU_GPU
    SurfaceInteraction(const Point3fi &pi, const Point2f &uv, const Vector3f &wo,
                       const Vector3f &dpdu, const Vector3f &dpdv, const Normal3f &dndu,
                       const Normal3f &dndv, Float time, bool flipNormal)
        : Interaction(pi, Normal3f(Normalize(Cross(dpdu, dpdv))), uv, wo, time),
          dpdu(dpdu),
          dpdv(dpdv),
          dndu(dndu),
          dndv(dndv) {
        // Initialize shading geometry from true geometry
        shading.n = n;
        shading.dpdu = dpdu;
        shading.dpdv = dpdv;
        shading.dndu = dndu;
        shading.dndv = dndv;

        // Adjust normal based on orientation and handedness
        if (flipNormal) {
            n *= -1;
            shading.n *= -1;
        }
    }

    PBRT_CPU_GPU
    SurfaceInteraction(const Point3fi &pi, const Point2f &uv, const Vector3f &wo,
                       const Vector3f &dpdu, const Vector3f &dpdv, const Normal3f &dndu,
                       const Normal3f &dndv, Float time, bool flipNormal, int faceIndex)
        : SurfaceInteraction(pi, uv, wo, dpdu, dpdv, dndu, dndv, time, flipNormal) {
        this->faceIndex = faceIndex;
    }

    PBRT_CPU_GPU
    void SetShadingGeometry(const Normal3f &ns, const Vector3f &dpdus,
                            const Vector3f &dpdvs, const Normal3f &dndus,
                            const Normal3f &dndvs, bool orientationIsAuthoritative) {
        // Compute _shading.n_ for _SurfaceInteraction_
        shading.n = ns;
        DCHECK_NE(shading.n, Normal3f(0, 0, 0));
        if (orientationIsAuthoritative)
            n = FaceForward(n, shading.n);
        else
            shading.n = FaceForward(shading.n, n);

        // Initialize _shading_ partial derivative values
        shading.dpdu = dpdus;
        shading.dpdv = dpdvs;
        shading.dndu = dndus;
        shading.dndv = dndvs;
        while (LengthSquared(shading.dpdu) > 1e16f ||
               LengthSquared(shading.dpdv) > 1e16f) {
            shading.dpdu /= 1e8f;
            shading.dpdv /= 1e8f;
        }
    }

    std::string ToString() const;

    void SetIntersectionProperties(MaterialHandle mtl, LightHandle area,
                                   const MediumInterface *primitiveMediumInterface,
                                   MediumHandle rayMedium) {
        material = mtl;
        areaLight = area;
        CHECK_GE(Dot(n, shading.n), 0.);
        // Set medium properties at surface intersection
        if (primitiveMediumInterface && primitiveMediumInterface->IsMediumTransition())
            mediumInterface = primitiveMediumInterface;
        else
            medium = rayMedium;
    }

    PBRT_CPU_GPU
    void ComputeDifferentials(const RayDifferential &r, CameraHandle camera,
                              int samplesPerPixel);

    PBRT_CPU_GPU
    void SkipIntersection(RayDifferential *ray, Float t) const;

    using Interaction::SpawnRay;
    PBRT_CPU_GPU
    RayDifferential SpawnRay(const RayDifferential &rayi, const BSDF &bsdf, Vector3f wi,
                             int /*BxDFFlags*/ flags, Float eta) const;

    PBRT_CPU_GPU
    BSDF GetBSDF(const RayDifferential &ray, SampledWavelengths &lambda,
                 CameraHandle camera, ScratchBuffer &scratchBuffer,
                 SamplerHandle sampler);
    PBRT_CPU_GPU
    BSSRDFHandle GetBSSRDF(const RayDifferential &ray, SampledWavelengths &lambda,
                           CameraHandle camera, ScratchBuffer &scratchBuffer);

    PBRT_CPU_GPU
    SampledSpectrum Le(Vector3f w, const SampledWavelengths &lambda) const;

    // SurfaceInteraction Public Members
    Vector3f dpdu, dpdv;
    Normal3f dndu, dndv;
    struct {
        Normal3f n;
        Vector3f dpdu, dpdv;
        Normal3f dndu, dndv;
    } shading;
    int faceIndex = 0;
    MaterialHandle material;
    LightHandle areaLight;
    Vector3f dpdx, dpdy;
    Float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;
};

}  // namespace pbrt

#endif  // PBRT_INTERACTION_H
