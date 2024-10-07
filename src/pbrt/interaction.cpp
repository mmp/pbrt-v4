// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/interaction.h>

#include <pbrt/base/camera.h>
#include <pbrt/cameras.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>

#include <cmath>

namespace pbrt {

std::string Interaction::ToString() const {
    return StringPrintf(
        "[ Interaction pi: %s n: %s uv: %s wo: %s time: %s "
        "medium: %s mediumInterface: %s ]",
        pi, n, uv, wo, time, medium ? medium.ToString().c_str() : "(nullptr)",
        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)");
}

std::string MediumInteraction::ToString() const {
    return StringPrintf(
        "[ MediumInteraction pi: %s n: %s uv: %s wo: %s time: %s "
        "medium: %s mediumInterface: %s phase: %s ]",
        pi, n, uv, wo, time, medium ? medium.ToString().c_str() : "(nullptr)",
        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)",
        phase ? phase.ToString().c_str() : "(nullptr)");
}

// SurfaceInteraction Method Definitions
PBRT_CPU_GPU void SurfaceInteraction::ComputeDifferentials(const RayDifferential &ray, Camera camera,
                                              int samplesPerPixel) {
    if (GetOptions().disableTextureFiltering) {
        dudx = dudy = dvdx = dvdy = 0;
        dpdx = dpdy = Vector3f(0, 0, 0);
        return;
    }
    if (ray.hasDifferentials && Dot(n, ray.rxDirection) != 0 &&
        Dot(n, ray.ryDirection) != 0) {
        // Estimate screen-space change in $\pt{}$ using ray differentials
        // Compute auxiliary intersection points with plane, _px_ and _py_
        Float d = -Dot(n, Vector3f(p()));
        Float tx = (-Dot(n, Vector3f(ray.rxOrigin)) - d) / Dot(n, ray.rxDirection);
        DCHECK(!IsInf(tx) && !IsNaN(tx));
        Point3f px = ray.rxOrigin + tx * ray.rxDirection;
        Float ty = (-Dot(n, Vector3f(ray.ryOrigin)) - d) / Dot(n, ray.ryDirection);
        DCHECK(!IsInf(ty) && !IsNaN(ty));
        Point3f py = ray.ryOrigin + ty * ray.ryDirection;

        dpdx = px - p();
        dpdy = py - p();

    } else {
        // Approximate screen-space change in $\pt{}$ based on camera projection
        camera.Approximate_dp_dxy(p(), n, time, samplesPerPixel, &dpdx, &dpdy);
    }
    // Estimate screen-space change in $(u,v)$
    // Compute $\transpose{\XFORM{A}} \XFORM{A}$ and its determinant
    Float ata00 = Dot(dpdu, dpdu), ata01 = Dot(dpdu, dpdv);
    Float ata11 = Dot(dpdv, dpdv);
    Float invDet = 1 / DifferenceOfProducts(ata00, ata11, ata01, ata01);
    invDet = IsFinite(invDet) ? invDet : 0.f;

    // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
    Float atb0x = Dot(dpdu, dpdx), atb1x = Dot(dpdv, dpdx);
    Float atb0y = Dot(dpdu, dpdy), atb1y = Dot(dpdv, dpdy);

    // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
    dudx = DifferenceOfProducts(ata11, atb0x, ata01, atb1x) * invDet;
    dvdx = DifferenceOfProducts(ata00, atb1x, ata01, atb0x) * invDet;
    dudy = DifferenceOfProducts(ata11, atb0y, ata01, atb1y) * invDet;
    dvdy = DifferenceOfProducts(ata00, atb1y, ata01, atb0y) * invDet;

    // Clamp derivatives of $u$ and $v$ to reasonable values
    dudx = IsFinite(dudx) ? Clamp(dudx, -1e8f, 1e8f) : 0.f;
    dvdx = IsFinite(dvdx) ? Clamp(dvdx, -1e8f, 1e8f) : 0.f;
    dudy = IsFinite(dudy) ? Clamp(dudy, -1e8f, 1e8f) : 0.f;
    dvdy = IsFinite(dvdy) ? Clamp(dvdy, -1e8f, 1e8f) : 0.f;
}

PBRT_CPU_GPU void SurfaceInteraction::SkipIntersection(RayDifferential *ray, Float t) const {
    *((Ray *)ray) = SpawnRay(ray->d);
    if (ray->hasDifferentials) {
        ray->rxOrigin = ray->rxOrigin + t * ray->rxDirection;
        ray->ryOrigin = ray->ryOrigin + t * ray->ryDirection;
    }
}

PBRT_CPU_GPU RayDifferential SurfaceInteraction::SpawnRay(const RayDifferential &rayi,
                                             const BSDF &bsdf, Vector3f wi, int flags,
                                             Float eta) const {
    RayDifferential rd(SpawnRay(wi));
    if (rayi.hasDifferentials) {
        // Compute ray differentials for specular reflection or transmission
        // Compute common factors for specular ray differentials
        Normal3f n = shading.n;
        Normal3f dndx = shading.dndu * dudx + shading.dndv * dvdx;
        Normal3f dndy = shading.dndu * dudy + shading.dndv * dvdy;
        Vector3f dwodx = -rayi.rxDirection - wo, dwody = -rayi.ryDirection - wo;

        if (flags == BxDFFlags::SpecularReflection) {
            // Initialize origins of specular differential rays
            rd.hasDifferentials = true;
            rd.rxOrigin = p() + dpdx;
            rd.ryOrigin = p() + dpdy;

            // Compute differential reflected directions
            Float dwoDotn_dx = Dot(dwodx, n) + Dot(wo, dndx);
            Float dwoDotn_dy = Dot(dwody, n) + Dot(wo, dndy);
            rd.rxDirection =
                wi - dwodx + 2 * Vector3f(Dot(wo, n) * dndx + dwoDotn_dx * n);
            rd.ryDirection =
                wi - dwody + 2 * Vector3f(Dot(wo, n) * dndy + dwoDotn_dy * n);

        } else if (flags == BxDFFlags::SpecularTransmission) {
            // Initialize origins of specular differential rays
            rd.hasDifferentials = true;
            rd.rxOrigin = p() + dpdx;
            rd.ryOrigin = p() + dpdy;

            // Compute differential transmitted directions
            // Find oriented surface normal for transmission
            if (Dot(wo, n) < 0) {
                n = -n;
                dndx = -dndx;
                dndy = -dndy;
            }

            // Compute partial derivatives of $\mu$
            Float dwoDotn_dx = Dot(dwodx, n) + Dot(wo, dndx);
            Float dwoDotn_dy = Dot(dwody, n) + Dot(wo, dndy);
            Float mu = Dot(wo, n) / eta - AbsDot(wi, n);
            Float dmudx = dwoDotn_dx * (1 / eta + 1 / Sqr(eta) * Dot(wo, n) / Dot(wi, n));
            Float dmudy = dwoDotn_dy * (1 / eta + 1 / Sqr(eta) * Dot(wo, n) / Dot(wi, n));

            rd.rxDirection = wi - eta * dwodx + Vector3f(mu * dndx + dmudx * n);
            rd.ryDirection = wi - eta * dwody + Vector3f(mu * dndy + dmudy * n);
        }
    }
    // Squash potentially troublesome differentials
    if (LengthSquared(rd.rxDirection) > 1e16f || LengthSquared(rd.ryDirection) > 1e16f ||
        LengthSquared(Vector3f(rd.rxOrigin)) > 1e16f ||
        LengthSquared(Vector3f(rd.ryOrigin)) > 1e16f)
        rd.hasDifferentials = false;

    return rd;
}

BSDF SurfaceInteraction::GetBSDF(const RayDifferential &ray, SampledWavelengths &lambda,
                                 Camera camera, ScratchBuffer &scratchBuffer,
                                 Sampler sampler) {
    // Estimate $(u,v)$ and position differentials at intersection point
    ComputeDifferentials(ray, camera, sampler.SamplesPerPixel());

    // Resolve _MixMaterial_ if necessary
    while (material.Is<MixMaterial>()) {
        MixMaterial *mix = material.Cast<MixMaterial>();
        material = mix->ChooseMaterial(UniversalTextureEvaluator(), *this);
    }

    // Return unset _BSDF_ if surface has a null material
    if (!material)
        return {};

    // Evaluate normal or bump map, if present
    FloatTexture displacement = material.GetDisplacement();
    const Image *normalMap = material.GetNormalMap();
    if (displacement || normalMap) {
        // Get shading $\dpdu$ and $\dpdv$ using normal or bump map
        Vector3f dpdu, dpdv;
        if (normalMap)
            NormalMap(*normalMap, *this, &dpdu, &dpdv);
        else
            BumpMap(UniversalTextureEvaluator(), displacement, *this, &dpdu, &dpdv);

        Normal3f ns(Normalize(Cross(dpdu, dpdv)));
        SetShadingGeometry(ns, dpdu, dpdv, shading.dndu, shading.dndv, false);
    }

    // Return BSDF for surface interaction
    BSDF bsdf =
        material.GetBSDF(UniversalTextureEvaluator(), *this, lambda, scratchBuffer);
    if (bsdf && GetOptions().forceDiffuse) {
        // Override _bsdf_ with diffuse equivalent
        SampledSpectrum r = bsdf.rho(wo, {sampler.Get1D()}, {sampler.Get2D()});
        bsdf = BSDF(shading.n, shading.dpdu, scratchBuffer.Alloc<DiffuseBxDF>(r));
    }
    return bsdf;
}

BSSRDF SurfaceInteraction::GetBSSRDF(const RayDifferential &ray,
                                     SampledWavelengths &lambda, Camera camera,
                                     ScratchBuffer &scratchBuffer) {
    // Resolve _MixMaterial_ if necessary
    while (material.Is<MixMaterial>()) {
        MixMaterial *mix = material.Cast<MixMaterial>();
        material = mix->ChooseMaterial(UniversalTextureEvaluator(), *this);
    }

    return material.GetBSSRDF(UniversalTextureEvaluator(), *this, lambda, scratchBuffer);
}

PBRT_CPU_GPU SampledSpectrum SurfaceInteraction::Le(Vector3f w,
                                       const SampledWavelengths &lambda) const {
    return areaLight ? areaLight.L(p(), n, uv, w, lambda) : SampledSpectrum(0.f);
}

std::string SurfaceInteraction::ToString() const {
    return StringPrintf(
        "[ SurfaceInteraction pi: %s n: %s uv: %s wo: %s time: %s "
        "medium: %s mediumInterface: %s dpdu: %s dpdv: %s dndu: %s dndv: %s "
        "shading.n: %s shading.dpdu: %s shading.dpdv: %s "
        "shading.dndu: %s shading.dndv: %s material: %s "
        "areaLight: %s dpdx: %s dpdy: %s dudx: %f dvdx: %f "
        "dudy: %f dvdy: %f faceIndex: %d ]",
        pi, n, uv, wo, time, medium ? medium.ToString().c_str() : "(nullptr)",
        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)", dpdu, dpdv,
        dndu, dndv, shading.n, shading.dpdu, shading.dpdv, shading.dndu, shading.dndv,
        material ? material.ToString().c_str() : "(nullptr)",
        areaLight ? areaLight.ToString().c_str() : "(nullptr)", dpdx, dpdy, dudx, dvdx,
        dudy, dvdy, faceIndex);
}

}  // namespace pbrt
