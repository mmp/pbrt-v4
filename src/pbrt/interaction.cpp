// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/interaction.h>

#include <pbrt/base/camera.h>
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
        "sigma_a: %s sigma_s: %s sigma_maj: %s Le: %s medium: %s mediumInterface: %s "
        "phase: %s ]",
        pi, n, uv, wo, time, sigma_a, sigma_s, sigma_maj, Le,
        medium ? medium.ToString().c_str() : "(nullptr)",
        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)",
        phase ? phase.ToString().c_str() : "(nullptr)");
}

// SurfaceInteraction Method Definitions
void SurfaceInteraction::ComputeDifferentials(const RayDifferential &ray,
                                              CameraHandle camera, int samplesPerPixel) {
    if (ray.hasDifferentials && Dot(n, ray.rxDirection) != 0 &&
        Dot(n, ray.ryDirection) != 0) {
        // Estimate screen-space change in $\pt{}$
        // Compute auxiliary intersection points with plane, _px_ and _py_
        Float d = -Dot(n, Vector3f(p()));
        Float tx = (-Dot(n, Vector3f(ray.rxOrigin)) - d) / Dot(n, ray.rxDirection);
        CHECK(!IsInf(tx) && !IsNaN(tx));
        Point3f px = ray.rxOrigin + tx * ray.rxDirection;
        Float ty = (-Dot(n, Vector3f(ray.ryOrigin)) - d) / Dot(n, ray.ryDirection);
        CHECK(!IsInf(ty) && !IsNaN(ty));
        Point3f py = ray.ryOrigin + ty * ray.ryDirection;

        dpdx = px - p();
        dpdy = py - p();

    } else
        camera.ApproximatedPdxy(*this, samplesPerPixel);
    // Estimate screen-space change in $(u,v)$
    Float a00 = Dot(dpdu, dpdu), a01 = Dot(dpdu, dpdv), a11 = Dot(dpdv, dpdv);
    Float invDet = 1 / (DifferenceOfProducts(a00, a11, a01, a01));

    Float b0x = Dot(dpdu, dpdx), b1x = Dot(dpdv, dpdx);
    Float b0y = Dot(dpdu, dpdy), b1y = Dot(dpdv, dpdy);

    /* Set the UV partials to zero if dpdu and/or dpdv == 0 */
    invDet = std::isfinite(invDet) ? invDet : 0.f;

    dudx = DifferenceOfProducts(a11, b0x, a01, b1x) * invDet;
    dvdx = DifferenceOfProducts(a00, b1x, a01, b0x) * invDet;

    dudy = DifferenceOfProducts(a11, b0y, a01, b1y) * invDet;
    dvdy = DifferenceOfProducts(a00, b1y, a01, b0y) * invDet;

    dudx = std::isfinite(dudx) ? Clamp(dudx, -1e8f, 1e8f) : 0.f;
    dvdx = std::isfinite(dvdx) ? Clamp(dvdx, -1e8f, 1e8f) : 0.f;
    dudy = std::isfinite(dudy) ? Clamp(dudy, -1e8f, 1e8f) : 0.f;
    dvdy = std::isfinite(dvdy) ? Clamp(dvdy, -1e8f, 1e8f) : 0.f;
}

void SurfaceInteraction::SkipIntersection(RayDifferential *ray, Float t) const {
    *((Ray *)ray) = SpawnRay(ray->d);
    if (ray->hasDifferentials) {
        ray->rxOrigin = ray->rxOrigin + t * ray->rxDirection;
        ray->ryOrigin = ray->ryOrigin + t * ray->ryDirection;
    }
}

RayDifferential SurfaceInteraction::SpawnRay(const RayDifferential &rayi,
                                             const BSDF &bsdf, Vector3f wi, int flags,
                                             Float eta) const {
    RayDifferential rd(SpawnRay(wi));
    if (rayi.hasDifferentials) {
        // Compute ray differentials for specular reflection or transmission
        // Compute common factors for specular ray differentials
        Normal3f ns = shading.n;
        Normal3f dndx = shading.dndu * dudx + shading.dndv * dvdx;
        Normal3f dndy = shading.dndu * dudy + shading.dndv * dvdy;
        Vector3f dwodx = -rayi.rxDirection - wo, dwody = -rayi.ryDirection - wo;

        if (flags == BxDFFlags::SpecularReflection) {
            // Initialize origins of specular differential rays
            rd.hasDifferentials = true;
            rd.rxOrigin = p() + dpdx;
            rd.ryOrigin = p() + dpdy;

            // Compute differential reflected directions
            Float dwoDotNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dwoDotNdy = Dot(dwody, ns) + Dot(wo, dndy);
            rd.rxDirection =
                wi - dwodx + 2 * Vector3f(Dot(wo, ns) * dndx + dwoDotNdx * ns);
            rd.ryDirection =
                wi - dwody + 2 * Vector3f(Dot(wo, ns) * dndy + dwoDotNdy * ns);

        } else if (flags == BxDFFlags::SpecularTransmission) {
            // Initialize origins of specular differential rays
            rd.hasDifferentials = true;
            rd.rxOrigin = p() + dpdx;
            rd.ryOrigin = p() + dpdy;

            // Compute differential transmitted directions
            // Find _eta_ and oriented surface normal for transmission
            eta = 1 / eta;
            if (Dot(wo, ns) < 0) {
                ns = -ns;
                dndx = -dndx;
                dndy = -dndy;
            }

            // Compute partial derivatives of $\mu$
            Float dwoDotNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dwoDotNdy = Dot(dwody, ns) + Dot(wo, dndy);
            Float mu = eta * Dot(wo, ns) - AbsDot(wi, ns);
            Float dmudx = (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dwoDotNdx;
            Float dmudy = (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dwoDotNdy;

            rd.rxDirection = wi - eta * dwodx + Vector3f(mu * dndx + dmudx * ns);
            rd.ryDirection = wi - eta * dwody + Vector3f(mu * dndy + dmudy * ns);
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
                                 CameraHandle camera, ScratchBuffer &scratchBuffer,
                                 SamplerHandle sampler) {
    ComputeDifferentials(ray, camera, sampler.SamplesPerPixel());
    // Resolve _MixMaterial_ if necessary
    while (material.Is<MixMaterial>()) {
        MixMaterial *mix = material.CastOrNullptr<MixMaterial>();
        material = mix->ChooseMaterial(UniversalTextureEvaluator(), *this);
    }

    // Return unset _BSDF_ if surface has a null material
    if (!material)
        return {};

    // Evaluate bump map and compute shading normal
    FloatTextureHandle displacement = material.GetDisplacement();
    if (displacement) {
        Vector3f dpdu, dpdv;
        Bump(UniversalTextureEvaluator(), displacement, *this, &dpdu, &dpdv);
        SetShadingGeometry(Normal3f(Normalize(Cross(dpdu, dpdv))), dpdu, dpdv,
                           shading.dndu, shading.dndv, false);
    }

    // Return BSDF for surface interaction
    BSDF bsdf =
        material.GetBSDF(UniversalTextureEvaluator(), *this, lambda, scratchBuffer);
    if (bsdf && GetOptions().forceDiffuse) {
        // Override _bsdf_ with diffuse equivalent
        SampledSpectrum r = bsdf.rho(wo, {sampler.Get1D()}, {sampler.Get2D()});
        bsdf = BSDF(wo, n, shading.n, shading.dpdu,
                    scratchBuffer.Alloc<IdealDiffuseBxDF>(r));
    }
    return bsdf;
}

BSSRDFHandle SurfaceInteraction::GetBSSRDF(const RayDifferential &ray,
                                           SampledWavelengths &lambda,
                                           CameraHandle camera,
                                           ScratchBuffer &scratchBuffer) {
    // Resolve _MixMaterial_ if necessary
    while (material.Is<MixMaterial>()) {
        MixMaterial *mix = material.CastOrNullptr<MixMaterial>();
        material = mix->ChooseMaterial(UniversalTextureEvaluator(), *this);
    }

    return material.GetBSSRDF(UniversalTextureEvaluator(), *this, lambda, scratchBuffer);
}

SampledSpectrum SurfaceInteraction::Le(Vector3f w,
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
