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
BSDF SurfaceInteraction::GetBSDF(const RayDifferential &ray, SampledWavelengths &lambda,
                                 CameraHandle camera, ScratchBuffer &scratchBuffer,
                                 SamplerHandle sampler) {
    ComputeDifferentials(ray, camera);
    while (material.Is<MixMaterial>()) {
        MixMaterial *mix = material.CastOrNullptr<MixMaterial>();
        material = mix->ChooseMaterial(UniversalTextureEvaluator(), *this);
    }
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
        SampledSpectrum r = bsdf.rho(wo, {sampler.Get1D()}, {sampler.Get2D()});
        bsdf = BSDF(wo, n, shading.n, shading.dpdu,
                    scratchBuffer.Alloc<IdealDiffuseBxDF>(r), bsdf.eta);
    }
    return bsdf;
}

BSSRDFHandle SurfaceInteraction::GetBSSRDF(const RayDifferential &ray,
                                           SampledWavelengths &lambda,
                                           CameraHandle camera,
                                           ScratchBuffer &scratchBuffer) {
    while (material.Is<MixMaterial>()) {
        MixMaterial *mix = material.CastOrNullptr<MixMaterial>();
        material = mix->ChooseMaterial(UniversalTextureEvaluator(), *this);
    }
    return material.GetBSSRDF(UniversalTextureEvaluator(), *this, lambda, scratchBuffer);
}

void SurfaceInteraction::ComputeDifferentials(const RayDifferential &ray,
                                              CameraHandle camera) const {
    if (ray.hasDifferentials && Dot(n, ray.rxDirection) != 0 &&
        Dot(n, ray.ryDirection) != 0) {
        // Estimate screen-space change in $\pt{}$
        // Compute auxiliary intersection points with plane, _px_ and _py_
        Float d = -Dot(n, Vector3f(p()));
        Float tx = (-Dot(n, Vector3f(ray.rxOrigin)) - d) / Dot(n, ray.rxDirection);
        CHECK(!std::isinf(tx) && !std::isnan(tx));
        Point3f px = ray.rxOrigin + tx * ray.rxDirection;
        Float ty = (-Dot(n, Vector3f(ray.ryOrigin)) - d) / Dot(n, ray.ryDirection);
        CHECK(!std::isinf(ty) && !std::isnan(ty));
        Point3f py = ray.ryOrigin + ty * ray.ryDirection;

        dpdx = px - p();
        dpdy = py - p();

    } else
        camera.ApproximatedPdxy(*this);
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

RayDifferential SurfaceInteraction::SpawnRay(const RayDifferential &rayi,
                                             const BSDF &bsdf, const Vector3f &wi,
                                             BxDFFlags flags) const {
    RayDifferential rd(SpawnRay(wi));
    if (rayi.hasDifferentials) {
        // Compute ray differentials for specular reflection or transmission
        // Compute common factors for specular ray differentials
        Normal3f ns = shading.n;
        Normal3f dndx = shading.dndu * dudx + shading.dndv * dvdx;
        Normal3f dndy = shading.dndu * dudy + shading.dndv * dvdy;
        Vector3f dwodx = -rayi.rxDirection - wo, dwody = -rayi.ryDirection - wo;

        if (flags == (BxDFFlags::Reflection | BxDFFlags::Specular)) {
            // Initialize origins of specular differential rays
            rd.hasDifferentials = true;
            rd.rxOrigin = p() + dpdx;
            rd.ryOrigin = p() + dpdy;

            // Compute differential reflected directions
            Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);
            rd.rxDirection = wi - dwodx + 2.f * Vector3f(Dot(wo, ns) * dndx + dDNdx * ns);
            rd.ryDirection = wi - dwody + 2.f * Vector3f(Dot(wo, ns) * dndy + dDNdy * ns);

        } else if (flags == (BxDFFlags::Transmission | BxDFFlags::Specular)) {
            // Initialize origins of specular differential rays
            rd.hasDifferentials = true;
            rd.rxOrigin = p() + dpdx;
            rd.ryOrigin = p() + dpdy;

            // Compute differential transmitted directions
            // NOTE: eta coming in is now 1/eta from the derivation below, so
            // there's a 1/ here now...
            Float eta = 1 / bsdf.eta;
            if (Dot(wo, ns) < 0) {
                ns = -ns;
                dndx = -dndx;
                dndy = -dndy;
            }
            Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);
            // Compute partial derivatives of $\mu$
            Float mu = eta * Dot(wo, ns) - AbsDot(wi, ns);
            Float dmudx = (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdx;
            Float dmudy = (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdy;

            rd.rxDirection = wi - eta * dwodx + Vector3f(mu * dndx + dmudx * ns);
            rd.ryDirection = wi - eta * dwody + Vector3f(mu * dndy + dmudy * ns);
        }
    }
    // Squash potentially troublesome differentials
    // After many specuar bounces (e.g. the Transparent Machines scenes),
    // differentials can drift off to have large magnitudes, which ends up
    // leaving a trail of Infs and NaNs in their wake. We'll disable the
    // differentials when this seems to be happening.
    //
    // TODO: this is unsatisfying and would be nice to address in a more
    // principled way.
    if (LengthSquared(rd.rxDirection) > 1e16f || LengthSquared(rd.ryDirection) > 1e16f ||
        LengthSquared(Vector3f(rd.rxOrigin)) > 1e16f ||
        LengthSquared(Vector3f(rd.ryOrigin)) > 1e16f)
        rd.hasDifferentials = false;

    return rd;
}

void SurfaceInteraction::SkipIntersection(RayDifferential *ray, Float t) const {
    *((Ray *)ray) = SpawnRay(ray->d);
    if (ray->hasDifferentials) {
        ray->rxOrigin = ray->rxOrigin + t * ray->rxDirection;
        ray->ryOrigin = ray->ryOrigin + t * ray->ryDirection;
    }
}

SampledSpectrum SurfaceInteraction::Le(const Vector3f &w,
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
