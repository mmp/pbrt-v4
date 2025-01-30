// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_INTERSECT_H
#define PBRT_WAVEFRONT_INTERSECT_H

#include <pbrt/pbrt.h>

#include <pbrt/util/spectrum.h>
#include <pbrt/wavefront/workitems.h>

namespace pbrt {

// Wavefront Ray Intersection Enqueuing Functions
inline PBRT_CPU_GPU void EnqueueWorkAfterMiss(RayWorkItem r,
                                              MediumSampleQueue *mediumSampleQueue,
                                              EscapedRayQueue *escapedRayQueue) {
    if (r.ray.medium) {
        PBRT_DBG("Adding miss ray to mediumSampleQueue. "
                 "ray %f %f %f d %f %f %f beta %f %f %f %f\n",
                 r.ray.o.x, r.ray.o.y, r.ray.o.z, r.ray.d.x, r.ray.d.y, r.ray.d.z,
                 r.beta[0], r.beta[1], r.beta[2], r.beta[3]);
        mediumSampleQueue->Push(r, Infinity);
    } else if (escapedRayQueue) {
        PBRT_DBG("Adding ray to escapedRayQueue pixel index %d\n", r.pixelIndex);
        escapedRayQueue->Push(r);
    }
}

inline PBRT_CPU_GPU void RecordShadowRayResult(const ShadowRayWorkItem w,
                                               SOA<PixelSampleState> *pixelSampleState,
                                               bool foundIntersection) {
    if (foundIntersection) {
        PBRT_DBG("Shadow ray was occluded\n");
        return;
    }
    SampledSpectrum Ld = w.Ld / (w.r_u + w.r_l).Average();
    PBRT_DBG("Unoccluded shadow ray. Final Ld %f %f %f %f "
             "(sr.Ld %f %f %f %f r_u %f %f %f %f r_l %f %f %f %f)\n",
             Ld[0], Ld[1], Ld[2], Ld[3], w.Ld[0], w.Ld[1], w.Ld[2], w.Ld[3], w.r_u[0],
             w.r_u[1], w.r_u[2], w.r_u[3], w.r_l[0], w.r_l[1], w.r_l[2], w.r_l[3]);

    SampledSpectrum Lpixel = pixelSampleState->L[w.pixelIndex];
    pixelSampleState->L[w.pixelIndex] = Lpixel + Ld;
}

inline PBRT_CPU_GPU void EnqueueWorkAfterIntersection(
    RayWorkItem r, Medium rayMedium, float tMax, SurfaceInteraction intr,
    MediumSampleQueue *mediumSampleQueue, RayQueue *nextRayQueue,
    HitAreaLightQueue *hitAreaLightQueue, MaterialEvalQueue *basicEvalMaterialQueue,
    MaterialEvalQueue *universalEvalMaterialQueue) {
    MediumInterface mediumInterface =
        intr.mediumInterface ? *intr.mediumInterface : MediumInterface(rayMedium);

    if (rayMedium) {
        CHECK(mediumSampleQueue);
        PBRT_DBG("Enqueuing into medium sample queue\n");
        mediumSampleQueue->Push(MediumSampleWorkItem{r.ray,
                                                     r.depth,
                                                     tMax,
                                                     r.lambda,
                                                     r.beta,
                                                     r.r_u,
                                                     r.r_l,
                                                     r.pixelIndex,
                                                     r.prevIntrCtx,
                                                     r.specularBounce,
                                                     r.anyNonSpecularBounces,
                                                     r.etaScale,
                                                     intr.areaLight,
                                                     intr.pi,
                                                     intr.n,
                                                     intr.dpdu,
                                                     intr.dpdv,
                                                     -r.ray.d,
                                                     intr.uv,
                                                     intr.material,
                                                     intr.shading.n,
                                                     intr.shading.dpdu,
                                                     intr.shading.dpdv,
                                                     intr.shading.dndu,
                                                     intr.shading.dndv,
                                                     intr.faceIndex,
                                                     mediumInterface});
        return;
    }

    // FIXME: this is all basically duplicate code w/medium.cpp
    Material material = intr.material;

    const MixMaterial *mix = material.CastOrNullptr<MixMaterial>();
    while (mix) {
        MaterialEvalContext ctx(intr);
        material = mix->ChooseMaterial(BasicTextureEvaluator(), ctx);
        mix = material.CastOrNullptr<MixMaterial>();
    }

    if (!material) {
        PBRT_DBG("Enqueuing into medium transition queue: pixel index %d \n",
                 r.pixelIndex);
        Ray newRay = intr.SpawnRay(r.ray.d);
        nextRayQueue->PushIndirectRay(newRay, r.depth, r.prevIntrCtx, r.beta, r.r_u,
                                      r.r_l, r.lambda, r.etaScale, r.specularBounce,
                                      r.anyNonSpecularBounces, r.pixelIndex);
        return;
    }

    if (intr.areaLight) {
        PBRT_DBG("Ray hit an area light: adding to hitAreaLightQueue pixel index %d\n",
                 r.pixelIndex);
        // TODO: intr.wo == -ray.d?
        hitAreaLightQueue->Push(HitAreaLightWorkItem{
            intr.areaLight, intr.p(), intr.n, intr.uv, intr.wo, r.lambda, r.depth, r.beta,
            r.r_u, r.r_l, r.prevIntrCtx, (int)r.specularBounce, r.pixelIndex});
    }

    FloatTexture displacement = material.GetDisplacement();

    MaterialEvalQueue *q =
        (material.CanEvaluateTextures(BasicTextureEvaluator()) &&
         (!displacement || BasicTextureEvaluator().CanEvaluate({displacement}, {})))
            ? basicEvalMaterialQueue
            : universalEvalMaterialQueue;

    PBRT_DBG("Enqueuing for material eval, mtl tag %d\n", material.Tag());

    auto enqueue = [=](auto ptr) {
        using Material = typename std::remove_reference_t<decltype(*ptr)>;
        q->Push(MaterialEvalWorkItem<Material>{ptr,
                                               intr.pi,
                                               intr.n,
                                               intr.dpdu,
                                               intr.dpdv,
                                               intr.time,
                                               r.depth,
                                               intr.shading.n,
                                               intr.shading.dpdu,
                                               intr.shading.dpdv,
                                               intr.shading.dndu,
                                               intr.shading.dndv,
                                               intr.uv,
                                               intr.faceIndex,
                                               r.lambda,
                                               r.pixelIndex,
                                               r.anyNonSpecularBounces,
                                               intr.wo,
                                               r.beta,
                                               r.r_u,
                                               r.etaScale,
                                               mediumInterface});
    };
    material.Dispatch(enqueue);

    PBRT_DBG("Closest hit found intersection at t %f\n", tMax);
}

struct TransmittanceTraceResult {
    bool hit;
    Point3f pHit;
    Material material;
};

template <typename T, typename S>
inline PBRT_CPU_GPU void TraceTransmittance(ShadowRayWorkItem sr,
                                            SOA<PixelSampleState> *pixelSampleState,
                                            T trace, S spawnTo) {
    SampledWavelengths lambda = sr.lambda;

    SampledSpectrum Ld = sr.Ld;

    Ray ray = sr.ray;
    Float tMax = sr.tMax;
    Point3f pLight = ray(tMax);
    RNG rng(Hash(ray.o), Hash(ray.d));

    SampledSpectrum T_ray(1.f);
    SampledSpectrum r_u(1.f), r_l(1.f);

    while (ray.d != Vector3f(0, 0, 0)) {
        PBRT_DBG(
            "Tracing shadow tr shadow ray pixel index %d o %f %f %f d %f %f %f tMax %f\n",
            sr.pixelIndex, ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z, tMax);

        TransmittanceTraceResult result = trace(ray, tMax);

        if (result.hit && result.material) {
            PBRT_DBG("Hit opaque. Bye\n");
            // Hit opaque surface
            T_ray = SampledSpectrum(0.f);
            break;
        }

        if (ray.medium) {
            PBRT_DBG("Ray medium %p. Will sample tmaj...\n", ray.medium.ptr());

            Float tEnd = !result.hit
                             ? tMax
                             : (Distance(ray.o, Point3f(result.pHit)) / Length(ray.d));
            SampledSpectrum T_maj = SampleT_maj(
                ray, tEnd, rng.Uniform<Float>(), rng, lambda,
                [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                    SampledSpectrum T_maj) {
                    SampledSpectrum sigma_n =
                        ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);

                    // ratio-tracking: only evaluate null scattering
                    Float pr = T_maj[0] * sigma_maj[0];
                    T_ray *= T_maj * sigma_n / pr;
                    r_l *= T_maj * sigma_maj / pr;
                    r_u *= T_maj * sigma_n / pr;

                    // Possibly terminate transmittance computation using Russian roulette
                    SampledSpectrum Tr = T_ray / (r_l + r_u).Average();
                    if (Tr.MaxComponentValue() < 0.05f) {
                        Float q = 0.75f;
                        if (rng.Uniform<Float>() < q)
                            T_ray = SampledSpectrum(0.);
                        else
                            T_ray /= 1 - q;
                    }

                    PBRT_DBG(
                        "T_maj %f %f %f %f sigma_n %f %f %f %f sigma_maj %f %f %f %f\n",
                        T_maj[0], T_maj[1], T_maj[2], T_maj[3], sigma_n[0], sigma_n[1],
                        sigma_n[2], sigma_n[3], sigma_maj[0], sigma_maj[1], sigma_maj[2],
                        sigma_maj[3]);
                    PBRT_DBG(
                        "T_ray %f %f %f %f r_l %f %f %f %f r_u %f %f %f %f\n",
                        T_ray[0], T_ray[1], T_ray[2], T_ray[3], r_l[0], r_l[1],
                        r_l[2], r_l[3], r_u[0], r_u[1], r_u[2], r_u[3]);

                    if (!T_ray)
                        return false;

                    return true;
                });
            T_ray *= T_maj / T_maj[0];
            r_l *= T_maj / T_maj[0];
            r_u *= T_maj / T_maj[0];
        }

        if (!result.hit || !T_ray)
            // done
            break;

        ray = spawnTo(pLight);
    }

    PBRT_DBG("Final T_ray %.9g %.9g %.9g %.9g sr.r_u %.9g %.9g %.9g %.9g "
             "r_u %.9g %.9g %.9g %.9g\n",
             T_ray[0], T_ray[1], T_ray[2], T_ray[3], sr.r_u[0], sr.r_u[1],
             sr.r_u[2], sr.r_u[3], r_u[0], r_u[1], r_u[2], r_u[3]);
    PBRT_DBG("sr.r_l %.9g %.9g %.9g %.9g r_l %.9g %.9g %.9g %.9g\n",
             sr.r_l[0], sr.r_l[1], sr.r_l[2], sr.r_l[3], r_l[0],
             r_l[1], r_l[2], r_l[3]);
    PBRT_DBG("scaled throughput %.9g %.9g %.9g %.9g\n",
             T_ray[0] / (sr.r_u * r_u + sr.r_l * r_l).Average(),
             T_ray[1] / (sr.r_u * r_u + sr.r_l * r_l).Average(),
             T_ray[2] / (sr.r_u * r_u + sr.r_l * r_l).Average(),
             T_ray[3] / (sr.r_u * r_u + sr.r_l * r_l).Average());

    if (T_ray) {
        // FIXME/reconcile: this takes r_l as input while
        // e.g. VolPathIntegrator::SampleLd() does not...
        Ld *= T_ray / (sr.r_u * r_u + sr.r_l * r_l).Average();

        PBRT_DBG("Setting final Ld for shadow ray pixel index %d = as %f %f %f %f\n",
                 sr.pixelIndex, Ld[0], Ld[1], Ld[2], Ld[3]);

        SampledSpectrum Lpixel = pixelSampleState->L[sr.pixelIndex];
        pixelSampleState->L[sr.pixelIndex] = Lpixel + Ld;
    }
}

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_INTERSECT_H
