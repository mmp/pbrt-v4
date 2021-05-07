
#ifndef PBRT_WAVEFRONT_INTERSECT_H
#define PBRT_WAVEFRONT_INTERSECT_H

#include <pbrt/pbrt.h>

#include <pbrt/wavefront/workitems.h>
#include <pbrt/util/spectrum.h>

namespace pbrt {

inline PBRT_CPU_GPU void
EnqueueWorkAfterMiss(RayWorkItem r, MediumSampleQueue *mediumSampleQueue,
                     EscapedRayQueue *escapedRayQueue) {
    if (r.ray.medium) {
        PBRT_DBG("Adding miss ray to mediumSampleQueue. "
                 "ray %f %f %f d %f %f %f T_hat %f %f %f %f\n",
                 r.ray.o.x, r.ray.o.y, r.ray.o.z, r.ray.d.x, r.ray.d.y, r.ray.d.z,
                 r.T_hat[0], r.T_hat[1], r.T_hat[2], r.T_hat[3]);
        mediumSampleQueue->Push(r.ray, Infinity, r.lambda, r.T_hat, r.uniPathPDF,
                                r.lightPathPDF, r.pixelIndex, r.prevIntrCtx,
                                r.isSpecularBounce,
                                r.anyNonSpecularBounces, r.etaScale);
    } else if (escapedRayQueue) {
        PBRT_DBG("Adding ray to escapedRayQueue pixel index %d\n", r.pixelIndex);
        escapedRayQueue->Push(
                              EscapedRayWorkItem{r.ray.o, r.ray.d, r.lambda, r.pixelIndex, (int)r.isSpecularBounce,
                                                     r.T_hat, r.uniPathPDF, r.lightPathPDF, r.prevIntrCtx});
    }
}

inline PBRT_CPU_GPU void
EnqueueWorkAfterIntersection(RayWorkItem r, Medium rayMedium, float tMax, SurfaceInteraction intr,
                             MediumSampleQueue *mediumSampleQueue,
                             RayQueue *nextRayQueue,
                             HitAreaLightQueue *hitAreaLightQueue,
                             MaterialEvalQueue *basicEvalMaterialQueue,
                             MaterialEvalQueue *universalEvalMaterialQueue,
                             MediumInterface mediumInterface) {
    if (rayMedium) {
        assert(mediumSampleQueue);
        PBRT_DBG("Enqueuing into medium sample queue\n");
        mediumSampleQueue->Push(
            MediumSampleWorkItem{r.ray,
                                 tMax,
                                 r.lambda,
                                 r.T_hat,
                                 r.uniPathPDF,
                                 r.lightPathPDF,
                                 r.pixelIndex,
                                 r.prevIntrCtx,
                                 r.isSpecularBounce,
                                 r.anyNonSpecularBounces,
                                 r.etaScale,
                                 intr.areaLight,
                                 intr.pi,
                                 intr.n,
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
        nextRayQueue->PushIndirectRay(
            newRay, r.prevIntrCtx, r.T_hat, r.uniPathPDF, r.lightPathPDF, r.lambda,
            r.etaScale, r.isSpecularBounce, r.anyNonSpecularBounces, r.pixelIndex);
        return;
    }

    if (intr.areaLight) {
        PBRT_DBG("Ray hit an area light: adding to hitAreaLightQueue pixel index %d\n",
                 r.pixelIndex);
        Ray ray = r.ray;
        // TODO: intr.wo == -ray.d?
        hitAreaLightQueue->Push(HitAreaLightWorkItem{
            intr.areaLight, intr.p(), intr.n, intr.uv, intr.wo, r.lambda,
            r.T_hat, r.uniPathPDF, r.lightPathPDF, r.prevIntrCtx,
            (int)r.isSpecularBounce, r.pixelIndex});
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
        q->Push(MaterialEvalWorkItem<Material>{
            ptr, intr.pi, intr.n, intr.shading.n,
            intr.shading.dpdu, intr.shading.dpdv, intr.shading.dndu, intr.shading.dndv,
                intr.uv, intr.faceIndex, r.lambda, r.anyNonSpecularBounces, intr.wo, r.pixelIndex,
                r.T_hat, r.uniPathPDF, r.etaScale,
                mediumInterface, intr.time});
    };
    material.Dispatch(enqueue);

    PBRT_DBG("Closest hit found intersection at t %f\n", tMax);
}


inline PBRT_CPU_GPU void
RecordShadowRayIntersection(const ShadowRayWorkItem w, SOA<PixelSampleState> *pixelSampleState,
                            bool foundIntersection) {
    if (foundIntersection) {
        PBRT_DBG("Shadow ray was occluded\n");
        return;
    }

    SampledSpectrum Ld = w.Ld / (w.uniPathPDF + w.lightPathPDF).Average();
    PBRT_DBG("Unoccluded shadow ray. Final Ld %f %f %f %f "
             "(sr.Ld %f %f %f %f uniPathPDF %f %f %f %f lightPathPDF %f %f %f %f)\n",
             Ld[0], Ld[1], Ld[2], Ld[3],
             w.Ld[0], w.Ld[1], w.Ld[2], w.Ld[3],
             w.uniPathPDF[0], w.uniPathPDF[1], w.uniPathPDF[2], w.uniPathPDF[3],
             w.lightPathPDF[0], w.lightPathPDF[1], w.lightPathPDF[2], w.lightPathPDF[3]);

    SampledSpectrum Lpixel = pixelSampleState->L[w.pixelIndex];
    pixelSampleState->L[w.pixelIndex] = Lpixel + Ld;
}

struct TransmittanceTraceResult {
     bool hit;
     Point3f pHit;
     Material material;
};

inline PBRT_CPU_GPU
void rescale(SampledSpectrum &T_hat, SampledSpectrum &lightPathPDF,
             SampledSpectrum &uniPathPDF) {
    if (T_hat.MaxComponentValue() > 0x1p24f ||
        lightPathPDF.MaxComponentValue() > 0x1p24f ||
        uniPathPDF.MaxComponentValue() > 0x1p24f) {
        T_hat *= 1.f / 0x1p24f;
        lightPathPDF *= 1.f / 0x1p24f;
        uniPathPDF *= 1.f / 0x1p24f;
    } else if (T_hat.MaxComponentValue() < 0x1p-24f ||
               lightPathPDF.MaxComponentValue() < 0x1p-24f ||
               uniPathPDF.MaxComponentValue() < 0x1p-24f) {
        T_hat *= 0x1p24f;
        lightPathPDF *= 0x1p24f;
        uniPathPDF *= 0x1p24f;
    }
}

 template <typename T, typename S>
inline PBRT_CPU_GPU void
TraceTransmittance(ShadowRayWorkItem sr, SOA<PixelSampleState> *pixelSampleState,
                   T trace, S spawnTo) {
    SampledWavelengths lambda = sr.lambda;

    SampledSpectrum Ld = sr.Ld;

    Ray ray = sr.ray;
    Float tMax = sr.tMax;
    Point3f pLight = ray(tMax);
    RNG rng(Hash(ray.o), Hash(ray.d));

    SampledSpectrum T_ray(1.f);
    SampledSpectrum uniPathPDF(1.f), lightPathPDF(1.f);

    while (ray.d != Vector3f(0, 0, 0)) {
        PBRT_DBG("Tracing shadow tr shadow ray pixel index %d o %f %f %f d %f %f %f tMax %f\n",
                 sr.pixelIndex, ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z,
                 tMax);

        TransmittanceTraceResult result = trace(ray, tMax);

        if (result.hit && result.material) {
            PBRT_DBG("Hit opaque. Bye\n");
            // Hit opaque surface
            T_ray = SampledSpectrum(0.f);
            break;
        }

        if (ray.medium) {
            PBRT_DBG("Ray medium %p. Will sample tmaj...\n", ray.medium.ptr());

            Float tEnd =
                !result.hit ? tMax : (Distance(ray.o, Point3f(result.pHit)) / Length(ray.d));
            SampledSpectrum T_maj =
                ray.medium.SampleT_maj(ray, tEnd, rng.Uniform<Float>(), rng, lambda,
                                  [&](const MediumSample &mediumSample) {
                                      const SampledSpectrum &T_maj = mediumSample.T_maj;
                                      const MediumInteraction &intr = mediumSample.intr;
                                      SampledSpectrum sigma_n = intr.sigma_n();

                                      // ratio-tracking: only evaluate null scattering
                                      T_ray *= T_maj * sigma_n;
                                      lightPathPDF *= T_maj * intr.sigma_maj;
                                      uniPathPDF *= T_maj * sigma_n;

                                      // Possibly terminate transmittance computation using Russian roulette
                                      SampledSpectrum Tr = T_ray / (lightPathPDF + uniPathPDF).Average();
                                      if (Tr.MaxComponentValue() < 0.05f) {
                                          Float q = 0.75f;
                                          if (rng.Uniform<Float>() < q)
                                              T_ray = SampledSpectrum(0.);
                                          else {
                                              lightPathPDF *= 1 - q;
                                              uniPathPDF *= 1 - q;
                                          }
                                      }

                                      PBRT_DBG("T_maj %f %f %f %f sigma_n %f %f %f %f sigma_maj %f %f %f %f\n",
                                               T_maj[0], T_maj[1], T_maj[2], T_maj[3],
                                               sigma_n[0], sigma_n[1], sigma_n[2], sigma_n[3],
                                               intr.sigma_maj[0], intr.sigma_maj[1], intr.sigma_maj[2],
                                               intr.sigma_maj[3]);
                                      PBRT_DBG("T_ray %f %f %f %f lightPathPDF %f %f %f %f uniPathPDF %f %f %f %f\n",
                                               T_ray[0], T_ray[1], T_ray[2], T_ray[3],
                                               lightPathPDF[0], lightPathPDF[1], lightPathPDF[2], lightPathPDF[3],
                                               uniPathPDF[0], uniPathPDF[1], uniPathPDF[2], uniPathPDF[3]);

                                      if (!T_ray)
                                          return false;

                                      rescale(T_ray, lightPathPDF, uniPathPDF);

                                      return true;
                                  });
            T_ray *= T_maj;
            lightPathPDF *= T_maj;
            uniPathPDF *= T_maj;
        }

        if (!result.hit || !T_ray)
            // done
            break;

        ray = spawnTo(pLight);
    }

    PBRT_DBG("Final T_ray %.9g %.9g %.9g %.9g sr.uniPathPDF %.9g %.9g %.9g %.9g uniPathPDF %.9g %.9g %.9g %.9g\n",
             T_ray[0], T_ray[1], T_ray[2], T_ray[3],
             sr.uniPathPDF[0], sr.uniPathPDF[1], sr.uniPathPDF[2], sr.uniPathPDF[3],
             uniPathPDF[0], uniPathPDF[1], uniPathPDF[2], uniPathPDF[3]);
    PBRT_DBG("sr.lightPathPDF %.9g %.9g %.9g %.9g lightPathPDF %.9g %.9g %.9g %.9g\n",
             sr.lightPathPDF[0], sr.lightPathPDF[1], sr.lightPathPDF[2], sr.lightPathPDF[3],
             lightPathPDF[0], lightPathPDF[1], lightPathPDF[2], lightPathPDF[3]);
    PBRT_DBG("scaled throughput %.9g %.9g %.9g %.9g\n",
             T_ray[0] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average(),
             T_ray[1] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average(),
             T_ray[2] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average(),
             T_ray[3] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average());

    if (T_ray) {
        // FIXME/reconcile: this takes lightPathPDF as input while
        // e.g. VolPathIntegrator::SampleLd() does not...
        Ld *= T_ray / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average();

        PBRT_DBG("Setting final Ld for shadow ray pixel index %d = as %f %f %f %f\n",
                 sr.pixelIndex, Ld[0], Ld[1], Ld[2], Ld[3]);

        SampledSpectrum Lpixel = pixelSampleState->L[sr.pixelIndex];
        pixelSampleState->L[sr.pixelIndex] = Lpixel + Ld;
    }
}

} // namespace pbrt

#endif // PBRT_WAVEFRONT_INTERSECT_H
