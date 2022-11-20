// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/integrator.h>

#include <pbrt/media.h>

namespace pbrt {

// SampleMediumScatteringCallback Definition
struct SampleMediumScatteringCallback {
    int wavefrontDepth;
    WavefrontPathIntegrator *integrator;
    template <typename PhaseFunction>
    void operator()() {
        integrator->SampleMediumScattering<PhaseFunction>(wavefrontDepth);
    }
};

// WavefrontPathIntegrator Participating Media Methods
void WavefrontPathIntegrator::SampleMediumInteraction(int wavefrontDepth) {
    if (!haveMedia)
        return;

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);
    ForAllQueued(
        "Sample medium interaction", mediumSampleQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(MediumSampleWorkItem w) {
            Ray ray = w.ray;
            Float tMax = w.tMax;

            PBRT_DBG("Sampling medium interaction pixel index %d depth %d ray %f %f %f d "
                     "%f %f "
                     "%f tMax %f\n",
                     w.pixelIndex, w.depth, ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y,
                     ray.d.z, tMax);

            SampledWavelengths lambda = w.lambda;
            SampledSpectrum beta = w.beta;
            SampledSpectrum r_u = w.r_u;
            SampledSpectrum r_l = w.r_l;
            SampledSpectrum L(0.f);
            RNG rng(Hash(ray.o, tMax), Hash(ray.d));

            PBRT_DBG("Lambdas %f %f %f %f\n", lambda[0], lambda[1], lambda[2], lambda[3]);
            PBRT_DBG("Medium sample beta %f %f %f %f r_u %f %f %f %f r_l %f %f "
                     "%f %f\n",
                     beta[0], beta[1], beta[2], beta[3], r_u[0], r_u[1],
                     r_u[2], r_u[3], r_l[0], r_l[1], r_l[2],
                     r_l[3]);

            // Sample the medium according to T_maj, the homogeneous
            // transmission function based on the majorant.
            bool scattered = false;

            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Float uDist = rng.Uniform<Float>();
            Float uMode = rng.Uniform<Float>();

            SampledSpectrum T_maj = SampleT_maj(
                ray, tMax, uDist, rng, lambda,
                [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                    SampledSpectrum T_maj) {
                    PBRT_DBG("Medium event T_maj %f %f %f %f sigma_a %f %f %f %f sigma_s "
                             "%f %f "
                             "%f %f\n",
                             T_maj[0], T_maj[1], T_maj[2], T_maj[3], mp.sigma_a[0],
                             mp.sigma_a[1], mp.sigma_a[2], mp.sigma_a[3], mp.sigma_s[0],
                             mp.sigma_s[1], mp.sigma_s[2], mp.sigma_s[3]);

                    // Add emission, if present.  Always do this and scale
                    // by sigma_a/sigma_maj rather than only doing it
                    // (without scaling) at absorption events.
                    if (w.depth < maxDepth && mp.Le) {
                        Float pr = sigma_maj[0] * T_maj[0];
                        SampledSpectrum r_e = r_u * sigma_maj * T_maj / pr;

                        // Update _L_ for medium emission
                        if (r_e)
                            L += beta * mp.sigma_a * T_maj * mp.Le /
                                 (pr * r_e.Average());
                    }

                    // Compute probabilities for each type of scattering.
                    Float pAbsorb = mp.sigma_a[0] / sigma_maj[0];
                    Float pScatter = mp.sigma_s[0] / sigma_maj[0];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);
                    PBRT_DBG("Medium scattering probabilities: %f %f %f\n", pAbsorb,
                             pScatter, pNull);

                    // And randomly choose one.
                    int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, uMode);

                    if (mode == 0) {
                        // Absorption--done.
                        PBRT_DBG("absorbed\n");
                        beta = SampledSpectrum(0.f);
                        // Tell the medium to stop traversal.
                        return false;
                    } else if (mode == 1) {
                        // Scattering.
                        PBRT_DBG("scattered\n");
                        Float pr = T_maj[0] * mp.sigma_s[0];
                        beta *= T_maj * mp.sigma_s / pr;
                        r_u *= T_maj * mp.sigma_s / pr;

                        // Enqueue medium scattering work.
                        auto enqueue = [=](auto ptr) {
                            using PhaseFunction = typename std::remove_const_t<
                                std::remove_reference_t<decltype(*ptr)>>;
                            mediumScatterQueue->Push(MediumScatterWorkItem<PhaseFunction>{
                                p, w.depth, lambda, beta, r_u, ptr, -ray.d, ray.time,
                                w.etaScale, ray.medium, w.pixelIndex});
                        };
                        DCHECK_RARE(1e-6f, !beta);
                        if (beta && r_u)
                            mp.phase.Dispatch(enqueue);

                        scattered = true;

                        return false;
                    } else {
                        // Null scattering.
                        PBRT_DBG("null-scattered\n");
                        SampledSpectrum sigma_n =
                            ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);

                        Float pr = T_maj[0] * sigma_n[0];
                        beta *= T_maj * sigma_n / pr;
                        if (pr == 0)
                            beta = SampledSpectrum(0.f);
                        r_u *= T_maj * sigma_n / pr;
                        r_l *= T_maj * sigma_maj / pr;

                        uMode = rng.Uniform<Float>();

                        return beta && r_u;
                    }
                });
            if (!scattered && beta) {
                beta *= T_maj / T_maj[0];
                r_u *= T_maj / T_maj[0];
                r_l *= T_maj / T_maj[0];
            }

            PBRT_DBG("Post ray medium sample L %f %f %f %f beta %f %f %f %f\n", L[0],
                     L[1], L[2], L[3], beta[0], beta[1], beta[2], beta[3]);
            PBRT_DBG("Post ray medium sample r_u %f %f %f %f r_l %f %f %f %f\n",
                     r_u[0], r_u[1], r_u[2], r_u[3], r_l[0],
                     r_l[1], r_l[2], r_l[3]);

            // Add any emission found to its pixel sample's L value.
            if (L) {
                SampledSpectrum Lp = pixelSampleState.L[w.pixelIndex];
                pixelSampleState.L[w.pixelIndex] = Lp + L;
                PBRT_DBG("Added emitted radiance %f %f %f %f at pixel index %d\n", L[0],
                         L[1], L[2], L[3], w.pixelIndex);
            }

            // There's no more work to do if there was a scattering event in
            // the medium.
            if (scattered || !beta || !r_u || w.depth == maxDepth)
                return;

            // Otherwise, enqueue bump and medium stuff...
            // FIXME: this is all basically duplicate code w/optix.cu
            if (w.tMax == Infinity) {
                // no intersection
                if (escapedRayQueue) {
                    PBRT_DBG("Adding ray to escapedRayQueue pixel index %d depth %d\n",
                             w.pixelIndex, w.depth);
                    escapedRayQueue->Push(EscapedRayWorkItem{
                        ray.o, ray.d, w.depth, lambda, w.pixelIndex, beta,
                        (int)w.specularBounce, r_u, r_l, w.prevIntrCtx});
                }
                return;
            }

            Material material = w.material;

            const MixMaterial *mix = material.CastOrNullptr<MixMaterial>();
            while (mix) {
                SurfaceInteraction intr(w.pi, w.uv, w.wo, w.dpdus, w.dpdvs, w.dndus,
                                        w.dndvs, ray.time, false /* flip normal */);
                intr.faceIndex = w.faceIndex;
                MaterialEvalContext ctx(intr);
                material = mix->ChooseMaterial(BasicTextureEvaluator(), ctx);
                mix = material.CastOrNullptr<MixMaterial>();
            }

            if (!material) {
                Interaction intr(w.pi, w.n);
                intr.mediumInterface = &w.mediumInterface;
                Ray newRay = intr.SpawnRay(ray.d);
                nextRayQueue->PushIndirectRay(
                    newRay, w.depth, w.prevIntrCtx, beta, r_u, r_l, lambda,
                    w.etaScale, w.specularBounce, w.anyNonSpecularBounces, w.pixelIndex);
                return;
            }

            if (w.areaLight) {
                PBRT_DBG(
                    "Ray hit an area light: adding to hitAreaLightQueue pixel index %d "
                    "depth %d\n",
                    w.pixelIndex, w.depth);
                hitAreaLightQueue->Push(HitAreaLightWorkItem{
                    w.areaLight, Point3f(w.pi), w.n, w.uv, -ray.d, lambda, w.depth, beta,
                    r_u, r_l, w.prevIntrCtx, w.specularBounce, w.pixelIndex});
            }

            FloatTexture displacement = material.GetDisplacement();

            MaterialEvalQueue *q =
                (material.CanEvaluateTextures(BasicTextureEvaluator()) &&
                 (!displacement ||
                  BasicTextureEvaluator().CanEvaluate({displacement}, {})))
                    ? basicEvalMaterialQueue
                    : universalEvalMaterialQueue;

            PBRT_DBG("Enqueuing for material eval, mtl tag %d", material.Tag());

            auto enqueue = [=](auto ptr) {
                using Material = typename std::remove_reference_t<decltype(*ptr)>;
                q->Push<MaterialEvalWorkItem<Material>>(
                    MaterialEvalWorkItem<Material>{ptr,
                                                   w.pi,
                                                   w.n,
                                                   w.dpdu,
                                                   w.dpdv,
                                                   ray.time,
                                                   w.depth,
                                                   w.ns,
                                                   w.dpdus,
                                                   w.dpdvs,
                                                   w.dndus,
                                                   w.dndvs,
                                                   w.uv,
                                                   w.faceIndex,
                                                   lambda,
                                                   w.pixelIndex,
                                                   w.anyNonSpecularBounces,
                                                   -ray.d,
                                                   beta,
                                                   r_u,
                                                   w.etaScale,
                                                   w.mediumInterface});
            };
            material.Dispatch(enqueue);
        });

    if (wavefrontDepth == maxDepth)
        return;

    ForEachType(SampleMediumScatteringCallback{wavefrontDepth, this},
                PhaseFunction::Types());
}

template <typename ConcretePhaseFunction>
void WavefrontPathIntegrator::SampleMediumScattering(int wavefrontDepth) {
    RayQueue *currentRayQueue = CurrentRayQueue(wavefrontDepth);
    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);

    std::string desc =
        std::string("Sample direct/indirect - ") + ConcretePhaseFunction::Name();
    ForAllQueued(
        desc.c_str(),
        mediumScatterQueue->Get<MediumScatterWorkItem<ConcretePhaseFunction>>(),
        maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const MediumScatterWorkItem<ConcretePhaseFunction> w) {
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Vector3f wo = w.wo;

            // Sample direct lighting at medium scattering event.  First,
            // choose a light source.
            LightSampleContext ctx(Point3fi(w.p), Normal3f(0, 0, 0), Normal3f(0, 0, 0));
            pstd::optional<SampledLight> sampledLight =
                lightSampler.Sample(ctx, raySamples.direct.uc);

            if (sampledLight) {
                Light light = sampledLight->light;
                // And now sample a point on the light.
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx, raySamples.direct.u, w.lambda, true);
                if (ls && ls->L && ls->pdf > 0) {
                    Vector3f wi = ls->wi;
                    SampledSpectrum beta = w.beta * w.phase->p(wo, wi);

                    PBRT_DBG("Phase phase beta %f %f %f %f\n", beta[0], beta[1], beta[2],
                             beta[3]);

                    // Compute PDFs for direct lighting MIS calculation.
                    Float lightPDF = ls->pdf * sampledLight->p;
                    Float phasePDF =
                        IsDeltaLight(light.Type()) ? 0.f : w.phase->PDF(wo, wi);
                    SampledSpectrum r_u = w.r_u * phasePDF;
                    SampledSpectrum r_l = w.r_u * lightPDF;

                    SampledSpectrum Ld = beta * ls->L;
                    Ray ray(w.p, ls->pLight.p() - w.p, w.time, w.medium);

                    // Enqueue shadow ray
                    shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon,
                                                           w.lambda, Ld, r_u, r_l,
                                                           w.pixelIndex});

                    PBRT_DBG("Enqueued medium shadow ray depth %d "
                             "Ld %f %f %f %f r_u %f %f %f %f "
                             "r_l %f %f %f %f pixel index %d\n",
                             w.depth, Ld[0], Ld[1], Ld[2], Ld[3], r_u[0], r_u[1],
                             r_u[2], r_u[3], r_l[0], r_l[1], r_l[2],
                             r_l[3], w.pixelIndex);
                }
            }

            // Sample indirect lighting.
            pstd::optional<PhaseFunctionSample> phaseSample =
                w.phase->Sample_p(wo, raySamples.indirect.u);
            if (!phaseSample || phaseSample->pdf == 0)
                return;

            SampledSpectrum beta = w.beta * phaseSample->p / phaseSample->pdf;
            SampledSpectrum r_u = w.r_u;
            SampledSpectrum r_l = w.r_u / phaseSample->pdf;

            // Russian roulette
            // TODO: should we even bother? Generally beta is one here,
            // due to the way scattering events are scattered and because we're
            // sampling exactly from the phase function's distribution...
            SampledSpectrum rrBeta = beta * w.etaScale / r_u.Average();
            if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                if (raySamples.indirect.rr < q) {
                    PBRT_DBG("RR terminated medium indirect with q %f pixel index %d\n",
                             q, w.pixelIndex);
                    return;
                }
                beta /= 1 - q;
            }

            Ray ray(w.p, phaseSample->wi, w.time, w.medium);
            bool specularBounce = false;
            bool anyNonSpecularBounces = true;

            // Spawn indirect ray.
            nextRayQueue->PushIndirectRay(ray, w.depth + 1, ctx, beta, r_u, r_l,
                                          w.lambda, w.etaScale, specularBounce,
                                          anyNonSpecularBounces, w.pixelIndex);
            PBRT_DBG("Enqueuing indirect medium ray at depth %d pixel index %d\n",
                     w.depth + 1, w.pixelIndex);
        });
}

}  // namespace pbrt
