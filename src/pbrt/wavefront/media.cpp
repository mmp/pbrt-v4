// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/integrator.h>

#include <pbrt/media.h>

namespace pbrt {

// It's not unususal for these values to have very large or very small
// magnitudes after multiple (null) scattering events, even though in the
// end ratios like T_hat/uniPathPDF are generally around 1.  To avoid overflow,
// we rescale all three of them by the same factor when they become large.
PBRT_CPU_GPU
static inline void rescale(SampledSpectrum &T_hat, SampledSpectrum &lightPathPDF,
                           SampledSpectrum &uniPathPDF) {
    // Note that no precision is lost in the rescaling since we're always
    // multiplying by an exact power of 2.
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
            SampledSpectrum T_hat = w.T_hat;
            SampledSpectrum uniPathPDF = w.uniPathPDF;
            SampledSpectrum lightPathPDF = w.lightPathPDF;
            SampledSpectrum L(0.f);
            RNG rng(Hash(tMax), Hash(ray.d));

            PBRT_DBG("Lambdas %f %f %f %f\n", lambda[0], lambda[1], lambda[2], lambda[3]);
            PBRT_DBG("Medium sample T_hat %f %f %f %f uniPathPDF %f %f %f %f "
                     "lightPathPDF %f %f %f %f\n",
                     T_hat[0], T_hat[1], T_hat[2], T_hat[3], uniPathPDF[0], uniPathPDF[1],
                     uniPathPDF[2], uniPathPDF[3], lightPathPDF[0], lightPathPDF[1],
                     lightPathPDF[2], lightPathPDF[3]);

            // Sample the medium according to T_maj, the homogeneous
            // transmission function based on the majorant.
            bool scattered = false;

            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Float uDist = raySamples.media.uDist;
            Float uMode = raySamples.media.uMode;

            SampledSpectrum T_maj = ray.medium.SampleT_maj(
                ray, tMax, uDist, rng, lambda, [&](const MediumSample &mediumSample) {
                    rescale(T_hat, uniPathPDF, lightPathPDF);

                    const MediumInteraction &intr = mediumSample.intr;
                    const SampledSpectrum &sigma_a = intr.sigma_a;
                    const SampledSpectrum &sigma_s = intr.sigma_s;
                    const SampledSpectrum &T_maj = mediumSample.T_maj;

                    PBRT_DBG("Medium event T_maj %f %f %f %f sigma_a %f %f %f %f sigma_s "
                             "%f %f "
                             "%f %f\n",
                             T_maj[0], T_maj[1], T_maj[2], T_maj[3], sigma_a[0],
                             sigma_a[1], sigma_a[2], sigma_a[3], sigma_s[0], sigma_s[1],
                             sigma_s[2], sigma_s[3]);

                    // Add emission, if present.  Always do this and scale
                    // by sigma_a/sigma_maj rather than only doing it
                    // (without scaling) at absorption events.
                    if (w.depth < maxDepth && intr.Le)
                        L += T_hat * intr.Le * sigma_a /
                             (intr.sigma_maj[0] * uniPathPDF.Average());

                    // Compute probabilities for each type of scattering.
                    Float pAbsorb = sigma_a[0] / intr.sigma_maj[0];
                    Float pScatter = sigma_s[0] / intr.sigma_maj[0];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);
                    PBRT_DBG("Medium scattering probabilities: %f %f %f\n", pAbsorb,
                             pScatter, pNull);

                    // And randomly choose one.
                    int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, uMode);

                    if (mode == 0) {
                        // Absorption--done.
                        PBRT_DBG("absorbed\n");
                        T_hat = SampledSpectrum(0.f);
                        // Tell the medium to stop traversal.
                        return false;
                    } else if (mode == 1) {
                        // Scattering.
                        PBRT_DBG("scattered\n");
                        T_hat *= T_maj * sigma_s;
                        uniPathPDF *= T_maj * sigma_s;

                        // Enqueue medium scattering work.
                        auto enqueue = [=](auto ptr) {
                            using PhaseFunction =
                                typename std::remove_const_t<std::remove_reference_t<decltype(*ptr)>>;
                            mediumScatterQueue->Push(MediumScatterWorkItem<PhaseFunction>{
                                intr.p(), w.depth, lambda, T_hat, uniPathPDF, ptr, -ray.d,
                                ray.time, w.etaScale, ray.medium, w.pixelIndex});
                        };
                        DCHECK_RARE(1e-6f, !T_hat);
                        if (T_hat)
                            intr.phase.Dispatch(enqueue);

                        scattered = true;

                        return false;
                    } else {
                        // Null scattering.
                        PBRT_DBG("null-scattered\n");
                        SampledSpectrum sigma_n = intr.sigma_n();

                        T_hat *= T_maj * sigma_n;
                        uniPathPDF *= T_maj * sigma_n;
                        lightPathPDF *= T_maj * intr.sigma_maj;

                        uMode = rng.Uniform<Float>();

                        return true;
                    }
                });
            if (!scattered && T_hat) {
                T_hat *= T_maj;
                uniPathPDF *= T_maj;
                lightPathPDF *= T_maj;
            }

            PBRT_DBG("Post ray medium sample L %f %f %f %f T_hat %f %f %f %f\n", L[0],
                     L[1], L[2], L[3], T_hat[0], T_hat[1], T_hat[2], T_hat[3]);
            PBRT_DBG("Post ray medium sample uniPathPDF %f %f %f %f lightPathPDF %f %f "
                     "%f %f\n",
                     uniPathPDF[0], uniPathPDF[1], uniPathPDF[2], uniPathPDF[3],
                     lightPathPDF[0], lightPathPDF[1], lightPathPDF[2], lightPathPDF[3]);

            // Add any emission found to its pixel sample's L value.
            if (L) {
                SampledSpectrum Lp = pixelSampleState.L[w.pixelIndex];
                pixelSampleState.L[w.pixelIndex] = Lp + L;
                PBRT_DBG("Added emitted radiance %f %f %f %f at pixel index %d\n", L[0],
                         L[1], L[2], L[3], w.pixelIndex);
            }

            // There's no more work to do if there was a scattering event in
            // the medium.
            if (scattered || !T_hat || w.depth == maxDepth)
                return;

            // Otherwise, enqueue bump and medium stuff...
            // FIXME: this is all basically duplicate code w/optix.cu
            if (w.tMax == Infinity) {
                // no intersection
                if (escapedRayQueue) {
                    PBRT_DBG("Adding ray to escapedRayQueue pixel index %d depth %d\n",
                             w.pixelIndex, w.depth);
                    escapedRayQueue->Push(
                        EscapedRayWorkItem{ray.o, ray.d, w.depth, lambda, w.pixelIndex,
                                           T_hat, (int)w.isSpecularBounce, uniPathPDF,
                                           lightPathPDF, w.prevIntrCtx});
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
                nextRayQueue->PushIndirectRay(newRay, w.depth, w.prevIntrCtx, T_hat,
                                              uniPathPDF, lightPathPDF, lambda,
                                              w.etaScale, w.isSpecularBounce,
                                              w.anyNonSpecularBounces, w.pixelIndex);
                return;
            }

            if (w.areaLight) {
                PBRT_DBG(
                    "Ray hit an area light: adding to hitAreaLightQueue pixel index %d "
                    "depth %d\n",
                    w.pixelIndex, w.depth);
                hitAreaLightQueue->Push(HitAreaLightWorkItem{
                    w.areaLight, Point3f(w.pi), w.n, w.uv, -ray.d, lambda, w.depth, T_hat,
                    uniPathPDF, lightPathPDF, w.prevIntrCtx, w.isSpecularBounce,
                    w.pixelIndex});
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
                                                   T_hat,
                                                   uniPathPDF,
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
                pstd::optional<LightLiSample> ls = light.SampleLi(
                    ctx, raySamples.direct.u, w.lambda, LightSamplingMode::WithMIS);
                if (ls && ls->L && ls->pdf > 0) {
                    Vector3f wi = ls->wi;
                    SampledSpectrum T_hat = w.T_hat * w.phase->p(wo, wi);

                    PBRT_DBG("Phase phase T_hat %f %f %f %f\n", T_hat[0], T_hat[1],
                             T_hat[2], T_hat[3]);

                    // Compute PDFs for direct lighting MIS calculation.
                    Float lightPDF = ls->pdf * sampledLight->pdf;
                    Float phasePDF =
                        IsDeltaLight(light.Type()) ? 0.f : w.phase->PDF(wo, wi);
                    SampledSpectrum uniPathPDF = w.uniPathPDF * phasePDF;
                    SampledSpectrum lightPathPDF = w.uniPathPDF * lightPDF;

                    SampledSpectrum Ld = T_hat * ls->L;
                    Ray ray(w.p, ls->pLight.p() - w.p, w.time, w.medium);

                    // Enqueue shadow ray
                    shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon,
                                                           w.lambda, Ld, uniPathPDF,
                                                           lightPathPDF, w.pixelIndex});

                    PBRT_DBG("Enqueued medium shadow ray depth %d "
                             "Ld %f %f %f %f uniPathPDF %f %f %f %f "
                             "lightPathPDF %f %f %f %f pixel index %d\n",
                             w.depth, Ld[0], Ld[1], Ld[2], Ld[3], uniPathPDF[0],
                             uniPathPDF[1], uniPathPDF[2], uniPathPDF[3], lightPathPDF[0],
                             lightPathPDF[1], lightPathPDF[2], lightPathPDF[3],
                             w.pixelIndex);
                }
            }

            // Sample indirect lighting.
            pstd::optional<PhaseFunctionSample> phaseSample =
                w.phase->Sample_p(wo, raySamples.indirect.u);
            if (!phaseSample || phaseSample->pdf == 0)
                return;

            SampledSpectrum T_hat = w.T_hat * phaseSample->p;
            SampledSpectrum uniPathPDF = w.uniPathPDF * phaseSample->pdf;
            SampledSpectrum lightPathPDF = w.uniPathPDF;

            // Russian roulette
            // TODO: should we even bother? Generally T_hat/uniPathPDF is one here,
            // due to the way scattering events are scattered and because we're
            // sampling exactly from the phase function's distribution...
            SampledSpectrum rrBeta = T_hat * w.etaScale / uniPathPDF.Average();
            if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                if (raySamples.indirect.rr < q) {
                    PBRT_DBG("RR terminated medium indirect with q %f pixel index %d\n",
                             q, w.pixelIndex);
                    return;
                }
                uniPathPDF *= 1 - q;
                lightPathPDF *= 1 - q;
            }

            Ray ray(w.p, phaseSample->wi, w.time, w.medium);
            bool isSpecularBounce = false;
            bool anyNonSpecularBounces = true;

            // Spawn indirect ray.
            nextRayQueue->PushIndirectRay(
                ray, w.depth + 1, ctx, T_hat, uniPathPDF, lightPathPDF, w.lambda,
                w.etaScale, isSpecularBounce, anyNonSpecularBounces, w.pixelIndex);
            PBRT_DBG("Enqueuing indirect medium ray at depth %d pixel index %d\n",
                     w.depth + 1, w.pixelIndex);
        });
}

}  // namespace pbrt
