// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/pathintegrator.h>

#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/media.h>

namespace pbrt {

// It's not unususal for these values to have very large or very small
// magnitudes after multiple (null) scattering events, even though in the
// end ratios like beta/uniPathPDF are generally around 1.  To avoid overflow,
// we rescale all three of them by the same factor when they become large.
PBRT_CPU_GPU
static inline void rescale(SampledSpectrum &beta, SampledSpectrum &lightPathPDF,
                           SampledSpectrum &uniPathPDF) {
    // Note that no precision is lost in the rescaling since we're always
    // multiplying by an exact power of 2.
    if (beta.MaxComponentValue() > 0x1p24f ||
        lightPathPDF.MaxComponentValue() > 0x1p24f ||
        uniPathPDF.MaxComponentValue() > 0x1p24f) {
        beta *= 1.f / 0x1p24f;
        lightPathPDF *= 1.f / 0x1p24f;
        uniPathPDF *= 1.f / 0x1p24f;
    } else if (beta.MaxComponentValue() < 0x1p-24f ||
               lightPathPDF.MaxComponentValue() < 0x1p-24f ||
               uniPathPDF.MaxComponentValue() < 0x1p-24f) {
        beta *= 0x1p24f;
        lightPathPDF *= 0x1p24f;
        uniPathPDF *= 0x1p24f;
    }
}

// GPUPathIntegrator Participating Media Methods
void GPUPathIntegrator::SampleMediumInteraction(int depth) {
    ForAllQueued(
        "Sample medium interaction", mediumSampleQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(MediumSampleWorkItem ms, int index) {
            Ray ray = ms.ray;
            Float tMax = ms.tMax;

            PBRT_DBG("Sampling medium interaction pixel index %d depth %d ray %f %f %f d "
                     "%f %f "
                     "%f tMax %f\n",
                     ms.pixelIndex, depth, ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y,
                     ray.d.z, tMax);

            SampledWavelengths lambda = ms.lambda;
            SampledSpectrum beta = ms.beta;
            SampledSpectrum uniPathPDF = ms.uniPathPDF;
            SampledSpectrum lightPathPDF = ms.lightPathPDF;
            SampledSpectrum L(0.f);
            RNG rng(Hash(tMax), Hash(ray.d));

            PBRT_DBG("Lambdas %f %f %f %f\n", lambda[0], lambda[1], lambda[2], lambda[3]);
            PBRT_DBG("Medium sample beta %f %f %f %f uniPathPDF %f %f %f %f lightPathPDF "
                     "%f %f %f %f\n",
                     beta[0], beta[1], beta[2], beta[3], uniPathPDF[0], uniPathPDF[1],
                     uniPathPDF[2], uniPathPDF[3], lightPathPDF[0], lightPathPDF[1],
                     lightPathPDF[2], lightPathPDF[3]);

            // Sample the medium according to T_maj, the homogeneous
            // transmission function based on the majorant.
            bool scattered = false;
            SampledSpectrum Tmaj = ray.medium.SampleTmaj(
                ray, tMax, rng, lambda, [&](const MediumSample &mediumSample) {
                    rescale(beta, uniPathPDF, lightPathPDF);

                    const MediumInteraction &intr = mediumSample.intr;
                    const SampledSpectrum &sigma_a = intr.sigma_a;
                    const SampledSpectrum &sigma_s = intr.sigma_s;
                    const SampledSpectrum &Tmaj = mediumSample.Tmaj;

                    PBRT_DBG(
                        "Medium event Tmaj %f %f %f %f sigma_a %f %f %f %f sigma_s %f %f "
                        "%f %f\n",
                        Tmaj[0], Tmaj[1], Tmaj[2], Tmaj[3], sigma_a[0], sigma_a[1],
                        sigma_a[2], sigma_a[3], sigma_s[0], sigma_s[1], sigma_s[2],
                        sigma_s[3]);

                    // Add emission, if present.  Always do this and scale
                    // by sigma_a/sigma_maj rather than only doing it
                    // (without scaling) at absorption events.
                    if (depth < maxDepth && intr.Le)
                        L += beta * intr.Le * sigma_a /
                             (intr.sigma_maj[0] * uniPathPDF.Average());

                    // Compute probabilities for each type of scattering.
                    Float pAbsorb = sigma_a[0] / intr.sigma_maj[0];
                    Float pScatter = sigma_s[0] / intr.sigma_maj[0];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);
                    PBRT_DBG("Medium scattering probabilities: %f %f %f\n", pAbsorb,
                             pScatter, pNull);

                    // And randomly choose one.
                    Float um = rng.Uniform<Float>();
                    int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, um);

                    if (mode == 0) {
                        // Absorption--done.
                        PBRT_DBG("absorbed\n");
                        beta = SampledSpectrum(0.f);
                        // Tell the medium to stop traveral.
                        return false;
                    } else if (mode == 1) {
                        // Scattering.
                        PBRT_DBG("scattered\n");
                        beta *= Tmaj * sigma_s;
                        uniPathPDF *= Tmaj * sigma_s;

                        // TODO: don't hard code a phase function.
                        const HGPhaseFunction *phase =
                            intr.phase.CastOrNullptr<HGPhaseFunction>();
                        // Enqueue medium scattering work.
                        mediumScatterQueue->Push(MediumScatterWorkItem{
                            intr.p(), lambda, beta, uniPathPDF, *phase, -ray.d,
                            ms.etaScale, ray.medium, ms.pixelIndex});
                        scattered = true;

                        return false;
                    } else {
                        // Null scattering.
                        PBRT_DBG("null-scattered\n");
                        SampledSpectrum sigma_n = intr.sigma_n();

                        beta *= Tmaj * sigma_n;
                        uniPathPDF *= Tmaj * sigma_n;
                        lightPathPDF *= Tmaj * intr.sigma_maj;

                        return true;
                    }
                });
            if (!scattered && beta) {
                beta *= Tmaj;
                uniPathPDF *= Tmaj;
                lightPathPDF *= Tmaj;
            }

            PBRT_DBG("Post ray medium sample L %f %f %f %f beta %f %f %f %f\n", L[0],
                     L[1], L[2], L[3], beta[0], beta[1], beta[2], beta[3]);
            PBRT_DBG("Post ray medium sample uniPathPDF %f %f %f %f lightPathPDF %f %f "
                     "%f %f\n",
                     uniPathPDF[0], uniPathPDF[1], uniPathPDF[2], uniPathPDF[3],
                     lightPathPDF[0], lightPathPDF[1], lightPathPDF[2], lightPathPDF[3]);

            // Add any emission found to its pixel sample's L value.
            if (L) {
                SampledSpectrum Lp = pixelSampleState.L[ms.pixelIndex];
                pixelSampleState.L[ms.pixelIndex] = Lp + SafeDiv(L, lambda.PDF());
                PBRT_DBG("Added emitted radiance %f %f %f %f at pixel index %d\n", L[0],
                         L[1], L[2], L[3], ms.pixelIndex);
            }

            // There's no more work to do if there was a scattering event in
            // the medium.
            if (scattered || !beta || depth == maxDepth)
                return;

            // Otherwise, enqueue bump and medium stuff...
            // FIXME: this is all basically duplicate code w/optix.cu
            if (ms.tMax == Infinity) {
                // no intersection
                if (escapedRayQueue) {
                    PBRT_DBG("Adding ray to escapedRayQueue pixel index %d depth %d\n",
                             ms.pixelIndex, depth);
                    escapedRayQueue->Push(EscapedRayWorkItem{
                        beta, uniPathPDF, lightPathPDF, lambda, ray.o, ray.d,
                        ms.prevIntrCtx, (int)ms.isSpecularBounce, ms.pixelIndex});
                }
            }

            MaterialHandle material = ms.material;
            if (!material) {
                Interaction intr(ms.pi, ms.n);
                intr.mediumInterface = &ms.mediumInterface;
                Ray newRay = intr.SpawnRay(ray.d);
                mediumTransitionQueue->Push(MediumTransitionWorkItem{
                    newRay, lambda, beta, uniPathPDF, lightPathPDF, ms.prevIntrCtx,
                    ms.isSpecularBounce, ms.anyNonSpecularBounces, ms.etaScale,
                    ms.pixelIndex});
#if 0
                // WHY NOT THIS?
                rayQueues[(depth + 1) & 1]->PushIndirect(newRay, ms.prevIntrCtx,
                                                         beta, uniPathPDF, lightPathPDF, lambda, ms.etaScale,
                                                         ms.isSpecularBounce, ms.anyNonSpecularBounces,
                                                         ms.pixelIndex);
#endif
                return;
            }

            if (ms.areaLight) {
                PBRT_DBG(
                    "Ray hit an area light: adding to hitAreaLightQueue pixel index %d "
                    "depth %d\n",
                    ms.pixelIndex, depth);
                hitAreaLightQueue->Push(HitAreaLightWorkItem{
                    ms.areaLight, lambda, beta, uniPathPDF, lightPathPDF, Point3f(ms.pi),
                    ms.n, ms.uv, -ray.d, ms.prevIntrCtx, ms.isSpecularBounce,
                    ms.pixelIndex});
            }

            FloatTextureHandle displacement = material.GetDisplacement();

            MaterialEvalQueue *q =
                (material.CanEvaluateTextures(BasicTextureEvaluator()) &&
                 (!displacement ||
                  BasicTextureEvaluator().CanEvaluate({displacement}, {})))
                    ? basicEvalMaterialQueue
                    : universalEvalMaterialQueue;

            PBRT_DBG("Enqueuing for material eval, mtl tag %d", material.Tag());

            auto enqueue = [=](auto ptr) {
                using Material = typename std::remove_reference_t<decltype(*ptr)>;
                q->Push<MaterialEvalWorkItem<Material>>(MaterialEvalWorkItem<Material>{
                    ptr, lambda, beta, uniPathPDF, ms.pi, ms.n, ms.ns, ms.dpdus, ms.dpdvs,
                    ms.dndus, ms.dndvs, -ray.d, ms.uv, ray.time, ms.anyNonSpecularBounces,
                    ms.etaScale, ms.mediumInterface, ms.pixelIndex});
            };
            material.Dispatch(enqueue);
        });

    if (depth == maxDepth)
        return;

    RayQueue *currentRayQueue = CurrentRayQueue(depth);
    RayQueue *nextRayQueue = NextRayQueue(depth);

    using PhaseFunction = HGPhaseFunction;
    std::string desc = std::string("Sample direct/indirect - Henyey Greenstein");
    ForAllQueued(
        desc.c_str(), mediumScatterQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(MediumScatterWorkItem ms, int index) {
            RaySamples raySamples = pixelSampleState.samples[ms.pixelIndex];
            Float time = 0;  // TODO: FIXME
            Vector3f wo = ms.wo;

            // Sample direct lighting at medium scattering event.  First,
            // choose a light source.
            LightSampleContext ctx(Point3fi(ms.p), Normal3f(0, 0, 0), Normal3f(0, 0, 0));
            pstd::optional<SampledLight> sampledLight =
                lightSampler.Sample(ctx, raySamples.direct.uc);

            if (sampledLight) {
                LightHandle light = sampledLight->light;
                // And now sample a point on the light.
                pstd::optional<LightLiSample> ls = light.SampleLi(
                    ctx, raySamples.direct.u, ms.lambda, LightSamplingMode::WithMIS);
                if (ls && ls->L && ls->pdf > 0) {
                    Vector3f wi = ls->wi;
                    SampledSpectrum beta = ms.beta * ms.phase.p(wo, wi);

                    PBRT_DBG("Phase phase beta %f %f %f %f\n", beta[0], beta[1], beta[2],
                             beta[3]);

                    // Compute PDFs for direct lighting MIS calculation.
                    Float lightPDF = ls->pdf * sampledLight->pdf;
                    Float phasePDF =
                        IsDeltaLight(light.Type()) ? 0.f : ms.phase.PDF(wo, wi);
                    SampledSpectrum uniPathPDF = ms.uniPathPDF * phasePDF;
                    SampledSpectrum lightPathPDF = ms.uniPathPDF * lightPDF;

                    SampledSpectrum Ld = SafeDiv(beta * ls->L, ms.lambda.PDF());
                    Ray ray(ms.p, ls->pLight.p() - ms.p, time, ms.medium);

                    // Enqueue shadow ray
                    shadowRayQueue->Push(ray, 1 - ShadowEpsilon, ms.lambda, Ld,
                                         uniPathPDF, lightPathPDF, ms.pixelIndex);

                    PBRT_DBG("Enqueued medium shadow ray depth %d "
                             "Ld %f %f %f %f uniPathPDF %f %f %f %f "
                             "lightPathPDF %f %f %f %f pixel index %d\n",
                             depth, Ld[0], Ld[1], Ld[2], Ld[3], uniPathPDF[0],
                             uniPathPDF[1], uniPathPDF[2], uniPathPDF[3], lightPathPDF[0],
                             lightPathPDF[1], lightPathPDF[2], lightPathPDF[3],
                             ms.pixelIndex);
                }
            }

            // Sample indirect lighting.
            pstd::optional<PhaseFunctionSample> phaseSample =
                ms.phase.Sample_p(wo, raySamples.indirect.u);
            if (!phaseSample || phaseSample->pdf == 0)
                return;

            SampledSpectrum beta = ms.beta * phaseSample->p;
            SampledSpectrum uniPathPDF = ms.uniPathPDF * phaseSample->pdf;
            SampledSpectrum lightPathPDF = ms.uniPathPDF;

            // Russian roulette
            // TODO: should we even bother? Generally beta/uniPathPDF is one here,
            // due to the way scattering events are scattered and because we're
            // sampling exactly from the phase function's distribution...
            SampledSpectrum rrBeta = beta * ms.etaScale / uniPathPDF.Average();
            if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
                Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                if (raySamples.indirect.rr < q) {
                    PBRT_DBG("RR terminated medium indirect with q %f pixel index %d\n",
                             q, ms.pixelIndex);
                    return;
                }
                uniPathPDF *= 1 - q;
                lightPathPDF *= 1 - q;
            }

            Ray ray(ms.p, phaseSample->wi, time, ms.medium);
            bool isSpecularBounce = false;
            bool anyNonSpecularBounces = true;

            // Spawn indirect ray.
            nextRayQueue->PushIndirect(ray, ctx, beta, uniPathPDF, lightPathPDF,
                                       ms.lambda, ms.etaScale, isSpecularBounce,
                                       anyNonSpecularBounces, ms.pixelIndex);
            PBRT_DBG("Enqueuing indirect medium ray at depth %d pixel index %d\n",
                     depth + 1, ms.pixelIndex);
        });
}

void GPUPathIntegrator::HandleMediumTransitions(int depth) {
    RayQueue *rayQueue = NextRayQueue(depth);

    ForAllQueued(
        "Handle medium transitions", mediumTransitionQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(MediumTransitionWorkItem mt, int index) {
            // Have to do this here, later, since we can't be writing into
            // the other ray queue in optix closest hit.  (Wait--really?
            // Why not? Basically boils down to current indirect enqueue (and other
            // places?))
            // TODO: figure this out...
            rayQueue->PushIndirect(mt.ray, mt.prevIntrCtx, mt.beta, mt.uniPathPDF,
                                   mt.lightPathPDF, mt.lambda, mt.etaScale,
                                   mt.isSpecularBounce, mt.anyNonSpecularBounces,
                                   mt.pixelIndex);
            PBRT_DBG("Enqueuied ray after medium transition at depth %d pixel index %d",
                     depth + 1, mt.pixelIndex);
        });
}

}  // namespace pbrt
