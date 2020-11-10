// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/bssrdf.h>
#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/interaction.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/samplers.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

namespace pbrt {

void GPUPathIntegrator::SampleSubsurface(int depth) {
    RayQueue *rayQueue = CurrentRayQueue(depth);
    RayQueue *nextRayQueue = NextRayQueue(depth);

    ForAllQueued(
        "Get BSSRDF and enqueue probe ray", bssrdfEvalQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(const GetBSSRDFAndProbeRayWorkItem be, int index) {
            using BSSRDF = typename SubsurfaceMaterial::BSSRDF;
            BSSRDF bssrdf;
            const SubsurfaceMaterial *material = be.material.Cast<SubsurfaceMaterial>();
            MaterialEvalContext ctx = be.GetMaterialEvalContext();
            SampledWavelengths lambda = be.lambda;
            material->GetBSSRDF(BasicTextureEvaluator(), ctx, lambda, &bssrdf);

            RaySamples raySamples = pixelSampleState.samples[be.pixelIndex];
            Float uc = raySamples.subsurface.uc;
            Point2f u = raySamples.subsurface.u;

            pstd::optional<BSSRDFProbeSegment> probeSeg = bssrdf.SampleSp(uc, u);
            if (probeSeg)
                subsurfaceScatterQueue->Push(probeSeg->p0, probeSeg->p1, material, bssrdf,
                                             lambda, be.beta, be.uniPathPDF, be.mediumInterface,
                                             be.etaScale, be.pixelIndex);
        });

    accel->IntersectOneRandom(maxQueueSize, subsurfaceScatterQueue);

    ForAllQueued(
        "Handle out-scattering after SSS", subsurfaceScatterQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(SubsurfaceScatterWorkItem s, int index) {
            if (s.weight == 0)
                return;

            using BSSRDF = TabulatedBSSRDF;
            BSSRDF bssrdf = s.bssrdf;
            using BxDF = typename BSSRDF::BxDF;
            BxDF bxdf;

            SubsurfaceInteraction &intr = s.ssi;
            BSSRDFSample bssrdfSample = bssrdf.ProbeIntersectionToSample(intr, &bxdf);

            if (!bssrdfSample.Sp || bssrdfSample.pdf == 0)
                return;

            SampledSpectrum betap = s.beta * bssrdfSample.Sp * s.weight / bssrdfSample.pdf;
            SampledWavelengths lambda = s.lambda;
            RaySamples raySamples = pixelSampleState.samples[s.pixelIndex];
            Vector3f wo = bssrdfSample.wo;
            BSDF &bsdf = bssrdfSample.Sw;
            Float time = 0;  // TODO: pipe through

            // NOTE: the remainder is copied from the Material/BSDF eval method.
            // Will unify into shared fragments in the book...

            // Indirect...
            {
                Point2f u = raySamples.indirect.u;
                Float uc = raySamples.indirect.uc;

                pstd::optional<BSDFSample> bsdfSample = bsdf.Sample_f<BxDF>(wo, uc, u);
                if (bsdfSample) {
                    Vector3f wi = bsdfSample->wi;
                    SampledSpectrum beta = betap * bsdfSample->f * AbsDot(wi, intr.ns);
                    SampledSpectrum uniPathPDF = s.uniPathPDF, lightPathPDF = uniPathPDF;

                    PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n", BxDF::Name(),
                        bsdfSample->f[0] * AbsDot(wi, intr.ns), bsdfSample->pdf,
                        bsdfSample->f[0] * AbsDot(wi, intr.ns) / bsdfSample->pdf);

                    if (bsdfSample->pdfIsProportional) {
                        Float pdf = bsdf.PDF(wo, wi);
                        beta *= pdf / bsdfSample->pdf;
                        uniPathPDF *= pdf;
                        PBRT_DBG("Sampled PDF is proportional: pdf %f\n", pdf);
                    } else
                        uniPathPDF *= bsdfSample->pdf;

                    Float etaScale = s.etaScale;
                    if (bsdfSample->IsTransmission())
                        etaScale *= Sqr(bsdf.eta);

                    // Russian roulette
                    SampledSpectrum rrBeta = beta * etaScale / uniPathPDF.Average();
                    if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
                        Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                        if (raySamples.indirect.rr < q) {
                            beta = SampledSpectrum(0.f);
                            PBRT_DBG("Path terminated with RR\n");
                        }
                        uniPathPDF *= 1 - q;
                        lightPathPDF *= 1 - q;
                    }

                    if (beta) {
                        Ray ray = SpawnRay(intr.pi, intr.n, time, wi);
                        if (haveMedia)
                            // TODO: should always just take outside in this case?
                            ray.medium = Dot(ray.d, intr.n) > 0
                                             ? s.mediumInterface.outside
                                             : s.mediumInterface.inside;

                        // || rather than | is intentional, to avoid the read if
                        // possible...
                        bool anyNonSpecularBounces = true;

                        nextRayQueue->PushIndirect(
                            ray, intr.pi, intr.n, intr.ns, beta, uniPathPDF, lightPathPDF, lambda,
                            etaScale, bsdfSample->IsSpecular(), anyNonSpecularBounces,
                            s.pixelIndex);

                        PBRT_DBG("Spawned indirect ray at depth %d. "
                            "Specular %d Beta %f %f %f %f uniPathPDF %f %f %f %f lightPathPDF %f "
                            "%f %f %f "
                            "beta/uniPathPDF %f %f %f %f\n",
                            depth + 1, int(bsdfSample->IsSpecular()),
                            beta[0], beta[1], beta[2], beta[3], uniPathPDF[0], uniPathPDF[1],
                            uniPathPDF[2], uniPathPDF[3], lightPathPDF[0], lightPathPDF[1], lightPathPDF[2],
                            lightPathPDF[3], SafeDiv(beta, uniPathPDF)[0], SafeDiv(beta, uniPathPDF)[1],
                            SafeDiv(beta, uniPathPDF)[2], SafeDiv(beta, uniPathPDF)[3]);
                    }
                }
            }

            // Direct lighting...
            if (bsdf.IsNonSpecular()) {
                LightSampleContext ctx(intr.pi, intr.n, intr.ns);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                if (!sampledLight)
                    return;
                LightHandle light = sampledLight->light;

                pstd::optional<LightLiSample> ls = light.SampleLi(ctx, raySamples.direct.u, lambda,
                                                  LightSamplingMode::WithMIS);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;

                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<BxDF>(wo, wi);
                if (!f)
                    return;

                SampledSpectrum beta = betap * f * AbsDot(wi, intr.ns);

                PBRT_DBG("depth %d beta %f %f %f %f f %f %f %f %f ls.L %f %f %f %f ls.pdf "
                    "%f\n",
                    depth, beta[0], beta[1], beta[2], beta[3], f[0], f[1], f[2], f[3],
                    ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                Float lightPDF = ls->pdf * sampledLight->pdf;
                // This causes uniPathPDF to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<BxDF>(wo, wi);
                SampledSpectrum uniPathPDF = s.uniPathPDF * bsdfPDF;
                SampledSpectrum lightPathPDF = s.uniPathPDF * lightPDF;

                SampledSpectrum Ld = SafeDiv(beta * ls->L, lambda.PDF());

                PBRT_DBG("depth %d Ld %f %f %f %f "
                    "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                    depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0], beta[1], beta[2], beta[3],
                    SafeDiv(beta, uniPathPDF)[0], SafeDiv(beta, uniPathPDF)[1],
                    SafeDiv(beta, uniPathPDF)[2], SafeDiv(beta, uniPathPDF)[3],
                    SafeDiv(Ld, uniPathPDF)[0], SafeDiv(Ld, uniPathPDF)[1],
                    SafeDiv(Ld, uniPathPDF)[2], SafeDiv(Ld, uniPathPDF)[3]);

                Ray ray = SpawnRayTo(intr.pi, intr.n, time, ls->pLight.pi, ls->pLight.n);
                if (haveMedia)
                    // TODO: as above, always take outside here?
                    ray.medium = Dot(ray.d, intr.n) > 0 ? s.mediumInterface.outside
                                                        : s.mediumInterface.inside;

                shadowRayQueue->Push(ray, 1 - ShadowEpsilon, lambda, Ld, uniPathPDF, lightPathPDF,
                                     s.pixelIndex);
            }
        });

    TraceShadowRays(depth);
}

}  // namespace pbrt
