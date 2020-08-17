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

#ifdef PBRT_GPU_DBG
#ifndef TO_STRING
#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x
#endif  // !TO_STRING
#define DBG(...) printf(__FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#else
#define DBG(...)
#endif

namespace pbrt {

void GPUPathIntegrator::SampleSubsurface(int depth) {
    ForAllQueued(
        "Get BSSRDF and enqueue probe ray", bssrdfEvalQueue, maxQueueSize,
        [=] PBRT_GPU(const GetBSSRDFAndProbeRayWorkItem be, int index) {
            using BSSRDF = typename SubsurfaceMaterial::BSSRDF;
            BSSRDF bssrdf;
            const SubsurfaceMaterial *material = be.material.Cast<SubsurfaceMaterial>();
            MaterialEvalContext ctx = be.GetMaterialEvalContext();
            SampledWavelengths lambda = be.lambda;
            material->GetBSSRDF(BasicTextureEvaluator(), ctx, lambda, &bssrdf);

            RaySamples raySamples = rayQueues[depth & 1]->raySamples[be.rayIndex];
            Float uc = raySamples.subsurface.uc;
            Point2f u = raySamples.subsurface.u;

            BSSRDFProbeSegment probeSeg = bssrdf.Sample(uc, u);
            if (probeSeg)
                subsurfaceScatterQueue->Push(probeSeg.p0, probeSeg.p1, material, bssrdf,
                                             be.beta, be.pdfUni, be.mediumInterface,
                                             be.rayIndex);
        });

    auto events = accel->IntersectOneRandom(maxQueueSize, subsurfaceScatterQueue);
    struct IsectRandomHack {};
    GetGPUKernelStats<IsectRandomHack>("Tracing subsurface scattering probe rays")
        .launchEvents.push_back(events);

    ForAllQueued(
        "Handle out-scattering after SSS", subsurfaceScatterQueue, maxQueueSize,
        [=] PBRT_GPU(SubsurfaceScatterWorkItem s, int index) {
            if (s.weight == 0)
                return;

            using BSSRDF = TabulatedBSSRDF;
            BSSRDF bssrdf = s.bssrdf;
            using BxDF = typename BSSRDF::BxDF;
            BxDF bxdf;

            SubsurfaceInteraction &intr = s.ssi;
            BSSRDFSample bssrdfSample = bssrdf.ProbeIntersectionToSample(intr, &bxdf);

            if (!bssrdfSample.S || bssrdfSample.pdf == 0)
                return;

            SampledSpectrum betap = s.beta * bssrdfSample.S * s.weight / bssrdfSample.pdf;
            SampledWavelengths lambda = rayQueues[depth & 1]->lambda[s.rayIndex];
            Float etaScale = rayQueues[depth & 1]->etaScale[s.rayIndex];
            RaySamples raySamples = rayQueues[depth & 1]->raySamples[s.rayIndex];
            Vector3f wo = bssrdfSample.wo;
            BSDF &bsdf = bssrdfSample.bsdf;
            Float time = 0;  // TODO: pipe through

            // NOTE: the remainder is copied from the Material/BSDF eval method.
            // Will unify into shared fragments in the book...

            // Indirect...
            {
                Point2f u = raySamples.indirect.u;
                Float uc = raySamples.indirect.uc;

                BSDFSample bsdfSample = bsdf.Sample_f<BxDF>(wo, uc, u);
                if (bsdfSample && bsdfSample.f) {
                    Vector3f wi = bsdfSample.wi;
                    SampledSpectrum beta = betap * bsdfSample.f * AbsDot(wi, intr.ns);
                    SampledSpectrum pdfUni = s.pdfUni, pdfNEE = pdfUni;

                    DBG("%s f*cos[0] %f bsdfSample.pdf %f f*cos/pdf %f\n", BxDF::Name(),
                        bsdfSample.f[0] * AbsDot(wi, intr.ns), bsdfSample.pdf,
                        bsdfSample.f[0] * AbsDot(wi, intr.ns) / bsdfSample.pdf);

                    if (bsdf.SampledPDFIsProportional()) {
                        Float pdf = bsdf.PDF(wo, wi);
                        beta *= pdf / bsdfSample.pdf;
                        pdfUni *= pdf;
                        DBG("Sampled PDF is proportional: pdf %f\n", pdf);
                    } else
                        pdfUni *= bsdfSample.pdf;

                    if (bsdfSample.IsTransmission())
                        etaScale *= Sqr(bsdf.eta);

                    // Russian roulette
                    SampledSpectrum rrBeta = beta * etaScale / pdfUni.Average();
                    if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
                        Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                        if (raySamples.indirect.rr < q) {
                            beta = SampledSpectrum(0.f);
                            DBG("Path terminated with RR\n");
                        }
                        pdfUni *= 1 - q;
                        pdfNEE *= 1 - q;
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
                        int pixelIndex = rayQueues[depth & 1]->pixelIndex[s.rayIndex];

                        rayQueues[(depth + 1) & 1]->PushIndirect(
                            ray, intr.pi, intr.n, intr.ns, beta, pdfUni, pdfNEE, lambda,
                            etaScale, bsdfSample.IsSpecular(), anyNonSpecularBounces,
                            pixelIndex);

                        DBG("Spawned indirect ray at depth %d from prev index %d. "
                            "Specular %d Beta %f %f %f %f pdfUni %f %f %f %f pdfNEE %f "
                            "%f %f %f "
                            "beta/pdfUni %f %f %f %f\n",
                            depth + 1, int(s.rayIndex), int(bsdfSample.IsSpecular()),
                            beta[0], beta[1], beta[2], beta[3], pdfUni[0], pdfUni[1],
                            pdfUni[2], pdfUni[3], pdfNEE[0], pdfNEE[1], pdfNEE[2],
                            pdfNEE[3], SafeDiv(beta, pdfUni)[0], SafeDiv(beta, pdfUni)[1],
                            SafeDiv(beta, pdfUni)[2], SafeDiv(beta, pdfUni)[3]);
                    }
                }
            }

            // Direct lighting...
            if (!bsdf.IsSpecular()) {
                LightSampleContext ctx(intr.pi, intr.n, intr.ns);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                LightHandle light = sampledLight->light;
                if (!light)
                    return;

                LightLiSample ls = light.SampleLi(ctx, raySamples.direct.u, lambda,
                                                  LightSamplingMode::WithMIS);
                if (!ls || !ls.L)
                    return;

                Vector3f wi = ls.wi;
                SampledSpectrum f = bsdf.f<BxDF>(wo, wi);
                if (!f)
                    return;

                SampledSpectrum beta = betap * f * AbsDot(wi, intr.ns);

                DBG("depth %d beta %f %f %f %f f %f %f %f %f ls.L %f %f %f %f ls.pdf "
                    "%f\n",
                    depth, beta[0], beta[1], beta[2], beta[3], f[0], f[1], f[2], f[3],
                    ls.L[0], ls.L[1], ls.L[2], ls.L[3], ls.pdf);

                Float lightPDF = ls.pdf * sampledLight->pdf;
                // This causes pdfUni to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<BxDF>(wo, wi);
                SampledSpectrum pdfUni = s.pdfUni * bsdfPDF;
                SampledSpectrum pdfNEE = s.pdfUni * lightPDF;

                SampledSpectrum Ld = beta * ls.L;

                DBG("depth %d Ld %f %f %f %f "
                    "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                    depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0], beta[1], beta[2], beta[3],
                    SafeDiv(beta, pdfUni)[0], SafeDiv(beta, pdfUni)[1],
                    SafeDiv(beta, pdfUni)[2], SafeDiv(beta, pdfUni)[3],
                    SafeDiv(Ld, pdfUni)[0], SafeDiv(Ld, pdfUni)[1],
                    SafeDiv(Ld, pdfUni)[2], SafeDiv(Ld, pdfUni)[3]);

                Ray ray = SpawnRayTo(intr.pi, intr.n, time, ls.pLight.pi, ls.pLight.n);
                if (haveMedia)
                    // TODO: as above, always take outside here?
                    ray.medium = Dot(ray.d, intr.n) > 0 ? s.mediumInterface.outside
                                                        : s.mediumInterface.inside;

                int pixelIndex = rayQueues[depth & 1]->pixelIndex[s.rayIndex];
                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                       pdfUni, pdfNEE, pixelIndex});
            }
        });

    TraceShadowRays(depth);
}

}  // namespace pbrt
