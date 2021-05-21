// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/samplers.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/wavefront/integrator.h>

namespace pbrt {

// WavefrontPathIntegrator Subsurface Scattering Methods
void WavefrontPathIntegrator::SampleSubsurface(int wavefrontDepth) {
    if (!haveSubsurface)
        return;

    RayQueue *rayQueue = CurrentRayQueue(wavefrontDepth);
    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);

    ForAllQueued(
        "Get BSSRDF and enqueue probe ray", bssrdfEvalQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const GetBSSRDFAndProbeRayWorkItem w) {
            using BSSRDF = typename SubsurfaceMaterial::BSSRDF;
            BSSRDF bssrdf;
            const SubsurfaceMaterial *material = w.material.Cast<SubsurfaceMaterial>();
            MaterialEvalContext ctx = w.GetMaterialEvalContext();
            SampledWavelengths lambda = w.lambda;
            material->GetBSSRDF(BasicTextureEvaluator(), ctx, lambda, &bssrdf);

            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Float uc = raySamples.subsurface.uc;
            Point2f u = raySamples.subsurface.u;

            pstd::optional<BSSRDFProbeSegment> probeSeg = bssrdf.SampleSp(uc, u);
            if (probeSeg)
                subsurfaceScatterQueue->Push(
                    probeSeg->p0, probeSeg->p1, w.depth, material, bssrdf, lambda,
                    w.T_hat, w.uniPathPDF, w.mediumInterface, w.etaScale, w.pixelIndex);
        });

    aggregate->IntersectOneRandom(maxQueueSize, subsurfaceScatterQueue);

    ForAllQueued(
        "Handle out-scattering after SSS", subsurfaceScatterQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(SubsurfaceScatterWorkItem w) {
            if (w.reservoirPDF == 0)
                return;

            using BSSRDF = TabulatedBSSRDF;
            BSSRDF bssrdf = w.bssrdf;
            using ConcreteBxDF = typename BSSRDF::BxDF;
            ConcreteBxDF bxdf;

            SubsurfaceInteraction &intr = w.ssi;
            BSSRDFSample bssrdfSample = bssrdf.ProbeIntersectionToSample(intr, &bxdf);

            if (!bssrdfSample.Sp || !bssrdfSample.pdf)
                return;

            SampledSpectrum T_hatp = w.T_hat * bssrdfSample.Sp;
            SampledSpectrum uniPathPDF = w.uniPathPDF * w.reservoirPDF * bssrdfSample.pdf;
            SampledWavelengths lambda = w.lambda;
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Vector3f wo = bssrdfSample.wo;
            BSDF &bsdf = bssrdfSample.Sw;
            Float time = 0;  // TODO: pipe through

            // NOTE: the remainder is copied from the Material/BSDF eval method.
            // Will unify into shared fragments in the book...

            // Indirect...
            {
                Point2f u = raySamples.indirect.u;
                Float uc = raySamples.indirect.uc;

                pstd::optional<BSDFSample> bsdfSample =
                    bsdf.Sample_f<ConcreteBxDF>(wo, uc, u);
                if (bsdfSample) {
                    Vector3f wi = bsdfSample->wi;
                    SampledSpectrum T_hat = T_hatp * bsdfSample->f * AbsDot(wi, intr.ns);
                    SampledSpectrum indirUniPathPDF = uniPathPDF,
                                    lightPathPDF = uniPathPDF;

                    PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n",
                             ConcreteBxDF::Name(), bsdfSample->f[0] * AbsDot(wi, intr.ns),
                             bsdfSample->pdf,
                             bsdfSample->f[0] * AbsDot(wi, intr.ns) / bsdfSample->pdf);

                    if (bsdfSample->pdfIsProportional) {
                        Float pdf = bsdf.PDF(wo, wi);
                        T_hat *= pdf / bsdfSample->pdf;
                        indirUniPathPDF *= pdf;
                        PBRT_DBG("Sampled PDF is proportional: pdf %f\n", pdf);
                    } else
                        indirUniPathPDF *= bsdfSample->pdf;

                    Float etaScale = w.etaScale;
                    if (bsdfSample->IsTransmission())
                        etaScale *= Sqr(bsdfSample->eta);

                    // Russian roulette
                    SampledSpectrum rrBeta = T_hat * etaScale / indirUniPathPDF.Average();
                    if (rrBeta.MaxComponentValue() < 1 && w.depth > 1) {
                        Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                        if (raySamples.indirect.rr < q) {
                            T_hat = SampledSpectrum(0.f);
                            PBRT_DBG("Path terminated with RR\n");
                        }
                        indirUniPathPDF *= 1 - q;
                        lightPathPDF *= 1 - q;
                    }

                    if (T_hat) {
                        Ray ray = SpawnRay(intr.pi, intr.n, time, wi);
                        if (haveMedia)
                            // TODO: should always just take outside in this case?
                            ray.medium = Dot(ray.d, intr.n) > 0
                                             ? w.mediumInterface.outside
                                             : w.mediumInterface.inside;

                        // || rather than | is intentional, to avoid the read if
                        // possible...
                        bool anyNonSpecularBounces = true;

                        LightSampleContext ctx(intr.pi, intr.n, intr.ns);
                        nextRayQueue->PushIndirectRay(
                            ray, w.depth + 1, ctx, T_hat, indirUniPathPDF, lightPathPDF,
                            lambda, etaScale, bsdfSample->IsSpecular(),
                            anyNonSpecularBounces, w.pixelIndex);

                        PBRT_DBG("Spawned indirect ray at depth %d. "
                                 "Specular %d T_Hat %f %f %f %f indirUniPathPDF %f %f %f "
                                 "%f lightPathPDF %f "
                                 "%f %f %f "
                                 "T_hat/indirUniPathPDF %f %f %f %f\n",
                                 w.depth + 1, int(bsdfSample->IsSpecular()), T_hat[0],
                                 T_hat[1], T_hat[2], T_hat[3], indirUniPathPDF[0],
                                 indirUniPathPDF[1], indirUniPathPDF[2],
                                 indirUniPathPDF[3], lightPathPDF[0], lightPathPDF[1],
                                 lightPathPDF[2], lightPathPDF[3],
                                 SafeDiv(T_hat, indirUniPathPDF)[0],
                                 SafeDiv(T_hat, indirUniPathPDF)[1],
                                 SafeDiv(T_hat, indirUniPathPDF)[2],
                                 SafeDiv(T_hat, indirUniPathPDF)[3]);
                    }
                }
            }

            // Direct lighting...
            if (IsNonSpecular(bsdf.Flags())) {
                LightSampleContext ctx(intr.pi, intr.n, intr.ns);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                if (!sampledLight)
                    return;
                Light light = sampledLight->light;

                pstd::optional<LightLiSample> ls = light.SampleLi(
                    ctx, raySamples.direct.u, lambda, LightSamplingMode::WithMIS);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;

                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                if (!f)
                    return;

                SampledSpectrum T_hat = T_hatp * f * AbsDot(wi, intr.ns);

                PBRT_DBG(
                    "depth %d T_hat %f %f %f %f f %f %f %f %f ls.L %f %f %f %f ls.pdf "
                    "%f\n",
                    w.depth, T_hat[0], T_hat[1], T_hat[2], T_hat[3], f[0], f[1], f[2],
                    f[3], ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                Float lightPDF = ls->pdf * sampledLight->pdf;
                // This causes uniPathPDF to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF =
                    IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<ConcreteBxDF>(wo, wi);
                SampledSpectrum lightPathPDF = uniPathPDF * lightPDF;
                uniPathPDF *= bsdfPDF;

                SampledSpectrum Ld = T_hat * ls->L;

                PBRT_DBG(
                    "depth %d Ld %f %f %f %f "
                    "new T_hat %f %f %f %f T_hat/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                    w.depth, Ld[0], Ld[1], Ld[2], Ld[3], T_hat[0], T_hat[1], T_hat[2],
                    T_hat[3], SafeDiv(T_hat, uniPathPDF)[0],
                    SafeDiv(T_hat, uniPathPDF)[1], SafeDiv(T_hat, uniPathPDF)[2],
                    SafeDiv(T_hat, uniPathPDF)[3], SafeDiv(Ld, uniPathPDF)[0],
                    SafeDiv(Ld, uniPathPDF)[1], SafeDiv(Ld, uniPathPDF)[2],
                    SafeDiv(Ld, uniPathPDF)[3]);

                Ray ray = SpawnRayTo(intr.pi, intr.n, time, ls->pLight.pi, ls->pLight.n);
                if (haveMedia)
                    // TODO: as above, always take outside here?
                    ray.medium = Dot(ray.d, intr.n) > 0 ? w.mediumInterface.outside
                                                        : w.mediumInterface.inside;

                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                       uniPathPDF, lightPathPDF,
                                                       w.pixelIndex});
            }
        });

    TraceShadowRays(wavefrontDepth);
}

}  // namespace pbrt
