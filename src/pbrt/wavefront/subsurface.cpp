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

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);

    ForAllQueued(
        "Get BSSRDF and enqueue probe ray", bssrdfEvalQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const GetBSSRDFAndProbeRayWorkItem w) {
            const SubsurfaceMaterial *material = w.material.Cast<SubsurfaceMaterial>();
            MaterialEvalContext ctx = w.GetMaterialEvalContext();
            SampledWavelengths lambda = w.lambda;
            using BSSRDF = typename SubsurfaceMaterial::BSSRDF;
            BSSRDF bssrdf = material->GetBSSRDF(BasicTextureEvaluator(), ctx, lambda);

            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Float uc = raySamples.subsurface.uc;
            Point2f u = raySamples.subsurface.u;

            pstd::optional<BSSRDFProbeSegment> probeSeg = bssrdf.SampleSp(uc, u);
            if (probeSeg)
                subsurfaceScatterQueue->Push(probeSeg->p0, probeSeg->p1, w.depth,
                                             material, bssrdf, lambda, w.beta, w.r_u,
                                             w.mediumInterface, w.etaScale, w.pixelIndex);
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

            Float pr = w.reservoirPDF * bssrdfSample.pdf[0];
            SampledSpectrum betap = w.beta * bssrdfSample.Sp / pr;
            SampledSpectrum r_u = w.r_u * bssrdfSample.pdf / bssrdfSample.pdf[0];
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
                    SampledSpectrum beta =
                        betap * bsdfSample->f * AbsDot(wi, intr.ns) / bsdfSample->pdf;
                    SampledSpectrum indir_r_u = r_u;

                    PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n",
                             ConcreteBxDF::Name(), bsdfSample->f[0] * AbsDot(wi, intr.ns),
                             bsdfSample->pdf,
                             bsdfSample->f[0] * AbsDot(wi, intr.ns) / bsdfSample->pdf);

                    SampledSpectrum r_l;
                    if (bsdfSample->pdfIsProportional)
                        r_l = r_u / bsdf.PDF<ConcreteBxDF>(wo, bsdfSample->wi);
                    else
                        r_l = r_u / bsdfSample->pdf;

                    Float etaScale = w.etaScale;
                    if (bsdfSample->IsTransmission())
                        etaScale *= Sqr(bsdfSample->eta);

                    // Russian roulette
                    SampledSpectrum rrBeta = beta * etaScale / indir_r_u.Average();
                    if (rrBeta.MaxComponentValue() < 1 && w.depth > 1) {
                        Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                        if (raySamples.indirect.rr < q) {
                            beta = SampledSpectrum(0.f);
                            PBRT_DBG("Path terminated with RR\n");
                        } else
                            beta /= 1 - q;
                    }

                    if (beta) {
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
                            ray, w.depth + 1, ctx, beta, indir_r_u, r_l, lambda,
                            etaScale, bsdfSample->IsSpecular(), anyNonSpecularBounces,
                            w.pixelIndex);

                        PBRT_DBG("Spawned indirect ray at depth %d. "
                                 "Specular %d beta %f %f %f %f indir_r_u %f %f %f %f "
                                 "r_l %f "
                                 "%f %f %f "
                                 "beta/indir_r_u %f %f %f %f\n",
                                 w.depth + 1, int(bsdfSample->IsSpecular()), beta[0],
                                 beta[1], beta[2], beta[3], indir_r_u[0],
                                 indir_r_u[1], indir_r_u[2], indir_r_u[3],
                                 r_l[0], r_l[1], r_l[2], r_l[3],
                                 SafeDiv(beta, indir_r_u)[0],
                                 SafeDiv(beta, indir_r_u)[1],
                                 SafeDiv(beta, indir_r_u)[2],
                                 SafeDiv(beta, indir_r_u)[3]);
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

                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx, raySamples.direct.u, lambda, true);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;

                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                if (!f)
                    return;

                SampledSpectrum beta = betap * f * AbsDot(wi, intr.ns);

                PBRT_DBG(
                    "depth %d beta %f %f %f %f f %f %f %f %f ls.L %f %f %f %f ls.pdf "
                    "%f\n",
                    w.depth, beta[0], beta[1], beta[2], beta[3], f[0], f[1], f[2], f[3],
                    ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                Float lightPDF = ls->pdf * sampledLight->p;
                // This causes r_u to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF =
                    IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<ConcreteBxDF>(wo, wi);
                SampledSpectrum r_l = r_u * lightPDF;
                r_u *= bsdfPDF;

                SampledSpectrum Ld = beta * ls->L;

                PBRT_DBG("depth %d Ld %f %f %f %f "
                         "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                         w.depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0], beta[1], beta[2],
                         beta[3], SafeDiv(beta, r_u)[0], SafeDiv(beta, r_u)[1],
                         SafeDiv(beta, r_u)[2], SafeDiv(beta, r_u)[3],
                         SafeDiv(Ld, r_u)[0], SafeDiv(Ld, r_u)[1],
                         SafeDiv(Ld, r_u)[2], SafeDiv(Ld, r_u)[3]);

                Ray ray = SpawnRayTo(intr.pi, intr.n, time, ls->pLight.pi, ls->pLight.n);
                if (haveMedia)
                    // TODO: as above, always take outside here?
                    ray.medium = Dot(ray.d, intr.n) > 0 ? w.mediumInterface.outside
                                                        : w.mediumInterface.inside;

                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                       r_u, r_l, w.pixelIndex});
            }
        });

    TraceShadowRays(wavefrontDepth);
}

}  // namespace pbrt
