// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/bxdfs.h>
#include <pbrt/cameras.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <type_traits>

#ifdef PBRT_GPU_DBG
#ifndef TO_STRING
#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x
#endif  // !TO_STRING
#define DBG(...) printf(__FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#else
#define DBG(...)
#endif  // PBRT_GPU_DBG

namespace pbrt {

template <typename Material, typename TextureEvaluator>
void GPUPathIntegrator::EvaluateMaterialAndBSDF(TextureEvaluator texEval,
                                                MaterialEvalQueue *evalQueue, int depth) {
    std::string name = StringPrintf(
        "%s + BxDF Eval (%s tex)", Material::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    ForAllQueued(
        name.c_str(), evalQueue->Get<Material>(), maxQueueSize,
        [=] PBRT_GPU(const MaterialEvalWorkItem<Material> me, int index) {
            const Material *material = me.material;

            Normal3f ns = me.ns;
            Vector3f dpdus = me.dpdus;

            FloatTextureHandle displacement = material->GetDisplacement();
            if (displacement) {
                // Compute shading normal (and shading dpdu) via bump mapping.
                DCHECK(texEval.CanEvaluate({displacement}, {}));

                BumpEvalContext bctx = me.GetBumpEvalContext();
                Vector3f dpdvs;
                Bump(texEval, displacement, bctx, &dpdus, &dpdvs);

                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, me.n);
            }

            // Evaluate the material (and thence, its textures), to get the BSDF.
            SampledWavelengths lambda = me.lambda;
            MaterialEvalContext ctx = me.GetMaterialEvalContext(ns, dpdus);
            using BxDF = typename Material::BxDF;
            BxDF bxdf;
            BSDF bsdf = material->GetBSDF(texEval, ctx, lambda, &bxdf);

            // BSDF regularization, if appropriate.
            if (regularize && me.anyNonSpecularBounces)
                bsdf.Regularize();

            if (depth == 0 && initializeVisibleSurface) {
                SurfaceInteraction intr;
                intr.pi = me.pi;
                intr.n = me.n;
                intr.shading.n = ns;
                intr.wo = me.wo;
                // TODO: intr.time

                // Estimate BSDF's albedo
                constexpr int nRhoSamples = 16;
                SampledSpectrum rho(0.f);
                for (int i = 0; i < nRhoSamples; ++i) {
                    // Generate sample for hemispherical-directional reflectance
                    Float uc = RadicalInverse(0, i + 1);
                    Point2f u(RadicalInverse(1, i + 1), RadicalInverse(2, i + 1));

                    // Estimate one term of $\rho_\roman{hd}$
                    BSDFSample bs = bsdf.Sample_f(me.wo, uc, u);
                    if (bs && bs.pdf > 0)
                        rho += bs.f * AbsDot(bs.wi, ns) / bs.pdf;
                }
                SampledSpectrum albedo = rho / nRhoSamples;

                pixelSampleState.visibleSurface[me.pixelIndex] =
                    VisibleSurface(intr, camera.GetCameraTransform(), albedo, lambda);
            }

            Vector3f wo = me.wo;
            RaySamples raySamples = rayQueues[depth & 1]->raySamples[me.rayIndex];

            // Sample indirect lighting
            BSDFSample bsdfSample =
                bsdf.Sample_f<BxDF>(wo, raySamples.indirect.uc, raySamples.indirect.u);
            if (bsdfSample && bsdfSample.f) {
                Vector3f wi = bsdfSample.wi;
                SampledSpectrum beta = me.beta * bsdfSample.f * AbsDot(wi, ns);
                SampledSpectrum pdfUni = me.pdfUni, pdfNEE = pdfUni;

                DBG("%s f*cos[0] %f bsdfSample.pdf %f f*cos/pdf %f\n", BxDF::Name(),
                    bsdfSample.f[0] * AbsDot(wi, ns), bsdfSample.pdf,
                    bsdfSample.f[0] * AbsDot(wi, ns) / bsdfSample.pdf);

                if (bsdf.SampledPDFIsProportional()) {
                    // The PDFs need to be handled slightly differently for
                    // stochastically-sampled layered materials..
                    Float pdf = bsdf.PDF(wo, wi);
                    beta *= pdf / bsdfSample.pdf;
                    pdfUni *= pdf;
                } else
                    pdfUni *= bsdfSample.pdf;

                Float etaScale = me.etaScale;
                if (bsdfSample.IsTransmission())
                    etaScale *= Sqr(bsdf.eta);

                // Russian roulette
                SampledSpectrum rrBeta = beta * etaScale / pdfUni.Average();
                if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (raySamples.indirect.rr < q) {
                        beta = SampledSpectrum(0.f);
                        DBG("Path terminated with RR ray index %d\n", me.rayIndex);
                    }
                    pdfUni *= 1 - q;
                    pdfNEE *= 1 - q;
                }

                if (beta) {
                    if (bsdfSample.IsTransmission() &&
                        material->HasSubsurfaceScattering()) {
                        // There's a BSSRDF and the sampled ray scattered
                        // into the surface; enqueue a work item for
                        // subsurface scattering rather than tracing the
                        // ray.
                        bssrdfEvalQueue->Push(material, lambda, beta, pdfUni,
                                              Point3f(me.pi), wo, me.n, ns,
                                              dpdus, me.uv, me.mediumInterface,
                                              me.rayIndex);
                    } else {
                        Ray ray = SpawnRay(me.pi, me.n, me.time, wi);
                        if (haveMedia)
                            ray.medium = Dot(ray.d, me.n) > 0 ? me.mediumInterface.outside
                                                              : me.mediumInterface.inside;

                        // || rather than | is intentional, to avoid the read if
                        // possible...
                        bool anyNonSpecularBounces =
                            !bsdfSample.IsSpecular() || me.anyNonSpecularBounces;

                        // Spawn indriect ray.
                        rayQueues[(depth + 1) & 1]->PushIndirect(
                            ray, me.pi, me.n, ns, beta, pdfUni, pdfNEE, lambda, etaScale,
                            bsdfSample.IsSpecular(), anyNonSpecularBounces,
                            me.pixelIndex);

                        DBG("Spawned indirect ray at depth %d from prev ray index %d. "
                            "Specular %d Beta %f %f %f %f pdfUni %f %f %f %f pdfNEE %f "
                            "%f %f %f beta/pdfUni %f %f %f %f\n",
                            depth + 1, int(me.rayIndex), int(bsdfSample.IsSpecular()),
                            beta[0], beta[1], beta[2], beta[3], pdfUni[0], pdfUni[1],
                            pdfUni[2], pdfUni[3], pdfNEE[0], pdfNEE[1], pdfNEE[2],
                            pdfNEE[3], SafeDiv(beta, pdfUni)[0], SafeDiv(beta, pdfUni)[1],
                            SafeDiv(beta, pdfUni)[2], SafeDiv(beta, pdfUni)[3]);
                    }
                }
            }

            // Sample direct lighting.
            if (!bsdf.IsSpecular()) {
                // Choose a light source using the LightSampler.
                LightSampleContext ctx(me.pi, me.n, ns);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                if (!sampledLight)
                    return;
                LightHandle light = sampledLight->light;

                // Remarkably, this substantially improves L1 cache hits with
                // CoatedDiffuseBxDF and gives about a 60% perf. benefit.
                __syncthreads();

                // And now sample the light source itself.
                LightLiSample ls = light.SampleLi(ctx, raySamples.direct.u, lambda,
                                                  LightSamplingMode::WithMIS);
                if (!ls || !ls.L)
                    return;

                Vector3f wi = ls.wi;
                SampledSpectrum f = bsdf.f<BxDF>(wo, wi);
                if (!f)
                    return;

                SampledSpectrum beta = me.beta * f * AbsDot(wi, ns);

                DBG("ray index %d depth %d beta %f %f %f %f f %f %f %f %f ls.L %f %f %f "
                    "%f ls.pdf %f\n",
                    me.rayIndex, depth, beta[0], beta[1], beta[2], beta[3], f[0], f[1],
                    f[2], f[3], ls.L[0], ls.L[1], ls.L[2], ls.L[3], ls.pdf);

                // Compute light and BSDF PDFs for MIS.
                Float lightPDF = ls.pdf * sampledLight->pdf;
                // This causes pdfUni to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<BxDF>(wo, wi);
                SampledSpectrum pdfUni = me.pdfUni * bsdfPDF;
                SampledSpectrum pdfNEE = me.pdfUni * lightPDF;

                SampledSpectrum Ld = beta * ls.L;

                Ray ray = SpawnRayTo(me.pi, me.n, me.time, ls.pLight.pi, ls.pLight.n);
                if (haveMedia)
                    ray.medium = Dot(ray.d, me.n) > 0 ? me.mediumInterface.outside
                                                      : me.mediumInterface.inside;

                shadowRayQueue->Push(ray, 1 - ShadowEpsilon, lambda, Ld,
                                     pdfUni, pdfNEE, me.pixelIndex);

                DBG("ray index %d spawned shadow ray depth %d Ld %f %f %f %f "
                    "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                    me.rayIndex, depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0], beta[1],
                    beta[2], beta[3], SafeDiv(beta, pdfUni)[0], SafeDiv(beta, pdfUni)[1],
                    SafeDiv(beta, pdfUni)[2], SafeDiv(beta, pdfUni)[3],
                    SafeDiv(Ld, pdfUni)[0], SafeDiv(Ld, pdfUni)[1],
                    SafeDiv(Ld, pdfUni)[2], SafeDiv(Ld, pdfUni)[3]);
            }
        });
}

template <typename Material>
void GPUPathIntegrator::EvaluateMaterialAndBSDF(int depth) {
    if (haveBasicEvalMaterial[MaterialHandle::TypeIndex<Material>()])
        EvaluateMaterialAndBSDF<Material>(BasicTextureEvaluator(), basicEvalMaterialQueue,
                                          depth);

    if (haveUniversalEvalMaterial[MaterialHandle::TypeIndex<Material>()])
        EvaluateMaterialAndBSDF<Material>(UniversalTextureEvaluator(),
                                          universalEvalMaterialQueue, depth);
}

struct EvaluateMaterialCallback {
    int depth;
    GPUPathIntegrator *integrator;
    template <typename Material>
    void operator()() {
        if constexpr (!std::is_same_v<Material, MixMaterial>)
            integrator->EvaluateMaterialAndBSDF<Material>(depth);
    }
};

template <typename TextureEvaluator>
void GPUPathIntegrator::ResolveMixMaterial(TextureEvaluator texEval,
                                           MaterialEvalQueue *evalQueue) {
    std::string name = StringPrintf("Resolve MixMaterial (%s tex)",
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    ForAllQueued(
        name.c_str(), evalQueue->Get<MixMaterial>(), maxQueueSize,
        [=] PBRT_GPU(const MaterialEvalWorkItem<MixMaterial> me, int index) {
            MaterialEvalContext ctx = me.GetMaterialEvalContext(me.ns, me.dpdus);
            MaterialHandle m = me.material->ChooseMaterial(TextureEvaluator(), ctx);

            auto enqueue = [=](auto ptr) {
                using NewMaterial = typename std::remove_reference_t<decltype(*ptr)>;
                evalQueue->Push<NewMaterial>(MaterialEvalWorkItem<NewMaterial>{
                        ptr, me.lambda, me.beta, me.pdfUni, me.pi, me.n, me.ns,
                        me.dpdus, me.dpdvs, me.dndus, me.dndvs,
                        me.wo, me.uv, me.time, me.anyNonSpecularBounces, me.etaScale,
                        me.mediumInterface, me.rayIndex, me.pixelIndex});
            };
            m.Dispatch(enqueue);
        });
}

void GPUPathIntegrator::EvaluateMaterialsAndBSDFs(int depth) {
    // Resolve Mix materials first
    if (haveBasicEvalMaterial[MaterialHandle::TypeIndex<MixMaterial>()])
        ResolveMixMaterial(BasicTextureEvaluator(),
                           basicEvalMaterialQueue);

    if (haveUniversalEvalMaterial[MaterialHandle::TypeIndex<MixMaterial>()])
        ResolveMixMaterial(UniversalTextureEvaluator(),
                           universalEvalMaterialQueue);

    MaterialHandle::ForEachType(EvaluateMaterialCallback{depth, this});
}

}  // namespace pbrt
