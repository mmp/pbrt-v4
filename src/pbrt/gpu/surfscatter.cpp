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

namespace pbrt {

PBRT_CPU_GPU
static inline void rescale(SampledSpectrum &T_hat, SampledSpectrum &lightPathPDF,
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

// GPUPathIntegrator Surface Scattering Methods
template <typename Material, typename TextureEvaluator>
void GPUPathIntegrator::EvaluateMaterialAndBSDF(TextureEvaluator texEval,
                                                MaterialEvalQueue *evalQueue, int depth) {
    std::string name = StringPrintf(
        "%s + BxDF Eval (%s tex)", Material::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");
    RayQueue *rayQueue = CurrentRayQueue(depth);
    RayQueue *nextRayQueue = NextRayQueue(depth);

    ForAllQueued(
        name.c_str(), evalQueue->Get<MaterialEvalWorkItem<Material>>(), maxQueueSize,
        PBRT_GPU_LAMBDA(const MaterialEvalWorkItem<Material> me, int index) {
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
                    pstd::optional<BSDFSample> bs = bsdf.Sample_f<BxDF>(me.wo, uc, u);
                    if (bs)
                        rho += bs->f * AbsDot(bs->wi, ns) / bs->pdf;
                }
                SampledSpectrum albedo = rho / nRhoSamples;

                pixelSampleState.visibleSurface[me.pixelIndex] =
                    VisibleSurface(intr, camera.GetCameraTransform(), albedo, lambda);
            }

            Vector3f wo = me.wo;
            RaySamples raySamples = pixelSampleState.samples[me.pixelIndex];

            // Sample indirect lighting
            pstd::optional<BSDFSample> bsdfSample =
                bsdf.Sample_f<BxDF>(wo, raySamples.indirect.uc, raySamples.indirect.u);
            if (bsdfSample) {
                Vector3f wi = bsdfSample->wi;
                SampledSpectrum T_hat = me.T_hat * bsdfSample->f * AbsDot(wi, ns);
                SampledSpectrum uniPathPDF = me.uniPathPDF, lightPathPDF = uniPathPDF;

                PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n", BxDF::Name(),
                         bsdfSample->f[0] * AbsDot(wi, ns), bsdfSample->pdf,
                         bsdfSample->f[0] * AbsDot(wi, ns) / bsdfSample->pdf);

                if (bsdfSample->pdfIsProportional) {
                    // The PDFs need to be handled slightly differently for
                    // stochastically-sampled layered materials..
                    Float pdf = bsdf.PDF<BxDF>(wo, wi);
                    T_hat *= pdf / bsdfSample->pdf;
                    uniPathPDF *= pdf;
                } else
                    uniPathPDF *= bsdfSample->pdf;
                rescale(T_hat, uniPathPDF, lightPathPDF);

                Float etaScale = me.etaScale;
                if (bsdfSample->IsTransmission())
                    etaScale *= Sqr(bsdf.eta);

                // Russian roulette
                SampledSpectrum rrBeta = T_hat * etaScale / uniPathPDF.Average();
                if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (raySamples.indirect.rr < q) {
                        T_hat = SampledSpectrum(0.f);
                        PBRT_DBG("Path terminated with RR\n");
                    }
                    uniPathPDF *= 1 - q;
                    lightPathPDF *= 1 - q;
                }

                if (T_hat) {
                    if (bsdfSample->IsTransmission() &&
                        material->HasSubsurfaceScattering()) {
                        // There's a BSSRDF and the sampled ray scattered
                        // into the surface; enqueue a work item for
                        // subsurface scattering rather than tracing the
                        // ray.
                        bssrdfEvalQueue->Push(material, lambda, T_hat, uniPathPDF,
                                              Point3f(me.pi), wo, me.n, ns, dpdus, me.uv,
                                              me.mediumInterface, etaScale,
                                              me.pixelIndex);
                    } else {
                        Ray ray = SpawnRay(me.pi, me.n, me.time, wi);
                        if (haveMedia)
                            ray.medium = Dot(ray.d, me.n) > 0 ? me.mediumInterface.outside
                                                              : me.mediumInterface.inside;

                        // || rather than | is intentional, to avoid the read if
                        // possible...
                        bool anyNonSpecularBounces =
                            !bsdfSample->IsSpecular() || me.anyNonSpecularBounces;

                        // Spawn indriect ray.
                        LightSampleContext ctx(
                            me.pi, me.n,
                            ns);  // Note: slightly different than context below. Problem?
                        nextRayQueue->PushIndirect(ray, ctx, T_hat, uniPathPDF,
                                                   lightPathPDF, lambda, etaScale,
                                                   bsdfSample->IsSpecular(),
                                                   anyNonSpecularBounces, me.pixelIndex);

                        PBRT_DBG(
                            "Spawned indirect ray at depth %d from me.index %d. "
                            "Specular %d T_Hat %f %f %f %f uniPathPDF %f %f %f %f "
                            "lightPathPDF %f "
                            "%f %f %f T_hat/uniPathPDF %f %f %f %f\n",
                            depth + 1, index, int(bsdfSample->IsSpecular()), T_hat[0],
                            T_hat[1], T_hat[2], T_hat[3], uniPathPDF[0], uniPathPDF[1],
                            uniPathPDF[2], uniPathPDF[3], lightPathPDF[0],
                            lightPathPDF[1], lightPathPDF[2], lightPathPDF[3],
                            SafeDiv(T_hat, uniPathPDF)[0], SafeDiv(T_hat, uniPathPDF)[1],
                            SafeDiv(T_hat, uniPathPDF)[2], SafeDiv(T_hat, uniPathPDF)[3]);
                    }
                }
            }

            // Sample direct lighting.
            if (bsdf.IsNonSpecular()) {
                // Choose a light source using the LightSampler.
                LightSampleContext ctx(me.pi, me.n, ns);
                if (bsdf.HasReflection() && !bsdf.HasTransmission())
                    ctx.pi = OffsetRayOrigin(ctx.pi, me.n, wo);
                else if (bsdf.HasTransmission() && !bsdf.HasReflection())
                    ctx.pi = OffsetRayOrigin(ctx.pi, me.n, -wo);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                if (!sampledLight)
                    return;
                LightHandle light = sampledLight->light;

                // Remarkably, this substantially improves L1 cache hits with
                // CoatedDiffuseBxDF and gives about a 60% perf. benefit.
                __syncthreads();

                // And now sample the light source itself.
                pstd::optional<LightLiSample> ls = light.SampleLi(
                    ctx, raySamples.direct.u, lambda, LightSamplingMode::WithMIS);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;

                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<BxDF>(wo, wi);
                if (!f)
                    return;

                SampledSpectrum T_hat = me.T_hat * f * AbsDot(wi, ns);
                PBRT_DBG("me.T_hat %f %f %f %f f %f %f %f %f dot %f\n", me.T_hat[0],
                         me.T_hat[1], me.T_hat[2], me.T_hat[3], f[0], f[1], f[2], f[3],
                         AbsDot(wi, ns));

                PBRT_DBG(
                    "me index %d depth %d T_hat %f %f %f %f f %f %f %f %f ls.L %f %f %f "
                    "%f ls.pdf %f\n",
                    index, depth, T_hat[0], T_hat[1], T_hat[2], T_hat[3], f[0], f[1],
                    f[2], f[3], ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                // Compute light and BSDF PDFs for MIS.
                Float lightPDF = ls->pdf * sampledLight->pdf;
                // This causes uniPathPDF to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<BxDF>(wo, wi);
                SampledSpectrum uniPathPDF = me.uniPathPDF * bsdfPDF;
                SampledSpectrum lightPathPDF = me.uniPathPDF * lightPDF;

                SampledSpectrum Ld = SafeDiv(T_hat * ls->L, lambda.PDF());

                Ray ray = SpawnRayTo(me.pi, me.n, me.time, ls->pLight.pi, ls->pLight.n);
                if (haveMedia)
                    ray.medium = Dot(ray.d, me.n) > 0 ? me.mediumInterface.outside
                                                      : me.mediumInterface.inside;

                shadowRayQueue->Push(ray, 1 - ShadowEpsilon, lambda, Ld, uniPathPDF,
                                     lightPathPDF, me.pixelIndex);

                PBRT_DBG(
                    "me.index %d spawned shadow ray depth %d Ld %f %f %f %f "
                    "new T_hat %f %f %f %f T_hat/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                    index, depth, Ld[0], Ld[1], Ld[2], Ld[3], T_hat[0], T_hat[1],
                    T_hat[2], T_hat[3], SafeDiv(T_hat, uniPathPDF)[0],
                    SafeDiv(T_hat, uniPathPDF)[1], SafeDiv(T_hat, uniPathPDF)[2],
                    SafeDiv(T_hat, uniPathPDF)[3], SafeDiv(Ld, uniPathPDF)[0],
                    SafeDiv(Ld, uniPathPDF)[1], SafeDiv(Ld, uniPathPDF)[2],
                    SafeDiv(Ld, uniPathPDF)[3]);
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
        // MixMaterial is resolved immediately in the closest hit shader,
        // so we don't need to worry about it here.
        if constexpr (!std::is_same_v<Material, MixMaterial>)
            integrator->EvaluateMaterialAndBSDF<Material>(depth);
    }
};

void GPUPathIntegrator::EvaluateMaterialsAndBSDFs(int depth) {
    MaterialHandle::ForEachType(EvaluateMaterialCallback{depth, this});
}

}  // namespace pbrt
