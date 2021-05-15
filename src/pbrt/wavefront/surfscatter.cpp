// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/bxdfs.h>
#include <pbrt/cameras.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/integrator.h>

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

// EvaluateMaterialCallback Definition
struct EvaluateMaterialCallback {
    int wavefrontDepth;
    WavefrontPathIntegrator *integrator;
    // EvaluateMaterialCallback Public Methods
    template <typename Mtl>
    void operator()() {
        if constexpr (!std::is_same_v<Mtl, MixMaterial>)
            integrator->EvaluateMaterialAndBSDF<Mtl>(wavefrontDepth);
    }
};

// WavefrontPathIntegrator Surface Scattering Methods
void WavefrontPathIntegrator::EvaluateMaterialsAndBSDFs(int wavefrontDepth) {
    ForEachType(EvaluateMaterialCallback{wavefrontDepth, this}, Material::Types());
}

template <typename Mtl>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(int wavefrontDepth) {
    if (haveBasicEvalMaterial[Material::TypeIndex<Mtl>()])
        EvaluateMaterialAndBSDF<Mtl>(BasicTextureEvaluator(), basicEvalMaterialQueue,
                                     wavefrontDepth);
    if (haveUniversalEvalMaterial[Material::TypeIndex<Mtl>()])
        EvaluateMaterialAndBSDF<Mtl>(UniversalTextureEvaluator(),
                                     universalEvalMaterialQueue, wavefrontDepth);
}

template <typename Mtl, typename TextureEvaluator>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(TextureEvaluator texEval,
                                                      MaterialEvalQueue *evalQueue,
                                                      int wavefrontDepth) {
    // Construct _name_ for material/texture evaluator kernel
    std::string name = StringPrintf(
        "%s + BxDF Eval (%s tex)", Mtl::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);
    ForAllQueued(
        name.c_str(), evalQueue->Get<MaterialEvalWorkItem<Mtl>>(), maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const MaterialEvalWorkItem<Mtl> w) {
            // Evaluate material and BSDF for ray intersection
            // Compute differentials for position and $(u,v)$ at intersection point
            Vector3f dpdx, dpdy;
            Float dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
            camera.Approximate_dp_dxy(Point3f(w.pi), w.n, w.time, samplesPerPixel, &dpdx,
                                      &dpdy);
            Vector3f dpdu = w.dpdu, dpdv = w.dpdv;
            // Estimate screen-space change in $(u,v)$
            Float a00 = Dot(dpdu, dpdu), a01 = Dot(dpdu, dpdv), a11 = Dot(dpdv, dpdv);
            Float invDet = 1 / (DifferenceOfProducts(a00, a11, a01, a01));

            Float b0x = Dot(dpdu, dpdx), b1x = Dot(dpdv, dpdx);
            Float b0y = Dot(dpdu, dpdy), b1y = Dot(dpdv, dpdy);

            /* Set the UV partials to zero if dpdu and/or dpdv == 0 */
            invDet = IsFinite(invDet) ? invDet : 0.f;

            dudx = DifferenceOfProducts(a11, b0x, a01, b1x) * invDet;
            dvdx = DifferenceOfProducts(a00, b1x, a01, b0x) * invDet;

            dudy = DifferenceOfProducts(a11, b0y, a01, b1y) * invDet;
            dvdy = DifferenceOfProducts(a00, b1y, a01, b0y) * invDet;

            dudx = IsFinite(dudx) ? Clamp(dudx, -1e8f, 1e8f) : 0.f;
            dvdx = IsFinite(dvdx) ? Clamp(dvdx, -1e8f, 1e8f) : 0.f;
            dudy = IsFinite(dudy) ? Clamp(dudy, -1e8f, 1e8f) : 0.f;
            dvdy = IsFinite(dvdy) ? Clamp(dvdy, -1e8f, 1e8f) : 0.f;

            // Apply bump mapping if material has a displacement texture
            Normal3f ns = w.ns;
            Vector3f dpdus = w.dpdus;
            FloatTexture displacement = w.material->GetDisplacement();
            const Image *normalMap = w.material->GetNormalMap();
            if (displacement || normalMap) {
                if (displacement)
                    DCHECK(texEval.CanEvaluate({displacement}, {}));
                BumpEvalContext bctx = w.GetBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                Bump(texEval, displacement, normalMap, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);
            }

            // Get BSDF at intersection point
            SampledWavelengths lambda = w.lambda;
            MaterialEvalContext ctx =
                w.GetMaterialEvalContext(dudx, dudy, dvdx, dvdy, ns, dpdus);
            using BxDF = typename Mtl::BxDF;
            BxDF bxdf;
            BSDF bsdf = w.material->GetBSDF(texEval, ctx, lambda, &bxdf);
            if (lambda.SecondaryTerminated())
                pixelSampleState.lambda[w.pixelIndex] = lambda;

            // Regularize BSDF, if appropriate
            if (regularize && w.anyNonSpecularBounces)
                bsdf.Regularize();

            // Initialize _VisibleSurface_ at first intersection if necessary
            if (w.depth == 0 && initializeVisibleSurface) {
                SurfaceInteraction isect;
                isect.pi = w.pi;
                isect.n = w.n;
                isect.shading.n = ns;
                isect.wo = w.wo;
                isect.time = w.time;
                isect.dpdx = dpdx;
                isect.dpdy = dpdy;

                // Estimate BSDF's albedo
                // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
                constexpr int nRhoSamples = 16;
                const Float ucRho[nRhoSamples] = {
                    0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                    0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                    0.9674518,  0.2995429,  0.5083201, 0.047338516};
                const Point2f uRho[nRhoSamples] = {
                    Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                    Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                    Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                    Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                    Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                    Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                    Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                    Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

                SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

                pixelSampleState.visibleSurface[w.pixelIndex] =
                    VisibleSurface(isect, camera.GetCameraTransform(), albedo, lambda);
            }

            // Sample BSDF and enqueue indirect ray at intersection point
            Vector3f wo = w.wo;
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            pstd::optional<BSDFSample> bsdfSample =
                bsdf.Sample_f<BxDF>(wo, raySamples.indirect.uc, raySamples.indirect.u);
            if (bsdfSample) {
                // Compute updated path throughput and PDFs and enqueue indirect ray
                Vector3f wi = bsdfSample->wi;
                SampledSpectrum T_hat = w.T_hat * bsdfSample->f * AbsDot(wi, ns);
                SampledSpectrum uniPathPDF = w.uniPathPDF, lightPathPDF = w.uniPathPDF;

                PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n", BxDF::Name(),
                         bsdfSample->f[0] * AbsDot(wi, ns), bsdfSample->pdf,
                         bsdfSample->f[0] * AbsDot(wi, ns) / bsdfSample->pdf);

                // Update _uniPathPDF_ based on BSDF sample PDF
                if (bsdfSample->pdfIsProportional) {
                    Float pdf = bsdf.PDF<BxDF>(wo, wi);
                    T_hat *= pdf / bsdfSample->pdf;
                    uniPathPDF *= pdf;
                } else
                    uniPathPDF *= bsdfSample->pdf;

                rescale(T_hat, uniPathPDF, lightPathPDF);
                // Update _etaScale_ accounting for BSDF scattering
                Float etaScale = w.etaScale;
                if (bsdfSample->IsTransmission())
                    etaScale *= Sqr(bsdfSample->eta);

                // Apply Russian roulette to indirect ray based on weighted path
                // throughput
                SampledSpectrum rrBeta = T_hat * etaScale / uniPathPDF.Average();
                // Note: depth >= 1 here to match VolPathIntegrator (which increments
                // depth earlier).
                if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (raySamples.indirect.rr < q) {
                        T_hat = SampledSpectrum(0.f);
                        PBRT_DBG("Path terminated with RR\n");
                    }
                    uniPathPDF *= 1 - q;
                    lightPathPDF *= 1 - q;
                }

                if (T_hat) {
                    // Enqueue ray in BSSRDF or indirect ray queue, as appropriate
                    if (bsdfSample->IsTransmission() &&
                        w.material->HasSubsurfaceScattering()) {
                        bssrdfEvalQueue->Push(w.material, lambda, T_hat, uniPathPDF,
                                              Point3f(w.pi), wo, w.n, ns, dpdus, w.uv,
                                              w.depth, w.mediumInterface, etaScale,
                                              w.pixelIndex);
                    } else {
                        // Initialize spawned ray and enqueue for next ray depth
                        Ray ray = SpawnRay(w.pi, w.n, w.time, wi);
                        // Initialize _ray_ medium if media are present
                        if (haveMedia)
                            ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                             : w.mediumInterface.inside;

                        bool anyNonSpecularBounces =
                            !bsdfSample->IsSpecular() || w.anyNonSpecularBounces;
                        // NOTE: slightly different than context below. Problem?
                        LightSampleContext ctx(w.pi, w.n, ns);
                        nextRayQueue->PushIndirectRay(
                            ray, w.depth + 1, ctx, T_hat, uniPathPDF, lightPathPDF,
                            lambda, etaScale, bsdfSample->IsSpecular(),
                            anyNonSpecularBounces, w.pixelIndex);

                        PBRT_DBG(
                            "Spawned indirect ray at depth %d from w.index %d. "
                            "Specular %d T_Hat %f %f %f %f uniPathPDF %f %f %f %f "
                            "lightPathPDF %f "
                            "%f %f %f T_hat/uniPathPDF %f %f %f %f\n",
                            w.depth + 1, w.pixelIndex, int(bsdfSample->IsSpecular()),
                            T_hat[0], T_hat[1], T_hat[2], T_hat[3], uniPathPDF[0],
                            uniPathPDF[1], uniPathPDF[2], uniPathPDF[3], lightPathPDF[0],
                            lightPathPDF[1], lightPathPDF[2], lightPathPDF[3],
                            SafeDiv(T_hat, uniPathPDF)[0], SafeDiv(T_hat, uniPathPDF)[1],
                            SafeDiv(T_hat, uniPathPDF)[2], SafeDiv(T_hat, uniPathPDF)[3]);
                    }
                }
            }

            // Sample light and enqueue shadow ray at intersection point
            BxDFFlags flags = bsdf.Flags();
            if (IsNonSpecular(flags)) {
                // Choose a light source using the _LightSampler_
                LightSampleContext ctx(w.pi, w.n, ns);
                if (IsReflective(flags) && !IsTransmissive(flags))
                    ctx.pi = OffsetRayOrigin(ctx.pi, w.n, wo);
                else if (IsTransmissive(flags) && IsReflective(flags))
                    ctx.pi = OffsetRayOrigin(ctx.pi, w.n, -wo);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                if (!sampledLight)
                    return;
                Light light = sampledLight->light;

                // Sample light source and evaluate BSDF for direct lighting
                pstd::optional<LightLiSample> ls = light.SampleLi(
                    ctx, raySamples.direct.u, lambda, LightSamplingMode::WithMIS);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;
                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<BxDF>(wo, wi);
                if (!f)
                    return;

                // Compute path throughput and path PDFs for light sample
                SampledSpectrum T_hat = w.T_hat * f * AbsDot(wi, ns);
                PBRT_DBG("w.T_hat %f %f %f %f f %f %f %f %f dot %f\n", w.T_hat[0],
                         w.T_hat[1], w.T_hat[2], w.T_hat[3], f[0], f[1], f[2], f[3],
                         AbsDot(wi, ns));

                PBRT_DBG(
                    "me index %d depth %d T_hat %f %f %f %f f %f %f %f %f ls.L %f %f %f "
                    "%f ls.pdf %f\n",
                    w.pixelIndex, w.depth, T_hat[0], T_hat[1], T_hat[2], T_hat[3], f[0],
                    f[1], f[2], f[3], ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                Float lightPDF = ls->pdf * sampledLight->pdf;
                // This causes uniPathPDF to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<BxDF>(wo, wi);
                SampledSpectrum uniPathPDF = w.uniPathPDF * bsdfPDF;
                SampledSpectrum lightPathPDF = w.uniPathPDF * lightPDF;

                // Enqueue shadow ray with tentative radiance contribution
                SampledSpectrum Ld = T_hat * ls->L;
                Ray ray = SpawnRayTo(w.pi, w.n, w.time, ls->pLight.pi, ls->pLight.n);
                // Initialize _ray_ medium if media are present
                if (haveMedia)
                    ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                     : w.mediumInterface.inside;

                shadowRayQueue->Push(ray, 1 - ShadowEpsilon, lambda, Ld, uniPathPDF,
                                     lightPathPDF, w.pixelIndex);

                PBRT_DBG(
                    "w.index %d spawned shadow ray depth %d Ld %f %f %f %f "
                    "new T_hat %f %f %f %f T_hat/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                    w.pixelIndex, w.depth, Ld[0], Ld[1], Ld[2], Ld[3], T_hat[0], T_hat[1],
                    T_hat[2], T_hat[3], SafeDiv(T_hat, uniPathPDF)[0],
                    SafeDiv(T_hat, uniPathPDF)[1], SafeDiv(T_hat, uniPathPDF)[2],
                    SafeDiv(T_hat, uniPathPDF)[3], SafeDiv(Ld, uniPathPDF)[0],
                    SafeDiv(Ld, uniPathPDF)[1], SafeDiv(Ld, uniPathPDF)[2],
                    SafeDiv(Ld, uniPathPDF)[3]);
            }
        });
}

}  // namespace pbrt
