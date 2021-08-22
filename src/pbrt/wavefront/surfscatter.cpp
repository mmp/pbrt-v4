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
static inline void rescale(SampledSpectrum *T_hat, SampledSpectrum *lightPathPDF,
                           SampledSpectrum *uniPathPDF) {
    if (T_hat->MaxComponentValue() > 0x1p24f ||
        lightPathPDF->MaxComponentValue() > 0x1p24f ||
        uniPathPDF->MaxComponentValue() > 0x1p24f) {
        *T_hat *= 1.f / 0x1p24f;
        *lightPathPDF *= 1.f / 0x1p24f;
        *uniPathPDF *= 1.f / 0x1p24f;
    } else if (T_hat->MaxComponentValue() < 0x1p-24f ||
               lightPathPDF->MaxComponentValue() < 0x1p-24f ||
               uniPathPDF->MaxComponentValue() < 0x1p-24f) {
        *T_hat *= 0x1p24f;
        *lightPathPDF *= 0x1p24f;
        *uniPathPDF *= 0x1p24f;
    }
}

// EvaluateMaterialCallback Definition
struct EvaluateMaterialCallback {
    int wavefrontDepth;
    WavefrontPathIntegrator *integrator;
    // EvaluateMaterialCallback Public Methods
    template <typename ConcreteMaterial>
    void operator()() {
        if constexpr (!std::is_same_v<ConcreteMaterial, MixMaterial>)
            integrator->EvaluateMaterialAndBSDF<ConcreteMaterial>(wavefrontDepth);
    }
};

// WavefrontPathIntegrator Surface Scattering Methods
void WavefrontPathIntegrator::EvaluateMaterialsAndBSDFs(int wavefrontDepth) {
    ForEachType(EvaluateMaterialCallback{wavefrontDepth, this}, Material::Types());
}

template <typename ConcreteMaterial>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(int wavefrontDepth) {
    int index = Material::TypeIndex<ConcreteMaterial>();
    if (haveBasicEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, BasicTextureEvaluator>(
            basicEvalMaterialQueue, wavefrontDepth);
    if (haveUniversalEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, UniversalTextureEvaluator>(
            universalEvalMaterialQueue, wavefrontDepth);
}

template <typename ConcreteMaterial, typename TextureEvaluator>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(MaterialEvalQueue *evalQueue,
                                                      int wavefrontDepth) {
    // Get BSDF for items in _evalQueue_ and sample illumination
    // Construct _desc_ for material/texture evaluation kernel
    std::string desc = StringPrintf(
        "%s + BxDF eval (%s tex)", ConcreteMaterial::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);
    auto queue = evalQueue->Get<MaterialEvalWorkItem<ConcreteMaterial>>();
    ForAllQueued(
        desc.c_str(), queue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const MaterialEvalWorkItem<ConcreteMaterial> w) {
            // Evaluate material and BSDF for ray intersection
            TextureEvaluator texEval;
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

            // Compute shading normal if bump or normal mapping is being used
            Normal3f ns = w.ns;
            Vector3f dpdus = w.dpdus;
            FloatTexture displacement = w.material->GetDisplacement();
            const Image *normalMap = w.material->GetNormalMap();
            if (displacement || normalMap) {
                // Call _Bump()_ to find shading geometry
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
            using ConcreteBxDF = typename ConcreteMaterial::BxDF;
            ConcreteBxDF bxdf;
            BSDF bsdf = w.material->GetBSDF(texEval, ctx, lambda, &bxdf);
            // Handle terminated secondary wavelengths after BSDF creation
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
                isect.uv = w.uv;
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
                    VisibleSurface(isect, albedo, lambda);
            }

            // Sample BSDF and enqueue indirect ray at intersection point
            Vector3f wo = w.wo;
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            pstd::optional<BSDFSample> bsdfSample = bsdf.Sample_f<ConcreteBxDF>(
                wo, raySamples.indirect.uc, raySamples.indirect.u);
            if (bsdfSample) {
                // Compute updated path throughput and PDFs and enqueue indirect ray
                Vector3f wi = bsdfSample->wi;
                SampledSpectrum beta = w.beta * bsdfSample->f * AbsDot(wi, ns);
                SampledSpectrum inv_w_u = w.inv_w_u, inv_w_l = w.inv_w_u;

                PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n",
                         ConcreteBxDF::Name(), bsdfSample->f[0] * AbsDot(wi, ns),
                         bsdfSample->pdf,
                         bsdfSample->f[0] * AbsDot(wi, ns) / bsdfSample->pdf);

                // Update _inv_w_u_ based on BSDF sample PDF
                if (bsdfSample->pdfIsProportional) {
                    Float pdf = bsdf.PDF<ConcreteBxDF>(wo, wi);
                    beta *= pdf / bsdfSample->pdf;
                    inv_w_u *= pdf;
                } else
                    inv_w_u *= bsdfSample->pdf;

                rescale(&beta, &inv_w_u, &inv_w_l);
                // Update _etaScale_ accounting for BSDF scattering
                Float etaScale = w.etaScale;
                if (bsdfSample->IsTransmission())
                    etaScale *= Sqr(bsdfSample->eta);

                // Apply Russian roulette to indirect ray based on weighted path
                // throughput
                SampledSpectrum rrBeta = beta * etaScale / inv_w_u.Average();
                // Note: depth >= 1 here to match VolPathIntegrator (which increments
                // depth earlier).
                if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (raySamples.indirect.rr < q) {
                        beta = SampledSpectrum(0.f);
                        PBRT_DBG("Path terminated with RR\n");
                    }
                    inv_w_u *= 1 - q;
                    inv_w_l *= 1 - q;
                }

                if (beta) {
                    // Enqueue ray in indirect ray queue or BSSRDF queue, as appropriate
                    if (bsdfSample->IsTransmission() &&
                        w.material->HasSubsurfaceScattering()) {
                        bssrdfEvalQueue->Push(w.material, lambda, beta, inv_w_u,
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
                            ray, w.depth + 1, ctx, beta, inv_w_u, inv_w_l,
                            lambda, etaScale, bsdfSample->IsSpecular(),
                            anyNonSpecularBounces, w.pixelIndex);

                        PBRT_DBG(
                            "Spawned indirect ray at depth %d from w.index %d. "
                            "Specular %d T_Hat %f %f %f %f inv_w_u %f %f %f %f "
                            "inv_w_l %f "
                            "%f %f %f beta/inv_w_u %f %f %f %f\n",
                            w.depth + 1, w.pixelIndex, int(bsdfSample->IsSpecular()),
                            beta[0], beta[1], beta[2], beta[3], inv_w_u[0],
                            inv_w_u[1], inv_w_u[2], inv_w_u[3], inv_w_l[0],
                            inv_w_l[1], inv_w_l[2], inv_w_l[3],
                            SafeDiv(beta, inv_w_u)[0], SafeDiv(beta, inv_w_u)[1],
                            SafeDiv(beta, inv_w_u)[2], SafeDiv(beta, inv_w_u)[3]);
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
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx, raySamples.direct.u, lambda, true);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;
                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                if (!f)
                    return;

                // Compute path throughput and path PDFs for light sample
                SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                PBRT_DBG("w.beta %f %f %f %f f %f %f %f %f dot %f\n", w.beta[0],
                         w.beta[1], w.beta[2], w.beta[3], f[0], f[1], f[2], f[3],
                         AbsDot(wi, ns));

                PBRT_DBG(
                    "me index %d depth %d beta %f %f %f %f f %f %f %f %f ls.L %f %f %f "
                    "%f ls.pdf %f\n",
                    w.pixelIndex, w.depth, beta[0], beta[1], beta[2], beta[3], f[0],
                    f[1], f[2], f[3], ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                Float lightPDF = ls->pdf * sampledLight->p;
                // This causes inv_w_u to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF =
                    IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<ConcreteBxDF>(wo, wi);
                SampledSpectrum inv_w_u = w.inv_w_u * bsdfPDF;
                SampledSpectrum inv_w_l = w.inv_w_u * lightPDF;

                // Enqueue shadow ray with tentative radiance contribution
                SampledSpectrum Ld = beta * ls->L;
                Ray ray = SpawnRayTo(w.pi, w.n, w.time, ls->pLight.pi, ls->pLight.n);
                // Initialize _ray_ medium if media are present
                if (haveMedia)
                    ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                     : w.mediumInterface.inside;

                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                       inv_w_u, inv_w_l,
                                                       w.pixelIndex});

                PBRT_DBG(
                    "w.index %d spawned shadow ray depth %d Ld %f %f %f %f "
                    "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                    w.pixelIndex, w.depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0], beta[1],
                    beta[2], beta[3], SafeDiv(beta, inv_w_u)[0],
                    SafeDiv(beta, inv_w_u)[1], SafeDiv(beta, inv_w_u)[2],
                    SafeDiv(beta, inv_w_u)[3], SafeDiv(Ld, inv_w_u)[0],
                    SafeDiv(Ld, inv_w_u)[1], SafeDiv(Ld, inv_w_u)[2],
                    SafeDiv(Ld, inv_w_u)[3]);
            }
        });
}

}  // namespace pbrt
