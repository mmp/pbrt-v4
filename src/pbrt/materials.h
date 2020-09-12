// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_MATERIALS_H
#define PBRT_MATERIALS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bssrdf.h>
#include <pbrt/base/material.h>
#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>

#include <memory>
#include <type_traits>

namespace pbrt {

// MaterialEvalContext Definition
struct MaterialEvalContext : public TextureEvalContext {
    MaterialEvalContext() = default;
    PBRT_CPU_GPU
    MaterialEvalContext(const SurfaceInteraction &si)
        : TextureEvalContext(si),
          wo(si.wo),
          n(si.n),
          ns(si.shading.n),
          dpdus(si.shading.dpdu) {}

    Vector3f wo;
    Normal3f n, ns;
    Vector3f dpdus;
};

// BumpEvalContext Definition
struct BumpEvalContext {
    BumpEvalContext() = default;
    PBRT_CPU_GPU
    BumpEvalContext(const SurfaceInteraction &si)
        : p(si.p()),
          uv(si.uv),
          dudx(si.dudx),
          dudy(si.dudy),
          dvdx(si.dvdx),
          dvdy(si.dvdy),
          dpdx(si.dpdx),
          dpdy(si.dpdy),
          faceIndex(si.faceIndex) {
        shading.n = si.shading.n;
        shading.dpdu = si.shading.dpdu;
        shading.dpdv = si.shading.dpdv;
        shading.dndu = si.shading.dndu;
        shading.dndv = si.shading.dndv;
    }

    PBRT_CPU_GPU
    operator TextureEvalContext() const {
        return TextureEvalContext(p, dpdx, dpdy, uv, dudx, dudy, dvdx, dvdy, faceIndex);
    }

    Point3f p;
    Point2f uv;
    struct {
        Normal3f n;
        Vector3f dpdu, dpdv;
        Normal3f dndu, dndv;
    } shading;
    Float dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
    Vector3f dpdx, dpdy;
    int faceIndex = 0;
};

// Bump-mapping Function Definitions
template <typename TextureEvaluator>
PBRT_CPU_GPU void Bump(TextureEvaluator texEval, FloatTextureHandle displacement,
                       const BumpEvalContext &si, Vector3f *dpdu, Vector3f *dpdv) {
    DCHECK(displacement != nullptr);
    DCHECK(texEval.CanEvaluate({displacement}, {}));
    // Compute offset positions and evaluate displacement texture
    TextureEvalContext shiftedCtx = si;
    // Shift _shiftedCtx_ _du_ in the $u$ direction
    Float du = .5f * (std::abs(si.dudx) + std::abs(si.dudy));
    if (du == 0)
        du = .0005f;
    shiftedCtx.p = si.p + du * si.shading.dpdu;
    shiftedCtx.uv = si.uv + Vector2f(du, 0.f);

    Float uDisplace = texEval(displacement, shiftedCtx);
    // Shift _shiftedCtx_ _dv_ in the $v$ direction
    Float dv = .5f * (std::abs(si.dvdx) + std::abs(si.dvdy));
    if (dv == 0)
        dv = .0005f;
    shiftedCtx.p = si.p + dv * si.shading.dpdv;
    shiftedCtx.uv = si.uv + Vector2f(0.f, dv);

    Float vDisplace = texEval(displacement, shiftedCtx);
    Float displace = texEval(displacement, si);

    // Compute bump-mapped differential geometry
    *dpdu = si.shading.dpdu + (uDisplace - displace) / du * Vector3f(si.shading.n) +
            displace * Vector3f(si.shading.dndu);
    *dpdv = si.shading.dpdv + (vDisplace - displace) / dv * Vector3f(si.shading.n) +
            displace * Vector3f(si.shading.dndv);
}

// DielectricMaterial Definition
class DielectricMaterial {
  public:
    using BxDF = DielectricInterfaceBxDF;
    using BSSRDF = void;
    // DielectricMaterial Public Methods
    DielectricMaterial(FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                       FloatTextureHandle etaF, SpectrumTextureHandle etaS,
                       FloatTextureHandle displacement, bool remapRoughness)
        : displacement(displacement),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          etaF(etaF),
          etaS(etaS),
          remapRoughness(remapRoughness) {
        CHECK((bool)etaF ^ (bool)etaS);
    }

    static const char *Name() { return "DielectricMaterial"; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({etaF, uRoughness, vRoughness}, {etaS});
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static DielectricMaterial *Create(const TextureParameterDictionary &parameters,
                                      const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda,
                              DielectricInterfaceBxDF *bxdf) const {
        // Compute index of refraction for dielectric material
        Float eta;
        if (etaF)
            eta = texEval(etaF, ctx);
        else {
            eta = texEval(etaS, ctx, lambda)[0];
            lambda.TerminateSecondary();
        }

        // Create microfacet distribution for \use{DielectricMaterial}
        Float urough = texEval(uRoughness, ctx), vrough = texEval(vRoughness, ctx);
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        TrowbridgeReitzDistribution distrib(urough, vrough);

        // Return BSDF for \use{DielectricMaterial}
        *bxdf = DielectricInterfaceBxDF(eta, distrib);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf, eta);
    }

  private:
    // DielectricMaterial Private Members
    FloatTextureHandle displacement;
    FloatTextureHandle uRoughness, vRoughness, etaF;
    SpectrumTextureHandle etaS;
    bool remapRoughness;
};

// ThinDielectricMaterial Definition
class ThinDielectricMaterial {
  public:
    using BxDF = ThinDielectricBxDF;
    using BSSRDF = void;
    // ThinDielectricMaterial Public Methods
    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({etaF}, {etaS});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda,
                              ThinDielectricBxDF *bxdf) const {
        // Compute index of refraction for dielectric material
        Float eta;
        if (etaF)
            eta = texEval(etaF, ctx);
        else {
            eta = texEval(etaS, ctx, lambda)[0];
            lambda.TerminateSecondary();
        }

        // Return BSDF for \use{ThinDielectricMaterial}
        *bxdf = ThinDielectricBxDF(eta);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf, eta);
    }

    PBRT_CPU_GPU
    bool IsTransparent() const { return true; }

    ThinDielectricMaterial(FloatTextureHandle etaF, SpectrumTextureHandle etaS,
                           FloatTextureHandle displacement)
        : displacement(displacement), etaF(etaF), etaS(etaS) {
        CHECK((bool)etaF ^ (bool)etaS);
    }

    static const char *Name() { return "ThinDielectricMaterial"; }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static ThinDielectricMaterial *Create(const TextureParameterDictionary &parameters,
                                          const FileLoc *loc, Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

  private:
    // ThinDielectricMaterial Private Data
    FloatTextureHandle displacement;
    FloatTextureHandle etaF;
    SpectrumTextureHandle etaS;
};

// MixMaterial Definition
class MixMaterial {
  public:
    using BxDF = void;  // shouldn't be accessed...
    using BSSRDF = void;
    // MixMaterial Public Methods
    MixMaterial(MaterialHandle m[2], FloatTextureHandle amount) : amount(amount) {
        materials[0] = m[0];
        materials[1] = m[1];
    }

    PBRT_CPU_GPU
    MaterialHandle GetMaterial(int i) const { return materials[i]; }

    static const char *Name() { return "MixMaterial"; }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const {
#ifndef PBRT_IS_GPU_CODE
        LOG_FATAL("Shouldn't be called");
#endif
        return nullptr;
    }

    static MixMaterial *Create(MaterialHandle materials[2],
                               const TextureParameterDictionary &parameters,
                               const FileLoc *loc, Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {
#ifndef PBRT_IS_GPU_CODE
        LOG_FATAL("Shouldn't be called");
#endif
    }

    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({amount}, {});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU MaterialHandle ChooseMaterial(TextureEvaluator texEval,
                                               MaterialEvalContext ctx) const {
        Float amt = texEval(amount, ctx);
        if (amt <= 0)
            return materials[0];
        if (amt >= 1)
            return materials[1];

        Float u = uint32_t(Hash(ctx.p, ctx.wo)) * 0x1p-32;
        return (amt < u) ? materials[0] : materials[1];
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda, void *bxdf) const {
#ifndef PBRT_IS_GPU_CODE
        LOG_FATAL("Shouldn't be called");
#endif
        return {};
    }

    PBRT_CPU_GPU
    bool IsTransparent() const {
#ifdef PBRT_IS_GPU_CODE
        return false;
#else
        return materials[0].IsTransparent() || materials[1].IsTransparent();
#endif
    }

  private:
    // MixMaterial Private Members
    FloatTextureHandle amount;
    MaterialHandle materials[2];
};

// HairMaterial Definition
class HairMaterial {
  public:
    using BxDF = HairBxDF;
    using BSSRDF = void;

    // HairMaterial Public Methods
    HairMaterial(SpectrumTextureHandle sigma_a, SpectrumTextureHandle color,
                 FloatTextureHandle eumelanin, FloatTextureHandle pheomelanin,
                 FloatTextureHandle eta, FloatTextureHandle beta_m,
                 FloatTextureHandle beta_n, FloatTextureHandle alpha)
        : sigma_a(sigma_a),
          color(color),
          eumelanin(eumelanin),
          pheomelanin(pheomelanin),
          eta(eta),
          beta_m(beta_m),
          beta_n(beta_n),
          alpha(alpha) {}

    static const char *Name() { return "HairMaterial"; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({eumelanin, pheomelanin, eta, beta_m, beta_n, alpha},
                                   {sigma_a, color});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda, HairBxDF *bxdf) const {
        Float bm = std::max<Float>(1e-2, texEval(beta_m, ctx));
        Float bn = std::max<Float>(1e-2, texEval(beta_n, ctx));
        Float a = texEval(alpha, ctx);
        Float e = texEval(eta, ctx);

        SampledSpectrum sig_a;
        if (sigma_a)
            sig_a = ClampZero(texEval(sigma_a, ctx, lambda));
        else if (color) {
            SampledSpectrum c = Clamp(texEval(color, ctx, lambda), 0, 1);
            sig_a = HairBxDF::SigmaAFromReflectance(c, bn, lambda);
        } else {
            CHECK(eumelanin || pheomelanin);
            sig_a = HairBxDF::SigmaAFromConcentration(
                        std::max(Float(0), eumelanin ? texEval(eumelanin, ctx) : 0),
                        std::max(Float(0), pheomelanin ? texEval(pheomelanin, ctx) : 0))
                        .Sample(lambda);
        }

        // Offset along width
        Float h = -1 + 2 * ctx.uv[1];
        *bxdf = HairBxDF(h, e, sig_a, bm, bn, a);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf, e);
    }

    static HairMaterial *Create(const TextureParameterDictionary &parameters,
                                const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return nullptr; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

  private:
    // HairMaterial Private Data
    SpectrumTextureHandle sigma_a, color;
    FloatTextureHandle eumelanin, pheomelanin, eta;
    FloatTextureHandle beta_m, beta_n, alpha;
};

// DiffuseMaterial Definition
class DiffuseMaterial {
  public:
    using BxDF = DiffuseBxDF;
    using BSSRDF = void;
    // DiffuseMaterial Public Methods
    static const char *Name() { return "DiffuseMaterial"; }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static DiffuseMaterial *Create(const TextureParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

    DiffuseMaterial(SpectrumTextureHandle reflectance, FloatTextureHandle sigma,
                    FloatTextureHandle displacement)
        : displacement(displacement), reflectance(reflectance), sigma(sigma) {}

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({sigma}, {reflectance});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda, DiffuseBxDF *bxdf) const {
        // Evaluate textures for _DiffuseMaterial_ and allocate BSDF
        SampledSpectrum r = Clamp(texEval(reflectance, ctx, lambda), 0, 1);
        Float sig = Clamp(texEval(sigma, ctx), 0, 90);
        *bxdf = DiffuseBxDF(r, SampledSpectrum(0), sig);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf);
    }

  private:
    // DiffuseMaterial Private Members
    FloatTextureHandle displacement;
    SpectrumTextureHandle reflectance;
    FloatTextureHandle sigma;
};

// ConductorMaterial Definition
class ConductorMaterial {
  public:
    using BxDF = ConductorBxDF;
    using BSSRDF = void;

    // ConductorMaterial Public Methods
    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({uRoughness, vRoughness}, {eta, k});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda, ConductorBxDF *bxdf) const {
        // Return BSDF for _ConductorMaterial_
        Float uRough = texEval(uRoughness, ctx), vRough = texEval(vRoughness, ctx);
        if (remapRoughness) {
            uRough = TrowbridgeReitzDistribution::RoughnessToAlpha(uRough);
            vRough = TrowbridgeReitzDistribution::RoughnessToAlpha(vRough);
        }
        SampledSpectrum etas = texEval(eta, ctx, lambda);
        SampledSpectrum ks = texEval(k, ctx, lambda);

        TrowbridgeReitzDistribution distrib(uRough, vRough);
        *bxdf = ConductorBxDF(distrib, etas, ks);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf);
    }

    ConductorMaterial(SpectrumTextureHandle eta, SpectrumTextureHandle k,
                      FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                      FloatTextureHandle displacement, bool remapRoughness)
        : displacement(displacement),
          eta(eta),
          k(k),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          remapRoughness(remapRoughness) {}

    static const char *Name() { return "ConductorMaterial"; }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static ConductorMaterial *Create(const TextureParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

  private:
    // ConductorMaterial Private Data
    FloatTextureHandle displacement;
    SpectrumTextureHandle eta, k;
    FloatTextureHandle uRoughness, vRoughness;
    bool remapRoughness;
};

// CoatedDiffuseMaterial Definition
class CoatedDiffuseMaterial {
  public:
    using BxDF = CoatedDiffuseBxDF;
    using BSSRDF = void;
    // CoatedDiffuseMaterial Public Methods
    CoatedDiffuseMaterial(SpectrumTextureHandle reflectance,
                          FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                          FloatTextureHandle thickness, SpectrumTextureHandle albedo,
                          FloatTextureHandle g, FloatTextureHandle eta,
                          FloatTextureHandle displacement, bool remapRoughness,
                          LayeredBxDFConfig config)
        : displacement(displacement),
          reflectance(reflectance),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          thickness(thickness),
          albedo(albedo),
          g(g),
          eta(eta),
          remapRoughness(remapRoughness),
          config(config) {}

    static const char *Name() { return "CoatedDiffuseMaterial"; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({uRoughness, vRoughness, thickness, g, eta},
                                   {reflectance, albedo});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, const MaterialEvalContext &ctx,
                              SampledWavelengths &lambda, CoatedDiffuseBxDF *bxdf) const {
        // Initialize diffuse component of plastic material
        SampledSpectrum r = Clamp(texEval(reflectance, ctx, lambda), 0, 1);

        // Create microfacet distribution _distrib_ for coated diffuse material
        Float urough = texEval(uRoughness, ctx);
        Float vrough = texEval(vRoughness, ctx);
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        TrowbridgeReitzDistribution distrib(urough, vrough);

        Float thick = texEval(thickness, ctx);
        Float e = texEval(eta, ctx);
        SampledSpectrum a = Clamp(texEval(albedo, ctx, lambda), 0, 1);
        Float gg = Clamp(texEval(g, ctx), -1, 1);

        *bxdf = CoatedDiffuseBxDF(DielectricInterfaceBxDF(e, distrib),
                                  IdealDiffuseBxDF(r), thick, a, gg, config);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static CoatedDiffuseMaterial *Create(const TextureParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, const MaterialEvalContext &ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

  private:
    // CoatedDiffuseMaterial Private Members
    FloatTextureHandle displacement;
    SpectrumTextureHandle reflectance, albedo;
    FloatTextureHandle uRoughness, vRoughness, thickness, g, eta;
    bool remapRoughness;
    LayeredBxDFConfig config;
};

// CoatedConductorMaterial Definition
class CoatedConductorMaterial {
  public:
    using BxDF = CoatedConductorBxDF;
    using BSSRDF = void;
    // CoatedConductorMaterial Public Methods
    CoatedConductorMaterial(FloatTextureHandle interfaceURoughness,
                            FloatTextureHandle interfaceVRoughness,
                            FloatTextureHandle thickness, FloatTextureHandle interfaceEta,
                            FloatTextureHandle g, SpectrumTextureHandle albedo,
                            FloatTextureHandle conductorURoughness,
                            FloatTextureHandle conductorVRoughness,
                            SpectrumTextureHandle conductorEta, SpectrumTextureHandle k,
                            FloatTextureHandle displacement, bool remapRoughness,
                            LayeredBxDFConfig config)
        : displacement(displacement),
          interfaceURoughness(interfaceURoughness),
          interfaceVRoughness(interfaceVRoughness),
          thickness(thickness),
          interfaceEta(interfaceEta),
          albedo(albedo),
          g(g),
          conductorURoughness(conductorURoughness),
          conductorVRoughness(conductorVRoughness),
          conductorEta(conductorEta),
          k(k),
          remapRoughness(remapRoughness),
          config(config) {}

    static const char *Name() { return "CoatedConductorMaterial"; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate(
            {interfaceURoughness, interfaceVRoughness, thickness, g, interfaceEta,
             conductorURoughness, conductorVRoughness},
            {conductorEta, k, albedo});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, const MaterialEvalContext &ctx,
                              SampledWavelengths &lambda,
                              CoatedConductorBxDF *bxdf) const {
        Float iurough = texEval(interfaceURoughness, ctx);
        Float ivrough = texEval(interfaceVRoughness, ctx);
        if (remapRoughness) {
            iurough = TrowbridgeReitzDistribution::RoughnessToAlpha(iurough);
            ivrough = TrowbridgeReitzDistribution::RoughnessToAlpha(ivrough);
        }
        TrowbridgeReitzDistribution interfaceDistrib(iurough, ivrough);

        Float thick = texEval(thickness, ctx);
        Float ieta = texEval(interfaceEta, ctx);

        SampledSpectrum ce = texEval(conductorEta, ctx, lambda);
        SampledSpectrum ck = texEval(k, ctx, lambda);
        Float curough = texEval(conductorURoughness, ctx);
        Float cvrough = texEval(conductorVRoughness, ctx);
        if (remapRoughness) {
            curough = TrowbridgeReitzDistribution::RoughnessToAlpha(curough);
            cvrough = TrowbridgeReitzDistribution::RoughnessToAlpha(cvrough);
        }
        TrowbridgeReitzDistribution conductorDistrib(curough, cvrough);

        SampledSpectrum a = Clamp(texEval(albedo, ctx, lambda), 0, 1);
        Float gg = Clamp(texEval(g, ctx), -1, 1);

        *bxdf = CoatedConductorBxDF(DielectricInterfaceBxDF(ieta, interfaceDistrib),
                                    ConductorBxDF(conductorDistrib, ce, ck), thick, a, gg,
                                    config);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static CoatedConductorMaterial *Create(const TextureParameterDictionary &parameters,
                                           const FileLoc *loc, Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, const MaterialEvalContext &ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

  private:
    // CoatedConductorMaterial Private Members
    FloatTextureHandle displacement;
    FloatTextureHandle interfaceURoughness, interfaceVRoughness, thickness, interfaceEta;
    FloatTextureHandle g;
    SpectrumTextureHandle albedo;
    FloatTextureHandle conductorURoughness, conductorVRoughness;
    SpectrumTextureHandle conductorEta, k;
    bool remapRoughness;
    LayeredBxDFConfig config;
};

// SubsurfaceMaterial Definition
class SubsurfaceMaterial {
  public:
    using BxDF = DielectricInterfaceBxDF;
    using BSSRDF = TabulatedBSSRDF;
    // SubsurfaceMaterial Public Methods
    SubsurfaceMaterial(Float scale, SpectrumTextureHandle sigma_a,
                       SpectrumTextureHandle sigma_s, SpectrumTextureHandle reflectance,
                       SpectrumTextureHandle mfp, Float g, Float eta,
                       FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                       FloatTextureHandle displacement, bool remapRoughness,
                       Allocator alloc)
        : displacement(displacement),
          scale(scale),
          sigma_a(sigma_a),
          sigma_s(sigma_s),
          reflectance(reflectance),
          mfp(mfp),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          eta(eta),
          remapRoughness(remapRoughness),
          table(100, 64, alloc) {
        ComputeBeamDiffusionBSSRDF(g, eta, &table);
    }

    static const char *Name() { return "SubsurfaceMaterial"; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({uRoughness, vRoughness}, {sigma_a, sigma_s});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, const MaterialEvalContext &ctx,
                              SampledWavelengths &lambda,
                              DielectricInterfaceBxDF *bxdf) const {
        // Initialize BSDF for _SubsurfaceMaterial_

        Float urough = texEval(uRoughness, ctx), vrough = texEval(vRoughness, ctx);
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        TrowbridgeReitzDistribution distrib(urough, vrough);

        // Initialize _bsdf_ for smooth or rough dielectric
        *bxdf = DielectricInterfaceBxDF(eta, distrib);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf, eta);
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, const MaterialEvalContext &ctx,
                                SampledWavelengths &lambda,
                                TabulatedBSSRDF *bssrdf) const {
        SampledSpectrum sig_a, sig_s;
        if (sigma_a && sigma_s) {
            // Evaluate textures for $\sigma_\roman{a}$ and $\sigma_\roman{s}$
            sig_a = ClampZero(scale * texEval(sigma_a, ctx, lambda));
            sig_s = ClampZero(scale * texEval(sigma_s, ctx, lambda));

        } else {
            // Compute _sig_a_ and _sig_s_ from reflectance and mfp
            DCHECK(reflectance && mfp);
            SampledSpectrum mfree = ClampZero(scale * texEval(mfp, ctx, lambda));
            SampledSpectrum r = Clamp(texEval(reflectance, ctx, lambda), 0, 1);
            SubsurfaceFromDiffuse(table, r, mfree, &sig_a, &sig_s);
        }
        *bssrdf = TabulatedBSSRDF(ctx.p, ctx.dpdus, ctx.ns, ctx.wo, 0 /* FIXME: si.time*/,
                                  eta, sig_a, sig_s, &table);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }
    PBRT_CPU_GPU bool IsTransparent() const { return false; }

    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return true; }

    static SubsurfaceMaterial *Create(const TextureParameterDictionary &parameters,
                                      const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // SubsurfaceMaterial Private Members
    FloatTextureHandle displacement;
    Float scale;
    SpectrumTextureHandle sigma_a, sigma_s, reflectance, mfp;
    FloatTextureHandle uRoughness, vRoughness;
    Float eta;
    bool remapRoughness;
    BSSRDFTable table;
};

// DiffuseTransmissionMaterial Definition
class DiffuseTransmissionMaterial {
  public:
    using BxDF = DiffuseBxDF;
    using BSSRDF = void;
    // DiffuseTransmissionMaterial Public Methods
    DiffuseTransmissionMaterial(SpectrumTextureHandle reflectance,
                                SpectrumTextureHandle transmittance,
                                FloatTextureHandle sigma, FloatTextureHandle displacement,
                                Float scale)
        : displacement(displacement),
          reflectance(reflectance),
          transmittance(transmittance),
          sigma(sigma),
          scale(scale) {}

    static const char *Name() { return "DiffuseTransmissionMaterial"; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return texEval.CanEvaluate({sigma}, {reflectance, transmittance});
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda, DiffuseBxDF *bxdf) const {
        SampledSpectrum r = Clamp(scale * texEval(reflectance, ctx, lambda), 0, 1);
        SampledSpectrum t = Clamp(scale * texEval(transmittance, ctx, lambda), 0, 1);
        Float s = texEval(sigma, ctx);
        *bxdf = DiffuseBxDF(r, t, s);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static DiffuseTransmissionMaterial *Create(
        const TextureParameterDictionary &parameters, const FileLoc *loc,
        Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

  private:
    // DiffuseTransmissionMaterial Private Data
    FloatTextureHandle displacement;
    SpectrumTextureHandle reflectance, transmittance;
    FloatTextureHandle sigma;
    Float scale;
};

// MeasuredMaterial Definition
class MeasuredMaterial {
  public:
    using BxDF = MeasuredBxDF;
    using BSSRDF = void;
    // MeasuredMaterial Public Methods
    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                              SampledWavelengths &lambda, MeasuredBxDF *bxdf) const {
        *bxdf = MeasuredBxDF(brdf, lambda);
        return BSDF(ctx.wo, ctx.n, ctx.ns, ctx.dpdus, bxdf);
    }

    MeasuredMaterial(const std::string &filename, FloatTextureHandle displacement,
                     Allocator alloc);

    static const char *Name() { return "MeasuredMaterial"; }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU bool CanEvaluateTextures(TextureEvaluator texEval) const {
        return true;
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static MeasuredMaterial *Create(const TextureParameterDictionary &parameters,
                                    const FileLoc *loc, Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU void GetBSSRDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                SampledWavelengths &lambda, void *) const {}

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU static constexpr bool HasSubsurfaceScattering() { return false; }

    std::string ToString() const;

  private:
    // MeasuredMaterial Private Members
    FloatTextureHandle displacement;
    const MeasuredBRDF *brdf;
};

template <typename TextureEvaluator>
inline bool MaterialHandle::CanEvaluateTextures(TextureEvaluator texEval) const {
    auto eval = [&](auto ptr) { return ptr->CanEvaluateTextures(texEval); };
    return Dispatch(eval);
}

template <typename TextureEvaluator>
inline BSDF MaterialHandle::GetBSDF(TextureEvaluator texEval, MaterialEvalContext ctx,
                                    SampledWavelengths &lambda,
                                    ScratchBuffer &scratchBuffer) const {
    auto get = [&](auto ptr) -> BSDF {
        using Material = typename std::remove_reference<decltype(*ptr)>::type;
        if constexpr (std::is_same_v<Material, MixMaterial>)
            return {};
        else {
            using BxDF = typename Material::BxDF;
            BxDF *bxdf = (BxDF *)scratchBuffer.Alloc(sizeof(BxDF), alignof(BxDF));
            return ptr->GetBSDF(texEval, ctx, lambda, bxdf);
        }
    };
    return Dispatch(get);
}

template <typename TextureEvaluator>
inline BSSRDFHandle MaterialHandle::GetBSSRDF(TextureEvaluator texEval,
                                              MaterialEvalContext ctx,
                                              SampledWavelengths &lambda,
                                              ScratchBuffer &scratchBuffer) const {
    auto get = [&](auto ptr) -> BSSRDFHandle {
        using Material = typename std::remove_reference<decltype(*ptr)>::type;
        using BSSRDF = typename Material::BSSRDF;
        if constexpr (std::is_same_v<BSSRDF, void>)
            return nullptr;
        else {
            BSSRDF *bssrdf =
                (BSSRDF *)scratchBuffer.Alloc(sizeof(BSSRDF), alignof(BSSRDF));
            ptr->GetBSSRDF(texEval, ctx, lambda, bssrdf);
            return bssrdf;
        }
    };
    return Dispatch(get);
}

inline bool MaterialHandle::IsTransparent() const {
    auto transp = [&](auto ptr) { return ptr->IsTransparent(); };
    return Dispatch(transp);
}

inline bool MaterialHandle::HasSubsurfaceScattering() const {
    auto has = [&](auto ptr) { return ptr->HasSubsurfaceScattering(); };
    return Dispatch(has);
}

inline FloatTextureHandle MaterialHandle::GetDisplacement() const {
    auto disp = [&](auto ptr) { return ptr->GetDisplacement(); };
    return Dispatch(disp);
}

}  // namespace pbrt

#endif  // PBRT_MATERIALS_H
