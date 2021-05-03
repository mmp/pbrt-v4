// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_INTEGRATOR_H
#define PBRT_WAVEFRONT_INTEGRATOR_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/base/sampler.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/options.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/wavefront/workitems.h>
#include <pbrt/wavefront/workqueue.h>

namespace pbrt {

class ParsedScene;

// WavefrontAggregate Definition
class WavefrontAggregate {
  public:
    virtual ~WavefrontAggregate() = default;

    virtual Bounds3f Bounds() const = 0;

    virtual void IntersectClosest(int maxRays, EscapedRayQueue *escapedRayQueue,
                                  HitAreaLightQueue *hitAreaLightQueue,
                                  MaterialEvalQueue *basicEvalMaterialQueue,
                                  MaterialEvalQueue *universalEvalMaterialQueue,
                                  MediumSampleQueue *mediumSampleQueue,
                                  RayQueue *rayQueue, RayQueue *nextRayQueue) const = 0;

    virtual void IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                                 SOA<PixelSampleState> *pixelSampleState) const = 0;

    virtual void IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                                   SOA<PixelSampleState> *pixelSampleState) const = 0;

    virtual void IntersectOneRandom(
        int maxRays, SubsurfaceScatterQueue *subsurfaceScatterQueue) const = 0;
};

// WavefrontPathIntegrator Definition
class WavefrontPathIntegrator {
  public:
    // WavefrontPathIntegrator Public Methods
    Float Render();

    void GenerateCameraRays(int y0, int sampleIndex);
    template <typename Sampler>
    void GenerateCameraRays(int y0, int sampleIndex);

    void GenerateRaySamples(int depth, int sampleIndex);
    template <typename Sampler>
    void GenerateRaySamples(int depth, int sampleIndex);

    void TraceShadowRays(int depth);
    void SampleMediumInteraction(int depth);
    void SampleSubsurface(int depth);

    void HandleEscapedRays(int depth);
    void HandleRayFoundEmission(int depth);

    void EvaluateMaterialsAndBSDFs(int depth);
    template <typename Mtl>
    void EvaluateMaterialAndBSDF(int depth);
    template <typename Mtl, typename TextureEvaluator>
    void EvaluateMaterialAndBSDF(TextureEvaluator texEval, MaterialEvalQueue *evalQueue,
                                 int depth);

    void SampleDirect(int depth);
    template <typename BxDF>
    void SampleDirect(int depth);

    void SampleIndirect(int depth);
    template <typename BxDF>
    void SampleIndirect(int depth);

    void UpdateFilm();

    template <typename F>
    void ParallelFor(const char *description, int nItems, F &&func) {
        if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
            GPUParallelFor(description, nItems, func);
#else
            LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
        } else
            pbrt::ParallelFor(0, nItems, func);
    }

    template <typename F>
    void Do(const char *description, F &&func) {
        if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
            GPUParallelFor(description, 1, [=] PBRT_GPU(int) mutable { func(); });
#else
            LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
        } else
            func();
    }

    WavefrontPathIntegrator(Allocator alloc, ParsedScene &scene);

    RayQueue *CurrentRayQueue(int depth) { return rayQueues[depth & 1]; }
    RayQueue *NextRayQueue(int depth) { return rayQueues[(depth + 1) & 1]; }

    void IntersectClosest(RayQueue *rayQueue, EscapedRayQueue *escapedRayQueue,
                          HitAreaLightQueue *hitAreaLightQueue,
                          MaterialEvalQueue *basicEvalMaterialQueue,
                          MaterialEvalQueue *universalEvalMaterialQueue,
                          MediumSampleQueue *mediumSampleQueue,
                          RayQueue *nextRayQueue) const;

    // WavefrontPathIntegrator Member Variables
    bool initializeVisibleSurface;
    bool haveSubsurface;
    bool haveMedia;
    pstd::array<bool, Material::NumTags()> haveBasicEvalMaterial;
    pstd::array<bool, Material::NumTags()> haveUniversalEvalMaterial;

    struct Stats {
        Stats(int maxDepth, Allocator alloc);

        std::string Print() const;

        // Note: not atomics: tid 0 always updates them for everyone...
        uint64_t cameraRays = 0;
        pstd::vector<uint64_t> indirectRays, shadowRays;
    };
    Stats *stats;

    WavefrontAggregate *aggregate = nullptr;

    Filter filter;
    Film film;
    Sampler sampler;
    Camera camera;
    pstd::vector<Light> *envLights;
    LightSampler lightSampler;

    int maxDepth;
    bool regularize;

    int scanlinesPerPass, maxQueueSize;

    SOA<PixelSampleState> pixelSampleState;

    RayQueue *rayQueues[2];

    MediumSampleQueue *mediumSampleQueue = nullptr;
    MediumScatterQueue *mediumScatterQueue = nullptr;

    EscapedRayQueue *escapedRayQueue = nullptr;

    HitAreaLightQueue *hitAreaLightQueue = nullptr;

    MaterialEvalQueue *basicEvalMaterialQueue = nullptr;
    MaterialEvalQueue *universalEvalMaterialQueue = nullptr;

    ShadowRayQueue *shadowRayQueue = nullptr;

    GetBSSRDFAndProbeRayQueue *bssrdfEvalQueue = nullptr;
    SubsurfaceScatterQueue *subsurfaceScatterQueue = nullptr;
};

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_INTEGRATOR_H
