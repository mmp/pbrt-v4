// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_PATHINTEGRATOR_H
#define PBRT_GPU_PATHINTEGRATOR_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/workitems.h>
#include <pbrt/gpu/workqueue.h>
#include <pbrt/util/pstd.h>

namespace pbrt {

class ParsedScene;
class GPUAccel;

void GPUInit();
void GPURender(ParsedScene &scene);

// GPUPathIntegrator Definition
class GPUPathIntegrator {
  public:
    // GPUPathIntegrator Public Methods
    void Render();

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
    template <typename Material>
    void EvaluateMaterialAndBSDF(int depth);
    template <typename Material, typename TextureEvaluator>
    void EvaluateMaterialAndBSDF(TextureEvaluator texEval, MaterialEvalQueue *evalQueue,
                                 int depth);

    void SampleDirect(int depth);
    template <typename BxDF>
    void SampleDirect(int depth);

    void SampleIndirect(int depth);
    template <typename BxDF>
    void SampleIndirect(int depth);

    void UpdateFilm();

    GPUPathIntegrator(Allocator alloc, const ParsedScene &scene);

    RayQueue *CurrentRayQueue(int depth) { return rayQueues[depth & 1]; }
    RayQueue *NextRayQueue(int depth) { return rayQueues[(depth + 1) & 1]; }

    void IntersectClosest(RayQueue *rayQueue, EscapedRayQueue *escapedRayQueue,
                          HitAreaLightQueue *hitAreaLightQueue,
                          MaterialEvalQueue *basicEvalMaterialQueue,
                          MaterialEvalQueue *universalEvalMaterialQueue,
                          MediumSampleQueue *mediumSampleQueue,
                          RayQueue *nextRayQueue) const;

    // GPUPathIntegrator Member Variables
    bool initializeVisibleSurface;
    bool haveSubsurface;
    bool haveMedia;
    pstd::array<bool, MaterialHandle::NumTags()> haveBasicEvalMaterial;
    pstd::array<bool, MaterialHandle::NumTags()> haveUniversalEvalMaterial;

    GPUAccel *accel = nullptr;

    struct Stats {
        Stats(int maxDepth, Allocator alloc);

        std::string Print() const;

        // Note: not atomics: tid 0 always updates them for everyone...
        uint64_t cameraRays = 0;
        pstd::vector<uint64_t> indirectRays, shadowRays;
    };
    Stats *stats;

    FilterHandle filter;
    FilmHandle film;
    SamplerHandle sampler;
    CameraHandle camera;
    pstd::vector<LightHandle> *envLights;
    LightSamplerHandle lightSampler;

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

#endif  // PBRT_GPU_PATHINTEGRATOR_H
