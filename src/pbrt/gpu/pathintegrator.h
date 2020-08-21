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

class GPUPathIntegrator {
  public:
    GPUPathIntegrator(Allocator alloc, const ParsedScene &scene);

    void Render(ImageMetadata *metadata);

    void GenerateCameraRays(int y0, int sampleIndex);
    template <typename Sampler>
    void GenerateCameraRays(int y0, int sampleIndex);

    void GenerateRaySamples(int depth, int sampleIndex);
    template <typename Sampler>
    void GenerateRaySamples(int depth, int sampleIndex);

    void TraceShadowRays(int depth);
    void SampleMediumInteraction(int depth);
    void HandleMediumTransitions(int depth);
    void SampleSubsurface(int depth);

    void HandleEscapedRays(int depth);
    void HandleRayFoundEmission(int depth);

    void EvaluateMaterialsAndBSDFs(int depth);
    template <typename Material>
    void EvaluateMaterialAndBSDF(int depth);
    template <typename Material, typename TextureEvaluator>
    void EvaluateMaterialAndBSDF(TextureEvaluator texEval, MaterialEvalQueue *evalQueue,
                                 int depth);
    template <typename TextureEvaluator>
    void ResolveMixMaterial(TextureEvaluator texEval,
                            MaterialEvalQueue *evalQueue);

    void SampleDirect(int depth);
    template <typename BxDF>
    void SampleDirect(int depth);

    void SampleIndirect(int depth);
    template <typename BxDF>
    void SampleIndirect(int depth);

    void UpdateFilm();

    FilterHandle filter;
    FilmHandle film;
    SamplerHandle sampler;
    CameraHandle camera;
    LightHandle envLight;
    LightSamplerHandle lightSampler;

    int maxDepth;
    bool regularize;
    int maxQueueSize, scanlinesPerPass;

    // Various properties of the scene
    bool initializeVisibleSurface;
    bool haveSubsurface;
    bool haveMedia;
    pstd::array<bool, MaterialHandle::NumTags()> haveBasicEvalMaterial;
    pstd::array<bool, MaterialHandle::NumTags()> haveUniversalEvalMaterial;

    GPUAccel *accel = nullptr;

    SOA<PixelSampleState> pixelSampleState;

    RayQueue *rayQueues[2] = {nullptr, nullptr};

    ShadowRayQueue *shadowRayQueue = nullptr;

    EscapedRayQueue *escapedRayQueue = nullptr;
    HitAreaLightQueue *hitAreaLightQueue = nullptr;

    MaterialEvalQueue *basicEvalMaterialQueue = nullptr;
    MaterialEvalQueue *universalEvalMaterialQueue = nullptr;

    GetBSSRDFAndProbeRayQueue *bssrdfEvalQueue = nullptr;
    SubsurfaceScatterQueue *subsurfaceScatterQueue = nullptr;

    MediumTransitionQueue *mediumTransitionQueue = nullptr;
    MediumSampleQueue *mediumSampleQueue = nullptr;
    MediumScatterQueue *mediumScatterQueue = nullptr;

    struct Stats {
        Stats(int maxDepth, Allocator alloc);

        std::string Print() const;

        // Note: not atomics: tid 0 always updates them for everyone...
        uint64_t cameraRays = 0;
        pstd::vector<uint64_t> indirectRays, shadowRays;
    };
    Stats *stats;
};

}  // namespace pbrt

#endif  // PBRT_GPU_PATHINTEGRATOR_H
