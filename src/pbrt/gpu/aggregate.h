// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_AGGREGATE_H
#define PBRT_GPU_AGGREGATE_H

#include <pbrt/pbrt.h>

#include <pbrt/gpu/optix.h>
#include <pbrt/materials.h>
#include <pbrt/parsedscene.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/soa.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/integrator.h>
#include <pbrt/wavefront/workitems.h>

#include <map>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace pbrt {

class OptiXAggregate : public WavefrontAggregate {
  public:
    OptiXAggregate(const ParsedScene &scene, Allocator alloc, NamedTextures &textures,
             const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
             const std::map<std::string, Medium> &media,
             const std::map<std::string, pbrt::Material> &namedMaterials,
             const std::vector<pbrt::Material> &materials);

    Bounds3f Bounds() const { return bounds; }

    void IntersectClosest(
        int maxRays, const RayQueue *rayQueue, EscapedRayQueue *escapedRayQueue,
        HitAreaLightQueue *hitAreaLightQueue, MaterialEvalQueue *basicEvalMaterialQueue,
        MaterialEvalQueue *universalEvalMaterialQueue,
        MediumSampleQueue *mediumSampleQueue, RayQueue *nextRayQueue) const;

    void IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                         SOA<PixelSampleState> *pixelSampleState) const;

    void IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                           SOA<PixelSampleState> *pixelSampleState) const;

    void IntersectOneRandom(int maxRays, SubsurfaceScatterQueue *subsurfaceScatterQueue) const;

  private:
    struct HitgroupRecord;

    OptixTraversableHandle createGASForTriangles(
        const std::vector<ShapeSceneEntity> &shapes, const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG, const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle createGASForBLPs(
        const std::vector<ShapeSceneEntity> &shapes, const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG, const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle createGASForQuadrics(
        const std::vector<ShapeSceneEntity> &shapes, const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG, const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle buildBVH(const std::vector<OptixBuildInput> &buildInputs);

    Allocator alloc;
    Bounds3f bounds;
    CUstream cudaStream;
    OptixDeviceContext optixContext;
    OptixModule optixModule;
    OptixPipeline optixPipeline;

    struct ParamBufferState {
        bool used = false;
        cudaEvent_t finishedEvent;
        CUdeviceptr ptr = 0;
        void *hostPtr = nullptr;
    };
    mutable std::vector<ParamBufferState> paramsPool;
    mutable size_t nextParamOffset = 0;

    ParamBufferState &getParamBuffer(const RayIntersectParameters &) const;

    pstd::vector<HitgroupRecord> intersectHGRecords;
    pstd::vector<HitgroupRecord> shadowHGRecords;
    pstd::vector<HitgroupRecord> randomHitHGRecords;
    OptixShaderBindingTable intersectSBT = {}, shadowSBT = {}, shadowTrSBT = {};
    OptixShaderBindingTable randomHitSBT = {};
    OptixTraversableHandle rootTraversable = {};
};

} // namespace pbrt

#endif // PBRT_GPU_AGGREGATE_H
