// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_OPTIX_AGGREGATE_H
#define PBRT_GPU_OPTIX_AGGREGATE_H

#include <pbrt/pbrt.h>

#include <pbrt/gpu/memory.h>
#include <pbrt/gpu/optix/optix.h>
#include <pbrt/scene.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/soa.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/integrator.h>
#include <pbrt/wavefront/workitems.h>

#include <map>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace pbrt {

class OptiXAggregate : public WavefrontAggregate {
  public:
    OptiXAggregate(const BasicScene &scene, CUDATrackedMemoryResource *memoryResource,
                   NamedTextures &textures,
                   const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
                   const std::map<std::string, Medium> &media,
                   const std::map<std::string, pbrt::Material> &namedMaterials,
                   const std::vector<pbrt::Material> &materials);

    Bounds3f Bounds() const { return bounds; }

    void IntersectClosest(int maxRays, const RayQueue *rayQueue,
                          EscapedRayQueue *escapedRayQueue,
                          HitAreaLightQueue *hitAreaLightQueue,
                          MaterialEvalQueue *basicEvalMaterialQueue,
                          MaterialEvalQueue *universalEvalMaterialQueue,
                          MediumSampleQueue *mediumSampleQueue,
                          RayQueue *nextRayQueue) const;

    void IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                         SOA<PixelSampleState> *pixelSampleState) const;

    void IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                           SOA<PixelSampleState> *pixelSampleState) const;

    void IntersectOneRandom(int maxRays,
                            SubsurfaceScatterQueue *subsurfaceScatterQueue) const;

    // WAR: The enclosing parent function ("PreparePLYMeshes") for an
    // extended __device__ lambda cannot have private or protected access
    // within its class, so it's public...
    static std::map<int, TriQuadMesh> PreparePLYMeshes(
        const std::vector<ShapeSceneEntity> &shapes,
        const std::map<std::string, FloatTexture> &floatTextures);

  private:
    struct HitgroupRecord;

    struct BVH {
        BVH() = default;
        BVH(size_t size);

        OptixTraversableHandle traversableHandle = {};
        std::vector<HitgroupRecord> intersectHGRecords;
        std::vector<HitgroupRecord> shadowHGRecords;
        std::vector<HitgroupRecord> randomHitHGRecords;
        Bounds3f bounds;
    };

    static BVH buildBVHForTriangles(
        const std::vector<ShapeSceneEntity> &shapes,
        const std::map<int, TriQuadMesh> &plyMeshes, OptixDeviceContext optixContext,
        const OptixProgramGroup &intersectPG, const OptixProgramGroup &shadowPG,
        const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        ThreadLocal<Allocator> &threadAllocators,
        ThreadLocal<cudaStream_t> &threadCUDAStreams);

    static BilinearPatchMesh *diceCurveToBLP(const ShapeSceneEntity &shape, int nDiceU,
                                             int nDiceV, Allocator alloc);

    static BVH buildBVHForBLPs(
        const std::vector<ShapeSceneEntity> &shapes, OptixDeviceContext optixContext,
        const OptixProgramGroup &intersectPG, const OptixProgramGroup &shadowPG,
        const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        ThreadLocal<Allocator> &threadAllocators,
        ThreadLocal<cudaStream_t> &threadCUDAStreams);

    static BVH buildBVHForQuadrics(
        const std::vector<ShapeSceneEntity> &shapes, OptixDeviceContext optixContext,
        const OptixProgramGroup &intersectPG, const OptixProgramGroup &shadowPG,
        const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        ThreadLocal<Allocator> &threadAllocators,
        ThreadLocal<cudaStream_t> &threadCUDAStreams);

    int addHGRecords(const BVH &bvh);

    static OptixModule createOptiXModule(OptixDeviceContext optixContext,
                                         const char *ptx);
    static OptixPipelineCompileOptions getPipelineCompileOptions();

    OptixProgramGroup createRaygenPG(const char *entrypoint) const;
    OptixProgramGroup createMissPG(const char *entrypoint) const;
    OptixProgramGroup createIntersectionPG(const char *closest, const char *any,
                                           const char *intersect) const;

    static OptixTraversableHandle buildOptixBVH(
        OptixDeviceContext optixContext, const std::vector<OptixBuildInput> &buildInputs,
        ThreadLocal<cudaStream_t> &threadCUDAStreams);

    CUDATrackedMemoryResource *memoryResource;
    std::mutex boundsMutex;
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

}  // namespace pbrt

#endif  // PBRT_GPU_AGGREGATE_H
