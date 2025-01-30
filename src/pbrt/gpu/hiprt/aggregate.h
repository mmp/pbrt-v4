#ifndef PBRT_HIPRT_AGGREGATE_H
#define PBRT_HIPRT_AGGREGATE_H

#include <pbrt/pbrt.h>

#include <pbrt/gpu/common.h>
#include <pbrt/gpu/memory.h>
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

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

namespace pbrt {

class HiprtAggregate : public WavefrontAggregate {
  public:
    HiprtAggregate(const BasicScene &scene, CUDATrackedMemoryResource *memoryResource,
                   NamedTextures &textures,
                   const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
                   const std::map<std::string, Medium> &media,
                   const std::map<std::string, pbrt::Material> &namedMaterials,
                   const std::vector<pbrt::Material> &materials, int maxQueueSize);

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
    static constexpr size_t BlockSize = 64u;

    struct ParamBufferState {
        bool used = false;
        hipEvent_t finishedEvent;
        hipDeviceptr_t ptr = 0;
        void *hostPtr = nullptr;
    };

    mutable std::vector<ParamBufferState> paramsPool;
    mutable size_t nextParamOffset = 0;

    struct Module {
        hipModule_t hipModule;
        hipFunction_t closestFunction;
        hipFunction_t shadowFunction;
        hipFunction_t shadowTrFunction;
        hipFunction_t oneRandomFunction;
        hiprtFuncTable funcTable;
    };

    struct GeometryContainer {
        GeometryContainer() = default;
        GeometryContainer(size_t size) : hgRecords(size) {}

        void resize(size_t size) {
            hgRecords.resize(size);
        }

        uint32_t offset;
        hiprtScene scene;
        std::vector<HitgroupRecord> hgRecords;
    };

    struct GeometryGroup {
        GeometryContainer triGeomContainer;
        GeometryContainer blpGeomContainer;
        GeometryContainer quadricGeomContainer;
    };

    struct InstanceContainer {
        void appendRecords(GeometryContainer &g) {
            g.offset = hgRecords.size();
            hgRecords.insert(hgRecords.end(), g.hgRecords.begin(), g.hgRecords.end());
        }

        void insertInstance(GeometryContainer &g, const hiprtFrameMatrix& m) { 
            if (g.scene != nullptr) {
                hiprtInstance instance;
                instance.type = hiprtInstanceTypeScene;
                instance.scene = g.scene;
                instances.push_back(instance);
                transforms.push_back(m);
                offsets.push_back(g.offset);
            }
        }

        std::vector<uint32_t> offsets;
        std::vector<hiprtFrameMatrix> transforms;
        std::vector<hiprtInstance> instances;
        std::vector<HitgroupRecord> hgRecords;
    };

    ParamBufferState &getParamBuffer(const RayIntersectParameters &) const;

    static BilinearPatchMesh *diceCurveToBLP(const ShapeSceneEntity &shape, int nDiceU,
                                             int nDiceV, Allocator alloc);

    static GeometryContainer buildBVHForTriangles(
        const std::vector<ShapeSceneEntity> &shapes,
        const std::map<int, TriQuadMesh> &plyMeshes, hiprtContext context,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        ThreadLocal<Allocator> &threadAllocators,
        ThreadLocal<hipStream_t> &threadCUDAStreams);

    static GeometryContainer buildBVHForBLPs(
        const std::vector<ShapeSceneEntity> &shapes, hiprtContext context,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        ThreadLocal<Allocator> &threadAllocators,
        ThreadLocal<hipStream_t> &threadCUDAStreams);

    static GeometryContainer buildBVHForQuadrics(
        const std::vector<ShapeSceneEntity> &shapes, hiprtContext context,
        const std::map<std::string, FloatTexture> &floatTextures,
        const std::map<std::string, Material> &namedMaterials,
        const std::vector<Material> &materials,
        const std::map<std::string, Medium> &media,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        ThreadLocal<Allocator> &threadAllocators,
        ThreadLocal<hipStream_t> &threadCUDAStreams);

    static hiprtScene buildBVHForInstances(
        const std::vector<hiprtInstance> &instances,
        const std::vector<hiprtFrameMatrix> &transforms, hiprtContext context,
        ThreadLocal<Allocator> &threadAllocators,
        ThreadLocal<hipStream_t> &threadCUDAStreams);

    static Module compileHiprtModule(hiprtContext context);

    static void hiprtLaunch(hipFunction_t func, int nx, int ny, void **args,
                            hipStream_t cudaStream, size_t sharedMemoryBytes = 0);

    static void hiprtLaunch(hipFunction_t func, int nx, int ny, int tx, int ty,
                            void **args, hipStream_t cudaStream,
                            size_t sharedMemoryBytes = 0);
    Module module;
    hiprtContext context;

    hiprtScene scene;
    hiprtGlobalStackBuffer globalStackBuffer;
    hiprtGlobalStackBuffer globalInstanceStackBuffer;
    std::vector<GeometryGroup> geomGroups;
    HitgroupRecord *hgRecords;
    uint32_t *offsets;
    
    Bounds3f bounds;
    CUDATrackedMemoryResource *memoryResource;
    hipStream_t cudaStream;
};

}  // namespace pbrt

#endif  // PBRT_HIPRT_AGGREGATE_H
