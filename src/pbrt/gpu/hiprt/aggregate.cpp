// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/hiprt/aggregate.h>

#include <pbrt/gpu/common.h>
#include <pbrt/gpu/util.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/scene.h>
#include <pbrt/textures.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/log.h>
#include <pbrt/util/loopsubdiv.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/splines.h>
#include <pbrt/util/stats.h>
#include <pbrt/wavefront/intersect.h>

#include <atomic>
#include <fstream>
#include <mutex>
#include <unordered_map>

#include <hiprt/hiprt.h>

#define HIPRT_CHECK(EXPR)                                                    \
    do {                                                                     \
        hiprtError res = EXPR;                                               \
        if (res != hiprtSuccess)                                             \
            LOG_FATAL("HIPRT call " #EXPR " failed with code %d", int(res)); \
    } while (false) /* eat semicolon */

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Acceleration structures", gpuBVHBytes);
STAT_COUNTER("Geometry/Triangles added from displacement mapping", displacedTrisDelta);
STAT_COUNTER("Geometry/Curves", nCurves);
STAT_COUNTER("Geometry/Bilinear patches created for diced curves", nBLPsForCurves);

static Material getMaterial(const ShapeSceneEntity &shape,
                            const std::map<std::string, Material> &namedMaterials,
                            const std::vector<Material> &materials) {
    if (!shape.materialName.empty()) {
        auto iter = namedMaterials.find(shape.materialName);
        if (iter == namedMaterials.end())
            ErrorExit(&shape.loc, "%s: material not defined", shape.materialName);
        return iter->second;
    } else {
        CHECK_NE(shape.materialIndex, -1);
        return materials[shape.materialIndex];
    }
}

static FloatTexture getAlphaTexture(
    const ShapeSceneEntity &shape,
    const std::map<std::string, FloatTexture> &floatTextures, Allocator alloc) {
    FloatTexture alphaTexture;

    std::string alphaTexName = shape.parameters.GetTexture("alpha");
    if (alphaTexName.empty()) {
        if (Float alpha = shape.parameters.GetOneFloat("alpha", 1.f); alpha < 1.f)
            alphaTexture = alloc.new_object<FloatConstantTexture>(alpha);
        else
            return nullptr;
    } else {
        auto iter = floatTextures.find(alphaTexName);
        if (iter == floatTextures.end())
            ErrorExit(&shape.loc, "%s: alpha texture not defined.", alphaTexName);

        alphaTexture = iter->second;
    }

    if (!BasicTextureEvaluator().CanEvaluate({alphaTexture}, {})) {
        // It would be nice to just use the UniversalTextureEvaluator (maybe
        // always), but optix complains "Error: Found call graph recursion"...
        Warning(&shape.loc,
                "%s: alpha texture too complex for BasicTextureEvaluator "
                "(need fallback path). Ignoring for now.",
                alphaTexName);
        alphaTexture = nullptr;
    }

    return alphaTexture;
}

static MediumInterface *getMediumInterface(const ShapeSceneEntity &shape,
                                           const std::map<std::string, Medium> &media,
                                           Allocator alloc) {
    if (shape.insideMedium.empty() && shape.outsideMedium.empty())
        return nullptr;

    auto getMedium = [&](const std::string &name) -> Medium {
        if (name.empty())
            return nullptr;

        auto iter = media.find(name);
        if (iter == media.end())
            ErrorExit(&shape.loc, "%s: medium not defined", name);
        return iter->second;
    };

    return alloc.new_object<MediumInterface>(getMedium(shape.insideMedium),
                                             getMedium(shape.outsideMedium));
}

HiprtAggregate::HiprtAggregate(
    const BasicScene &scene, CUDATrackedMemoryResource *memoryResource,
    NamedTextures &textures,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    const std::map<std::string, Medium> &media,
    const std::map<std::string, pbrt::Material> &namedMaterials,
    const std::vector<pbrt::Material> &materials, int maxQueueSize)
    : memoryResource(memoryResource), cudaStream(nullptr), scene(nullptr) {
    hipCtx_t cudaContext;
    CUDA_CHECK(hipCtxGetCurrent(&cudaContext));
    CHECK(cudaContext != nullptr);

#if defined(PBRT_IS_WINDOWS)
    // On Windows, it is unfortunately necessary to disable
    // multithreading here.  The issue is that GPU managed memory can
    // only be accessed by one of the CPU or the GPU at a time; the
    // program crashes if this is restriction is violated.  Thus, it's
    // bad news if we are simultaneously, say, reading PLY files on
    // the CPU and storing them in managed memory while an OptiX
    // kernel is running on the GPU to build a BVH... (Issue #164).
    if (Options->useGPU)
        DisableThreadPool();
#endif  // PBRT_IS_WINDOWS

    ThreadLocal<hipStream_t> threadCUDAStreams([]() {
        hipStream_t stream;
        hipStreamCreate(&stream);
        return stream;
    });

    paramsPool.resize(256);  // should be plenty
    for (ParamBufferState &ps : paramsPool) {
        void *ptr;
        CUDA_CHECK(hipMalloc(&ptr, sizeof(RayIntersectParameters)));
        ps.ptr = (hipDeviceptr_t)ptr;
        CUDA_CHECK(hipEventCreate(&ps.finishedEvent));
        CUDA_CHECK(hipHostMalloc(&ps.hostPtr, sizeof(RayIntersectParameters)));
    }

    // Create HIPRT context
    LOG_VERBOSE("Starting HIPRT initialization");

    int current_device;
    CUDA_CHECK(hipGetDevice(&current_device));

    hiprtContextCreationInput ctxInput;
    ctxInput.ctxt = cudaContext;
    ctxInput.device = current_device;
    ctxInput.deviceType = hiprtDeviceAMD;
    HIPRT_CHECK(hiprtCreateContext(HIPRT_API_VERSION, ctxInput, context));

#ifdef NDEBUG
    hiprtSetLogLevel(hiprtLogLevelError);
#endif
    LOG_VERBOSE("HIPRT version %d.%d.%x successfully initialized", HIPRT_MAJOR_VERSION,
                HIPRT_MINOR_VERSION, HIPRT_PATCH_VERSION);

    // HIPRT module
    module = compileHiprtModule(context);

    LOG_VERBOSE("Finished HIPRT initialization");

    // Note: do not delete the pointers in threadBufferResources, since doing
    // so would cause the memory they manage to be freed.
    ThreadLocal<Allocator> threadAllocators([memoryResource]() {
        pstd::pmr::monotonic_buffer_resource *resource =
            new pstd::pmr::monotonic_buffer_resource(1024 * 1024, memoryResource);
        return Allocator(resource);
    });

    ///////////////////////////////////////////////////////////////////////////
    // Build top-level acceleration structures for non-instanced shapes
    LOG_VERBOSE("Starting to create shapes and acceleration structures");
    for (const auto &shape : scene.shapes)
        if (shape.name != "sphere" && shape.name != "cylinder" && shape.name != "disk" &&
            shape.name != "trianglemesh" && shape.name != "plymesh" &&
            shape.name != "loopsubdiv" && shape.name != "bilinearmesh" &&
            shape.name != "curve")
            ErrorExit(&shape.loc, "%s: unknown shape", shape.name);

    LOG_VERBOSE("Starting to read PLY meshes");
    std::map<int, TriQuadMesh> plyMeshes =
        PreparePLYMeshes(scene.shapes, textures.floatTextures);
    LOG_VERBOSE("Finished reading PLY meshes");

    LOG_VERBOSE("Starting to build geometries (BLAS)");
    geomGroups.resize(scene.instanceDefinitions.size() + 1);

    geomGroups[0].triGeomContainer = buildBVHForTriangles(
        scene.shapes, plyMeshes, context, textures.floatTextures, namedMaterials,
        materials, media, shapeIndexToAreaLights, threadAllocators, threadCUDAStreams);

    geomGroups[0].blpGeomContainer = buildBVHForBLPs(
        scene.shapes, context, textures.floatTextures, namedMaterials, materials, media,
        shapeIndexToAreaLights, threadAllocators, threadCUDAStreams);

    geomGroups[0].quadricGeomContainer = buildBVHForQuadrics(
        scene.shapes, context, textures.floatTextures, namedMaterials, materials, media,
        shapeIndexToAreaLights, threadAllocators, threadCUDAStreams);
    LOG_VERBOSE("Finished building geometries (BLAS)");

    ///////////////////////////////////////////////////////////////////////////
    // Create instanced geometries
    LOG_VERBOSE("Starting to build instanced geometries",
                scene.instanceDefinitions.size());

    InstanceContainer instanceContainer;
    instanceContainer.appendRecords(geomGroups[0].triGeomContainer);
    instanceContainer.appendRecords(geomGroups[0].blpGeomContainer);
    instanceContainer.appendRecords(geomGroups[0].quadricGeomContainer);

    std::vector<InternedString> allInstanceNames;
    for (const auto &def : scene.instanceDefinitions)
        allInstanceNames.push_back(def.first);

    std::unordered_map<InternedString, GeometryGroup, InternedStringHash> instanceMap;
    for (int i = 0; i < scene.instanceDefinitions.size(); ++i) {
        InternedString name = allInstanceNames[i];
        auto iter = scene.instanceDefinitions.find(name);
        CHECK(iter != scene.instanceDefinitions.end());
        const auto &def = *iter;

        if (!def.second->animatedShapes.empty())
            Warning("Ignoring %d animated shapes in instance \"%s\".",
                    def.second->animatedShapes.size(), def.first);

        std::map<int, TriQuadMesh> meshes =
            PreparePLYMeshes(def.second->shapes, textures.floatTextures);

        GeometryGroup &geomGroup = geomGroups[i + 1];

        geomGroup.triGeomContainer = buildBVHForTriangles(
            def.second->shapes, meshes, context, textures.floatTextures, namedMaterials,
            materials, media, {}, threadAllocators, threadCUDAStreams);

        geomGroup.blpGeomContainer = buildBVHForBLPs(
            def.second->shapes, context, textures.floatTextures, namedMaterials,
            materials, media, {}, threadAllocators, threadCUDAStreams);

        geomGroup.quadricGeomContainer = buildBVHForQuadrics(
            def.second->shapes, context, textures.floatTextures, namedMaterials,
            materials, media, {}, threadAllocators, threadCUDAStreams);

        instanceContainer.appendRecords(geomGroup.triGeomContainer);
        instanceContainer.appendRecords(geomGroup.blpGeomContainer);
        instanceContainer.appendRecords(geomGroup.quadricGeomContainer);

        meshes.clear();
        instanceMap[def.first] = geomGroup;
    }
    LOG_VERBOSE("Finished building instanced geometries");

    ///////////////////////////////////////////////////////////////////////////
    // Instancing
    LOG_VERBOSE("Starting to build scene (TLAS)");
    hiprtFrameMatrix identity{};
    for (size_t i = 0; i < 3; ++i)
        identity.matrix[i][i] = 1.0f;

    instanceContainer.insertInstance(geomGroups[0].triGeomContainer, identity);
    instanceContainer.insertInstance(geomGroups[0].blpGeomContainer, identity);
    instanceContainer.insertInstance(geomGroups[0].quadricGeomContainer, identity);
    size_t nNoInstancedGeoms = instanceContainer.instances.size();

    for (size_t i = 0; i < scene.instances.size(); ++i) {
        const auto &sceneInstance = scene.instances[i];
        auto iter = instanceMap.find(sceneInstance.name);

        if (iter != instanceMap.end()) {
            SquareMatrix<4> O2WTransfrom = sceneInstance.renderFromInstance->GetMatrix();
            SquareMatrix<4> W2OTransfrom = Inverse(O2WTransfrom).value();

            hiprtFrameMatrix transform;
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 4; ++k)
                    transform.matrix[j][k] = O2WTransfrom[j][k];

            GeometryGroup &geomGroup = iter->second;
            instanceContainer.insertInstance(geomGroup.triGeomContainer, transform);
            instanceContainer.insertInstance(geomGroup.blpGeomContainer, transform);
            instanceContainer.insertInstance(geomGroup.quadricGeomContainer, transform);
        }
    }

    this->scene =
        buildBVHForInstances(instanceContainer.instances, instanceContainer.transforms,
                             context, threadAllocators, threadCUDAStreams);
    LOG_VERBOSE("Finished building scene (TLAS)");

    ///////////////////////////////////////////////////////////////////////////
    // Bounds
    hiprtFloat3 aabbMin, aabbMax;
    HIPRT_CHECK(hiprtExportSceneAabb(context, this->scene, aabbMin, aabbMax));
    bounds.pMin = {aabbMin.x, aabbMin.y, aabbMin.z};
    bounds.pMax = {aabbMax.x, aabbMax.y, aabbMax.z};

    ///////////////////////////////////////////////////////////////////////////
    // Copy hitgroup records and offsets
    hipStream_t buildStream = threadCUDAStreams.Get();
    CUDA_CHECK(hipMalloc(&hgRecords,
                         instanceContainer.hgRecords.size() * sizeof(HitgroupRecord)));
    CUDA_CHECK(hipMemcpyAsync((void *)hgRecords, instanceContainer.hgRecords.data(),
                              sizeof(HitgroupRecord) * instanceContainer.hgRecords.size(),
                              hipMemcpyHostToDevice, buildStream));
    CUDA_CHECK(hipMalloc(&offsets, instanceContainer.offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(hipMemcpyAsync((void *)offsets, instanceContainer.offsets.data(),
                              sizeof(uint32_t) * instanceContainer.offsets.size(),
                              hipMemcpyHostToDevice, buildStream));

    ///////////////////////////////////////////////////////////////////////////
    // Create stack buffers
    constexpr uint32_t StackSize = 64;
    hiprtGlobalStackBufferInput stackBufferInput{hiprtStackTypeGlobal,
                                                 hiprtStackEntryTypeInteger, StackSize,
                                                 (uint32_t)maxQueueSize};
    HIPRT_CHECK(
        hiprtCreateGlobalStackBuffer(context, stackBufferInput, globalStackBuffer));

    constexpr uint32_t InstanceStackSize = 1;
    hiprtGlobalStackBufferInput instanceStackBufferInput{
        hiprtStackTypeGlobal, hiprtStackEntryTypeInstance, InstanceStackSize,
        (uint32_t)maxQueueSize};
    HIPRT_CHECK(hiprtCreateGlobalStackBuffer(context, instanceStackBufferInput,
                                             globalInstanceStackBuffer));

#if defined(PBRT_IS_WINDOWS)
    if (Options->useGPU)
        ReenableThreadPool();
#endif  // PBRT_IS_WINDOWS
}

void HiprtAggregate::IntersectClosest(int maxRays, const RayQueue *rayQueue,
                                      EscapedRayQueue *escapedRayQueue,
                                      HitAreaLightQueue *hitAreaLightQueue,
                                      MaterialEvalQueue *basicEvalMaterialQueue,
                                      MaterialEvalQueue *universalEvalMaterialQueue,
                                      MediumSampleQueue *mediumSampleQueue,
                                      RayQueue *nextRayQueue) const {
    std::pair<hipEvent_t, hipEvent_t> events =
        GetProfilerEvents("Trace closest hit rays");

    hipEventRecord(events.first);

    if (scene) {
        RayIntersectParameters params;
        params.traversable = scene;
        params.rayQueue = rayQueue;
        params.nextRayQueue = nextRayQueue;
        params.escapedRayQueue = escapedRayQueue;
        params.hitAreaLightQueue = hitAreaLightQueue;
        params.basicEvalMaterialQueue = basicEvalMaterialQueue;
        params.universalEvalMaterialQueue = universalEvalMaterialQueue;
        params.mediumSampleQueue = mediumSampleQueue;
        params.globalStackBuffer = globalStackBuffer;
        params.globalInstanceStackBuffer = globalInstanceStackBuffer;
        params.funcTable = module.funcTable;
        params.hgRecords = hgRecords;
        params.offsets = offsets;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect closest");
#endif

        hipDeviceptr_t paramsPtr;
        size_t paramsSize;
        CUDA_CHECK(
            hipModuleGetGlobal(&paramsPtr, &paramsSize, module.hipModule, "paramBuffer"));
        CUDA_CHECK(hipMemcpyAsync((void *)paramsPtr, (const void *)pbs.ptr,
                                  sizeof(params), hipMemcpyDeviceToDevice, cudaStream));

        hiprtLaunch(module.closestFunction, maxRays, 1, BlockSize, 1, nullptr,
                    cudaStream);
        CUDA_CHECK(hipEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(hipDeviceSynchronize());
        LOG_VERBOSE("Post-sync triangle intersect closest");
#endif
    }

    hipEventRecord(events.second);
};

void HiprtAggregate::IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                                     SOA<PixelSampleState> *pixelSampleState) const {
    std::pair<hipEvent_t, hipEvent_t> events = GetProfilerEvents("Trace shadow rays");

    hipEventRecord(events.first);

    if (scene) {
        RayIntersectParameters params;
        params.traversable = scene;
        params.shadowRayQueue = shadowRayQueue;
        params.pixelSampleState = *pixelSampleState;
        params.globalStackBuffer = globalStackBuffer;
        params.globalInstanceStackBuffer = globalInstanceStackBuffer;
        params.funcTable = module.funcTable;
        params.hgRecords = hgRecords;
        params.offsets = offsets;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect shadow");
#endif

        hipDeviceptr_t paramsPtr;
        size_t paramsSize;
        CUDA_CHECK(
            hipModuleGetGlobal(&paramsPtr, &paramsSize, module.hipModule, "paramBuffer"));
        CUDA_CHECK(hipMemcpyAsync((void *)paramsPtr, (const void *)pbs.ptr,
                                  sizeof(params), hipMemcpyDeviceToDevice, cudaStream));

        hiprtLaunch(module.shadowFunction, maxRays, 1, BlockSize, 1, nullptr, cudaStream);
        CUDA_CHECK(hipEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(hipDeviceSynchronize());
        LOG_VERBOSE("Post-sync intersect shadow");
#endif
    }

    hipEventRecord(events.second);
}

void HiprtAggregate::IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                                       SOA<PixelSampleState> *pixelSampleState) const {
    std::pair<hipEvent_t, hipEvent_t> events =
        GetProfilerEvents("Tracing shadow Tr rays");

    hipEventRecord(events.first);

    if (scene) {
        RayIntersectParameters params;
        params.traversable = scene;
        params.shadowRayQueue = shadowRayQueue;
        params.pixelSampleState = *pixelSampleState;
        params.globalStackBuffer = globalStackBuffer;
        params.globalInstanceStackBuffer = globalInstanceStackBuffer;
        params.funcTable = module.funcTable;
        params.hgRecords = hgRecords;
        params.offsets = offsets;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect shadow Tr");
#endif
        hipDeviceptr_t paramsPtr;
        size_t paramsSize;
        CUDA_CHECK(
            hipModuleGetGlobal(&paramsPtr, &paramsSize, module.hipModule, "paramBuffer"));
        CUDA_CHECK(hipMemcpyAsync((void *)paramsPtr, (const void *)pbs.ptr,
                                  sizeof(params), hipMemcpyDeviceToDevice, cudaStream));

        hiprtLaunch(module.shadowTrFunction, maxRays, 1, BlockSize, 1, nullptr,
                    cudaStream);
        CUDA_CHECK(hipEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(hipDeviceSynchronize());
        LOG_VERBOSE("Post-sync intersect shadow Tr");
#endif
    }

    hipEventRecord(events.second);
}

void HiprtAggregate::IntersectOneRandom(
    int maxRays, SubsurfaceScatterQueue *subsurfaceScatterQueue) const {
    std::pair<hipEvent_t, hipEvent_t> events =
        GetProfilerEvents("Tracing subsurface scattering probe rays");

    hipEventRecord(events.first);

    if (scene) {
        RayIntersectParameters params;
        params.traversable = scene;
        params.subsurfaceScatterQueue = subsurfaceScatterQueue;
        params.globalStackBuffer = globalStackBuffer;
        params.globalInstanceStackBuffer = globalInstanceStackBuffer;
        params.funcTable = module.funcTable;
        params.hgRecords = hgRecords;
        params.offsets = offsets;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect random");
#endif
        hipDeviceptr_t paramsPtr;
        size_t paramsSize;
        CUDA_CHECK(
            hipModuleGetGlobal(&paramsPtr, &paramsSize, module.hipModule, "paramBuffer"));
        CUDA_CHECK(hipMemcpyAsync((void *)paramsPtr, (const void *)pbs.ptr,
                                  sizeof(params), hipMemcpyDeviceToDevice, cudaStream));

        hiprtLaunch(module.oneRandomFunction, maxRays, 1, BlockSize, 1, nullptr,
                    cudaStream);
        CUDA_CHECK(hipEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(hipDeviceSynchronize());
        LOG_VERBOSE("Post-sync intersect random");
#endif
    }

    hipEventRecord(events.second);
}

std::map<int, TriQuadMesh> HiprtAggregate::PreparePLYMeshes(
    const std::vector<ShapeSceneEntity> &shapes,
    const std::map<std::string, FloatTexture> &floatTextures) {
    std::map<int, TriQuadMesh> plyMeshes;
    std::mutex mutex;
    ParallelFor(0, shapes.size(), [&](int64_t i) {
        const auto &shape = shapes[i];
        if (shape.name != "plymesh")
            return;

        std::string filename =
            ResolveFilename(shape.parameters.GetOneString("filename", ""));
        if (filename.empty())
            ErrorExit(&shape.loc, "plymesh: \"filename\" must be provided.");
        TriQuadMesh plyMesh = TriQuadMesh::ReadPLY(filename);  // todo: alloc
        if (!plyMesh.triIndices.empty() || !plyMesh.quadIndices.empty()) {
            plyMesh.ConvertToOnlyTriangles();

            Float edgeLength =
                shape.parameters.GetOneFloat("displacement.edgelength", 1.f);
            edgeLength *= Options->displacementEdgeScale;

            std::string displacementTexName = shape.parameters.GetTexture("displacement");
            if (!displacementTexName.empty()) {
                auto iter = floatTextures.find(displacementTexName);
                if (iter == floatTextures.end())
                    ErrorExit(&shape.loc, "%s: no such texture defined.",
                              displacementTexName);
                FloatTexture displacement = iter->second;

                LOG_VERBOSE("Starting to displace mesh \"%s\" with \"%s\"", filename,
                            displacementTexName);

                size_t origNumTris = plyMesh.triIndices.size() / 3;

                plyMesh = plyMesh.Displace(
                    [&](Point3f v0, Point3f v1) {
                        v0 = (*shape.renderFromObject)(v0);
                        v1 = (*shape.renderFromObject)(v1);
                        return Distance(v0, v1);
                    },
                    edgeLength,
                    [&](Point3f *pCPU, const Normal3f *nCPU, const Point2f *uvCPU,
                        int nVertices) {
                        Point3f *p;
                        Normal3f *n;
                        Point2f *uv;
                        CUDA_CHECK(hipMallocManaged(&p, nVertices * sizeof(Point3f)));
                        CUDA_CHECK(hipMallocManaged(&n, nVertices * sizeof(Normal3f)));
                        CUDA_CHECK(hipMallocManaged(&uv, nVertices * sizeof(Point2f)));

                        std::memcpy(p, pCPU, nVertices * sizeof(Point3f));
                        std::memcpy(n, nCPU, nVertices * sizeof(Normal3f));
                        std::memcpy(uv, uvCPU, nVertices * sizeof(Point2f));

                        GPUParallelFor(
                            "Evaluate Displacement", nVertices, [=] PBRT_GPU(int i) {
                                TextureEvalContext ctx;
                                ctx.p = p[i];
                                ctx.uv = uv[i];
                                Float d = UniversalTextureEvaluator()(displacement, ctx);
                                p[i] += Vector3f(d * n[i]);
                            });
                        GPUWait();

                        std::memcpy(pCPU, p, nVertices * sizeof(Point3f));

                        CUDA_CHECK(hipFree(p));
                        CUDA_CHECK(hipFree(n));
                        CUDA_CHECK(hipFree(uv));
                    },
                    &shape.loc);

                displacedTrisDelta += plyMesh.triIndices.size() / 3 - origNumTris;

                LOG_VERBOSE("Finished displacing mesh \"%s\" with \"%s\" -> %d tris",
                            filename, displacementTexName, plyMesh.triIndices.size() / 3);
            }
        }

        std::lock_guard<std::mutex> lock(mutex);
        plyMeshes[i] = std::move(plyMesh);
    });

    return plyMeshes;
}

HiprtAggregate::ParamBufferState &HiprtAggregate::getParamBuffer(
    const RayIntersectParameters &params) const {
    CHECK(nextParamOffset < paramsPool.size());

    ParamBufferState &pbs = paramsPool[nextParamOffset];
    if (++nextParamOffset == paramsPool.size())
        nextParamOffset = 0;
    if (!pbs.used)
        pbs.used = true;
    else
        CUDA_CHECK(hipEventSynchronize(pbs.finishedEvent));

    // Copy to host-side pinned memory
    memcpy(pbs.hostPtr, &params, sizeof(params));
    CUDA_CHECK(hipMemcpyAsync((void *)pbs.ptr, pbs.hostPtr, sizeof(params),
                              hipMemcpyHostToDevice, cudaStream));

    return pbs;
}

BilinearPatchMesh *HiprtAggregate::diceCurveToBLP(const ShapeSceneEntity &shape,
                                                  int nDiceU, int nDiceV,
                                                  Allocator alloc) {
    CHECK_EQ(shape.name, "curve");
    const ParameterDictionary &parameters = shape.parameters;
    const FileLoc *loc = &shape.loc;

    ++nCurves;

    // Extract parameters; the following ~90 lines of code are,
    // unfortunately, copied from Curve::Create.  We would like to avoid
    // the overhead of splitting the curve and creating Curve objects, so
    // here we go..
    Float width = parameters.GetOneFloat("width", 1.f);
    Float width0 = parameters.GetOneFloat("width0", width);
    Float width1 = parameters.GetOneFloat("width1", width);

    int degree = parameters.GetOneInt("degree", 3);
    if (degree != 2 && degree != 3) {
        Error(loc, "Invalid degree %d: only degree 2 and 3 curves are supported.",
              degree);
        return {};
    }

    std::string basis = parameters.GetOneString("basis", "bezier");
    if (basis != "bezier" && basis != "bspline") {
        Error(loc,
              "Invalid basis \"%s\": only \"bezier\" and \"bspline\" are "
              "supported.",
              basis);
        return {};
    }

    int nSegments;
    std::vector<Point3f> cp = parameters.GetPoint3fArray("P");
    bool bezierBasis = (basis == "bezier");
    if (bezierBasis) {
        // After the first segment, which uses degree+1 control points,
        // subsequent segments reuse the last control point of the previous
        // one and then use degree more control points.
        if (((cp.size() - 1 - degree) % degree) != 0) {
            Error(loc,
                  "Invalid number of control points %d: for the degree %d "
                  "Bezier basis %d + n * %d are required, for n >= 0.",
                  (int)cp.size(), degree, degree + 1, degree);
            return {};
        }
        nSegments = (cp.size() - 1) / degree;
    } else {
        if (cp.size() < degree + 1) {
            Error(loc,
                  "Invalid number of control points %d: for the degree %d "
                  "b-spline basis, must have >= %d.",
                  int(cp.size()), degree, degree + 1);
            return {};
        }
        nSegments = cp.size() - degree;
    }

    CurveType type;
    std::string curveType = parameters.GetOneString("type", "flat");
    if (curveType == "flat")
        type = CurveType::Flat;
    else if (curveType == "ribbon")
        type = CurveType::Ribbon;
    else if (curveType == "cylinder")
        type = CurveType::Cylinder;
    else {
        Error(loc, R"(Unknown curve type "%s".  Using "cylinder".)", curveType);
        type = CurveType::Cylinder;
    }

    std::vector<Normal3f> n = parameters.GetNormal3fArray("N");
    if (!n.empty()) {
        if (type != CurveType::Ribbon) {
            Warning("Curve normals are only used with \"ribbon\" type curves.");
            n = {};
        } else if (n.size() != nSegments + 1) {
            Error(loc,
                  "Invalid number of normals %d: must provide %d normals for "
                  "ribbon curves with %d segments.",
                  int(n.size()), nSegments + 1, nSegments);
            return {};
        }
        for (Normal3f &nn : n)
            Normalize(nn);
    } else if (type == CurveType::Ribbon) {
        Error(loc, "Must provide normals \"N\" at curve endpoints with ribbon "
                   "curves.");
        return {};
    }

    // Start dicing...
    std::vector<int> blpIndices;
    std::vector<Point3f> blpP;
    std::vector<Normal3f> blpN;
    std::vector<Point2f> blpUV;

    int lastCPOffset = -1;
    pstd::array<Point3f, 4> segCpBezier;

    for (int i = 0; i <= nDiceU; ++i) {
        Float u = Float(i) / Float(nDiceU);
        Float width = Lerp(u, width0, width1);

        int segmentIndex = int(u * nSegments);
        if (segmentIndex == nSegments)  // u == 1...
            --segmentIndex;

        // Compute offset into original control points for current u
        int cpOffset;
        if (bezierBasis)
            cpOffset = segmentIndex * degree;
        else
            // Uniform b-spline.
            cpOffset = segmentIndex;

        if (cpOffset != lastCPOffset) {
            // update segCpBezier
            if (bezierBasis) {
                if (degree == 2) {
                    // Elevate to degree 3.
                    segCpBezier = ElevateQuadraticBezierToCubic(
                        pstd::MakeConstSpan(cp).subspan(cpOffset, 3));
                } else {
                    // All set.
                    for (int i = 0; i < 4; ++i)
                        segCpBezier[i] = cp[cpOffset + i];
                }
            } else {
                // Uniform b-spline.
                if (degree == 2) {
                    pstd::array<Point3f, 3> bezCp = QuadraticBSplineToBezier(
                        pstd::MakeConstSpan(cp).subspan(cpOffset, 3));
                    segCpBezier =
                        ElevateQuadraticBezierToCubic(pstd::MakeConstSpan(bezCp));
                } else {
                    segCpBezier = CubicBSplineToBezier(
                        pstd::MakeConstSpan(cp).subspan(cpOffset, 4));
                }
            }
            lastCPOffset = cpOffset;
        }

        Float uSeg = (u * nSegments) - segmentIndex;
        DCHECK(uSeg >= 0 && uSeg <= 1);

        Vector3f dpdu;
        Point3f p = EvaluateCubicBezier(segCpBezier, uSeg, &dpdu);

        switch (type) {
        case CurveType::Ribbon: {
            Float normalAngle = AngleBetween(n[segmentIndex], n[segmentIndex + 1]);
            Float invSinNormalAngle = 1 / std::sin(normalAngle);

            Normal3f nu;
            if (normalAngle == 0)
                nu = n[segmentIndex];
            else {
                Float sin0 = std::sin((1 - uSeg) * normalAngle) * invSinNormalAngle;
                Float sin1 = std::sin(uSeg * normalAngle) * invSinNormalAngle;
                nu = sin0 * n[segmentIndex] + sin1 * n[segmentIndex + 1];
            }
            Vector3f dpdv = Normalize(Cross(nu, dpdu)) * width;

            blpP.push_back(p - dpdv / 2);
            blpP.push_back(p + dpdv / 2);
            blpUV.push_back(Point2f(u, 0));
            blpUV.push_back(Point2f(u, 1));

            if (i > 0) {
                blpIndices.push_back(2 * (i - 1));
                blpIndices.push_back(2 * (i - 1) + 1);
                blpIndices.push_back(2 * i);
                blpIndices.push_back(2 * i + 1);
            }
            break;
        }
        case CurveType::Flat:
        case CurveType::Cylinder: {
            Vector3f ortho[2];
            CoordinateSystem(Normalize(dpdu), &ortho[0], &ortho[1]);
            ortho[0] *= width / 2;
            ortho[1] *= width / 2;

            // Repeat the first/last vertex so we can assign different
            // texture coordinates...
            for (int v = 0; v <= nDiceV; ++v) {
                Float angle = Float(v) / nDiceV * 2 * Pi;
                blpP.push_back(p + ortho[0] * std::cos(angle) +
                               ortho[1] * std::sin(angle));
                blpN.push_back(Normal3f(Normalize(blpP.back() - p)));
                blpUV.push_back(Point2f(u, Float(v) / nDiceV));
            }

            if (i > 0) {
                for (int v = 0; v < nDiceV; ++v) {
                    // Indexing is funny due to doubled-up last vertex
                    blpIndices.push_back((nDiceV + 1) * (i - 1) + v);
                    blpIndices.push_back((nDiceV + 1) * (i - 1) + v + 1);
                    blpIndices.push_back((nDiceV + 1) * i + v);
                    blpIndices.push_back((nDiceV + 1) * i + v + 1);
                }
            }
            break;
        }
        }
    }

    nBLPsForCurves += blpIndices.size() / 4;

    return alloc.new_object<BilinearPatchMesh>(
        *shape.renderFromObject, shape.reverseOrientation, blpIndices, blpP, blpN, blpUV,
        std::vector<int>(), nullptr, alloc);
}

HiprtAggregate::GeometryContainer HiprtAggregate::buildBVHForTriangles(
    const std::vector<ShapeSceneEntity> &shapes,
    const std::map<int, TriQuadMesh> &plyMeshes, hiprtContext context,
    const std::map<std::string, FloatTexture> &floatTextures,
    const std::map<std::string, Material> &namedMaterials,
    const std::vector<Material> &materials, const std::map<std::string, Medium> &media,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    ThreadLocal<Allocator> &threadAllocators,
    ThreadLocal<hipStream_t> &threadCUDAStreams) {
    // Count how many of the shapes are triangle meshes
    std::vector<size_t> meshIndexToShapeIndex;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto &shape = shapes[i];
        if (shape.name == "trianglemesh" || shape.name == "plymesh" ||
            shape.name == "loopsubdiv")
            meshIndexToShapeIndex.push_back(i);
    }

    size_t nMeshes = meshIndexToShapeIndex.size();
    if (nMeshes == 0)
        return {};

    LOG_VERBOSE("Building triangle BLAS");

    std::vector<TriangleMesh *> meshes(nMeshes, nullptr);
    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        Allocator alloc = threadAllocators.Get();
        size_t shapeIndex = meshIndexToShapeIndex[meshIndex];
        const auto &shape = shapes[shapeIndex];

        TriangleMesh *mesh = nullptr;
        if (shape.name == "trianglemesh") {
            mesh = Triangle::CreateMesh(shape.renderFromObject, shape.reverseOrientation,
                                        shape.parameters, &shape.loc, alloc);
            CHECK(mesh != nullptr);
        } else if (shape.name == "loopsubdiv") {
            // Copied from pbrt/shapes.cpp... :-p
            int nLevels = shape.parameters.GetOneInt("levels", 3);
            std::vector<int> vertexIndices = shape.parameters.GetIntArray("indices");
            if (vertexIndices.empty())
                ErrorExit(&shape.loc, "Vertex indices \"indices\" not "
                                      "provided for LoopSubdiv shape.");

            std::vector<Point3f> P = shape.parameters.GetPoint3fArray("P");
            if (P.empty())
                ErrorExit(&shape.loc, "Vertex positions \"P\" not provided "
                                      "for LoopSubdiv shape.");

            // don't actually use this for now...
            std::string scheme = shape.parameters.GetOneString("scheme", "loop");

            mesh = LoopSubdivide(shape.renderFromObject, shape.reverseOrientation,
                                 nLevels, vertexIndices, P, alloc);
            CHECK(mesh != nullptr);
        } else if (shape.name == "plymesh") {
            auto plyIter = plyMeshes.find(shapeIndex);
            CHECK(plyIter != plyMeshes.end());
            const TriQuadMesh &plyMesh = plyIter->second;

            if (!plyMesh.quadIndices.empty() && shape.lightIndex != -1) {
#if 0
                // If you'd like to know what they are...
                for (int i = 0; i < plyMesh.quadIndices.size(); ++i)
                    Printf("%s\n", plyMesh.p[plyMesh.quadIndices[i]]);
#endif
                // This would be nice to fix, but it involves some
                // plumbing and it's a rare case. The underlying issue
                // is that when we create AreaLights for emissive
                // shapes earlier, we're not expecting this..
                std::string filename =
                    ResolveFilename(shape.parameters.GetOneString("filename", ""));
                ErrorExit(&shape.loc,
                          "%s: PLY file being used as an area light has quads--"
                          "this is currently unsupported. Please replace them with "
                          "\"bilinearmesh\" "
                          "shapes as a workaround. (Sorry!).",
                          filename);
            }

            mesh = alloc.new_object<TriangleMesh>(
                *shape.renderFromObject, shape.reverseOrientation, plyMesh.triIndices,
                plyMesh.p, std::vector<Vector3f>(), plyMesh.n, plyMesh.uv,
                plyMesh.faceIndices, alloc);
        } else
            LOG_FATAL("Logic error in GPUAggregate::buildBVHForTriangles()");

        meshes[meshIndex] = mesh;
    });

    GeometryContainer geomContainer(nMeshes);

    ParallelFor(0, nMeshes, [&](int64_t startIndex, int64_t endIndex) {
        Allocator alloc = threadAllocators.Get();

        for (int meshIndex = startIndex; meshIndex < endIndex; ++meshIndex) {
            TriangleMesh *mesh = meshes[meshIndex];
            size_t shapeIndex = meshIndexToShapeIndex[meshIndex];
            const auto &shape = shapes[shapeIndex];
            FloatTexture alphaTexture = getAlphaTexture(shape, floatTextures, alloc);
            Material material = getMaterial(shape, namedMaterials, materials);

            // Do this here, after the alpha texture has been consumed.
            shape.parameters.ReportUnused();

            HitgroupRecord &hgRecord = geomContainer.hgRecords[meshIndex];
            hgRecord.type = HitgroupRecord::TriangleMesh;
            hgRecord.triRec.mesh = mesh;
            hgRecord.triRec.material = material;
            hgRecord.triRec.alphaTexture = alphaTexture;
            hgRecord.triRec.areaLights = {};
            if (shape.lightIndex != -1) {
                if (!material)
                    Warning(&shape.loc, "Ignoring area light specification for shape "
                                        "with \"interface\" material.");
                else {
                    // Note: this will hit if we try to have an instance as an area
                    // light.
                    auto iter = shapeIndexToAreaLights.find(shapeIndex);
                    CHECK(iter != shapeIndexToAreaLights.end());
                    CHECK_EQ(iter->second->size(), mesh->nTriangles);
                    hgRecord.triRec.areaLights = pstd::MakeSpan(*iter->second);
                }
            }
            hgRecord.triRec.mediumInterface = getMediumInterface(shape, media, alloc);
        }
    });

    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
    options.batchBuildMaxPrimCount = 512u;

    size_t nTris = 0;
    size_t nVerts = 0;
    std::vector<size_t> triOffsets;
    std::vector<size_t> vertsOffsets;
    for (size_t meshIndex = 0; meshIndex < nMeshes; ++meshIndex) {
        TriangleMesh *mesh = meshes[meshIndex];
        triOffsets.push_back(nTris);
        vertsOffsets.push_back(nVerts);
        nTris += mesh->nTriangles;
        nVerts += mesh->nVertices;
    }

    hiprtInt3 *tris;
    CUDA_CHECK(hipMalloc(&tris, sizeof(hiprtInt3) * nTris));

    hiprtFloat3 *verts;
    CUDA_CHECK(hipMalloc(&verts, sizeof(hiprtFloat3) * nVerts));

    std::vector<hiprtInt3> triBuffer(nTris);
    std::vector<hiprtFloat3> vertBuffer(nVerts);

    std::vector<hiprtGeometryBuildInput> geomInputs(nMeshes);
    std::vector<hiprtGeometry> geoms(nMeshes);
    std::vector<hiprtGeometry *> geomAddrs(nMeshes);
    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        TriangleMesh *mesh = meshes[meshIndex];
        size_t triOffset = triOffsets[meshIndex];
        size_t vertOffset = vertsOffsets[meshIndex];

        hiprtTriangleMeshPrimitive prim;
        prim.triangleCount = mesh->nTriangles;
        prim.triangleStride = sizeof(hiprtInt3);
        prim.triangleIndices = tris + triOffset;
        prim.vertexCount = mesh->nVertices;
        prim.vertexStride = sizeof(hiprtFloat3);
        prim.vertices = verts + vertOffset;

        hiprtGeometryBuildInput &geomInput = geomInputs[meshIndex];
        geomInput.type = hiprtPrimitiveTypeTriangleMesh;
        geomInput.primitive.triangleMesh = prim;
        geomInput.geomType = 0;

        geomAddrs[meshIndex] = &geoms[meshIndex];

        std::memcpy(&triBuffer[triOffset], mesh->vertexIndices,
                    sizeof(hiprtInt3) * mesh->nTriangles);
        std::memcpy(&vertBuffer[vertOffset], mesh->p,
                    sizeof(hiprtFloat3) * mesh->nVertices);
    });

    hipStream_t buildStream = threadCUDAStreams.Get();
    CUDA_CHECK(hipMemcpyAsync(tris, triBuffer.data(), sizeof(hiprtInt3) * nTris,
                              hipMemcpyHostToDevice, buildStream));
    CUDA_CHECK(hipMemcpyAsync(verts, vertBuffer.data(), sizeof(hiprtFloat3) * nVerts,
                              hipMemcpyHostToDevice, buildStream));

    size_t geomTempSize;
    HIPRT_CHECK(hiprtGetGeometriesBuildTemporaryBufferSize(
        context, nMeshes, geomInputs.data(), options, geomTempSize));

    hiprtDevicePtr tempBuffer;
    CUDA_CHECK(hipMalloc(&tempBuffer, geomTempSize));

    HIPRT_CHECK(hiprtCreateGeometries(context, nMeshes, geomInputs.data(), options,
                                      geomAddrs.data()));

    LOG_VERBOSE("Starting to build triangle mesh geometries");
    HIPRT_CHECK(hiprtBuildGeometries(context, hiprtBuildOperationBuild, nMeshes,
                                     geomInputs.data(), options, tempBuffer, buildStream,
                                     geoms.data()));
    LOG_VERBOSE("Finished building triangle mesh geometries");

    CUDA_CHECK(hipFree(tris));
    CUDA_CHECK(hipFree(verts));
    CUDA_CHECK(hipFree(tempBuffer));

    hiprtFrameMatrix identity{};
    for (size_t i = 0; i < 3; ++i)
        identity.matrix[i][i] = 1.0f;

    std::vector<hiprtInstance> instances(nMeshes);
    std::vector<hiprtFrameMatrix> transforms(nMeshes);
    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        instances[meshIndex].type = hiprtInstanceTypeGeometry;
        instances[meshIndex].geometry = geoms[meshIndex];
        transforms[meshIndex] = identity;
    });

    geomContainer.scene = buildBVHForInstances(instances, transforms, context,
                                               threadAllocators, threadCUDAStreams);

    return geomContainer;
}

HiprtAggregate::GeometryContainer HiprtAggregate::buildBVHForBLPs(
    const std::vector<ShapeSceneEntity> &shapes, hiprtContext context,
    const std::map<std::string, FloatTexture> &floatTextures,
    const std::map<std::string, Material> &namedMaterials,
    const std::vector<Material> &materials, const std::map<std::string, Medium> &media,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    ThreadLocal<Allocator> &threadAllocators,
    ThreadLocal<hipStream_t> &threadCUDAStreams) {
    // Count how many BLP meshes there are in shapes
    std::vector<size_t> meshIndexToShapeIndex;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto &shape = shapes[i];
        if (shape.name == "bilinearmesh" || shape.name == "curve")
            meshIndexToShapeIndex.push_back(i);
    }

    size_t nMeshes = meshIndexToShapeIndex.size();
    if (nMeshes == 0)
        return {};

    LOG_VERBOSE("Building bilinear patch BLAS");

    // Create meshes
    std::vector<BilinearPatchMesh *> meshes(nMeshes, nullptr);
    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        Allocator alloc = threadAllocators.Get();
        size_t shapeIndex = meshIndexToShapeIndex[meshIndex];
        const auto &shape = shapes[shapeIndex];

        if (shape.name == "bilinearmesh") {
            BilinearPatchMesh *mesh = BilinearPatch::CreateMesh(
                shape.renderFromObject, shape.reverseOrientation, shape.parameters,
                &shape.loc, alloc);
            meshes[meshIndex] = mesh;
        } else if (shape.name == "curve") {
            BilinearPatchMesh *curveMesh =
                diceCurveToBLP(shape, 5 /* nseg */, 5 /* nvert */, alloc);
            if (curveMesh) {
                meshes[meshIndex] = curveMesh;
            }
        }
    });

    GeometryContainer geomContainer(nMeshes);

    ParallelFor(0, nMeshes, [&](int64_t startIndex, int64_t endIndex) {
        Allocator alloc = threadAllocators.Get();

        for (int meshIndex = startIndex; meshIndex < endIndex; ++meshIndex) {
            BilinearPatchMesh *mesh = meshes[meshIndex];
            size_t shapeIndex = meshIndexToShapeIndex[meshIndex];
            const auto &shape = shapes[shapeIndex];
            Material material = getMaterial(shape, namedMaterials, materials);
            FloatTexture alphaTexture = getAlphaTexture(shape, floatTextures, alloc);

            // After "alpha" has been consumed...
            shape.parameters.ReportUnused();

            HitgroupRecord &hgRecord = geomContainer.hgRecords[meshIndex];
            hgRecord.type = HitgroupRecord::BilinearMesh;
            hgRecord.blpRec.mesh = mesh;
            hgRecord.blpRec.material = material;
            hgRecord.blpRec.alphaTexture = alphaTexture;
            hgRecord.blpRec.areaLights = {};
            if (shape.lightIndex != -1) {
                if (!material)
                    Warning(&shape.loc,
                            "Ignoring area light specification for shape with "
                            "\"interface\" material.");
                else {
                    auto iter = shapeIndexToAreaLights.find(shapeIndex);
                    // Note: this will hit if we try to have an instance as an area
                    // light.
                    CHECK(iter != shapeIndexToAreaLights.end());
                    CHECK_EQ(iter->second->size(), mesh->nPatches);
                    hgRecord.blpRec.areaLights = pstd::MakeSpan(*iter->second);
                }
            }
            hgRecord.blpRec.mediumInterface = getMediumInterface(shape, media, alloc);
        }
    });

    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
    options.batchBuildMaxPrimCount = 512u;

    size_t nAabbs = 0;
    std::vector<size_t> aabbOffsets;
    for (size_t meshIndex = 0; meshIndex < nMeshes; ++meshIndex) {
        BilinearPatchMesh *mesh = meshes[meshIndex];
        aabbOffsets.push_back(nAabbs);
        nAabbs += mesh->nPatches;
    }

    hiprtFloat4 *aabbs;
    CUDA_CHECK(hipMalloc(&aabbs, sizeof(hiprtFloat4) * 2 * nAabbs));

    std::vector<hiprtFloat4> aabbBuffer(2 * nAabbs);

    std::vector<hiprtGeometryBuildInput> geomInputs(nMeshes);
    std::vector<hiprtGeometry> geoms(nMeshes);
    std::vector<hiprtGeometry *> geomAddrs(nMeshes);
    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        BilinearPatchMesh *mesh = meshes[meshIndex];
        size_t aabbIndex = aabbOffsets[meshIndex];

        hiprtAABBListPrimitive prim;
        prim.aabbCount = mesh->nPatches;
        prim.aabbStride = 2 * sizeof(hiprtFloat4);
        prim.aabbs = aabbs + 2 * aabbIndex;

        hiprtGeometryBuildInput &geomInput = geomInputs[meshIndex];
        geomInput.type = hiprtPrimitiveTypeAABBList;
        geomInput.primitive.aabbList = prim;
        geomInput.geomType = 1;

        geomAddrs[meshIndex] = &geoms[meshIndex];

        for (int patchIndex = 0; patchIndex < mesh->nPatches; ++patchIndex) {
            Bounds3f patchBounds;
            for (int i = 0; i < 4; ++i)
                patchBounds =
                    Union(patchBounds, mesh->p[mesh->vertexIndices[4 * patchIndex + i]]);
            hiprtFloat4 aabbMin =
                hiprtFloat4{float(patchBounds.pMin.x), float(patchBounds.pMin.y),
                            float(patchBounds.pMin.z), 0.0f};
            hiprtFloat4 aabbMax =
                hiprtFloat4{float(patchBounds.pMax.x), float(patchBounds.pMax.y),
                            float(patchBounds.pMax.z), 0.0f};
            aabbBuffer[2 * aabbIndex + 0] = aabbMin;
            aabbBuffer[2 * aabbIndex + 1] = aabbMax;
            ++aabbIndex;
        }
    });

    hipStream_t buildStream = threadCUDAStreams.Get();
    CUDA_CHECK(hipMemcpyAsync(aabbs, aabbBuffer.data(), sizeof(hiprtFloat4) * 2 * nAabbs,
                              hipMemcpyHostToDevice, buildStream));

    size_t geomTempSize;
    HIPRT_CHECK(hiprtGetGeometriesBuildTemporaryBufferSize(
        context, nMeshes, geomInputs.data(), options, geomTempSize));

    hiprtDevicePtr tempBuffer;
    CUDA_CHECK(hipMalloc(&tempBuffer, geomTempSize));

    HIPRT_CHECK(hiprtCreateGeometries(context, nMeshes, geomInputs.data(), options,
                                      geomAddrs.data()));

    LOG_VERBOSE("Starting to build bilinear patch mesh geometries");
    HIPRT_CHECK(hiprtBuildGeometries(context, hiprtBuildOperationBuild, nMeshes,
                                     geomInputs.data(), options, tempBuffer, buildStream,
                                     geoms.data()));
    LOG_VERBOSE("Finished building bilinear patch mesh geometries");

    CUDA_CHECK(hipFree(aabbs));
    CUDA_CHECK(hipFree(tempBuffer));

    hiprtFrameMatrix identity{};
    for (size_t i = 0; i < 3; ++i)
        identity.matrix[i][i] = 1.0f;

    std::vector<hiprtInstance> instances(nMeshes);
    std::vector<hiprtFrameMatrix> transforms(nMeshes);
    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        instances[meshIndex].type = hiprtInstanceTypeGeometry;
        instances[meshIndex].geometry = geoms[meshIndex];
        transforms[meshIndex] = identity;
    });

    geomContainer.scene = buildBVHForInstances(instances, transforms, context,
                                               threadAllocators, threadCUDAStreams);

    return geomContainer;
}

HiprtAggregate::GeometryContainer HiprtAggregate::buildBVHForQuadrics(
    const std::vector<ShapeSceneEntity> &shapes, hiprtContext context,
    const std::map<std::string, FloatTexture> &floatTextures,
    const std::map<std::string, Material> &namedMaterials,
    const std::vector<Material> &materials, const std::map<std::string, Medium> &media,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    ThreadLocal<Allocator> &threadAllocators,
    ThreadLocal<hipStream_t> &threadCUDAStreams) {
    int nQuadrics = 0;
    for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
        const auto &s = shapes[shapeIndex];
        if (s.name == "sphere" || s.name == "cylinder" || s.name == "disk")
            ++nQuadrics;
    }

    if (nQuadrics == 0)
        return {};

    LOG_VERBOSE("Building quadric BLAS");

    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
    options.batchBuildMaxPrimCount = 512u;

    Allocator alloc = threadAllocators.Get();
    hiprtFloat4 *aabbs;
    CUDA_CHECK(hipMalloc(&aabbs, sizeof(hiprtFloat4) * 2 * nQuadrics));

    std::vector<hiprtFloat4> aabbBuffer(2 * nQuadrics);

    int quadricIndex = 0;
    GeometryContainer geomContainer(nQuadrics);
    for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
        const auto &s = shapes[shapeIndex];
        if (s.name != "sphere" && s.name != "cylinder" && s.name != "disk")
            continue;

        pstd::vector<Shape> shapes = Shape::Create(
            s.name, s.renderFromObject, s.objectFromRender, s.reverseOrientation,
            s.parameters, floatTextures, &s.loc, alloc);
        if (shapes.empty())
            continue;
        CHECK_EQ(1, shapes.size());
        Shape shape = shapes[0];

        Bounds3f shapeBounds = shape.Bounds();
        hiprtFloat4 aabbMin =
            hiprtFloat4{float(shapeBounds.pMin.x), float(shapeBounds.pMin.y),
                        float(shapeBounds.pMin.z), 0.0f};
        hiprtFloat4 aabbMax =
            hiprtFloat4{float(shapeBounds.pMax.x), float(shapeBounds.pMax.y),
                        float(shapeBounds.pMax.z), 0.0f};
        aabbBuffer[2 * quadricIndex + 0] = aabbMin;
        aabbBuffer[2 * quadricIndex + 1] = aabbMax;

        // Find alpha texture, if present.
        Material material = getMaterial(s, namedMaterials, materials);
        FloatTexture alphaTexture = getAlphaTexture(s, floatTextures, alloc);

        // Once again, after any alpha texture is created...
        s.parameters.ReportUnused();

        HitgroupRecord &hgRecord = geomContainer.hgRecords[quadricIndex];
        hgRecord.type = HitgroupRecord::Quadric;
        hgRecord.quadricRec.shape = shape;
        hgRecord.quadricRec.material = material;
        hgRecord.quadricRec.alphaTexture = alphaTexture;
        hgRecord.quadricRec.areaLight = nullptr;
        if (s.lightIndex != -1) {
            if (!material)
                Warning(&s.loc, "Ignoring area light specification for shape with "
                                "\"interface\" material.");
            else {
                auto iter = shapeIndexToAreaLights.find(shapeIndex);
                // Note: this will hit if we try to have an instance as an area
                // light.
                CHECK(iter != shapeIndexToAreaLights.end());
                CHECK_EQ(iter->second->size(), 1);
                hgRecord.quadricRec.areaLight = (*iter->second)[0];
            }
        }
        hgRecord.quadricRec.mediumInterface = getMediumInterface(s, media, alloc);

        ++quadricIndex;
    }
    nQuadrics = quadricIndex;
    geomContainer.resize(nQuadrics);

    std::vector<hiprtGeometryBuildInput> geomInputs(nQuadrics);
    std::vector<hiprtGeometry> geoms(nQuadrics);
    std::vector<hiprtGeometry *> geomAddrs(nQuadrics);
    ParallelFor(0, nQuadrics, [&](int64_t quadricIndex) {
        hiprtAABBListPrimitive prim;
        prim.aabbCount = 1;
        prim.aabbStride = 2 * sizeof(hiprtFloat4);
        prim.aabbs = aabbs + 2 * quadricIndex;

        hiprtGeometryBuildInput &geomInput = geomInputs[quadricIndex];
        geomInput.type = hiprtPrimitiveTypeAABBList;
        geomInput.primitive.aabbList = prim;
        geomInput.geomType = 2;

        geomAddrs[quadricIndex] = &geoms[quadricIndex];
    });

    hipStream_t buildStream = threadCUDAStreams.Get();
    CUDA_CHECK(hipMemcpyAsync(aabbs, aabbBuffer.data(),
                              sizeof(hiprtFloat4) * 2 * nQuadrics, hipMemcpyHostToDevice,
                              buildStream));

    size_t geomTempSize;
    HIPRT_CHECK(hiprtGetGeometriesBuildTemporaryBufferSize(
        context, nQuadrics, geomInputs.data(), options, geomTempSize));

    hiprtDevicePtr tempBuffer;
    CUDA_CHECK(hipMalloc(&tempBuffer, geomTempSize));

    HIPRT_CHECK(hiprtCreateGeometries(context, nQuadrics, geomInputs.data(), options,
                                      geomAddrs.data()));

    LOG_VERBOSE("Starting to build quadric geometries");
    HIPRT_CHECK(hiprtBuildGeometries(context, hiprtBuildOperationBuild, nQuadrics,
                                     geomInputs.data(), options, tempBuffer, buildStream,
                                     geoms.data()));
    LOG_VERBOSE("Finished building quadric geometries");

    CUDA_CHECK(hipFree(aabbs));
    CUDA_CHECK(hipFree(tempBuffer));

    hiprtFrameMatrix identity{};
    for (size_t i = 0; i < 3; ++i)
        identity.matrix[i][i] = 1.0f;

    std::vector<hiprtInstance> instances(nQuadrics);
    std::vector<hiprtFrameMatrix> transforms(nQuadrics);
    ParallelFor(0, nQuadrics, [&](int64_t quadricIndex) {
        instances[quadricIndex].type = hiprtInstanceTypeGeometry;
        instances[quadricIndex].geometry = geoms[quadricIndex];
        transforms[quadricIndex] = identity;
    });

    geomContainer.scene = buildBVHForInstances(instances, transforms, context,
                                               threadAllocators, threadCUDAStreams);

    return geomContainer;
}

hiprtScene HiprtAggregate::buildBVHForInstances(
    const std::vector<hiprtInstance> &instances,
    const std::vector<hiprtFrameMatrix> &transforms, hiprtContext context,
    ThreadLocal<Allocator> &threadAllocators,
    ThreadLocal<hipStream_t> &threadCUDAStreams) {
    CHECK(transforms.size() == instances.size());
    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;
    options.batchBuildMaxPrimCount = 512u;

    hiprtSceneBuildInput sceneInput;
    sceneInput.instanceCount = instances.size();
    sceneInput.frameCount = instances.size();
    sceneInput.instanceFrames = nullptr;
    sceneInput.instanceMasks = nullptr;
    sceneInput.instanceTransformHeaders = nullptr;
    sceneInput.instances = nullptr;
    sceneInput.frameType = hiprtFrameTypeMatrix;

    size_t sceneTempSize;
    HIPRT_CHECK(hiprtGetSceneBuildTemporaryBufferSize(context, sceneInput, options,
                                                      sceneTempSize));

    hiprtDevicePtr tempBuffer;
    CUDA_CHECK(hipMalloc(&tempBuffer, sceneTempSize));
    CUDA_CHECK(
        hipMalloc(&sceneInput.instances, sizeof(hiprtInstance) * instances.size()));
    CUDA_CHECK(hipMalloc(&sceneInput.instanceFrames,
                         sizeof(hiprtFrameMatrix) * transforms.size()));

    hipStream_t buildStream = threadCUDAStreams.Get();
    CUDA_CHECK(hipMemcpyAsync((void *)sceneInput.instances, instances.data(),
                              sizeof(hiprtInstance) * instances.size(),
                              hipMemcpyHostToDevice, buildStream));
    CUDA_CHECK(hipMemcpyAsync((void *)sceneInput.instanceFrames, transforms.data(),
                              sizeof(hiprtFrameMatrix) * transforms.size(),
                              hipMemcpyHostToDevice, buildStream));

    hiprtScene scene;
    HIPRT_CHECK(hiprtCreateScene(context, sceneInput, options, scene));

    LOG_VERBOSE("Started to build scene");
    HIPRT_CHECK(hiprtBuildScene(context, hiprtBuildOperationBuild, sceneInput, options,
                                tempBuffer, buildStream, scene));
    LOG_VERBOSE("Finished building scene");

    CUDA_CHECK(hipFree(tempBuffer));
    CUDA_CHECK(hipFree(sceneInput.instances));
    CUDA_CHECK(hipFree(sceneInput.instanceFrames));

    return scene;
}

static void loadFile(const std::string& path, std::vector<char> &dst) {
    std::fstream f(path, std::ios::binary | std::ios::in);
    if (f.is_open()) {
        size_t sizeFile;
        f.seekg(0, std::fstream::end);
        size_t size = sizeFile = (size_t)f.tellg();
        dst.resize(size);
        f.seekg(0, std::fstream::beg);
        f.read(dst.data(), size);
        f.close();
    }
}

HiprtAggregate::Module HiprtAggregate::compileHiprtModule(hiprtContext context) {
    const std::string path = "../src/pbrt/gpu/hiprt/hiprt.cu";
    const std::string closestFunction = "__raygen__findClosest";
    const std::string shadowFunction = "__raygen__shadow";
    const std::string shadowTrFunction = "__raygen__shadow_Tr";
    const std::string oneRandomFunction = "__raygen__randomHit";
    const std::string binFilename = "hiprt.hipfb";

    std::vector<char> binary;
    loadFile(binFilename, binary);

    Module module;
    CUDA_CHECK(hipModuleLoadData(&module.hipModule, binary.data()));

    HIPRT_CHECK(hiprtCreateFuncTable(context, 3, 1, module.funcTable));

    CUDA_CHECK(hipModuleGetFunction(&module.closestFunction, module.hipModule,
                                    closestFunction.c_str()));
    CUDA_CHECK(hipModuleGetFunction(&module.shadowFunction, module.hipModule,
                                    shadowFunction.c_str()));
    CUDA_CHECK(hipModuleGetFunction(&module.shadowTrFunction, module.hipModule,
                                    shadowTrFunction.c_str()));
    CUDA_CHECK(hipModuleGetFunction(&module.oneRandomFunction, module.hipModule,
                                    oneRandomFunction.c_str()));

    return module;
}

void HiprtAggregate::hiprtLaunch(hipFunction_t func, int nx, int ny, int tx, int ty,
                                 void **args, hipStream_t cudaStream,
                                 size_t sharedMemoryBytes) {
    int3 tpb = {tx, ty, 1};
    int3 nb;
    nb.x = (nx + tpb.x - 1) / tpb.x;
    nb.y = (ny + tpb.y - 1) / tpb.y;
    CUDA_CHECK(hipModuleLaunchKernel(func, nb.x, nb.y, 1, tpb.x, tpb.y, 1,
                                     sharedMemoryBytes, cudaStream, args, 0));
}

}  // namespace pbrt
