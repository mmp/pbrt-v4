// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/optix/aggregate.h>

#include <pbrt/gpu/optix/optix.h>
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
#include <mutex>
#include <unordered_map>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#ifdef NVTX
#ifdef UNICODE
#undef UNICODE
#endif  // UNICODE
#include <nvtx3/nvToolsExt.h>
#ifdef RGB
#undef RGB
#endif  // RGB
#endif

#define OPTIX_CHECK(EXPR)                                                           \
    do {                                                                            \
        OptixResult res = EXPR;                                                     \
        if (res != OPTIX_SUCCESS)                                                   \
            LOG_FATAL("OptiX call " #EXPR " failed with code %d: \"%s\"", int(res), \
                      optixGetErrorString(res));                                    \
    } while (false) /* eat semicolon */

#define OPTIX_CHECK_WITH_LOG(EXPR, LOG)                                             \
    do {                                                                            \
        OptixResult res = EXPR;                                                     \
        if (res != OPTIX_SUCCESS)                                                   \
            LOG_FATAL("OptiX call " #EXPR " failed with code %d: \"%s\"\nLogs: %s", \
                      int(res), optixGetErrorString(res), LOG);                     \
    } while (false) /* eat semicolon */

namespace pbrt {

OptiXAggregate::BVH::BVH(size_t size) {
    intersectHGRecords.resize(size);
    shadowHGRecords.resize(size);
    randomHitHGRecords.resize(size);
}

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) OptiXAggregate::HitgroupRecord {
    HitgroupRecord() {}
    HitgroupRecord(const HitgroupRecord &r) { memcpy(this, &r, sizeof(HitgroupRecord)); }
    HitgroupRecord &operator=(const HitgroupRecord &r) {
        if (this != &r)
            memcpy(this, &r, sizeof(HitgroupRecord));
        return *this;
    }

    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    union {
        TriangleMeshRecord triRec;
        BilinearMeshRecord bilinearRec;
        QuadricRecord quadricRec;
    };
};

extern "C" {
extern const unsigned char PBRT_EMBEDDED_PTX[];
}

template <typename T>
static CUdeviceptr CopyToDevice(pstd::span<const T> buffer) {
    void *ptr;
    size_t size = buffer.size() * sizeof(buffer[0]);
    CUDA_CHECK(cudaMalloc(&ptr, size));
    CUDA_CHECK(cudaMemcpy(ptr, buffer.data(), size, cudaMemcpyHostToDevice));
    return CUdeviceptr(ptr);
}

STAT_MEMORY_COUNTER("Memory/Acceleration structures", gpuBVHBytes);

OptixTraversableHandle OptiXAggregate::buildOptixBVH(
    OptixDeviceContext optixContext, const std::vector<OptixBuildInput> &buildInputs,
    ThreadLocal<cudaStream_t> &threadCUDAStreams,
    std::vector<CUdeviceptr> &bvhBuffers) {
    if (buildInputs.empty())
        return {};

    // Figure out memory requirements.
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions,
                                             buildInputs.data(), buildInputs.size(),
                                             &blasBufferSizes));

    uint64_t *compactedSizePtr;
    CUDA_CHECK(cudaMalloc(&compactedSizePtr, sizeof(uint64_t)));
    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = (CUdeviceptr)compactedSizePtr;

    // Allocate buffers.
    void *tempBuffer;
    CUDA_CHECK(cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes));
    void *outputBuffer;
    CUDA_CHECK(cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes));

    // Build.
    cudaStream_t buildStream = threadCUDAStreams.Get();
    OptixTraversableHandle traversableHandle{0};
    OPTIX_CHECK(optixAccelBuild(
        optixContext, buildStream, &accelOptions, buildInputs.data(), buildInputs.size(),
        CUdeviceptr(tempBuffer), blasBufferSizes.tempSizeInBytes,
        CUdeviceptr(outputBuffer), blasBufferSizes.outputSizeInBytes, &traversableHandle,
        &emitDesc, 1));

    CUDA_CHECK(cudaFree(tempBuffer));

    CUDA_CHECK(cudaStreamSynchronize(buildStream));
    uint64_t compactedSize;
    CUDA_CHECK(cudaMemcpyAsync(&compactedSize, compactedSizePtr, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, buildStream));
    CUDA_CHECK(cudaStreamSynchronize(buildStream));

    if (compactedSize >= blasBufferSizes.outputSizeInBytes) {
        // No need to compact...
        gpuBVHBytes += blasBufferSizes.outputSizeInBytes;

        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        bvhBuffers.push_back((CUdeviceptr)outputBuffer);
    } else {
        // Compact the acceleration structure
        gpuBVHBytes += compactedSize;

        void *asBuffer;
        CUDA_CHECK(cudaMalloc(&asBuffer, compactedSize));

        OPTIX_CHECK(optixAccelCompact(optixContext, buildStream, traversableHandle,
                                      CUdeviceptr(asBuffer), compactedSize,
                                      &traversableHandle));
        CUDA_CHECK(cudaStreamSynchronize(buildStream));

        CUDA_CHECK(cudaFree(outputBuffer));

        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        bvhBuffers.push_back((CUdeviceptr)asBuffer);
    }

    CUDA_CHECK(cudaFree(compactedSizePtr));
    return traversableHandle;
}

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

static int getOptixGeometryFlags(bool isTriangle, FloatTexture alphaTexture) {
    if (alphaTexture && isTriangle)
        // Need anyhit
        return OPTIX_GEOMETRY_FLAG_NONE;
    else
        return OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
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

STAT_COUNTER("Geometry/Triangles added from displacement mapping", displacedTrisDelta);

std::map<int, TriQuadMesh> OptiXAggregate::PreparePLYMeshes(
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
                shape.parameters.GetOneFloat("edgelength", 1.f);
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
                        CUDA_CHECK(cudaMallocManaged(&p, nVertices * sizeof(Point3f)));
                        CUDA_CHECK(cudaMallocManaged(&n, nVertices * sizeof(Normal3f)));
                        CUDA_CHECK(cudaMallocManaged(&uv, nVertices * sizeof(Point2f)));

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

                        CUDA_CHECK(cudaFree(p));
                        CUDA_CHECK(cudaFree(n));
                        CUDA_CHECK(cudaFree(uv));
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

OptiXAggregate::BVH OptiXAggregate::buildBVHForTriangles(
    const std::vector<ShapeSceneEntity> &shapes,
    const std::map<int, TriQuadMesh> &plyMeshes, OptixDeviceContext optixContext,
    const OptixProgramGroup &intersectPG, const OptixProgramGroup &shadowPG,
    const OptixProgramGroup &randomHitPG,
    const std::map<std::string, FloatTexture> &floatTextures,
    const std::map<std::string, Material> &namedMaterials,
    const std::vector<Material> &materials, const std::map<std::string, Medium> &media,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    ThreadLocal<Allocator> &threadAllocators,
    ThreadLocal<cudaStream_t> &threadCUDAStreams,
    std::vector<CUdeviceptr> &bvhBuffers) {
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

    std::vector<TriangleMesh *> meshes(nMeshes, nullptr);
    std::vector<Bounds3f> meshBounds(nMeshes);
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

        Bounds3f bounds;
        for (size_t i = 0; i < mesh->nVertices; ++i)
            bounds = Union(bounds, mesh->p[i]);

        meshes[meshIndex] = mesh;
        meshBounds[meshIndex] = bounds;
    });

    BVH bvh(nMeshes);
    std::vector<OptixBuildInput> optixBuildInputs(nMeshes);
    std::vector<CUdeviceptr> pDeviceDevicePtrs(nMeshes);
    std::vector<uint32_t> triangleInputFlags(nMeshes);

    std::mutex boundsMutex;
    ParallelFor(0, nMeshes, [&](int64_t startIndex, int64_t endIndex) {
        Allocator alloc = threadAllocators.Get();
        Bounds3f localBounds;

        for (int meshIndex = startIndex; meshIndex < endIndex; ++meshIndex) {
            int shapeIndex = meshIndexToShapeIndex[meshIndex];
            TriangleMesh *mesh = meshes[meshIndex];
            const auto &shape = shapes[shapeIndex];

            OptixBuildInput &input = optixBuildInputs[meshIndex];

            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.numVertices = mesh->nVertices;
#ifdef PBRT_FLOAT_AS_DOUBLE
            // Convert the vertex positions to 32-bit floats before giving
            // them to OptiX, since it doesn't support double-precision
            // geometry.
            input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
            float *pGPU;
            std::vector<float> p32(3 * mesh->nVertices);
            for (int i = 0; i < mesh->nVertices; i++) {
                p32[3*i] = mesh->p[i].x;
                p32[3*i+1] = mesh->p[i].y;
                p32[3*i+2] = mesh->p[i].z;
            }
            CUDA_CHECK(cudaMalloc(&pGPU, mesh->nVertices * 3 * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(pGPU, p32.data(), mesh->nVertices * 3 *  sizeof(float),
                                  cudaMemcpyHostToDevice));
#else
            input.triangleArray.vertexStrideInBytes = sizeof(Point3f);
            Point3f *pGPU;
            CUDA_CHECK(cudaMalloc(&pGPU, mesh->nVertices * sizeof(Point3f)));
            CUDA_CHECK(cudaMemcpy(pGPU, mesh->p, mesh->nVertices * sizeof(Point3f),
                                  cudaMemcpyHostToDevice));
#endif
            pDeviceDevicePtrs[meshIndex] = CUdeviceptr(pGPU);
            input.triangleArray.vertexBuffers = &pDeviceDevicePtrs[meshIndex];

            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes = 3 * sizeof(int);
            input.triangleArray.numIndexTriplets = mesh->nTriangles;
            int *indicesGPU;
            CUDA_CHECK(cudaMalloc(&indicesGPU, mesh->nTriangles * 3 * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(indicesGPU, mesh->vertexIndices, mesh->nTriangles * 3 * sizeof(int),
                                  cudaMemcpyHostToDevice));
            input.triangleArray.indexBuffer = CUdeviceptr(indicesGPU);

            FloatTexture alphaTexture = getAlphaTexture(shape, floatTextures, alloc);
            Material material = getMaterial(shape, namedMaterials, materials);
            triangleInputFlags[meshIndex] = getOptixGeometryFlags(true, alphaTexture);
            input.triangleArray.flags = &triangleInputFlags[meshIndex];

            // Do this here, after the alpha texture has been consumed.
            shape.parameters.ReportUnused();

            input.triangleArray.numSbtRecords = 1;
            input.triangleArray.sbtIndexOffsetBuffer = CUdeviceptr(nullptr);
            input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            HitgroupRecord hgRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
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

            bvh.intersectHGRecords[meshIndex] = hgRecord;

            OPTIX_CHECK(optixSbtRecordPackHeader(randomHitPG, &hgRecord));
            bvh.randomHitHGRecords[meshIndex] = hgRecord;

            OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
            bvh.shadowHGRecords[meshIndex] = hgRecord;

            localBounds = Union(localBounds, meshBounds[meshIndex]);
        }

        std::lock_guard<std::mutex> lock(boundsMutex);
        bvh.bounds = Union(bvh.bounds, localBounds);
    });

    bvh.traversableHandle =
        buildOptixBVH(optixContext, optixBuildInputs, threadCUDAStreams, bvhBuffers);

    return bvh;
}

STAT_COUNTER("Geometry/Curves", nCurves);
STAT_COUNTER("Geometry/Bilinear patches created for diced curves", nBLPsForCurves);

BilinearPatchMesh *OptiXAggregate::diceCurveToBLP(const ShapeSceneEntity &shape,
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

OptiXAggregate::BVH OptiXAggregate::buildBVHForBLPs(
    const std::vector<ShapeSceneEntity> &shapes, OptixDeviceContext optixContext,
    const OptixProgramGroup &intersectPG, const OptixProgramGroup &shadowPG,
    const OptixProgramGroup &randomHitPG,
    const std::map<std::string, FloatTexture> &floatTextures,
    const std::map<std::string, Material> &namedMaterials,
    const std::vector<Material> &materials, const std::map<std::string, Medium> &media,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    ThreadLocal<Allocator> &threadAllocators,
    ThreadLocal<cudaStream_t> &threadCUDAStreams,
    std::vector<CUdeviceptr> &bvhBuffers) {
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

    // Create meshes
    std::vector<BilinearPatchMesh *> meshes(nMeshes, nullptr);
    std::atomic<int> nPatches = 0;

    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        Allocator alloc = threadAllocators.Get();
        size_t shapeIndex = meshIndexToShapeIndex[meshIndex];
        const auto &shape = shapes[shapeIndex];

        if (shape.name == "bilinearmesh") {
            BilinearPatchMesh *mesh = BilinearPatch::CreateMesh(
                shape.renderFromObject, shape.reverseOrientation, shape.parameters,
                &shape.loc, alloc);
            meshes[meshIndex] = mesh;
            nPatches += mesh->nPatches;
        } else if (shape.name == "curve") {
            BilinearPatchMesh *curveMesh =
                diceCurveToBLP(shape, 5 /* nseg */, 5 /* nvert */, alloc);
            if (curveMesh) {
                meshes[meshIndex] = curveMesh;
                nPatches += curveMesh->nPatches;
            }
        }
    });

    // Figure out where each mesh starts using entries of the aabb array
    std::vector<int> meshAABBStartIndex(nMeshes);
    int aabbIndex = 0;
    for (size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
        meshAABBStartIndex[meshIndex] = aabbIndex;
        aabbIndex += meshes[meshIndex]->nPatches;
    }

    // Create build inputs
    BVH bvh(nMeshes);
    int buildInputIndex = 0;
    std::vector<OptixBuildInput> optixBuildInputs(nMeshes);
    std::vector<OptixAabb> aabbs(nPatches);
    OptixAabb *deviceAABBs;
    CUDA_CHECK(cudaMalloc(&deviceAABBs, sizeof(OptixAabb) * nPatches));
    std::vector<CUdeviceptr> aabbDevicePtrs(nMeshes);
    std::vector<uint32_t> flags(nMeshes);

    std::mutex boundsMutex;
    ParallelFor(0, nMeshes, [&](int64_t meshIndex) {
        Allocator alloc = threadAllocators.Get();
        BilinearPatchMesh *mesh = meshes[meshIndex];

        OptixBuildInput &input = optixBuildInputs[meshIndex];
        input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.numSbtRecords = 1;
        input.customPrimitiveArray.numPrimitives = mesh->nPatches;
        int aabbIndex = meshAABBStartIndex[meshIndex];
        aabbDevicePtrs[meshIndex] = CUdeviceptr(&deviceAABBs[aabbIndex]);
        input.customPrimitiveArray.aabbBuffers = &aabbDevicePtrs[meshIndex];
        input.customPrimitiveArray.flags = &flags[meshIndex];

        Bounds3f meshBounds;
        for (int patchIndex = 0; patchIndex < mesh->nPatches; ++patchIndex) {
            Bounds3f patchBounds;
            for (int i = 0; i < 4; ++i)
                patchBounds =
                    Union(patchBounds, mesh->p[mesh->vertexIndices[4 * patchIndex + i]]);

            OptixAabb aabb = {float(patchBounds.pMin.x), float(patchBounds.pMin.y),
                              float(patchBounds.pMin.z), float(patchBounds.pMax.x),
                              float(patchBounds.pMax.y), float(patchBounds.pMax.z)};
            aabbs[aabbIndex++] = aabb;

            meshBounds = Union(meshBounds, patchBounds);
        }

        size_t shapeIndex = meshIndexToShapeIndex[meshIndex];
        const auto &shape = shapes[shapeIndex];
        Material material = getMaterial(shape, namedMaterials, materials);
        FloatTexture alphaTexture = getAlphaTexture(shape, floatTextures, alloc);

        // After "alpha" has been consumed...
        shape.parameters.ReportUnused();

        flags[meshIndex] = getOptixGeometryFlags(false, alphaTexture);

        HitgroupRecord hgRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
        hgRecord.bilinearRec.mesh = mesh;
        hgRecord.bilinearRec.material = material;
        hgRecord.bilinearRec.alphaTexture = alphaTexture;
        hgRecord.bilinearRec.areaLights = {};
        if (shape.lightIndex != -1) {
            if (!material)
                Warning(&shape.loc, "Ignoring area light specification for shape with "
                                    "\"interface\" material.");
            else {
                auto iter = shapeIndexToAreaLights.find(shapeIndex);
                // Note: this will hit if we try to have an instance as an area
                // light.
                CHECK(iter != shapeIndexToAreaLights.end());
                CHECK_EQ(iter->second->size(), mesh->nPatches);
                hgRecord.bilinearRec.areaLights = pstd::MakeSpan(*iter->second);
            }
        }
        hgRecord.bilinearRec.mediumInterface = getMediumInterface(shape, media, alloc);

        bvh.intersectHGRecords[meshIndex] = hgRecord;

        OPTIX_CHECK(optixSbtRecordPackHeader(randomHitPG, &hgRecord));
        bvh.randomHitHGRecords[meshIndex] = hgRecord;

        OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
        bvh.shadowHGRecords[meshIndex] = hgRecord;

        std::lock_guard<std::mutex> lock(boundsMutex);
        bvh.bounds = Union(bvh.bounds, meshBounds);
    });

    CUDA_CHECK(cudaMemcpyAsync(deviceAABBs, aabbs.data(),
                               aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice,
                               threadCUDAStreams.Get()));

    bvh.traversableHandle =
        buildOptixBVH(optixContext, optixBuildInputs, threadCUDAStreams, bvhBuffers);

    CUDA_CHECK(cudaFree(deviceAABBs));

    return bvh;
}

OptiXAggregate::BVH OptiXAggregate::buildBVHForQuadrics(
    const std::vector<ShapeSceneEntity> &shapes, OptixDeviceContext optixContext,
    const OptixProgramGroup &intersectPG, const OptixProgramGroup &shadowPG,
    const OptixProgramGroup &randomHitPG,
    const std::map<std::string, FloatTexture> &floatTextures,
    const std::map<std::string, Material> &namedMaterials,
    const std::vector<Material> &materials, const std::map<std::string, Medium> &media,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    ThreadLocal<Allocator> &threadAllocators,
    ThreadLocal<cudaStream_t> &threadCUDAStreams,
    std::vector<CUdeviceptr> &bvhBuffers) {
    int nQuadrics = 0;
    for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
        const auto &s = shapes[shapeIndex];
        if (s.name == "sphere" || s.name == "cylinder" || s.name == "disk")
            ++nQuadrics;
    }

    if (nQuadrics == 0)
        return {};

    Allocator alloc = threadAllocators.Get();
    BVH bvh(nQuadrics);
    std::vector<OptixBuildInput> optixBuildInputs(nQuadrics);
    OptixAabb *deviceShapeAABBs;
    CUDA_CHECK(cudaMalloc(&deviceShapeAABBs, sizeof(OptixAabb) * nQuadrics));
    std::vector<OptixAabb> shapeAABBs(nQuadrics);
    std::vector<CUdeviceptr> aabbDevicePtrs(nQuadrics);
    std::vector<uint32_t> flags(nQuadrics);

    int quadricIndex = 0;
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

        OptixBuildInput &input = optixBuildInputs[quadricIndex];
        input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.numSbtRecords = 1;
        input.customPrimitiveArray.numPrimitives = 1;
        input.customPrimitiveArray.flags = &flags[quadricIndex];

        Bounds3f shapeBounds = shape.Bounds();
        OptixAabb aabb = {float(shapeBounds.pMin.x), float(shapeBounds.pMin.y),
                          float(shapeBounds.pMin.z), float(shapeBounds.pMax.x),
                          float(shapeBounds.pMax.y), float(shapeBounds.pMax.z)};
        shapeAABBs[quadricIndex] = aabb;
        aabbDevicePtrs[quadricIndex] = CUdeviceptr(&deviceShapeAABBs[quadricIndex]);
        input.customPrimitiveArray.aabbBuffers = &aabbDevicePtrs[quadricIndex];

        bvh.bounds = Union(bvh.bounds, shapeBounds);

        // Find alpha texture, if present.
        Material material = getMaterial(s, namedMaterials, materials);
        FloatTexture alphaTexture = getAlphaTexture(s, floatTextures, alloc);
        flags[quadricIndex] = getOptixGeometryFlags(false, alphaTexture);

        // Once again, after any alpha texture is created...
        s.parameters.ReportUnused();

        HitgroupRecord hgRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
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

        bvh.intersectHGRecords[quadricIndex] = hgRecord;

        OPTIX_CHECK(optixSbtRecordPackHeader(randomHitPG, &hgRecord));
        bvh.randomHitHGRecords[quadricIndex] = hgRecord;

        OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
        bvh.shadowHGRecords[quadricIndex] = hgRecord;

        ++quadricIndex;
    }

    CUDA_CHECK(cudaMemcpyAsync(deviceShapeAABBs, shapeAABBs.data(),
                               shapeAABBs.size() * sizeof(shapeAABBs[0]),
                               cudaMemcpyHostToDevice, threadCUDAStreams.Get()));

    bvh.traversableHandle =
        buildOptixBVH(optixContext, optixBuildInputs, threadCUDAStreams, bvhBuffers);

    CUDA_CHECK(cudaFree(deviceShapeAABBs));

    return bvh;
}

static void logCallback(unsigned int level, const char *tag, const char *message,
                        void *cbdata) {
    if (level <= 2)
        LOG_ERROR("OptiX: %s: %s", tag, message);
    else
        LOG_VERBOSE("OptiX: %s: %s", tag, message);
}

int OptiXAggregate::addHGRecords(const BVH &bvh) {
    if (bvh.intersectHGRecords.empty())
        return -1;

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    int sbtOffset = intersectHGRecords.size();
    intersectHGRecords.insert(intersectHGRecords.end(), bvh.intersectHGRecords.begin(),
                              bvh.intersectHGRecords.end());
    shadowHGRecords.insert(shadowHGRecords.end(), bvh.shadowHGRecords.begin(),
                           bvh.shadowHGRecords.end());
    randomHitHGRecords.insert(randomHitHGRecords.end(), bvh.randomHitHGRecords.begin(),
                              bvh.randomHitHGRecords.end());
    return sbtOffset;
}

OptixPipelineCompileOptions OptiXAggregate::getPipelineCompileOptions() {
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 4;
    // OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.exceptionFlags =
        (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH);
#if (OPTIX_VERSION < 80000)
    // This flag is removed since OptiX 8.0.0
    pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_DEBUG;
#endif
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    return pipelineCompileOptions;
}

OptixModule OptiXAggregate::createOptiXModule(OptixDeviceContext optixContext,
                                              const char *ptx) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    // TODO: REVIEW THIS
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#if (OPTIX_VERSION >= 70400)
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
#else
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif  // OPTIX_VERSION
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    // Workaround driver 510/511 bug with debug builds.
    // (See https://github.com/mmp/pbrt-v4/issues/226).
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipelineCompileOptions = getPipelineCompileOptions();

    char log[4096];
    size_t logSize = sizeof(log);
    OptixModule optixModule;

#if (OPTIX_VERSION >= 70700)
#define OPTIX_MODULE_CREATE_FN optixModuleCreate
#else
#define OPTIX_MODULE_CREATE_FN optixModuleCreateFromPTX
#endif

    OPTIX_CHECK_WITH_LOG(
        OPTIX_MODULE_CREATE_FN(
            optixContext, &moduleCompileOptions, &pipelineCompileOptions,
            ptx, strlen(ptx), log, &logSize, &optixModule
        ),
        log
    );

    LOG_VERBOSE("%s", log);

    return optixModule;
}

OptixProgramGroup OptiXAggregate::createRaygenPG(const char *entrypoint) const {
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = optixModule;
    desc.raygen.entryFunctionName = entrypoint;

    char log[4096];
    size_t logSize = sizeof(log);
    OptixProgramGroup pg;
    OPTIX_CHECK_WITH_LOG(
        optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions, log, &logSize, &pg),
        log);
    LOG_VERBOSE("%s", log);

    return pg;
}

OptixProgramGroup OptiXAggregate::createMissPG(const char *entrypoint) const {
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = optixModule;
    desc.miss.entryFunctionName = entrypoint;

    char log[4096];
    size_t logSize = sizeof(log);
    OptixProgramGroup pg;
    OPTIX_CHECK_WITH_LOG(
        optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions, log, &logSize, &pg),
        log);
    LOG_VERBOSE("%s", log);

    return pg;
}

OptixProgramGroup OptiXAggregate::createIntersectionPG(const char *closest,
                                                       const char *any,
                                                       const char *intersect) const {
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    if (closest) {
        desc.hitgroup.moduleCH = optixModule;
        desc.hitgroup.entryFunctionNameCH = closest;
    }
    if (any) {
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = any;
    }
    if (intersect) {
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = intersect;
    }

    char log[4096];
    size_t logSize = sizeof(log);
    OptixProgramGroup pg;
    OPTIX_CHECK_WITH_LOG(
        optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions, log, &logSize, &pg),
        log);
    LOG_VERBOSE("%s", log);

    return pg;
}

OptiXAggregate::OptiXAggregate(
    const BasicScene &scene, CUDATrackedMemoryResource *memoryResource,
    NamedTextures &textures,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    const std::map<std::string, Medium> &media,
    const std::map<std::string, pbrt::Material> &namedMaterials,
    const std::vector<pbrt::Material> &materials)
    : memoryResource(memoryResource), cudaStream(nullptr) {
    CUcontext cudaContext;
    CU_CHECK(cuCtxGetCurrent(&cudaContext));
    CHECK(cudaContext != nullptr);

#ifdef PBRT_IS_WINDOWS
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

    ThreadLocal<cudaStream_t> threadCUDAStreams([]() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    });

    paramsPool.resize(256);  // should be plenty
    for (ParamBufferState &ps : paramsPool) {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(RayIntersectParameters)));
        ps.ptr = (CUdeviceptr)ptr;
        CUDA_CHECK(cudaEventCreate(&ps.finishedEvent));
        CUDA_CHECK(cudaMallocHost(&ps.hostPtr, sizeof(RayIntersectParameters)));
    }

    // Create OptiX context
    LOG_VERBOSE("Starting OptiX initialization");
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions ctxOptions = {};
#ifndef NDEBUG
    ctxOptions.logCallbackLevel = 4;  // status/progress
#else
    ctxOptions.logCallbackLevel = 2;  // error
#endif
    ctxOptions.logCallbackFunction = logCallback;
#if (OPTIX_VERSION >= 70200) && !defined(NDEBUG)
    ctxOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &ctxOptions, &optixContext));

    LOG_VERBOSE("Optix version %d.%d.%d successfully initialized", OPTIX_VERSION / 10000,
                (OPTIX_VERSION % 10000) / 100, OPTIX_VERSION % 100);

    // OptiX module
    optixModule = createOptiXModule(optixContext, (const char *)PBRT_EMBEDDED_PTX);

    // Optix program groups...
    char log[4096];
    size_t logSize = sizeof(log);

    OptixProgramGroup raygenPGClosest = createRaygenPG("__raygen__findClosest");
    OptixProgramGroup missPGNoOp = createMissPG("__miss__noop");
    OptixProgramGroup hitPGTriangle =
        createIntersectionPG("__closesthit__triangle", "__anyhit__triangle", nullptr);
    OptixProgramGroup hitPGBilinearPatch = createIntersectionPG(
        "__closesthit__bilinearPatch", nullptr, "__intersection__bilinearPatch");
    OptixProgramGroup hitPGQuadric =
        createIntersectionPG("__closesthit__quadric", nullptr, "__intersection__quadric");

    OptixProgramGroup raygenPGShadow = createRaygenPG("__raygen__shadow");
    OptixProgramGroup missPGShadow = createMissPG("__miss__shadow");
    OptixProgramGroup anyhitPGShadowTriangle =
        createIntersectionPG(nullptr, "__anyhit__shadowTriangle", nullptr);

    OptixProgramGroup raygenPGShadowTr = createRaygenPG("__raygen__shadow_Tr");
    OptixProgramGroup missPGShadowTr = createMissPG("__miss__shadow_Tr");

    OptixProgramGroup anyhitPGShadowBilinearPatch = createIntersectionPG(
        nullptr, "__anyhit__shadowBilinearPatch", "__intersection__bilinearPatch");
    OptixProgramGroup anyhitPGShadowQuadric = createIntersectionPG(
        nullptr, "__anyhit__shadowQuadric", "__intersection__quadric");

    OptixProgramGroup raygenPGRandomHit = createRaygenPG("__raygen__randomHit");
    OptixProgramGroup hitPGRandomHitTriangle =
        createIntersectionPG("__closesthit__randomHitTriangle", nullptr, nullptr);
    OptixProgramGroup hitPGRandomHitBilinearPatch = createIntersectionPG(
        "__closesthit__randomHitBilinearPatch", nullptr, "__intersection__bilinearPatch");
    OptixProgramGroup hitPGRandomHitQuadric = createIntersectionPG(
        "__closesthit__randomHitQuadric", nullptr, "__intersection__quadric");

    // Optix pipeline...
    OptixProgramGroup allPGs[] = {raygenPGClosest,
                                  missPGNoOp,
                                  hitPGTriangle,
                                  hitPGBilinearPatch,
                                  hitPGQuadric,
                                  raygenPGShadow,
                                  missPGShadow,
                                  anyhitPGShadowTriangle,
                                  anyhitPGShadowBilinearPatch,
                                  anyhitPGShadowQuadric,
                                  raygenPGShadowTr,
                                  missPGShadowTr,
                                  raygenPGRandomHit,
                                  hitPGRandomHitTriangle,
                                  hitPGRandomHitBilinearPatch,
                                  hitPGRandomHitQuadric};

    programGroups.assign(std::begin(allPGs), std::end(allPGs));

    OptixPipelineCompileOptions pipelineCompileOptions = getPipelineCompileOptions();

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 2;
#if (OPTIX_VERSION < 70700)
#ifndef NDEBUG
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
#endif // OPTIX_VERSION

    OPTIX_CHECK_WITH_LOG(
        optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions,
                            allPGs, sizeof(allPGs) / sizeof(allPGs[0]), log, &logSize,
                            &optixPipeline),
        log);
    LOG_VERBOSE("%s", log);

#if 0
    OPTIX_CHECK(optixPipelineSetStackSize(
        optixPipeline,
        0, /* direct callables from intersect or any-hit */
        0, /* direct callables from raygen, miss, or closest hit */
        4 * 1024, /* continuation stack */
        2 /* max graph depth. NOTE: this is 3 when we have motion xforms... */));
#endif

    // Shader binding tables...
    // Hitgroups are done as meshes are processed

    // Closest intersection
    Allocator alloc(memoryResource);
    RaygenRecord *raygenClosestRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGClosest, raygenClosestRecord));
    intersectSBT.raygenRecord = (CUdeviceptr)raygenClosestRecord;

    MissRecord *missNoOpRecord = alloc.new_object<MissRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGNoOp, missNoOpRecord));
    intersectSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    intersectSBT.missRecordStrideInBytes = sizeof(MissRecord);
    intersectSBT.missRecordCount = 1;

    // Shadow
    RaygenRecord *raygenShadowRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGShadow, raygenShadowRecord));
    shadowSBT.raygenRecord = (CUdeviceptr)raygenShadowRecord;

    MissRecord *missShadowRecord = alloc.new_object<MissRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGShadow, missShadowRecord));
    shadowSBT.missRecordBase = (CUdeviceptr)missShadowRecord;
    shadowSBT.missRecordStrideInBytes = sizeof(MissRecord);
    shadowSBT.missRecordCount = 1;

    // Shadow + Tr
    RaygenRecord *raygenShadowTrRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGShadowTr, raygenShadowTrRecord));
    shadowTrSBT.raygenRecord = (CUdeviceptr)raygenShadowTrRecord;

    MissRecord *missShadowTrRecord = alloc.new_object<MissRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGShadowTr, missShadowTrRecord));
    shadowTrSBT.missRecordBase = (CUdeviceptr)missShadowTrRecord;
    shadowTrSBT.missRecordStrideInBytes = sizeof(MissRecord);
    shadowTrSBT.missRecordCount = 1;

    // Random hit
    RaygenRecord *raygenRandomHitRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGRandomHit, raygenRandomHitRecord));
    randomHitSBT.raygenRecord = (CUdeviceptr)raygenRandomHitRecord;
    randomHitSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    randomHitSBT.missRecordStrideInBytes = sizeof(MissRecord);
    randomHitSBT.missRecordCount = 1;

    LOG_VERBOSE("Finished OptiX initialization");

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

    struct GAS {
        BVH bvh;
        int sbtOffset;
    };
    AsyncJob<GAS> *triJob = RunAsync([&]() {
        BVH triangleBVH = buildBVHForTriangles(
            scene.shapes, plyMeshes, optixContext, hitPGTriangle, anyhitPGShadowTriangle,
            hitPGRandomHitTriangle, textures.floatTextures, namedMaterials, materials,
            media, shapeIndexToAreaLights, threadAllocators, threadCUDAStreams, bvhBuffers);
        int sbtOffset = addHGRecords(triangleBVH);
        return GAS{triangleBVH, sbtOffset};
    });

    AsyncJob<GAS> *blpJob = RunAsync([&]() {
        BVH blpBVH =
            buildBVHForBLPs(scene.shapes, optixContext, hitPGBilinearPatch,
                            anyhitPGShadowBilinearPatch, hitPGRandomHitBilinearPatch,
                            textures.floatTextures, namedMaterials, materials, media,
                            shapeIndexToAreaLights, threadAllocators, threadCUDAStreams, bvhBuffers);
        int bilinearSBTOffset = addHGRecords(blpBVH);
        return GAS{blpBVH, bilinearSBTOffset};
    });

    AsyncJob<GAS> *quadricJob = RunAsync([&]() {
        BVH quadricBVH = buildBVHForQuadrics(
            scene.shapes, optixContext, hitPGQuadric, anyhitPGShadowQuadric,
            hitPGRandomHitQuadric, textures.floatTextures, namedMaterials, materials,
            media, shapeIndexToAreaLights, threadAllocators, threadCUDAStreams, bvhBuffers);
        int quadricSBTOffset = addHGRecords(quadricBVH);
        return GAS{quadricBVH, quadricSBTOffset};
    });

    ///////////////////////////////////////////////////////////////////////////
    // Create IASes for instance definitions
    // TODO: better name here...
    struct Instance {
        OptixTraversableHandle handles[3] = {};
        int sbtOffsets[3] = {-1, -1, -1};
        Bounds3f bounds;

        int NumValidHandles() const {
            return (handles[0] ? 1 : 0) + (handles[1] ? 1 : 0) + (handles[2] ? 1 : 0);
        }
    };

    LOG_VERBOSE("Starting to create IASes for %d instance definitions",
                scene.instanceDefinitions.size());
    std::vector<InternedString> allInstanceNames;
    for (const auto &def : scene.instanceDefinitions)
        allInstanceNames.push_back(def.first);

    std::unordered_map<InternedString, Instance, InternedStringHash> instanceMap;
    std::mutex instanceMapMutex;
    ParallelFor(0, scene.instanceDefinitions.size(), [&](int64_t i) {
        InternedString name = allInstanceNames[i];
        auto iter = scene.instanceDefinitions.find(name);
        CHECK(iter != scene.instanceDefinitions.end());
        const auto &def = *iter;

        if (!def.second->animatedShapes.empty())
            Warning("Ignoring %d animated shapes in instance \"%s\".",
                    def.second->animatedShapes.size(), def.first);

        Instance inst;

        std::map<int, TriQuadMesh> meshes =
            PreparePLYMeshes(def.second->shapes, textures.floatTextures);

        BVH triangleBVH = buildBVHForTriangles(
            def.second->shapes, meshes, optixContext, hitPGTriangle,
            anyhitPGShadowTriangle, hitPGRandomHitTriangle, textures.floatTextures,
            namedMaterials, materials, media, {}, threadAllocators, threadCUDAStreams, bvhBuffers);
        meshes.clear();
        if (triangleBVH.traversableHandle) {
            inst.handles[0] = triangleBVH.traversableHandle;
            inst.sbtOffsets[0] = addHGRecords(triangleBVH);
            inst.bounds = triangleBVH.bounds;
        }

        BVH blpBVH =
            buildBVHForBLPs(def.second->shapes, optixContext, hitPGBilinearPatch,
                            anyhitPGShadowBilinearPatch, hitPGRandomHitBilinearPatch,
                            textures.floatTextures, namedMaterials, materials, media, {},
                            threadAllocators, threadCUDAStreams, bvhBuffers);
        if (blpBVH.traversableHandle) {
            inst.handles[1] = blpBVH.traversableHandle;
            inst.sbtOffsets[1] = addHGRecords(blpBVH);
            inst.bounds = Union(inst.bounds, blpBVH.bounds);
        }

        BVH quadricBVH = buildBVHForQuadrics(
            def.second->shapes, optixContext, hitPGQuadric, anyhitPGShadowQuadric,
            hitPGRandomHitQuadric, textures.floatTextures, namedMaterials, materials,
            media, {}, threadAllocators, threadCUDAStreams, bvhBuffers);
        if (quadricBVH.traversableHandle) {
            inst.handles[2] = quadricBVH.traversableHandle;
            inst.sbtOffsets[2] = addHGRecords(quadricBVH);
            inst.bounds = Union(inst.bounds, quadricBVH.bounds);
        }

        std::lock_guard<std::mutex> lock(instanceMapMutex);
        instanceMap[def.first] = inst;
    });

    LOG_VERBOSE("Finished creating IASes for instance definitions");

    ///////////////////////////////////////////////////////////////////////////
    // Create OptixInstances for instances
    LOG_VERBOSE("Starting to create %d OptixInstances", scene.instances.size());

    // Get the appropriate instanceMap iterator for each instance use, just
    // once, and in parallel.  While we're at it, count the total number of
    // OptixInstances that will be added to iasInstances.
    std::vector<
        std::unordered_map<InternedString, Instance, InternedStringHash>::const_iterator>
        instanceMapIters(scene.instances.size());
    std::atomic<int> totalOptixInstances{0};
    std::vector<int> numValidHandles(scene.instances.size());
    ParallelFor(0, scene.instances.size(), [&](int64_t indexBegin, int64_t indexEnd) {
        int localTotalInstances = 0;

        for (int64_t i = indexBegin; i < indexEnd; ++i) {
            const auto &sceneInstance = scene.instances[i];
            auto iter = instanceMap.find(sceneInstance.name);
            instanceMapIters[i] = iter;

            if (iter != instanceMap.end()) {
                int nHandles = iter->second.NumValidHandles();
                localTotalInstances += nHandles;
                numValidHandles[i] = nHandles;
            }
        }

        // Don't hammer the atomic; at least accumulate sums locally for a
        // range of them before we add to it.
        totalOptixInstances += localTotalInstances;
    });

    std::vector<OptixInstance> iasInstances;
    iasInstances.reserve(3 + totalOptixInstances);

    // Consume futures for top-level non-instanced geometry acceleration structures.
    OptixInstance gasInstance = {};
    float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    memcpy(gasInstance.transform, identity, 12 * sizeof(float));
    gasInstance.visibilityMask = 255;
    gasInstance.flags =
        OPTIX_INSTANCE_FLAG_NONE;  // TODO: OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT
    LOG_VERBOSE("Starting to consume top-level GAS futures");
    for (AsyncJob<GAS> *job : {triJob, blpJob, quadricJob}) {
        GAS gas = job->GetResult();
        if (gas.bvh.traversableHandle) {
            gasInstance.traversableHandle = gas.bvh.traversableHandle;
            gasInstance.sbtOffset = gas.sbtOffset;
            iasInstances.push_back(gasInstance);

            bounds = Union(bounds, gas.bvh.bounds);
        }
    }
    LOG_VERBOSE("Finished consuming top-level GAS futures");

    // Resize iasInstances to be just the right size for the OptixInstances
    // to come
    size_t iasInstancesOffset = iasInstances.size();
    iasInstances.resize(iasInstances.size() + totalOptixInstances);

    // Compute the staring offset in iasInstances for each instance use.
    std::vector<size_t> instanceIASOffset(scene.instances.size());
    for (size_t i = 0; i < scene.instances.size(); ++i) {
        instanceIASOffset[i] = iasInstancesOffset;
        iasInstancesOffset += numValidHandles[i];
    }

    // Now loop over all of the instance uses in parallel.
    ParallelFor(0, scene.instances.size(), [&](int64_t indexBegin, int64_t indexEnd) {
        Bounds3f localBounds;

        for (int64_t index = indexBegin; index < indexEnd; ++index) {
            const auto &sceneInstance = scene.instances[index];
            auto iter = instanceMapIters[index];
            if (iter == instanceMap.end())
                ErrorExit(&sceneInstance.loc, "%s: object instance not defined.",
                          sceneInstance.name);

            if (sceneInstance.renderFromInstance == nullptr) {
                Warning(&sceneInstance.loc,
                        "%s: object instance has animated transformation. TODO",
                        sceneInstance.name);
                continue;
            }

            const Instance &in = iter->second;
            if (in.bounds.IsDegenerate())
                // Empty instance. Nothing to do (and don't update bounds!)
                continue;

            localBounds =
                Union(localBounds, (*sceneInstance.renderFromInstance)(in.bounds));

            size_t iasOffset = instanceIASOffset[index];
            for (int i = 0; i < 3; ++i) {
                if (!in.handles[i])
                    continue;

                OptixInstance optixInstance = {};
                SquareMatrix<4> renderFromInstance =
                    sceneInstance.renderFromInstance->GetMatrix();
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 4; ++k)
                        optixInstance.transform[4 * j + k] = renderFromInstance[j][k];
                optixInstance.visibilityMask = 255;
                optixInstance.sbtOffset = in.sbtOffsets[i];
                optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;  // TODO:
                // OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT
                optixInstance.traversableHandle = in.handles[i];
                iasInstances[iasOffset] = optixInstance;
                ++iasOffset;
            }
        }

        // As before with totalOptixInstances, try to limit how many times
        // we update the global bounds so we're not hammering on it.
        std::lock_guard<std::mutex> lock(boundsMutex);
        bounds = Union(bounds, localBounds);
    });
    LOG_VERBOSE("Finished creating OptixInstances");

    ///////////////////////////////////////////////////////////////////////////
    // Build the top-level IAS
    LOG_VERBOSE("Starting to build top-level IAS");
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    CUdeviceptr instanceDevicePtr = CopyToDevice(pstd::MakeConstSpan(iasInstances));
    buildInput.instanceArray.instances = instanceDevicePtr;
    buildInput.instanceArray.numInstances = iasInstances.size();

    rootTraversable = buildOptixBVH(optixContext, {buildInput}, threadCUDAStreams, bvhBuffers);

    CUDA_CHECK(cudaFree((void *)instanceDevicePtr));
    LOG_VERBOSE("Finished building top-level IAS");

    LOG_VERBOSE("Finished creating shapes and acceleration structures");

    if (!scene.animatedShapes.empty())
        Warning("Ignoring %d animated shapes", scene.animatedShapes.size());

    ///////////////////////////////////////////////////////////////////////////
    // Final SBT initialization
    CUdeviceptr isectHGRBDevicePtr =
        CopyToDevice(pstd::MakeConstSpan(intersectHGRecords));
    intersectSBT.hitgroupRecordBase = isectHGRBDevicePtr;
    intersectSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    intersectSBT.hitgroupRecordCount = intersectHGRecords.size();

    CUdeviceptr shadowHGRBDevicePtr = CopyToDevice(pstd::MakeConstSpan(shadowHGRecords));
    shadowSBT.hitgroupRecordBase = shadowHGRBDevicePtr;
    shadowSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    shadowSBT.hitgroupRecordCount = shadowHGRecords.size();

    // Still want to run the closest hit shaders...
    shadowTrSBT.hitgroupRecordBase = isectHGRBDevicePtr;
    shadowTrSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    shadowTrSBT.hitgroupRecordCount = intersectHGRecords.size();

    CUdeviceptr randomHitHGRBDevicePtr =
        CopyToDevice(pstd::MakeConstSpan(randomHitHGRecords));
    randomHitSBT.hitgroupRecordBase = randomHitHGRBDevicePtr;
    randomHitSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    randomHitSBT.hitgroupRecordCount = randomHitHGRecords.size();

#ifdef PBRT_IS_WINDOWS
    if (Options->useGPU)
        ReenableThreadPool();
#endif  // PBRT_IS_WINDOWS
}

OptiXAggregate::~OptiXAggregate() {
    if (intersectSBT.hitgroupRecordBase)
        CUDA_CHECK(cudaFree((void *)intersectSBT.hitgroupRecordBase));
    if (shadowSBT.hitgroupRecordBase)
        CUDA_CHECK(cudaFree((void *)shadowSBT.hitgroupRecordBase));
    if (randomHitSBT.hitgroupRecordBase)
        CUDA_CHECK(cudaFree((void *)randomHitSBT.hitgroupRecordBase));

    Allocator alloc(memoryResource);
    if (intersectSBT.raygenRecord) alloc.delete_object((RaygenRecord*)intersectSBT.raygenRecord);
    if (intersectSBT.missRecordBase) alloc.delete_object((MissRecord*)intersectSBT.missRecordBase);
    
    if (shadowSBT.raygenRecord) alloc.delete_object((RaygenRecord*)shadowSBT.raygenRecord);
    if (shadowSBT.missRecordBase) alloc.delete_object((MissRecord*)shadowSBT.missRecordBase);
    
    if (shadowTrSBT.raygenRecord) alloc.delete_object((RaygenRecord*)shadowTrSBT.raygenRecord);
    if (shadowTrSBT.missRecordBase) alloc.delete_object((MissRecord*)shadowTrSBT.missRecordBase);
    
    if (randomHitSBT.raygenRecord) alloc.delete_object((RaygenRecord*)randomHitSBT.raygenRecord);

    for (CUdeviceptr ptr : bvhBuffers)
        CUDA_CHECK(cudaFree((void *)ptr));

    for (ParamBufferState &ps : paramsPool) {
        if (ps.ptr) CUDA_CHECK(cudaFree((void *)ps.ptr));
        if (ps.hostPtr) CUDA_CHECK(cudaFreeHost(ps.hostPtr));
        if (ps.finishedEvent) CUDA_CHECK(cudaEventDestroy(ps.finishedEvent));
    }

    if (optixPipeline) OPTIX_CHECK(optixPipelineDestroy(optixPipeline));
    for (auto pg : programGroups)
        if (pg) OPTIX_CHECK(optixProgramGroupDestroy(pg));
    if (optixModule) OPTIX_CHECK(optixModuleDestroy(optixModule));
    if (optixContext) OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
}

OptiXAggregate::ParamBufferState &OptiXAggregate::getParamBuffer(
    const RayIntersectParameters &params) const {
    CHECK(nextParamOffset < paramsPool.size());

    ParamBufferState &pbs = paramsPool[nextParamOffset];
    if (++nextParamOffset == paramsPool.size())
        nextParamOffset = 0;
    if (!pbs.used)
        pbs.used = true;
    else
        CUDA_CHECK(cudaEventSynchronize(pbs.finishedEvent));

    // Copy to host-side pinned memory
    memcpy(pbs.hostPtr, &params, sizeof(params));
    CUDA_CHECK(cudaMemcpyAsync((void *)pbs.ptr, pbs.hostPtr, sizeof(params),
                               cudaMemcpyHostToDevice));

    return pbs;
}

void OptiXAggregate::IntersectClosest(int maxRays, const RayQueue *rayQueue,
                                      EscapedRayQueue *escapedRayQueue,
                                      HitAreaLightQueue *hitAreaLightQueue,
                                      MaterialEvalQueue *basicEvalMaterialQueue,
                                      MaterialEvalQueue *universalEvalMaterialQueue,
                                      MediumSampleQueue *mediumSampleQueue,
                                      RayQueue *nextRayQueue) const {
    std::pair<cudaEvent_t, cudaEvent_t> events =
        GetProfilerEvents("Trace closest hit rays");

    cudaEventRecord(events.first);

    if (rootTraversable) {
        RayIntersectParameters params;
        params.traversable = rootTraversable;
        params.rayQueue = rayQueue;
        params.nextRayQueue = nextRayQueue;
        params.escapedRayQueue = escapedRayQueue;
        params.hitAreaLightQueue = hitAreaLightQueue;
        params.basicEvalMaterialQueue = basicEvalMaterialQueue;
        params.universalEvalMaterialQueue = universalEvalMaterialQueue;
        params.mediumSampleQueue = mediumSampleQueue;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect closest");
#endif
#ifdef NVTX
        nvtxRangePush("OptiXAggregate::IntersectClosest");
#endif

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &intersectSBT, maxRays, 1,
                                1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifdef NVTX
        nvtxRangePop();
#endif
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync triangle intersect closest");
#endif
    }

    cudaEventRecord(events.second);
};

void OptiXAggregate::IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                                     SOA<PixelSampleState> *pixelSampleState) const {
    std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents("Trace shadow rays");

    cudaEventRecord(events.first);

    if (rootTraversable) {
        RayIntersectParameters params;
        params.traversable = rootTraversable;
        params.shadowRayQueue = shadowRayQueue;
        params.pixelSampleState = *pixelSampleState;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect shadow");
#endif
#ifdef NVTX
        nvtxRangePush("OptiXAggregate::IntersectShadow");
#endif

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &shadowSBT, maxRays, 1,
                                1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifdef NVTX
        nvtxRangePop();
#endif
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync intersect shadow");
#endif
    }

    cudaEventRecord(events.second);
}

void OptiXAggregate::IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                                       SOA<PixelSampleState> *pixelSampleState) const {
    std::pair<cudaEvent_t, cudaEvent_t> events =
        GetProfilerEvents("Tracing shadow Tr rays");

    cudaEventRecord(events.first);

    if (rootTraversable) {
        RayIntersectParameters params;
        params.traversable = rootTraversable;
        params.shadowRayQueue = shadowRayQueue;
        params.pixelSampleState = *pixelSampleState;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect shadow Tr");
#endif
#ifdef NVTX
        nvtxRangePush("OptiXAggregate::IntersectShadowTr");
#endif

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &shadowTrSBT, maxRays, 1,
                                1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifdef NVTX
        nvtxRangePop();
#endif
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync intersect shadow Tr");
#endif
    }

    cudaEventRecord(events.second);
}

void OptiXAggregate::IntersectOneRandom(
    int maxRays, SubsurfaceScatterQueue *subsurfaceScatterQueue) const {
    std::pair<cudaEvent_t, cudaEvent_t> events =
        GetProfilerEvents("Tracing subsurface scattering probe rays");

    cudaEventRecord(events.first);

    if (rootTraversable) {
        RayIntersectParameters params;
        params.traversable = rootTraversable;
        params.subsurfaceScatterQueue = subsurfaceScatterQueue;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching intersect random");
#endif
#ifdef NVTX
        nvtxRangePush("OptiXAggregate::IntersectOneRandom");
#endif

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &randomHitSBT, maxRays, 1,
                                1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifdef NVTX
        nvtxRangePop();
#endif
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync intersect random");
#endif
    }

    cudaEventRecord(events.second);
}

}  // namespace pbrt
