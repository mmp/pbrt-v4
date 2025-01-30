#include <hip/hip_runtime.h>
#include <hiprt/hiprt_device.h>

__device__ bool __filter__alphaKilled(const hiprtRay &ray, const void *data,
                                      void *payload, const hiprtHit &hit);
__device__ bool __intersection__bilinearPatch(const hiprtRay &ray, const void *data,
                                              void *payload, hiprtHit &hit);
__device__ bool __intersection__quadric(const hiprtRay &ray, const void *data,
                                        void *payload, hiprtHit &hit);

HIPRT_DEVICE bool intersectFunc(uint32_t geomType, uint32_t rayType,
                                const hiprtFuncTableHeader &tableHeader,
                                const hiprtRay &ray, void *payload, hiprtHit &hit) {
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    const void *data = tableHeader.funcDataSets[index].intersectFuncData;
    switch (index) {
    case 1: {
        return __intersection__bilinearPatch(ray, data, payload, hit);
    }
    case 2: {
        return __intersection__quadric(ray, data, payload, hit);
    }
    default: {
        return false;
    }
    }
}

HIPRT_DEVICE bool filterFunc(uint32_t geomType, uint32_t rayType,
                             const hiprtFuncTableHeader &tableHeader, const hiprtRay &ray,
                             void *payload, const hiprtHit &hit) {
    const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
    const void *data = tableHeader.funcDataSets[index].filterFuncData;
    switch (index) {
    case 0: {
        return __filter__alphaKilled(ray, data, payload, hit);
    }
    default: {
        return false;
    }
    }
}

// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/gpu/common.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/float.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/intersect.h>

// Make various functions visible to HIPRT, which doesn't get to link
// shader code with the HIP code in the main executable...
#include <pbrt/util/color.cpp>
#include <pbrt/util/colorspace.cpp>
#include <pbrt/util/log.cpp>
#include <pbrt/util/noise.cpp>
#include <pbrt/util/spectrum.cpp>
#include <pbrt/util/transform.cpp>

#include <utility>

using namespace pbrt;

alignas(alignof(RayIntersectParameters)) __constant__
    unsigned char paramBuffer[sizeof(RayIntersectParameters)];
#define params (*(RayIntersectParameters *)paramBuffer)

///////////////////////////////////////////////////////////////////////////
// Utility functions

template <bool AnyHit>
__device__ inline hiprtHit Trace(hiprtScene scene, Ray ray, Float tMax, uint32_t &missed,
                                 void *payload) {
    hiprtRay hiprtRay;
    hiprtRay.origin = make_float3(ray.o.x, ray.o.y, ray.o.z);
    hiprtRay.direction = make_float3(ray.d.x, ray.d.y, ray.d.z);
    hiprtRay.minT = 1e-7f;
    hiprtRay.maxT = tMax;

    __shared__ int sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
    hiprtSharedStackBuffer sharedStackBuffer{SHARED_STACK_SIZE, sharedStackCache};
    hiprtGlobalStack stack(params.globalStackBuffer, sharedStackBuffer);

    hiprtSharedStackBuffer sharedInstanceStackBuffer{};
    hiprtGlobalInstanceStack instanceStack(params.globalInstanceStackBuffer,
                                           sharedInstanceStackBuffer);

    hiprtHit hit;
    if constexpr (!AnyHit) {
        hiprtSceneTraversalClosestCustomStack<hiprtGlobalStack, hiprtGlobalInstanceStack>
            tr(scene, hiprtRay, stack, instanceStack, hiprtFullRayMask,
               hiprtTraversalHintDefault, payload, params.funcTable, 0, ray.time);
        hit = tr.getNextHit();
    } else {
        hiprtSceneTraversalAnyHitCustomStack<hiprtGlobalStack, hiprtGlobalInstanceStack>
            tr(scene, hiprtRay, stack, instanceStack, hiprtFullRayMask,
               hiprtTraversalHintDefault, payload, params.funcTable, 0, ray.time);
        hit = tr.getNextHit();
        if (hit.t == 0.0f)
            hit.primID = hiprtInvalidValue;
    }

    missed = uint32_t(hit.primID == hiprtInvalidValue);
    return hit;
}

static __device__ uint32_t recordIndex(const hiprtHit &hit) {
    return hit.instanceIDs[1] + params.offsets[hit.instanceIDs[0]];
}

static __forceinline__ __device__ Transform getWorldFromInstance(const hiprtHit &hit) {
    hiprtFrameMatrix hiprtWorldFromObjM =
        hiprtGetObjectToWorldFrameMatrix(params.traversable, hit.instanceIDs, 0.0f);
    hiprtFrameMatrix hiprtObjFromWorldM =
        hiprtGetWorldToObjectFrameMatrix(params.traversable, hit.instanceIDs, 0.0f);
    SquareMatrix<4> worldFromObjM, objFromWorldM;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            worldFromObjM[i][j] = hiprtWorldFromObjM.matrix[i][j];
            objFromWorldM[i][j] = hiprtObjFromWorldM.matrix[i][j];
        }
    }
    return Transform(worldFromObjM, objFromWorldM);
}

static __forceinline__ __device__ SurfaceInteraction
getTriangleIntersection(const Ray &rayWorld, const hiprtHit &hit) {
    const TriangleMeshRecord &rec = params.hgRecords[recordIndex(hit)].triRec;

    float b1 = hit.uv.x;
    float b2 = hit.uv.y;
    float b0 = 1 - b1 - b2;

    float3 rd = make_float3(rayWorld.d.x, rayWorld.d.y, rayWorld.d.z);
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    Transform worldFromInstance = getWorldFromInstance(hit);

    Float time = rayWorld.time;
    wo = worldFromInstance.ApplyInverse(wo);

    TriangleIntersection ti{b0, b1, b2, hit.t};
    SurfaceInteraction intr =
        Triangle::InteractionFromIntersection(rec.mesh, hit.primID, ti, time, wo);
    return worldFromInstance(intr);
}

static __forceinline__ __device__ SurfaceInteraction
getBilinearPatchIntersection(const Ray &rayWorld, const hiprtHit &hit, Point2f uv) {
    BilinearMeshRecord &rec = params.hgRecords[recordIndex(hit)].blpRec;

    float3 rd = make_float3(rayWorld.d.x, rayWorld.d.y, rayWorld.d.z);
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    return BilinearPatch::InteractionFromIntersection(rec.mesh, hit.primID, uv,
                                                      rayWorld.time, wo);
}

static __device__ inline SurfaceInteraction getQuadricIntersection(
    const Ray &rayWorld, const hiprtHit &hit, const QuadricIntersection &si) {
    QuadricRecord &rec = params.hgRecords[recordIndex(hit)].quadricRec;

    float3 rd = make_float3(rayWorld.d.x, rayWorld.d.y, rayWorld.d.z);
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);
    Float time = rayWorld.time;

    SurfaceInteraction intr;
    if (const Sphere *sphere = rec.shape.CastOrNullptr<Sphere>())
        intr = sphere->InteractionFromIntersection(si, wo, time);
    else if (const Cylinder *cylinder = rec.shape.CastOrNullptr<Cylinder>())
        intr = cylinder->InteractionFromIntersection(si, wo, time);
    else if (const Disk *disk = rec.shape.CastOrNullptr<Disk>())
        intr = disk->InteractionFromIntersection(si, wo, time);
    else
        CHECK(!"unexpected quadric");

    return intr;
}

///////////////////////////////////////////////////////////////////////////
// Intersection and filter functions

__device__ bool __filter__alphaKilled(const hiprtRay &ray, const void *data,
                                      void *payload, const hiprtHit &hit) {
    const TriangleMeshRecord &rec = params.hgRecords[recordIndex(hit)].triRec;
    if (!rec.alphaTexture)
        return false;

    Ray rayWorld = *(Ray *)payload;

    SurfaceInteraction intr = getTriangleIntersection(rayWorld, hit);

    BasicTextureEvaluator eval;
    Float alpha = eval(rec.alphaTexture, intr);
    if (alpha >= 1)
        return false;
    if (alpha <= 0)
        return true;
    else {
        float3 o = make_float3(rayWorld.o.x, rayWorld.o.y, rayWorld.o.z);
        float3 d = make_float3(rayWorld.d.x, rayWorld.d.y, rayWorld.d.z);
        Float u = HashFloat(o, d);
        return u > alpha;
    }
}

__device__ bool __intersection__bilinearPatch(const hiprtRay &ray_, const void *data,
                                              void *payload, hiprtHit &hit) {
    BilinearMeshRecord &rec = params.hgRecords[recordIndex(hit)].blpRec;

    float3 org = ray_.origin;
    float3 dir = ray_.direction;
    Float tMax = ray_.maxT;
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));
    Ray rayWorld = *(Ray *)payload;

    int vertexIndex = 4 * hit.primID;
    Point3f p00 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex]];
    Point3f p10 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 1]];
    Point3f p01 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 2]];
    Point3f p11 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 3]];
    pstd::optional<BilinearIntersection> isect =
        IntersectBilinearPatch(ray, tMax, p00, p10, p01, p11);

    if (!isect)
        return false;

    if (rec.alphaTexture) {
        SurfaceInteraction intr = getBilinearPatchIntersection(rayWorld, hit, isect->uv);
        BasicTextureEvaluator eval;
        Float alpha = eval(rec.alphaTexture, intr);
        if (alpha < 1) {
            if (alpha == 0)
                // No hit
                return false;

            float3 o = make_float3(rayWorld.o.x, rayWorld.o.y, rayWorld.o.z);
            float3 d = make_float3(rayWorld.d.x, rayWorld.d.y, rayWorld.d.z);
            Float u = HashFloat(o, d);
            if (u > alpha)
                // no hit
                return false;
        }
    }

    hit.t = isect->t;
    hit.uv.x = isect->uv[0];
    hit.uv.y = isect->uv[1];

    return true;
}

__device__ bool __intersection__quadric(const hiprtRay &ray_, const void *data,
                                        void *payload, hiprtHit &hit) {
    QuadricRecord &rec = params.hgRecords[recordIndex(hit)].quadricRec;

    float3 org = ray_.origin;
    float3 dir = ray_.direction;
    Float tMax = ray_.maxT;
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));
    pstd::optional<QuadricIntersection> isect;

    if (const Sphere *sphere = rec.shape.CastOrNullptr<Sphere>())
        isect = sphere->BasicIntersect(ray, tMax);
    else if (const Cylinder *cylinder = rec.shape.CastOrNullptr<Cylinder>())
        isect = cylinder->BasicIntersect(ray, tMax);
    else if (const Disk *disk = rec.shape.CastOrNullptr<Disk>())
        isect = disk->BasicIntersect(ray, tMax);

    if (!isect)
        return false;

    Ray rayWorld = *(Ray *)payload;

    if (rec.alphaTexture) {
        SurfaceInteraction intr = getQuadricIntersection(rayWorld, hit, *isect);

        BasicTextureEvaluator eval;
        Float alpha = eval(rec.alphaTexture, intr);
        if (alpha < 1) {
            if (alpha == 0)
                // No hit
                return false;

            float3 o = make_float3(rayWorld.o.x, rayWorld.o.y, rayWorld.o.z);
            float3 d = make_float3(rayWorld.d.x, rayWorld.d.y, rayWorld.d.z);
            Float u = HashFloat(o.x, o.y, o.z, d.x, d.y, d.z);
            if (u > alpha)
                // no hit
                return false;
        }
    }

    hit.t = isect->tHit;
    hit.normal = make_float3(isect->pObj.x, isect->pObj.y, isect->pObj.z);
    hit.uv.x = isect->phi;

    return true;
}

///////////////////////////////////////////////////////////////////////////
// Closest hit

struct ClosestHitContext {
    ClosestHitContext() = default;
    __device__ ClosestHitContext(Medium rayMedium, bool shadowRay)
        : rayMedium(rayMedium), shadowRay(shadowRay) {}

    Medium rayMedium;
    bool shadowRay;

    // out
    Point3fi piHit;
    Normal3f nHit;
    Material material;
    MediumInterface mediumInterface;

    __device__ Ray SpawnRayTo(const Point3f &p) const {
        Interaction intr(piHit, nHit);
        intr.mediumInterface = &mediumInterface;
        return intr.SpawnRayTo(p);
    }
};

static __forceinline__ __device__ void ProcessClosestIntersection(
    SurfaceInteraction intr, const hiprtHit &hit, ClosestHitContext &ctx) {
    int rayIndex = blockIdx.x * blockDim.x + threadIdx.x;

    Medium rayMedium = ctx.rayMedium;

    if (intr.mediumInterface)
        ctx.mediumInterface = *intr.mediumInterface;
    else
        ctx.mediumInterface = MediumInterface(rayMedium);

    ctx.piHit = intr.pi;
    ctx.nHit = intr.n;
    ctx.material = intr.material;

    if (ctx.shadowRay)
        return;

    // We only have the ray queue (and it only makes sense to access) for
    // regular closest hit rays.
    RayWorkItem r = (*params.rayQueue)[rayIndex];

    EnqueueWorkAfterIntersection(r, rayMedium, hit.t, intr, params.mediumSampleQueue,
                                 params.nextRayQueue, params.hitAreaLightQueue,
                                 params.basicEvalMaterialQueue,
                                 params.universalEvalMaterialQueue);
}
__device__ void closesthitTriangle(const Ray &rayWorld, const hiprtHit &hit,
                                   ClosestHitContext &ctx) {
    const TriangleMeshRecord &rec = params.hgRecords[recordIndex(hit)].triRec;

    SurfaceInteraction intr = getTriangleIntersection(rayWorld, hit);

    if (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
        intr.mediumInterface = rec.mediumInterface;
    intr.material = rec.material;
    if (!rec.areaLights.empty())
        intr.areaLight = rec.areaLights[hit.primID];

    ProcessClosestIntersection(intr, hit, ctx);
}

__device__ void closesthitBilinearPatch(const Ray &rayWorld, const hiprtHit &hit,
                                        ClosestHitContext &ctx) {
    BilinearMeshRecord &rec = params.hgRecords[recordIndex(hit)].blpRec;

    Point2f uv(hit.uv.x, hit.uv.y);

    SurfaceInteraction intr = getBilinearPatchIntersection(rayWorld, hit, uv);
    if (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
        intr.mediumInterface = rec.mediumInterface;
    intr.material = rec.material;
    if (!rec.areaLights.empty())
        intr.areaLight = rec.areaLights[hit.primID];

    Transform worldFromInstance = getWorldFromInstance(hit);
    intr = worldFromInstance(intr);

    ProcessClosestIntersection(intr, hit, ctx);
}

__device__ void closesthitQuadric(const Ray &rayWorld, const hiprtHit &hit,
                                  ClosestHitContext &ctx) {
    QuadricRecord &rec = params.hgRecords[recordIndex(hit)].quadricRec;
    QuadricIntersection qi;
    qi.pObj = Point3f(hit.normal.x, hit.normal.y, hit.normal.z);
    qi.phi = hit.uv.x;

    SurfaceInteraction intr = getQuadricIntersection(rayWorld, hit, qi);
    if (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
        intr.mediumInterface = rec.mediumInterface;
    intr.material = rec.material;
    if (rec.areaLight)
        intr.areaLight = rec.areaLight;

    Transform worldFromInstance = getWorldFromInstance(hit);
    intr = worldFromInstance(intr);

    ProcessClosestIntersection(intr, hit, ctx);
}

extern "C" __global__ void __raygen__findClosest() {
    int rayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIndex >= params.rayQueue->Size())
        return;

    RayWorkItem r = (*params.rayQueue)[rayIndex];
    Ray ray = r.ray;
    Float tMax = 1e30f;

    PBRT_DBG("ray o %f %f %f dir %f %f %f tmax %f\n", ray.o.x, ray.o.y, ray.o.z, ray.d.x,
             ray.d.y, ray.d.z, tMax);

    uint32_t missed = 0;
    hiprtHit hit = Trace<false>(params.traversable, ray, tMax, missed, &ray);

    if (missed) {
        EnqueueWorkAfterMiss(r, params.mediumSampleQueue, params.escapedRayQueue);
        return;
    }

    ClosestHitContext ctx(ray.medium, false);
    if (params.hgRecords[recordIndex(hit)].type == HitgroupRecord::TriangleMesh)
        closesthitTriangle(ray, hit, ctx);
    else if (params.hgRecords[recordIndex(hit)].type == HitgroupRecord::BilinearMesh)
        closesthitBilinearPatch(ray, hit, ctx);
    else if (params.hgRecords[recordIndex(hit)].type == HitgroupRecord::Quadric)
        closesthitQuadric(ray, hit, ctx);
    else
        CHECK(!"unexpected primitive type");
}

///////////////////////////////////////////////////////////////////////////
// Shadow rays

extern "C" __global__ void __raygen__shadow() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.shadowRayQueue->Size())
        return;

    ShadowRayWorkItem sr = (*params.shadowRayQueue)[index];
    PBRT_DBG("Tracing shadow ray index %d o %f %f %f d %f %f %f\n", index, sr.ray.o.x,
             sr.ray.o.y, sr.ray.o.z, sr.ray.d.x, sr.ray.d.y, sr.ray.d.z);

    uint32_t missed = 0;
    Trace<true>(params.traversable, sr.ray, sr.tMax, missed, &sr.ray);

    RecordShadowRayResult(sr, &params.pixelSampleState, !missed);
}

extern "C" __global__ void __raygen__shadow_Tr() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.shadowRayQueue->Size())
        return;

    ShadowRayWorkItem sr = (*params.shadowRayQueue)[index];

    ClosestHitContext ctx;

    TraceTransmittance(
        sr, &params.pixelSampleState,
        [&](Ray ray, Float tMax) -> TransmittanceTraceResult {
            uint32_t missed = 0;
            hiprtHit hit = Trace<false>(params.traversable, ray, tMax, missed,
                                        &ray);  // the closest hit is actually used

            ctx = ClosestHitContext(ray.medium, true);

            if (!missed) {
                if (params.hgRecords[recordIndex(hit)].type ==
                    HitgroupRecord::TriangleMesh)
                    closesthitTriangle(ray, hit, ctx);
                else if (params.hgRecords[recordIndex(hit)].type ==
                         HitgroupRecord::BilinearMesh)
                    closesthitBilinearPatch(ray, hit, ctx);
                else if (params.hgRecords[recordIndex(hit)].type ==
                         HitgroupRecord::Quadric)
                    closesthitQuadric(ray, hit, ctx);
                else
                    CHECK(!"unexpected primitive type");
            }

            return TransmittanceTraceResult{!missed, Point3f(ctx.piHit), ctx.material};
        },
        [&](Point3f p) -> Ray { return ctx.SpawnRayTo(p); });
}

///////////////////////////////////////////////////////////////////////////
// Random hit (for subsurface scattering)

struct RandomHitPayload {
    WeightedReservoirSampler<SubsurfaceInteraction> wrs;
    Material material;
    pstd::optional<SurfaceInteraction> intr;
};

__device__ void closesthitRandomHitTriangle(const Ray &rayWorld, const hiprtHit &hit,
                                            RandomHitPayload *p) {
    const TriangleMeshRecord &rec = params.hgRecords[recordIndex(hit)].triRec;

    PBRT_DBG("Anyhit triangle for random hit: rec.material %p params.materials %p\n",
             rec.material.ptr(), p->material.ptr());

    SurfaceInteraction intr = getTriangleIntersection(rayWorld, hit);
    p->intr = intr;

    if (rec.material == p->material)
        p->wrs.Add([&] __device__() { return intr; }, 1.f);
}

__device__ void closesthitRandomHitBilinearPatch(const Ray &rayWorld, const hiprtHit &hit,
                                                 RandomHitPayload *p) {
    BilinearMeshRecord &rec = params.hgRecords[recordIndex(hit)].blpRec;

    PBRT_DBG("Anyhit blp for random hit: rec.material %p params.materials %p\n",
             rec.material.ptr(), p->material.ptr());

    Point2f uv(hit.uv.x, hit.uv.y);
    SurfaceInteraction intr = getBilinearPatchIntersection(rayWorld, hit, uv);
    p->intr = intr;

    if (rec.material == p->material)
        p->wrs.Add([&] __device__() { return intr; }, 1.f);
}

__device__ void closesthitRandomHitQuadric(const Ray &rayWorld, const hiprtHit &hit,
                                           RandomHitPayload *p) {
    QuadricRecord &rec = params.hgRecords[recordIndex(hit)].quadricRec;

    PBRT_DBG("Anyhit quadric for random hit: rec.material %p params.materials %p\n",
             rec.material.ptr(), p->material.ptr());

    QuadricIntersection qi;
    qi.pObj = Point3f(hit.normal.x, hit.normal.y, hit.normal.z);
    qi.phi = hit.uv.x;

    SurfaceInteraction intr = getQuadricIntersection(rayWorld, hit, qi);
    p->intr = intr;

    if (rec.material == p->material)
        p->wrs.Add([&] __device__() { return intr; }, 1.f);
}

extern "C" __global__ void __raygen__randomHit() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.subsurfaceScatterQueue->Size())
        return;

    SubsurfaceScatterWorkItem s = (*params.subsurfaceScatterQueue)[index];

    Ray ray(s.p0, s.p1 - s.p0);

    RandomHitPayload payload;
    payload.wrs.Seed(Hash(s.p0, s.p1));
    payload.material = s.material;

    PBRT_DBG("Randomhit raygen ray.o %f %f %f ray.d %f %f %f\n", ray.o.x, ray.o.y,
             ray.o.z, ray.d.x, ray.d.y, ray.d.z);

    int depth = 0;
    while (LengthSquared(ray.d) > 0 && ++depth < 100) {
        uint32_t missed = 0;
        hiprtHit hit =
            Trace<false>(params.traversable, ray, 1.f /* tMax */, missed, &ray);

        if (!missed) {
            if (params.hgRecords[recordIndex(hit)].type == HitgroupRecord::TriangleMesh)
                closesthitRandomHitTriangle(ray, hit, &payload);
            else if (params.hgRecords[recordIndex(hit)].type ==
                     HitgroupRecord::BilinearMesh)
                closesthitRandomHitBilinearPatch(ray, hit, &payload);
            else if (params.hgRecords[recordIndex(hit)].type == HitgroupRecord::Quadric)
                closesthitRandomHitQuadric(ray, hit, &payload);
            else
                CHECK(!"unexpected primitive type");
        }

        if (payload.intr) {
            ray = payload.intr->SpawnRayTo(s.p1);
            payload.intr.reset();
        } else
            break;
    }

    if (payload.wrs.HasSample() &&
        payload.wrs.WeightSum() > 0) {  // TODO: latter check shouldn't be needed...
        const SubsurfaceInteraction &si = payload.wrs.GetSample();

        params.subsurfaceScatterQueue->reservoirPDF[index] =
            payload.wrs.SampleProbability();
        params.subsurfaceScatterQueue->ssi[index] = payload.wrs.GetSample();
    } else
        params.subsurfaceScatterQueue->reservoirPDF[index] = 0;
}
