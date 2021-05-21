// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/gpu/aggregate.h>
#include <pbrt/gpu/optix.h>
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

// Make various functions visible to OptiX, which doesn't get to link
// shader code with the CUDA code in the main executable...
#include <pbrt/util/color.cpp>
#include <pbrt/util/colorspace.cpp>
#include <pbrt/util/log.cpp>
#include <pbrt/util/noise.cpp>
#include <pbrt/util/spectrum.cpp>
#include <pbrt/util/transform.cpp>

#include <optix_device.h>

#include <utility>

using namespace pbrt;

extern "C" {
extern __constant__ pbrt::RayIntersectParameters params;
}

///////////////////////////////////////////////////////////////////////////
// Utility functions

// Payload management
__device__ inline uint32_t packPointer0(void *ptr) {
    uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    return uptr >> 32;
}

__device__ inline uint32_t packPointer1(void *ptr) {
    uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    return uint32_t(uptr);
}

template <typename T>
static __forceinline__ __device__ T *getPayload() {
    uint32_t p0 = optixGetPayload_0(), p1 = optixGetPayload_1();
    const uint64_t uptr = (uint64_t(p0) << 32) | p1;
    return reinterpret_cast<T *>(uptr);
}

template <typename... Args>
__device__ inline void Trace(OptixTraversableHandle traversable, Ray ray, Float tMin,
                             Float tMax, OptixRayFlags flags, Args &&... payload) {
    optixTrace(traversable, make_float3(ray.o.x, ray.o.y, ray.o.z),
               make_float3(ray.d.x, ray.d.y, ray.d.z), tMin, tMax, ray.time,
               OptixVisibilityMask(255), flags, 0, /* ray type */
               1,                                  /* number of ray types */
               0,                                  /* missSBTIndex */
               std::forward<Args>(payload)...);
}

///////////////////////////////////////////////////////////////////////////
// Closest hit

struct ClosestHitContext {
    ClosestHitContext() = default;
    PBRT_GPU
    ClosestHitContext(Medium rayMedium, bool shadowRay)
        : rayMedium(rayMedium), shadowRay(shadowRay) {}

    Medium rayMedium;
    bool shadowRay;

    // out
    Point3fi piHit;
    Normal3f nHit;
    Material material;
    MediumInterface mediumInterface;

    PBRT_GPU
    Ray SpawnRayTo(const Point3f &p) const {
        Interaction intr(piHit, nHit);
        intr.mediumInterface = &mediumInterface;
        return intr.SpawnRayTo(p);
    }
};

extern "C" __global__ void __raygen__findClosest() {
    int rayIndex(optixGetLaunchIndex().x);
    if (rayIndex >= params.rayQueue->Size())
        return;

    RayWorkItem r = (*params.rayQueue)[rayIndex];
    Ray ray = r.ray;
    Float tMax = 1e30f;

    ClosestHitContext ctx(ray.medium, false);
    uint32_t p0 = packPointer0(&ctx), p1 = packPointer1(&ctx);

    PBRT_DBG("ray o %f %f %f dir %f %f %f tmax %f\n", ray.o.x, ray.o.y, ray.o.z, ray.d.x,
        ray.d.y, ray.d.z, tMax);

    uint32_t missed = 0;
    Trace(params.traversable, ray, 0.f /* tMin */, tMax, OPTIX_RAY_FLAG_NONE, p0, p1,
          missed);

    if (missed)
        EnqueueWorkAfterMiss(r, params.mediumSampleQueue, params.escapedRayQueue);
}

extern "C" __global__ void __miss__noop() {
    optixSetPayload_2(1);
}

static __forceinline__ __device__ void ProcessClosestIntersection(
    SurfaceInteraction intr) {
    int rayIndex = optixGetLaunchIndex().x;

    Medium rayMedium = getPayload<ClosestHitContext>()->rayMedium;
    if (intr.mediumInterface)
        getPayload<ClosestHitContext>()->mediumInterface = *intr.mediumInterface;
    else
        getPayload<ClosestHitContext>()->mediumInterface = MediumInterface(rayMedium);

    getPayload<ClosestHitContext>()->piHit = intr.pi;
    getPayload<ClosestHitContext>()->nHit = intr.n;
    getPayload<ClosestHitContext>()->material = intr.material;

    if (getPayload<ClosestHitContext>()->shadowRay)
        return;

    // We only have the ray queue (and it only makes sense to access) for
    // regular closest hit rays.
    RayWorkItem r = (*params.rayQueue)[rayIndex];

    EnqueueWorkAfterIntersection(r, rayMedium, optixGetRayTmax(), intr, params.mediumSampleQueue,
                                 params.nextRayQueue, params.hitAreaLightQueue,
                                 params.basicEvalMaterialQueue,
                                 params.universalEvalMaterialQueue);
}

static __forceinline__ __device__ Transform getWorldFromInstance() {
    assert(optixGetTransformListSize() == 1);
    float worldFromObj[12], objFromWorld[12];
    optixGetObjectToWorldTransformMatrix(worldFromObj);
    optixGetWorldToObjectTransformMatrix(objFromWorld);
    SquareMatrix<4> worldFromObjM(worldFromObj[0], worldFromObj[1], worldFromObj[2],
                                  worldFromObj[3], worldFromObj[4], worldFromObj[5],
                                  worldFromObj[6], worldFromObj[7], worldFromObj[8],
                                  worldFromObj[9], worldFromObj[10], worldFromObj[11],
                                  0.f, 0.f, 0.f, 1.f);
    SquareMatrix<4> objFromWorldM(objFromWorld[0], objFromWorld[1], objFromWorld[2],
                                  objFromWorld[3], objFromWorld[4], objFromWorld[5],
                                  objFromWorld[6], objFromWorld[7], objFromWorld[8],
                                  objFromWorld[9], objFromWorld[10], objFromWorld[11],
                                  0.f, 0.f, 0.f, 1.f);

    return Transform(worldFromObjM, objFromWorldM);
}

///////////////////////////////////////////////////////////////////////////
// Triangles

static __forceinline__ __device__ SurfaceInteraction
getTriangleIntersection() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    float b1 = optixGetTriangleBarycentrics().x;
    float b2 = optixGetTriangleBarycentrics().y;
    float b0 = 1 - b1 - b2;

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    Transform worldFromInstance = getWorldFromInstance();

    Float time = optixGetRayTime();
    wo = worldFromInstance.ApplyInverse(wo);

    TriangleIntersection ti{b0, b1, b2, optixGetRayTmax()};
    SurfaceInteraction intr =
        Triangle::InteractionFromIntersection(rec.mesh, optixGetPrimitiveIndex(),
                                              ti, time, wo);
    return worldFromInstance(intr);
}

static __forceinline__ __device__ bool alphaKilled(const TriangleMeshRecord &rec) {
    if (!rec.alphaTexture)
        return false;

    SurfaceInteraction intr = getTriangleIntersection();

    BasicTextureEvaluator eval;
    Float alpha = eval(rec.alphaTexture, intr);
    if (alpha >= 1)
        return false;
    if (alpha <= 0)
        return true;
    else {
        float3 o = optixGetWorldRayOrigin();
        float3 d = optixGetWorldRayDirection();
        Float u = HashFloat(o, d);
        return u > alpha;
    }
}

extern "C" __global__ void __closesthit__triangle() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    SurfaceInteraction intr = getTriangleIntersection();

    if (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
        intr.mediumInterface = rec.mediumInterface;
    intr.material = rec.material;
    if (!rec.areaLights.empty())
        intr.areaLight = rec.areaLights[optixGetPrimitiveIndex()];

    ProcessClosestIntersection(intr);
}

extern "C" __global__ void __anyhit__triangle() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    if (alphaKilled(rec))
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__shadowTriangle() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    if (alphaKilled(rec))
        optixIgnoreIntersection();
}

///////////////////////////////////////////////////////////////////////////
// Shadow rays

extern "C" __global__ void __raygen__shadow() {
    int index = optixGetLaunchIndex().x;
    if (index >= params.shadowRayQueue->Size())
        return;

    ShadowRayWorkItem sr = (*params.shadowRayQueue)[index];
    PBRT_DBG("Tracing shadow ray index %d o %f %f %f d %f %f %f\n",
             index, sr.ray.o.x, sr.ray.o.y, sr.ray.o.z,
             sr.ray.d.x, sr.ray.d.y, sr.ray.d.z);

    uint32_t missed = 0;
    Trace(params.traversable, sr.ray, 1e-5f /* tMin */, sr.tMax, OPTIX_RAY_FLAG_NONE,
          missed);

    RecordShadowRayIntersection(sr, &params.pixelSampleState, !missed);
}

extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __raygen__shadow_Tr() {
    PBRT_DBG("raygen sahadow tr %d\n", optixGetLaunchIndex().x);
    int index = optixGetLaunchIndex().x;
    if (index >= params.shadowRayQueue->Size())
        return;

    ShadowRayWorkItem sr = (*params.shadowRayQueue)[index];

    ClosestHitContext ctx;

    TraceTransmittance(sr, &params.pixelSampleState,
                       [&](Ray ray, Float tMax) -> TransmittanceTraceResult {
                           ctx = ClosestHitContext(ray.medium, true);
                           uint32_t p0 = packPointer0(&ctx), p1 = packPointer1(&ctx);

                           uint32_t missed = 0;

                           Trace(params.traversable, ray, 1e-5f /* tMin */, tMax, OPTIX_RAY_FLAG_NONE, p0,
                                 p1, missed);

                           return TransmittanceTraceResult{!missed, Point3f(ctx.piHit), ctx.material};
                       },
                       [&](Point3f p) -> Ray {
                           return ctx.SpawnRayTo(p);
                       });
}

extern "C" __global__ void __miss__shadow_Tr() {
    optixSetPayload_2(1);
}

/////////////////////////////////////////////////////////////////////////////////////
// Quadrics

static __device__ inline SurfaceInteraction getQuadricIntersection(
    const QuadricIntersection &si) {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);
    Float time = optixGetRayTime();

    SurfaceInteraction intr;
    if (const Sphere *sphere = rec.shape.CastOrNullptr<Sphere>())
        intr = sphere->InteractionFromIntersection(si, wo, time);
    else if (const Cylinder *cylinder = rec.shape.CastOrNullptr<Cylinder>())
        intr = cylinder->InteractionFromIntersection(si, wo, time);
    else if (const Disk *disk = rec.shape.CastOrNullptr<Disk>())
        intr = disk->InteractionFromIntersection(si, wo, time);
    else
        assert(!"unexpected quadric");

    return intr;
}

extern "C" __global__ void __closesthit__quadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());
    QuadricIntersection qi;
    qi.pObj =
        Point3f(BitsToFloat(optixGetAttribute_0()), BitsToFloat(optixGetAttribute_1()),
                BitsToFloat(optixGetAttribute_2()));
    qi.phi = BitsToFloat(optixGetAttribute_3());

    SurfaceInteraction intr = getQuadricIntersection(qi);
    if (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
        intr.mediumInterface = rec.mediumInterface;
    intr.material = rec.material;
    if (rec.areaLight)
        intr.areaLight = rec.areaLight;

    Transform worldFromInstance = getWorldFromInstance();
    intr = worldFromInstance(intr);

    ProcessClosestIntersection(intr);
}

extern "C" __global__ void __anyhit__shadowQuadric() {
}

extern "C" __global__ void __intersection__quadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    float3 org = optixGetObjectRayOrigin();
    float3 dir = optixGetObjectRayDirection();
    Float tMax = optixGetRayTmax();
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));
    pstd::optional<QuadricIntersection> isect;

    if (const Sphere *sphere = rec.shape.CastOrNullptr<Sphere>())
        isect = sphere->BasicIntersect(ray, tMax);
    else if (const Cylinder *cylinder = rec.shape.CastOrNullptr<Cylinder>())
        isect = cylinder->BasicIntersect(ray, tMax);
    else if (const Disk *disk = rec.shape.CastOrNullptr<Disk>())
        isect = disk->BasicIntersect(ray, tMax);

    if (!isect)
        return;

    if (rec.alphaTexture) {
        SurfaceInteraction intr = getQuadricIntersection(*isect);

        BasicTextureEvaluator eval;
        Float alpha = eval(rec.alphaTexture, intr);
        if (alpha < 1) {
            if (alpha == 0)
                // No hit
                return;

            float3 o = optixGetWorldRayOrigin();
            float3 d = optixGetWorldRayDirection();
            Float u = HashFloat(o.x, o.y, o.z, d.x, d.y, d.z);
            if (u > alpha)
                // no hit
                return;
        }
    }

    optixReportIntersection(isect->tHit, 0 /* hit kind */, FloatToBits(isect->pObj.x),
                            FloatToBits(isect->pObj.y), FloatToBits(isect->pObj.z),
                            FloatToBits(isect->phi));
}

///////////////////////////////////////////////////////////////////////////
// Bilinear patches

static __forceinline__ __device__ SurfaceInteraction
getBilinearPatchIntersection(Point2f uv) {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    return BilinearPatch::InteractionFromIntersection(rec.mesh, optixGetPrimitiveIndex(),
                                                      uv, optixGetRayTime(), wo);
}

extern "C" __global__ void __closesthit__bilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    Point2f uv(BitsToFloat(optixGetAttribute_0()), BitsToFloat(optixGetAttribute_1()));

    SurfaceInteraction intr = getBilinearPatchIntersection(uv);
    if (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
        intr.mediumInterface = rec.mediumInterface;
    intr.material = rec.material;
    if (!rec.areaLights.empty())
        intr.areaLight = rec.areaLights[optixGetPrimitiveIndex()];

    Transform worldFromInstance = getWorldFromInstance();
    intr = worldFromInstance(intr);

    ProcessClosestIntersection(intr);
}

extern "C" __global__ void __anyhit__shadowBilinearPatch() {
}

extern "C" __global__ void __intersection__bilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    float3 org = optixGetObjectRayOrigin();
    float3 dir = optixGetObjectRayDirection();
    Float tMax = optixGetRayTmax();
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));

    int vertexIndex = 4 * optixGetPrimitiveIndex();
    Point3f p00 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex]];
    Point3f p10 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 1]];
    Point3f p01 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 2]];
    Point3f p11 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 3]];
    pstd::optional<BilinearIntersection> isect =
        IntersectBilinearPatch(ray, tMax, p00, p10, p01, p11);

    if (!isect)
        return;

    if (rec.alphaTexture) {
        SurfaceInteraction intr = getBilinearPatchIntersection(isect->uv);
        BasicTextureEvaluator eval;
        Float alpha = eval(rec.alphaTexture, intr);
        if (alpha < 1) {
            if (alpha == 0)
                // No hit
                return;

            float3 o = optixGetWorldRayOrigin();
            float3 d = optixGetWorldRayDirection();
            Float u = HashFloat(o, d);
            if (u > alpha)
                // no hit
                return;
        }
    }

    optixReportIntersection(isect->t, 0 /* hit kind */, FloatToBits(isect->uv[0]),
                            FloatToBits(isect->uv[1]));
}

///////////////////////////////////////////////////////////////////////////
// Random hit (for subsurface scattering)

struct RandomHitPayload {
    WeightedReservoirSampler<SubsurfaceInteraction> wrs;
    Material material;
    pstd::optional<SurfaceInteraction> intr;
};

extern "C" __global__ void __raygen__randomHit() {
    // Keep as uint32_t so can pass directly to optixTrace.
    uint32_t index = optixGetLaunchIndex().x;
    if (index >= params.subsurfaceScatterQueue->Size())
        return;

    SubsurfaceScatterWorkItem s = (*params.subsurfaceScatterQueue)[index];

    Ray ray(s.p0, s.p1 - s.p0);

    RandomHitPayload payload;
    payload.wrs.Seed(Hash(s.p0, s.p1));
    payload.material = s.material;

    uint32_t ptr0 = packPointer0(&payload), ptr1 = packPointer1(&payload);

    PBRT_DBG("Randomhit raygen ray.o %f %f %f ray.d %f %f %f tMax %f\n", ray.o.x, ray.o.y,
        ray.o.z, ray.d.x, ray.d.y, ray.d.z, tMax);

    while (true) {
        Trace(params.traversable, ray, 0.f /* tMin */, 1.f /* tMax */,
              OPTIX_RAY_FLAG_NONE, ptr0, ptr1);

        if (payload.intr) {
            ray = payload.intr->SpawnRayTo(s.p1);
            payload.intr.reset();
        } else
            break;
    }

    if (payload.wrs.HasSample() &&
        payload.wrs.WeightSum() > 0) {  // TODO: latter check shouldn't be needed...
        const SubsurfaceInteraction &si = payload.wrs.GetSample();
        PBRT_DBG("optix si p %f %f %f n %f %f %f\n", si.p().x, si.p().y, si.p().z, si.n.x,
            si.n.y, si.n.z);

        params.subsurfaceScatterQueue->reservoirPDF[index] = payload.wrs.SamplePDF();
        params.subsurfaceScatterQueue->ssi[index] = payload.wrs.GetSample();
    } else
        params.subsurfaceScatterQueue->reservoirPDF[index] = 0;
}

extern "C" __global__ void __closesthit__randomHitTriangle() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    RandomHitPayload *p = getPayload<RandomHitPayload>();

    PBRT_DBG("Anyhit triangle for random hit: rec.material %p params.materials %p\n",
        rec.material.ptr(), p->material.ptr());

    SurfaceInteraction intr = getTriangleIntersection();
    p->intr = intr;

    if (rec.material == p->material)
        p->wrs.Add([&] PBRT_CPU_GPU() { return intr; }, 1.f);
}

extern "C" __global__ void __closesthit__randomHitBilinearPatch() {
    BilinearMeshRecord &rec = *(BilinearMeshRecord *)optixGetSbtDataPointer();

    RandomHitPayload *p = getPayload<RandomHitPayload>();

    PBRT_DBG("Anyhit blp for random hit: rec.material %p params.materials %p\n",
        rec.material.ptr(), p->material.ptr());

    Point2f uv(BitsToFloat(optixGetAttribute_0()),
               BitsToFloat(optixGetAttribute_1()));
    SurfaceInteraction intr = getBilinearPatchIntersection(uv);
    p->intr = intr;

    if (rec.material == p->material)
        p->wrs.Add([&] PBRT_CPU_GPU() { return intr; }, 1.f);
}

extern "C" __global__ void __closesthit__randomHitQuadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    RandomHitPayload *p = getPayload<RandomHitPayload>();

    PBRT_DBG("Anyhit quadric for random hit: rec.material %p params.materials %p\n",
        rec.material.ptr(), p->material.ptr());

    QuadricIntersection qi;
    qi.pObj = Point3f(BitsToFloat(optixGetAttribute_0()),
                      BitsToFloat(optixGetAttribute_1()),
                      BitsToFloat(optixGetAttribute_2()));
    qi.phi = BitsToFloat(optixGetAttribute_3());

    SurfaceInteraction intr = getQuadricIntersection(qi);
    p->intr = intr;

    if (rec.material == p->material)
        p->wrs.Add([&] PBRT_CPU_GPU() { return intr; }, 1.f);
}
