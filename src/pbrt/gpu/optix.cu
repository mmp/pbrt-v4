// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/gpu/accel.h>
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

// Make various functions visible to OptiX, which doesn't get to link
// shader code with the CUDA code in the main executable...
#include <pbrt/util/color.cpp>
#include <pbrt/util/colorspace.cpp>
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
    PBRT_GPU
    ClosestHitContext(MediumHandle rayMedium, bool shadowRay)
        : rayMedium(rayMedium), shadowRay(shadowRay) {}

    MediumHandle rayMedium;
    bool shadowRay;

    // out
    Point3fi piHit;
    Normal3f nHit;
    MaterialHandle material;
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

    if (missed) {
        if (ray.medium) {
            PBRT_DBG("Adding miss ray to mediumSampleQueue. "
                "ray %f %f %f d %f %f %f T_hat %f %f %f %f\n",
                r.ray.o.x, r.ray.o.y, r.ray.o.z, r.ray.d.x, r.ray.d.y, r.ray.d.z,
                r.T_hat[0], r.T_hat[1], r.T_hat[2], r.T_hat[3]);
            params.mediumSampleQueue->Push(r.ray, Infinity, r.lambda, r.T_hat, r.uniPathPDF,
                                           r.lightPathPDF, r.pixelIndex, r.prevIntrCtx,
                                           r.isSpecularBounce,
                                           r.anyNonSpecularBounces, r.etaScale);
        } else if (params.escapedRayQueue) {
            PBRT_DBG("Adding ray to escapedRayQueue ray index %d pixel index %d\n", rayIndex,
                     r.pixelIndex);
            params.escapedRayQueue->Push(EscapedRayWorkItem{
                ray.o, ray.d, r.lambda, r.pixelIndex, (int)r.isSpecularBounce,
                r.T_hat, r.uniPathPDF, r.lightPathPDF, r.prevIntrCtx});
        }
    }
}

extern "C" __global__ void __miss__noop() {
    optixSetPayload_2(1);
}

static __forceinline__ __device__ void ProcessClosestIntersection(
    SurfaceInteraction intr) {
    int rayIndex = optixGetLaunchIndex().x;

    MediumHandle rayMedium = getPayload<ClosestHitContext>()->rayMedium;
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

    if (rayMedium) {
        assert(params.mediumSampleQueue);
        PBRT_DBG("Enqueuing into medium sample queue\n");
        params.mediumSampleQueue->Push(
            MediumSampleWorkItem{r.ray,
                                 optixGetRayTmax(),
                                 r.lambda,
                                 r.T_hat,
                                 r.uniPathPDF,
                                 r.lightPathPDF,
                                 r.pixelIndex,
                                 r.prevIntrCtx,
                                 r.isSpecularBounce,
                                 r.anyNonSpecularBounces,
                                 r.etaScale,
                                 intr.areaLight,
                                 intr.pi,
                                 intr.n,
                                 -r.ray.d,
                                 intr.uv,
                                 intr.material,
                                 intr.shading.n,
                                 intr.shading.dpdu,
                                 intr.shading.dpdv,
                                 intr.shading.dndu,
                                 intr.shading.dndv,
                                 getPayload<ClosestHitContext>()->mediumInterface});
        return;
    }

    // FIXME: this is all basically duplicate code w/medium.cpp
    MaterialHandle material = intr.material;

    const MixMaterial *mix = material.CastOrNullptr<MixMaterial>();
    if (mix) {
        MaterialEvalContext ctx(intr);
        material = mix->ChooseMaterial(BasicTextureEvaluator(), ctx);
    }

    if (!material) {
        PBRT_DBG("Enqueuing into medium transition queue: ray index %d pixel index %d \n",
                 rayIndex, r.pixelIndex);
        Ray newRay = intr.SpawnRay(r.ray.d);
        params.nextRayQueue->PushIndirectRay(
            newRay, r.prevIntrCtx, r.T_hat, r.uniPathPDF, r.lightPathPDF, r.lambda,
            r.etaScale, r.isSpecularBounce, r.anyNonSpecularBounces, r.pixelIndex);
        return;
    }

    if (intr.areaLight) {
        PBRT_DBG("Ray hit an area light: adding to hitAreaLightQueue ray index %d pixel index "
                 "%d\n", rayIndex, r.pixelIndex);
        Ray ray = r.ray;
        // TODO: intr.wo == -ray.d?
        params.hitAreaLightQueue->Push(HitAreaLightWorkItem{
            intr.areaLight, intr.p(), intr.n, intr.uv, intr.wo, r.lambda,
            r.T_hat, r.uniPathPDF, r.lightPathPDF, r.prevIntrCtx,
            (int)r.isSpecularBounce, r.pixelIndex});
    }

    FloatTextureHandle displacement = material.GetDisplacement();

    MaterialEvalQueue *q =
        (material.CanEvaluateTextures(BasicTextureEvaluator()) &&
         (!displacement || BasicTextureEvaluator().CanEvaluate({displacement}, {})))
            ? params.basicEvalMaterialQueue
            : params.universalEvalMaterialQueue;

    PBRT_DBG("Enqueuing for material eval, mtl tag %d\n", material.Tag());

    auto enqueue = [=](auto ptr) {
        using Material = typename std::remove_reference_t<decltype(*ptr)>;
        q->Push(MaterialEvalWorkItem<Material>{
            ptr, intr.pi, intr.n, intr.shading.n,
            intr.shading.dpdu, intr.shading.dpdv, intr.shading.dndu, intr.shading.dndv,
                intr.uv, r.lambda, r.anyNonSpecularBounces, intr.wo, r.pixelIndex,
                r.T_hat, r.uniPathPDF, r.etaScale,
                getPayload<ClosestHitContext>()->mediumInterface, intr.time});
    };
    material.Dispatch(enqueue);

    PBRT_DBG("Closest hit found intersection at t %f\n", optixGetRayTmax());
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

    Transform worldFromInstance(worldFromObjM, objFromWorldM);

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
        Float u = uint32_t(Hash(o, d)) * 0x1p-32f;
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

    if (rec.material && rec.material.IsTransparent())
        optixIgnoreIntersection();

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

    SampledSpectrum Ld;
    if (missed) {
        Ld = sr.Ld / (sr.uniPathPDF + sr.lightPathPDF).Average();
        PBRT_DBG("Unoccluded shadow ray. Final Ld %f %f %f %f "
                 "(sr.Ld %f %f %f %f uniPathPDF %f %f %f %f lightPathPDF %f %f %f %f)\n",
                 Ld[0], Ld[1], Ld[2], Ld[3],
                 sr.Ld[0], sr.Ld[1], sr.Ld[2], sr.Ld[3],
                 sr.uniPathPDF[0], sr.uniPathPDF[1], sr.uniPathPDF[2], sr.uniPathPDF[3],
                 sr.lightPathPDF[0], sr.lightPathPDF[1], sr.lightPathPDF[2], sr.lightPathPDF[3]);

        SampledSpectrum Lpixel = params.pixelSampleState->L[sr.pixelIndex];
        params.pixelSampleState->L[sr.pixelIndex] = Lpixel + Ld;
    } else {
        PBRT_DBG("Shadow ray was occluded\n");
    }
}

extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(1);
}

__device__
inline void rescale(SampledSpectrum &T_hat, SampledSpectrum &lightPathPDF,
                    SampledSpectrum &uniPathPDF) {
    if (T_hat.MaxComponentValue() > 0x1p24f ||
        lightPathPDF.MaxComponentValue() > 0x1p24f ||
        uniPathPDF.MaxComponentValue() > 0x1p24f) {
        T_hat *= 1.f / 0x1p24f;
        lightPathPDF *= 1.f / 0x1p24f;
        uniPathPDF *= 1.f / 0x1p24f;
    } else if (T_hat.MaxComponentValue() < 0x1p-24f ||
               lightPathPDF.MaxComponentValue() < 0x1p-24f ||
               uniPathPDF.MaxComponentValue() < 0x1p-24f) {
        T_hat *= 0x1p24f;
        lightPathPDF *= 0x1p24f;
        uniPathPDF *= 0x1p24f;
    }
}

extern "C" __global__ void __raygen__shadow_Tr() {
    PBRT_DBG("raygen sahadow tr %d\n", optixGetLaunchIndex().x);
    int index = optixGetLaunchIndex().x;
    if (index >= params.shadowRayQueue->Size())
        return;

    ShadowRayWorkItem sr = (*params.shadowRayQueue)[index];
    SampledWavelengths lambda = sr.lambda;

    SampledSpectrum Ld = sr.Ld;
    PBRT_DBG("Initial Ld %f %f %f %f shadow ray index %d pixel index %d\n", Ld[0], Ld[1],
        Ld[2], Ld[3], index, sr.pixelIndex);

    Ray ray = sr.ray;
    Float tMax = sr.tMax;
    Point3f pLight = ray(tMax);
    RNG rng(Hash(ray.o), Hash(ray.d));

    SampledSpectrum T_ray(1.f);
    SampledSpectrum uniPathPDF(1.f), lightPathPDF(1.f);

    while (ray.d != Vector3f(0, 0, 0)) {
        ClosestHitContext ctx(ray.medium, true);
        uint32_t p0 = packPointer0(&ctx), p1 = packPointer1(&ctx);

        PBRT_DBG("Tracing shadow tr shadow ray index %d pixel index %d "
            "ray %f %f %f d %f %f %f tMax %f\n",
            index, sr.pixelIndex, ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z,
            tMax);

        uint32_t missed = 0;

        Trace(params.traversable, ray, 1e-5f /* tMin */, tMax, OPTIX_RAY_FLAG_NONE, p0,
              p1, missed);

        if (!missed && ctx.material) {
            PBRT_DBG("Hit opaque. Bye\n");
            // Hit opaque surface
            T_ray = SampledSpectrum(0.f);
            break;
        }

        if (ray.medium) {
            PBRT_DBG("Ray medium %p. Will sample tmaj...\n", ray.medium.ptr());

            Float tEnd =
                missed ? tMax : (Distance(ray.o, Point3f(ctx.piHit)) / Length(ray.d));
            SampledSpectrum Tmaj =
                ray.medium.SampleTmaj(ray, tEnd, rng.Uniform<Float>(), rng, lambda,
                                  [&](const MediumSample &mediumSample) {
                                      const SampledSpectrum &Tmaj = mediumSample.Tmaj;
                                      const MediumInteraction &intr = mediumSample.intr;
                                      SampledSpectrum sigma_n = intr.sigma_n();

                                      // ratio-tracking: only evaluate null scattering
                                      T_ray *= Tmaj * sigma_n;
                                      lightPathPDF *= Tmaj * intr.sigma_maj;
                                      uniPathPDF *= Tmaj * sigma_n;

                                      // Possibly terminate transmittance computation using Russian roulette
                                      SampledSpectrum Tr = T_ray / (lightPathPDF + uniPathPDF).Average();
                                      if (Tr.MaxComponentValue() < 0.05f) {
                                          Float q = 0.75f;
                                          if (rng.Uniform<Float>() < q)
                                              T_ray = SampledSpectrum(0.);
                                          else {
                                              lightPathPDF *= 1 - q;
                                              uniPathPDF *= 1 - q;
                                          }
                                      }

                                      PBRT_DBG("Tmaj %f %f %f %f sigma_n %f %f %f %f sigma_maj %f %f %f %f\n",
                                               Tmaj[0], Tmaj[1], Tmaj[2], Tmaj[3],
                                               sigma_n[0], sigma_n[1], sigma_n[2], sigma_n[3],
                                               intr.sigma_maj[0], intr.sigma_maj[1], intr.sigma_maj[2],
                                               intr.sigma_maj[3]);
                                      PBRT_DBG("T_ray %f %f %f %f lightPathPDF %f %f %f %f uniPathPDF %f %f %f %f\n",
                                               T_ray[0], T_ray[1], T_ray[2], T_ray[3],
                                               lightPathPDF[0], lightPathPDF[1], lightPathPDF[2], lightPathPDF[3],
                                               uniPathPDF[0], uniPathPDF[1], uniPathPDF[2], uniPathPDF[3]);

                                      if (!T_ray)
                                          return false;

                                      rescale(T_ray, lightPathPDF, uniPathPDF);

                                      return true;
                                  });
            T_ray *= Tmaj;
            lightPathPDF *= Tmaj;
            uniPathPDF *= Tmaj;
        }

        if (missed || !T_ray)
            // done
            break;

        ray = ctx.SpawnRayTo(pLight);
    }

    PBRT_DBG("Final T_ray %.9g %.9g %.9g %.9g sr.uniPathPDF %.9g %.9g %.9g %.9g uniPathPDF %.9g %.9g %.9g %.9g\n",
             T_ray[0], T_ray[1], T_ray[2], T_ray[3],
             sr.uniPathPDF[0], sr.uniPathPDF[1], sr.uniPathPDF[2], sr.uniPathPDF[3],
             uniPathPDF[0], uniPathPDF[1], uniPathPDF[2], uniPathPDF[3]);
    PBRT_DBG("sr.lightPathPDF %.9g %.9g %.9g %.9g lightPathPDF %.9g %.9g %.9g %.9g\n",
             sr.lightPathPDF[0], sr.lightPathPDF[1], sr.lightPathPDF[2], sr.lightPathPDF[3],
             lightPathPDF[0], lightPathPDF[1], lightPathPDF[2], lightPathPDF[3]);
    PBRT_DBG("scaled throughput %.9g %.9g %.9g %.9g\n",
             T_ray[0] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average(),
             T_ray[1] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average(),
             T_ray[2] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average(),
             T_ray[3] / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average());

    if (T_ray) {
        // FIXME/reconcile: this takes lightPathPDF as input while
        // e.g. VolPathIntegrator::SampleLd() does not...
        Ld *= T_ray / (sr.uniPathPDF * uniPathPDF + sr.lightPathPDF * lightPathPDF).Average();

        PBRT_DBG("Setting final Ld for shadow ray index %d pixel index %d = as %f %f %f %f\n",
                 index, sr.pixelIndex, Ld[0], Ld[1], Ld[2], Ld[3]);

        SampledSpectrum Lpixel = params.pixelSampleState->L[sr.pixelIndex];
        params.pixelSampleState->L[sr.pixelIndex] = Lpixel + Ld;
    }
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

    ProcessClosestIntersection(intr);
}

extern "C" __global__ void __anyhit__shadowQuadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    if (rec.material && rec.material.IsTransparent())
        optixIgnoreIntersection();
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
            Float u = uint32_t(Hash(o.x, o.y, o.z, d.x, d.y, d.z)) * 0x1p-32f;
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

    ProcessClosestIntersection(intr);
}

extern "C" __global__ void __anyhit__shadowBilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    if (rec.material && rec.material.IsTransparent())
        optixIgnoreIntersection();
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
            Float u = uint32_t(Hash(o, d)) * 0x1p-32f;
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
    MaterialHandle material;
};

extern "C" __global__ void __raygen__randomHit() {
    // Keep as uint32_t so can pass directly to optixTrace.
    uint32_t index = optixGetLaunchIndex().x;
    if (index >= params.subsurfaceScatterQueue->Size())
        return;

    SubsurfaceScatterWorkItem s = (*params.subsurfaceScatterQueue)[index];

    Ray ray(s.p0, s.p1 - s.p0);
    Float tMax = 1.f;

    RandomHitPayload payload;
    payload.wrs.Seed(Hash(s.p0, s.p1));
    payload.material = s.material;

    uint32_t ptr0 = packPointer0(&payload), ptr1 = packPointer1(&payload);

    PBRT_DBG("Randomhit raygen ray.o %f %f %f ray.d %f %f %f tMax %f\n", ray.o.x, ray.o.y,
        ray.o.z, ray.d.x, ray.d.y, ray.d.z, tMax);

    Trace(params.traversable, ray, 0.f /* tMin */, tMax, OPTIX_RAY_FLAG_NONE, ptr0, ptr1);

    if (payload.wrs.HasSample() &&
        payload.wrs.WeightSum() > 0) {  // TODO: latter check shouldn't be needed...
        const SubsurfaceInteraction &si = payload.wrs.GetSample();
        PBRT_DBG("optix si p %f %f %f n %f %f %f\n", si.p().x, si.p().y, si.p().z, si.n.x,
            si.n.y, si.n.z);

        params.subsurfaceScatterQueue->weight[index] = payload.wrs.WeightSum();
        params.subsurfaceScatterQueue->ssi[index] = payload.wrs.GetSample();
    } else
        params.subsurfaceScatterQueue->weight[index] = 0;
}

extern "C" __global__ void __anyhit__randomHitTriangle() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    RandomHitPayload *p = getPayload<RandomHitPayload>();

    PBRT_DBG("Anyhit triangle for random hit: rec.material %p params.materials %p\n",
        rec.material.ptr(), p->material.ptr());

    if (rec.material == p->material)
        p->wrs.Add([&] PBRT_CPU_GPU() { return getTriangleIntersection(); }, 1.f);

    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__randomHitBilinearPatch() {
    BilinearMeshRecord &rec = *(BilinearMeshRecord *)optixGetSbtDataPointer();

    RandomHitPayload *p = getPayload<RandomHitPayload>();

    PBRT_DBG("Anyhit blp for random hit: rec.material %p params.materials %p\n",
        rec.material.ptr(), p->material.ptr());

    if (rec.material == p->material)
        p->wrs.Add(
            [&] PBRT_CPU_GPU() {
                Point2f uv(BitsToFloat(optixGetAttribute_0()),
                           BitsToFloat(optixGetAttribute_1()));
                return getBilinearPatchIntersection(uv);
            },
            1.f);

    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__randomHitQuadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    RandomHitPayload *p = getPayload<RandomHitPayload>();

    PBRT_DBG("Anyhit quadric for random hit: rec.material %p params.materials %p\n",
        rec.material.ptr(), p->material.ptr());

    if (rec.material == p->material) {
        p->wrs.Add(
            [&] PBRT_CPU_GPU() {
                QuadricIntersection qi;
                qi.pObj = Point3f(BitsToFloat(optixGetAttribute_0()),
                                  BitsToFloat(optixGetAttribute_1()),
                                  BitsToFloat(optixGetAttribute_2()));
                qi.phi = BitsToFloat(optixGetAttribute_3());

                return getQuadricIntersection(qi);
            },
            1.f);
    }

    optixIgnoreIntersection();
}
