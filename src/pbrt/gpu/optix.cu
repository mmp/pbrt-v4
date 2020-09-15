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
                "ray %f %f %f d %f %f %f beta %f %f %f %f\n",
                r.ray.o.x, r.ray.o.y, r.ray.o.z, r.ray.d.x, r.ray.d.y, r.ray.d.z,
                r.beta[0], r.beta[1], r.beta[2], r.beta[3]);
            params.mediumSampleQueue->Push(r.ray, Infinity, r.lambda, r.beta, r.pdfUni,
                                           r.pdfNEE, r.pixelIndex, r.piPrev,
                                           r.nPrev, r.nsPrev, r.isSpecularBounce,
                                           r.anyNonSpecularBounces, r.etaScale);
        } else if (params.escapedRayQueue) {
            PBRT_DBG("Adding ray to escapedRayQueue ray index %d pixel index %d\n", rayIndex,
                     r.pixelIndex);
            params.escapedRayQueue->Push(EscapedRayWorkItem{
                r.beta, r.pdfUni, r.pdfNEE, r.lambda, ray.o, ray.d, r.piPrev, r.nPrev,
                r.nsPrev, (int)r.isSpecularBounce, r.pixelIndex});
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
                                 r.beta,
                                 r.pdfUni,
                                 r.pdfNEE,
                                 r.pixelIndex,
                                 r.piPrev,
                                 r.nPrev,
                                 r.nsPrev,
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
        params.mediumTransitionQueue->Push(MediumTransitionWorkItem{
            newRay, r.lambda, r.beta, r.pdfUni, r.pdfNEE, r.piPrev, r.nPrev, r.nsPrev,
            r.isSpecularBounce, r.anyNonSpecularBounces, r.etaScale, r.pixelIndex});
        return;
    }

    if (intr.areaLight) {
        PBRT_DBG("Ray hit an area light: adding to hitAreaLightQueue ray index %d pixel index "
                 "%d\n", rayIndex, r.pixelIndex);
        Ray ray = r.ray;
        // TODO: intr.wo == -ray.d?
        params.hitAreaLightQueue->Push(HitAreaLightWorkItem{
            intr.areaLight, r.lambda, r.beta, r.pdfUni, r.pdfNEE, intr.p(), intr.n,
            intr.uv, intr.wo, r.piPrev, ray.d, ray.time, r.nPrev, r.nsPrev,
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
        q->Push<Material>(MaterialEvalWorkItem<Material>{
            ptr, r.lambda, r.beta, r.pdfUni, intr.pi, intr.n, intr.shading.n,
            intr.shading.dpdu, intr.shading.dpdv, intr.shading.dndu, intr.shading.dndv,
            intr.wo, intr.uv, intr.time, r.anyNonSpecularBounces, r.etaScale,
            getPayload<ClosestHitContext>()->mediumInterface, r.pixelIndex});
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
        Float u = uint32_t(Hash(o.x, o.y, o.z, d.x, d.y, d.z)) * 0x1p-32f;
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
        Ld = sr.Ld / (sr.pdfUni + sr.pdfNEE).Average();
        PBRT_DBG("Unoccluded shadow ray. Final Ld %f %f %f %f "
                 "(sr.Ld %f %f %f %f pdfUni %f %f %f %f pdfNEE %f %f %f %f)\n",
                 Ld[0], Ld[1], Ld[2], Ld[3],
                 sr.Ld[0], sr.Ld[1], sr.Ld[2], sr.Ld[3],
                 sr.pdfUni[0], sr.pdfUni[1], sr.pdfUni[2], sr.pdfUni[3],
                 sr.pdfNEE[0], sr.pdfNEE[1], sr.pdfNEE[2], sr.pdfNEE[3]);
    } else {
        PBRT_DBG("Shadow ray was occluded\n");
        Ld = SampledSpectrum(0.);
    }

    params.shadowRayQueue->Ld[index] = Ld;
}

extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(1);
}

__device__
inline void rescale(SampledSpectrum &beta, SampledSpectrum &pdfLight,
                    SampledSpectrum &pdfUni) {
    if (beta.MaxComponentValue() > 0x1p24f ||
        pdfLight.MaxComponentValue() > 0x1p24f ||
        pdfUni.MaxComponentValue() > 0x1p24f) {
        beta *= 1.f / 0x1p24f;
        pdfLight *= 1.f / 0x1p24f;
        pdfUni *= 1.f / 0x1p24f;
    } else if (beta.MaxComponentValue() < 0x1p-24f ||
               pdfLight.MaxComponentValue() < 0x1p-24f ||
               pdfUni.MaxComponentValue() < 0x1p-24f) {
        beta *= 0x1p24f;
        pdfLight *= 0x1p24f;
        pdfUni *= 0x1p24f;
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

    SampledSpectrum throughput(1.f);
    SampledSpectrum pdfUni(1.f), pdfNEE(1.f);

    while (true) {
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
            throughput = SampledSpectrum(0.f);
            break;
        }

        if (ray.medium) {
            PBRT_DBG("Ray medium %p. Will sample tmaj...\n", ray.medium.ptr());

            Float tEnd =
                missed ? tMax : (Distance(ray.o, Point3f(ctx.piHit)) / Length(ray.d));
            ray.medium.SampleTmaj(ray, tEnd, rng, lambda,
                                  [&](const MediumSample &mediumSample) {
                                      if (!mediumSample.intr)
                                          // FIXME: include last Tmaj?
                                          return false;

                                      const SampledSpectrum &Tmaj = mediumSample.Tmaj;
                                      const MediumInteraction &intr = *mediumSample.intr;
                                      SampledSpectrum sigma_n = intr.sigma_n();

                                      // ratio-tracking: only evaluate null scattering
                                      throughput *= Tmaj * sigma_n;
                                      pdfNEE *= Tmaj * intr.sigma_maj;
                                      pdfUni *= Tmaj * sigma_n;

                                      Float pSurvive = throughput.MaxComponentValue() /
                                          (pdfNEE + pdfUni).Average();

                                      PBRT_DBG("throughput %f %f %f %f pdfNEE %f %f %f %f pdfUni %f %f %f %f "
                                               "pSurvive %f\n",
                                               throughput[0], throughput[1], throughput[2], throughput[3],
                                               pdfNEE[0], pdfNEE[1], pdfNEE[2], pdfNEE[3],
                                               pdfUni[0], pdfUni[1], pdfUni[2], pdfUni[3], pSurvive);

                                      if (pSurvive < .25f) {
                                          if (rng.Uniform<Float>() > pSurvive)
                                              throughput = SampledSpectrum(0.);
                                          else
                                              throughput /= pSurvive;
                                      }

                                      PBRT_DBG("Tmaj %f %f %f %f sigma_n %f %f %f %f sigma_maj %f %f %f %f\n",
                                               Tmaj[0], Tmaj[1], Tmaj[2], Tmaj[3],
                                               sigma_n[0], sigma_n[1], sigma_n[2], sigma_n[3],
                                               intr.sigma_maj[0], intr.sigma_maj[1], intr.sigma_maj[2],
                                               intr.sigma_maj[3]);
                                      PBRT_DBG("throughput %f %f %f %f pdfNEE %f %f %f %f pdfUni %f %f %f %f\n",
                                               throughput[0], throughput[1], throughput[2], throughput[3],
                                               pdfNEE[0], pdfNEE[1], pdfNEE[2], pdfNEE[3],
                                               pdfUni[0], pdfUni[1], pdfUni[2], pdfUni[3]);

                                      if (!throughput)
                                          return false;

                                      rescale(throughput, pdfNEE, pdfUni);

                                      return true;
                                  });
        }

        if (missed || !throughput)
            // done
            break;

        ray = ctx.SpawnRayTo(pLight);

        if (ray.d == Vector3f(0, 0, 0))
            break;
    }

    PBRT_DBG("Final throughput %.9g %.9g %.9g %.9g sr.pdfUni %.9g %.9g %.9g %.9g pdfUni %.9g %.9g %.9g %.9g\n",
             throughput[0], throughput[1], throughput[2], throughput[3],
             sr.pdfUni[0], sr.pdfUni[1], sr.pdfUni[2], sr.pdfUni[3],
             pdfUni[0], pdfUni[1], pdfUni[2], pdfUni[3]);
    PBRT_DBG("sr.pdfNEE %.9g %.9g %.9g %.9g pdfNEE %.9g %.9g %.9g %.9g\n",
             sr.pdfNEE[0], sr.pdfNEE[1], sr.pdfNEE[2], sr.pdfNEE[3],
             pdfNEE[0], pdfNEE[1], pdfNEE[2], pdfNEE[3]);
    PBRT_DBG("scaled throughput %.9g %.9g %.9g %.9g\n",
             throughput[0] / (sr.pdfUni * pdfUni + sr.pdfNEE * pdfNEE).Average(),
             throughput[1] / (sr.pdfUni * pdfUni + sr.pdfNEE * pdfNEE).Average(),
             throughput[2] / (sr.pdfUni * pdfUni + sr.pdfNEE * pdfNEE).Average(),
             throughput[3] / (sr.pdfUni * pdfUni + sr.pdfNEE * pdfNEE).Average());

    if (!throughput)
        Ld = SampledSpectrum(0.f);
    else
        Ld *= throughput / (sr.pdfUni * pdfUni + sr.pdfNEE * pdfNEE).Average();

    PBRT_DBG("Setting final Ld for shadow ray index %d pixel index %d = as %f %f %f %f\n",
        index, sr.pixelIndex, Ld[0], Ld[1], Ld[2], Ld[3]);

    params.shadowRayQueue->Ld[index] = Ld;
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
            Float u = uint32_t(Hash(o.x, o.y, o.z, d.x, d.y, d.z)) * 0x1p-32f;
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
