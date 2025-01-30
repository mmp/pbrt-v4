#ifndef PBRT_GPU_COMMON_H
#define PBRT_GPU_COMMON_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/shape.h>
#include <pbrt/base/texture.h>
#include <pbrt/util/pstd.h>
#include <pbrt/wavefront/workitems.h>
#include <pbrt/wavefront/workqueue.h>

#if defined(__HIPCC__)
#include <hiprt/hiprt.h>
#include <hiprt/hiprt_vec.h>
#else
#include <optix.h>
#endif

namespace pbrt {

class TriangleMesh;
class BilinearPatchMesh;

struct TriangleMeshRecord {
    const TriangleMesh *mesh;
    Material material;
    FloatTexture alphaTexture;
    pstd::span<Light> areaLights;
    MediumInterface *mediumInterface;
};

struct BilinearMeshRecord {
    const BilinearPatchMesh *mesh;
    Material material;
    FloatTexture alphaTexture;
    pstd::span<Light> areaLights;
    MediumInterface *mediumInterface;
};

struct QuadricRecord {
    Shape shape;
    Material material;
    FloatTexture alphaTexture;
    Light areaLight;
    MediumInterface *mediumInterface;
};

#if defined(__HIP_PLATFORM_AMD__)
static constexpr size_t HitgroupAlignment = 16u;

struct alignas(HitgroupAlignment) HitgroupRecord {
    PBRT_CPU_GPU HitgroupRecord() {}
    PBRT_CPU_GPU HitgroupRecord(const HitgroupRecord &r) {
        memcpy(this, &r, sizeof(HitgroupRecord));
    }
    PBRT_CPU_GPU HitgroupRecord &operator=(const HitgroupRecord &r) {
        if (this != &r)
            memcpy(this, &r, sizeof(HitgroupRecord));
        return *this;
    }

    union {
        TriangleMeshRecord triRec;
        BilinearMeshRecord blpRec;
        QuadricRecord quadricRec;
    };
    enum { TriangleMesh, BilinearMesh, Quadric } type;
};
#endif

struct RayIntersectParameters {
#if defined(__HIPCC__)
    hiprtScene traversable;
#else
    OptixTraversableHandle traversable;
#endif

    const RayQueue *rayQueue;

    // Closest hit
    RayQueue *nextRayQueue;
    EscapedRayQueue *escapedRayQueue;
    HitAreaLightQueue *hitAreaLightQueue;
    MaterialEvalQueue *basicEvalMaterialQueue, *universalEvalMaterialQueue;
    MediumSampleQueue *mediumSampleQueue;

    // Shadow rays
    ShadowRayQueue *shadowRayQueue;
    SOA<PixelSampleState> pixelSampleState;

    // Subsurface scattering...
    SubsurfaceScatterQueue *subsurfaceScatterQueue;

#if defined(__HIPCC__)
    // Stack buffers
    hiprtGlobalStackBuffer globalStackBuffer;
    hiprtGlobalStackBuffer globalInstanceStackBuffer;
    // Custom function table
    hiprtFuncTable funcTable;
    // Hitgroup records
    HitgroupRecord *hgRecords;
    // Offsets for hitgroup records
    uint32_t *offsets;
#endif
};
}  // namespace pbrt

#endif  // PBRT_GPU_COMMON_H
