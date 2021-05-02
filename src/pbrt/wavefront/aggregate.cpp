// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/aggregate.h>

#include <pbrt/cpu/aggregates.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/parsedscene.h>
#include <pbrt/textures.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/log.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/stats.h>
#include <pbrt/wavefront/intersect.h>

namespace pbrt {

// FIXME: copied in gpu/aggregate.cpp
static void updateMaterialNeeds(Material m, pstd::array<bool, Material::NumTags()> *haveBasicEvalMaterial,
                                pstd::array<bool, Material::NumTags()> *haveUniversalEvalMaterial,
                                bool *haveSubsurface) {
    if (!m)
        return;

    if (MixMaterial *mix = m.CastOrNullptr<MixMaterial>(); mix) {
        updateMaterialNeeds(mix->GetMaterial(0), haveBasicEvalMaterial, haveUniversalEvalMaterial,
                            haveSubsurface);
        updateMaterialNeeds(mix->GetMaterial(1), haveBasicEvalMaterial, haveUniversalEvalMaterial,
                            haveSubsurface);
        return;
    }

    *haveSubsurface |= m.HasSubsurfaceScattering();

    FloatTexture displace = m.GetDisplacement();
    if (m.CanEvaluateTextures(BasicTextureEvaluator()) &&
        (!displace || BasicTextureEvaluator().CanEvaluate({displace}, {})))
        (*haveBasicEvalMaterial)[m.Tag()] = true;
    else
        (*haveUniversalEvalMaterial)[m.Tag()] = true;
}

CPUAggregate::CPUAggregate(ParsedScene &scene, Allocator alloc,
         NamedTextures &textures,
         const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
         const std::map<std::string, Medium> &media,
         pstd::array<bool, Material::NumTags()> *haveBasicEvalMaterial,
         pstd::array<bool, Material::NumTags()> *haveUniversalEvalMaterial,
         bool *haveSubsurface) {
    ParsedScene::Scene s = scene.CreateAggregate(alloc, textures, shapeIndexToAreaLights, media);

    aggregate = s.aggregate;

    for (Material m : s.materials)
        updateMaterialNeeds(m, haveBasicEvalMaterial, haveUniversalEvalMaterial,
                            haveSubsurface);
    for (const auto &m : s.namedMaterials)
        updateMaterialNeeds(m.second, haveBasicEvalMaterial, haveUniversalEvalMaterial,
                            haveSubsurface);
}

void CPUAggregate::IntersectClosest(int maxRays, EscapedRayQueue *escapedRayQueue,
                                HitAreaLightQueue *hitAreaLightQueue,
                                MaterialEvalQueue *basicEvalMaterialQueue,
                                MaterialEvalQueue *universalEvalMaterialQueue,
                                MediumSampleQueue *mediumSampleQueue,
                                RayQueue *rayQueue, RayQueue *nextRayQueue) const {
    ParallelFor(0, rayQueue->Size(),
                [=](int index) {
                    const RayWorkItem r = (*rayQueue)[index];
                    pstd::optional<ShapeIntersection> si = aggregate.Intersect(r.ray, Infinity);
                    if (!si) {
                        EnqueueWorkAfterMiss(r, mediumSampleQueue, escapedRayQueue);
                    } else {
                        const SurfaceInteraction &intr = si->intr;
                        Float tHit = Distance(r.ray.o, intr.p()) / Length(r.ray.d);
                        MediumInterface mediumInterface = intr.mediumInterface ? *intr.mediumInterface :
                            MediumInterface(r.ray.medium);

                        EnqueueWorkAfterIntersection(r, r.ray.medium /* FIXME DIFF THAN OPTIX WOT */, tHit,
                                                     si->intr, mediumSampleQueue,
                                                     nextRayQueue, hitAreaLightQueue,
                                                     basicEvalMaterialQueue, universalEvalMaterialQueue,
                                                     mediumInterface);
                    }
                });
}

void CPUAggregate::IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                               SOA<PixelSampleState> *pixelSampleState) const {
    ParallelFor(0, shadowRayQueue->Size(),
                [=](int index) {
                    const ShadowRayWorkItem w = (*shadowRayQueue)[index];
                    RecordShadowRayIntersection(w, pixelSampleState,
                                                aggregate.IntersectP(w.ray, w.tMax));
                });
}

void CPUAggregate::IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                                 SOA<PixelSampleState> *pixelSampleState) const {
    ParallelFor(0, shadowRayQueue->Size(),
                [=](int index) {
                    const ShadowRayWorkItem w = (*shadowRayQueue)[index];
                    pstd::optional<ShapeIntersection> si;
                    TraceTransmittance(w, pixelSampleState,
                                       [&](Ray ray, Float tMax) -> TransmittanceTraceResult {
                                           si = aggregate.Intersect(ray, tMax);

                                           if (!si)
                                               return TransmittanceTraceResult{false, Point3f(), Material()};
                                           else
                                               return TransmittanceTraceResult{true, si->intr.p(), si->intr.material};
                                       },
                                       [&](Point3f p) -> Ray {
                                           return si->intr.SpawnRayTo(p);
                                       });
                });
}

void CPUAggregate::IntersectOneRandom(int maxRays, SubsurfaceScatterQueue *subsurfaceScatterQueue) const {
    ParallelFor(0, subsurfaceScatterQueue->Size(),
                [=](int index) {
                    const SubsurfaceScatterWorkItem &w = (*subsurfaceScatterQueue)[index];
                    uint64_t seed = Hash(w.p0, w.p1);

                    WeightedReservoirSampler<SubsurfaceInteraction> wrs(seed);
                    Interaction base(w.p0, 0.f /* FIXME time */, Medium());
                    while (true) {
                        Ray r = base.SpawnRayTo(w.p1);
                        if (r.d == Vector3f(0, 0, 0))
                            break;
                        pstd::optional<ShapeIntersection> si = aggregate.Intersect(r, 1);
                        if (!si)
                            break;
                        base = si->intr;
                        if (si->intr.material == w.material)
                            wrs.Add(SubsurfaceInteraction(si->intr), 1.f);
                    }

                    if (wrs.HasSample()) {
                        subsurfaceScatterQueue->reservoirPDF[index] = wrs.SamplePDF();
                        subsurfaceScatterQueue->ssi[index] = wrs.GetSample();
                    } else
                        subsurfaceScatterQueue->reservoirPDF[index] = 0;
                });
}

}  // namespace pbrt
