// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/primitive.h>

#include <pbrt/cpu/aggregates.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

Bounds3f PrimitiveHandle::Bounds() const {
    auto bounds = [&](auto ptr) { return ptr->Bounds(); };
    return DispatchCPU(bounds);
}

pstd::optional<ShapeIntersection> PrimitiveHandle::Intersect(const Ray &r,
                                                             Float tMax) const {
    auto isect = [&](auto ptr) { return ptr->Intersect(r, tMax); };
    return DispatchCPU(isect);
}

bool PrimitiveHandle::IntersectP(const Ray &r, Float tMax) const {
    auto isectp = [&](auto ptr) { return ptr->IntersectP(r, tMax); };
    return DispatchCPU(isectp);
}

// GeometricPrimitive Method Definitions
GeometricPrimitive::GeometricPrimitive(ShapeHandle shape, MaterialHandle material,
                                       LightHandle areaLight,
                                       const MediumInterface &mediumInterface,
                                       FloatTextureHandle alpha)
    : shape(shape),
      material(material),
      areaLight(areaLight),
      mediumInterface(mediumInterface),
      alpha(alpha) {
    primitiveMemory += sizeof(*this);
}

Bounds3f GeometricPrimitive::Bounds() const {
    return shape.Bounds();
}

pstd::optional<ShapeIntersection> GeometricPrimitive::Intersect(const Ray &r,
                                                                Float tMax) const {
    pstd::optional<ShapeIntersection> si = shape.Intersect(r, tMax);
    if (!si)
        return {};
    CHECK_LT(si->tHit, 1.001 * tMax);
    // Test intersection against alpha texture, if present
    if (alpha) {
        if (Float a = alpha.Evaluate(si->intr); a < 1) {
            // Potentially skip intersection, depending on stochastic alpha test
            Float u = (a <= 0) ? 1.f : (uint32_t(Hash(r.o, r.d)) * 0x1p-32f);
            if (u > a) {
                // Ignore this intersection and trace a new ray
                Ray rNext = si->intr.SpawnRay(r.d);
                pstd::optional<ShapeIntersection> siNext =
                    Intersect(rNext, tMax - si->tHit);
                if (siNext)
                    siNext->tHit += si->tHit;
                return siNext;
            }
        }
    }

    // Initialize _SurfaceInteraction_ after _Shape_ intersection
    si->intr.SetIntersectionProperties(material, areaLight, &mediumInterface, r.medium);

    return si;
}

bool GeometricPrimitive::IntersectP(const Ray &r, Float tMax) const {
    // Skip shadow intersection test for transparent materials
    if (material && material.IsTransparent())
        return false;

    // Perform shadow intersection with test, handling alpha if necessary
    if (alpha)
        return Intersect(r, tMax).has_value();
    else
        return shape.IntersectP(r, tMax);
}

// SimplePrimitive Method Definitions
SimplePrimitive::SimplePrimitive(ShapeHandle shape, MaterialHandle material)
    : shape(shape), material(material) {
    primitiveMemory += sizeof(*this);
}

Bounds3f SimplePrimitive::Bounds() const {
    return shape.Bounds();
}

bool SimplePrimitive::IntersectP(const Ray &r, Float tMax) const {
    if (material && material.IsTransparent())
        return false;
    return shape.IntersectP(r, tMax);
}

pstd::optional<ShapeIntersection> SimplePrimitive::Intersect(const Ray &r,
                                                             Float tMax) const {
    pstd::optional<ShapeIntersection> si = shape.Intersect(r, tMax);
    if (!si)
        return {};

    si->intr.SetIntersectionProperties(material, nullptr, nullptr, r.medium);

    return si;
}

// TransformedPrimitive Method Definitions
pstd::optional<ShapeIntersection> TransformedPrimitive::Intersect(const Ray &r,
                                                                  Float tMax) const {
    // Transform ray to primitive-space and intersect with primitive
    Ray ray = renderFromPrimitive->ApplyInverse(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si)
        return {};
    CHECK_LT(si->tHit, 1.001 * tMax);

    // Return transformed instance's intersection information
    si->intr = (*renderFromPrimitive)(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool TransformedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = renderFromPrimitive->ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

// AnimatedPrimitive Method Definitions
AnimatedPrimitive::AnimatedPrimitive(PrimitiveHandle p,
                                     const AnimatedTransform &renderFromPrimitive)
    : primitive(p), renderFromPrimitive(renderFromPrimitive) {
    primitiveMemory += sizeof(*this);
    CHECK(renderFromPrimitive.IsAnimated());
}

pstd::optional<ShapeIntersection> AnimatedPrimitive::Intersect(const Ray &r,
                                                               Float tMax) const {
    // Compute _ray_ after transformation by _renderFromPrimitive_
    Transform interpRenderFromPrimitive = renderFromPrimitive.Interpolate(r.time);
    Ray ray = interpRenderFromPrimitive.ApplyInverse(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si)
        return {};

    // Transform instance's intersection data to render space
    si->intr = interpRenderFromPrimitive(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool AnimatedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = renderFromPrimitive.ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

}  // namespace pbrt
