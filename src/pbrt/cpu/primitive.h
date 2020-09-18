// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_PRIMITIVE_H
#define PBRT_CPU_PRIMITIVE_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/shape.h>
#include <pbrt/base/texture.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>

#include <memory>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Primitives", primitiveMemory);

class SimplePrimitive;
class GeometricPrimitive;
class TransformedPrimitive;
class AnimatedPrimitive;
class BVHAggregate;
class KdTreeAggregate;

// PrimitiveHandle Definition
class PrimitiveHandle
    : public TaggedPointer<SimplePrimitive, GeometricPrimitive, TransformedPrimitive,
                           AnimatedPrimitive, BVHAggregate, KdTreeAggregate> {
  public:
    // Primitive Interface
    using TaggedPointer::TaggedPointer;

    Bounds3f Bounds() const;

    pstd::optional<ShapeIntersection> Intersect(const Ray &r,
                                                Float tMax = Infinity) const;
    bool IntersectP(const Ray &r, Float tMax = Infinity) const;
};

// GeometricPrimitive Definition
class GeometricPrimitive {
  public:
    // GeometricPrimitive Public Methods
    GeometricPrimitive(ShapeHandle shape, MaterialHandle material, LightHandle areaLight,
                       const MediumInterface &mediumInterface,
                       FloatTextureHandle alpha = nullptr);
    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;

  private:
    // GeometricPrimitive Private Members
    ShapeHandle shape;
    MaterialHandle material;
    LightHandle areaLight;
    MediumInterface mediumInterface;
    FloatTextureHandle alpha;
};

// SimplePrimitive Definition
class SimplePrimitive {
  public:
    // SimplePrimitive Public Methods
    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    SimplePrimitive(ShapeHandle shape, MaterialHandle material);

  private:
    // SimplePrimitive Private Members
    ShapeHandle shape;
    MaterialHandle material;
};

// TransformedPrimitive Definition
class TransformedPrimitive {
  public:
    // TransformedPrimitive Public Methods
    TransformedPrimitive(PrimitiveHandle primitive, const Transform *renderFromPrimitive)
        : primitive(primitive), renderFromPrimitive(renderFromPrimitive) {
        primitiveMemory += sizeof(*this);
    }

    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;

    Bounds3f Bounds() const { return (*renderFromPrimitive)(primitive.Bounds()); }

  private:
    // TransformedPrimitive Private Members
    PrimitiveHandle primitive;
    const Transform *renderFromPrimitive;
};

// AnimatedPrimitive Definition
class AnimatedPrimitive {
  public:
    // AnimatedPrimitive Public Methods
    Bounds3f Bounds() const {
        return renderFromPrimitive.MotionBounds(primitive.Bounds());
    }

    AnimatedPrimitive(PrimitiveHandle primitive,
                      const AnimatedTransform &renderFromPrimitive);
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;

  private:
    // AnimatedPrimitive Private Members
    PrimitiveHandle primitive;
    AnimatedTransform renderFromPrimitive;
};

}  // namespace pbrt

#endif  // PBRT_CPU_PRIMITIVE_H
