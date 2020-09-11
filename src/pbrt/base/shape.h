// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_SHAPE_H
#define PBRT_BASE_SHAPE_H

#include <pbrt/pbrt.h>

#include <pbrt/util/buffercache.h>
#include <pbrt/util/float.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// Shape Declarations
class Triangle;
class BilinearPatch;
class Curve;
class Sphere;
class Cylinder;
class Disk;

struct ShapeSample;
struct ShapeIntersection;
struct ShapeSampleContext;

// ShapeHandle Definition
class ShapeHandle
    : public TaggedPointer<Sphere, Cylinder, Disk, Triangle, BilinearPatch, Curve> {
  public:
    // Shape Interface
    using TaggedPointer::TaggedPointer;

    static pstd::vector<ShapeHandle> Create(const std::string &name,
                                            const Transform *renderFromObject,
                                            const Transform *objectFromRender,
                                            bool reverseOrientation,
                                            const ParameterDictionary &parameters,
                                            const FileLoc *loc, Allocator alloc);
    std::string ToString() const;

    PBRT_CPU_GPU inline Bounds3f Bounds() const;

    PBRT_CPU_GPU inline DirectionCone NormalBounds() const;

    PBRT_CPU_GPU inline pstd::optional<ShapeIntersection> Intersect(
        const Ray &ray, Float tMax = Infinity) const;

    PBRT_CPU_GPU inline bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    PBRT_CPU_GPU inline Float Area() const;

    PBRT_CPU_GPU inline pstd::optional<ShapeSample> Sample(const Point2f &u) const;

    PBRT_CPU_GPU inline Float PDF(const Interaction &) const;

    PBRT_CPU_GPU inline pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx,
                                                           const Point2f &u) const;

    PBRT_CPU_GPU inline Float PDF(const ShapeSampleContext &ctx,
                                  const Vector3f &wi) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_SHAPE_H
