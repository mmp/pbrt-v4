// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_FILTER_H
#define PBRT_BASE_FILTER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

// Filter Declarations
struct FilterSample;
class BoxFilter;
class GaussianFilter;
class MitchellFilter;
class LanczosSincFilter;
class TriangleFilter;

// Filter Definition
class Filter : public TaggedPointer<BoxFilter, GaussianFilter, MitchellFilter,
                                    LanczosSincFilter, TriangleFilter> {
  public:
    // Filter Interface
    using TaggedPointer::TaggedPointer;

    static Filter Create(const std::string &name, const ParameterDictionary &parameters,
                         const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU inline Vector2f Radius() const;

    PBRT_CPU_GPU inline Float Evaluate(const Point2f &p) const;

    PBRT_CPU_GPU inline FilterSample Sample(const Point2f &u) const;

    PBRT_CPU_GPU inline Float Integral() const;

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_FILTER_H
