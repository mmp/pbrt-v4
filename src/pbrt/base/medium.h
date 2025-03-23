// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_MEDIUM_H
#define PBRT_BASE_MEDIUM_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>

#include <string>
#include <vector>

namespace pbrt {

// PhaseFunctionSample Definition
struct PhaseFunctionSample {
    Float p;
    Vector3f wi;
    Float pdf;
};

// PhaseFunction Definition
class HGPhaseFunction;

class PhaseFunction : public TaggedPointer<HGPhaseFunction> {
  public:
    // PhaseFunction Interface
    using TaggedPointer::TaggedPointer;

    std::string ToString() const;

    PBRT_CPU_GPU inline Float p(Vector3f wo, Vector3f wi) const;

    PBRT_CPU_GPU inline pstd::optional<PhaseFunctionSample> Sample_p(Vector3f wo,
                                                                     Point2f u) const;

    PBRT_CPU_GPU inline Float PDF(Vector3f wo, Vector3f wi) const;
};

class HomogeneousMedium;
class GridMedium;
class RGBGridMedium;
class CloudMedium;
class NanoVDBMedium;

struct MediumProperties;

// RayMajorantSegment Definition
struct RayMajorantSegment {
    Float tMin, tMax;
    SampledSpectrum sigma_maj;
    std::string ToString() const;
};

// RayMajorantIterator Definition
class HomogeneousMajorantIterator;
class DDAMajorantIterator;

class RayMajorantIterator
    : public TaggedPointer<HomogeneousMajorantIterator, DDAMajorantIterator> {
  public:
    using TaggedPointer::TaggedPointer;

    PBRT_CPU_GPU
    pstd::optional<RayMajorantSegment> Next();

    std::string ToString() const;
};

// Medium Definition
class Medium
    : public TaggedPointer<  // Medium Types
          HomogeneousMedium, GridMedium, RGBGridMedium, CloudMedium, NanoVDBMedium

          > {
  public:
    // Medium Interface
    using TaggedPointer::TaggedPointer;

    static Medium Create(const std::string &name, const ParameterDictionary &parameters,
                         const Transform &renderFromMedium, const FileLoc *loc,
                         Allocator alloc);

    std::string ToString() const;

    bool IsEmissive() const;

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const;

    // Medium Public Methods
    RayMajorantIterator SampleRay(Ray ray, Float tMax, const SampledWavelengths &lambda,
                                  ScratchBuffer &buf) const;
};

// MediumInterface Definition
struct MediumInterface {
    // MediumInterface Public Methods
    std::string ToString() const;

    MediumInterface() = default;
    PBRT_CPU_GPU
    MediumInterface(Medium medium) : inside(medium), outside(medium) {}
    PBRT_CPU_GPU
    MediumInterface(Medium inside, Medium outside) : inside(inside), outside(outside) {}

    PBRT_CPU_GPU
    bool IsMediumTransition() const { return inside != outside; }

    // MediumInterface Public Members
    Medium inside, outside;
};

}  // namespace pbrt

#endif  // PBRT_BASE_MEDIUM_H
