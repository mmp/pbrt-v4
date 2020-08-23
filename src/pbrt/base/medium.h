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
    PBRT_CPU_GPU operator bool() const { return pdf > 0; }
    Float p;
    Vector3f wi;
    Float pdf = 0;
};

// PhaseFunctionHandle Definition
class HGPhaseFunction;

class PhaseFunctionHandle : public TaggedPointer<HGPhaseFunction> {
  public:
    // PhaseFunctionHandle Interface
    using TaggedPointer::TaggedPointer;

    std::string ToString() const;

    PBRT_CPU_GPU inline Float p(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_CPU_GPU inline PhaseFunctionSample Sample_p(const Vector3f &wo,
                                                     const Point2f &u) const;
    PBRT_CPU_GPU inline Float PDF(const Vector3f &wo, const Vector3f &wi) const;
};

class HomogeneousMedium;
template <typename DensityProvider>
class GeneralMedium;
class UniformGridMediumProvider;
using UniformGridMedium = GeneralMedium<UniformGridMediumProvider>;
struct MediumSample;

// MediumHandle Definition
class MediumHandle : public TaggedPointer<HomogeneousMedium, UniformGridMedium> {
  public:
    // MediumHandle Interface
    using TaggedPointer::TaggedPointer;

    static MediumHandle Create(const std::string &name,
                               const ParameterDictionary &parameters,
                               const Transform &renderFromMedium, const FileLoc *loc,
                               Allocator alloc);

    std::string ToString() const;

    template <typename F>
    PBRT_CPU_GPU void SampleTmaj(const Ray &ray, Float tMax, RNG &rng,
                                 const SampledWavelengths &lambda, F callback) const;

    bool IsEmissive() const;
};

// MediumInterface Definition
struct MediumInterface {
    // MediumInterface Public Methods
    std::string ToString() const;

    MediumInterface() = default;
    PBRT_CPU_GPU
    MediumInterface(MediumHandle medium) : inside(medium), outside(medium) {}
    PBRT_CPU_GPU
    MediumInterface(MediumHandle inside, MediumHandle outside)
        : inside(inside), outside(outside) {}

    PBRT_CPU_GPU
    bool IsMediumTransition() const { return inside != outside; }

    // MediumInterface Public Members
    MediumHandle inside, outside;
};

}  // namespace pbrt

#endif  // PBRT_BASE_MEDIUM_H
