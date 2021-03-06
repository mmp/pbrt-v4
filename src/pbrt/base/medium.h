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
template <typename Provider>
class CuboidMedium;
class UniformGridMediumProvider;
// UniformGridMedium Definition
using UniformGridMedium = CuboidMedium<UniformGridMediumProvider>;

class CloudMediumProvider;
// CloudMedium Definition
using CloudMedium = CuboidMedium<CloudMediumProvider>;

class NanoVDBMediumProvider;
// NanoVDBMedium Definition
using NanoVDBMedium = CuboidMedium<NanoVDBMediumProvider>;

struct MediumProperties;
struct MediumSample;

// MediumDensity Definition
struct MediumDensity {
    PBRT_CPU_GPU
    MediumDensity(Float d) : sigma_a(d), sigma_s(d) {}
    PBRT_CPU_GPU
    MediumDensity(SampledSpectrum sigma_a, SampledSpectrum sigma_s)
        : sigma_a(sigma_a), sigma_s(sigma_s) {}
    SampledSpectrum sigma_a, sigma_s;
};

// Medium Definition
class Medium : public TaggedPointer<HomogeneousMedium, UniformGridMedium, CloudMedium,
                                    NanoVDBMedium> {
  public:
    // Medium Interface
    using TaggedPointer::TaggedPointer;

    static Medium Create(const std::string &name, const ParameterDictionary &parameters,
                         const Transform &renderFromMedium, const FileLoc *loc,
                         Allocator alloc);

    std::string ToString() const;

    bool IsEmissive() const;

    PBRT_CPU_GPU
    MediumProperties Sample(Point3f p, const SampledWavelengths &lambda) const;

    template <typename F>
    PBRT_CPU_GPU SampledSpectrum SampleT_maj(Ray ray, Float tMax, Float u, RNG &rng,
                                             const SampledWavelengths &lambda,
                                             F callback) const;
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
