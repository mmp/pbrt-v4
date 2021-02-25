// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_LIGHTSAMPLER_H
#define PBRT_BASE_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

// SampledLight Definition
struct SampledLight {
    Light light;
    Float pdf = 0;
    std::string ToString() const;
};

class UniformLightSampler;
class PowerLightSampler;
class BVHLightSampler;
class ExhaustiveLightSampler;

// LightSampler Definition
class LightSampler : public TaggedPointer<UniformLightSampler, PowerLightSampler,
                                          ExhaustiveLightSampler, BVHLightSampler> {
  public:
    // LightSampler Interface
    using TaggedPointer::TaggedPointer;

    static LightSampler Create(const std::string &name, pstd::span<const Light> lights,
                               Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(const LightSampleContext &ctx,
                                                            Float u) const;

    PBRT_CPU_GPU inline Float PDF(const LightSampleContext &ctx, Light light) const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(Float u) const;
    PBRT_CPU_GPU inline Float PDF(Light light) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHTSAMPLER_H
