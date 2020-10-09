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
    LightHandle light;
    Float pdf = 0;
    std::string ToString() const;
};

class UniformLightSampler;
class PowerLightSampler;
class BVHLightSampler;
class ExhaustiveLightSampler;

// LightSamplerHandle Definition
class LightSamplerHandle : public TaggedPointer<UniformLightSampler, PowerLightSampler,
                                                BVHLightSampler, ExhaustiveLightSampler> {
  public:
    // LightSampler Interface
    using TaggedPointer::TaggedPointer;

    static LightSamplerHandle Create(const std::string &name,
                                     pstd::span<const LightHandle> lights,
                                     Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(const LightSampleContext &ctx,
                                                            Float u) const;

    PBRT_CPU_GPU inline Float PDF(const LightSampleContext &ctx, LightHandle light) const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(Float u) const;
    PBRT_CPU_GPU inline Float PDF(LightHandle light) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHTSAMPLER_H
