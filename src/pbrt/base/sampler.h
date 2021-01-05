// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_SAMPLER_H
#define PBRT_BASE_SAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <string>
#include <vector>

namespace pbrt {

// CameraSample Definition
struct CameraSample {
    Point2f pFilm;
    Point2f pLens;
    Float time = 0;
    Float weight = 1;
    std::string ToString() const;
};

// Sampler Declarations
class HaltonSampler;
class PaddedSobolSampler;
class PMJ02BNSampler;
class RandomSampler;
class SobolSampler;
class StratifiedSampler;
class ZSobolSampler;
class MLTSampler;
class DebugMLTSampler;

// SamplerHandle Definition
class SamplerHandle
    : public TaggedPointer<RandomSampler, StratifiedSampler, HaltonSampler,
                           PaddedSobolSampler, SobolSampler, ZSobolSampler,
                           PMJ02BNSampler, MLTSampler, DebugMLTSampler> {
  public:
    // Sampler Interface
    using TaggedPointer::TaggedPointer;

    static SamplerHandle Create(const std::string &name,
                                const ParameterDictionary &parameters,
                                Point2i fullResolution, const FileLoc *loc,
                                Allocator alloc);

    PBRT_CPU_GPU inline int SamplesPerPixel() const;

    PBRT_CPU_GPU inline void StartPixelSample(const Point2i &p, int sampleIndex,
                                              int dimension = 0);

    PBRT_CPU_GPU inline Float Get1D();
    PBRT_CPU_GPU inline Point2f Get2D();

    PBRT_CPU_GPU inline Point2f GetPixel2D();

    std::vector<SamplerHandle> Clone(int n, Allocator alloc = {});

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_SAMPLER_H
