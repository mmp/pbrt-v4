// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/samplers.h>

#include <type_traits>

namespace pbrt {

// GPUPathIntegrator Sampler Methods
void GPUPathIntegrator::GenerateRaySamples(int depth, int sampleIndex) {
    auto generateSamples = [=](auto sampler) {
        using Sampler = std::remove_reference_t<decltype(*sampler)>;
        if constexpr (!std::is_same_v<Sampler, MLTSampler> &&
                      !std::is_same_v<Sampler, DebugMLTSampler>)
            GenerateRaySamples<Sampler>(depth, sampleIndex);
    };
    sampler.DispatchCPU(generateSamples);
}

template <typename Sampler>
void GPUPathIntegrator::GenerateRaySamples(int depth, int sampleIndex) {
    // Generate description string _desc_ for ray sample generation
    std::string desc = std::string("Generate ray samples - ") + Sampler::Name();

    RayQueue *rayQueue = CurrentRayQueue(depth);
    ForAllQueued(
        desc.c_str(), rayQueue, maxQueueSize, PBRT_GPU_LAMBDA(const RayWorkItem w) {
            // Generate samples for ray segment at current sample index
            // Find first sample dimension
            int dimension = 5 + 7 * depth;
            if (haveSubsurface)
                dimension += 3 * depth;
            if (haveMedia)
                dimension += 2 * depth;

            // Initialize _Sampler_ for pixel, sample index, and dimension
            Sampler pixelSampler = *sampler.Cast<Sampler>();
            Point2i pPixel = pixelSampleState.pPixel[w.pixelIndex];
            pixelSampler.StartPixelSample(pPixel, sampleIndex, dimension);

            // Initialize _RaySamples_ structure with sample values
            RaySamples rs;
            rs.direct.u = pixelSampler.Get2D();
            rs.direct.uc = pixelSampler.Get1D();
            // Initialize indirect and possibly subsurface samples in _rs_
            rs.indirect.u = pixelSampler.Get2D();
            rs.indirect.uc = pixelSampler.Get1D();
            rs.indirect.rr = pixelSampler.Get1D();
            rs.haveSubsurface = haveSubsurface;
            if (haveSubsurface) {
                rs.subsurface.uc = pixelSampler.Get1D();
                rs.subsurface.u = pixelSampler.Get2D();
            }
            rs.haveMedia = haveMedia;
            if (haveMedia) {
                rs.media.uDist = pixelSampler.Get1D();
                rs.media.uMode = pixelSampler.Get1D();
            }

            // Store _RaySamples_ in pixel sample state
            pixelSampleState.samples[w.pixelIndex] = rs;
        });
}

}  // namespace pbrt
