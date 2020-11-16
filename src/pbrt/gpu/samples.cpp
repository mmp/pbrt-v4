// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/samplers.h>

#include <type_traits>

namespace pbrt {

// GPUPathIntegrator Sampler Methods
template <typename Sampler>
void GPUPathIntegrator::GenerateRaySamples(int depth, int sampleIndex) {
    std::string desc = std::string("Generate ray samples - ") + Sampler::Name();
    RayQueue *rayQueue = CurrentRayQueue(depth);

    ForAllQueued(
        desc.c_str(), rayQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(const RayWorkItem w, int index) {
            // Figure out how many dimensions have been consumed so far: 5
            // are used for the initial camera sample and then either 7 or
            // 10 per ray, depending on whether there's subsurface
            // scattering.
            int dimension = 5 + 7 * depth;
            if (haveSubsurface)
                dimension += 3 * depth;

            // Initialize a Sampler
            Sampler pixelSampler = *sampler.Cast<Sampler>();
            Point2i pPixel = pixelSampleState.pPixel[w.pixelIndex];
            pixelSampler.StartPixelSample(pPixel, sampleIndex, dimension);

            // Generate the samples for the ray and store them with it in
            // the ray queue.
            RaySamples rs;
            rs.direct.u = pixelSampler.Get2D();
            rs.direct.uc = pixelSampler.Get1D();
            rs.indirect.u = pixelSampler.Get2D();
            rs.indirect.uc = pixelSampler.Get1D();
            rs.indirect.rr = pixelSampler.Get1D();
            rs.haveSubsurface = haveSubsurface;
            if (haveSubsurface) {
                rs.subsurface.uc = pixelSampler.Get1D();
                rs.subsurface.u = pixelSampler.Get2D();
            }

            pixelSampleState.samples[w.pixelIndex] = rs;
        });
}

void GPUPathIntegrator::GenerateRaySamples(int depth, int sampleIndex) {
    auto generateSamples = [=](auto sampler) {
        using Sampler = std::remove_reference_t<decltype(*sampler)>;
        if constexpr (!std::is_same_v<Sampler, MLTSampler> &&
                      !std::is_same_v<Sampler, DebugMLTSampler>)
            GenerateRaySamples<Sampler>(depth, sampleIndex);
    };
    // Call the appropriate GenerateRaySamples specialization based on the
    // Sampler's actual type.
    sampler.DispatchCPU(generateSamples);
}

}  // namespace pbrt
