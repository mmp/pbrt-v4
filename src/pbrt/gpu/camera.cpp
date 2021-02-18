// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/cameras.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/options.h>
#include <pbrt/samplers.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

// GPUPathIntegrator Camera Ray Methods
void GPUPathIntegrator::GenerateCameraRays(int y0, int sampleIndex) {
    // Define _generateRays_ lambda function
    auto generateRays = [=](auto sampler) {
        using Sampler = std::remove_reference_t<decltype(*sampler)>;
        if constexpr (!std::is_same_v<Sampler, MLTSampler> &&
                      !std::is_same_v<Sampler, DebugMLTSampler>)
            GenerateCameraRays<Sampler>(y0, sampleIndex);
    };

    sampler.DispatchCPU(generateRays);
}

template <typename Sampler>
void GPUPathIntegrator::GenerateCameraRays(int y0, int sampleIndex) {
    RayQueue *rayQueue = CurrentRayQueue(0);
    GPUParallelFor(
        "Generate Camera rays", maxQueueSize, PBRT_GPU_LAMBDA(int pixelIndex) {
            // Enqueue camera ray and set pixel state for sample
            // Compute pixel coordinates for _pixelIndex_
            Bounds2i pixelBounds = film.PixelBounds();
            int xResolution = pixelBounds.pMax.x - pixelBounds.pMin.x;
            Point2i pPixel(pixelBounds.pMin.x + pixelIndex % xResolution,
                           y0 + pixelIndex / xResolution);
            pixelSampleState.pPixel[pixelIndex] = pPixel;

            // Test pixel coordinates against pixel bounds
            if (!InsideExclusive(pPixel, pixelBounds))
                return;

            // Initialize _Sampler_ for current pixel and sample
            Sampler pixelSampler = *sampler.Cast<Sampler>();
            pixelSampler.StartPixelSample(pPixel, sampleIndex, 0);

            // Sample wavelengths for ray path
            Float lu = pixelSampler.Get1D();
            if (GetOptions().disableWavelengthJitter)
                lu = 0.5f;
            SampledWavelengths lambda = film.SampleWavelengths(lu);

            // Compute _CameraSample_ and generate ray
            CameraSample cameraSample = GetCameraSample(pixelSampler, pPixel, filter);
            pstd::optional<CameraRay> cameraRay =
                camera.GenerateRay(cameraSample, lambda);

            // Initialize remainder of _PixelSampleState_ for ray
            pixelSampleState.L[pixelIndex] = SampledSpectrum(0.f);
            pixelSampleState.lambda[pixelIndex] = lambda;
            pixelSampleState.filterWeight[pixelIndex] = cameraSample.filterWeight;
            if (initializeVisibleSurface)
                pixelSampleState.visibleSurface[pixelIndex] = VisibleSurface();

            // Enqueue camera ray for intersection tests
            if (cameraRay) {
                rayQueue->PushCameraRay(cameraRay->ray, lambda, pixelIndex);
                pixelSampleState.cameraRayWeight[pixelIndex] = cameraRay->weight;
            } else
                pixelSampleState.cameraRayWeight[pixelIndex] = SampledSpectrum(0);
        });
}

}  // namespace pbrt
