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
    auto generateRays = [=](auto sampler) {
        using Sampler = std::remove_reference_t<decltype(*sampler)>;
        if constexpr (!std::is_same_v<Sampler, MLTSampler> &&
                      !std::is_same_v<Sampler, DebugMLTSampler>)
            GenerateCameraRays<Sampler>(y0, sampleIndex);
    };
    // Somewhat surprisingly, GenerateCameraRays() is specialized on the
    // type of the Sampler being used and not on, say, the Camera.  By
    // specializing on the sampler type, the particular Sampler used can be
    // stack allocated (rather than living in global memory), which in turn
    // allows its state to be stored in registers in the
    // GenerateCameraRays() kernel. There's little benefit from
    // specializing on the Camera since its state is read-only and shared
    // among all of the threads, so caches well in practice.
    sampler.DispatchCPU(generateRays);
}

template <typename Sampler>
void GPUPathIntegrator::GenerateCameraRays(int y0, int sampleIndex) {
    RayQueue *rayQueue = CurrentRayQueue(0);
    GPUParallelFor(
        "Generate Camera rays", maxQueueSize, PBRT_GPU_LAMBDA(int pixelIndex) {
            // Initialize _pPixel_ and test against pixel bounds
            Vector2i resolution = film.PixelBounds().Diagonal();
            Bounds2i pixelBounds = film.PixelBounds();

            Point2i pPixel(pixelBounds.pMin.x + int(pixelIndex) % resolution.x,
                           pixelBounds.pMin.y + y0 + int(pixelIndex) / resolution.x);
            pixelSampleState.pPixel[pixelIndex] = pPixel;

            // If we've split the image into multiple spans of scanlines,
            // then in the final pass, we may have a few more threads
            // launched than there are remaining pixels. Bail out without
            // enqueuing a ray if so.
            if (!InsideExclusive(pPixel, pixelBounds))
                return;

            // Initialize _Sampler_ for current pixel and sample
            Sampler pixelSampler = *sampler.Cast<Sampler>();
            pixelSampler.StartPixelSample(pPixel, sampleIndex, 0);

            // Sample wavelengths for ray path
            Float lu = RadicalInverse(1, sampleIndex) + BlueNoise(47, pPixel);
            if (lu >= 1)
                lu -= 1;
            if (GetOptions().disableWavelengthJitter)
                lu = 0.5f;
            SampledWavelengths lambda = film.SampleWavelengths(lu);

            // Compute _CameraSample_ and generate ray
            CameraSample cameraSample = GetCameraSample(pixelSampler, pPixel, filter);
            pstd::optional<CameraRay> cameraRay =
                camera.GenerateRay(cameraSample, lambda);

            // Initialize remainder of _PixelSampleState_ for ray
            // Initialize the rest of the pixel sample's state.
            pixelSampleState.L[pixelIndex] = SampledSpectrum(0.f);
            pixelSampleState.lambda[pixelIndex] = lambda;
            pixelSampleState.filterWeight[pixelIndex] = cameraSample.weight;
            if (initializeVisibleSurface)
                pixelSampleState.visibleSurface[pixelIndex] = VisibleSurface();

            // Enqueue camera ray for intersection tests
            if (cameraRay) {
                // Enqueue the camera ray if the camera gave us one with
                // non-zero weight. (RealisticCamera doesn't always return
                // a ray, e.g. in the case of vignetting...)
                rayQueue->PushCameraRay(cameraRay->ray, lambda, pixelIndex);
                pixelSampleState.cameraRayWeight[pixelIndex] = cameraRay->weight;
            } else
                pixelSampleState.cameraRayWeight[pixelIndex] = SampledSpectrum(0);
        });
}

}  // namespace pbrt
