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

#ifdef PBRT_GPU_DBG
#ifndef TO_STRING
#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x
#endif  // !TO_STRING
#define DBG(...) printf(__FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#else
#define DBG(...)
#endif  // PBRT_GPU_DBG

namespace pbrt {

template <typename Sampler>
void GPUPathIntegrator::GenerateCameraRays(int y0, int sampleIndex) {
    Vector2i resolution = film.PixelBounds().Diagonal();
    Bounds2i pixelBounds = film.PixelBounds();

    GPUParallelFor("Generate Camera rays", maxQueueSize, [=] PBRT_GPU(int pixelIndex) {
        Point2i pPixel(pixelBounds.pMin.x + int(pixelIndex) % resolution.x,
                       pixelBounds.pMin.y + y0 + int(pixelIndex) / resolution.x);
        pixelSampleState.pPixel[pixelIndex] = pPixel;

        // If we've split the image into multiple spans of scanlines,
        // then in the final pass, we may have a few more threads
        // launched than there are remaining pixels. Bail out without
        // enqueuing a ray if so.
        if (!InsideExclusive(pPixel, pixelBounds))
            return;

        // Initialize the Sampler for the current pixel and sample.
        Sampler sampler = *this->sampler.Cast<Sampler>();
        sampler.StartPixelSample(pPixel, sampleIndex, 0);

        // Sample wavelengths for the ray path for the pixel sample.
        // Use a blue noise pattern rather than the Sampler.
        Float lu = RadicalInverse(1, sampleIndex) + BlueNoise(47, pPixel.x, pPixel.y);
        if (lu >= 1)
            lu -= 1;
        if (GetOptions().disableWavelengthJitter)
            lu = 0.5f;
        SampledWavelengths lambda = film.SampleWavelengths(lu);

        // Generate samples for the camera ray and the ray itself.
        CameraSample cameraSample = GetCameraSample(sampler, pPixel, filter);
        CameraRay cameraRay = camera.GenerateRay(cameraSample, lambda);

        // Initialize the rest of the pixel sample's state.
        pixelSampleState.L[pixelIndex] = SampledSpectrum(0.f);
        pixelSampleState.lambda[pixelIndex] = lambda;
        pixelSampleState.cameraRayWeight[pixelIndex] = cameraRay.weight;
        pixelSampleState.filterWeight[pixelIndex] = cameraSample.weight;
        if (initializeVisibleSurface)
            pixelSampleState.visibleSurface[pixelIndex] = VisibleSurface();

        if (cameraRay.weight)
            // Enqueue the camera ray if the camera gave us one with
            // non-zero weight. (RealisticCamera doesn't always return
            // a ray, e.g. in the case of vignetting...)
            rayQueues[0]->PushCameraRay(cameraRay.ray, lambda, pixelIndex);
    });
}

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

}  // namespace pbrt
