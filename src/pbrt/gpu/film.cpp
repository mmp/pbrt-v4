// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/film.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>

#ifdef PBRT_GPU_DBG
#ifndef TO_STRING
#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x
#endif  // !TO_STRING
#define DBG(...) printf(__FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#else
#define DBG(...)
#endif

namespace pbrt {

void GPUPathIntegrator::UpdateFilm() {
    GPUParallelFor("Update Film", maxQueueSize, [=] PBRT_GPU(int pixelIndex) {
        Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
        if (!InsideExclusive(pPixel, film.PixelBounds()))
            return;

        // Compute final weighted radiance value
        SampledSpectrum Lw = SampledSpectrum(pixelSampleState.L[pixelIndex]) *
                             pixelSampleState.cameraRayWeight[pixelIndex];

        SampledWavelengths lambda = pixelSampleState.lambda[pixelIndex];
        Float filterWeight = pixelSampleState.filterWeight[pixelIndex];

        if (initializeVisibleSurface) {
            VisibleSurface visibleSurface = pixelSampleState.visibleSurface[pixelIndex];
            film.AddSample(pPixel, Lw, lambda, &visibleSurface, filterWeight);
        } else
            film.AddSample(pPixel, Lw, lambda, nullptr, filterWeight);
    });
}

}  // namespace pbrt
