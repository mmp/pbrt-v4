// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/film.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>

namespace pbrt {

// GPUPathIntegrator Film Methods
void GPUPathIntegrator::UpdateFilm() {
    GPUParallelFor(
        "Update Film", maxQueueSize, PBRT_GPU_LAMBDA(int pixelIndex) {
            // Check pixel against film bounds
            Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
            if (!InsideExclusive(pPixel, film.PixelBounds()))
                return;

            // Compute final weighted radiance value
            SampledSpectrum Lw = SampledSpectrum(pixelSampleState.L[pixelIndex]) *
                                 pixelSampleState.cameraRayWeight[pixelIndex];
            SampledWavelengths lambda = pixelSampleState.lambda[pixelIndex];
            Float filterWeight = pixelSampleState.filterWeight[pixelIndex];

            PBRT_DBG("Adding Lw %f %f %f %f at pixel (%d, %d)", Lw[0], Lw[1], Lw[2],
                     Lw[3], pPixel.x, pPixel.y);
            // Provide sample radiance value to film
            if (initializeVisibleSurface) {
                VisibleSurface visibleSurface =
                    pixelSampleState.visibleSurface[pixelIndex];
                film.AddSample(pPixel, Lw, lambda, &visibleSurface, filterWeight);
            } else
                film.AddSample(pPixel, Lw, lambda, nullptr, filterWeight);
        });
}

}  // namespace pbrt
