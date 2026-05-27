// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/film.h>
#include <pbrt/wavefront/integrator.h>

namespace pbrt {

// WavefrontPathIntegrator Film Methods
void WavefrontPathIntegrator::UpdateFilm() {
    ParallelFor(
        "Update film", maxQueueSize, PBRT_CPU_GPU_LAMBDA(int pixelIndex) {
            // Check pixel against film bounds
            Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
            if (!InsideExclusive(pPixel, film.PixelBounds()))
                return;

            // Compute final weighted radiance value
            SampledSpectrum Lw = SampledSpectrum(pixelSampleState.L[pixelIndex]) *
                                 pixelSampleState.cameraRayWeight[pixelIndex];

            PBRT_DBG("Adding Lw %f %f %f %f at pixel (%d, %d)\n", Lw[0], Lw[1], Lw[2],
                     Lw[3], pPixel.x, pPixel.y);
            // Provide sample radiance value to film
            SampledWavelengths lambda = pixelSampleState.lambda[pixelIndex];
            Float filterWeight = pixelSampleState.filterWeight[pixelIndex];

#ifdef PBRT_BUILD_NRC
            // NRC milestone 2: record target RGB for the first-hit captured
            // earlier in EvaluateMaterialAndBSDF for this pixelIndex.
            if (nrcTargets != nullptr && nrcValid != nullptr &&
                nrcValid[pixelIndex]) {
                RGB rgb = film.GetPixelSensor()->ToSensorRGB(Lw, lambda);
                float *t = nrcTargets +
                           size_t(pixelIndex) * kNRCOutputDims;
                t[0] = float(rgb.r);
                t[1] = float(rgb.g);
                t[2] = float(rgb.b);
            }
#endif

            if (initializeVisibleSurface) {
                // Call _Film::AddSample()_ with _VisibleSurface_ for pixel sample
                VisibleSurface visibleSurface =
                    pixelSampleState.visibleSurface[pixelIndex];
                film.AddSample(pPixel, Lw, lambda, &visibleSurface, filterWeight);

            } else
                film.AddSample(pPixel, Lw, lambda, nullptr, filterWeight);
        });
}

}  // namespace pbrt
