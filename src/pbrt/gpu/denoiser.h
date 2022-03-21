
#ifndef PBRT_GPU_DENOISER_H
#define PBRT_GPU_DENOISER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/color.h>
#include <pbrt/util/vecmath.h>

#include <optix.h>

namespace pbrt {

class Denoiser {
 public:
    Denoiser(Vector2i resolution, bool haveAlbedoAndNormal);

    void Denoise(RGB *rgb, Normal3f *n, RGB *albedo, RGB *result);

 private:
    Vector2i resolution;
    bool haveAlbedoAndNormal;
    OptixDenoiser denoiserHandle;
    OptixDenoiserSizes memorySizes;
    void *denoiserState, *scratchBuffer, *intensity;
};

} // namespace pbrt

#endif // PBRT_GPU_DENOISER_H
