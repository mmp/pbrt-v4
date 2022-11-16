// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/options.h>

#if defined(PBRT_BUILD_GPU_RENDERER)
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/util/print.h>

namespace pbrt {

PBRTOptions *Options;

#if defined(PBRT_BUILD_GPU_RENDERER)
__constant__ BasicPBRTOptions OptionsGPU;

void CopyOptionsToGPU() {
    CUDA_CHECK(cudaMemcpyToSymbol(OptionsGPU, Options, sizeof(OptionsGPU)));
}
#endif

std::string ToString(const RenderingCoordinateSystem &r) {
    if (r == RenderingCoordinateSystem::Camera)
        return "RenderingCoordinateSystem::Camera";
    else if (r == RenderingCoordinateSystem::CameraWorld)
        return "RenderingCoordinateSystem::CameraWorld";
    else {
        CHECK(r == RenderingCoordinateSystem::World);
        return "RenderingCoordinateSystem::World";
    }
}

std::string PBRTOptions::ToString() const {
    return StringPrintf(
        "[ PBRTOptions seed: %s quiet: %s disablePixelJitter: %s "
        "disableWavelengthJitter: %s disableTextureFiltering: %s disableImageTextures: %s "
        "forceDiffuse: %s useGPU: %s wavefront: %s interactive: %s fullscreen %s "
        "renderingSpace: %s nThreads: %s logLevel: %s logFile: %s logUtilization: %s "
        "writePartialImages: %s recordPixelStatistics: %s "
        "printStatistics: %s pixelSamples: %s gpuDevice: %s quickRender: %s upgrade: %s "
        "imageFile: %s mseReferenceImage: %s mseReferenceOutput: %s debugStart: %s "
        "displayServer: %s cropWindow: %s pixelBounds: %s pixelMaterial: %s "
        "displacementEdgeScale: %f ]",
        seed, quiet, disablePixelJitter, disableWavelengthJitter, disableTextureFiltering,
        disableImageTextures, forceDiffuse, useGPU, wavefront, interactive, fullscreen,
        renderingSpace, nThreads, logLevel, logFile, logUtilization, writePartialImages,
        recordPixelStatistics, printStatistics, pixelSamples, gpuDevice, quickRender, upgrade,
        imageFile, mseReferenceImage, mseReferenceOutput, debugStart, displayServer, cropWindow,
        pixelBounds, pixelMaterial, displacementEdgeScale);
}

}  // namespace pbrt
