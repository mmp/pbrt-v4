// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/options.h>

#include <pbrt/util/print.h>

namespace pbrt {

PBRTOptions *Options;

#if defined(PBRT_BUILD_GPU_RENDERER)
__constant__ BasicPBRTOptions OptionsGPU;
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
        "disableWavelengthJitter: %s "
        "forceDiffuse: %s useGPU: %s wavefront: %s renderingSpace: %s nThreads: %s "
        "logLevel: %s logFile: %s writePartialImages: %s recordPixelStatistics: %s "
        "printStatistics: %s pixelSamples: %s gpuDevice: %s quickRender: %s upgrade: %s "
        "imageFile: %s mseReferenceImage: %s mseReferenceOutput: %s debugStart: %s "
        "displayServer: %s cropWindow: %s pixelBounds: %s pixelMaterial: %s ]",
        seed, quiet, disablePixelJitter, disableWavelengthJitter, forceDiffuse, useGPU,
        wavefront, renderingSpace, nThreads, logLevel, logFile, writePartialImages,
        recordPixelStatistics, printStatistics, pixelSamples, gpuDevice, quickRender,
        upgrade, imageFile, mseReferenceImage, mseReferenceOutput, debugStart,
        displayServer, cropWindow, pixelBounds, pixelMaterial);
}

}  // namespace pbrt
