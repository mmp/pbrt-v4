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

std::string PBRTOptions::ToString() const {
    return StringPrintf(
        "[ PBRTOptions nThreads: %d seed: %d quickRender: %s quiet: %s "
        "recordPixelStatistics: %s upgrade: %s disablePixelJitter: %s "
        "disableWavelengthJitter: %s forceDiffuse: %s useGPU: %s "
        "imageFile: %s mseReferenceImage: %s mseReferenceOutput: %s "
        "debugStart: %s displayServer: %s cropWindow: %s pixelBounds: %s ]",
        nThreads, seed, quickRender, quiet, recordPixelStatistics, upgrade,
        disablePixelJitter, disableWavelengthJitter, forceDiffuse, useGPU, imageFile,
        mseReferenceImage, mseReferenceOutput, debugStart, displayServer, cropWindow,
        pixelBounds);
}

}  // namespace pbrt
