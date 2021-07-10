// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_OPTIONS_H
#define PBRT_OPTIONS_H

#include <pbrt/pbrt.h>
#include <pbrt/util/log.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// RenderingCoordinateSystem Definition
enum class RenderingCoordinateSystem { Camera, CameraWorld, World };
std::string ToString(const RenderingCoordinateSystem &);

// BasicPBRTOptions Definition
struct BasicPBRTOptions {
    int seed = 0;
    bool quiet = false;
    bool disablePixelJitter = false, disableWavelengthJitter = false;
    bool forceDiffuse = false;
    bool useGPU = false;
    bool wavefront = false;
    RenderingCoordinateSystem renderingSpace = RenderingCoordinateSystem::CameraWorld;
};

// PBRTOptions Definition
struct PBRTOptions : BasicPBRTOptions {
    int nThreads = 0;
    LogLevel logLevel = LogLevel::Error;
    std::string logFile;
    bool writePartialImages = false;
    bool recordPixelStatistics = false;
    bool printStatistics = false;
    pstd::optional<int> pixelSamples;
    pstd::optional<int> gpuDevice;
    bool quickRender = false;
    bool upgrade = false;
    std::string imageFile;
    std::string mseReferenceImage, mseReferenceOutput;
    std::string debugStart;
    std::string displayServer;
    pstd::optional<Bounds2f> cropWindow;
    pstd::optional<Bounds2i> pixelBounds;
    pstd::optional<Point2i> pixelMaterial;
    Float displacementEdgeScale = 1;

    std::string ToString() const;
};

// Options Global Variable Declaration
extern PBRTOptions *Options;

#if defined(PBRT_BUILD_GPU_RENDERER) && defined(__CUDACC__)
extern __constant__ BasicPBRTOptions OptionsGPU;
#endif

// Options Inline Functions
PBRT_CPU_GPU inline const BasicPBRTOptions &GetOptions();

PBRT_CPU_GPU inline const BasicPBRTOptions &GetOptions() {
#if defined(PBRT_IS_GPU_CODE)
    return OptionsGPU;
#else
    return *Options;
#endif
}

}  // namespace pbrt

#endif  // PBRT_OPTIONS_H
