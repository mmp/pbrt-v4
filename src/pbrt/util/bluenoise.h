// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_BLUENOISE_H
#define PBRT_UTIL_BLUENOISE_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

static constexpr int BlueNoiseResolution = 128;
static constexpr int NumBlueNoiseTextures = 48;

extern PBRT_CONST uint16_t
    BlueNoiseTextures[NumBlueNoiseTextures][BlueNoiseResolution][BlueNoiseResolution];

// Blue noise lookup functions
PBRT_CPU_GPU inline float BlueNoise(int tableIndex, Point2i p);

PBRT_CPU_GPU
inline float BlueNoise(int textureIndex, Point2i p) {
    CHECK(textureIndex >= 0 && p.x >= 0 && p.y >= 0);
    textureIndex %= NumBlueNoiseTextures;
    int x = p.x % BlueNoiseResolution, y = p.y % BlueNoiseResolution;
    return BlueNoiseTextures[textureIndex][x][y] / 65535.f;
}

}  // namespace pbrt

#endif  // PBRT_UTIL_BLUENOISE_H
