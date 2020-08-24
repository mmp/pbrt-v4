// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_NOISE_H
#define PBRT_UTIL_NOISE_H

#include <pbrt/pbrt.h>

namespace pbrt {

PBRT_CPU_GPU
Float Noise(Float x, Float y = .5f, Float z = .5f);
PBRT_CPU_GPU
Float Noise(const Point3f &p);
PBRT_CPU_GPU
Float FBm(const Point3f &p, const Vector3f &dpdx, const Vector3f &dpdy, Float omega,
          int octaves);
PBRT_CPU_GPU
Float Turbulence(const Point3f &p, const Vector3f &dpdx, const Vector3f &dpdy,
                 Float omega, int octaves);

}  // namespace pbrt

#endif  // PBRT_UTIL_NOISE_H
