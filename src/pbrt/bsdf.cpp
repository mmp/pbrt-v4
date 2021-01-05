// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/bsdf.h>

#include <pbrt/util/spectrum.h>

namespace pbrt {

std::string BSDFSample::ToString() const {
    return StringPrintf("[ BSDFSample f: %s wi: %s pdf: %s flags: %s ]", f, wi, pdf,
                        flags);
}

// BSDF Method Definitions
std::string BSDF::ToString() const {
    return StringPrintf("[ BSDF bxdf: %s shadingFrame: %s ng: %s ]", bxdf, shadingFrame,
                        ng);
}

}  // namespace pbrt
