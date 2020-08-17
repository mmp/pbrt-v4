// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/float.h>

#include <pbrt/util/print.h>

namespace pbrt {

std::string Half::ToString() const {
    return StringPrintf("%f", (float)(*this));
}

}  // namespace pbrt
