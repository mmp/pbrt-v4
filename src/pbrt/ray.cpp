// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/ray.h>

#include <pbrt/util/print.h>

namespace pbrt {

std::string Ray::ToString() const {
    return StringPrintf("[ o: %s d: %s time: %f, medium: %s ]", o, d, time, medium);
}

std::string RayDifferential::ToString() const {
    return StringPrintf("[ ray: %s differentials: %s xo: %s xd: %s yo: %s yd: %s ]",
                        ((const Ray &)(*this)), hasDifferentials ? "true" : "false",
                        rxOrigin, rxDirection, ryOrigin, ryDirection);
}

}  // namespace pbrt
