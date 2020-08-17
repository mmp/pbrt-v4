// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace pbrt {

template <>
std::string internal::ToString2<FloatInterval>(FloatInterval x, FloatInterval y) {
    return StringPrintf("[ %s %s ]", x, y);
}

template <>
std::string internal::ToString3<FloatInterval>(FloatInterval x, FloatInterval y,
                                               FloatInterval z) {
    return StringPrintf("[ %s %s %s ]", x, y, z);
}

template <typename T>
std::string internal::ToString2(T x, T y) {
    if (std::is_floating_point<T>::value)
        return StringPrintf("[ %f, %f ]", x, y);
    else
        return StringPrintf("[ %d, %d ]", x, y);
}

template <typename T>
std::string internal::ToString3(T x, T y, T z) {
    if (std::is_floating_point<T>::value)
        return StringPrintf("[ %f, %f, %f ]", x, y, z);
    else
        return StringPrintf("[ %d, %d, %d ]", x, y, z);
}

template std::string internal::ToString2(float, float);
template std::string internal::ToString2(double, double);
template std::string internal::ToString2(int, int);
template std::string internal::ToString3(float, float, float);
template std::string internal::ToString3(double, double, double);
template std::string internal::ToString3(int, int, int);

// Quaternion Method Definitions
std::string Quaternion::ToString() const {
    return StringPrintf("[ %f, %f, %f, %f ]", v.x, v.y, v.z, w);
}

std::string DirectionCone::ToString() const {
    return StringPrintf("[ DirectionCone w: %s cosTheta: %f ]", w, cosTheta);
}

}  // namespace pbrt
