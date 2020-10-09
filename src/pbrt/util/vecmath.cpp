// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace pbrt {

template <>
std::string internal::ToString2<Interval>(Interval x, Interval y) {
    return StringPrintf("[ %s %s ]", x, y);
}

template <>
std::string internal::ToString3<Interval>(Interval x, Interval y, Interval z) {
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

// DirectionCone Function Definitions
DirectionCone Union(const DirectionCone &a, const DirectionCone &b) {
    // Handle the cases where one or both cones are empty
    if (a.empty)
        return b;
    if (b.empty)
        return a;

    // Handle the cases where one cone is inside the other
    Float theta_a = SafeACos(a.cosTheta), theta_b = SafeACos(b.cosTheta);
    Float theta_d = AngleBetween(a.w, b.w);
    if (std::min(theta_d + theta_b, Pi) <= theta_a)
        return a;
    if (std::min(theta_d + theta_a, Pi) <= theta_b)
        return b;

    // Compute the spread angle of the merged cone, $\theta_o$
    Float theta_o = (theta_a + theta_d + theta_b) / 2;
    if (theta_o >= Pi)
        return DirectionCone::EntireSphere();

    // Find the merged cone's axis and return cone union
    Float theta_r = theta_o - theta_a;
    Vector3f wr = Cross(a.w, b.w);
    if (LengthSquared(wr) == 0)
        return DirectionCone::EntireSphere();
    Vector3f w = Rotate(Degrees(theta_r), wr)(a.w);
    return DirectionCone(w, std::cos(theta_o));
}

std::string DirectionCone::ToString() const {
    return StringPrintf("[ DirectionCone w: %s cosTheta: %f empty: %s ]", w, cosTheta,
                        empty);
}

}  // namespace pbrt
