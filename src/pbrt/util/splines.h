// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SPLINES_H
#define PBRT_UTIL_SPLINES_H

#include <pbrt/pbrt.h>

#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

// Bezier Inline Functions
template <typename P>
PBRT_CPU_GPU inline P BlossomCubicBezier(pstd::span<const P> p, Float u0, Float u1,
                                         Float u2) {
    P a[3] = {Lerp(u0, p[0], p[1]), Lerp(u0, p[1], p[2]), Lerp(u0, p[2], p[3])};
    P b[2] = {Lerp(u1, a[0], a[1]), Lerp(u1, a[1], a[2])};
    return Lerp(u2, b[0], b[1]);
}

template <typename P>
PBRT_CPU_GPU inline P EvaluateCubicBezier(pstd::span<const P> cp, Float u) {
    return BlossomCubicBezier(cp, u, u, u);
}

PBRT_CPU_GPU inline Point3f EvaluateCubicBezier(pstd::span<const Point3f> cp, Float u,
                                                Vector3f *deriv) {
    Point3f cp1[3] = {Lerp(u, cp[0], cp[1]), Lerp(u, cp[1], cp[2]),
                      Lerp(u, cp[2], cp[3])};
    Point3f cp2[2] = {Lerp(u, cp1[0], cp1[1]), Lerp(u, cp1[1], cp1[2])};
    if (deriv) {
        // Compute B\'ezier curve derivative at $u$
        if (LengthSquared(cp2[1] - cp2[0]) > 0)
            *deriv = 3 * (cp2[1] - cp2[0]);
        else
            *deriv = cp[3] - cp[0];
    }
    return Lerp(u, cp2[0], cp2[1]);
}

PBRT_CPU_GPU inline pstd::array<Point3f, 7> SubdivideCubicBezier(
    pstd::span<const Point3f> cp) {
    return {cp[0],
            (cp[0] + cp[1]) / 2,
            (cp[0] + 2 * cp[1] + cp[2]) / 4,
            (cp[0] + 3 * cp[1] + 3 * cp[2] + cp[3]) / 8,
            (cp[1] + 2 * cp[2] + cp[3]) / 4,
            (cp[2] + cp[3]) / 2,
            cp[3]};
}

PBRT_CPU_GPU inline pstd::array<Point3f, 4> CubicBezierControlPoints(
    pstd::span<const Point3f> cp, Float uMin, Float uMax) {
    return {BlossomCubicBezier(cp, uMin, uMin, uMin),
            BlossomCubicBezier(cp, uMin, uMin, uMax),
            BlossomCubicBezier(cp, uMin, uMax, uMax),
            BlossomCubicBezier(cp, uMax, uMax, uMax)};
}

PBRT_CPU_GPU inline Bounds3f BoundCubicBezier(pstd::span<const Point3f> cp) {
    return Union(Bounds3f(cp[0], cp[1]), Bounds3f(cp[2], cp[3]));
}

PBRT_CPU_GPU inline Bounds3f BoundCubicBezier(pstd::span<const Point3f> cp, Float uMin,
                                              Float uMax) {
    if (uMin == 0 && uMax == 1)
        return BoundCubicBezier(cp);
    auto cpSeg = CubicBezierControlPoints(cp, uMin, uMax);
    return BoundCubicBezier(pstd::span<const Point3f>(cpSeg));
}

PBRT_CPU_GPU inline pstd::array<Point3f, 4> ElevateQuadraticBezierToCubic(
    pstd::span<const Point3f> cp) {
    return {cp[0], Lerp(2.f / 3.f, cp[0], cp[1]), Lerp(1.f / 3.f, cp[1], cp[2]), cp[2]};
}

PBRT_CPU_GPU inline pstd::array<Point3f, 3> QuadraticBSplineToBezier(
    pstd::span<const Point3f> cp) {
    // We can compute equivalent Bezier control points via some blossoming.
    // We have three control points and a uniform knot vector; we'll label
    // the points p01, p12, and p23.  We want the Bezier control points of
    // the equivalent curve, which are p11, p12, and p22.  We already have
    // p12.
    Point3f p11 = Lerp(0.5, cp[0], cp[1]);
    Point3f p22 = Lerp(0.5, cp[1], cp[2]);
    return {p11, cp[1], p22};
}

PBRT_CPU_GPU inline pstd::array<Point3f, 4> CubicBSplineToBezier(
    pstd::span<const Point3f> cp) {
    // Blossom from p012, p123, p234, and p345 to the Bezier control points
    // p222, p223, p233, and p333.
    // https://people.eecs.berkeley.edu/~sequin/CS284/IMGS/cubicbsplinepoints.gif
    Point3f p012 = cp[0], p123 = cp[1], p234 = cp[2], p345 = cp[3];

    Point3f p122 = Lerp(2.f / 3.f, p012, p123);
    Point3f p223 = Lerp(1.f / 3.f, p123, p234);
    Point3f p233 = Lerp(2.f / 3.f, p123, p234);
    Point3f p334 = Lerp(1.f / 3.f, p234, p345);

    Point3f p222 = Lerp(0.5f, p122, p223);
    Point3f p333 = Lerp(0.5f, p233, p334);

    return {p222, p223, p233, p333};
}

}  // namespace pbrt

#endif  // PBRT_UTIL_SPLINES_H
