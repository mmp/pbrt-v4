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

// Bezier Spline Inline Functions
template <typename P>
PBRT_CPU_GPU inline P BlossomCubicBezier(pstd::span<const P> p, Float u0, Float u1,
                                         Float u2) {
    P a[3] = {Lerp(u0, p[0], p[1]), Lerp(u0, p[1], p[2]), Lerp(u0, p[2], p[3])};
    P b[2] = {Lerp(u1, a[0], a[1]), Lerp(u1, a[1], a[2])};
    return Lerp(u2, b[0], b[1]);
}

template <typename P>
PBRT_CPU_GPU inline pstd::array<P, 7> SubdivideCubicBezier(pstd::span<const P> cp) {
    return {cp[0],
            (cp[0] + cp[1]) / 2,
            (cp[0] + 2 * cp[1] + cp[2]) / 4,
            (cp[0] + 3 * cp[1] + 3 * cp[2] + cp[3]) / 8,
            (cp[1] + 2 * cp[2] + cp[3]) / 4,
            (cp[2] + cp[3]) / 2,
            cp[3]};
}

template <typename P, typename V>
PBRT_CPU_GPU inline P EvaluateCubicBezier(pstd::span<const P> cp, Float u, V *deriv) {
    P cp1[3] = {Lerp(u, cp[0], cp[1]), Lerp(u, cp[1], cp[2]), Lerp(u, cp[2], cp[3])};
    P cp2[2] = {Lerp(u, cp1[0], cp1[1]), Lerp(u, cp1[1], cp1[2])};
    if (deriv != nullptr) {
        if (LengthSquared(cp2[1] - cp2[0]) > 0)
            *deriv = 3 * (cp2[1] - cp2[0]);
        else {
            // For a cubic Bezier, if the first three control points (say) are
            // coincident, then the derivative of the curve is legitimately
            // (0,0,0) at u=0.  This is problematic for us, though, since we'd
            // like to be able to compute a surface normal there.  In that case,
            // just punt and take the difference between the first and last
            // control points, which ain't great, but will hopefully do.
            *deriv = cp[3] - cp[0];
        }
    }
    return Lerp(u, cp2[0], cp2[1]);
}

template <typename P>
PBRT_CPU_GPU inline P EvaluateCubicBezier(pstd::span<const P> cp, Float u) {
    P cp1[3] = {Lerp(u, cp[0], cp[1]), Lerp(u, cp[1], cp[2]), Lerp(u, cp[2], cp[3])};
    P cp2[2] = {Lerp(u, cp1[0], cp1[1]), Lerp(u, cp1[1], cp1[2])};
    return Lerp(u, cp2[0], cp2[1]);
}

template <typename P>
PBRT_CPU_GPU inline pstd::array<P, 4> CubicBezierControlPoints(pstd::span<const P> cp,
                                                               Float uMin, Float uMax) {
    return {BlossomCubicBezier(cp, uMin, uMin, uMin),
            BlossomCubicBezier(cp, uMin, uMin, uMax),
            BlossomCubicBezier(cp, uMin, uMax, uMax),
            BlossomCubicBezier(cp, uMax, uMax, uMax)};
}

template <typename B, typename P>
PBRT_CPU_GPU inline B BoundCubicBezier(pstd::span<const P> cp) {
#if 1
    return Union(B(cp[0], cp[1]), B(cp[2], cp[3]));
#else
    // http://iquilezles.org/www/articles/bezierbbox/bezierbbox.htm
    B bounds;
    for (int i = 0; i < P::nDimensions; ++i) {
        Float mi = std::min(cp[0][i], cp[3][i]), ma = std::max(cp[0][i], cp[3][i]);

        Float c = -1.0 * cp[0][i] + 1.0 * cp[1][i];
        Float b = 1.0 * cp[0][i] - 2.0 * cp[1][i] + 1.0 * cp[2][i];
        Float a = -1.0 * cp[0][i] + 3.0 * cp[1][i] - 3.0 * cp[2][i] + 1.0 * cp[3][i];

        Float h = b * b - a * c;

        if (h > 0) {
            Float g = std::sqrt(h);

            Float t1 = Clamp((-b - g) / a, 0, 1);
            Float s1 = 1 - t1;
            Float t2 = Clamp((-b + g) / a, 0, 1);
            Float s2 = 1 - t2;
            Float q1 = (s1 * s1 * s1 * cp[0][i] + 3.0 * s1 * s1 * t1 * cp[1][i] +
                        3.0 * s1 * t1 * t1 * cp[2][i] + t1 * t1 * t1 * cp[3][i]);
            Float q2 = (s2 * s2 * s2 * cp[0][i] + 3.0 * s2 * s2 * t2 * cp[1][i] +
                        3.0 * s2 * t2 * t2 * cp[2][i] + t2 * t2 * t2 * cp[3][i]);

            mi = std::min(mi, std::min(q1, q2));
            ma = std::max(ma, std::max(q1, q2));
        }
        bounds.pMin[i] = mi;
        bounds.pMax[i] = ma;
    }

    return bounds;
#endif
}

template <typename B, typename P>
PBRT_CPU_GPU inline B BoundCubicBezier(pstd::span<const P> cp, Float uMin, Float uMax) {
    if (uMin == 0 && uMax == 1)
        return BoundCubicBezier<B>(cp);
    pstd::array<P, 4> cpSeg = CubicBezierControlPoints(cp, uMin, uMax);
    return BoundCubicBezier<B>(pstd::MakeConstSpan(cpSeg));
}

template <typename P>
PBRT_CPU_GPU inline pstd::array<P, 4> ElevateQuadraticBezierToCubic(
    pstd::span<const P> cp) {
    return {cp[0], Lerp(2.f / 3.f, cp[0], cp[1]), Lerp(1.f / 3.f, cp[1], cp[2]), cp[2]};
}

template <typename P>
PBRT_CPU_GPU inline pstd::array<P, 3> QuadraticBSplineToBezier(pstd::span<const P> cp) {
    // We can compute equivalent Bezier control points via some blossiming.
    // We have three control points and a uniform knot vector; we'll label
    // the points p01, p12, and p23.  We want the Bezier control points of
    // the equivalent curve, which are p11, p12, and p22.  We already have
    // p12.
    P p11 = Lerp(0.5, cp[0], cp[1]);
    P p22 = Lerp(0.5, cp[1], cp[2]);
    return {p11, cp[1], p22};
}

template <typename P>
PBRT_CPU_GPU inline pstd::array<P, 4> CubicBSplineToBezier(pstd::span<const P> cp) {
    // Blossom from p012, p123, p234, and p345 to the Bezier control points
    // p222, p223, p233, and p333.
    // https://people.eecs.berkeley.edu/~sequin/CS284/IMGS/cubicbsplinepoints.gif
    P p012 = cp[0], p123 = cp[1], p234 = cp[2], p345 = cp[3];

    P p122 = Lerp(2.f / 3.f, p012, p123);
    P p223 = Lerp(1.f / 3.f, p123, p234);
    P p233 = Lerp(2.f / 3.f, p123, p234);
    P p334 = Lerp(1.f / 3.f, p234, p345);

    P p222 = Lerp(0.5f, p122, p223);
    P p333 = Lerp(0.5f, p233, p334);

    return {p222, p223, p233, p333};
}

}  // namespace pbrt

#endif  // PBRT_UTIL_SPLINES_H
