// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// Include this first, since it has a method named Infinity(), and we
// #define that for __CUDA_ARCH__ builds.
#include <gtest/gtest.h>

#include <pbrt/util/sampling.h>

#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <set>

namespace pbrt {

// Sampling Function Definitions
pstd::array<Float, 3> SampleSphericalTriangle(const pstd::array<Point3f, 3> &v, Point3f p,
                                              Point2f u, Float *pdf) {
    if (pdf)
        *pdf = 0;
    // Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
    Vector3f a(v[0] - p), b(v[1] - p), c(v[2] - p);
    CHECK_GT(LengthSquared(a), 0);
    CHECK_GT(LengthSquared(b), 0);
    CHECK_GT(LengthSquared(c), 0);
    a = Normalize(a);
    b = Normalize(b);
    c = Normalize(c);

    // Compute normalized cross products of all direction pairs
    Vector3f n_ab = Cross(a, b), n_bc = Cross(b, c), n_ca = Cross(c, a);
    if (LengthSquared(n_ab) == 0 || LengthSquared(n_bc) == 0 || LengthSquared(n_ca) == 0)
        return {};
    n_ab = Normalize(n_ab);
    n_bc = Normalize(n_bc);
    n_ca = Normalize(n_ca);

    // Find angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
    Float alpha = AngleBetween(n_ab, -n_ca);
    Float beta = AngleBetween(n_bc, -n_ab);
    Float gamma = AngleBetween(n_ca, -n_bc);

    // Uniformly sample triangle area $A$ to compute $A'$
    Float A_pi = alpha + beta + gamma;
    Float Ap_pi = Lerp(u[0], Pi, A_pi);
    if (pdf) {
        Float A = A_pi - Pi;
        *pdf = (A <= 0) ? 0 : 1 / A;
    }

    // Find $\cos \beta'$ for point along _b_ for sampled area
    Float cosAlpha = std::cos(alpha), sinAlpha = std::sin(alpha);
    Float sinPhi = std::sin(Ap_pi) * cosAlpha - std::cos(Ap_pi) * sinAlpha;
    Float cosPhi = std::cos(Ap_pi) * cosAlpha + std::sin(Ap_pi) * sinAlpha;
    Float k1 = cosPhi + cosAlpha;
    Float k2 = sinPhi - sinAlpha * Dot(a, b) /* cos c */;
    Float cosBp = (k2 + (DifferenceOfProducts(k2, cosPhi, k1, sinPhi)) * cosAlpha) /
                  ((SumOfProducts(k2, sinPhi, k1, cosPhi)) * sinAlpha);
    // Happens if the triangle basically covers the entire hemisphere.
    // We currently depend on calling code to detect this case, which
    // is sort of ugly/unfortunate.
    CHECK(!IsNaN(cosBp));
    cosBp = Clamp(cosBp, -1, 1);

    // Sample $c'$ along the arc between $b'$ and $a$
    Float sinBp = SafeSqrt(1 - Sqr(cosBp));
    Vector3f cp = cosBp * a + sinBp * Normalize(GramSchmidt(c, a));

    // Compute sampled spherical triangle direction and return barycentrics
    Float cosTheta = 1 - u[1] * (1 - Dot(cp, b));
    Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    Vector3f w = cosTheta * b + sinTheta * Normalize(GramSchmidt(cp, b));
    // Find barycentric coordinates for sampled direction _w_
    Vector3f e1 = v[1] - v[0], e2 = v[2] - v[0];
    Vector3f s1 = Cross(w, e2);
    Float divisor = Dot(s1, e1);
    CHECK_RARE(1e-6, divisor == 0);
    if (divisor == 0) {
        // This happens with triangles that cover (nearly) the whole
        // hemisphere.
        return {1.f / 3.f, 1.f / 3.f, 1.f / 3.f};
    }
    Float invDivisor = 1 / divisor;
    Vector3f s = p - v[0];
    Float b1 = Dot(s, s1) * invDivisor;
    Float b2 = Dot(w, Cross(s, e1)) * invDivisor;

    // Return clamped barycentrics for sampled direction
    b1 = Clamp(b1, 0, 1);
    b2 = Clamp(b2, 0, 1);
    if (b1 + b2 > 1) {
        b1 /= b1 + b2;
        b2 /= b1 + b2;
    }
    return {Float(1 - b1 - b2), Float(b1), Float(b2)};
}

// Via Jim Arvo's SphTri.C
Point2f InvertSphericalTriangleSample(const pstd::array<Point3f, 3> &v, Point3f p,
                                      Vector3f w) {
    // Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
    Vector3f a(v[0] - p), b(v[1] - p), c(v[2] - p);
    CHECK_GT(LengthSquared(a), 0);
    CHECK_GT(LengthSquared(b), 0);
    CHECK_GT(LengthSquared(c), 0);
    a = Normalize(a);
    b = Normalize(b);
    c = Normalize(c);

    // Compute normalized cross products of all direction pairs
    Vector3f n_ab = Cross(a, b), n_bc = Cross(b, c), n_ca = Cross(c, a);
    if (LengthSquared(n_ab) == 0 || LengthSquared(n_bc) == 0 || LengthSquared(n_ca) == 0)
        return {};
    n_ab = Normalize(n_ab);
    n_bc = Normalize(n_bc);
    n_ca = Normalize(n_ca);

    // Find angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
    Float alpha = AngleBetween(n_ab, -n_ca);
    Float beta = AngleBetween(n_bc, -n_ab);
    Float gamma = AngleBetween(n_ca, -n_bc);

    // Find vertex $\VEC{c'}$ along $\VEC{a}\VEC{c}$ arc for $\w{}$
    Vector3f cp = Normalize(Cross(Cross(b, w), Cross(c, a)));
    if (Dot(cp, a + c) < 0)
        cp = -cp;

    // Invert uniform area sampling to find _u0_
    Float u0;
    if (Dot(a, cp) > 0.99999847691f /* 0.1 degrees */)
        u0 = 0;
    else {
        // Compute area $A'$ of subtriangle
        Vector3f n_cpb = Cross(cp, b), n_acp = Cross(a, cp);
        CHECK_RARE(1e-5, LengthSquared(n_cpb) == 0 || LengthSquared(n_acp) == 0);
        if (LengthSquared(n_cpb) == 0 || LengthSquared(n_acp) == 0)
            return Point2f(0.5, 0.5);
        n_cpb = Normalize(n_cpb);
        n_acp = Normalize(n_acp);
        Float Ap = alpha + AngleBetween(n_ab, n_cpb) + AngleBetween(n_acp, -n_cpb) - Pi;

        // Compute sample _u0_ that gives the area $A'$
        Float A = alpha + beta + gamma - Pi;
        u0 = Ap / A;
    }

    // Invert arc sampling to find _u1_ and return result
    Float u1 = (1 - Dot(w, b)) / (1 - Dot(cp, b));
    return Point2f(Clamp(u0, 0, 1), Clamp(u1, 0, 1));
}

Point3f SampleSphericalRectangle(Point3f pRef, Point3f s, Vector3f ex, Vector3f ey,
                                 Point2f u, Float *pdf) {
    // Compute local reference frame and transform rectangle coordinates
    Float exl = Length(ex), eyl = Length(ey);
    Frame R = Frame::FromXY(ex / exl, ey / eyl);
    Vector3f dLocal = R.ToLocal(s - pRef);
    Float z0 = dLocal.z;

    // flip 'z' to make it point against 'Q'
    if (z0 > 0) {
        R.z = -R.z;
        z0 *= -1;
    }
    Float x0 = dLocal.x, y0 = dLocal.y;
    Float x1 = x0 + exl, y1 = y0 + eyl;

    // Find plane normals to rectangle edges and compute internal angles
    Vector3f v00(x0, y0, z0), v01(x0, y1, z0);
    Vector3f v10(x1, y0, z0), v11(x1, y1, z0);
    Vector3f n0 = Normalize(Cross(v00, v10)), n1 = Normalize(Cross(v10, v11));
    Vector3f n2 = Normalize(Cross(v11, v01)), n3 = Normalize(Cross(v01, v00));
    Float g0 = AngleBetween(-n0, n1), g1 = AngleBetween(-n1, n2);
    Float g2 = AngleBetween(-n2, n3), g3 = AngleBetween(-n3, n0);

    // Compute spherical rectangle solid angle and PDF
    Float solidAngle = g0 + g1 + g2 + g3 - 2 * Pi;
    CHECK_RARE(1e-5, solidAngle <= 0);
    if (solidAngle <= 0) {
        if (pdf)
            *pdf = 0;
        return Point3f(s + u[0] * ex + u[1] * ey);
    }
    if (pdf)
        *pdf = std::max<Float>(0, 1 / solidAngle);
    if (solidAngle < 1e-3)
        return Point3f(s + u[0] * ex + u[1] * ey);

    // Sample _cu_ for spherical rectangle sample
    Float b0 = n0.z, b1 = n2.z;
    Float au = u[0] * (g0 + g1 - 2 * Pi) + (u[0] - 1) * (g2 + g3);
    Float fu = (std::cos(au) * b0 - b1) / std::sin(au);
    Float cu = pstd::copysign(1 / std::sqrt(Sqr(fu) + Sqr(b0)), fu);
    cu = Clamp(cu, -OneMinusEpsilon, OneMinusEpsilon);  // avoid NaNs

    // Find _xu_ along $x$ edge for spherical rectangle sample
    Float xu = -(cu * z0) / SafeSqrt(1 - Sqr(cu));
    xu = Clamp(xu, x0, x1);

    // Find _xv_ along $y$ edge for spherical rectangle sample
    Float dd = std::sqrt(Sqr(xu) + Sqr(z0));
    Float h0 = y0 / std::sqrt(Sqr(dd) + Sqr(y0));
    Float h1 = y1 / std::sqrt(Sqr(dd) + Sqr(y1));
    Float hv = h0 + u[1] * (h1 - h0), hvsq = Sqr(hv);
    Float yv = (hvsq < 1 - 1e-6f) ? (hv * dd) / std::sqrt(1 - hvsq) : y1;

    // Return spherical triangle sample in original coordinate system
    return pRef + R.FromLocal(Vector3f(xu, yv, z0));
}

Point2f InvertSphericalRectangleSample(Point3f pRef, Point3f s, Vector3f ex, Vector3f ey,
                                       Point3f pRect) {
    // TODO: Delete anything unused in the below...

    // SphQuadInit()
    // local reference system 'R'
    Float exl = Length(ex), eyl = Length(ey);
    Frame R = Frame::FromXY(ex / exl, ey / eyl);

    // compute rectangle coords in local reference system
    Vector3f d = s - pRef;
    Vector3f dLocal = R.ToLocal(d);
    Float z0 = dLocal.z;

    // flip 'z' to make it point against 'Q'
    if (z0 > 0) {
        R.z = -R.z;
        z0 *= -1;
    }
    Float z0sq = Sqr(z0);
    Float x0 = dLocal.x;
    Float y0 = dLocal.y;
    Float x1 = x0 + exl;
    Float y1 = y0 + eyl;
    Float y0sq = Sqr(y0), y1sq = Sqr(y1);

    // create vectors to four vertices
    Vector3f v00(x0, y0, z0), v01(x0, y1, z0);
    Vector3f v10(x1, y0, z0), v11(x1, y1, z0);

    // compute normals to edges
    Vector3f n0 = Normalize(Cross(v00, v10));
    Vector3f n1 = Normalize(Cross(v10, v11));
    Vector3f n2 = Normalize(Cross(v11, v01));
    Vector3f n3 = Normalize(Cross(v01, v00));

    // compute internal angles (gamma_i)
    Float g0 = AngleBetween(-n0, n1);
    Float g1 = AngleBetween(-n1, n2);
    Float g2 = AngleBetween(-n2, n3);
    Float g3 = AngleBetween(-n3, n0);

    // compute predefined constants
    Float b0 = n0.z, b1 = n2.z, b0sq = Sqr(b0), b1sq = Sqr(b1);

    // compute solid angle from internal angles
    Float solidAngle = double(g0) + double(g1) + double(g2) + double(g3) - 2. * Pi;

    // TODO: this (rarely) goes differently than sample. figure out why...
    if (solidAngle < 1e-3) {
        Vector3f pq = pRect - s;
        return Point2f(Dot(pq, ex) / LengthSquared(ex), Dot(pq, ey) / LengthSquared(ey));
    }

    Vector3f v = R.ToLocal(pRect - pRef);
    Float xu = v.x, yv = v.y;

    xu = Clamp(xu, x0, x1);  // avoid Infs
    if (xu == 0)
        xu = 1e-10;

    // Doing all this in double actually makes things slightly worse???!?
    // Float fusq = (1 - b0sq * Sqr(cu)) / Sqr(cu);
    // Float fusq = 1 / Sqr(cu) - b0sq;  // more stable
    Float invcusq = 1 + z0sq / Sqr(xu);
    Float fusq = invcusq - b0sq;  // the winner so far
    Float fu = pstd::copysign(std::sqrt(fusq), xu);
    // Note, though have 1 + z^2/x^2 - b0^2, which isn't great if b0 \approx 1
    // double fusq = 1. - Sqr(double(b0)) + Sqr(double(z0) / double(xu));  //
    // this is worse?? double fu = pstd::copysign(std::sqrt(fusq), cu);
    CHECK_RARE(1e-6, fu == 0);

    // State of the floating point world: in the bad cases, about half the
    // error seems to come from inaccuracy in fu and half comes from
    // inaccuracy in sqrt/au.
    //
    // For fu, the main issue comes adding a small value to 1+ in invcusq
    // and then having b0sq be close to one, so having catastrophic
    // cancellation affect fusq. Approximating it as z0sq / Sqr(xu) when
    // b0sq is close to one doesn't help, however..
    //
    // For au, DifferenceOfProducts doesn't seem to help with the two
    // factors. Furthermore, while it would be nice to think about this
    // like atan(y/x) and then rewrite/simplify y/x, we need to do so in a
    // way that doesn't flip the sign of x and y, which would be fine if we
    // were computing y/x, but messes up atan2's quadrant-determinations...

    Float sqrt = SafeSqrt(DifferenceOfProducts(b0, b0, b1, b1) + fusq);
    // No benefit to difference of products here...
    Float au = std::atan2(-(b1 * fu) - pstd::copysign(b0 * sqrt, fu * b0),
                          b0 * b1 - sqrt * std::abs(fu));
    if (au > 0)
        au -= 2 * Pi;

    if (fu == 0)
        au = Pi;
    Float u0 = (au + g2 + g3) / solidAngle;

    Float ddsq = Sqr(xu) + z0sq;
    Float dd = std::sqrt(ddsq);
    Float h0 = y0 / std::sqrt(ddsq + y0sq);
    Float h1 = y1 / std::sqrt(ddsq + y1sq);
    Float yvsq = Sqr(yv);

    Float u1[2] = {(DifferenceOfProducts(h0, h0, h0, h1) -
                    std::abs(h0 - h1) * std::sqrt(yvsq * (ddsq + yvsq)) / (ddsq + yvsq)) /
                       Sqr(h0 - h1),
                   (DifferenceOfProducts(h0, h0, h0, h1) +
                    std::abs(h0 - h1) * std::sqrt(yvsq * (ddsq + yvsq)) / (ddsq + yvsq)) /
                       Sqr(h0 - h1)};

    // TODO: yuck is there a better way to figure out which is the right
    // solution?
    Float hv[2] = {Lerp(u1[0], h0, h1), Lerp(u1[1], h0, h1)};
    Float hvsq[2] = {Sqr(hv[0]), Sqr(hv[1])};
    Float yz[2] = {(hv[0] * dd) / std::sqrt(1 - hvsq[0]),
                   (hv[1] * dd) / std::sqrt(1 - hvsq[1])};

    Point2f u = (std::abs(yz[0] - yv) < std::abs(yz[1] - yv))
                    ? Point2f(Clamp(u0, 0, 1), u1[0])
                    : Point2f(Clamp(u0, 0, 1), u1[1]);

    return u;
}

Vector3f SampleHenyeyGreenstein(Vector3f wo, Float g, Point2f u, Float *pdf) {
    // Compute $\cos \theta$ for Henyey--Greenstein sample
    Float cosTheta;
    if (std::abs(g) < 1e-3f)
        cosTheta = 1 - 2 * u[0];
    else
        cosTheta =
            -1 / (2 * g) * (1 + Sqr(g) - Sqr((1 - Sqr(g)) / (1 + g - 2 * g * u[0])));

    // Compute direction _wi_ for Henyey--Greenstein sample
    Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    Float phi = 2 * Pi * u[1];
    Frame wFrame = Frame::FromZ(wo);
    Vector3f wi = wFrame.FromLocal(SphericalDirection(sinTheta, cosTheta, phi));

    if (pdf)
        *pdf = HenyeyGreenstein(cosTheta, g);
    return wi;
}

Point2f RejectionSampleDisk(RNG &rng) {
    Point2f p;
    do {
        p.x = 1 - 2 * rng.Uniform<Float>();
        p.y = 1 - 2 * rng.Uniform<Float>();
    } while (Sqr(p.x) + Sqr(p.y) > 1);
    return p;
}

Float SampleCatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> f,
                       pstd::span<const Float> F, Float u, Float *fval, Float *pdf) {
    CHECK_EQ(nodes.size(), f.size());
    CHECK_EQ(f.size(), F.size());
    // Map _u_ to a spline interval by inverting _F_
    u *= F.back();
    int i = FindInterval(F.size(), [&](int i) { return F[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    Float x0 = nodes[i], x1 = nodes[i + 1];
    Float f0 = f[i], f1 = f[i + 1];
    Float width = x1 - x0;

    // Approximate derivatives using finite differences
    Float d0 = (i > 0) ? width * (f1 - f[i - 1]) / (x1 - nodes[i - 1]) : (f1 - f0);
    Float d1 = (i + 2 < nodes.size()) ? width * (f[i + 2] - f0) / (nodes[i + 2] - x0)
                                      : (f1 - f0);

    // Rescale _u_ for continuous spline sampling step
    u = (u - F[i]) / width;

    // Invert definite integral over spline segment
    Float Fhat, fhat;
    auto eval = [&](Float t) -> std::pair<Float, Float> {
        Fhat = EvaluatePolynomial(t, 0, f0, 0.5f * d0,
                                  (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                                  0.25f * (d0 + d1) + 0.5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    if (fval)
        *fval = fhat;
    if (pdf)
        *pdf = fhat / F.back();
    return x0 + width * t;
}

Float SampleCatmullRom2D(pstd::span<const Float> nodes1, pstd::span<const Float> nodes2,
                         pstd::span<const Float> values, pstd::span<const Float> cdf,
                         Float alpha, Float u, Float *fval, Float *pdf) {
    // Determine offset and coefficients for the _alpha_ parameter
    int offset;
    Float weights[4];
    if (!CatmullRomWeights(nodes1, alpha, &offset, weights))
        return 0;

    // Define a lambda function to interpolate table entries
    auto interpolate = [&](pstd::span<const Float> array, int idx) {
        Float v = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
                v += array[(offset + i) * nodes2.size() + idx] * weights[i];
        return v;
    };

    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    Float maximum = interpolate(cdf, nodes2.size() - 1);
    u *= maximum;
    int idx =
        FindInterval(nodes2.size(), [&](int i) { return interpolate(cdf, i) <= u; });

    // Look up node positions and interpolated function values
    Float f0 = interpolate(values, idx), f1 = interpolate(values, idx + 1);
    Float x0 = nodes2[idx], x1 = nodes2[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;

    // Rescale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;

    // Approximate derivatives using finite differences of the interpolant
    if (idx > 0)
        d0 = width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[idx - 1]);
    else
        d0 = f1 - f0;
    if (idx + 2 < nodes2.size())
        d1 = width * (interpolate(values, idx + 2) - f0) / (nodes2[idx + 2] - x0);
    else
        d1 = f1 - f0;

    // Invert definite integral over spline segment
    Float Fhat, fhat;
    auto eval = [&](Float t) -> std::pair<Float, Float> {
        Fhat = EvaluatePolynomial(t, 0, f0, 0.5f * d0,
                                  (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                                  0.25f * (d0 + d1) + 0.5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    if (fval)
        *fval = fhat;
    if (pdf)
        *pdf = fhat / maximum;
    return x0 + width * t;
}

pstd::vector<Float> Sample1DFunction(std::function<Float(Float)> f, int nSteps,
                                     int nSamples, Float min, Float max,
                                     Allocator alloc) {
    pstd::vector<Float> values(nSteps, Float(0), alloc);
    for (int i = 0; i < nSteps; ++i) {
        double accum = 0;
        // One extra so that we sample at the very start and the very end.
        for (int j = 0; j < nSamples + 1; ++j) {
            Float delta = Float(j) / nSamples;
            Float v = Lerp((i + delta) / Float(nSteps), min, max);
            Float fv = std::abs(f(v));
            accum = std::max<double>(accum, fv);
        }
        // There's actually no need for the divide by nSamples, since
        // these are normalzed into a PDF anyway.
        values[i] = accum;
    }
    return values;
}

Array2D<Float> Sample2DFunction(std::function<Float(Float, Float)> f, int nu, int nv,
                                int nSamples, Bounds2f domain, Allocator alloc) {
    std::vector<Point2f> samples(nSamples);
    for (int i = 0; i < nSamples; ++i)
        samples[i] = Point2f(RadicalInverse(0, i), RadicalInverse(1, i));
    // Check the corners, too.
    samples.push_back(Point2f(0, 1));
    samples.push_back(Point2f(1, 0));
    samples.push_back(Point2f(1, 1));

    Array2D<Float> values(nu, nv, alloc);
    for (int v = 0; v < nv; ++v) {
        for (int u = 0; u < nu; ++u) {
            double accum = 0;
            for (size_t i = 0; i < samples.size(); ++i) {
                Point2f p = domain.Lerp(
                    Point2f((u + samples[i][0]) / nu, (v + samples[i][1]) / nv));
                Float fuv = std::abs(f(p.x, p.y));
                accum = std::max<double>(accum, fuv);
            }
            // There's actually no need for the divide by nSamples, since
            // these are normalzed into a PDF anyway.
            values(u, v) = accum;
        }
    }

    return values;
}

void PiecewiseConstant1D::TestCompareDistributions(const PiecewiseConstant1D &da,
                                                   const PiecewiseConstant1D &db,
                                                   Float eps) {
    ASSERT_EQ(da.func.size(), db.func.size());
    ASSERT_EQ(da.cdf.size(), db.cdf.size());
    ASSERT_EQ(da.min, db.min);
    ASSERT_EQ(da.max, db.max);
    for (size_t i = 0; i < da.func.size(); ++i) {
        Float pdfa = da.func[i] / da.funcInt, pdfb = db.func[i] / db.funcInt;
        Float err = std::abs(pdfa - pdfb) / ((pdfa + pdfb) / 2);
        EXPECT_LT(err, eps) << pdfa << " - " << pdfb;
    }
}

void PiecewiseConstant2D::TestCompareDistributions(const PiecewiseConstant2D &da,
                                                   const PiecewiseConstant2D &db,
                                                   Float eps) {
    PiecewiseConstant1D::TestCompareDistributions(da.pMarginal, db.pMarginal, eps);

    ASSERT_EQ(da.pConditionalV.size(), db.pConditionalV.size());
    ASSERT_EQ(da.domain, db.domain);
    for (size_t i = 0; i < da.pConditionalV.size(); ++i)
        PiecewiseConstant1D::TestCompareDistributions(da.pConditionalV[i],
                                                      db.pConditionalV[i], eps);
}

// AliasTable Method Definitions
AliasTable::AliasTable(pstd::span<const Float> weights, Allocator alloc)
    : bins(weights.size(), alloc) {
    // Normalize _weights_ to compute alias table PDF
    Float sum = std::accumulate(weights.begin(), weights.end(), 0.);
    CHECK_GT(sum, 0);
    for (size_t i = 0; i < weights.size(); ++i)
        bins[i].p = weights[i] / sum;

    // Create alias table work lists
    struct Outcome {
        Float pHat;
        size_t index;
    };
    std::vector<Outcome> under, over;
    for (size_t i = 0; i < bins.size(); ++i) {
        // Add outcome _i_ to an alias table work list
        Float pHat = bins[i].p * bins.size();
        if (pHat < 1)
            under.push_back(Outcome{pHat, i});
        else
            over.push_back(Outcome{pHat, i});
    }

    // Process under and over work item together
    while (!under.empty() && !over.empty()) {
        // Remove an item from each alias table work list
        Outcome un = under.back(), ov = over.back();
        under.pop_back();
        over.pop_back();

        // Initialize probability and alias for _un_
        bins[un.index].q = un.pHat;
        bins[un.index].alias = ov.index;

        // Push excess probability on to work list
        Float pExcess = un.pHat + ov.pHat - 1;
        if (pExcess < 1)
            under.push_back(Outcome{pExcess, ov.index});
        else
            over.push_back(Outcome{pExcess, ov.index});
    }

    // Handle remaining alias table work items
    while (!over.empty()) {
        Outcome ov = over.back();
        over.pop_back();
        bins[ov.index].q = 1;
        bins[ov.index].alias = -1;
    }
    while (!under.empty()) {
        Outcome un = under.back();
        under.pop_back();
        bins[un.index].q = 1;
        bins[un.index].alias = -1;
    }
}

int AliasTable::Sample(Float u, Float *pmf, Float *uRemapped) const {
    // Compute alias table _offset_ and remapped random sample _up_
    int offset = std::min<int>(u * bins.size(), bins.size() - 1);
    Float up = std::min<Float>(u * bins.size() - offset, OneMinusEpsilon);

    if (up < bins[offset].q) {
        // Return sample for alias table at _offset_
        DCHECK_GT(bins[offset].p, 0);
        if (pmf)
            *pmf = bins[offset].p;
        if (uRemapped)
            *uRemapped = std::min<Float>(up / bins[offset].q, OneMinusEpsilon);
        return offset;

    } else {
        // Return sample for alias table at _alias[offset]_
        int alias = bins[offset].alias;
        DCHECK_GE(alias, 0);
        DCHECK_GT(bins[alias].p, 0);
        if (pmf)
            *pmf = bins[alias].p;
        if (uRemapped)
            *uRemapped = std::min<Float>((up - bins[offset].q) / (1 - bins[offset].q),
                                         OneMinusEpsilon);
        return alias;
    }
}

std::string AliasTable::ToString() const {
    std::string s = "[ AliasTable bins: [ ";
    for (const auto &b : bins)
        s += StringPrintf("[ Bin q: %f p: %f alias: %d ] ", b.q, b.p, b.alias);
    return s + "] ]";
}

std::string SummedAreaTable::ToString() const {
    return StringPrintf("[ SummedAreaTable sum: %s ]", sum);
}

}  // namespace pbrt
