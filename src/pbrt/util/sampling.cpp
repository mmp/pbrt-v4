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
Point2f RejectionSampleDisk(RNG &rng) {
    Point2f p;
    do {
        p.x = 1 - 2 * rng.Uniform<Float>();
        p.y = 1 - 2 * rng.Uniform<Float>();
    } while (p.x * p.x + p.y * p.y > 1);
    return p;
}

pstd::array<Float, 3> SampleSphericalTriangle(const pstd::array<Point3f, 3> &v,
                                              const Point3f &p, const Point2f &u,
                                              Float *pdf) {
    if (pdf != nullptr)
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

    // Compute angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
    Float alpha = AngleBetween(n_ab, -n_ca);
    Float beta = AngleBetween(n_bc, -n_ab);
    Float gamma = AngleBetween(n_ca, -n_bc);

    // Uniformly sample triangle area $A$ to compute $A'$
    Float A = alpha + beta + gamma - Pi;
    if (A <= 0)
        return {};
    if (pdf != nullptr)
        *pdf = 1 / A;
    Float Ap = u[0] * A;

    // Compute $\sin \beta'$ and $\cos \beta'$ for point along _b_ for sampled area
    Float cosAlpha = std::cos(alpha), sinAlpha = std::sin(alpha);

    Float sinPhi = std::sin(Ap) * cosAlpha - std::cos(Ap) * sinAlpha;
    Float cosPhi = std::cos(Ap) * cosAlpha + std::sin(Ap) * sinAlpha;

    Float uu = cosPhi - cosAlpha;
    Float vv = sinPhi + sinAlpha * Dot(a, b) /* cos c */;
    Float cosBetap = (((vv * cosPhi - uu * sinPhi) * cosAlpha - vv) /
                      ((vv * sinPhi + uu * cosPhi) * sinAlpha));
    // Happens if the triangle basically covers the entire hemisphere.
    // We currently depend on calling code to detect this case, which
    // is sort of ugly/unfortunate.
    CHECK(!IsNaN(cosBetap));
    cosBetap = Clamp(cosBetap, -1, 1);
    Float sinBetap = SafeSqrt(1 - cosBetap * cosBetap);

    // Compute $c'$ along the arc between $b'$ and $a$
    // Gram-Schmidt
    auto GS = [](const Vector3f &a, const Vector3f &b) {
        return Normalize(a - Dot(a, b) * b);
    };
    Vector3f cp = cosBetap * a + sinBetap * GS(c, a);

    // Compute sampled spherical triangle direction and return barycentrics
    Float cosTheta = 1 - u[1] * (1 - Dot(cp, b));
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    Vector3f w = cosTheta * b + sinTheta * GS(cp, b);
    // Find barycentric coordinates for sampled direction _w_
    Vector3f e1(v[1] - v[0]), e2(v[2] - v[0]);
    Vector3f s1 = Cross(w, e2);
    Float divisor = Dot(s1, e1);
    CHECK_RARE(1e-6, divisor == 0);
    if (divisor == 0) {
        // This happens with triangles that cover (nearly) the whole
        // hemisphere.
        return {1.f / 3.f, 1.f / 3.f, 1.f / 3.f};
    }
    Float invDivisor = 1 / divisor;
    Vector3f s(p - v[0]);
    Float b1 = Dot(s, s1) * invDivisor;
    Vector3f s2 = Cross(s, e1);
    Float b2 = Dot(w, s2) * invDivisor;

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
Point2f InvertSphericalTriangleSample(const pstd::array<Point3f, 3> &v, const Point3f &p,
                                      const Vector3f &w) {
    using Vector3d = Vector3<double>;
    Vector3d a(v[0] - p), b(v[1] - p), c(v[2] - p);
    CHECK_GT(LengthSquared(a), 0);
    CHECK_GT(LengthSquared(b), 0);
    CHECK_GT(LengthSquared(c), 0);
    a = Normalize(a);
    b = Normalize(b);
    c = Normalize(c);

    // TODO: have a shared snippet that goes from here to computing
    // alpha/beta/gamma, use it also in Triangle::SolidAngle().
    Vector3d axb = Cross(a, b), bxc = Cross(b, c), cxa = Cross(c, a);
    CHECK_RARE(1e-5, LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 ||
                         LengthSquared(cxa) == 0);
    if (LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 || LengthSquared(cxa) == 0)
        return Point2f(0.5, 0.5);

    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxa = Normalize(cxa);

    // See comment in Triangle::SolidAngle() for ordering...
    double alpha = AngleBetween(cxa, -axb);
    double beta = AngleBetween(axb, -bxc);
    double gamma = AngleBetween(bxc, -cxa);

    // Spherical area of the triangle.
    double A = alpha + beta + gamma - Pi;

    // Assume that w is normalized...

    // Compute the new C vertex, which lies on the arc defined by b-w
    // and the arc defined by a-c.
    Vector3d cp = Normalize(Cross(Cross(b, Vector3d(w)), Cross(c, a)));

    // Adjust the sign of cp.  Make sure it's on the arc between A and C.
    if (Dot(cp, a + c) < 0)
        cp = -cp;

    // Compute x1, the area of the sub-triangle over the original area.
    // The AngleBetween() calls are computing the dihedral angles (a, b, cp)
    // and (a, cp, b) respectively, FWIW...
    Vector3d cnxb = Cross(cp, b), axcn = Cross(a, cp);
    CHECK_RARE(1e-5, LengthSquared(cnxb) == 0 || LengthSquared(axcn) == 0);
    if (LengthSquared(cnxb) == 0 || LengthSquared(axcn) == 0)
        return Point2f(0.5, 0.5);
    cnxb = Normalize(cnxb);
    axcn = Normalize(axcn);

    Float sub_area = alpha + AngleBetween(axb, cnxb) + AngleBetween(axcn, -cnxb) - Pi;
    Float u0 = sub_area / A;

    // Now compute the second coordinate using the new C vertex.
    Float z = Dot(Vector3d(w), b);
    Float u1 = (1 - z) / (1 - Dot(cp, b));

    return Point2f(Clamp(u0, 0, 1), Clamp(u1, 0, 1));
}

Point3f SampleSphericalRectangle(const Point3f &pRef, const Point3f &s,
                                 const Vector3f &ex, const Vector3f &ey, const Point2f &u,
                                 Float *pdf) {
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
    Float g0 = AngleBetween(-n0, n1);
    Float g1 = AngleBetween(-n1, n2);
    Float g2 = AngleBetween(-n2, n3);
    Float g3 = AngleBetween(-n3, n0);

    // Compute spherical rectangle solid angle and PDF
    Float solidAngle = double(g0) + double(g1) + double(g2) + double(g3) - 2. * Pi;
    CHECK_RARE(1e-5, solidAngle <= 0);
    if (solidAngle <= 0) {
        if (pdf != nullptr)
            *pdf = 0;
        return Point3f(s + u[0] * ex + u[1] * ey);
    }
    if (pdf != nullptr)
        *pdf = std::max<Float>(0, 1 / solidAngle);
    if (solidAngle < 1e-3)
        return Point3f(s + u[0] * ex + u[1] * ey);

    // Sample _cu_ for spherical rectangle sample
    Float b0 = n0.z, b1 = n2.z;
    // Float au = u[0] * solidAngle + k;   // original
    Float au = u[0] * solidAngle - g2 - g3;
    Float fu = (std::cos(au) * b0 - b1) / std::sin(au);
    Float cu = std::copysign(1 / std::sqrt(Sqr(fu) + Sqr(b0)), fu);
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

Point2f InvertSphericalRectangleSample(const Point3f &pRef, const Point3f &s,
                                       const Vector3f &ex, const Vector3f &ey,
                                       const Point3f &pQuad) {
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
        Vector3f pq = pQuad - s;
        return Point2f(Dot(pq, ex) / LengthSquared(ex), Dot(pq, ey) / LengthSquared(ey));
    }

    Vector3f v = R.ToLocal(pQuad - pRef);
    Float xu = v.x, yv = v.y;

    xu = Clamp(xu, x0, x1);  // avoid Infs
    if (xu == 0)
        xu = 1e-10;

    // DOing all this in double actually makes things slightly worse???!?
    // Float fusq = (1 - b0sq * Sqr(cu)) / Sqr(cu);
    // Float fusq = 1 / Sqr(cu) - b0sq;  // more stable
    Float invcusq = 1 + z0sq / Sqr(xu);
    Float fusq = invcusq - b0sq;  // the winner so far
    Float fu = std::copysign(std::sqrt(fusq), xu);
    // Note, though have 1 + z^2/x^2 - b0^2, which isn't great if b0 \approx 1
    // double fusq = 1. - Sqr(double(b0)) + Sqr(double(z0) / double(xu));  //
    // this is worse?? double fu = std::copysign(std::sqrt(fusq), cu);
    CHECK_RARE(1e-6, fu == 0);

    // State of the floating point world: in the bad cases, about half the
    // error seems to come from inaccuracy in fu and half comes from
    // inaccuracy in sqrt/au.
    //
    // For fu, the main issue comes adding a small value to 1+ in invcusq
    // and then having b0sq be close to one, so having catistrophic
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
    Float au = std::atan2(-(b1 * fu) - std::copysign(b0 * sqrt, fu * b0),
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

Vector3f SampleHenyeyGreenstein(const Vector3f &wo, Float g, const Point2f &u,
                                Float *pdf) {
    // Compute $\cos \theta$ for Henyey--Greenstein sample
    Float cosTheta;
    if (std::abs(g) < 1e-3)
        cosTheta = 1 - 2 * u[0];
    else {
        Float sqrTerm = (1 - g * g) / (1 + g - 2 * g * u[0]);
        cosTheta = -(1 + g * g - sqrTerm * sqrTerm) / (2 * g);
    }

    // Compute direction _wi_ for Henyey--Greenstein sample
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    Float phi = 2 * Pi * u[1];
    Frame wFrame = Frame::FromZ(wo);
    Vector3f wi = wFrame.FromLocal(SphericalDirection(sinTheta, cosTheta, phi));

    if (pdf)
        *pdf = HenyeyGreenstein(cosTheta, g);
    return wi;
}

Float SampleCatmullRom(pstd::span<const Float> x, pstd::span<const Float> f,
                       pstd::span<const Float> F, Float u, Float *fval, Float *pdf) {
    CHECK_EQ(x.size(), f.size());
    CHECK_EQ(f.size(), F.size());
    // Map _u_ to a spline interval by inverting _F_
    u *= F.back();
    int i = FindInterval(F.size(), [&](int i) { return F[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    Float x0 = x[i], x1 = x[i + 1];
    Float f0 = f[i], f1 = f[i + 1];
    Float width = x1 - x0;

    // Approximate derivatives using finite differences
    Float d0, d1;
    if (i > 0)
        d0 = width * (f1 - f[i - 1]) / (x1 - x[i - 1]);
    else
        d0 = f1 - f0;
    if (i + 2 < x.size())
        d1 = width * (f[i + 2] - f0) / (x[i + 2] - x0);
    else
        d1 = f1 - f0;

    // Re-scale _u_ for continous spline sampling step
    u = (u - F[i]) / width;

    // Invert definite integral over spline segment
    Float Fhat, fhat;
    auto eval = [&](Float t) -> std::pair<Float, Float> {
        Fhat =
            EvaluatePolynomial(t, 0, f0, .5f * d0, (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                               .25f * (d0 + d1) + .5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    if (fval != nullptr)
        *fval = fhat;
    if (pdf != nullptr)
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
        Float value = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
                value += array[(offset + i) * nodes2.size() + idx] * weights[i];
        return value;
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

    // Re-scale _u_ using the interpolated _cdf_
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
        Fhat =
            EvaluatePolynomial(t, 0, f0, .5f * d0, (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                               .25f * (d0 + d1) + .5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    if (fval != nullptr)
        *fval = fhat;
    if (pdf != nullptr)
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

    ASSERT_EQ(da.pConditionalY.size(), db.pConditionalY.size());
    ASSERT_EQ(da.domain, db.domain);
    for (size_t i = 0; i < da.pConditionalY.size(); ++i)
        PiecewiseConstant1D::TestCompareDistributions(da.pConditionalY[i],
                                                      db.pConditionalY[i], eps);
}

// AliasTable Method Definitions
AliasTable::AliasTable(pstd::span<const Float> weights, Allocator alloc)
    : bins(weights.size(), alloc) {
    // Normalize _weights_ to compute alias table PDF
    Float sum = std::accumulate(weights.begin(), weights.end(), 0.);
    for (size_t i = 0; i < weights.size(); ++i)
        bins[i].pdf = weights[i] / sum;

    // Create alias table work lists
    struct Outcome {
        Float pHat;
        size_t index;
    };
    std::vector<Outcome> under, over;
    for (size_t i = 0; i < bins.size(); ++i) {
        // Add outcome _i_ to an alias table work list
        Float pHat = bins[i].pdf * bins.size();
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
        Float pExcess = (un.pHat + ov.pHat) - 1;
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

int AliasTable::Sample(Float u, Float *pdfOut, Float *uRemapped) const {
    // Compute alias table _offset_ and remapped random sample _up_
    int offset = std::min<int>(u * bins.size(), bins.size() - 1);
    Float up = std::min<Float>(u * bins.size() - offset, OneMinusEpsilon);

    if (up < bins[offset].q) {
        // Return sample for alias table at _offset_
        DCHECK_GT(bins[offset].pdf, 0);
        if (pdfOut)
            *pdfOut = bins[offset].pdf;
        if (uRemapped)
            *uRemapped = std::min<Float>(up / bins[offset].q, OneMinusEpsilon);
        return offset;

    } else {
        // Return sample for alias table at _alias[offset]_
        int alias = bins[offset].alias;
        DCHECK_GE(alias, 0);
        DCHECK_GT(bins[offset].pdf, 0);
        if (pdfOut)
            *pdfOut = bins[alias].pdf;
        if (uRemapped)
            *uRemapped = std::min<Float>((up - bins[offset].q) / (1 - bins[offset].q),
                                         OneMinusEpsilon);
        return alias;
    }
}

std::string AliasTable::ToString() const {
    std::string s = "[ AliasTable bins: [ ";
    for (const auto &b : bins)
        s += StringPrintf("[ Bin q: %f pdf: %f alias: %d ] ", b.q, b.pdf, b.alias);
    return s + "] ]";
}

std::string SummedAreaTable::ToString() const {
    return StringPrintf("[ SummedAreaTable sum: %s ]", sum);
}

}  // namespace pbrt
