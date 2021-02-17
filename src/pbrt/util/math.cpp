// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/math.h>

#include <pbrt/util/check.h>
#include <pbrt/util/print.h>
#include <pbrt/util/vecmath.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace pbrt {

std::string CompensatedFloat::ToString() const {
    return StringPrintf("[ CompensatedFloat v: %f err: %f ]", v, err);
}

template <typename Float>
std::string CompensatedSum<Float>::ToString() const {
    return StringPrintf("[ CompensatedSum sum: %s c: %s ]", sum, c);
}

template <int N>
std::string SquareMatrix<N>::ToString() const {
    std::string s = "[ [";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            s += StringPrintf(" %f", m[i][j]);
            if (j < N - 1)
                s += ',';
            else
                s += " ]";
        }
        if (i < N - 1)
            s += ", [";
    }
    s += " ]";
    return s;
}

// General case
template <int N>
pstd::optional<SquareMatrix<N>> Inverse(const SquareMatrix<N> &m) {
    int indxc[N], indxr[N];
    int ipiv[N] = {0};
    Float minv[N][N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            minv[i][j] = m[i][j];
    for (int i = 0; i < N; i++) {
        int irow = 0, icol = 0;
        Float big = 0.f;
        // Choose pivot
        for (int j = 0; j < N; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < N; k++) {
                    if (ipiv[k] == 0) {
                        if (std::abs(minv[j][k]) >= big) {
                            big = std::abs(minv[j][k]);
                            irow = j;
                            icol = k;
                        }
                    } else if (ipiv[k] > 1)
                        return {};  // singular
                }
            }
        }
        ++ipiv[icol];
        // Swap rows _irow_ and _icol_ for pivot
        if (irow != icol) {
            for (int k = 0; k < N; ++k)
                pstd::swap(minv[irow][k], minv[icol][k]);
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (minv[icol][icol] == 0.f)
            return {};  // singular

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        Float pivinv = 1. / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < N; j++)
            minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < N; j++) {
            if (j != icol) {
                Float save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < N; k++)
                    minv[j][k] = FMA(-minv[icol][k], save, minv[j][k]);
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = N - 1; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < N; k++)
                pstd::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
        }
    }
    return SquareMatrix<N>(minv);
}

template class SquareMatrix<2>;
template pstd::optional<SquareMatrix<2>> Inverse(const SquareMatrix<2> &);
template SquareMatrix<2> operator*(const SquareMatrix<2> &m1, const SquareMatrix<2> &m2);

template class SquareMatrix<3>;
template class SquareMatrix<4>;

int NextPrime(int x) {
    if (x == 2)
        return 3;
    if ((x & 1) == 0)
        ++x;  // make it odd

    std::vector<int> smallPrimes{2};
    // NOTE: isPrime w.r.t. smallPrims...
    auto isPrime = [&smallPrimes](int n) {
        for (int p : smallPrimes)
            if (n != p && (n % p) == 0)
                return false;
        return true;
    };

    // Initialize smallPrimes
    // Up to about 2B, the biggest gap between primes:
    // https://en.wikipedia.org/wiki/Prime_gap
    const int maxPrimeGap = 320;
    for (int n = 3; n < int(std::sqrt(x + maxPrimeGap)) + 1; n += 2)
        if (isPrime(n))
            smallPrimes.push_back(n);

    while (!isPrime(x))
        x += 2;

    return x;
}

#ifndef PBRT_IS_GPU_CODE
#ifdef PBRT_FLOAT_AS_DOUBLE
const Interval Interval::Pi(3.1415926535897931, 3.1415926535897936);
#else
const Interval Interval::Pi = Interval(3.1415925f, 3.14159274f);
#endif
#endif

std::string Interval::ToString() const {
    return StringPrintf("[ Interval [%f, %f] ]", low, high);
}

// Spline Interpolation Function Definitions
bool CatmullRomWeights(pstd::span<const Float> nodes, Float x, int *offset,
                       pstd::span<Float> weights) {
    CHECK_GE(weights.size(), 4);
    // Return _false_ if _x_ is out of bounds
    if (!(x >= nodes.front() && x <= nodes.back()))
        return false;

    // Search for the interval _idx_ containing _x_
    int idx = FindInterval(nodes.size(), [&](int i) { return nodes[i] <= x; });
    *offset = idx - 1;
    Float x0 = nodes[idx], x1 = nodes[idx + 1];

    // Compute the $t$ parameter and powers
    Float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;

    // Compute initial node weights $w_1$ and $w_2$
    weights[1] = 2 * t3 - 3 * t2 + 1;
    weights[2] = -2 * t3 + 3 * t2;

    // Compute first node weight $w_0$
    if (idx > 0) {
        Float w0 = (t3 - 2 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        Float w0 = t3 - 2 * t2 + t;
        weights[0] = 0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    // Compute last node weight $w_3$
    if (idx + 2 < nodes.size()) {
        Float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        Float w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0;
    }

    return true;
}

Float CatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> f, Float x) {
    CHECK_EQ(nodes.size(), f.size());
    if (!(x >= nodes.front() && x <= nodes.back()))
        return 0;
    int idx = FindInterval(nodes.size(), [&](int i) { return nodes[i] <= x; });
    Float x0 = nodes[idx], x1 = nodes[idx + 1];
    Float f0 = f[idx], f1 = f[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;
    if (idx > 0)
        d0 = width * (f1 - f[idx - 1]) / (x1 - nodes[idx - 1]);
    else
        d0 = f1 - f0;

    if (idx + 2 < nodes.size())
        d1 = width * (f[idx + 2] - f0) / (nodes[idx + 2] - x0);
    else
        d1 = f1 - f0;

    Float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * f0 + (-2 * t3 + 3 * t2) * f1 + (t3 - 2 * t2 + t) * d0 +
           (t3 - t2) * d1;
}

Float InvertCatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> f,
                       Float u) {
    // Stop when _u_ is out of bounds
    if (!(u > f.front()))
        return nodes.front();
    else if (!(u < f.back()))
        return nodes.back();

    // Map _u_ to a spline interval by inverting _f_
    int i = FindInterval(f.size(), [&](int i) { return f[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    Float x0 = nodes[i], x1 = nodes[i + 1];
    Float f0 = f[i], f1 = f[i + 1];
    Float width = x1 - x0;

    // Approximate derivatives using finite differences
    Float d0 = (i > 0) ? width * (f1 - f[i - 1]) / (x1 - nodes[i - 1]) : (f1 - f0);
    Float d1 = (i + 2 < nodes.size()) ? width * (f[i + 2] - f0) / (nodes[i + 2] - x0)
                                      : (f1 - f0);

    // Invert the spline interpolant using Newton-Bisection
    auto eval = [&](Float t) -> std::pair<Float, Float> {
        // Compute powers of _t_
        Float t2 = t * t, t3 = t2 * t;

        // Set _Fhat_ using Equation (\ref{eq:cubicspline-as-basisfunctions})
        Float Fhat = (2 * t3 - 3 * t2 + 1) * f0 + (-2 * t3 + 3 * t2) * f1 +
                     (t3 - 2 * t2 + t) * d0 + (t3 - t2) * d1;

        // Set _fhat_ using Equation (\ref{eq:cubicspline-derivative})
        Float fhat = (6 * t2 - 6 * t) * f0 + (-6 * t2 + 6 * t) * f1 +
                     (3 * t2 - 4 * t + 1) * d0 + (3 * t2 - 2 * t) * d1;

        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);
    return x0 + t * width;
}

Float IntegrateCatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> f,
                          pstd::span<Float> cdf) {
    CHECK_EQ(nodes.size(), f.size());
    Float sum = 0;
    cdf[0] = 0;
    for (int i = 0; i < nodes.size() - 1; ++i) {
        // Look up $x_i$ and function values of spline segment _i_
        Float x0 = nodes[i], x1 = nodes[i + 1];
        Float f0 = f[i], f1 = f[i + 1];
        Float width = x1 - x0;

        // Approximate derivatives using finite differences
        Float d0 = (i > 0) ? width * (f1 - f[i - 1]) / (x1 - nodes[i - 1]) : (f1 - f0);
        Float d1 = (i + 2 < nodes.size()) ? width * (f[i + 2] - f0) / (nodes[i + 2] - x0)
                                          : (f1 - f0);

        // Keep a running sum and build a cumulative distribution function
        sum += width * ((f0 + f1) / 2 + (d0 - d1) / 12);
        cdf[i + 1] = sum;
    }
    return sum;
}

// Square--Sphere Mapping Function Definitions
// Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
Vector3f EqualAreaSquareToSphere(Point2f p) {
    CHECK(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
    // Transform _p_ to $[-1,1]^2$ and compute absolute values
    Float u = 2 * p.x - 1, v = 2 * p.y - 1;
    Float up = std::abs(u), vp = std::abs(v);

    // Compute radius _r_ as signed distance from diagonal
    Float signedDistance = 1 - (up + vp);
    Float d = std::abs(signedDistance);
    Float r = 1 - d;

    // Compute angle $\phi$ for square to sphere mapping
    Float phi = (r == 0 ? 1 : (vp - up) / r + 1) * Pi / 4;

    // Find $z$ coordinate for spherical direction
    Float z = std::copysign(1 - Sqr(r), signedDistance);

    // Compute $\cos \phi$ and $\sin \phi$ for original quadrant and return vector
    Float cosPhi = std::copysign(std::cos(phi), u);
    Float sinPhi = std::copysign(std::sin(phi), v);
    return Vector3f(cosPhi * r * SafeSqrt(2 - Sqr(r)), sinPhi * r * SafeSqrt(2 - Sqr(r)),
                    z);
}

// Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
Point2f EqualAreaSphereToSquare(Vector3f d) {
    DCHECK(LengthSquared(d) > .999 && LengthSquared(d) < 1.001);
    Float x = std::abs(d.x), y = std::abs(d.y), z = std::abs(d.z);

    // Compute the radius r
    Float r = SafeSqrt(1 - z);  // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    Float a = std::max(x, y), b = std::min(x, y);
    b = a == 0 ? 0 : b / a;

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    const Float t1 = 0.406758566246788489601959989e-5;
    const Float t2 = 0.636226545274016134946890922156;
    const Float t3 = 0.61572017898280213493197203466e-2;
    const Float t4 = -0.247333733281268944196501420480;
    const Float t5 = 0.881770664775316294736387951347e-1;
    const Float t6 = 0.419038818029165735901852432784e-1;
    const Float t7 = -0.251390972343483509333252996350e-1;
    Float phi = EvaluatePolynomial(b, t1, t2, t3, t4, t5, t6, t7);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y)
        phi = 1 - phi;

    // Find (u,v) based on (r,phi)
    Float v = phi * r;
    Float u = r - v;

    if (d.z < 0) {
        // southern hemisphere -> mirror u,v
        pstd::swap(u, v);
        u = 1 - u;
        v = 1 - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = std::copysign(u, d.x);
    v = std::copysign(v, d.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return Point2f(0.5f * (u + 1), 0.5f * (v + 1));
}

Point2f WrapEqualAreaSquare(Point2f uv) {
    if (uv[0] < 0) {
        uv[0] = -uv[0];     // mirror across u = 0
        uv[1] = 1 - uv[1];  // mirror across v = 0.5
    } else if (uv[0] > 1) {
        uv[0] = 2 - uv[0];  // mirror across u = 1
        uv[1] = 1 - uv[1];  // mirror across v = 0.5
    }
    if (uv[1] < 0) {
        uv[0] = 1 - uv[0];  // mirror across u = 0.5
        uv[1] = -uv[1];     // mirror across v = 0;
    } else if (uv[1] > 1) {
        uv[0] = 1 - uv[0];  // mirror across u = 0.5
        uv[1] = 2 - uv[1];  // mirror across v = 1
    }
    return uv;
}

}  // namespace pbrt
