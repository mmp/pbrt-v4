// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SAMPLING_H
#define PBRT_UTIL_SAMPLING_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/lowdiscrepancy.h>  // yuck: for Hammersley generator...
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

// Sampling Function Declarations
PBRT_CPU_GPU
pstd::array<Float, 3> SampleSphericalTriangle(const pstd::array<Point3f, 3> &v,
                                              const Point3f &p, const Point2f &u,
                                              Float *pdf = nullptr);

PBRT_CPU_GPU
Point2f InvertSphericalTriangleSample(const pstd::array<Point3f, 3> &v, const Point3f &p,
                                      const Vector3f &w);

PBRT_CPU_GPU
Point3f SampleSphericalRectangle(const Point3f &p, const Point3f &v00, const Vector3f &ex,
                                 const Vector3f &ey, const Point2f &u,
                                 Float *pdf = nullptr);

PBRT_CPU_GPU
Point2f InvertSphericalRectangleSample(const Point3f &pRef, const Point3f &v00,
                                       const Vector3f &ex, const Vector3f &ey,
                                       const Point3f &pQuad);

PBRT_CPU_GPU
Vector3f SampleHenyeyGreenstein(const Vector3f &wo, Float g, const Point2f &u,
                                Float *pdf = nullptr);

PBRT_CPU_GPU
Float SampleCatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> f,
                       pstd::span<const Float> cdf, Float sample, Float *fval = nullptr,
                       Float *pdf = nullptr);

PBRT_CPU_GPU
Float SampleCatmullRom2D(pstd::span<const Float> nodes1, pstd::span<const Float> nodes2,
                         pstd::span<const Float> values, pstd::span<const Float> cdf,
                         Float alpha, Float sample, Float *fval = nullptr,
                         Float *pdf = nullptr);

// Sampling Inline Functions
PBRT_CPU_GPU
inline int SampleDiscrete(pstd::span<const Float> weights, Float u, Float *pdf = nullptr,
                          Float *uRemapped = nullptr) {
    // Handle empty _weights_ for discrete sampling
    if (weights.empty()) {
        if (pdf != nullptr)
            *pdf = 0;
        return -1;
    }

    // Compute sum of _weights_
    Float sumWeights = 0;
    for (Float w : weights)
        sumWeights += w;

    // Find offset in _weights_ corresponding to _u_
    int offset = 0;
    while (offset < weights.size() && u >= weights[offset] / sumWeights) {
        u -= weights[offset] / sumWeights;
        ++offset;
    }
    CHECK_RARE(1e-6, offset == weights.size());
    if (offset == weights.size())
        offset = weights.size() - 1;

    // Compute PDF and remapped _u_ value, if necessary
    Float p = weights[offset] / sumWeights;
    if (pdf != nullptr)
        *pdf = p;
    if (uRemapped != nullptr)
        *uRemapped = std::min(u / p, OneMinusEpsilon);

    return offset;
}

PBRT_CPU_GPU
inline Float LinearPDF(Float x, Float a, Float b) {
    DCHECK(a >= 0 && b >= 0);
    if (x < 0 || x > 1)
        return 0;
    return 2 * Lerp(x, a, b) / (a + b);
}

PBRT_CPU_GPU
inline Float SampleLinear(Float u, Float a, Float b) {
    DCHECK(a >= 0 && b >= 0);
    Float x = u * (a + b) / (a + std::sqrt(Lerp(u, Sqr(a), Sqr(b))));
    return std::min(x, OneMinusEpsilon);
}

PBRT_CPU_GPU
inline Float InvertLinearSample(Float x, Float a, Float b) {
    return x * (a * (2 - x) + b * x) / (a + b);
}

PBRT_CPU_GPU
inline Float ExponentialPDF(Float x, Float a) {
    DCHECK_GT(a, 0);
    return a * std::exp(-a * x);
}

PBRT_CPU_GPU
inline Float SampleExponential(Float u, Float a) {
    DCHECK_GT(a, 0);
    return -std::log(1 - u) / a;
}

PBRT_CPU_GPU
inline Float InvertExponentialSample(Float x, Float a) {
    DCHECK_GT(a, 0);
    return 1 - std::exp(-a * x);
}

PBRT_CPU_GPU
inline Float LogisticPDF(Float x, Float s) {
    x = std::abs(x);
    return std::exp(-x / s) / (s * Sqr(1 + std::exp(-x / s)));
}

PBRT_CPU_GPU
inline Float SampleLogistic(Float u, Float s) {
    return -s * std::log(1 / u - 1);
}

PBRT_CPU_GPU
inline Float InvertLogisticSample(Float x, Float s) {
    return 1 / (1 + std::exp(-x / s));
}

PBRT_CPU_GPU
inline Float TrimmedLogisticPDF(Float x, Float s, Float a, Float b) {
    return Logistic(x, s) / (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

PBRT_CPU_GPU
inline Float SampleTrimmedLogistic(Float u, Float s, Float a, Float b) {
    DCHECK_LT(a, b);
    u = Lerp(u, InvertLogisticSample(a, s), InvertLogisticSample(b, s));
    Float x = SampleLogistic(u, s);
    DCHECK(!IsNaN(x));
    return Clamp(x, a, b);
}

PBRT_CPU_GPU
inline Float InvertTrimmedLogisticSample(Float x, Float s, Float a, Float b) {
    DCHECK(a <= x && x <= b);
    return (InvertLogisticSample(x, s) - InvertLogisticSample(a, s)) /
           (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

PBRT_CPU_GPU
inline Float SmoothStepPDF(Float x, Float a, Float b) {
    if (x < a || x > b)
        return 0;
    DCHECK_LT(a, b);
    return (2 / (b - a)) * SmoothStep(x, a, b);
}

PBRT_CPU_GPU
inline Float SampleSmoothStep(Float u, Float a, Float b) {
    DCHECK_LT(a, b);
    auto cdfMinusU = [=](Float x) -> std::pair<Float, Float> {
        Float t = (x - a) / (b - a);
        Float P = 2 * Pow<3>(t) - Pow<4>(t);
        Float PDeriv = SmoothStepPDF(x, a, b);
        return {P - u, PDeriv};
    };
    return NewtonBisection(a, b, cdfMinusU);
}

PBRT_CPU_GPU
inline Float InvertSmoothStepSample(Float x, Float a, Float b) {
    Float t = (x - a) / (b - a);
    auto P = [&](Float x) { return 2 * Pow<3>(t) - Pow<4>(t); };
    return (P(x) - P(a)) / (P(b) - P(a));
}

PBRT_CPU_GPU
inline Vector3f SampleUniformHemisphere(const Point2f &u) {
    Float z = u[0];
    Float r = SafeSqrt(1 - z * z);
    Float phi = 2 * Pi * u[1];
    return {r * std::cos(phi), r * std::sin(phi), z};
}

PBRT_CPU_GPU
inline Float UniformHemispherePDF() {
    return Inv2Pi;
}

PBRT_CPU_GPU
inline Point2f InvertUniformHemisphereSample(const Vector3f &v) {
    Float phi = std::atan2(v.y, v.x);
    if (phi < 0)
        phi += 2 * Pi;
    return Point2f(v.z, phi / (2 * Pi));
}

PBRT_CPU_GPU
inline Vector3f SampleUniformSphere(const Point2f &u) {
    Float z = 1 - 2 * u[0];
    Float r = SafeSqrt(1 - z * z);
    Float phi = 2 * Pi * u[1];
    return {r * std::cos(phi), r * std::sin(phi), z};
}

PBRT_CPU_GPU
inline Float UniformSpherePDF() {
    return Inv4Pi;
}

PBRT_CPU_GPU
inline Point2f InvertUniformSphereSample(const Vector3f &v) {
    Float phi = std::atan2(v.y, v.x);
    if (phi < 0)
        phi += 2 * Pi;
    return Point2f((1 - v.z) / 2, phi / (2 * Pi));
}

PBRT_CPU_GPU
inline Point2f SampleUniformDiskPolar(const Point2f &u) {
    Float r = std::sqrt(u[0]);
    Float theta = 2 * Pi * u[1];
    return {r * std::cos(theta), r * std::sin(theta)};
}

PBRT_CPU_GPU
inline Point2f InvertUniformDiskPolarSample(const Point2f &p) {
    Float phi = std::atan2(p.y, p.x);
    if (phi < 0)
        phi += 2 * Pi;
    return Point2f(Sqr(p.x) + Sqr(p.y), phi / (2 * Pi));
}

PBRT_CPU_GPU
inline Point2f SampleUniformDiskConcentric(const Point2f &u) {
    // Map _u_ to $[-1,1]^2$ and handle degeneracy at the origin
    Point2f uOffset = 2.f * u - Vector2f(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
        return {0, 0};

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * Point2f(std::cos(theta), std::sin(theta));
}

PBRT_CPU_GPU
inline Point2f InvertUniformDiskConcentricSample(const Point2f &p) {
    Float theta = std::atan2(p.y, p.x);  // -pi -> pi
    Float r = std::sqrt(Sqr(p.x) + Sqr(p.y));

    Point2f uo;
    // TODO: can we make this less branchy?
    if (std::abs(theta) < PiOver4 || std::abs(theta) > 3 * PiOver4) {
        uo.x = r = std::copysign(r, p.x);
        if (p.x < 0) {
            if (p.y < 0) {
                uo.y = (Pi + theta) * r / PiOver4;
            } else {
                uo.y = (theta - Pi) * r / PiOver4;
            }
        } else {
            uo.y = (theta * r) / PiOver4;
        }
    } else {
        uo.y = r = std::copysign(r, p.y);
        if (p.y < 0) {
            uo.x = -(PiOver2 + theta) * r / PiOver4;
        } else {
            uo.x = (PiOver2 - theta) * r / PiOver4;
        }
    }

    return {(uo.x + 1) / 2, (uo.y + 1) / 2};
}

PBRT_CPU_GPU
inline Vector3f SampleCosineHemisphere(const Point2f &u) {
    Point2f d = SampleUniformDiskConcentric(u);
    Float z = SafeSqrt(1 - d.x * d.x - d.y * d.y);
    return Vector3f(d.x, d.y, z);
}

PBRT_CPU_GPU
inline Float CosineHemispherePDF(Float cosTheta) {
    return cosTheta * InvPi;
}

PBRT_CPU_GPU
inline Point2f InvertCosineHemisphereSample(const Vector3f &v) {
    return InvertUniformDiskConcentricSample({v.x, v.y});
}

PBRT_CPU_GPU
inline Float UniformConePDF(Float cosThetaMax) {
    return 1 / (2 * Pi * (1 - cosThetaMax));
}

PBRT_CPU_GPU
inline Vector3f SampleUniformCone(const Point2f &u, Float cosThetaMax) {
    Float cosTheta = (1 - u[0]) + u[0] * cosThetaMax;
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    Float phi = u[1] * 2 * Pi;
    return SphericalDirection(sinTheta, cosTheta, phi);
}

PBRT_CPU_GPU
inline Point2f InvertUniformConeSample(const Vector3f &v, Float cosThetaMax) {
    Float cosTheta = v.z;
    Float phi = SphericalPhi(v);
    return {(cosTheta - 1) / (cosThetaMax - 1), phi / (2 * Pi)};
}

PBRT_CPU_GPU
inline Float BilinearPDF(Point2f p, pstd::span<const Float> w) {
    DCHECK_EQ(4, w.size());
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)
        return 0;
    if (w[0] + w[1] + w[2] + w[3] == 0)
        return 1;
    return 4 * Bilerp({p[0], p[1]}, w) / (w[0] + w[1] + w[2] + w[3]);
}

PBRT_CPU_GPU
inline Point2f SampleBilinear(Point2f u, pstd::span<const Float> w) {
    DCHECK_EQ(4, w.size());
    Point2f p;
    // Sample $v$ for bilinear marginal distribution
    Float v0 = w[0] + w[1], v1 = w[2] + w[3];
    p[1] = SampleLinear(u[1], v0, v1);

    // Sample $u$ for bilinear conditional distribution
    p[0] = SampleLinear(u[0], Lerp(p[1], w[0], w[2]), Lerp(p[1], w[1], w[3]));

    return p;
}

PBRT_CPU_GPU
inline Point2f InvertBilinearSample(Point2f p, pstd::span<const Float> v) {
    return {InvertLinearSample(p[0], Lerp(p[1], v[0], v[2]), Lerp(p[1], v[1], v[3])),
            InvertLinearSample(p[1], v[0] + v[1], v[2] + v[3])};
}

PBRT_CPU_GPU
inline Float BalanceHeuristic(int nf, Float fPdf, int ng, Float gPdf) {
    return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

PBRT_CPU_GPU
inline Float PowerHeuristic(int nf, Float fPdf, int ng, Float gPdf) {
    Float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

PBRT_CPU_GPU
inline Float XYZMatchingPDF(Float lambda) {
    if (lambda < 360 || lambda > 830)
        return 0;
    return 0.0039398042f / Sqr(std::cosh(0.0072f * (lambda - 538)));
}

PBRT_CPU_GPU
inline Float SampleXYZMatching(Float u) {
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}

PBRT_CPU_GPU inline pstd::array<Float, 3> SampleUniformTriangle(const Point2f &u) {
    Float b0, b1;
    if (u[0] < u[1]) {
        b0 = u[0] / 2;
        b1 = u[1] - b0;
    } else {
        b1 = u[1] / 2;
        b0 = u[0] - b1;
    }
    return {b0, b1, 1 - b0 - b1};
}

PBRT_CPU_GPU
inline Point2f InvertUniformTriangleSample(const pstd::array<Float, 3> &b) {
    if (b[0] > b[1]) {
        // b0 = u[0] - u[1] / 2, b1 = u[1] / 2
        return {b[0] + b[1], 2 * b[1]};
    } else {
        // b1 = u[1] - u[0] / 2, b0 = u[0] / 2
        return {2 * b[0], b[1] + b[0]};
    }
}

PBRT_CPU_GPU
inline Float TentPDF(Float x, Float radius) {
    if (std::abs(x) >= radius)
        return 0;
    return 1 / radius - std::abs(x) / Sqr(radius);
}

PBRT_CPU_GPU
inline Float SampleTent(Float u, Float radius) {
    if (SampleDiscrete({0.5f, 0.5f}, u, nullptr, &u) == 0)
        return -radius + radius * SampleLinear(u, 0, 1);
    else
        return radius * SampleLinear(u, 1, 0);
}

PBRT_CPU_GPU
inline Float InvertTentSample(Float x, Float radius) {
    if (x <= 0)
        return (1 - InvertLinearSample(-x / radius, 1, 0)) / 2;
    else
        return 0.5f + InvertLinearSample(x / radius, 1, 0) / 2;
}

PBRT_CPU_GPU
inline Float NormalPDF(Float x, Float mu = 0, Float sigma = 1) {
    return Gaussian(x, mu, sigma);
}

PBRT_CPU_GPU
inline Float SampleNormal(Float u, Float mu = 0, Float sigma = 1) {
    return mu + Sqrt2 * sigma * ErfInv(2 * u - 1);
}

PBRT_CPU_GPU
inline Float InvertNormalSample(Float x, Float mu = 0, Float sigma = 1) {
    return 0.5f * (1 + std::erf((x - mu) / (sigma * Sqrt2)));
}

PBRT_CPU_GPU
inline Point2f SampleTwoNormal(const Point2f &u, Float mu = 0, Float sigma = 1) {
    return {mu + sigma * std::sqrt(-2 * std::log(1 - u[0])) * std::cos(2 * Pi * u[1]),
            mu + sigma * std::sqrt(-2 * std::log(1 - u[0])) * std::sin(2 * Pi * u[1])};
}

PBRT_CPU_GPU
inline Vector3f SampleTrowbridgeReitz(Float alpha_x, Float alpha_y, const Point2f &u) {
    Float cosTheta = 0, phi = (2 * Pi) * u[1];
    if (alpha_x == alpha_y) {
        Float tanTheta2 = alpha_x * alpha_x * u[0] / (1 - u[0]);
        cosTheta = 1 / std::sqrt(1 + tanTheta2);
    } else {
        phi = std::atan(alpha_y / alpha_x * std::tan(2 * Pi * u[1] + .5f * Pi));
        if (u[1] > .5f)
            phi += Pi;
        Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
        Float alpha2 = 1 / (Sqr(cosPhi / alpha_x) + Sqr(sinPhi / alpha_y));
        Float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
        cosTheta = 1 / std::sqrt(1 + tanTheta2);
    }
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    return SphericalDirection(sinTheta, cosTheta, phi);
}

// Via Eric Heitz's jcgt sample code...
PBRT_CPU_GPU
inline Vector3f SampleTrowbridgeReitzVisibleArea(const Vector3f &w, Float alpha_x,
                                                 Float alpha_y, const Point2f &u) {
    // Section 3.2: transforming the view direction to the hemisphere
    // configuration
    Vector3f wh = Normalize(Vector3f(alpha_x * w.x, alpha_y * w.y, w.z));

    // Section 4.1: orthonormal basis. Can't use CoordinateSystem() since
    // T1 has to be in the tangent plane w.r.t. (0,0,1).
    Vector3f T1 =
        (wh.z < 0.99999f) ? Normalize(Cross(Vector3f(0, 0, 1), wh)) : Vector3f(1, 0, 0);
    Vector3f T2 = Cross(wh, T1);

    // Section 4.2: parameterization of the projected area
    Float r = std::sqrt(u[0]);
    Float phi = 2 * Pi * u[1];
    Float t1 = r * std::cos(phi), t2 = r * std::sin(phi);
    Float s = 0.5f * (1 + wh.z);
    t2 = (1 - s) * std::sqrt(1 - t1 * t1) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    Vector3f nh =
        t1 * T1 + t2 * T2 + std::sqrt(std::max<Float>(0, 1 - t1 * t1 - t2 * t2)) * wh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    CHECK_RARE(1e-6, nh.z == 0);
    return Normalize(
        Vector3f(alpha_x * nh.x, alpha_y * nh.y, std::max<Float>(1e-6f, nh.z)));
}

// Sample from e^(-c x), x from 0 to xMax
PBRT_CPU_GPU
inline Float SampleTrimmedExponential(Float u, Float c, Float xMax) {
    return std::log(1 - u * (1 - std::exp(-c * xMax))) / -c;
}

PBRT_CPU_GPU
inline Float TrimmedExponentialPDF(Float x, Float c, Float xMax) {
    if (x < 0 || x > xMax)
        return 0;
    return c / (1 - std::exp(-c * xMax)) * std::exp(-c * x);
}

PBRT_CPU_GPU
inline Float InvertTrimmedExponentialSample(Float x, Float c, Float xMax) {
    DCHECK(x >= 0 && x <= xMax);
    return (1 - std::exp(-c * x)) / (1 - std::exp(-c * xMax));
}

PBRT_CPU_GPU
inline Vector3f SampleUniformHemisphereConcentric(const Point2f &u) {
    // Map uniform random numbers to $[-1,1]^2$
    Point2f uOffset = 2.f * u - Vector2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0)
        return Vector3f(0, 0, 1);

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }

    return Vector3f(std::cos(theta) * r * std::sqrt(2 - r * r),
                    std::sin(theta) * r * std::sqrt(2 - r * r), 1 - r * r);
}

// VarianceEstimator Definition
template <typename Float = Float>
class VarianceEstimator {
  public:
    // VarianceEstimator Public Methods
    PBRT_CPU_GPU
    void Add(Float x) {
        ++n;
        Float delta = x - mean;
        mean += delta / n;
        Float delta2 = x - mean;
        S += delta * delta2;
    }

    PBRT_CPU_GPU
    Float Mean() const { return mean; }
    PBRT_CPU_GPU
    Float Variance() const { return (n > 1) ? S / (n - 1) : 0; }
    PBRT_CPU_GPU
    int64_t Count() const { return n; }
    PBRT_CPU_GPU
    Float RelativeVariance() const {
        return (n < 1 || mean == 0) ? 0 : Variance() / Mean();
    }

    PBRT_CPU_GPU
    void Merge(const VarianceEstimator &ve) {
        if (ve.n == 0)
            return;
        S = S + ve.S + Sqr(ve.mean - mean) * n * ve.n / (n + ve.n);
        mean = (n * mean + ve.n * ve.mean) / (n + ve.n);
        n += ve.n;
    }

  private:
    // VarianceEstimator Private Members
    Float mean = 0, S = 0;
    int64_t n = 0;
};

// WeightedReservoirSampler Definition
template <typename T>
class WeightedReservoirSampler {
  public:
    // WeightedReservoirSampler Public Methods
    WeightedReservoirSampler() = default;
    PBRT_CPU_GPU
    WeightedReservoirSampler(uint64_t rngSeed) : rng(rngSeed) {}

    PBRT_CPU_GPU
    void Seed(uint64_t seed) { rng.SetSequence(seed); }

    PBRT_CPU_GPU
    void Add(const T &sample, Float weight) {
        weightSum += weight;
        // Randomly add _sample_ to reservoir
        Float p = weight / weightSum;
        if (rng.Uniform<Float>() < p) {
            reservoir = sample;
            reservoirWeight = weight;
        }

        DCHECK_LT(weightSum, 1e80);
    }

    template <typename F>
    PBRT_CPU_GPU void Add(F func, Float weight) {
        // Process weighted reservoir sample via callback
        weightSum += weight;
        Float p = weight / weightSum;
        if (rng.Uniform<Float>() < p) {
            reservoir = func();
            reservoirWeight = weight;
        }
        DCHECK_LT(weightSum, 1e80);
    }

    PBRT_CPU_GPU
    void Copy(const WeightedReservoirSampler &wrs) {
        weightSum = wrs.weightSum;
        reservoir = wrs.reservoir;
        reservoirWeight = wrs.reservoirWeight;
    }

    PBRT_CPU_GPU
    int HasSample() const { return weightSum > 0; }
    PBRT_CPU_GPU
    const T &GetSample() const { return reservoir; }
    PBRT_CPU_GPU
    Float SamplePDF() const { return reservoirWeight / weightSum; }
    PBRT_CPU_GPU
    Float WeightSum() const { return weightSum; }

    PBRT_CPU_GPU
    void Reset() { reservoirWeight = weightSum = 0; }

    PBRT_CPU_GPU
    void Merge(const WeightedReservoirSampler &wrs) {
        DCHECK_LE(weightSum + wrs.WeightSum(), 1e80);
        if (wrs.HasSample()) {
            Add(wrs.reservoir, wrs.weightSum);
        }
    }

    std::string ToString() const {
        return StringPrintf("[ WeightedReservoirSampler rng: %s "
                            "weightSum: %f reservoir: %s reservoirWeight: %f ]",
                            rng, weightSum, reservoir, reservoirWeight);
    }

  private:
    // WeightedReservoirSampler Private Members
    RNG rng;
    Float weightSum = 0;
    Float reservoirWeight = 0;
    T reservoir;
};

// PiecewiseConstant1D Definition
class PiecewiseConstant1D {
  public:
    // PiecewiseConstant1D Public Methods
    PBRT_CPU_GPU
    size_t BytesUsed() const {
        return (func.capacity() + cdf.capacity()) * sizeof(Float);
    }

    static void TestCompareDistributions(const PiecewiseConstant1D &da,
                                         const PiecewiseConstant1D &db, Float eps = 1e-5);

    std::string ToString() const {
        return StringPrintf("[ PiecewiseConstant1D func: %s cdf: %s "
                            "min: %f max: %f funcInt: %f ]",
                            func, cdf, min, max, funcInt);
    }

    PiecewiseConstant1D() = default;
    PiecewiseConstant1D(Allocator alloc) : func(alloc), cdf(alloc) {}
    PiecewiseConstant1D(pstd::span<const Float> f, Allocator alloc = {})
        : PiecewiseConstant1D(f, 0., 1., alloc) {}

    PiecewiseConstant1D(pstd::span<const Float> f, Float min, Float max,
                        Allocator alloc = {})
        : func(f.begin(), f.end(), alloc), cdf(f.size() + 1, alloc), min(min), max(max) {
        CHECK_GT(max, min);
        // Compute integral of step function at $x_i$
        cdf[0] = 0;
        size_t n = f.size();
        for (size_t i = 1; i < n + 1; ++i) {
            CHECK_GE(func[i - 1], 0);
            cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n;
        }

        // Transform step function integral into CDF
        funcInt = cdf[n];
        if (funcInt == 0)
            for (size_t i = 1; i < n + 1; ++i)
                cdf[i] = Float(i) / Float(n);
        else
            for (size_t i = 1; i < n + 1; ++i)
                cdf[i] /= funcInt;
    }

    PBRT_CPU_GPU
    size_t size() const { return func.size(); }

    PBRT_CPU_GPU
    Float Sample(Float u, Float *pdf = nullptr, int *off = nullptr) const {
        // Find surrounding CDF segments and _offset_
        int offset =
            FindInterval((int)cdf.size(), [&](int index) { return cdf[index] <= u; });
        if (off)
            *off = offset;

        // Compute offset along CDF segment
        Float du = u - cdf[offset];
        if (cdf[offset + 1] - cdf[offset] > 0)
            du /= cdf[offset + 1] - cdf[offset];
        DCHECK(!IsNaN(du));

        // Compute PDF for sampled offset
        if (pdf != nullptr)
            *pdf = (funcInt > 0) ? func[offset] / funcInt : 0;

        // Return $x$ corresponding to sample
        return Lerp((offset + du) / size(), min, max);
    }

    PBRT_CPU_GPU
    pstd::optional<Float> Invert(Float x) const {
        // Compute offset to CDF values that bracket $x$
        if (x < min || x > max)
            return {};
        Float c = (x - min) / (max - min) * func.size();
        int offset = Clamp(int(c), 0, func.size() - 1);
        DCHECK(offset >= 0 && offset + 1 < cdf.size());

        // Linearly interpolate between adjacent CDF values to find sample value
        Float delta = c - offset;
        return Lerp(delta, cdf[offset], cdf[offset + 1]);
    }

    // PiecewiseConstant1D Public Members
    pstd::vector<Float> func, cdf;
    Float min, max;
    Float funcInt = 0;
};

// PiecewiseConstant2D Definition
class PiecewiseConstant2D {
  public:
    // PiecewiseConstant2D Public Methods
    PiecewiseConstant2D() = default;
    PiecewiseConstant2D(Allocator alloc) : pConditionalY(alloc), pMarginal(alloc) {}
    PiecewiseConstant2D(pstd::span<const Float> data, int nx, int ny,
                        Allocator alloc = {})
        : PiecewiseConstant2D(data, nx, ny, Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                              alloc) {}
    explicit PiecewiseConstant2D(const Array2D<Float> &data, Allocator alloc = {})
        : PiecewiseConstant2D(pstd::span<const Float>(data), data.xSize(), data.ySize(),
                              alloc) {}
    PiecewiseConstant2D(const Array2D<Float> &data, Bounds2f domain, Allocator alloc = {})
        : PiecewiseConstant2D(pstd::span<const Float>(data), data.xSize(), data.ySize(),
                              domain, alloc) {}

    PBRT_CPU_GPU
    size_t BytesUsed() const {
        return pConditionalY.size() *
                   (pConditionalY[0].BytesUsed() + sizeof(pConditionalY[0])) +
               pMarginal.BytesUsed();
    }

    PBRT_CPU_GPU
    Bounds2f Domain() const { return domain; }

    PBRT_CPU_GPU
    Point2i Resolution() const {
        return {int(pConditionalY[0].size()), int(pMarginal.size())};
    }

    std::string ToString() const {
        return StringPrintf("[ PiecewiseConstant2D domain: %s pConditionalY: %s "
                            "pMarginal: %s ]",
                            domain, pConditionalY, pMarginal);
    }
    static void TestCompareDistributions(const PiecewiseConstant2D &da,
                                         const PiecewiseConstant2D &db, Float eps = 1e-5);

    PiecewiseConstant2D(pstd::span<const Float> func, int nx, int ny, Bounds2f domain,
                        Allocator alloc = {})
        : domain(domain), pConditionalY(alloc), pMarginal(alloc) {
        CHECK_EQ(func.size(), (size_t)nx * (size_t)ny);
        pConditionalY.reserve(ny);
        for (int y = 0; y < ny; ++y)
            // Compute conditional sampling distribution for $\tilde{y}$
            // TODO: emplace_back is key so the alloc sticks. WHY?
            pConditionalY.emplace_back(func.subspan(y * nx, nx), domain.pMin[0],
                                       domain.pMax[0], alloc);

        // Compute marginal sampling distribution $p[\tilde{y}]$
        std::vector<Float> marginalFunc;
        marginalFunc.reserve(ny);
        for (int y = 0; y < ny; ++y)
            marginalFunc.push_back(pConditionalY[y].funcInt);
        pMarginal =
            PiecewiseConstant1D(marginalFunc, domain.pMin[1], domain.pMax[1], alloc);
    }

    PBRT_CPU_GPU
    Point2f Sample(const Point2f &u, Float *pdf = nullptr) const {
        Float pdfs[2];
        int y;
        Float d1 = pMarginal.Sample(u[1], &pdfs[1], &y);
        Float d0 = pConditionalY[y].Sample(u[0], &pdfs[0]);
        if (pdf != nullptr)
            *pdf = pdfs[0] * pdfs[1];
        return Point2f(d0, d1);
    }

    PBRT_CPU_GPU
    Float PDF(const Point2f &pr) const {
        Point2f p = Point2f(domain.Offset(pr));
        int ix =
            Clamp(int(p[0] * pConditionalY[0].size()), 0, pConditionalY[0].size() - 1);
        int iy = Clamp(int(p[1] * pMarginal.size()), 0, pMarginal.size() - 1);
        return pConditionalY[iy].func[ix] / pMarginal.funcInt;
    }

    PBRT_CPU_GPU
    pstd::optional<Point2f> Invert(const Point2f &p) const {
        pstd::optional<Float> mInv = pMarginal.Invert(p[1]);
        if (!mInv)
            return {};
        Float p1o = (p[1] - domain.pMin[1]) / (domain.pMax[1] - domain.pMin[1]);
        if (p1o < 0 || p1o > 1)
            return {};
        int offset = Clamp(p1o * pConditionalY.size(), 0, pConditionalY.size() - 1);
        pstd::optional<Float> cInv = pConditionalY[offset].Invert(p[0]);
        if (!cInv)
            return {};
        return Point2f(*cInv, *mInv);
    }

  private:
    // PiecewiseConstant2D Private Members
    Bounds2f domain;
    pstd::vector<PiecewiseConstant1D> pConditionalY;
    PiecewiseConstant1D pMarginal;
};

// AliasTable Definition
class AliasTable {
  public:
    // AliasTable Public Methods
    AliasTable() = default;
    AliasTable(Allocator alloc = {}) : bins(alloc) {}
    AliasTable(pstd::span<const Float> weights, Allocator alloc = {});

    PBRT_CPU_GPU
    int Sample(Float u, Float *pdf = nullptr, Float *uRemapped = nullptr) const;
    std::string ToString() const;

    PBRT_CPU_GPU
    size_t size() const { return bins.size(); }
    PBRT_CPU_GPU
    Float PDF(int index) const { return bins[index].pdf; }

  private:
    // AliasTable Private Members
    struct Bin {
        Float q, pdf;
        int alias;
    };
    pstd::vector<Bin> bins;
};

// SummedAreaTable Definition
class SummedAreaTable {
  public:
    // SummedAreaTable Public Methods
    SummedAreaTable(Allocator alloc) : sum(alloc) {}
    SummedAreaTable(const Array2D<Float> &values, Allocator alloc = {})
        : sum(integrate(values, alloc)) {}

    PBRT_CPU_GPU
    Float Sum(const Bounds2f &extent) const {
        double s = ((lookup(extent.pMax.x, extent.pMax.y) -
                     lookup(extent.pMin.x, extent.pMax.y)) +
                    (lookup(extent.pMin.x, extent.pMin.y) -
                     lookup(extent.pMax.x, extent.pMin.y)));
        return std::max<Float>(s, 0);
    }

    PBRT_CPU_GPU
    Float Average(const Bounds2f &extent) const { return Sum(extent) / extent.Area(); }

    std::string ToString() const;

  private:
    // SummedAreaTable Private Methods
    Array2D<double> integrate(const Array2D<Float> &values, Allocator alloc) {
        auto f = [&values](int x, int y) {
            return values(x, y) / (values.xSize() * values.ySize());
        };
        Array2D<double> result(values.xSize(), values.ySize(), alloc);
        result(0, 0) = f(0, 0);
        // Compute sums along first scanline and column
        for (int x = 1; x < result.xSize(); ++x)
            result(x, 0) = f(x, 0) + result(x - 1, 0);
        for (int y = 1; y < result.ySize(); ++y)
            result(0, y) = f(0, y) + result(0, y - 1);

        // Compute sums for the remainder of the entries
        for (int y = 1; y < result.ySize(); ++y)
            for (int x = 1; x < result.xSize(); ++x)
                result(x, y) = (f(x, y) + result(x - 1, y) + result(x, y - 1) -
                                result(x - 1, y - 1));

        return result;
    }

    PBRT_CPU_GPU
    double lookup(Float x, Float y) const {
        // Rescale $(x,y)$ to table resolution and compute integer coordinates
        x *= sum.xSize();
        y *= sum.ySize();
        int x0 = (int)x, y0 = (int)y;

        // Bilinearly interpolate between surrounding table values
        Float v00 = lookup(x0, y0), v10 = lookup(x0 + 1, y0);
        Float v01 = lookup(x0, y0 + 1), v11 = lookup(x0 + 1, y0 + 1);
        Float dx = x - int(x), dy = y - int(y);
        return (1 - dx) * (1 - dy) * v00 + (1 - dx) * dy * v01 + dx * (1 - dy) * v10 +
               dx * dy * v11;
    }

    PBRT_CPU_GPU
    double lookup(int x, int y) const {
        // Return zero at lower boundaries
        if (x == 0 || y == 0)
            return 0;

        // Reindex $(x,y)$ and return actual stored value
        x = std::min(x - 1, sum.xSize() - 1);
        y = std::min(y - 1, sum.ySize() - 1);
        return sum(x, y);
    }

    // SummedAreaTable Private Members
    Array2D<double> sum;
};

// WindowedPiecewiseConstant2D Definition
class WindowedPiecewiseConstant2D {
  public:
    // WindowedPiecewiseConstant2D Public Methods
    WindowedPiecewiseConstant2D(Allocator alloc) : sat(alloc), func(alloc) {}
    WindowedPiecewiseConstant2D(Array2D<Float> f, Allocator alloc = {})
        : sat(f, alloc), func(f, alloc) {}

    PBRT_CPU_GPU
    Point2f Sample(const Point2f &u, const Bounds2f &b, Float *pdf) const {
        // Handle zero-valued function for windowed sampling
        if (sat.Sum(b) == 0) {
            *pdf = 0;
            return {};
        }

        // Sample marginal windowed function in the first dimension
        Float sumb = sat.Sum(b);
        auto Px = [&, this](Float x) -> Float {
            Bounds2f bx = b;
            bx.pMax.x = x;
            return sat.Sum(bx) / sumb;
        };
        Point2f p;
        int nx = func.xSize();
        p.x = sample(Px, u[0], b.pMin.x, b.pMax.x, nx);

        // Sample conditional windowed function in the second dimension
        Bounds2f by(Point2f(std::floor(p.x * nx) / nx, b.pMin.y),
                    Point2f(std::ceil(p.x * nx) / nx, b.pMax.y));
        if (by.pMin.x == by.pMax.x)
            by.pMax.x += 1.f / nx;
        if (sat.Sum(by) == 0) {
            // This can happen when we're provided a really narrow initial
            // bounding box
            *pdf = 0;
            return {};
        }
        Float sumby = sat.Sum(by);
        auto Py = [&, this](Float y) -> Float {
            Bounds2f byy = by;
            byy.pMax.y = y;
            return sat.Sum(byy) / sumby;
        };
        p.y = sample(Py, u[1], b.pMin.y, b.pMax.y, func.ySize());

        // Compute PDF and return point sampled from windowed function
        *pdf = PDF(p, b);
        return p;
    }

    PBRT_CPU_GPU
    Float PDF(const Point2f &p, const Bounds2f &b) const {
        if (sat.Sum(b) == 0)
            return 0;
        return Eval(p) / sat.Sum(b);
    }

  private:
    // WindowedPiecewiseConstant2D Private Methods
    PBRT_CPU_GPU
    Float Eval(const Point2f &p) const {
        Point2i pi(std::min<int>(p[0] * func.xSize(), func.xSize() - 1),
                   std::min<int>(p[1] * func.ySize(), func.ySize() - 1));
        return func[pi];
    }

    template <typename F>
    PBRT_CPU_GPU static Float sample(F func, Float u, Float min, Float max, int n) {
        // Apply bisection to bracket _u_
        while (std::ceil(n * max) - std::floor(n * min) > 1) {
            DCHECK_LE(func(min), u);
            DCHECK_GE(func(max), u);
            Float mid = (min + max) / 2;
            if (func(mid) > u)
                max = mid;
            else
                min = mid;
        }

        // Find sample by interpolating between _min_ and _max_
        Float t = (u - func(min)) / (func(max) - func(min));
        return Clamp(Lerp(t, min, max), min, max);
    }

    // WindowedPiecewiseConstant2D Private Members
    SummedAreaTable sat;
    Array2D<Float> func;
};

pstd::vector<Float> Sample1DFunction(std::function<Float(Float)> f, int nSteps,
                                     int nSamples, Float min = 0, Float max = 1,
                                     Allocator alloc = {});

Array2D<Float> Sample2DFunction(std::function<Float(Float, Float)> f, int nu, int nv,
                                int nSamples,
                                Bounds2f domain = {Point2f(0, 0), Point2f(1, 1)},
                                Allocator alloc = {});

namespace detail {

template <typename Iterator>
class IndexingIterator {
  public:
    template <typename Generator>
    PBRT_CPU_GPU IndexingIterator(int i, int n, const Generator *) : i(i), n(n) {}

    PBRT_CPU_GPU
    bool operator==(const Iterator &it) const { return i == it.i; }
    PBRT_CPU_GPU
    bool operator!=(const Iterator &it) const { return !(*this == it); }
    PBRT_CPU_GPU
    Iterator &operator++() {
        ++i;
        return (Iterator &)*this;
    }
    PBRT_CPU_GPU
    Iterator operator++(int) const {
        Iterator it = *this;
        return ++it;
    }

  protected:
    int i, n;
};

template <typename Generator, typename Iterator>
class IndexingGenerator {
  public:
    PBRT_CPU_GPU
    IndexingGenerator(int n) : n(n) {}
    PBRT_CPU_GPU
    Iterator begin() const { return Iterator(0, n, (const Generator *)this); }
    PBRT_CPU_GPU
    Iterator end() const { return Iterator(n, n, (const Generator *)this); }

  protected:
    int n;
};

class Uniform1DIter;
class Uniform2DIter;
class Uniform3DIter;
class Hammersley2DIter;
class Hammersley3DIter;
class Stratified1DIter;
class Stratified2DIter;
class Stratified3DIter;
template <typename Iterator>
class RNGIterator;

template <typename Generator, typename Iterator>
class RNGGenerator : public IndexingGenerator<Generator, Iterator> {
  public:
    PBRT_CPU_GPU
    RNGGenerator(int n, uint64_t sequenceIndex = 0, uint64_t seed = PCG32_DEFAULT_STATE)
        : IndexingGenerator<Generator, Iterator>(n),
          sequenceIndex(sequenceIndex),
          seed(seed) {}

  protected:
    friend RNGIterator<Iterator>;
    uint64_t sequenceIndex, seed;
};

template <typename Iterator>
class RNGIterator : public IndexingIterator<Iterator> {
  public:
    template <typename Generator>
    PBRT_CPU_GPU RNGIterator(int i, int n,
                             const RNGGenerator<Generator, Iterator> *generator)
        : IndexingIterator<Iterator>(i, n, generator), rng(generator->sequenceIndex) {}

  protected:
    RNG rng;
};

}  // namespace detail

class Uniform1D : public detail::RNGGenerator<Uniform1D, detail::Uniform1DIter> {
  public:
    using detail::RNGGenerator<Uniform1D, detail::Uniform1DIter>::RNGGenerator;
};

class Uniform2D : public detail::RNGGenerator<Uniform2D, detail::Uniform2DIter> {
  public:
    using detail::RNGGenerator<Uniform2D, detail::Uniform2DIter>::RNGGenerator;
};

class Uniform3D : public detail::RNGGenerator<Uniform3D, detail::Uniform3DIter> {
  public:
    using detail::RNGGenerator<Uniform3D, detail::Uniform3DIter>::RNGGenerator;
};

class Hammersley2D
    : public detail::IndexingGenerator<Hammersley2D, detail::Hammersley2DIter> {
  public:
    using detail::IndexingGenerator<Hammersley2D,
                                    detail::Hammersley2DIter>::IndexingGenerator;
};

class Hammersley3D
    : public detail::IndexingGenerator<Hammersley3D, detail::Hammersley3DIter> {
  public:
    using detail::IndexingGenerator<Hammersley3D,
                                    detail::Hammersley3DIter>::IndexingGenerator;
};

class Stratified1D : public detail::RNGGenerator<Stratified1D, detail::Stratified1DIter> {
  public:
    using detail::RNGGenerator<Stratified1D, detail::Stratified1DIter>::RNGGenerator;
};

class Stratified2D : public detail::RNGGenerator<Stratified2D, detail::Stratified2DIter> {
  public:
    PBRT_CPU_GPU
    Stratified2D(int nx, int ny, uint64_t sequenceIndex = 0,
                 uint64_t seed = PCG32_DEFAULT_STATE)
        : detail::RNGGenerator<Stratified2D, detail::Stratified2DIter>(
              nx * ny, sequenceIndex, seed),
          nx(nx),
          ny(ny) {}

  private:
    friend detail::Stratified2DIter;
    int nx, ny;
};

class Stratified3D : public detail::RNGGenerator<Stratified3D, detail::Stratified3DIter> {
  public:
    PBRT_CPU_GPU
    Stratified3D(int nx, int ny, int nz, uint64_t sequenceIndex = 0,
                 uint64_t seed = PCG32_DEFAULT_STATE)
        : detail::RNGGenerator<Stratified3D, detail::Stratified3DIter>(
              nx * ny * nz, sequenceIndex, seed),
          nx(nx),
          ny(ny),
          nz(nz) {}

  private:
    friend detail::Stratified3DIter;
    int nx, ny, nz;
};

namespace detail {

class Uniform1DIter : public RNGIterator<Uniform1DIter> {
  public:
    using RNGIterator<Uniform1DIter>::RNGIterator;
    PBRT_CPU_GPU
    Float operator*() { return rng.Uniform<Float>(); }
};

class Uniform2DIter : public RNGIterator<Uniform2DIter> {
  public:
    using RNGIterator<Uniform2DIter>::RNGIterator;
    PBRT_CPU_GPU
    Point2f operator*() { return {rng.Uniform<Float>(), rng.Uniform<Float>()}; }
};

class Uniform3DIter : public RNGIterator<Uniform3DIter> {
  public:
    using RNGIterator<Uniform3DIter>::RNGIterator;
    PBRT_CPU_GPU
    Point3f operator*() {
        return {rng.Uniform<Float>(), rng.Uniform<Float>(), rng.Uniform<Float>()};
    }
};

class Stratified1DIter : public RNGIterator<Stratified1DIter> {
  public:
    using RNGIterator<Stratified1DIter>::RNGIterator;
    PBRT_CPU_GPU
    Float operator*() { return (i + rng.Uniform<Float>()) / n; }
};

class Stratified2DIter : public RNGIterator<Stratified2DIter> {
  public:
    PBRT_CPU_GPU
    Stratified2DIter(int i, int n, const Stratified2D *generator)
        : RNGIterator<Stratified2DIter>(i, n, generator),
          nx(generator->nx),
          ny(generator->ny) {}

    PBRT_CPU_GPU
    Point2f operator*() {
        int ix = i % nx, iy = i / nx;
        return {(ix + rng.Uniform<Float>()) / nx, (iy + rng.Uniform<Float>()) / ny};
    }

  private:
    int nx, ny;
};

class Stratified3DIter : public RNGIterator<Stratified3DIter> {
  public:
    PBRT_CPU_GPU
    Stratified3DIter(int i, int n, const Stratified3D *generator)
        : RNGIterator<Stratified3DIter>(i, n, generator),
          nx(generator->nx),
          ny(generator->ny),
          nz(generator->nz) {}

    PBRT_CPU_GPU
    Point3f operator*() {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);
        return {(ix + rng.Uniform<Float>()) / nx, (iy + rng.Uniform<Float>()) / ny,
                (iz + rng.Uniform<Float>()) / nz};
    }

  private:
    int nx, ny, nz;
};

class Hammersley2DIter : public IndexingIterator<Hammersley2DIter> {
  public:
    using IndexingIterator<Hammersley2DIter>::IndexingIterator;
    PBRT_CPU_GPU
    Point2f operator*() { return {Float(i) / Float(n), RadicalInverse(0, i)}; }
};

class Hammersley3DIter : public IndexingIterator<Hammersley3DIter> {
  public:
    using IndexingIterator<Hammersley3DIter>::IndexingIterator;
    PBRT_CPU_GPU
    Point3f operator*() {
        return {Float(i) / Float(n), RadicalInverse(0, i), RadicalInverse(1, i)};
    }
};

}  // namespace detail

// Both PiecewiseConstant2D and Hierarchical2DWarp work for the warp here
#if 0
template <typename W>
Image WarpedStrataVisualization(const W &warp, int xs = 16, int ys = 16) {
    Image im(PixelFormat::Half, {warp.Resolution().x / 2, warp.Resolution().y / 2}, { "R", "G", "B" });
    for (int y = 0; y < im.Resolution().y; ++y) {
        for (int x = 0; x < im.Resolution().x; ++x) {
            Point2f target = warp.Domain().Lerp({(x + .5f) / im.Resolution().x,
                                                 (y + .5f) / im.Resolution().y});
            if (warp.PDF(target) == 0) continue;

            pstd::optional<Point2f> u = warp.Invert(target);
            if (!u.has_value()) {
#if 0
                LOG(WARNING) << "No value at target " << target << ", though cont pdf = " <<
                    tabdist.PDF(target);
#endif
                continue;
            }

#if 1
            int tile = int(u->x * xs) + xs * int(u->y * ys);
            Float rgb[3] = { RadicalInverse(0, tile), RadicalInverse(1, tile),
                             RadicalInverse(2, tile) };
            im.SetChannels({x, int(y)}, {rgb[0], rgb[1], rgb[2]});
#else
            Float gray = ((int(u->x * xs) + int(u->y * ys)) & 1) ? 0.8 : 0.2;
            im.SetChannel({x, int(y)}, 0, gray);
#endif
        }
    }
    return im;
}
#endif

// PiecewiseLinear2D Implementation
// *****************************************************************************
// Marginal-conditional warp
// *****************************************************************************

/**
 * \brief Implements a marginal sample warping scheme for 2D distributions
 * with linear interpolation and an optional dependence on additional parameters
 *
 * This class takes a rectangular floating point array as input and constructs
 * internal data structures to efficiently map uniform variates from the unit
 * square <tt>[0, 1]^2</tt> to a function on <tt>[0, 1]^2</tt> that linearly
 * interpolates the input array.
 *
 * The mapping is constructed via the inversion method, which is applied to
 * a marginal distribution over rows, followed by a conditional distribution
 * over columns.
 *
 * The implementation also supports <em>conditional distributions</em>, i.e. 2D
 * distributions that depend on an arbitrary number of parameters (indicated
 * via the \c Dimension template parameter).
 *
 * In this case, the input array should have dimensions <tt>N0 x N1 x ... x Nn
 * x res[1] x res[0]</tt> (where the last dimension is contiguous in memory),
 * and the <tt>param_res</tt> should be set to <tt>{ N0, N1, ..., Nn }</tt>,
 * and <tt>param_values</tt> should contain the parameter values where the
 * distribution is discretized. Linear interpolation is used when sampling or
 * evaluating the distribution for in-between parameter values.
 */
template <size_t Dimension = 0>
class PiecewiseLinear2D {
  private:
    using FloatStorage = pstd::vector<float>;

#if !defined(_MSC_VER) && !defined(__CUDACC__)
    static constexpr size_t ArraySize = Dimension;
#else
    static constexpr size_t ArraySize = (Dimension != 0) ? Dimension : 1;
#endif

  public:
    PiecewiseLinear2D(Allocator alloc)
        : m_param_values(alloc),
          m_data(alloc),
          m_marginal_cdf(alloc),
          m_conditional_cdf(alloc) {
        for (int i = 0; i < ArraySize; ++i)
            m_param_values.emplace_back(alloc);
    }

    /**
     * Construct a marginal sample warping scheme for floating point
     * data of resolution \c size.
     *
     * \c param_res and \c param_values are only needed for conditional
     * distributions (see the text describing the PiecewiseLinear2D class).
     *
     * If \c normalize is set to \c false, the implementation will not
     * re-scale the distribution so that it integrates to \c 1. It can
     * still be sampled (proportionally), but returned density values
     * will reflect the unnormalized values.
     *
     * If \c build_cdf is set to \c false, the implementation will not
     * construct the cdf needed for sample warping, which saves memory in case
     * this functionality is not needed (e.g. if only the interpolation in \c
     * eval() is used).
     */
    PiecewiseLinear2D(Allocator alloc, const float *data, int xSize, int ySize,
                      pstd::array<int, Dimension> param_res = {},
                      pstd::array<const float *, Dimension> param_values = {},
                      bool normalize = true, bool build_cdf = true)
        : m_size(xSize, ySize),
          m_patch_size(1.f / (xSize - 1), 1.f / (ySize - 1)),
          m_inv_patch_size(m_size - Vector2i(1, 1)),
          m_param_values(alloc),
          m_data(alloc),
          m_marginal_cdf(alloc),
          m_conditional_cdf(alloc) {
        if (build_cdf && !normalize)
            LOG_FATAL("PiecewiseLinear2D: build_cdf implies normalize=true");

        /* Keep track of the dependence on additional parameters (optional) */
        uint32_t slices = 1;
        for (int i = 0; i < ArraySize; ++i)
            m_param_values.emplace_back(alloc);
        for (int i = (int)Dimension - 1; i >= 0; --i) {
            if (param_res[i] < 1)
                LOG_FATAL("PiecewiseLinear2D(): parameter resolution must be >= 1!");

            m_param_size[i] = param_res[i];
            m_param_values[i] = FloatStorage(param_res[i]);
            memcpy(m_param_values[i].data(), param_values[i],
                   sizeof(float) * param_res[i]);
            m_param_strides[i] = param_res[i] > 1 ? slices : 0;
            slices *= m_param_size[i];
        }

        uint32_t n_values = xSize * ySize;

        m_data = FloatStorage(slices * n_values);

        if (build_cdf) {
            m_marginal_cdf = FloatStorage(slices * m_size.y);
            m_conditional_cdf = FloatStorage(slices * n_values);

            float *marginal_cdf = m_marginal_cdf.data(),
                  *conditional_cdf = m_conditional_cdf.data(), *data_out = m_data.data();

            for (uint32_t slice = 0; slice < slices; ++slice) {
                /* Construct conditional CDF */
                for (int y = 0; y < m_size.y; ++y) {
                    double sum = 0.0;
                    size_t i = y * xSize;
                    conditional_cdf[i] = 0.f;
                    for (int x = 0; x < m_size.x - 1; ++x, ++i) {
                        sum += .5 * ((double)data[i] + (double)data[i + 1]);
                        conditional_cdf[i + 1] = (float)sum;
                    }
                }

                /* Construct marginal CDF */
                marginal_cdf[0] = 0.f;
                double sum = 0.0;
                for (int y = 0; y < m_size.y - 1; ++y) {
                    sum += .5 * ((double)conditional_cdf[(y + 1) * xSize - 1] +
                                 (double)conditional_cdf[(y + 2) * xSize - 1]);
                    marginal_cdf[y + 1] = (float)sum;
                }

                /* Normalize CDFs and PDF (if requested) */
                float normalization = 1.f / marginal_cdf[m_size.y - 1];
                for (size_t i = 0; i < n_values; ++i)
                    conditional_cdf[i] *= normalization;
                for (size_t i = 0; i < m_size.y; ++i)
                    marginal_cdf[i] *= normalization;
                for (size_t i = 0; i < n_values; ++i)
                    data_out[i] = data[i] * normalization;

                marginal_cdf += m_size.y;
                conditional_cdf += n_values;
                data_out += n_values;
                data += n_values;
            }
        } else {
            float *data_out = m_data.data();

            for (uint32_t slice = 0; slice < slices; ++slice) {
                float normalization = 1.f / HProd(m_inv_patch_size);
                if (normalize) {
                    double sum = 0.0;
                    for (uint32_t y = 0; y < m_size.y - 1; ++y) {
                        size_t i = y * xSize;
                        for (uint32_t x = 0; x < m_size.x - 1; ++x, ++i) {
                            float v00 = data[i], v10 = data[i + 1], v01 = data[i + xSize],
                                  v11 = data[i + 1 + xSize],
                                  avg = .25f * (v00 + v10 + v01 + v11);
                            sum += (double)avg;
                        }
                    }
                    normalization = float(1.0 / sum);
                }

                for (uint32_t k = 0; k < n_values; ++k)
                    data_out[k] = data[k] * normalization;

                data += n_values;
                data_out += n_values;
            }
        }
    }

    struct PLSample {
        Vector2f p;
        float pdf;
    };

    /**
     * \brief Given a uniformly distributed 2D sample, draw a sample from the
     * distribution (parameterized by \c param if applicable)
     *
     * Returns the warped sample and associated probability density.
     */
    PBRT_CPU_GPU
    PLSample Sample(Vector2f sample, const Float *param = nullptr) const {
        /* Avoid degeneracies at the extrema */
        sample[0] = Clamp(sample[0], 1 - OneMinusEpsilon, OneMinusEpsilon);
        sample[1] = Clamp(sample[1], 1 - OneMinusEpsilon, OneMinusEpsilon);

        /* Look up parameter-related indices and weights (if Dimension != 0) */
        float param_weight[2 * ArraySize];
        uint32_t slice_offset = 0u;
        for (size_t dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index = FindInterval(m_param_size[dim], [&](uint32_t idx) {
                return m_param_values[dim].data()[idx] <= param[dim];
            });

            float p0 = m_param_values[dim][param_index],
                  p1 = m_param_values[dim][param_index + 1];

            param_weight[2 * dim + 1] = Clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Sample the row first */
        uint32_t offset = 0;
        if (Dimension != 0)
            offset = slice_offset * m_size.y;

        auto fetch_marginal = [&](uint32_t idx) -> float {
            return lookup<Dimension>(m_marginal_cdf.data(), offset + idx, m_size.y,
                                     param_weight);
        };

        uint32_t row = FindInterval(
            m_size.y, [&](uint32_t idx) { return fetch_marginal(idx) < sample.y; });

        sample.y -= fetch_marginal(row);

        uint32_t slice_size = HProd(m_size);
        offset = row * m_size.x;
        if (Dimension != 0)
            offset += slice_offset * slice_size;

        float r0 = lookup<Dimension>(m_conditional_cdf.data(), offset + m_size.x - 1,
                                     slice_size, param_weight),
              r1 =
                  lookup<Dimension>(m_conditional_cdf.data(), offset + (m_size.x * 2 - 1),
                                    slice_size, param_weight);

        bool is_const = std::abs(r0 - r1) < 1e-4f * (r0 + r1);
        sample.y = is_const ? (2.f * sample.y)
                            : (r0 - SafeSqrt(r0 * r0 - 2.f * sample.y * (r0 - r1)));
        sample.y /= is_const ? (r0 + r1) : (r0 - r1);

        /* Sample the column next */
        sample.x *= (1.f - sample.y) * r0 + sample.y * r1;

        auto fetch_conditional = [&](uint32_t idx) -> float {
            float v0 = lookup<Dimension>(m_conditional_cdf.data(), offset + idx,
                                         slice_size, param_weight),
                  v1 = lookup<Dimension>(m_conditional_cdf.data() + m_size.x,
                                         offset + idx, slice_size, param_weight);

            return (1.f - sample.y) * v0 + sample.y * v1;
        };

        uint32_t col = FindInterval(
            m_size.x, [&](uint32_t idx) { return fetch_conditional(idx) < sample.x; });

        sample.x -= fetch_conditional(col);

        offset += col;

        float v00 = lookup<Dimension>(m_data.data(), offset, slice_size, param_weight),
              v10 =
                  lookup<Dimension>(m_data.data() + 1, offset, slice_size, param_weight),
              v01 = lookup<Dimension>(m_data.data() + m_size.x, offset, slice_size,
                                      param_weight),
              v11 = lookup<Dimension>(m_data.data() + m_size.x + 1, offset, slice_size,
                                      param_weight),
              c0 = std::fma((1.f - sample.y), v00, sample.y * v01),
              c1 = std::fma((1.f - sample.y), v10, sample.y * v11);

        is_const = std::abs(c0 - c1) < 1e-4f * (c0 + c1);
        sample.x = is_const ? (2.f * sample.x)
                            : (c0 - SafeSqrt(c0 * c0 - 2.f * sample.x * (c0 - c1)));
        sample.x /= is_const ? (c0 + c1) : (c0 - c1);

        return {Vector2f((col + sample.x) * m_patch_size.x,
                         (row + sample.y) * m_patch_size.y),
                ((1.f - sample.x) * c0 + sample.x * c1) * HProd(m_inv_patch_size)};
    }

    /// Inverse of the mapping implemented in \c Sample()
    PBRT_CPU_GPU
    PLSample Invert(Vector2f sample, const Float *param = nullptr) const {
        /* Look up parameter-related indices and weights (if Dimension != 0) */
        float param_weight[2 * ArraySize];
        uint32_t slice_offset = 0u;
        for (size_t dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index = FindInterval(m_param_size[dim], [&](uint32_t idx) {
                return m_param_values[dim][idx] <= param[dim];
            });

            float p0 = m_param_values[dim][param_index],
                  p1 = m_param_values[dim][param_index + 1];

            param_weight[2 * dim + 1] = Clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Fetch values at corners of bilinear patch */
        sample.x *= m_inv_patch_size.x;
        sample.y *= m_inv_patch_size.y;
        Vector2i pos = Min(Vector2i(sample), m_size - Vector2i(2, 2));
        sample -= Vector2f(pos);

        uint32_t offset = pos.x + pos.y * m_size.x;
        uint32_t slice_size = HProd(m_size);
        if (Dimension != 0)
            offset += slice_offset * slice_size;

        /* Invert the X component */
        float v00 = lookup<Dimension>(m_data.data(), offset, slice_size, param_weight),
              v10 =
                  lookup<Dimension>(m_data.data() + 1, offset, slice_size, param_weight),
              v01 = lookup<Dimension>(m_data.data() + m_size.x, offset, slice_size,
                                      param_weight),
              v11 = lookup<Dimension>(m_data.data() + m_size.x + 1, offset, slice_size,
                                      param_weight);

        Vector2f w1 = sample, w0 = Vector2f(1, 1) - w1;

        float c0 = std::fma(w0.y, v00, w1.y * v01), c1 = std::fma(w0.y, v10, w1.y * v11),
              pdf = std::fma(w0.x, c0, w1.x * c1);

        sample.x *= c0 + .5f * sample.x * (c1 - c0);

        float v0 = lookup<Dimension>(m_conditional_cdf.data(), offset, slice_size,
                                     param_weight),
              v1 = lookup<Dimension>(m_conditional_cdf.data() + m_size.x, offset,
                                     slice_size, param_weight);

        sample.x += (1.f - sample.y) * v0 + sample.y * v1;

        offset = pos.y * m_size.x;
        if (Dimension != 0)
            offset += slice_offset * slice_size;

        float r0 = lookup<Dimension>(m_conditional_cdf.data(), offset + m_size.x - 1,
                                     slice_size, param_weight),
              r1 =
                  lookup<Dimension>(m_conditional_cdf.data(), offset + (m_size.x * 2 - 1),
                                    slice_size, param_weight);

        sample.x /= (1.f - sample.y) * r0 + sample.y * r1;

        /* Invert the Y component */
        sample.y *= r0 + .5f * sample.y * (r1 - r0);

        offset = pos.y;
        if (Dimension != 0)
            offset += slice_offset * m_size.y;

        sample.y +=
            lookup<Dimension>(m_marginal_cdf.data(), offset, m_size.y, param_weight);

        return {sample, pdf * HProd(m_inv_patch_size)};
    }

    /**
     * \brief Evaluate the density at position \c pos. The distribution is
     * parameterized by \c param if applicable.
     */
    PBRT_CPU_GPU
    float Evaluate(Vector2f pos, const Float *param = nullptr) const {
        /* Look up parameter-related indices and weights (if Dimension != 0) */
        float param_weight[2 * ArraySize];
        uint32_t slice_offset = 0u;

        for (size_t dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index = FindInterval(m_param_size[dim], [&](uint32_t idx) {
                return m_param_values[dim][idx] <= param[dim];
            });

            float p0 = m_param_values[dim][param_index],
                  p1 = m_param_values[dim][param_index + 1];

            param_weight[2 * dim + 1] = Clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Compute linear interpolation weights */
        pos.x *= m_inv_patch_size.x;
        pos.y *= m_inv_patch_size.y;
        Vector2i offset = Min(Vector2i(pos), m_size - Vector2i(2, 2));

        Vector2f w1 = pos - Vector2f(Vector2i(offset)), w0 = Vector2f(1, 1) - w1;

        uint32_t index = offset.x + offset.y * m_size.x;

        uint32_t size = HProd(m_size);
        if (Dimension != 0)
            index += slice_offset * size;

        float v00 = lookup<Dimension>(m_data.data(), index, size, param_weight),
              v10 = lookup<Dimension>(m_data.data() + 1, index, size, param_weight),
              v01 =
                  lookup<Dimension>(m_data.data() + m_size.x, index, size, param_weight),
              v11 = lookup<Dimension>(m_data.data() + m_size.x + 1, index, size,
                                      param_weight);

        return std::fma(w0.y, std::fma(w0.x, v00, w1.x * v10),
                        w1.y * std::fma(w0.x, v01, w1.x * v11)) *
               HProd(m_inv_patch_size);
    }

    PBRT_CPU_GPU
    size_t BytesUsed() const {
        size_t sum = 4 * (m_data.capacity() + m_marginal_cdf.capacity() +
                          m_conditional_cdf.capacity());
        for (int i = 0; i < ArraySize; ++i)
            sum += m_param_values[i].capacity();
        return sum;
    }

  private:
    template <size_t Dim, std::enable_if_t<Dim != 0, int> = 0>
    PBRT_CPU_GPU float lookup(const float *data, uint32_t i0, uint32_t size,
                              const float *param_weight) const {
        uint32_t i1 = i0 + m_param_strides[Dim - 1] * size;

        float w0 = param_weight[2 * Dim - 2], w1 = param_weight[2 * Dim - 1],
              v0 = lookup<Dim - 1>(data, i0, size, param_weight),
              v1 = lookup<Dim - 1>(data, i1, size, param_weight);

        return std::fma(v0, w0, v1 * w1);
    }

    template <size_t Dim, std::enable_if_t<Dim == 0, int> = 0>
    PBRT_CPU_GPU float lookup(const float *data, uint32_t index, uint32_t,
                              const float *) const {
        return data[index];
    }

    /// Resolution of the discretized density function
    Vector2i m_size;

    /// Size of a bilinear patch in the unit square
    Vector2f m_patch_size, m_inv_patch_size;

    /// Resolution of each parameter (optional)
    uint32_t m_param_size[ArraySize];

    /// Stride per parameter in units of sizeof(float)
    uint32_t m_param_strides[ArraySize];

    /// Discretization of each parameter domain
    pstd::vector<FloatStorage> m_param_values;

    /// Density values
    FloatStorage m_data;

    /// Marginal and conditional PDFs
    FloatStorage m_marginal_cdf;
    FloatStorage m_conditional_cdf;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_SAMPLING_H
