// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_SHAPES_H
#define PBRT_SHAPES_H

#include <pbrt/pbrt.h>

#include <pbrt/base/shape.h>
#include <pbrt/interaction.h>
#include <pbrt/ray.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <map>
#include <memory>
#include <vector>

namespace pbrt {

// ShapeSample Definition
struct ShapeSample {
    Interaction intr;
    Float pdf;
    std::string ToString() const;
};

// ShapeSampleContext Definition
struct ShapeSampleContext {
    // ShapeSampleContext Public Methods
    ShapeSampleContext() = default;
    PBRT_CPU_GPU
    ShapeSampleContext(const Point3fi &pi, const Normal3f &n, const Normal3f &ns,
                       Float time)
        : pi(pi), n(n), ns(ns), time(time) {}
    PBRT_CPU_GPU
    ShapeSampleContext(const SurfaceInteraction &si)
        : pi(si.pi), n(si.n), ns(si.shading.n), time(si.time) {}
    PBRT_CPU_GPU
    ShapeSampleContext(const MediumInteraction &mi) : pi(mi.pi), time(mi.time) {}

    PBRT_CPU_GPU
    Point3f p() const { return Point3f(pi); }

    PBRT_CPU_GPU
    Point3f OffsetRayOrigin(const Vector3f &w) const;
    PBRT_CPU_GPU
    Point3f OffsetRayOrigin(const Point3f &pt) const;
    PBRT_CPU_GPU
    Ray SpawnRay(const Vector3f &w) const;

    Point3fi pi;
    Normal3f n, ns;
    Float time;
};

// ShapeSampleContext Inline Methods
PBRT_CPU_GPU inline Point3f ShapeSampleContext::OffsetRayOrigin(const Vector3f &w) const {
    // Copied from Interaction... :-p
    Float d = Dot(Abs(n), pi.Error());
    Vector3f offset = d * Vector3f(n);
    if (Dot(w, n) < 0)
        offset = -offset;
    Point3f po = Point3f(pi) + offset;
    // Round offset point _po_ away from _p_
    for (int i = 0; i < 3; ++i) {
        if (offset[i] > 0)
            po[i] = NextFloatUp(po[i]);
        else if (offset[i] < 0)
            po[i] = NextFloatDown(po[i]);
    }

    return po;
}

PBRT_CPU_GPU inline Point3f ShapeSampleContext::OffsetRayOrigin(const Point3f &pt) const {
    return OffsetRayOrigin(pt - p());
}

PBRT_CPU_GPU inline Ray ShapeSampleContext::SpawnRay(const Vector3f &w) const {
    // Note: doesn't set medium, but that's fine, since this is only
    // used by shapes to see if ray would have intersected them
    return Ray(OffsetRayOrigin(w), w, time);
}

// ShapeIntersection Definition
struct ShapeIntersection {
    SurfaceInteraction intr;
    Float tHit;
    std::string ToString() const;
};

// QuadricIntersection Definition
struct QuadricIntersection {
    Float tHit;
    Point3f pObj;
    Float phi;
};

// Sphere Definition
class Sphere {
  public:
    // Sphere Public Methods
    static Sphere *Create(const Transform *renderFromObject,
                          const Transform *objectFromRender, bool reverseOrientation,
                          const ParameterDictionary &parameters, const FileLoc *loc,
                          Allocator alloc);

    std::string ToString() const;

    Sphere(const Transform *renderFromObject, const Transform *objectFromRender,
           bool reverseOrientation, Float radius, Float zMin, Float zMax, Float phiMax)
        : renderFromObject(renderFromObject),
          objectFromRender(objectFromRender),
          reverseOrientation(reverseOrientation),
          transformSwapsHandedness(renderFromObject->SwapsHandedness()),
          radius(radius),
          zMin(Clamp(std::min(zMin, zMax), -radius, radius)),
          zMax(Clamp(std::max(zMin, zMax), -radius, radius)),
          thetaZMin(std::acos(Clamp(std::min(zMin, zMax) / radius, -1, 1))),
          thetaZMax(std::acos(Clamp(std::max(zMin, zMax) / radius, -1, 1))),
          phiMax(Radians(Clamp(phiMax, 0, 360))) {}

    PBRT_CPU_GPU
    Bounds3f Bounds() const;

    PBRT_CPU_GPU
    DirectionCone NormalBounds() const { return DirectionCone::EntireSphere(); }

    PBRT_CPU_GPU
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const {
        pstd::optional<QuadricIntersection> isect = BasicIntersect(ray, tMax);
        if (!isect)
            return {};
        SurfaceInteraction intr = InteractionFromIntersection(*isect, -ray.d, ray.time);
        return ShapeIntersection{intr, isect->tHit};
    }

    PBRT_CPU_GPU
    pstd::optional<QuadricIntersection> BasicIntersect(const Ray &r, Float tMax) const {
        Float phi;
        Point3f pHit;
        // Transform _Ray_ origin and direction to object space
        Point3fi oi = (*objectFromRender)(Point3fi(r.o));
        Vector3fi di = (*objectFromRender)(Vector3fi(r.d));

        // Solve quadratic equation to compute sphere _t0_ and _t1_
        FloatInterval t0, t1;
        // Compute sphere quadratic coefficients
        FloatInterval a = SumSquares(di.x, di.y, di.z);
        FloatInterval b = 2 * (di.x * oi.x + di.y * oi.y + di.z * oi.z);
        FloatInterval c = SumSquares(oi.x, oi.y, oi.z) - Sqr(FloatInterval(radius));

// Compute sphere quadratic discriminant _discrim_
#if 0
// Original
FloatInterval b2 = Sqr(b), ac = 4 * a * c;
FloatInterval odiscrim = b2 - ac; // b * b - FloatInterval(4) * a * c;
#endif
        FloatInterval f = b / (2 * a);  // (o . d) / LengthSquared(d)
        Point3fi fp = oi - f * di;
        // There's a bit more precision if you compute x^2-y^2 as (x+y)(x-y).
        FloatInterval sqrtf = Sqrt(SumSquares(fp.x, fp.y, fp.z));
        FloatInterval discrim =
            4 * a * ((FloatInterval(radius)) - sqrtf) * ((FloatInterval(radius)) + sqrtf);
        if (discrim.LowerBound() < 0)
            return {};

        // Compute quadratic $t$ values
        FloatInterval rootDiscrim = Sqrt(discrim);
        FloatInterval q;
        if ((Float)b < 0)
            q = -.5f * (b - rootDiscrim);
        else
            q = -.5f * (b + rootDiscrim);
        t0 = q / a;
        t1 = c / q;
        // Swap quadratic $t$ values so that _t0_ is the lesser
        if (t0.LowerBound() > t1.LowerBound())
            pstd::swap(t0, t1);

        // Check quadric shape _t0_ and _t1_ for nearest intersection
        if (t0.UpperBound() > tMax || t1.LowerBound() <= 0)
            return {};
        FloatInterval tShapeHit = t0;
        if (tShapeHit.LowerBound() <= 0) {
            tShapeHit = t1;
            if (tShapeHit.UpperBound() > tMax)
                return {};
        }

        // Compute sphere hit position and $\phi$
        pHit = Point3f(oi) + (Float)tShapeHit * Vector3f(di);
        // Refine sphere intersection point
        pHit *= radius / Distance(pHit, Point3f(0, 0, 0));

        if (pHit.x == 0 && pHit.y == 0)
            pHit.x = 1e-5f * radius;
        phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0)
            phi += 2 * Pi;

        // Test sphere intersection against clipping parameters
        if ((zMin > -radius && pHit.z < zMin) || (zMax < radius && pHit.z > zMax) ||
            phi > phiMax) {
            if (tShapeHit == t1)
                return {};
            if (t1.UpperBound() > tMax)
                return {};
            tShapeHit = t1;
            // Compute sphere hit position and $\phi$
            pHit = Point3f(oi) + (Float)tShapeHit * Vector3f(di);
            // Refine sphere intersection point
            pHit *= radius / Distance(pHit, Point3f(0, 0, 0));

            if (pHit.x == 0 && pHit.y == 0)
                pHit.x = 1e-5f * radius;
            phi = std::atan2(pHit.y, pHit.x);
            if (phi < 0)
                phi += 2 * Pi;

            if ((zMin > -radius && pHit.z < zMin) || (zMax < radius && pHit.z > zMax) ||
                phi > phiMax)
                return {};
        }

        // Return _QuadricIntersection_ for sphere intersection
        return QuadricIntersection{Float(tShapeHit), pHit, phi};
    }

    PBRT_CPU_GPU
    bool IntersectP(const Ray &r, Float tMax = Infinity) const {
        return BasicIntersect(r, tMax).has_value();
    }

    PBRT_CPU_GPU
    SurfaceInteraction InteractionFromIntersection(const QuadricIntersection &isect,
                                                   const Vector3f &wo, Float time) const {
        Point3f pHit = isect.pObj;
        Float phi = isect.phi;
        // Find parametric representation of sphere hit
        Float u = phi / phiMax;
        Float cosTheta = pHit.z / radius;
        Float theta = SafeACos(cosTheta);
        Float v = (theta - thetaZMin) / (thetaZMax - thetaZMin);
        // Compute sphere $\dpdu$ and $\dpdv$
        Float zRadius = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
        Float cosPhi = pHit.x / zRadius;
        Float sinPhi = pHit.y / zRadius;
        Vector3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Vector3f dpdv = (thetaZMax - thetaZMin) *
                        Vector3f(pHit.z * cosPhi, pHit.z * sinPhi, -radius * sinTheta);

        // Compute sphere $\dndu$ and $\dndv$
        Vector3f d2Pduu = -phiMax * phiMax * Vector3f(pHit.x, pHit.y, 0);
        Vector3f d2Pduv =
            (thetaZMax - thetaZMin) * pHit.z * phiMax * Vector3f(-sinPhi, cosPhi, 0.);
        Vector3f d2Pdvv = -(thetaZMax - thetaZMin) * (thetaZMax - thetaZMin) *
                          Vector3f(pHit.x, pHit.y, pHit.z);
        // Compute coefficients for fundamental forms
        Float E = Dot(dpdu, dpdu);
        Float F = Dot(dpdu, dpdv);
        Float G = Dot(dpdv, dpdv);
        Vector3f N = Normalize(Cross(dpdu, dpdv));
        Float e = Dot(N, d2Pduu);
        Float f = Dot(N, d2Pduv);
        Float g = Dot(N, d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        Float invEGF2 = 1 / (E * G - F * F);
        Normal3f dndu =
            Normal3f((f * F - e * G) * invEGF2 * dpdu + (e * F - f * E) * invEGF2 * dpdv);
        Normal3f dndv =
            Normal3f((g * F - f * G) * invEGF2 * dpdu + (f * F - g * E) * invEGF2 * dpdv);

        // Compute error bounds for sphere intersection
        Vector3f pError = gamma(5) * Abs((Vector3f)pHit);

        // Return _SurfaceInteraction_ for quadric intersection
        bool flipNormal = reverseOrientation ^ transformSwapsHandedness;
        Vector3f woObject = (*objectFromRender)(wo);
        return (*renderFromObject)(SurfaceInteraction(Point3fi(pHit, pError),
                                                      Point2f(u, v), woObject, dpdu, dpdv,
                                                      dndu, dndv, time, flipNormal));
    }

    PBRT_CPU_GPU
    Float Area() const { return phiMax * radius * (zMax - zMin); }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const Point2f &u) const;

    PBRT_CPU_GPU
    Float PDF(const Interaction &) const { return 1 / Area(); }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx,
                                       const Point2f &u) const {
        // Sample uniformly on sphere if $\pt{}$ is inside it
        Point3f pCenter = (*renderFromObject)(Point3f(0, 0, 0));
        Point3f pOrigin = ctx.OffsetRayOrigin(pCenter);
        if (DistanceSquared(pOrigin, pCenter) <= radius * radius) {
            // Uniformly sample shape and compute incident direction _wi_
            pstd::optional<ShapeSample> ss = Sample(u);
            DCHECK(ss.has_value());
            ss->intr.time = ctx.time;
            Vector3f wi = ss->intr.p() - ctx.p();
            if (LengthSquared(wi) == 0)
                return {};
            wi = Normalize(wi);

            // Convert uniform area sample PDF in _ss_ to solid angle measure
            ss->pdf /= AbsDot(ss->intr.n, -wi) / DistanceSquared(ctx.p(), ss->intr.p());
            if (IsInf(ss->pdf))
                return {};

            return ss;
        }

        // Sample sphere uniformly inside subtended cone
        // Compute quantities related the $\theta_\roman{max}$ for cone
        Float sinThetaMax = radius / Distance(ctx.p(), pCenter);
        Float sin2ThetaMax = sinThetaMax * sinThetaMax;
        Float cosThetaMax = SafeSqrt(1 - sin2ThetaMax);
        Float oneMinusCosThetaMax = 1 - cosThetaMax;

        // Compute $\theta$ and $\phi$ values for sample in cone
        Float cosTheta = (cosThetaMax - 1) * u[0] + 1;
        Float sin2Theta = 1 - cosTheta * cosTheta;
        if (sin2ThetaMax < 0.00068523f /* sin^2(1.5 deg) */) {
            // Compute cone sample via Taylor series expansion for small angles
            sin2Theta = sin2ThetaMax * u[0];
            cosTheta = std::sqrt(1 - sin2Theta);
            oneMinusCosThetaMax = sin2ThetaMax / 2;
        }

        // Compute angle $\alpha$ from center of sphere to sampled point on surface
        Float cosAlpha = sin2Theta / sinThetaMax +
                         cosTheta * SafeSqrt(1 - sin2Theta / Sqr(sinThetaMax));
        Float sinAlpha = SafeSqrt(1 - Sqr(cosAlpha));

        // Compute surface normal and sampled point on sphere
        Float phi = u[1] * 2 * Pi;
        Vector3f w = SphericalDirection(sinAlpha, cosAlpha, phi);
        Frame samplingFrame = Frame::FromZ(Normalize(pCenter - ctx.p()));
        Normal3f n(samplingFrame.FromLocal(-w));
        Point3f p = pCenter + radius * Point3f(n.x, n.y, n.z);
        if (reverseOrientation)
            n *= -1;

        // Return _ShapeSample_ for sampled point on sphere
        // Compute _pError_ for sampled point on sphere
        Vector3f pError = gamma(5) * Abs((Vector3f)p);

        DCHECK_NE(oneMinusCosThetaMax, 0);  // very small far away sphere
        return ShapeSample{Interaction(Point3fi(p, pError), n, ctx.time),
                           1 / (2 * Pi * oneMinusCosThetaMax)};
    }

    PBRT_CPU_GPU
    Float PDF(const ShapeSampleContext &ctx, const Vector3f &wi) const {
        Point3f pCenter = (*renderFromObject)(Point3f(0, 0, 0));
        Point3f pOrigin = ctx.OffsetRayOrigin(pCenter);
        if (DistanceSquared(pOrigin, pCenter) <= radius * radius) {
            // Return solid angle PDF for point inside sphere
            // Intersect sample ray with shape geometry
            Ray ray = ctx.SpawnRay(wi);
            pstd::optional<ShapeIntersection> isect = Intersect(ray);
            CHECK_RARE(1e-6, !isect.has_value());
            if (!isect)
                return 0;

            // Compute PDF in solid angle measure from shape intersection point
            Float pdf = (1 / Area()) / (AbsDot(isect->intr.n, -wi) /
                                        DistanceSquared(ctx.p(), isect->intr.p()));
            if (IsInf(pdf))
                pdf = 0;

            return pdf;
        }
        // Compute general solid angle sphere PDF
        Float sin2ThetaMax = radius * radius / DistanceSquared(ctx.p(), pCenter);
        Float cosThetaMax = SafeSqrt(1 - sin2ThetaMax);
        Float oneMinusCosThetaMax = 1 - cosThetaMax;
        // Compute more accurate _oneMinusCosThetaMax_ for small solid angle
        if (sin2ThetaMax < 0.00068523f /* sin^2(1.5 deg) */)
            oneMinusCosThetaMax = sin2ThetaMax / 2;

        return 1 / (2 * Pi * oneMinusCosThetaMax);
    }

  private:
    // Sphere Private Members
    Float radius;
    Float zMin, zMax;
    Float thetaZMin, thetaZMax, phiMax;
    const Transform *renderFromObject, *objectFromRender;
    bool reverseOrientation, transformSwapsHandedness;
};

// Disk Definition
class Disk {
  public:
    // Disk Public Methods
    Disk(const Transform *renderFromObject, const Transform *objectFromRender,
         bool reverseOrientation, Float height, Float radius, Float innerRadius,
         Float phiMax)
        : renderFromObject(renderFromObject),
          objectFromRender(objectFromRender),
          reverseOrientation(reverseOrientation),
          transformSwapsHandedness(renderFromObject->SwapsHandedness()),
          height(height),
          radius(radius),
          innerRadius(innerRadius),
          phiMax(Radians(Clamp(phiMax, 0, 360))) {}

    static Disk *Create(const Transform *renderFromObject,
                        const Transform *objectFromRender, bool reverseOrientation,
                        const ParameterDictionary &parameters, const FileLoc *loc,
                        Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    Float Area() const { return phiMax * 0.5f * (Sqr(radius) - Sqr(innerRadius)); }

    PBRT_CPU_GPU
    Bounds3f Bounds() const;

    PBRT_CPU_GPU
    DirectionCone NormalBounds() const;

    PBRT_CPU_GPU
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const {
        pstd::optional<QuadricIntersection> isect = BasicIntersect(ray, tMax);
        if (!isect)
            return {};
        SurfaceInteraction intr = InteractionFromIntersection(*isect, -ray.d, ray.time);
        return ShapeIntersection{intr, isect->tHit};
    }

    PBRT_CPU_GPU
    pstd::optional<QuadricIntersection> BasicIntersect(const Ray &r, Float tMax) const {
        // Transform _Ray_ origin and direction to object space
        Point3fi oi = (*objectFromRender)(Point3fi(r.o));
        Vector3fi di = (*objectFromRender)(Vector3fi(r.d));

        // Compute plane intersection for disk
        // Reject disk intersections for rays parallel to the disk's plane
        if (Float(di.z) == 0)
            return {};

        Float tShapeHit = (height - Float(oi.z)) / Float(di.z);
        if (tShapeHit <= 0 || tShapeHit >= tMax)
            return {};

        // See if hit point is inside disk radii and $\phimax$
        Point3f pHit = Point3f(oi) + (Float)tShapeHit * Vector3f(di);
        Float dist2 = pHit.x * pHit.x + pHit.y * pHit.y;
        if (dist2 > radius * radius || dist2 < innerRadius * innerRadius)
            return {};
        // Test disk $\phi$ value against $\phimax$
        Float phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0)
            phi += 2 * Pi;
        if (phi > phiMax)
            return {};

        // Return _QuadricIntersection_ for disk intersection
        return QuadricIntersection{tShapeHit, pHit, phi};
    }

    PBRT_CPU_GPU
    SurfaceInteraction InteractionFromIntersection(const QuadricIntersection &isect,
                                                   const Vector3f &wo, Float time) const {
        Point3f pHit = isect.pObj;
        Float phi = isect.phi;
        // Find parametric representation of disk hit
        Float u = phi / phiMax;
        Float rHit = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
        Float v = (radius - rHit) / (radius - innerRadius);
        Vector3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Vector3f dpdv = Vector3f(pHit.x, pHit.y, 0.) * (innerRadius - radius) / rHit;
        Normal3f dndu(0, 0, 0), dndv(0, 0, 0);

        // Refine disk intersection point
        pHit.z = height;

        // Compute error bounds for disk intersection
        Vector3f pError(0, 0, 0);

        // Return _SurfaceInteraction_ for quadric intersection
        bool flipNormal = reverseOrientation ^ transformSwapsHandedness;
        Vector3f woObject = (*objectFromRender)(wo);
        return (*renderFromObject)(SurfaceInteraction(Point3fi(pHit, pError),
                                                      Point2f(u, v), woObject, dpdu, dpdv,
                                                      dndu, dndv, time, flipNormal));
    }

    PBRT_CPU_GPU
    bool IntersectP(const Ray &r, Float tMax = Infinity) const {
        return BasicIntersect(r, tMax).has_value();
    }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const Point2f &u) const {
        Point2f pd = SampleUniformDiskConcentric(u);
        Point3f pObj(pd.x * radius, pd.y * radius, height);
        Point3fi pi = (*renderFromObject)(Point3fi(pObj));
        Normal3f n = Normalize((*renderFromObject)(Normal3f(0, 0, 1)));
        if (reverseOrientation)
            n *= -1;
        return ShapeSample{Interaction(pi, n), 1 / Area()};
    }

    PBRT_CPU_GPU
    Float PDF(const Interaction &) const { return 1 / Area(); }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx,
                                       const Point2f &u) const {
        // Uniformly sample shape and compute incident direction _wi_
        pstd::optional<ShapeSample> ss = Sample(u);
        DCHECK(ss.has_value());
        ss->intr.time = ctx.time;
        Vector3f wi = ss->intr.p() - ctx.p();
        if (LengthSquared(wi) == 0)
            return {};
        wi = Normalize(wi);

        // Convert uniform area sample PDF in _ss_ to solid angle measure
        ss->pdf /= AbsDot(ss->intr.n, -wi) / DistanceSquared(ctx.p(), ss->intr.p());
        if (IsInf(ss->pdf))
            return {};

        return ss;
    }

    PBRT_CPU_GPU
    Float PDF(const ShapeSampleContext &ctx, const Vector3f &wi) const {
        // Intersect sample ray with shape geometry
        Ray ray = ctx.SpawnRay(wi);
        pstd::optional<ShapeIntersection> isect = Intersect(ray);
        CHECK_RARE(1e-6, !isect.has_value());
        if (!isect)
            return 0;

        // Compute PDF in solid angle measure from shape intersection point
        Float pdf = (1 / Area()) / (AbsDot(isect->intr.n, -wi) /
                                    DistanceSquared(ctx.p(), isect->intr.p()));
        if (IsInf(pdf))
            pdf = 0;

        return pdf;
    }

  private:
    // Disk Private Members
    const Transform *renderFromObject, *objectFromRender;
    bool reverseOrientation, transformSwapsHandedness;
    Float height, radius, innerRadius, phiMax;
};

// Cylinder Definition
class Cylinder {
  public:
    // Cylinder Public Methods
    Cylinder(const Transform *renderFromObject, const Transform *objectFromRender,
             bool reverseOrientation, Float radius, Float zMin, Float zMax, Float phiMax);

    static Cylinder *Create(const Transform *renderFromObject,
                            const Transform *objectFromRender, bool reverseOrientation,
                            const ParameterDictionary &parameters, const FileLoc *loc,
                            Allocator alloc);

    PBRT_CPU_GPU
    Bounds3f Bounds() const;

    std::string ToString() const;

    PBRT_CPU_GPU
    Float Area() const { return (zMax - zMin) * radius * phiMax; }

    PBRT_CPU_GPU
    DirectionCone NormalBounds() const { return DirectionCone::EntireSphere(); }

    PBRT_CPU_GPU
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const {
        pstd::optional<QuadricIntersection> isect = BasicIntersect(ray, tMax);
        if (!isect)
            return {};
        SurfaceInteraction intr = InteractionFromIntersection(*isect, -ray.d, ray.time);
        return ShapeIntersection{intr, isect->tHit};
    }

    PBRT_CPU_GPU
    pstd::optional<QuadricIntersection> BasicIntersect(const Ray &r, Float tMax) const {
        Float phi;
        Point3f pHit;
        // Transform _Ray_ origin and direction to object space
        Point3fi oi = (*objectFromRender)(Point3fi(r.o));
        Vector3fi di = (*objectFromRender)(Vector3fi(r.d));

        // Solve quadratic equation to find cylinder _t0_ and _t1_ values
        FloatInterval t0, t1;
        // Compute cylinder quadratic coefficients
        FloatInterval a = SumSquares(di.x, di.y);
        FloatInterval b = 2 * (di.x * oi.x + di.y * oi.y);
        FloatInterval c = SumSquares(oi.x, oi.y) - Sqr(FloatInterval(radius));

        // Compute cylinder quadratic discriminant _discrim_
        // FloatInterval discrim = B * B - FloatInterval(4) * A * C;
        FloatInterval f = b / (2 * a);  // (o . d) / LengthSquared(d)
        FloatInterval fx = oi.x - f * di.x;
        FloatInterval fy = oi.y - f * di.y;
        FloatInterval sqrtf = Sqrt(SumSquares(fx, fy));
        FloatInterval discrim =
            4 * a * (FloatInterval(radius) + sqrtf) * (FloatInterval(radius) - sqrtf);
        if (discrim.LowerBound() < 0)
            return {};

        // Compute quadratic $t$ values
        FloatInterval rootDiscrim = Sqrt(discrim);
        FloatInterval q;
        if ((Float)b < 0)
            q = -.5f * (b - rootDiscrim);
        else
            q = -.5f * (b + rootDiscrim);
        t0 = q / a;
        t1 = c / q;
        // Swap quadratic $t$ values so that _t0_ is the lesser
        if (t0.LowerBound() > t1.LowerBound())
            pstd::swap(t0, t1);

        // Check quadric shape _t0_ and _t1_ for nearest intersection
        if (t0.UpperBound() > tMax || t1.LowerBound() <= 0)
            return {};
        FloatInterval tShapeHit = t0;
        if (tShapeHit.LowerBound() <= 0) {
            tShapeHit = t1;
            if (tShapeHit.UpperBound() > tMax)
                return {};
        }

        // Compute cylinder hit point and $\phi$
        pHit = Point3f(oi) + (Float)tShapeHit * Vector3f(di);
        // Refine cylinder intersection point
        Float hitRad = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
        pHit.x *= radius / hitRad;
        pHit.y *= radius / hitRad;

        phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0)
            phi += 2 * Pi;

        // Test cylinder intersection against clipping parameters
        if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
            if (tShapeHit == t1)
                return {};
            tShapeHit = t1;
            if (t1.UpperBound() > tMax)
                return {};
            // Compute cylinder hit point and $\phi$
            pHit = Point3f(oi) + (Float)tShapeHit * Vector3f(di);
            // Refine cylinder intersection point
            Float hitRad = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
            pHit.x *= radius / hitRad;
            pHit.y *= radius / hitRad;

            phi = std::atan2(pHit.y, pHit.x);
            if (phi < 0)
                phi += 2 * Pi;

            if (pHit.z < zMin || pHit.z > zMax || phi > phiMax)
                return {};
        }

        // Return _QuadricIntersection_ for cylinder intersection
        return QuadricIntersection{(Float)tShapeHit, pHit, phi};
    }

    PBRT_CPU_GPU
    bool IntersectP(const Ray &r, Float tMax = Infinity) const {
        return BasicIntersect(r, tMax).has_value();
    }

    PBRT_CPU_GPU
    SurfaceInteraction InteractionFromIntersection(const QuadricIntersection &isect,
                                                   const Vector3f &wo, Float time) const {
        Point3f pHit = isect.pObj;
        Float phi = isect.phi;
        // Find parametric representation of cylinder hit
        Float u = phi / phiMax;
        Float v = (pHit.z - zMin) / (zMax - zMin);
        // Compute cylinder $\dpdu$ and $\dpdv$
        Vector3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Vector3f dpdv(0, 0, zMax - zMin);

        // Compute cylinder $\dndu$ and $\dndv$
        Vector3f d2Pduu = -phiMax * phiMax * Vector3f(pHit.x, pHit.y, 0);
        Vector3f d2Pduv(0, 0, 0), d2Pdvv(0, 0, 0);
        // Compute coefficients for fundamental forms
        Float E = Dot(dpdu, dpdu);
        Float F = Dot(dpdu, dpdv);
        Float G = Dot(dpdv, dpdv);
        Vector3f N = Normalize(Cross(dpdu, dpdv));
        Float e = Dot(N, d2Pduu);
        Float f = Dot(N, d2Pduv);
        Float g = Dot(N, d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        Float invEGF2 = 1 / (E * G - F * F);
        Normal3f dndu =
            Normal3f((f * F - e * G) * invEGF2 * dpdu + (e * F - f * E) * invEGF2 * dpdv);
        Normal3f dndv =
            Normal3f((g * F - f * G) * invEGF2 * dpdu + (f * F - g * E) * invEGF2 * dpdv);

        // Compute error bounds for cylinder intersection
        Vector3f pError = gamma(3) * Abs(Vector3f(pHit.x, pHit.y, 0));

        // Return _SurfaceInteraction_ for quadric intersection
        bool flipNormal = reverseOrientation ^ transformSwapsHandedness;
        Vector3f woObject = (*objectFromRender)(wo);
        return (*renderFromObject)(SurfaceInteraction(Point3fi(pHit, pError),
                                                      Point2f(u, v), woObject, dpdu, dpdv,
                                                      dndu, dndv, time, flipNormal));
    }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const Point2f &u) const {
        Float z = Lerp(u[0], zMin, zMax);
        Float phi = u[1] * phiMax;
        // Compute cylinder sample position _pi_ and normal _n_ from $z$ and $\phi$
        Point3f pObj = Point3f(radius * std::cos(phi), radius * std::sin(phi), z);
        // Reproject _pObj_ to cylinder surface and compute _pObjError_
        Float hitRad = std::sqrt(pObj.x * pObj.x + pObj.y * pObj.y);
        pObj.x *= radius / hitRad;
        pObj.y *= radius / hitRad;
        Vector3f pObjError = gamma(3) * Abs(Vector3f(pObj.x, pObj.y, 0));

        Point3fi pi = (*renderFromObject)(Point3fi(pObj, pObjError));
        Normal3f n = Normalize((*renderFromObject)(Normal3f(pObj.x, pObj.y, 0)));
        if (reverseOrientation)
            n *= -1;

        return ShapeSample{Interaction(pi, n), 1 / Area()};
    }

    PBRT_CPU_GPU
    Float PDF(const Interaction &) const { return 1 / Area(); }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx,
                                       const Point2f &u) const {
        // Uniformly sample shape and compute incident direction _wi_
        pstd::optional<ShapeSample> ss = Sample(u);
        DCHECK(ss.has_value());
        ss->intr.time = ctx.time;
        Vector3f wi = ss->intr.p() - ctx.p();
        if (LengthSquared(wi) == 0)
            return {};
        wi = Normalize(wi);

        // Convert uniform area sample PDF in _ss_ to solid angle measure
        ss->pdf /= AbsDot(ss->intr.n, -wi) / DistanceSquared(ctx.p(), ss->intr.p());
        if (IsInf(ss->pdf))
            return {};

        return ss;
    }

    PBRT_CPU_GPU
    Float PDF(const ShapeSampleContext &ctx, const Vector3f &wi) const {
        // Intersect sample ray with shape geometry
        Ray ray = ctx.SpawnRay(wi);
        pstd::optional<ShapeIntersection> isect = Intersect(ray);
        CHECK_RARE(1e-6, !isect.has_value());
        if (!isect)
            return 0;

        // Compute PDF in solid angle measure from shape intersection point
        Float pdf = (1 / Area()) / (AbsDot(isect->intr.n, -wi) /
                                    DistanceSquared(ctx.p(), isect->intr.p()));
        if (IsInf(pdf))
            pdf = 0;

        return pdf;
    }

  private:
    // Cylinder Private Members
    const Transform *renderFromObject, *objectFromRender;
    bool reverseOrientation, transformSwapsHandedness;
    Float radius, zMin, zMax, phiMax;
};

// Cylinder Inline Methods
inline Cylinder::Cylinder(const Transform *renderFromObject,
                          const Transform *objectFromRender, bool reverseOrientation,
                          Float radius, Float zMin, Float zMax, Float phiMax)
    : renderFromObject(renderFromObject),
      objectFromRender(objectFromRender),
      reverseOrientation(reverseOrientation),
      transformSwapsHandedness(renderFromObject->SwapsHandedness()),
      radius(radius),
      zMin(std::min(zMin, zMax)),
      zMax(std::max(zMin, zMax)),
      phiMax(Radians(Clamp(phiMax, 0, 360))) {}

// Triangle Declarations
#if defined(PBRT_BUILD_GPU_RENDERER) && defined(__CUDACC__)
extern PBRT_GPU pstd::vector<const TriangleMesh *> *allTriangleMeshesGPU;
#endif

// TriangleIntersection Definition
struct TriangleIntersection {
    Float b0, b1, b2;
    Float t;
    std::string ToString() const;
};

// Triangle Function Declarations
PBRT_CPU_GPU
pstd::optional<TriangleIntersection> IntersectTriangle(const Ray &ray, Float tMax,
                                                       const Point3f &p0,
                                                       const Point3f &p1,
                                                       const Point3f &p2);

// Triangle Definition
class Triangle {
  public:
    // Triangle Public Methods
    static pstd::vector<ShapeHandle> CreateTriangles(const TriangleMesh *mesh,
                                                     Allocator alloc);

    Triangle() = default;
    Triangle(int meshIndex, int triIndex) : meshIndex(meshIndex), triIndex(triIndex) {}

    static void Init(Allocator alloc);

    PBRT_CPU_GPU
    Bounds3f Bounds() const;

    PBRT_CPU_GPU
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const;
    PBRT_CPU_GPU
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    PBRT_CPU_GPU
    Float Area() const {
        // Get triangle vertices in _p0_, _p1_, and _p2_
        const TriangleMesh *mesh = GetMesh();
        const int *v = &mesh->vertexIndices[3 * triIndex];
        Point3f p0 = mesh->p[v[0]], p1 = mesh->p[v[1]], p2 = mesh->p[v[2]];

        return 0.5f * Length(Cross(p1 - p0, p2 - p0));
    }

    PBRT_CPU_GPU
    DirectionCone NormalBounds() const;

    std::string ToString() const;

    static TriangleMesh *CreateMesh(const Transform *renderFromObject,
                                    bool reverseOrientation,
                                    const ParameterDictionary &parameters,
                                    const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    Float SolidAngle(const Point3f &p) const {
        // Get triangle vertices in _p0_, _p1_, and _p2_
        const TriangleMesh *mesh = GetMesh();
        const int *v = &mesh->vertexIndices[3 * triIndex];
        Point3f p0 = mesh->p[v[0]], p1 = mesh->p[v[1]], p2 = mesh->p[v[2]];

        return SphericalTriangleArea(Normalize(p0 - p), Normalize(p1 - p),
                                     Normalize(p2 - p));
    }

    PBRT_CPU_GPU
    static SurfaceInteraction InteractionFromIntersection(const TriangleMesh *mesh,
                                                          int triIndex,
                                                          const TriangleIntersection &ti,
                                                          Float time,
                                                          const Vector3f &wo) {
        const int *v = &mesh->vertexIndices[3 * triIndex];
        Point3f p0 = mesh->p[v[0]], p1 = mesh->p[v[1]], p2 = mesh->p[v[2]];
        // Compute triangle partial derivatives
        // Compute deltas and matrix determinant for triangle partial derivatives
        // Get triangle texture coordinates in _uv_ array
        pstd::array<Point2f, 3> uv =
            mesh->uv
                ? pstd::array<Point2f, 3>(
                      {mesh->uv[v[0]], mesh->uv[v[1]], mesh->uv[v[2]]})
                : pstd::array<Point2f, 3>({Point2f(0, 0), Point2f(1, 0), Point2f(1, 1)});

        Vector2f duv02 = uv[0] - uv[2], duv12 = uv[1] - uv[2];
        Vector3f dp02 = p0 - p2, dp12 = p1 - p2;
        Float determinant = DifferenceOfProducts(duv02[0], duv12[1], duv02[1], duv12[0]);

        Vector3f dpdu, dpdv;
        bool degenerateUV = std::abs(determinant) < 1e-9f;
        if (!degenerateUV) {
            // Compute triangle $\dpdu$ and $\dpdv$ via matrix inversion
            Float invdet = 1 / determinant;
            dpdu = DifferenceOfProducts(duv12[1], dp02, duv02[1], dp12) * invdet;
            dpdv = DifferenceOfProducts(duv02[0], dp12, duv12[0], dp02) * invdet;
        }
        // Handle degenerate triangle $(u,v)$ parameterization or partial derivatives
        if (degenerateUV || LengthSquared(Cross(dpdu, dpdv)) == 0) {
            Vector3f ng = Cross(p2 - p0, p1 - p0);
            if (LengthSquared(ng) == 0) {
                ng = Vector3f(Cross(Vector3<double>(p2 - p0), Vector3<double>(p1 - p0)));
                CHECK_NE(LengthSquared(ng), 0);
            }
            CoordinateSystem(Normalize(ng), &dpdu, &dpdv);
        }

        // Interpolate $(u,v)$ parametric coordinates and hit point
        Point3f pHit = ti.b0 * p0 + ti.b1 * p1 + ti.b2 * p2;
        Point2f uvHit = ti.b0 * uv[0] + ti.b1 * uv[1] + ti.b2 * uv[2];

        // Return _SurfaceInteraction_ for triangle hit
        int faceIndex = mesh->faceIndices ? mesh->faceIndices[triIndex] : 0;
        bool flipNormal = mesh->reverseOrientation ^ mesh->transformSwapsHandedness;
        // Compute error bounds _pError_ for triangle intersection
        Float xAbsSum =
            (std::abs(ti.b0 * p0.x) + std::abs(ti.b1 * p1.x) + std::abs(ti.b2 * p2.x));
        Float yAbsSum =
            (std::abs(ti.b0 * p0.y) + std::abs(ti.b1 * p1.y) + std::abs(ti.b2 * p2.y));
        Float zAbsSum =
            (std::abs(ti.b0 * p0.z) + std::abs(ti.b1 * p1.z) + std::abs(ti.b2 * p2.z));
        Vector3f pError = gamma(7) * Vector3f(xAbsSum, yAbsSum, zAbsSum);

        SurfaceInteraction isect(Point3fi(pHit, pError), uvHit, wo, dpdu, dpdv,
                                 Normal3f(), Normal3f(), time, flipNormal, faceIndex);
        // Set final surface normal and shading geometry for triangle
        // Override surface normal in _isect_ for triangle
        isect.n = isect.shading.n = Normal3f(Normalize(Cross(dp02, dp12)));
        if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
            isect.n = isect.shading.n = -isect.n;

        if (mesh->n || mesh->s) {
            // Initialize _Triangle_ shading geometry
            // Compute shading normal _ns_ for triangle
            Normal3f ns;
            if (mesh->n != nullptr) {
                ns =
                    ti.b0 * mesh->n[v[0]] + ti.b1 * mesh->n[v[1]] + ti.b2 * mesh->n[v[2]];
                ns = LengthSquared(ns) > 0 ? Normalize(ns) : isect.n;
            } else
                ns = isect.n;

            // Compute shading tangent _ss_ for triangle
            Vector3f ss;
            if (mesh->s != nullptr) {
                ss =
                    ti.b0 * mesh->s[v[0]] + ti.b1 * mesh->s[v[1]] + ti.b2 * mesh->s[v[2]];
                if (LengthSquared(ss) == 0)
                    ss = isect.dpdu;
            } else
                ss = isect.dpdu;

            // Compute shading bitangent _ts_ for triangle and adjust _ss_
            Vector3f ts = Cross(ns, ss);
            if (LengthSquared(ts) > 0)
                ss = Cross(ts, ns);
            else
                CoordinateSystem(ns, &ss, &ts);

            // Compute $\dndu$ and $\dndv$ for triangle shading geometry
            Normal3f dndu, dndv;
            if (mesh->n != nullptr) {
                // Compute deltas for triangle partial derivatives of normal
                Vector2f duv02 = uv[0] - uv[2];
                Vector2f duv12 = uv[1] - uv[2];
                Normal3f dn1 = mesh->n[v[0]] - mesh->n[v[2]];
                Normal3f dn2 = mesh->n[v[1]] - mesh->n[v[2]];

                Float determinant =
                    DifferenceOfProducts(duv02[0], duv12[1], duv02[1], duv12[0]);
                bool degenerateUV = std::abs(determinant) < 1e-9;
                if (degenerateUV) {
                    // We can still compute dndu and dndv, with respect to the
                    // same arbitrary coordinate system we use to compute dpdu
                    // and dpdv when this happens. It's important to do this
                    // (rather than giving up) so that ray differentials for
                    // rays reflected from triangles with degenerate
                    // parameterizations are still reasonable.
                    Vector3f dn = Cross(Vector3f(mesh->n[v[2]] - mesh->n[v[0]]),
                                        Vector3f(mesh->n[v[1]] - mesh->n[v[0]]));

                    if (LengthSquared(dn) == 0)
                        dndu = dndv = Normal3f(0, 0, 0);
                    else {
                        Vector3f dnu, dnv;
                        CoordinateSystem(dn, &dnu, &dnv);
                        dndu = Normal3f(dnu);
                        dndv = Normal3f(dnv);
                    }
                } else {
                    Float invDet = 1 / determinant;
                    dndu = DifferenceOfProducts(duv12[1], dn1, duv02[1], dn2) * invDet;
                    dndv = DifferenceOfProducts(duv02[0], dn2, duv12[0], dn1) * invDet;
                }
            } else
                dndu = dndv = Normal3f(0, 0, 0);

            isect.SetShadingGeometry(ns, ss, ts, dndu, dndv, true);
        }

        return isect;
    }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const Point2f &u) const {
        // Get triangle vertices in _p0_, _p1_, and _p2_
        const TriangleMesh *mesh = GetMesh();
        const int *v = &mesh->vertexIndices[3 * triIndex];
        Point3f p0 = mesh->p[v[0]], p1 = mesh->p[v[1]], p2 = mesh->p[v[2]];

        // Sample point on triangle uniformly by area
        pstd::array<Float, 3> b = SampleUniformTriangle(u);
        Point3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;

        // Compute surface normal for sampled point on triangle
        Normal3f n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
        if (mesh->n != nullptr) {
            Normal3f ns(b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] +
                        (1 - b[0] - b[1]) * mesh->n[v[2]]);
            n = FaceForward(n, ns);
        } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
            n *= -1;

        // Compute error bounds _pError_ for sampled point on triangle
        Point3f pAbsSum = Abs(b[0] * p0) + Abs(b[1] * p1) + Abs((1 - b[0] - b[1]) * p2);
        Vector3f pError = Vector3f(gamma(6) * pAbsSum);

        return ShapeSample{Interaction(Point3fi(p, pError), n), 1 / Area()};
    }

    PBRT_CPU_GPU
    Float PDF(const Interaction &) const { return 1 / Area(); }

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx, Point2f u) const {
        // Get triangle vertices in _p0_, _p1_, and _p2_
        const TriangleMesh *mesh = GetMesh();
        const int *v = &mesh->vertexIndices[3 * triIndex];
        Point3f p0 = mesh->p[v[0]], p1 = mesh->p[v[1]], p2 = mesh->p[v[2]];

        // Use uniform area sampling for numerically unstable cases
        Float solidAngle = SolidAngle(ctx.p());
        if (solidAngle < MinSphericalSampleArea || solidAngle > MaxSphericalSampleArea) {
            // Uniformly sample shape and compute incident direction _wi_
            pstd::optional<ShapeSample> ss = Sample(u);
            DCHECK(ss.has_value());
            ss->intr.time = ctx.time;
            Vector3f wi = ss->intr.p() - ctx.p();
            if (LengthSquared(wi) == 0)
                return {};
            wi = Normalize(wi);

            // Convert uniform area sample PDF in _ss_ to solid angle measure
            ss->pdf /= AbsDot(ss->intr.n, -wi) / DistanceSquared(ctx.p(), ss->intr.p());
            if (IsInf(ss->pdf))
                return {};

            return ss;
        }

        // Sample spherical triangle from reference point
        // Apply warp product sampling for cosine factor at reference point
        Float pdf = 1;
        if (ctx.ns != Normal3f(0, 0, 0)) {
            // Compute $\cos \theta$-based weights _w_ at sample domain corners
            Point3f rp = ctx.p();
            Vector3f wi[3] = {Normalize(p0 - rp), Normalize(p1 - rp), Normalize(p2 - rp)};
            pstd::array<Float, 4> w =
                pstd::array<Float, 4>{std::max<Float>(0.01, AbsDot(ctx.ns, wi[1])),
                                      std::max<Float>(0.01, AbsDot(ctx.ns, wi[1])),
                                      std::max<Float>(0.01, AbsDot(ctx.ns, wi[0])),
                                      std::max<Float>(0.01, AbsDot(ctx.ns, wi[2]))};

            u = SampleBilinear(u, w);
            DCHECK(u[0] >= 0 && u[0] < 1 && u[1] >= 0 && u[1] < 1);
            pdf *= BilinearPDF(u, w);
        }

        Float triPDF;
        pstd::array<Float, 3> b =
            SampleSphericalTriangle({p0, p1, p2}, ctx.p(), u, &triPDF);
        if (triPDF == 0)
            return {};
        pdf *= triPDF;
        Point3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;

        // Compute surface normal for sampled point on triangle
        Normal3f n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
        if (mesh->n != nullptr) {
            Normal3f ns(b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] +
                        (1 - b[0] - b[1]) * mesh->n[v[2]]);
            n = FaceForward(n, ns);
        } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
            n *= -1;

        // Compute error bounds _pError_ for sampled point on triangle
        Point3f pAbsSum = Abs(b[0] * p0) + Abs(b[1] * p1) + Abs((1 - b[0] - b[1]) * p2);
        Vector3f pError = Vector3f(gamma(6) * pAbsSum);

        // Return _ShapeSample_ for uniform solid angle sampled point on triangle
        return ShapeSample{Interaction(Point3fi(p, pError), n, ctx.time), pdf};
    }

    PBRT_CPU_GPU
    Float PDF(const ShapeSampleContext &ctx, const Vector3f &wi) const {
        Float solidAngle = SolidAngle(ctx.p());
        // Return PDF based on uniform area sampling for challenging triangles
        if (solidAngle < MinSphericalSampleArea || solidAngle > MaxSphericalSampleArea) {
            // Intersect sample ray with shape geometry
            Ray ray = ctx.SpawnRay(wi);
            pstd::optional<ShapeIntersection> isect = Intersect(ray);
            CHECK_RARE(1e-6, !isect.has_value());
            if (!isect)
                return 0;

            // Compute PDF in solid angle measure from shape intersection point
            Float pdf = (1 / Area()) / (AbsDot(isect->intr.n, -wi) /
                                        DistanceSquared(ctx.p(), isect->intr.p()));
            if (IsInf(pdf))
                pdf = 0;

            return pdf;
        }

        Float pdf = 1 / solidAngle;
        // Adjust PDF for warp product sampling of triangle $\cos \theta$ factor
        if (ctx.ns != Normal3f(0, 0, 0)) {
            // Get triangle vertices in _p0_, _p1_, and _p2_
            const TriangleMesh *mesh = GetMesh();
            const int *v = &mesh->vertexIndices[3 * triIndex];
            Point3f p0 = mesh->p[v[0]], p1 = mesh->p[v[1]], p2 = mesh->p[v[2]];

            Point2f u = InvertSphericalTriangleSample({p0, p1, p2}, ctx.p(), wi);
            // Compute $\cos \theta$-based weights _w_ at sample domain corners
            Point3f rp = ctx.p();
            Vector3f wi[3] = {Normalize(p0 - rp), Normalize(p1 - rp), Normalize(p2 - rp)};
            pstd::array<Float, 4> w =
                pstd::array<Float, 4>{std::max<Float>(0.01, AbsDot(ctx.ns, wi[1])),
                                      std::max<Float>(0.01, AbsDot(ctx.ns, wi[1])),
                                      std::max<Float>(0.01, AbsDot(ctx.ns, wi[0])),
                                      std::max<Float>(0.01, AbsDot(ctx.ns, wi[2]))};

            pdf *= BilinearPDF(u, w);
        }

        return pdf;
    }

  private:
    // Triangle Private Methods
    PBRT_CPU_GPU
    const TriangleMesh *GetMesh() const {
#ifdef PBRT_IS_GPU_CODE
        return (*allTriangleMeshesGPU)[meshIndex];
#else
        return (*allMeshes)[meshIndex];
#endif
    }

    // Triangle Private Members
    int meshIndex = -1, triIndex = -1;
    static pstd::vector<const TriangleMesh *> *allMeshes;
    static constexpr Float MinSphericalSampleArea = 2e-4;
    static constexpr Float MaxSphericalSampleArea = 6.22;
};

// CurveType Definition
enum class CurveType { Flat, Cylinder, Ribbon };

std::string ToString(CurveType type);

// CurveCommon Definition
struct CurveCommon {
    // CurveCommon Public Methods
    CurveCommon(pstd::span<const Point3f> c, Float w0, Float w1, CurveType type,
                pstd::span<const Normal3f> norm, const Transform *renderFromObject,
                const Transform *objectFromRender, bool reverseOrientation);

    std::string ToString() const;

    // CurveCommon Public Members
    CurveType type;
    Point3f cpObj[4];
    Float width[2];
    Normal3f n[2];
    Float normalAngle, invSinNormalAngle;
    const Transform *renderFromObject, *objectFromRender;
    bool reverseOrientation, transformSwapsHandedness;
};

// Curve Definition
class Curve {
  public:
    // Curve Public Methods
    static pstd::vector<ShapeHandle> Create(const Transform *renderFromObject,
                                            const Transform *objectFromRender,
                                            bool reverseOrientation,
                                            const ParameterDictionary &parameters,
                                            const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;
    PBRT_CPU_GPU
    Float Area() const;

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const Point2f &u) const;
    PBRT_CPU_GPU
    Float PDF(const Interaction &) const;

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx,
                                       const Point2f &u) const;
    PBRT_CPU_GPU
    Float PDF(const ShapeSampleContext &ctx, const Vector3f &wi) const;

    PBRT_CPU_GPU
    bool OrientationIsReversed() const { return common->reverseOrientation; }
    PBRT_CPU_GPU
    bool TransformSwapsHandedness() const { return common->transformSwapsHandedness; }

    std::string ToString() const;

    Curve(const CurveCommon *common, Float uMin, Float uMax)
        : common(common), uMin(uMin), uMax(uMax) {}

    PBRT_CPU_GPU
    DirectionCone NormalBounds() const { return DirectionCone::EntireSphere(); }

  private:
    // Curve Private Methods
    bool intersect(const Ray &r, Float tMax, pstd::optional<ShapeIntersection> *si) const;
    bool recursiveIntersect(const Ray &r, Float tMax, pstd::span<const Point3f> cp,
                            const Transform &ObjectFromRay, Float u0, Float u1, int depth,
                            pstd::optional<ShapeIntersection> *si) const;

    // Curve Private Members
    const CurveCommon *common;
    Float uMin, uMax;
};

// BilinearPatch Declarations
#if defined(PBRT_BUILD_GPU_RENDERER) && defined(__CUDACC__)
extern PBRT_GPU pstd::vector<const BilinearPatchMesh *> *allBilinearMeshesGPU;
#endif

// BilinearIntersection Definition
struct BilinearIntersection {
    Point2f uv;
    Float t;
    std::string ToString() const;
};

// Bilinear Patch Inline Functions
PBRT_CPU_GPU inline pstd::optional<BilinearIntersection> IntersectBilinearPatch(
    const Ray &ray, Float tMax, const Point3f &p00, const Point3f &p10,
    const Point3f &p01, const Point3f &p11) {
    // Find quadratic coefficients for distance from ray to $u$ line
    Vector3f qn = Cross(p10 - p00, p01 - p11);
    Vector3f e11 = p11 - p10, e00 = p01 - p00;
    Vector3f q00 = p00 - ray.o, q10 = p10 - ray.o;
    Float a = Dot(qn, ray.d);
    Float c = Dot(Cross(q00, ray.d), e00);
    Float b = Dot(Cross(q10, ray.d), e11) - (a + c);

    // Solve quadratic for bilinear patch intersection
    Float u1, u2;
    if (!Quadratic(a, b, c, &u1, &u2))
        return {};

    Float t = tMax, u, v;
    // Compute $(u,v)$ and ray $t$ corresponding to first quadratic root
    if (0 <= u1 && u1 <= 1) {
        Vector3f pa = Lerp(u1, q00, q10), pb = Lerp(u1, e00, e11);
        Vector3f n = Cross(ray.d, pb);
        Float det = Dot(n, n);
        n = Cross(n, pa);
        Float t1 = Dot(n, pb), v1 = Dot(n, ray.d);
        // Set _u_, _v_, and _t_ if intersection is valid
        if (t1 > 0 && 0 <= v1 && v1 <= det) {
            u = u1;
            v = v1 / det;
            t = t1 / det;
        }
    }

    // Compute $(u,v)$ and ray $t$ corresponding to second quadratic root
    if (0 <= u2 && u2 <= 1 && u2 != u1) {
        Vector3f pa = Lerp(u2, q00, q10), pb = Lerp(u2, e00, e11);
        Vector3f n = Cross(ray.d, pb);
        Float det = Dot(n, n);
        n = Cross(n, pa);
        Float t2 = Dot(n, pb) / det;
        Float v2 = Dot(n, ray.d);
        if (0 <= v2 && v2 <= det && t > t2 && t2 > 0) {
            t = t2;
            u = u2;
            v = v2 / det;
        }
    }

    // TODO: reject hits with sufficiently small t that we're not sure.
    // Check intersection $t$ against _tMax_ and possibly return intersection
    if (t >= tMax)
        return {};
    return BilinearIntersection{{u, v}, t};
}

// BilinearPatch Definition
class BilinearPatch {
  public:
    // BilinearPatch Public Methods
    BilinearPatch(int meshIndex, int blpIndex);

    static void Init(Allocator alloc);

    static BilinearPatchMesh *CreateMesh(const Transform *renderFromObject,
                                         bool reverseOrientation,
                                         const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc);

    static pstd::vector<ShapeHandle> CreatePatches(const BilinearPatchMesh *mesh,
                                                   Allocator alloc);

    PBRT_CPU_GPU
    Bounds3f Bounds() const;

    PBRT_CPU_GPU
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const;

    PBRT_CPU_GPU
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(const ShapeSampleContext &ctx, Point2f u) const;

    PBRT_CPU_GPU
    Float PDF(const ShapeSampleContext &ctx, const Vector3f &wi) const;

    PBRT_CPU_GPU
    pstd::optional<ShapeSample> Sample(Point2f u) const;

    PBRT_CPU_GPU
    Float PDF(const Interaction &) const;

    PBRT_CPU_GPU
    DirectionCone NormalBounds() const;

    std::string ToString() const;

    PBRT_CPU_GPU
    Float Area() const { return area; }

    PBRT_CPU_GPU
    static SurfaceInteraction InteractionFromIntersection(const BilinearPatchMesh *mesh,
                                                          int patchIndex,
                                                          const Point2f &uv, Float time,
                                                          const Vector3f &wo) {
        // Compute bilinear patch intersection point, $\dpdu$, and $\dpdv$
        const int *v = &mesh->vertexIndices[4 * patchIndex];
        Point3f p00 = mesh->p[v[0]], p10 = mesh->p[v[1]], p01 = mesh->p[v[2]],
                p11 = mesh->p[v[3]];
        Point3f pHit = Lerp(uv[0], Lerp(uv[1], p00, p01), Lerp(uv[1], p10, p11));
        Vector3f dpdu = Lerp(uv[1], p10, p11) - Lerp(uv[1], p00, p01);
        Vector3f dpdv = Lerp(uv[0], p01, p11) - Lerp(uv[0], p00, p10);

        Point2f uvTex = uv;
        if (mesh->uv != nullptr) {
            // Compute texture coordinates for bilinear patch intersection point
            const Point2f &uv00 = mesh->uv[v[0]], &uv10 = mesh->uv[v[1]];
            const Point2f &uv01 = mesh->uv[v[2]], &uv11 = mesh->uv[v[3]];
            uvTex = Lerp(uv[0], Lerp(uv[1], uv00, uv01), Lerp(uv[1], uv10, uv11));
            // Update bilinear patch $\dpdu$ and $\dpdv$ accounting for texture
            // coordinates Compute partial derivatives of $(u,v)$ with respect to
            // interpolated texture coordinates
            Float dsdu =
                -uv00[0] + uv10[0] + uv[1] * (uv00[0] - uv01[0] - uv10[0] + uv11[0]);
            Float dsdv =
                -uv00[0] + uv01[0] + uv[0] * (uv00[0] - uv01[0] - uv10[0] + uv11[0]);
            Float dtdu =
                -uv00[1] + uv10[1] + uv[1] * (uv00[1] - uv01[1] - uv10[1] + uv11[1]);
            Float dtdv =
                -uv00[1] + uv01[1] + uv[0] * (uv00[1] - uv01[1] - uv10[1] + uv11[1]);
            Float duds = std::abs(dsdu) < 1e-8f ? 0 : 1 / dsdu;
            Float dvds = std::abs(dsdv) < 1e-8f ? 0 : 1 / dsdv;
            Float dudt = std::abs(dtdu) < 1e-8f ? 0 : 1 / dtdu;
            Float dvdt = std::abs(dtdv) < 1e-8f ? 0 : 1 / dtdv;

            // Compute partial derivatives of $\pt{}$ with respect to interpolated texture
            // coordinates
            Vector3f dpds = dpdu * duds + dpdv * dvds;
            Vector3f dpdt = dpdu * dudt + dpdv * dvdt;

            // Set _dpdu_ and _dpdt_ to updated partial derivatives
            if (Cross(dpds, dpdt) != Vector3f(0, 0, 0)) {
                if (Dot(Cross(dpdu, dpdv), Cross(dpds, dpdt)) < 0)
                    dpdt = -dpdt;
                CHECK_GE(Dot(Normalize(Cross(dpdu, dpdv)), Normalize(Cross(dpds, dpdt))),
                         -1e-3);
                dpdu = dpds;
                dpdv = dpdt;
            }
        }
        // Find partial derivatives $\dndu$ and $\dndv$ for bilinear patch
        Vector3f d2Pduu(0, 0, 0), d2Pdvv(0, 0, 0);
        Vector3f d2Pduv(p00.x - p01.x - p10.x + p11.x, p00.y - p01.y - p10.y + p11.y,
                        p00.z - p01.z - p10.z + p11.z);
        // Compute coefficients for fundamental forms
        Float E = Dot(dpdu, dpdu);
        Float F = Dot(dpdu, dpdv);
        Float G = Dot(dpdv, dpdv);
        Vector3f N = Normalize(Cross(dpdu, dpdv));
        Float e = Dot(N, d2Pduu);
        Float f = Dot(N, d2Pduv);
        Float g = Dot(N, d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        Float invEGF2 = 1 / (E * G - F * F);
        Normal3f dndu =
            Normal3f((f * F - e * G) * invEGF2 * dpdu + (e * F - f * E) * invEGF2 * dpdv);
        Normal3f dndv =
            Normal3f((g * F - f * G) * invEGF2 * dpdu + (f * F - g * E) * invEGF2 * dpdv);

        // Initialize bilinear patch intersection point error _pError_
        Vector3f pError =
            gamma(6) * Vector3f(Max(Max(Abs(p00), Abs(p10)), Max(Abs(p01), Abs(p11))));

        // Initialize _SurfaceInteraction_ for bilinear patch intersection
        int faceIndex = mesh->faceIndices ? mesh->faceIndices[patchIndex] : 0;
        bool flipNormal = mesh->reverseOrientation ^ mesh->transformSwapsHandedness;
        SurfaceInteraction isect(Point3fi(pHit, pError), uvTex, wo, dpdu, dpdv, dndu,
                                 dndv, time, flipNormal, faceIndex);

        if (mesh->n != nullptr) {
            // Compute shading normals for bilinear patch intersection point
            Normal3f n00 = mesh->n[v[0]], n10 = mesh->n[v[1]], n01 = mesh->n[v[2]],
                     n11 = mesh->n[v[3]];
            Normal3f ns = Lerp(uv[0], Lerp(uv[1], n00, n01), Lerp(uv[1], n10, n11));
            if (LengthSquared(ns) > 0) {
                ns = Normalize(ns);
                Normal3f n = Normal3f(Normalize(isect.n));
                Vector3f axis = Cross(Vector3f(n), Vector3f(ns));
                if (LengthSquared(axis) > 1e-14f) {
                    // Set shading geometry for bilinear patch intersection
                    Normal3f dndu = Lerp(uv[1], n10, n11) - Lerp(uv[1], n00, n01);
                    Normal3f dndv = Lerp(uv[0], n01, n11) - Lerp(uv[0], n00, n10);
                    axis = Normalize(axis);
                    Float cosTheta = Dot(n, ns),
                          sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
                    Transform r = Rotate(sinTheta, cosTheta, axis);
                    Vector3f sdpdu = r(dpdu), sdpdv = r(dpdv);
                    sdpdu -= Dot(sdpdu, ns) * Vector3f(ns);
                    isect.SetShadingGeometry(ns, sdpdu, sdpdv, dndu, dndv, true);
                }
            }
        }
        return isect;
    }

  private:
    // BilinearPatch Private Methods
    PBRT_CPU_GPU
    const BilinearPatchMesh *GetMesh() const {
#ifdef PBRT_IS_GPU_CODE
        return (*allBilinearMeshesGPU)[meshIndex];
#else
        return (*allMeshes)[meshIndex];
#endif
    }

    PBRT_CPU_GPU
    bool IsRectangle() const {
        // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
        const BilinearPatchMesh *mesh = GetMesh();
        const int *v = &mesh->vertexIndices[4 * blpIndex];
        const Point3f &p00 = mesh->p[v[0]], &p10 = mesh->p[v[1]];
        const Point3f &p01 = mesh->p[v[2]], &p11 = mesh->p[v[3]];

        if (p00 == p01 || p01 == p11 || p11 == p10 || p10 == p00)
            return false;
        // Check if bilinear patch vertices are coplanar
        Normal3f n(Normalize(Cross(p10 - p00, p01 - p00)));
        if (AbsDot(Normalize(p11 - p00), n) > 1e-5f)
            return false;

        // Check if planar vertices form a rectangle
        Point3f pCenter = (p00 + p01 + p10 + p11) / 4;
        Float d2[4] = {DistanceSquared(p00, pCenter), DistanceSquared(p01, pCenter),
                       DistanceSquared(p10, pCenter), DistanceSquared(p11, pCenter)};
        for (int i = 1; i < 4; ++i)
            if (std::abs(d2[i] - d2[0]) / d2[0] > 1e-4f)
                return false;
        return true;
    }

    // BilinearPatch Private Members
    int meshIndex, blpIndex;
    Float area;
    static pstd::vector<const BilinearPatchMesh *> *allMeshes;
    static constexpr Float MinSphericalSampleArea = 1e-4;
};

inline Bounds3f ShapeHandle::Bounds() const {
    auto bounds = [&](auto ptr) { return ptr->Bounds(); };
    return Dispatch(bounds);
}

inline pstd::optional<ShapeIntersection> ShapeHandle::Intersect(const Ray &ray,
                                                                Float tMax) const {
    auto intr = [&](auto ptr) { return ptr->Intersect(ray, tMax); };
    return Dispatch(intr);
}

inline bool ShapeHandle::IntersectP(const Ray &ray, Float tMax) const {
    auto intr = [&](auto ptr) { return ptr->IntersectP(ray, tMax); };
    return Dispatch(intr);
}

inline Float ShapeHandle::Area() const {
    auto area = [&](auto ptr) { return ptr->Area(); };
    return Dispatch(area);
}

inline pstd::optional<ShapeSample> ShapeHandle::Sample(const Point2f &u) const {
    auto sample = [&](auto ptr) { return ptr->Sample(u); };
    return Dispatch(sample);
}

inline Float ShapeHandle::PDF(const Interaction &in) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(in); };
    return Dispatch(pdf);
}

inline pstd::optional<ShapeSample> ShapeHandle::Sample(const ShapeSampleContext &ctx,
                                                       const Point2f &u) const {
    auto sample = [&](auto ptr) { return ptr->Sample(ctx, u); };
    return Dispatch(sample);
}

inline Float ShapeHandle::PDF(const ShapeSampleContext &ctx, const Vector3f &wi) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(ctx, wi); };
    return Dispatch(pdf);
}

inline DirectionCone ShapeHandle::NormalBounds() const {
    auto nb = [&](auto ptr) { return ptr->NormalBounds(); };
    return Dispatch(nb);
}

}  // namespace pbrt

#endif  // PBRT_SHAPES_H
