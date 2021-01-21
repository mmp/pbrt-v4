// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_RAY_H
#define PBRT_RAY_H

#include <pbrt/pbrt.h>

#include <pbrt/base/medium.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// Ray Definition
class Ray {
  public:
    // Ray Public Methods
    PBRT_CPU_GPU
    bool HasNaN() const { return (o.HasNaN() || d.HasNaN()); }

    std::string ToString() const;

    Ray() = default;
    PBRT_CPU_GPU
    Ray(Point3f o, Vector3f d, Float time = 0.f, MediumHandle medium = nullptr)
        : o(o), d(d), time(time), medium(medium) {}

    PBRT_CPU_GPU
    Point3f operator()(Float t) const { return o + d * t; }

    // Ray Public Members
    Point3f o;
    Vector3f d;
    Float time = 0;
    MediumHandle medium = nullptr;
};

// RayDifferential Definition
class RayDifferential : public Ray {
  public:
    // RayDifferential Public Methods
    RayDifferential() = default;
    PBRT_CPU_GPU
    RayDifferential(Point3f o, Vector3f d, Float time = 0.f,
                    MediumHandle medium = nullptr)
        : Ray(o, d, time, medium) {}

    PBRT_CPU_GPU
    explicit RayDifferential(const Ray &ray) : Ray(ray) {}

    void ScaleDifferentials(Float s) {
        rxOrigin = o + (rxOrigin - o) * s;
        ryOrigin = o + (ryOrigin - o) * s;
        rxDirection = d + (rxDirection - d) * s;
        ryDirection = d + (ryDirection - d) * s;
    }

    PBRT_CPU_GPU
    bool HasNaN() const {
        return Ray::HasNaN() ||
               (hasDifferentials && (rxOrigin.HasNaN() || ryOrigin.HasNaN() ||
                                     rxDirection.HasNaN() || ryDirection.HasNaN()));
    }
    std::string ToString() const;

    // RayDifferential Public Members
    bool hasDifferentials = false;
    Point3f rxOrigin, ryOrigin;
    Vector3f rxDirection, ryDirection;
};

// Ray Inline Functions
PBRT_CPU_GPU inline Point3f OffsetRayOrigin(Point3fi pi, Normal3f n, Vector3f w) {
    // Find vector _offset_ to corner of error bounds and compute initial _po_
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

PBRT_CPU_GPU inline Ray SpawnRay(Point3fi pi, Normal3f n, Float time, Vector3f d) {
    return Ray(OffsetRayOrigin(pi, n, d), d, time);
}

PBRT_CPU_GPU inline Ray SpawnRayTo(Point3fi pFrom, Normal3f n, Float time, Point3f pTo) {
    Vector3f d = pTo - Point3f(pFrom);
    return SpawnRay(pFrom, n, time, d);
}

PBRT_CPU_GPU inline Ray SpawnRayTo(Point3fi pFrom, Normal3f nFrom, Float time,
                                   Point3fi pTo, Normal3f nTo) {
    Point3f pf = OffsetRayOrigin(pFrom, nFrom, Point3f(pTo) - Point3f(pFrom));
    Point3f pt = OffsetRayOrigin(pTo, nTo, pf - Point3f(pTo));
    return Ray(pf, pt - pf, time);
}

}  // namespace pbrt

#endif  // PBRT_RAY_H
