// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_TRANSFORM_H
#define PBRT_UTIL_TRANSFORM_H

#include <pbrt/pbrt.h>

#include <pbrt/ray.h>
#include <pbrt/util/float.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <stdio.h>
#include <cmath>
#include <limits>
#include <memory>

namespace pbrt {

// Transform Definition
class Transform {
  public:
    // Transform Public Methods
    PBRT_CPU_GPU
    inline Ray ApplyInverse(const Ray &r, Float *tMax = nullptr) const;
    PBRT_CPU_GPU
    inline RayDifferential ApplyInverse(const RayDifferential &r,
                                        Float *tMax = nullptr) const;
    template <typename T>
    PBRT_CPU_GPU inline Vector3<T> ApplyInverse(Vector3<T> v) const;
    template <typename T>
    PBRT_CPU_GPU inline Normal3<T> ApplyInverse(Normal3<T>) const;

    uint64_t Hash() const { return HashBuffer<sizeof(m)>(&m); }

    std::string ToString() const;

    Transform() = default;

    PBRT_CPU_GPU
    Transform(const SquareMatrix<4> &m) : m(m) {
        pstd::optional<SquareMatrix<4>> inv = Inverse(m);
        if (inv)
            mInv = *inv;
        else {
            // Initialize _mInv_ with not-a-number values
            Float NaN = std::numeric_limits<Float>::has_signaling_NaN
                            ? std::numeric_limits<Float>::signaling_NaN()
                            : std::numeric_limits<Float>::quiet_NaN();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    mInv[i][j] = NaN;
        }
    }

    PBRT_CPU_GPU
    Transform(const Float mat[4][4]) : Transform(SquareMatrix<4>(mat)) {}

    PBRT_CPU_GPU
    Transform(const SquareMatrix<4> &m, const SquareMatrix<4> &mInv) : m(m), mInv(mInv) {}

    PBRT_CPU_GPU
    const SquareMatrix<4> &GetMatrix() const { return m; }
    PBRT_CPU_GPU
    const SquareMatrix<4> &GetInverseMatrix() const { return mInv; }

    PBRT_CPU_GPU
    bool operator==(const Transform &t) const { return t.m == m; }
    PBRT_CPU_GPU
    bool operator!=(const Transform &t) const { return t.m != m; }
    PBRT_CPU_GPU
    bool IsIdentity() const { return m.IsIdentity(); }

    PBRT_CPU_GPU
    bool HasScale(Float tolerance = 1e-3f) const {
        Float la2 = LengthSquared((*this)(Vector3f(1, 0, 0)));
        Float lb2 = LengthSquared((*this)(Vector3f(0, 1, 0)));
        Float lc2 = LengthSquared((*this)(Vector3f(0, 0, 1)));
        return (std::abs(la2 - 1) > tolerance || std::abs(lb2 - 1) > tolerance ||
                std::abs(lc2 - 1) > tolerance);
    }

    template <typename T>
    PBRT_CPU_GPU Point3<T> operator()(Point3<T> p) const;

    template <typename T>
    PBRT_CPU_GPU inline Point3<T> ApplyInverse(Point3<T> p) const;

    template <typename T>
    PBRT_CPU_GPU Vector3<T> operator()(Vector3<T> v) const;

    template <typename T>
    PBRT_CPU_GPU Normal3<T> operator()(Normal3<T>) const;

    PBRT_CPU_GPU
    Ray operator()(const Ray &r, Float *tMax = nullptr) const;
    PBRT_CPU_GPU
    RayDifferential operator()(const RayDifferential &r, Float *tMax = nullptr) const;

    PBRT_CPU_GPU
    Bounds3f operator()(const Bounds3f &b) const;

    PBRT_CPU_GPU
    Transform operator*(const Transform &t2) const;

    PBRT_CPU_GPU
    bool SwapsHandedness() const;

    PBRT_CPU_GPU
    explicit Transform(const Frame &frame);

    PBRT_CPU_GPU
    explicit Transform(Quaternion q);

    PBRT_CPU_GPU
    explicit operator Quaternion() const;

    void Decompose(Vector3f *T, SquareMatrix<4> *R, SquareMatrix<4> *S) const;

    PBRT_CPU_GPU
    Interaction operator()(const Interaction &in) const;
    PBRT_CPU_GPU
    Interaction ApplyInverse(const Interaction &in) const;
    PBRT_CPU_GPU
    SurfaceInteraction operator()(const SurfaceInteraction &si) const;
    PBRT_CPU_GPU
    SurfaceInteraction ApplyInverse(const SurfaceInteraction &in) const;

    PBRT_CPU_GPU
    Point3fi operator()(const Point3fi &p) const {
        Float x = Float(p.x), y = Float(p.y), z = Float(p.z);
        // Compute transformed coordinates from point _x_, _y_, and _z_
        Float xp = (m[0][0] * x + m[0][1] * y) + (m[0][2] * z + m[0][3]);
        Float yp = (m[1][0] * x + m[1][1] * y) + (m[1][2] * z + m[1][3]);
        Float zp = (m[2][0] * x + m[2][1] * y) + (m[2][2] * z + m[2][3]);
        Float wp = (m[3][0] * x + m[3][1] * y) + (m[3][2] * z + m[3][3]);

        // Compute absolute error for transformed point, _pError_
        Vector3f pError;
        if (p.IsExact()) {
            // Compute error for transformed exact _p_
            pError.x = gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                   std::abs(m[0][2] * z) + std::abs(m[0][3]));
            pError.y = gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                   std::abs(m[1][2] * z) + std::abs(m[1][3]));
            pError.z = gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                   std::abs(m[2][2] * z) + std::abs(m[2][3]));

        } else {
            // Compute error for transformed approximate _p_
            Vector3f pInError = p.Error();
            pError.x = (gamma(3) + 1) * (std::abs(m[0][0]) * pInError.x +
                                         std::abs(m[0][1]) * pInError.y +
                                         std::abs(m[0][2]) * pInError.z) +
                       gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                   std::abs(m[0][2] * z) + std::abs(m[0][3]));
            pError.y = (gamma(3) + 1) * (std::abs(m[1][0]) * pInError.x +
                                         std::abs(m[1][1]) * pInError.y +
                                         std::abs(m[1][2]) * pInError.z) +
                       gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                   std::abs(m[1][2] * z) + std::abs(m[1][3]));
            pError.z = (gamma(3) + 1) * (std::abs(m[2][0]) * pInError.x +
                                         std::abs(m[2][1]) * pInError.y +
                                         std::abs(m[2][2]) * pInError.z) +
                       gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                   std::abs(m[2][2] * z) + std::abs(m[2][3]));
        }

        if (wp == 1)
            return Point3fi(Point3f(xp, yp, zp), pError);
        else
            return Point3fi(Point3f(xp, yp, zp), pError) / wp;
    }

    PBRT_CPU_GPU
    Vector3fi operator()(const Vector3fi &v) const;
    PBRT_CPU_GPU
    Point3fi ApplyInverse(const Point3fi &p) const;

  private:
    // Transform Private Members
    SquareMatrix<4> m, mInv;
};

// Transform Function Declarations
PBRT_CPU_GPU
Transform Translate(Vector3f delta);

PBRT_CPU_GPU
Transform Scale(Float x, Float y, Float z);

PBRT_CPU_GPU
Transform RotateX(Float theta);
PBRT_CPU_GPU
Transform RotateY(Float theta);
PBRT_CPU_GPU
Transform RotateZ(Float theta);

PBRT_CPU_GPU
Transform LookAt(Point3f pos, Point3f look, Vector3f up);

PBRT_CPU_GPU
Transform Orthographic(Float znear, Float zfar);

PBRT_CPU_GPU
Transform Perspective(Float fov, Float znear, Float zfar);

// Transform Inline Functions
PBRT_CPU_GPU inline Transform Inverse(const Transform &t) {
    return Transform(t.GetInverseMatrix(), t.GetMatrix());
}

PBRT_CPU_GPU inline Transform Transpose(const Transform &t) {
    return Transform(Transpose(t.GetMatrix()), Transpose(t.GetInverseMatrix()));
}

PBRT_CPU_GPU inline Transform Rotate(Float sinTheta, Float cosTheta, Vector3f axis) {
    Vector3f a = Normalize(axis);
    SquareMatrix<4> m;
    // Compute rotation of first basis vector
    m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m[0][3] = 0;

    // Compute rotations of second and third basis vectors
    m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m[1][3] = 0;

    m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m[2][3] = 0;

    return Transform(m, Transpose(m));
}

PBRT_CPU_GPU inline Transform Rotate(Float theta, Vector3f axis) {
    Float sinTheta = std::sin(Radians(theta));
    Float cosTheta = std::cos(Radians(theta));
    return Rotate(sinTheta, cosTheta, axis);
}

PBRT_CPU_GPU inline Transform RotateFromTo(Vector3f from, Vector3f to) {
    // Compute intermediate vector for vector reflection
    Vector3f refl;
    if (std::abs(from.x) < 0.72f && std::abs(to.x) < 0.72f)
        refl = Vector3f(1, 0, 0);
    else if (std::abs(from.y) < 0.72f && std::abs(to.y) < 0.72f)
        refl = Vector3f(0, 1, 0);
    else
        refl = Vector3f(0, 0, 1);

    // Initialize matrix _r_ for rotation
    Vector3f u = refl - from, v = refl - to;
    SquareMatrix<4> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            // Initialize matrix element _r[i][j]_
            r[i][j] = ((i == j) ? 1 : 0) - 2 / Dot(u, u) * u[i] * u[j] -
                      2 / Dot(v, v) * v[i] * v[j] +
                      4 * Dot(u, v) / (Dot(u, u) * Dot(v, v)) * v[i] * u[j];

    return Transform(r, Transpose(r));
}

inline Vector3fi Transform::operator()(const Vector3fi &v) const {
    Float x = Float(v.x), y = Float(v.y), z = Float(v.z);
    Vector3f vOutError;
    if (v.IsExact()) {
        vOutError.x = gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                  std::abs(m[0][2] * z));
        vOutError.y = gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                  std::abs(m[1][2] * z));
        vOutError.z = gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                  std::abs(m[2][2] * z));
    } else {
        Vector3f vInError = v.Error();
        vOutError.x = (gamma(3) + 1) * (std::abs(m[0][0]) * vInError.x +
                                        std::abs(m[0][1]) * vInError.y +
                                        std::abs(m[0][2]) * vInError.z) +
                      gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) +
                                  std::abs(m[0][2] * z));
        vOutError.y = (gamma(3) + 1) * (std::abs(m[1][0]) * vInError.x +
                                        std::abs(m[1][1]) * vInError.y +
                                        std::abs(m[1][2]) * vInError.z) +
                      gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) +
                                  std::abs(m[1][2] * z));
        vOutError.z = (gamma(3) + 1) * (std::abs(m[2][0]) * vInError.x +
                                        std::abs(m[2][1]) * vInError.y +
                                        std::abs(m[2][2]) * vInError.z) +
                      gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) +
                                  std::abs(m[2][2] * z));
    }

    Float xp = m[0][0] * x + m[0][1] * y + m[0][2] * z;
    Float yp = m[1][0] * x + m[1][1] * y + m[1][2] * z;
    Float zp = m[2][0] * x + m[2][1] * y + m[2][2] * z;

    return Vector3fi(Vector3f(xp, yp, zp), vOutError);
}

// Transform Inline Methods
template <typename T>
inline Point3<T> Transform::operator()(Point3<T> p) const {
    T xp = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
    T yp = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
    T zp = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
    T wp = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
inline Vector3<T> Transform::operator()(Vector3<T> v) const {
    return Vector3<T>(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                      m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                      m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
}

template <typename T>
inline Normal3<T> Transform::operator()(Normal3<T> n) const {
    T x = n.x, y = n.y, z = n.z;
    return Normal3<T>(mInv[0][0] * x + mInv[1][0] * y + mInv[2][0] * z,
                      mInv[0][1] * x + mInv[1][1] * y + mInv[2][1] * z,
                      mInv[0][2] * x + mInv[1][2] * y + mInv[2][2] * z);
}

inline Ray Transform::operator()(const Ray &r, Float *tMax) const {
    Point3fi o = (*this)(Point3fi(r.o));
    Vector3f d = (*this)(r.d);
    // Offset ray origin to edge of error bounds and compute _tMax_
    if (Float lengthSquared = LengthSquared(d); lengthSquared > 0) {
        Float dt = Dot(Abs(d), o.Error()) / lengthSquared;
        o += d * dt;
        if (tMax)
            *tMax -= dt;
    }

    return Ray(Point3f(o), d, r.time, r.medium);
}

inline RayDifferential Transform::operator()(const RayDifferential &r,
                                             Float *tMax) const {
    Ray tr = (*this)(Ray(r), tMax);
    RayDifferential ret(tr.o, tr.d, tr.time, tr.medium);
    ret.hasDifferentials = r.hasDifferentials;
    ret.rxOrigin = (*this)(r.rxOrigin);
    ret.ryOrigin = (*this)(r.ryOrigin);
    ret.rxDirection = (*this)(r.rxDirection);
    ret.ryDirection = (*this)(r.ryDirection);
    return ret;
}

inline Transform::Transform(const Frame &frame)
    : Transform(SquareMatrix<4>(frame.x.x, frame.x.y, frame.x.z, 0, frame.y.x, frame.y.y,
                                frame.y.z, 0, frame.z.x, frame.z.y, frame.z.z, 0, 0, 0, 0,
                                1)) {}

inline Transform::Transform(Quaternion q) {
    Float xx = q.v.x * q.v.x, yy = q.v.y * q.v.y, zz = q.v.z * q.v.z;
    Float xy = q.v.x * q.v.y, xz = q.v.x * q.v.z, yz = q.v.y * q.v.z;
    Float wx = q.v.x * q.w, wy = q.v.y * q.w, wz = q.v.z * q.w;

    mInv[0][0] = 1 - 2 * (yy + zz);
    mInv[0][1] = 2 * (xy + wz);
    mInv[0][2] = 2 * (xz - wy);
    mInv[1][0] = 2 * (xy - wz);
    mInv[1][1] = 1 - 2 * (xx + zz);
    mInv[1][2] = 2 * (yz + wx);
    mInv[2][0] = 2 * (xz + wy);
    mInv[2][1] = 2 * (yz - wx);
    mInv[2][2] = 1 - 2 * (xx + yy);

    // Transpose since we are left-handed.  Ugh.
    m = Transpose(mInv);
}

template <typename T>
inline Point3<T> Transform::ApplyInverse(Point3<T> p) const {
    T x = p.x, y = p.y, z = p.z;
    T xp = (mInv[0][0] * x + mInv[0][1] * y) + (mInv[0][2] * z + mInv[0][3]);
    T yp = (mInv[1][0] * x + mInv[1][1] * y) + (mInv[1][2] * z + mInv[1][3]);
    T zp = (mInv[2][0] * x + mInv[2][1] * y) + (mInv[2][2] * z + mInv[2][3]);
    T wp = (mInv[3][0] * x + mInv[3][1] * y) + (mInv[3][2] * z + mInv[3][3]);
    CHECK_NE(wp, 0);
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
inline Vector3<T> Transform::ApplyInverse(Vector3<T> v) const {
    T x = v.x, y = v.y, z = v.z;
    return Vector3<T>(mInv[0][0] * x + mInv[0][1] * y + mInv[0][2] * z,
                      mInv[1][0] * x + mInv[1][1] * y + mInv[1][2] * z,
                      mInv[2][0] * x + mInv[2][1] * y + mInv[2][2] * z);
}

template <typename T>
inline Normal3<T> Transform::ApplyInverse(Normal3<T> n) const {
    T x = n.x, y = n.y, z = n.z;
    return Normal3<T>(m[0][0] * x + m[1][0] * y + m[2][0] * z,
                      m[0][1] * x + m[1][1] * y + m[2][1] * z,
                      m[0][2] * x + m[1][2] * y + m[2][2] * z);
}

inline Ray Transform::ApplyInverse(const Ray &r, Float *tMax) const {
    Point3fi o = ApplyInverse(Point3fi(r.o));
    Vector3f d = ApplyInverse(r.d);
    // Offset ray origin to edge of error bounds and compute _tMax_
    Float lengthSquared = LengthSquared(d);
    if (lengthSquared > 0) {
        Vector3f oError(o.x.Width() / 2, o.y.Width() / 2, o.z.Width() / 2);
        Float dt = Dot(Abs(d), oError) / lengthSquared;
        o += d * dt;
        if (tMax)
            *tMax -= dt;
    }
    return Ray(Point3f(o), d, r.time, r.medium);
}

inline RayDifferential Transform::ApplyInverse(const RayDifferential &r,
                                               Float *tMax) const {
    Ray tr = ApplyInverse(Ray(r), tMax);
    RayDifferential ret(tr.o, tr.d, tr.time, tr.medium);
    ret.hasDifferentials = r.hasDifferentials;
    ret.rxOrigin = ApplyInverse(r.rxOrigin);
    ret.ryOrigin = ApplyInverse(r.ryOrigin);
    ret.rxDirection = ApplyInverse(r.rxDirection);
    ret.ryDirection = ApplyInverse(r.ryDirection);
    return ret;
}

// AnimatedTransform Definition
class AnimatedTransform {
  public:
    // AnimatedTransform Public Methods
    AnimatedTransform() = default;
    explicit AnimatedTransform(const Transform &t) : AnimatedTransform(t, 0, t, 1) {}
    AnimatedTransform(const Transform &startTransform, Float startTime,
                      const Transform &endTransform, Float endTime);

    PBRT_CPU_GPU
    bool IsAnimated() const { return actuallyAnimated; }

    PBRT_CPU_GPU
    Ray ApplyInverse(const Ray &r, Float *tMax = nullptr) const;

    PBRT_CPU_GPU
    Point3f ApplyInverse(Point3f p, Float time) const {
        if (!actuallyAnimated)
            return startTransform.ApplyInverse(p);
        return Interpolate(time).ApplyInverse(p);
    }
    PBRT_CPU_GPU
    Vector3f ApplyInverse(Vector3f v, Float time) const {
        if (!actuallyAnimated)
            return startTransform.ApplyInverse(v);
        return Interpolate(time).ApplyInverse(v);
    }
    PBRT_CPU_GPU
    Normal3f operator()(Normal3f n, Float time) const;
    PBRT_CPU_GPU
    Normal3f ApplyInverse(Normal3f n, Float time) const {
        if (!actuallyAnimated)
            return startTransform.ApplyInverse(n);
        return Interpolate(time).ApplyInverse(n);
    }
    PBRT_CPU_GPU
    Interaction operator()(const Interaction &it) const;
    PBRT_CPU_GPU
    Interaction ApplyInverse(const Interaction &it) const;
    PBRT_CPU_GPU
    SurfaceInteraction operator()(const SurfaceInteraction &it) const;
    PBRT_CPU_GPU
    SurfaceInteraction ApplyInverse(const SurfaceInteraction &it) const;
    PBRT_CPU_GPU
    bool HasScale() const { return startTransform.HasScale() || endTransform.HasScale(); }

    std::string ToString() const;

    PBRT_CPU_GPU
    Transform Interpolate(Float time) const;

    PBRT_CPU_GPU
    Ray operator()(const Ray &r, Float *tMax = nullptr) const;
    PBRT_CPU_GPU
    RayDifferential operator()(const RayDifferential &r, Float *tMax = nullptr) const;
    PBRT_CPU_GPU
    Point3f operator()(Point3f p, Float time) const;
    PBRT_CPU_GPU
    Vector3f operator()(Vector3f v, Float time) const;

    PBRT_CPU_GPU
    Bounds3f MotionBounds(const Bounds3f &b) const;

    PBRT_CPU_GPU
    Bounds3f BoundPointMotion(Point3f p) const;

    // AnimatedTransform Public Members
    Transform startTransform, endTransform;
    Float startTime = 0, endTime = 1;

  private:
    // AnimatedTransform Private Methods
    PBRT_CPU_GPU
    static void FindZeros(Float c1, Float c2, Float c3, Float c4, Float c5, Float theta,
                          Interval tInterval, pstd::span<Float> zeros, int *nZeros,
                          int depth = 8);

    // AnimatedTransform Private Members
    bool actuallyAnimated = false;
    Vector3f T[2];
    Quaternion R[2];
    SquareMatrix<4> S[2];
    bool hasRotation;
    struct DerivativeTerm {
        PBRT_CPU_GPU
        DerivativeTerm() {}
        PBRT_CPU_GPU
        DerivativeTerm(Float c, Float x, Float y, Float z) : kc(c), kx(x), ky(y), kz(z) {}
        Float kc, kx, ky, kz;
        PBRT_CPU_GPU
        Float Eval(Point3f p) const { return kc + kx * p.x + ky * p.y + kz * p.z; }
    };
    DerivativeTerm c1[3], c2[3], c3[3], c4[3], c5[3];
};

}  // namespace pbrt

#endif  // PBRT_UTIL_TRANSFORM_H
