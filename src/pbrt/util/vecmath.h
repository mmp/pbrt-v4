// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_VECMATH_H
#define PBRT_UTIL_VECMATH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>

namespace pbrt {

namespace internal {

template <typename T>
std::string ToString2(T x, T y);
template <typename T>
std::string ToString3(T x, T y, T z);

}  // namespace internal

extern template std::string internal::ToString2(float, float);
extern template std::string internal::ToString2(double, double);
extern template std::string internal::ToString2(int, int);
extern template std::string internal::ToString3(float, float, float);
extern template std::string internal::ToString3(double, double, double);
extern template std::string internal::ToString3(int, int, int);

namespace {

PBRT_CPU_GPU inline bool IsNaN(Interval fi) {
    return pbrt::IsNaN(Float(fi));
}

// TupleLength Definition
template <typename T>
struct TupleLength {
    using type = Float;
};

template <>
struct TupleLength<double> {
    using type = double;
};

template <>
struct TupleLength<long double> {
    using type = long double;
};

template <>
struct TupleLength<Interval> {
    using type = Interval;
};

}  // anonymous namespace

// Tuple2 Definition
template <template <typename> class Child, typename T>
class Tuple2 {
  public:
    // Tuple2 Public Methods
    static const int nDimensions = 2;

    Tuple2() = default;
    PBRT_CPU_GPU
    Tuple2(T x, T y) : x(x), y(y) { DCHECK(!HasNaN()); }
    PBRT_CPU_GPU
    bool HasNaN() const { return IsNaN(x) || IsNaN(y); }
#ifdef PBRT_DEBUG_BUILD
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    PBRT_CPU_GPU
    Tuple2(Child<T> c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
    }
    PBRT_CPU_GPU
    Child<T> &operator=(Child<T> c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        return static_cast<Child<T> &>(*this);
    }
#endif

    template <typename U>
    PBRT_CPU_GPU auto operator+(Child<U> c) const -> Child<decltype(T{} + U{})> {
        DCHECK(!c.HasNaN());
        return {x + c.x, y + c.y};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator+=(Child<U> c) {
        DCHECK(!c.HasNaN());
        x += c.x;
        y += c.y;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Child<U> c) const -> Child<decltype(T{} - U{})> {
        DCHECK(!c.HasNaN());
        return {x - c.x, y - c.y};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator-=(Child<U> c) {
        DCHECK(!c.HasNaN());
        x -= c.x;
        y -= c.y;
        return static_cast<Child<T> &>(*this);
    }

    PBRT_CPU_GPU
    bool operator==(Child<T> c) const { return x == c.x && y == c.y; }
    PBRT_CPU_GPU
    bool operator!=(Child<T> c) const { return x != c.x || y != c.y; }

    template <typename U>
    PBRT_CPU_GPU auto operator*(U s) const -> Child<decltype(T{} * U{})> {
        return {s * x, s * y};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator*=(U s) {
        DCHECK(!IsNaN(s));
        x *= s;
        y *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_CPU_GPU auto operator/(U d) const -> Child<decltype(T{} / U{})> {
        DCHECK(d != 0 && !IsNaN(d));
        return {x / d, y / d};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator/=(U d) {
        DCHECK_NE(d, 0);
        DCHECK(!IsNaN(d));
        x /= d;
        y /= d;
        return static_cast<Child<T> &>(*this);
    }

    PBRT_CPU_GPU
    Child<T> operator-() const { return {-x, -y}; }

    PBRT_CPU_GPU
    T operator[](int i) const {
        DCHECK(i >= 0 && i <= 1);
        return (i == 0) ? x : y;
    }

    PBRT_CPU_GPU
    T &operator[](int i) {
        DCHECK(i >= 0 && i <= 1);
        return (i == 0) ? x : y;
    }

    std::string ToString() const { return internal::ToString2(x, y); }

    // Tuple2 Public Members
    T x{}, y{};
};

// Tuple2 Inline Functions
template <template <class> class C, typename T, typename U>
PBRT_CPU_GPU inline auto operator*(U s, Tuple2<C, T> t) -> C<decltype(T{} * U{})> {
    DCHECK(!t.HasNaN());
    return t * s;
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Abs(Tuple2<C, T> t) {
    // "argument-dependent lookup..." (here and elsewhere)
    using std::abs;
    return {abs(t.x), abs(t.y)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Ceil(Tuple2<C, T> t) {
    using pstd::ceil;
    return {ceil(t.x), ceil(t.y)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Floor(Tuple2<C, T> t) {
    using pstd::floor;
    return {floor(t.x), floor(t.y)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline auto Lerp(Float t, Tuple2<C, T> t0, Tuple2<C, T> t1) {
    return (1 - t) * t0 + t * t1;
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> FMA(Float a, Tuple2<C, T> b, Tuple2<C, T> c) {
    return {FMA(a, b.x, c.x), FMA(a, b.y, c.y)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> FMA(Tuple2<C, T> a, Float b, Tuple2<C, T> c) {
    return FMA(b, a, c);
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Min(Tuple2<C, T> t0, Tuple2<C, T> t1) {
    using std::min;
    return {min(t0.x, t1.x), min(t0.y, t1.y)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline T MinComponentValue(Tuple2<C, T> t) {
    using std::min;
    return min({t.x, t.y});
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline int MinComponentIndex(Tuple2<C, T> t) {
    return (t.x < t.y) ? 0 : 1;
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Max(Tuple2<C, T> t0, Tuple2<C, T> t1) {
    using std::max;
    return {max(t0.x, t1.x), max(t0.y, t1.y)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline T MaxComponentValue(Tuple2<C, T> t) {
    using std::max;
    return max({t.x, t.y});
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline int MaxComponentIndex(Tuple2<C, T> t) {
    return (t.x > t.y) ? 0 : 1;
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Permute(Tuple2<C, T> t, pstd::array<int, 2> p) {
    return {t[p[0]], t[p[1]]};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline T HProd(Tuple2<C, T> t) {
    return t.x * t.y;
}

// Tuple3 Definition
template <template <typename> class Child, typename T>
class Tuple3 {
  public:
    // Tuple3 Public Methods
    Tuple3() = default;
    PBRT_CPU_GPU
    Tuple3(T x, T y, T z) : x(x), y(y), z(z) { DCHECK(!HasNaN()); }

    PBRT_CPU_GPU
    bool HasNaN() const { return IsNaN(x) || IsNaN(y) || IsNaN(z); }

    PBRT_CPU_GPU
    T operator[](int i) const {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    PBRT_CPU_GPU
    T &operator[](int i) {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    template <typename U>
    PBRT_CPU_GPU auto operator+(Child<U> c) const -> Child<decltype(T{} + U{})> {
        DCHECK(!c.HasNaN());
        return {x + c.x, y + c.y, z + c.z};
    }

    static const int nDimensions = 3;

#ifdef PBRT_DEBUG_BUILD
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    PBRT_CPU_GPU
    Tuple3(Child<T> c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        z = c.z;
    }

    PBRT_CPU_GPU
    Child<T> &operator=(Child<T> c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        z = c.z;
        return static_cast<Child<T> &>(*this);
    }
#endif

    template <typename U>
    PBRT_CPU_GPU Child<T> &operator+=(Child<U> c) {
        DCHECK(!c.HasNaN());
        x += c.x;
        y += c.y;
        z += c.z;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Child<U> c) const -> Child<decltype(T{} - U{})> {
        DCHECK(!c.HasNaN());
        return {x - c.x, y - c.y, z - c.z};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator-=(Child<U> c) {
        DCHECK(!c.HasNaN());
        x -= c.x;
        y -= c.y;
        z -= c.z;
        return static_cast<Child<T> &>(*this);
    }

    PBRT_CPU_GPU
    bool operator==(Child<T> c) const { return x == c.x && y == c.y && z == c.z; }
    PBRT_CPU_GPU
    bool operator!=(Child<T> c) const { return x != c.x || y != c.y || z != c.z; }

    template <typename U>
    PBRT_CPU_GPU auto operator*(U s) const -> Child<decltype(T{} * U{})> {
        return {s * x, s * y, s * z};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator*=(U s) {
        DCHECK(!IsNaN(s));
        x *= s;
        y *= s;
        z *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_CPU_GPU auto operator/(U d) const -> Child<decltype(T{} / U{})> {
        DCHECK_NE(d, 0);
        return {x / d, y / d, z / d};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator/=(U d) {
        DCHECK_NE(d, 0);
        x /= d;
        y /= d;
        z /= d;
        return static_cast<Child<T> &>(*this);
    }
    PBRT_CPU_GPU
    Child<T> operator-() const { return {-x, -y, -z}; }

    std::string ToString() const { return internal::ToString3(x, y, z); }

    // Tuple3 Public Members
    T x{}, y{}, z{};
};

// Tuple3 Inline Functions
template <template <class> class C, typename T, typename U>
PBRT_CPU_GPU inline auto operator*(U s, Tuple3<C, T> t) -> C<decltype(T{} * U{})> {
    return t * s;
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Abs(Tuple3<C, T> t) {
    using std::abs;
    return {abs(t.x), abs(t.y), abs(t.z)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Ceil(Tuple3<C, T> t) {
    using pstd::ceil;
    return {ceil(t.x), ceil(t.y), ceil(t.z)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Floor(Tuple3<C, T> t) {
    using pstd::floor;
    return {floor(t.x), floor(t.y), floor(t.z)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline auto Lerp(Float t, Tuple3<C, T> t0, Tuple3<C, T> t1) {
    return (1 - t) * t0 + t * t1;
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> FMA(Float a, Tuple3<C, T> b, Tuple3<C, T> c) {
    return {FMA(a, b.x, c.x), FMA(a, b.y, c.y), FMA(a, b.z, c.z)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> FMA(Tuple3<C, T> a, Float b, Tuple3<C, T> c) {
    return FMA(b, a, c);
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Min(Tuple3<C, T> t1, Tuple3<C, T> t2) {
    using std::min;
    return {min(t1.x, t2.x), min(t1.y, t2.y), min(t1.z, t2.z)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline T MinComponentValue(Tuple3<C, T> t) {
    using std::min;
    return min({t.x, t.y, t.z});
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline int MinComponentIndex(Tuple3<C, T> t) {
    return (t.x < t.y) ? ((t.x < t.z) ? 0 : 2) : ((t.y < t.z) ? 1 : 2);
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Max(Tuple3<C, T> t1, Tuple3<C, T> t2) {
    using std::max;
    return {max(t1.x, t2.x), max(t1.y, t2.y), max(t1.z, t2.z)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline T MaxComponentValue(Tuple3<C, T> t) {
    using std::max;
    return max({t.x, t.y, t.z});
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline int MaxComponentIndex(Tuple3<C, T> t) {
    return (t.x > t.y) ? ((t.x > t.z) ? 0 : 2) : ((t.y > t.z) ? 1 : 2);
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Permute(Tuple3<C, T> t, pstd::array<int, 3> p) {
    return {t[p[0]], t[p[1]], t[p[2]]};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline T HProd(Tuple3<C, T> t) {
    return t.x * t.y * t.z;
}

// Vector2 Definition
template <typename T>
class Vector2 : public Tuple2<Vector2, T> {
  public:
    // Vector2 Public Methods
    using Tuple2<Vector2, T>::x;
    using Tuple2<Vector2, T>::y;

    Vector2() = default;
    PBRT_CPU_GPU
    Vector2(T x, T y) : Tuple2<pbrt::Vector2, T>(x, y) {}
    template <typename U>
    PBRT_CPU_GPU explicit Vector2(Point2<U> p);
    template <typename U>
    PBRT_CPU_GPU explicit Vector2(Vector2<U> v)
        : Tuple2<pbrt::Vector2, T>(T(v.x), T(v.y)) {}
};

// Vector3 Definition
template <typename T>
class Vector3 : public Tuple3<Vector3, T> {
  public:
    // Vector3 Public Methods
    using Tuple3<Vector3, T>::x;
    using Tuple3<Vector3, T>::y;
    using Tuple3<Vector3, T>::z;

    Vector3() = default;
    PBRT_CPU_GPU
    Vector3(T x, T y, T z) : Tuple3<pbrt::Vector3, T>(x, y, z) {}

    template <typename U>
    PBRT_CPU_GPU explicit Vector3(Vector3<U> v)
        : Tuple3<pbrt::Vector3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    PBRT_CPU_GPU explicit Vector3(Point3<U> p);
    template <typename U>
    PBRT_CPU_GPU explicit Vector3(Normal3<U> n);
};

// Vector2* Definitions
using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;

// Vector3* Definitions
using Vector3f = Vector3<Float>;
using Vector3i = Vector3<int>;

// Vector3fi Definition
class Vector3fi : public Vector3<Interval> {
  public:
    // Vector3fi Public Methods
    using Vector3<Interval>::x;
    using Vector3<Interval>::y;
    using Vector3<Interval>::z;
    using Vector3<Interval>::HasNaN;
    using Vector3<Interval>::operator+;
    using Vector3<Interval>::operator+=;
    using Vector3<Interval>::operator*;
    using Vector3<Interval>::operator*=;

    Vector3fi() = default;
    PBRT_CPU_GPU
    Vector3fi(Float x, Float y, Float z)
        : Vector3<Interval>(Interval(x), Interval(y), Interval(z)) {}
    PBRT_CPU_GPU
    Vector3fi(Interval x, Interval y, Interval z) : Vector3<Interval>(x, y, z) {}
    PBRT_CPU_GPU
    Vector3fi(Vector3f p)
        : Vector3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}
    template <typename T>
    PBRT_CPU_GPU explicit Vector3fi(Point3<T> p)
        : Vector3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}

    PBRT_CPU_GPU Vector3fi(Vector3<Interval> pfi) : Vector3<Interval>(pfi) {}

    PBRT_CPU_GPU
    Vector3fi(Vector3f v, Vector3f e)
        : Vector3<Interval>(Interval::FromValueAndError(v.x, e.x),
                            Interval::FromValueAndError(v.y, e.y),
                            Interval::FromValueAndError(v.z, e.z)) {}

    PBRT_CPU_GPU
    Vector3f Error() const { return {x.Width() / 2, y.Width() / 2, z.Width() / 2}; }
    PBRT_CPU_GPU
    bool IsExact() const { return x.Width() == 0 && y.Width() == 0 && z.Width() == 0; }
};

// Point2 Definition
template <typename T>
class Point2 : public Tuple2<Point2, T> {
  public:
    // Point2 Public Methods
    using Tuple2<Point2, T>::x;
    using Tuple2<Point2, T>::y;
    using Tuple2<Point2, T>::HasNaN;
    using Tuple2<Point2, T>::operator+;
    using Tuple2<Point2, T>::operator+=;
    using Tuple2<Point2, T>::operator*;
    using Tuple2<Point2, T>::operator*=;

    PBRT_CPU_GPU
    Point2() { x = y = 0; }
    PBRT_CPU_GPU
    Point2(T x, T y) : Tuple2<pbrt::Point2, T>(x, y) {}
    template <typename U>
    PBRT_CPU_GPU explicit Point2(Point2<U> v) : Tuple2<pbrt::Point2, T>(T(v.x), T(v.y)) {}
    template <typename U>
    PBRT_CPU_GPU explicit Point2(Vector2<U> v)
        : Tuple2<pbrt::Point2, T>(T(v.x), T(v.y)) {}

    template <typename U>
    PBRT_CPU_GPU auto operator+(Vector2<U> v) const -> Point2<decltype(T{} + U{})> {
        DCHECK(!v.HasNaN());
        return {x + v.x, y + v.y};
    }
    template <typename U>
    PBRT_CPU_GPU Point2<T> &operator+=(Vector2<U> v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        return *this;
    }

    // We can't do using operator- above, since we don't want to pull in
    // the Point-Point -> Point one so that we can return a vector
    // instead...
    PBRT_CPU_GPU
    Point2<T> operator-() const { return {-x, -y}; }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Point2<U> p) const -> Vector2<decltype(T{} - U{})> {
        DCHECK(!p.HasNaN());
        return {x - p.x, y - p.y};
    }
    template <typename U>
    PBRT_CPU_GPU auto operator-(Vector2<U> v) const -> Point2<decltype(T{} - U{})> {
        DCHECK(!v.HasNaN());
        return {x - v.x, y - v.y};
    }
    template <typename U>
    PBRT_CPU_GPU Point2<T> &operator-=(Vector2<U> v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        return *this;
    }
};

// Point2 Inline Functions
PBRT_CPU_GPU inline Point2f InvertBilinear(Point2f p, pstd::span<const Point2f> v);

// https://www.iquilezles.org/www/articles/ibilinear/ibilinear.htm,
// with a fix for perfect quads
PBRT_CPU_GPU inline Point2f InvertBilinear(Point2f p, pstd::span<const Point2f> vert) {
    // The below assumes a quad (vs uv parametric layout) in v....
    Point2f a = vert[0], b = vert[1], c = vert[3], d = vert[2];
    Vector2f e = b - a, f = d - a, g = (a - b) + (c - d), h = p - a;

    auto cross2d = [](Vector2f a, Vector2f b) {
        return DifferenceOfProducts(a.x, b.y, a.y, b.x);
    };

    Float k2 = cross2d(g, f);
    Float k1 = cross2d(e, f) + cross2d(h, g);
    Float k0 = cross2d(h, e);

    // if edges are parallel, this is a linear equation
    if (std::abs(k2) < 0.001f) {
        if (std::abs(e.x * k1 - g.x * k0) < 1e-5f)
            return Point2f((h.y * k1 + f.y * k0) / (e.y * k1 - g.y * k0), -k0 / k1);
        else
            return Point2f((h.x * k1 + f.x * k0) / (e.x * k1 - g.x * k0), -k0 / k1);
    }

    Float v0, v1;
    if (!Quadratic(k2, k1, k0, &v0, &v1))
        return Point2f(0, 0);

    Float u = (h.x - f.x * v0) / (e.x + g.x * v0);
    if (u < 0 || u > 1 || v0 < 0 || v0 > 1)
        return Point2f((h.x - f.x * v1) / (e.x + g.x * v1), v1);
    return Point2f(u, v0);
}

// Point3 Definition
template <typename T>
class Point3 : public Tuple3<Point3, T> {
  public:
    // Point3 Public Methods
    using Tuple3<Point3, T>::x;
    using Tuple3<Point3, T>::y;
    using Tuple3<Point3, T>::z;
    using Tuple3<Point3, T>::HasNaN;
    using Tuple3<Point3, T>::operator+;
    using Tuple3<Point3, T>::operator+=;
    using Tuple3<Point3, T>::operator*;
    using Tuple3<Point3, T>::operator*=;

    Point3() = default;
    PBRT_CPU_GPU
    Point3(T x, T y, T z) : Tuple3<pbrt::Point3, T>(x, y, z) {}

    // We can't do using operator- above, since we don't want to pull in
    // the Point-Point -> Point one so that we can return a vector
    // instead...
    PBRT_CPU_GPU
    Point3<T> operator-() const { return {-x, -y, -z}; }

    template <typename U>
    PBRT_CPU_GPU explicit Point3(Point3<U> p)
        : Tuple3<pbrt::Point3, T>(T(p.x), T(p.y), T(p.z)) {}
    template <typename U>
    PBRT_CPU_GPU explicit Point3(Vector3<U> v)
        : Tuple3<pbrt::Point3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    PBRT_CPU_GPU auto operator+(Vector3<U> v) const -> Point3<decltype(T{} + U{})> {
        DCHECK(!v.HasNaN());
        return {x + v.x, y + v.y, z + v.z};
    }
    template <typename U>
    PBRT_CPU_GPU Point3<T> &operator+=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Vector3<U> v) const -> Point3<decltype(T{} - U{})> {
        DCHECK(!v.HasNaN());
        return {x - v.x, y - v.y, z - v.z};
    }
    template <typename U>
    PBRT_CPU_GPU Point3<T> &operator-=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Point3<U> p) const -> Vector3<decltype(T{} - U{})> {
        DCHECK(!p.HasNaN());
        return {x - p.x, y - p.y, z - p.z};
    }
};

// Point2* Definitions
using Point2f = Point2<Float>;
using Point2i = Point2<int>;

// Point3* Definitions
using Point3f = Point3<Float>;
using Point3i = Point3<int>;

// Point3fi Definition
class Point3fi : public Point3<Interval> {
  public:
    using Point3<Interval>::x;
    using Point3<Interval>::y;
    using Point3<Interval>::z;
    using Point3<Interval>::HasNaN;
    using Point3<Interval>::operator+;
    using Point3<Interval>::operator*;
    using Point3<Interval>::operator*=;

    Point3fi() = default;
    PBRT_CPU_GPU
    Point3fi(Interval x, Interval y, Interval z) : Point3<Interval>(x, y, z) {}
    PBRT_CPU_GPU
    Point3fi(Float x, Float y, Float z)
        : Point3<Interval>(Interval(x), Interval(y), Interval(z)) {}
    PBRT_CPU_GPU
    Point3fi(const Point3f &p)
        : Point3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}
    PBRT_CPU_GPU
    Point3fi(Point3<Interval> p) : Point3<Interval>(p) {}
    PBRT_CPU_GPU
    Point3fi(Point3f p, Vector3f e)
        : Point3<Interval>(Interval::FromValueAndError(p.x, e.x),
                           Interval::FromValueAndError(p.y, e.y),
                           Interval::FromValueAndError(p.z, e.z)) {}

    PBRT_CPU_GPU
    Vector3f Error() const { return {x.Width() / 2, y.Width() / 2, z.Width() / 2}; }
    PBRT_CPU_GPU
    bool IsExact() const { return x.Width() == 0 && y.Width() == 0 && z.Width() == 0; }

    // Meh--can't seem to get these from Point3 via using declarations...
    template <typename U>
    PBRT_CPU_GPU Point3fi operator+(Vector3<U> v) const {
        DCHECK(!v.HasNaN());
        return {x + v.x, y + v.y, z + v.z};
    }
    template <typename U>
    PBRT_CPU_GPU Point3fi &operator+=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    PBRT_CPU_GPU
    Point3fi operator-() const { return {-x, -y, -z}; }

    template <typename U>
    PBRT_CPU_GPU Point3fi operator-(Point3<U> p) const {
        DCHECK(!p.HasNaN());
        return {x - p.x, y - p.y, z - p.z};
    }
    template <typename U>
    PBRT_CPU_GPU Point3fi operator-(Vector3<U> v) const {
        DCHECK(!v.HasNaN());
        return {x - v.x, y - v.y, z - v.z};
    }
    template <typename U>
    PBRT_CPU_GPU Point3fi &operator-=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
};

// Normal3 Definition
template <typename T>
class Normal3 : public Tuple3<Normal3, T> {
  public:
    // Normal3 Public Methods
    using Tuple3<Normal3, T>::x;
    using Tuple3<Normal3, T>::y;
    using Tuple3<Normal3, T>::z;
    using Tuple3<Normal3, T>::HasNaN;
    using Tuple3<Normal3, T>::operator+;
    using Tuple3<Normal3, T>::operator*;
    using Tuple3<Normal3, T>::operator*=;

    Normal3() = default;
    PBRT_CPU_GPU
    Normal3(T x, T y, T z) : Tuple3<pbrt::Normal3, T>(x, y, z) {}
    template <typename U>
    PBRT_CPU_GPU explicit Normal3<T>(Normal3<U> v)
        : Tuple3<pbrt::Normal3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    PBRT_CPU_GPU explicit Normal3<T>(Vector3<U> v)
        : Tuple3<pbrt::Normal3, T>(T(v.x), T(v.y), T(v.z)) {}
};

using Normal3f = Normal3<Float>;

// Quaternion Definition
class Quaternion {
  public:
    // Quaternion Public Methods
    Quaternion() = default;

    PBRT_CPU_GPU
    Quaternion &operator+=(Quaternion q) {
        v += q.v;
        w += q.w;
        return *this;
    }

    PBRT_CPU_GPU
    Quaternion operator+(Quaternion q) const { return {v + q.v, w + q.w}; }
    PBRT_CPU_GPU
    Quaternion &operator-=(Quaternion q) {
        v -= q.v;
        w -= q.w;
        return *this;
    }
    PBRT_CPU_GPU
    Quaternion operator-() const { return {-v, -w}; }
    PBRT_CPU_GPU
    Quaternion operator-(Quaternion q) const { return {v - q.v, w - q.w}; }
    PBRT_CPU_GPU
    Quaternion &operator*=(Float f) {
        v *= f;
        w *= f;
        return *this;
    }
    PBRT_CPU_GPU
    Quaternion operator*(Float f) const { return {v * f, w * f}; }
    PBRT_CPU_GPU
    Quaternion &operator/=(Float f) {
        DCHECK_NE(0, f);
        v /= f;
        w /= f;
        return *this;
    }
    PBRT_CPU_GPU
    Quaternion operator/(Float f) const {
        DCHECK_NE(0, f);
        return {v / f, w / f};
    }

    std::string ToString() const;

    // Quaternion Public Members
    Vector3f v;
    Float w = 1;
};

// Vector2 Inline Functions
template <typename T>
template <typename U>
Vector2<T>::Vector2(Point2<U> p) : Tuple2<pbrt::Vector2, T>(T(p.x), T(p.y)) {}

template <typename T>
PBRT_CPU_GPU inline auto Dot(Vector2<T> v1, Vector2<T> v2) ->
    typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return SumOfProducts(v1.x, v2.x, v1.y, v2.y);
}

template <typename T>
PBRT_CPU_GPU inline auto AbsDot(Vector2<T> v1, Vector2<T> v2) ->
    typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return std::abs(Dot(v1, v2));
}

template <typename T>
PBRT_CPU_GPU inline auto LengthSquared(Vector2<T> v) -> typename TupleLength<T>::type {
    return Sqr(v.x) + Sqr(v.y);
}

template <typename T>
PBRT_CPU_GPU inline auto Length(Vector2<T> v) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(v));
}

template <typename T>
PBRT_CPU_GPU inline auto Normalize(Vector2<T> v) {
    return v / Length(v);
}

template <typename T>
PBRT_CPU_GPU inline auto Distance(Point2<T> p1, Point2<T> p2) ->
    typename TupleLength<T>::type {
    return Length(p1 - p2);
}

template <typename T>
PBRT_CPU_GPU inline auto DistanceSquared(Point2<T> p1, Point2<T> p2) ->
    typename TupleLength<T>::type {
    return LengthSquared(p1 - p2);
}

// Vector3 Inline Functions
template <typename T>
template <typename U>
Vector3<T>::Vector3(Point3<U> p) : Tuple3<pbrt::Vector3, T>(T(p.x), T(p.y), T(p.z)) {}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> Cross(Vector3<T> v1, Normal3<T> v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return {DifferenceOfProducts(v1.y, v2.z, v1.z, v2.y),
            DifferenceOfProducts(v1.z, v2.x, v1.x, v2.z),
            DifferenceOfProducts(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> Cross(Normal3<T> v1, Vector3<T> v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return {DifferenceOfProducts(v1.y, v2.z, v1.z, v2.y),
            DifferenceOfProducts(v1.z, v2.x, v1.x, v2.z),
            DifferenceOfProducts(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
PBRT_CPU_GPU inline T LengthSquared(Vector3<T> v) {
    return Sqr(v.x) + Sqr(v.y) + Sqr(v.z);
}

template <typename T>
PBRT_CPU_GPU inline auto Length(Vector3<T> v) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(v));
}

template <typename T>
PBRT_CPU_GPU inline auto Normalize(Vector3<T> v) {
    return v / Length(v);
}

template <typename T>
PBRT_CPU_GPU inline T Dot(Vector3<T> v, Vector3<T> w) {
    DCHECK(!v.HasNaN() && !w.HasNaN());
    return v.x * w.x + v.y * w.y + v.z * w.z;
}

// Equivalent to std::acos(Dot(a, b)), but more numerically stable.
// via http://www.plunk.org/~hatch/rightway.html
template <typename T>
PBRT_CPU_GPU inline Float AngleBetween(Vector3<T> v1, Vector3<T> v2) {
    if (Dot(v1, v2) < 0)
        return Pi - 2 * SafeASin(Length(v1 + v2) / 2);
    else
        return 2 * SafeASin(Length(v2 - v1) / 2);
}

template <typename T>
PBRT_CPU_GPU inline T AbsDot(Vector3<T> v1, Vector3<T> v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return std::abs(Dot(v1, v2));
}

template <typename T>
PBRT_CPU_GPU inline Float AngleBetween(Normal3<T> a, Normal3<T> b) {
    if (Dot(a, b) < 0)
        return Pi - 2 * SafeASin(Length(a + b) / 2);
    else
        return 2 * SafeASin(Length(b - a) / 2);
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> GramSchmidt(Vector3<T> v, Vector3<T> w) {
    return v - Dot(v, w) * w;
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> Cross(Vector3<T> v, Vector3<T> w) {
    DCHECK(!v.HasNaN() && !w.HasNaN());
    return {DifferenceOfProducts(v.y, w.z, v.z, w.y),
            DifferenceOfProducts(v.z, w.x, v.x, w.z),
            DifferenceOfProducts(v.x, w.y, v.y, w.x)};
}

template <typename T>
PBRT_CPU_GPU inline void CoordinateSystem(Vector3<T> v1, Vector3<T> *v2, Vector3<T> *v3) {
    Float sign = pstd::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * Sqr(v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + Sqr(v1.y) * a, -v1.y);
}

template <typename T>
PBRT_CPU_GPU inline void CoordinateSystem(Normal3<T> v1, Vector3<T> *v2, Vector3<T> *v3) {
    Float sign = pstd::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * Sqr(v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + Sqr(v1.y) * a, -v1.y);
}

template <typename T>
template <typename U>
Vector3<T>::Vector3(Normal3<U> n) : Tuple3<pbrt::Vector3, T>(T(n.x), T(n.y), T(n.z)) {}

// Point3 Inline Functions
template <typename T>
PBRT_CPU_GPU inline auto Distance(Point3<T> p1, Point3<T> p2) {
    return Length(p1 - p2);
}

template <typename T>
PBRT_CPU_GPU inline auto DistanceSquared(Point3<T> p1, Point3<T> p2) {
    return LengthSquared(p1 - p2);
}

// Normal3 Inline Functions
template <typename T>
PBRT_CPU_GPU inline auto LengthSquared(Normal3<T> n) -> typename TupleLength<T>::type {
    return Sqr(n.x) + Sqr(n.y) + Sqr(n.z);
}

template <typename T>
PBRT_CPU_GPU inline auto Length(Normal3<T> n) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(n));
}

template <typename T>
PBRT_CPU_GPU inline auto Normalize(Normal3<T> n) {
    return n / Length(n);
}

template <typename T>
PBRT_CPU_GPU inline auto Dot(Normal3<T> n, Vector3<T> v) ->
    typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
PBRT_CPU_GPU inline auto Dot(Vector3<T> v, Normal3<T> n) ->
    typename TupleLength<T>::type {
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
PBRT_CPU_GPU inline auto Dot(Normal3<T> n1, Normal3<T> n2) ->
    typename TupleLength<T>::type {
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return FMA(n1.x, n2.x, SumOfProducts(n1.y, n2.y, n1.z, n2.z));
}

template <typename T>
PBRT_CPU_GPU inline auto AbsDot(Normal3<T> n, Vector3<T> v) ->
    typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return std::abs(Dot(n, v));
}

template <typename T>
PBRT_CPU_GPU inline auto AbsDot(Vector3<T> v, Normal3<T> n) ->
    typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return abs(Dot(v, n));
}

template <typename T>
PBRT_CPU_GPU inline auto AbsDot(Normal3<T> n1, Normal3<T> n2) ->
    typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return abs(Dot(n1, n2));
}

template <typename T>
PBRT_CPU_GPU inline Normal3<T> FaceForward(Normal3<T> n, Vector3<T> v) {
    return (Dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
PBRT_CPU_GPU inline Normal3<T> FaceForward(Normal3<T> n, Normal3<T> n2) {
    return (Dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> FaceForward(Vector3<T> v, Vector3<T> v2) {
    return (Dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> FaceForward(Vector3<T> v, Normal3<T> n2) {
    return (Dot(v, n2) < 0.f) ? -v : v;
}

// Quaternion Inline Functions
PBRT_CPU_GPU
inline Quaternion operator*(Float f, Quaternion q) {
    return q * f;
}

PBRT_CPU_GPU inline Float Dot(Quaternion q1, Quaternion q2) {
    return Dot(q1.v, q2.v) + q1.w * q2.w;
}

PBRT_CPU_GPU inline Float Length(Quaternion q) {
    return std::sqrt(Dot(q, q));
}
PBRT_CPU_GPU inline Quaternion Normalize(Quaternion q) {
    DCHECK_GT(Length(q), 0);
    return q / Length(q);
}

PBRT_CPU_GPU inline Float AngleBetween(Quaternion q1, Quaternion q2) {
    if (Dot(q1, q2) < 0)
        return Pi - 2 * SafeASin(Length(q1 + q2) / 2);
    else
        return 2 * SafeASin(Length(q2 - q1) / 2);
}

// http://www.plunk.org/~hatch/rightway.html
PBRT_CPU_GPU inline Quaternion Slerp(Float t, Quaternion q1, Quaternion q2) {
    Float theta = AngleBetween(q1, q2);
    Float sinThetaOverTheta = SinXOverX(theta);
    return q1 * (1 - t) * SinXOverX((1 - t) * theta) / sinThetaOverTheta +
           q2 * t * SinXOverX(t * theta) / sinThetaOverTheta;
}

// Bounds2 Definition
template <typename T>
class Bounds2 {
  public:
    // Bounds2 Public Methods
    PBRT_CPU_GPU
    Bounds2() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Point2<T>(maxNum, maxNum);
        pMax = Point2<T>(minNum, minNum);
    }
    PBRT_CPU_GPU
    explicit Bounds2(Point2<T> p) : pMin(p), pMax(p) {}
    PBRT_CPU_GPU
    Bounds2(Point2<T> p1, Point2<T> p2) : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}
    template <typename U>
    PBRT_CPU_GPU explicit Bounds2(const Bounds2<U> &b) {
        if (b.IsEmpty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds2<T>();
        else {
            pMin = Point2<T>(b.pMin);
            pMax = Point2<T>(b.pMax);
        }
    }

    PBRT_CPU_GPU
    Vector2<T> Diagonal() const { return pMax - pMin; }

    PBRT_CPU_GPU
    T Area() const {
        Vector2<T> d = pMax - pMin;
        return d.x * d.y;
    }

    PBRT_CPU_GPU
    bool IsEmpty() const { return pMin.x >= pMax.x || pMin.y >= pMax.y; }

    PBRT_CPU_GPU
    bool IsDegenerate() const { return pMin.x > pMax.x || pMin.y > pMax.y; }

    PBRT_CPU_GPU
    int MaxDimension() const {
        Vector2<T> diag = Diagonal();
        if (diag.x > diag.y)
            return 0;
        else
            return 1;
    }
    PBRT_CPU_GPU
    Point2<T> operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    PBRT_CPU_GPU
    Point2<T> &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    PBRT_CPU_GPU
    bool operator==(const Bounds2<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    PBRT_CPU_GPU
    bool operator!=(const Bounds2<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    PBRT_CPU_GPU
    Point2<T> Corner(int corner) const {
        DCHECK(corner >= 0 && corner < 4);
        return Point2<T>((*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y);
    }
    PBRT_CPU_GPU
    Point2<T> Lerp(Point2f t) const {
        return Point2<T>(pbrt::Lerp(t.x, pMin.x, pMax.x),
                         pbrt::Lerp(t.y, pMin.y, pMax.y));
    }
    PBRT_CPU_GPU
    Vector2<T> Offset(Point2<T> p) const {
        Vector2<T> o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        return o;
    }
    PBRT_CPU_GPU
    void BoundingSphere(Point2<T> *c, Float *rad) const {
        *c = (pMin + pMax) / 2;
        *rad = Inside(*c, *this) ? Distance(*c, pMax) : 0;
    }

    std::string ToString() const { return StringPrintf("[ %s - %s ]", pMin, pMax); }

    // Bounds2 Public Members
    Point2<T> pMin, pMax;
};

// Bounds3 Definition
template <typename T>
class Bounds3 {
  public:
    // Bounds3 Public Methods
    PBRT_CPU_GPU
    Bounds3() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Point3<T>(maxNum, maxNum, maxNum);
        pMax = Point3<T>(minNum, minNum, minNum);
    }

    PBRT_CPU_GPU
    explicit Bounds3(Point3<T> p) : pMin(p), pMax(p) {}

    PBRT_CPU_GPU
    Bounds3(Point3<T> p1, Point3<T> p2) : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}

    PBRT_CPU_GPU
    Point3<T> operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    PBRT_CPU_GPU
    Point3<T> &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }

    PBRT_CPU_GPU
    Point3<T> Corner(int corner) const {
        DCHECK(corner >= 0 && corner < 8);
        return Point3<T>((*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }

    PBRT_CPU_GPU
    Vector3<T> Diagonal() const { return pMax - pMin; }

    PBRT_CPU_GPU
    T SurfaceArea() const {
        Vector3<T> d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    PBRT_CPU_GPU
    T Volume() const {
        Vector3<T> d = Diagonal();
        return d.x * d.y * d.z;
    }

    PBRT_CPU_GPU
    int MaxDimension() const {
        Vector3<T> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    PBRT_CPU_GPU
    Point3f Lerp(Point3f t) const {
        return Point3f(pbrt::Lerp(t.x, pMin.x, pMax.x), pbrt::Lerp(t.y, pMin.y, pMax.y),
                       pbrt::Lerp(t.z, pMin.z, pMax.z));
    }

    PBRT_CPU_GPU
    Vector3f Offset(Point3f p) const {
        Vector3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    PBRT_CPU_GPU
    void BoundingSphere(Point3<T> *center, Float *radius) const {
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }

    PBRT_CPU_GPU
    bool IsEmpty() const {
        return pMin.x >= pMax.x || pMin.y >= pMax.y || pMin.z >= pMax.z;
    }
    PBRT_CPU_GPU
    bool IsDegenerate() const {
        return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z;
    }

    template <typename U>
    PBRT_CPU_GPU explicit Bounds3(const Bounds3<U> &b) {
        if (b.IsEmpty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds3<T>();
        else {
            pMin = Point3<T>(b.pMin);
            pMax = Point3<T>(b.pMax);
        }
    }
    PBRT_CPU_GPU
    bool operator==(const Bounds3<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    PBRT_CPU_GPU
    bool operator!=(const Bounds3<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    PBRT_CPU_GPU
    bool IntersectP(Point3f o, Vector3f d, Float tMax = Infinity, Float *hitt0 = nullptr,
                    Float *hitt1 = nullptr) const;
    PBRT_CPU_GPU
    bool IntersectP(Point3f o, Vector3f d, Float tMax, Vector3f invDir,
                    const int dirIsNeg[3]) const;

    std::string ToString() const { return StringPrintf("[ %s - %s ]", pMin, pMax); }

    // Bounds3 Public Members
    Point3<T> pMin, pMax;
};

// Bounds[23][fi] Definitions
using Bounds2f = Bounds2<Float>;
using Bounds2i = Bounds2<int>;
using Bounds3f = Bounds3<Float>;
using Bounds3i = Bounds3<int>;

class Bounds2iIterator : public std::forward_iterator_tag {
  public:
    PBRT_CPU_GPU
    Bounds2iIterator(const Bounds2i &b, const Point2i &pt) : p(pt), bounds(&b) {}
    PBRT_CPU_GPU
    Bounds2iIterator operator++() {
        advance();
        return *this;
    }
    PBRT_CPU_GPU
    Bounds2iIterator operator++(int) {
        Bounds2iIterator old = *this;
        advance();
        return old;
    }
    PBRT_CPU_GPU
    bool operator==(const Bounds2iIterator &bi) const {
        return p == bi.p && bounds == bi.bounds;
    }
    PBRT_CPU_GPU
    bool operator!=(const Bounds2iIterator &bi) const {
        return p != bi.p || bounds != bi.bounds;
    }

    PBRT_CPU_GPU
    Point2i operator*() const { return p; }

  private:
    PBRT_CPU_GPU
    void advance() {
        ++p.x;
        if (p.x == bounds->pMax.x) {
            p.x = bounds->pMin.x;
            ++p.y;
        }
    }
    Point2i p;
    const Bounds2i *bounds;
};

// Bounds2 Inline Functions
template <typename T>
PBRT_CPU_GPU inline Bounds2<T> Union(const Bounds2<T> &b1, const Bounds2<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
PBRT_CPU_GPU inline Bounds2<T> Intersect(const Bounds2<T> &b1, const Bounds2<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> b;
    b.pMin = Max(b1.pMin, b2.pMin);
    b.pMax = Min(b1.pMax, b2.pMax);
    return b;
}

template <typename T>
PBRT_CPU_GPU inline bool Overlaps(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    bool x = (ba.pMax.x >= bb.pMin.x) && (ba.pMin.x <= bb.pMax.x);
    bool y = (ba.pMax.y >= bb.pMin.y) && (ba.pMin.y <= bb.pMax.y);
    return (x && y);
}

template <typename T>
PBRT_CPU_GPU inline bool Inside(Point2<T> pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x <= b.pMax.x && pt.y >= b.pMin.y && pt.y <= b.pMax.y);
}

template <typename T>
PBRT_CPU_GPU inline bool Inside(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    return (ba.pMin.x >= bb.pMin.x && ba.pMax.x <= bb.pMax.x && ba.pMin.y >= bb.pMin.y &&
            ba.pMax.y <= bb.pMax.y);
}

template <typename T>
PBRT_CPU_GPU inline bool InsideExclusive(Point2<T> pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x < b.pMax.x && pt.y >= b.pMin.y && pt.y < b.pMax.y);
}

template <typename T, typename U>
PBRT_CPU_GPU inline Bounds2<T> Expand(const Bounds2<T> &b, U delta) {
    Bounds2<T> ret;
    ret.pMin = b.pMin - Vector2<T>(delta, delta);
    ret.pMax = b.pMax + Vector2<T>(delta, delta);
    return ret;
}

// Bounds3 Inline Functions
template <typename T>
PBRT_CPU_GPU inline Bounds3<T> Union(const Bounds3<T> &b, Point3<T> p) {
    Bounds3<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T>
PBRT_CPU_GPU inline Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    Bounds3<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
PBRT_CPU_GPU inline Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    Bounds3<T> b;
    b.pMin = Max(b1.pMin, b2.pMin);
    b.pMax = Min(b1.pMax, b2.pMax);
    return b;
}

template <typename T>
PBRT_CPU_GPU inline bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

template <typename T>
PBRT_CPU_GPU inline bool Inside(Point3<T> p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y && p.y <= b.pMax.y &&
            p.z >= b.pMin.z && p.z <= b.pMax.z);
}

template <typename T>
PBRT_CPU_GPU inline bool InsideExclusive(Point3<T> p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y && p.y < b.pMax.y &&
            p.z >= b.pMin.z && p.z < b.pMax.z);
}

template <typename T, typename U>
PBRT_CPU_GPU inline auto DistanceSquared(Point3<T> p, const Bounds3<U> &b) {
    using TDist = decltype(T{} - U{});
    TDist dx = std::max<TDist>({0, b.pMin.x - p.x, p.x - b.pMax.x});
    TDist dy = std::max<TDist>({0, b.pMin.y - p.y, p.y - b.pMax.y});
    TDist dz = std::max<TDist>({0, b.pMin.z - p.z, p.z - b.pMax.z});
    return Sqr(dx) + Sqr(dy) + Sqr(dz);
}

template <typename T, typename U>
PBRT_CPU_GPU inline auto Distance(Point3<T> p, const Bounds3<U> &b) {
    auto dist2 = DistanceSquared(p, b);
    using TDist = typename TupleLength<decltype(dist2)>::type;
    return std::sqrt(TDist(dist2));
}

template <typename T, typename U>
PBRT_CPU_GPU inline Bounds3<T> Expand(const Bounds3<T> &b, U delta) {
    Bounds3<T> ret;
    ret.pMin = b.pMin - Vector3<T>(delta, delta, delta);
    ret.pMax = b.pMax + Vector3<T>(delta, delta, delta);
    return ret;
}

template <typename T>
PBRT_CPU_GPU inline bool Bounds3<T>::IntersectP(Point3f o, Vector3f d, Float tMax,
                                                Float *hitt0, Float *hitt1) const {
    Float t0 = 0, t1 = tMax;
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        Float invRayDir = 1 / d[i];
        Float tNear = (pMin[i] - o[i]) * invRayDir;
        Float tFar = (pMax[i] - o[i]) * invRayDir;
        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar)
            pstd::swap(tNear, tFar);
        // Update _tFar_ to ensure robust ray--bounds intersection
        tFar *= 1 + 2 * gamma(3);

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1)
            return false;
    }
    if (hitt0)
        *hitt0 = t0;
    if (hitt1)
        *hitt1 = t1;
    return true;
}

template <typename T>
PBRT_CPU_GPU inline bool Bounds3<T>::IntersectP(Point3f o, Vector3f d, Float raytMax,
                                                Vector3f invDir,
                                                const int dirIsNeg[3]) const {
    const Bounds3f &bounds = *this;
    // Check for ray intersection against $x$ and $y$ slabs
    Float tMin = (bounds[dirIsNeg[0]].x - o.x) * invDir.x;
    Float tMax = (bounds[1 - dirIsNeg[0]].x - o.x) * invDir.x;
    Float tyMin = (bounds[dirIsNeg[1]].y - o.y) * invDir.y;
    Float tyMax = (bounds[1 - dirIsNeg[1]].y - o.y) * invDir.y;
    // Update _tMax_ and _tyMax_ to ensure robust bounds intersection
    tMax *= 1 + 2 * gamma(3);
    tyMax *= 1 + 2 * gamma(3);

    if (tMin > tyMax || tyMin > tMax)
        return false;
    if (tyMin > tMin)
        tMin = tyMin;
    if (tyMax < tMax)
        tMax = tyMax;

    // Check for ray intersection against $z$ slab
    Float tzMin = (bounds[dirIsNeg[2]].z - o.z) * invDir.z;
    Float tzMax = (bounds[1 - dirIsNeg[2]].z - o.z) * invDir.z;
    // Update _tzMax_ to ensure robust bounds intersection
    tzMax *= 1 + 2 * gamma(3);

    if (tMin > tzMax || tzMin > tMax)
        return false;
    if (tzMin > tMin)
        tMin = tzMin;
    if (tzMax < tMax)
        tMax = tzMax;

    return (tMin < raytMax) && (tMax > 0);
}

PBRT_CPU_GPU
inline Bounds2iIterator begin(const Bounds2i &b) {
    return Bounds2iIterator(b, b.pMin);
}

PBRT_CPU_GPU
inline Bounds2iIterator end(const Bounds2i &b) {
    // Normally, the ending point is at the minimum x value and one past
    // the last valid y value.
    Point2i pEnd(b.pMin.x, b.pMax.y);
    // However, if the bounds are degenerate, override the end point to
    // equal the start point so that any attempt to iterate over the bounds
    // exits out immediately.
    if (b.pMin.x >= b.pMax.x || b.pMin.y >= b.pMax.y)
        pEnd = b.pMin;
    return Bounds2iIterator(b, pEnd);
}

template <typename T>
PBRT_CPU_GPU inline Bounds2<T> Union(const Bounds2<T> &b, Point2<T> p) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

// Spherical Geometry Inline Functions
PBRT_CPU_GPU inline Float SphericalTriangleArea(Vector3f a, Vector3f b, Vector3f c) {
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

    return std::abs(alpha + beta + gamma - Pi);
}

PBRT_CPU_GPU inline Float SphericalQuadArea(Vector3f a, Vector3f b, Vector3f c,
                                            Vector3f d);

PBRT_CPU_GPU inline Float SphericalQuadArea(Vector3f a, Vector3f b, Vector3f c,
                                            Vector3f d) {
    Vector3f axb = Cross(a, b), bxc = Cross(b, c);
    Vector3f cxd = Cross(c, d), dxa = Cross(d, a);
    if (LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 || LengthSquared(cxd) == 0 ||
        LengthSquared(dxa) == 0)
        return 0;
    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxd = Normalize(cxd);
    dxa = Normalize(dxa);

    Float alpha = AngleBetween(dxa, -axb);
    Float beta = AngleBetween(axb, -bxc);
    Float gamma = AngleBetween(bxc, -cxd);
    Float delta = AngleBetween(cxd, -dxa);

    return std::abs(alpha + beta + gamma + delta - 2 * Pi);
}

PBRT_CPU_GPU inline Vector3f SphericalDirection(Float sinTheta, Float cosTheta,
                                                Float phi) {
    DCHECK(sinTheta >= -1.0001 && sinTheta <= 1.0001);
    DCHECK(cosTheta >= -1.0001 && cosTheta <= 1.0001);
    return Vector3f(Clamp(sinTheta, -1, 1) * std::cos(phi),
                    Clamp(sinTheta, -1, 1) * std::sin(phi), Clamp(cosTheta, -1, 1));
}

PBRT_CPU_GPU inline Float SphericalTheta(Vector3f v) {
    return SafeACos(v.z);
}

PBRT_CPU_GPU inline Float SphericalPhi(Vector3f v) {
    Float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Pi) : p;
}

PBRT_CPU_GPU inline Float CosTheta(Vector3f w) {
    return w.z;
}
PBRT_CPU_GPU inline Float Cos2Theta(Vector3f w) {
    return Sqr(w.z);
}
PBRT_CPU_GPU inline Float AbsCosTheta(Vector3f w) {
    return std::abs(w.z);
}

PBRT_CPU_GPU inline Float Sin2Theta(Vector3f w) {
    return std::max<Float>(0, 1 - Cos2Theta(w));
}
PBRT_CPU_GPU inline Float SinTheta(Vector3f w) {
    return std::sqrt(Sin2Theta(w));
}

PBRT_CPU_GPU inline Float TanTheta(Vector3f w) {
    return SinTheta(w) / CosTheta(w);
}
PBRT_CPU_GPU inline Float Tan2Theta(Vector3f w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

PBRT_CPU_GPU inline Float CosPhi(Vector3f w) {
    Float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
}
PBRT_CPU_GPU inline Float SinPhi(Vector3f w) {
    Float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
}

PBRT_CPU_GPU inline Float CosDPhi(Vector3f wa, Vector3f wb) {
    Float waxy = Sqr(wa.x) + Sqr(wa.y), wbxy = Sqr(wb.x) + Sqr(wb.y);
    if (waxy == 0 || wbxy == 0)
        return 1;
    return Clamp((wa.x * wb.x + wa.y * wb.y) / std::sqrt(waxy * wbxy), -1, 1);
}

PBRT_CPU_GPU inline bool SameHemisphere(Vector3f w, Vector3f wp) {
    return w.z * wp.z > 0;
}

PBRT_CPU_GPU
inline bool SameHemisphere(Vector3f w, Normal3f wp) {
    return w.z * wp.z > 0;
}

// OctahedralVector Definition
class OctahedralVector {
  public:
    // OctahedralVector Public Methods
    OctahedralVector() = default;
    PBRT_CPU_GPU
    OctahedralVector(Vector3f v) {
        v /= std::abs(v.x) + std::abs(v.y) + std::abs(v.z);
        if (v.z >= 0) {
            x = Encode(v.x);
            y = Encode(v.y);
        } else {
            // Encode octahedral vector with $z < 0$
            x = Encode((1 - std::abs(v.y)) * Sign(v.x));
            y = Encode((1 - std::abs(v.x)) * Sign(v.y));
        }
    }

    PBRT_CPU_GPU
    explicit operator Vector3f() const {
        Vector3f v;
        v.x = -1 + 2 * (x / 65535.f);
        v.y = -1 + 2 * (y / 65535.f);
        v.z = 1 - (std::abs(v.x) + std::abs(v.y));
        // Reparameterize directions in the $z<0$ portion of the octahedron
        if (v.z < 0) {
            Float xo = v.x;
            v.x = (1 - std::abs(v.y)) * Sign(xo);
            v.y = (1 - std::abs(xo)) * Sign(v.y);
        }

        return Normalize(v);
    }

    std::string ToString() const {
        return StringPrintf("[ OctahedralVector x: %d y: %d ]", x, y);
    }

  private:
    // OctahedralVector Private Methods
    PBRT_CPU_GPU
    static Float Sign(Float v) { return std::copysign(1.f, v); }

    PBRT_CPU_GPU
    static uint16_t Encode(Float f) {
        return pstd::round(Clamp((f + 1) / 2, 0, 1) * 65535.f);
    }

    // OctahedralVector Private Members
    uint16_t x, y;
};

// DirectionCone Definition
class DirectionCone {
  public:
    // DirectionCone Public Methods
    DirectionCone() = default;
    PBRT_CPU_GPU
    DirectionCone(Vector3f w, Float cosTheta)
        : w(Normalize(w)), cosTheta(cosTheta), empty(false) {}
    PBRT_CPU_GPU
    explicit DirectionCone(Vector3f w) : DirectionCone(w, 1) {}

    PBRT_CPU_GPU
    static DirectionCone EntireSphere() { return DirectionCone(Vector3f(0, 0, 1), -1); }

    std::string ToString() const;

    PBRT_CPU_GPU
    Vector3f ClosestVectorInCone(Vector3f wp) const;

    // DirectionCone Public Members
    Vector3f w;
    Float cosTheta;
    bool empty = true;
};

// DirectionCone Inline Functions
PBRT_CPU_GPU inline bool Inside(const DirectionCone &d, Vector3f w) {
    return !d.empty && Dot(d.w, Normalize(w)) >= d.cosTheta;
}

PBRT_CPU_GPU inline DirectionCone BoundSubtendedDirections(const Bounds3f &b, Point3f p) {
    // Compute bounding sphere for _b_ and check if _p_ is inside
    Float radius;
    Point3f pCenter;
    b.BoundingSphere(&pCenter, &radius);
    if (DistanceSquared(p, pCenter) < radius * radius)
        return DirectionCone::EntireSphere();

    // Compute and return _DirectionCone_ for bounding sphere
    Vector3f w = Normalize(pCenter - p);
    Float sin2ThetaMax = Sqr(radius) / DistanceSquared(pCenter, p);
    Float cosThetaMax = SafeSqrt(1 - sin2ThetaMax);
    return DirectionCone(w, cosThetaMax);
}

PBRT_CPU_GPU
inline Vector3f DirectionCone::ClosestVectorInCone(Vector3f wp) const {
    DCHECK(!empty);
    wp = Normalize(wp);
    // Return provided vector if it is inside the cone
    if (Dot(wp, w) > cosTheta)
        return wp;

    // Find closest vector by rotating _wp_ until it touches the cone
    Float sinTheta = -SafeSqrt(1 - cosTheta * cosTheta);
    Vector3f a = Cross(wp, w);
    return cosTheta * w +
           (sinTheta / Length(a)) *
               Vector3f(w.x * (wp.y * w.y + wp.z * w.z) - wp.x * (Sqr(w.y) + Sqr(w.z)),
                        w.y * (wp.x * w.x + wp.z * w.z) - wp.y * (Sqr(w.x) + Sqr(w.z)),
                        w.z * (wp.x * w.x + wp.y * w.y) - wp.z * (Sqr(w.x) + Sqr(w.y)));
}

// DirectionCone Function Declarations
PBRT_CPU_GPU
DirectionCone Union(const DirectionCone &a, const DirectionCone &b);

// Frame Definition
class Frame {
  public:
    // Frame Public Methods
    PBRT_CPU_GPU
    Frame() : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}
    PBRT_CPU_GPU
    Frame(Vector3f x, Vector3f y, Vector3f z);

    PBRT_CPU_GPU
    static Frame FromXZ(Vector3f x, Vector3f z) { return Frame(x, Cross(z, x), z); }
    PBRT_CPU_GPU
    static Frame FromXY(Vector3f x, Vector3f y) { return Frame(x, y, Cross(x, y)); }

    PBRT_CPU_GPU
    static Frame FromZ(Vector3f z) {
        Vector3f x, y;
        CoordinateSystem(z, &x, &y);
        return Frame(x, y, z);
    }

    PBRT_CPU_GPU
    static Frame FromX(Vector3f x) {
        Vector3f y, z;
        CoordinateSystem(x, &y, &z);
        return Frame(x, y, z);
    }

    PBRT_CPU_GPU
    static Frame FromY(Vector3f y) {
        Vector3f x, z;
        CoordinateSystem(y, &z, &x);
        return Frame(x, y, z);
    }

    PBRT_CPU_GPU
    static Frame FromX(Normal3f x) {
        Vector3f y, z;
        CoordinateSystem(x, &y, &z);
        return Frame(Vector3f(x), y, z);
    }

    PBRT_CPU_GPU
    static Frame FromY(Normal3f y) {
        Vector3f x, z;
        CoordinateSystem(y, &z, &x);
        return Frame(x, Vector3f(y), z);
    }

    PBRT_CPU_GPU
    static Frame FromZ(Normal3f z) { return FromZ(Vector3f(z)); }

    PBRT_CPU_GPU
    Vector3f ToLocal(Vector3f v) const {
        return Vector3f(Dot(v, x), Dot(v, y), Dot(v, z));
    }

    PBRT_CPU_GPU
    Normal3f ToLocal(Normal3f n) const {
        return Normal3f(Dot(n, x), Dot(n, y), Dot(n, z));
    }

    PBRT_CPU_GPU
    Vector3f FromLocal(Vector3f v) const { return v.x * x + v.y * y + v.z * z; }

    PBRT_CPU_GPU
    Normal3f FromLocal(Normal3f n) const { return Normal3f(n.x * x + n.y * y + n.z * z); }

    std::string ToString() const {
        return StringPrintf("[ Frame x: %s y: %s z: %s ]", x, y, z);
    }

    // Frame Public Members
    Vector3f x, y, z;
};

// Frame Inline Functions
inline Frame::Frame(Vector3f x, Vector3f y, Vector3f z) : x(x), y(y), z(z) {
    DCHECK_LT(std::abs(LengthSquared(x) - 1), 1e-4);
    DCHECK_LT(std::abs(LengthSquared(y) - 1), 1e-4);
    DCHECK_LT(std::abs(LengthSquared(z) - 1), 1e-4);
    DCHECK_LT(std::abs(Dot(x, y)), 1e-4);
    DCHECK_LT(std::abs(Dot(y, z)), 1e-4);
    DCHECK_LT(std::abs(Dot(z, x)), 1e-4);
}

}  // namespace pbrt

#endif  // PBRT_UTIL_VECMATH_H
