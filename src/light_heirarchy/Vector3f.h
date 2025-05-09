#ifndef VECTOR3F_H
#define VECTOR3F_H

#include <cmath>

struct Vector3f {
    float x, y, z;

    Vector3f() : x(0), y(0), z(0) {}
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}

    float distanceSquared(const Vector3f& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    float distance(const Vector3f& other) const {
        return std::sqrt(distanceSquared(other));
    }

    float operator[](int index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        return z;
    }

    Vector3f operator+(const Vector3f& other) const {
        return Vector3f(x + other.x, y + other.y, z + other.z);
    }

    Vector3f operator-(const Vector3f& other) const {
        return Vector3f(x - other.x, y - other.y, z - other.z);
    }

    Vector3f& operator+=(const Vector3f& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vector3f& operator-=(const Vector3f& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vector3f& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vector3f& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    Vector3f abs() const {
        return Vector3f(std::abs(x), std::abs(y), std::abs(z));
    }

    Vector3f operator*(float scalar) const {
        return Vector3f(x * scalar, y * scalar, z * scalar);
    }

    Vector3f operator/(float scalar) const {
        return Vector3f(x / scalar, y / scalar, z / scalar);
    }
};
inline Vector3f operator*(float scalar, const Vector3f& vec) {
        return vec * scalar;
    }

#endif // VECTOR3F_H