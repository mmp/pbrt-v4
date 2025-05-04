#include <cmath>

struct Vector3f {
    float x, y, z;

    Vector3f(float x = 0, float y = 0, float z = 0)
        : x(x), y(y), z(z) {}

    float distanceSquared(const Vector3f& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    float distance(const Vector3f& other) const {
        return sqrt(distanceSquared(other));
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

    Vector3f abs() const {
        return Vector3f(fabs(x), fabs(y), fabs(z));
    }

    Vector3f operator/(float scalar) const {
        return Vector3f(x / scalar, y / scalar, z / scalar);
    }

    Vector3f operator*(float scalar) const {
        return Vector3f(x * scalar, y * scalar, z * scalar);
    }
};

inline Vector3f operator*(float scalar, const Vector3f& v) {
    return v * scalar;
}
