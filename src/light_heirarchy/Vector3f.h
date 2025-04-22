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

    float operator[](int index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        return z;
    }
};
