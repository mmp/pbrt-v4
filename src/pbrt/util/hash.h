// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_HASH_H
#define PBRT_UTIL_HASH_H

#include <pbrt/pbrt.h>

namespace pbrt {

// https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
PBRT_CPU_GPU
inline uint64_t MurmurHash64A(const void *key, int len, uint64_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995ull;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t *data = (const uint64_t *)key;
    const uint64_t *end = data + (len / 8);

    while (data != end) {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const unsigned char *data2 = (const unsigned char *)data;

    switch (len & 7) {
    case 7:
        h ^= uint64_t(data2[6]) << 48;
    case 6:
        h ^= uint64_t(data2[5]) << 40;
    case 5:
        h ^= uint64_t(data2[4]) << 32;
    case 4:
        h ^= uint64_t(data2[3]) << 24;
    case 3:
        h ^= uint64_t(data2[2]) << 16;
    case 2:
        h ^= uint64_t(data2[1]) << 8;
    case 1:
        h ^= uint64_t(data2[0]);
        h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

template <typename... Args>
PBRT_CPU_GPU inline uint64_t hashInternal(uint64_t hash, Args...);

template <>
PBRT_CPU_GPU inline uint64_t hashInternal(uint64_t hash) {
    return hash;
}

template <typename T, typename... Args>
PBRT_CPU_GPU inline uint64_t hashInternal(uint64_t hash, T v, Args... args) {
    return MurmurHash64A(&v, sizeof(v), hashInternal(hash, args...));
}

// Hashing Inline Functions
PBRT_CPU_GPU inline uint64_t HashBuffer(const void *ptr, size_t size, uint64_t seed = 0) {
    return MurmurHash64A(ptr, size, seed);
}

template <size_t size>
PBRT_CPU_GPU inline uint64_t HashBuffer(const void *ptr, uint64_t seed = 0) {
    return MurmurHash64A(ptr, size, seed);
}

template <typename... Args>
PBRT_CPU_GPU inline uint64_t Hash(Args... args);

template <typename... Args>
PBRT_CPU_GPU inline uint64_t Hash(Args... args) {
    return hashInternal(0, args...);
}

}  // namespace pbrt

#endif  // PBRT_UTIL_HASH_H
