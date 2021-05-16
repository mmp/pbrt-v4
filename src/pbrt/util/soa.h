// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SOA_H
#define PBRT_UTIL_SOA_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bssrdf.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/ray.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

struct alignas(16) Float4 {
    Float v[4];
};

PBRT_CPU_GPU
inline Float4 Load4(const Float4 *p) {
#if defined(PBRT_IS_GPU_CODE) && !defined(PBRT_FLOAT_AS_DOUBLE)
    float4 v = *(const float4 *)p;
    return {{v.x, v.y, v.z, v.w}};
#else
    return *p;
#endif
}

PBRT_CPU_GPU
inline void Store4(Float4 *p, Float4 v) {
#if defined(PBRT_IS_GPU_CODE) && !defined(PBRT_FLOAT_AS_DOUBLE)
    *(float4 *)p = make_float4(v.v[0], v.v[1], v.v[2], v.v[3]);
#else
    *p = v;
#endif
}

template <>
struct SOA<SampledSpectrum> {
    SOA() = default;
    SOA(int size, Allocator alloc) {
        if constexpr ((NSpectrumSamples % 4) == 0) {
            nAlloc = n4 * size;
            ptr4 = alloc.allocate_object<Float4>(nAlloc);
        } else {
            nAlloc = size * NSpectrumSamples;
            ptr1 = alloc.allocate_object<Float>(nAlloc);
        }
    }
    SOA &operator=(const SOA& s) {
        nAlloc = s.nAlloc;
        ptr4 = s.ptr4;
        ptr1 = s.ptr1;
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator[](int i) const {
        SampledSpectrum s;
        if constexpr ((NSpectrumSamples % 4) == 0) {
            int offset = n4 * i;
            DCHECK_LT(offset, nAlloc);
            for (int i = 0; i < n4; ++i, ++offset) {
                Float4 v4 = Load4(ptr4 + offset);
                for (int j = 0; j < 4; ++j)
                    s[4 * i + j] = v4.v[j];
            }
        } else {
            int offset = i * NSpectrumSamples;
            DCHECK_LT(offset, nAlloc);
            for (int i = 0; i < NSpectrumSamples; ++i)
                s[i] = ptr1[offset + i];
        }
        return s;
    }

    struct GetSetIndirector {
        PBRT_CPU_GPU
        operator SampledSpectrum() const {
            return soa->Load(index);  // (*(const SOA<SampledSpectrum> *)soa)[index];
        }
        PBRT_CPU_GPU
        void operator=(const SampledSpectrum &s) {
            if constexpr ((NSpectrumSamples % 4) == 0) {
                int offset = n4 * index;
                DCHECK_LT(offset, soa->nAlloc);
                for (int i = 0; i < n4; ++i, ++offset)
                    Store4(soa->ptr4 + offset,
                           {s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]});
            } else {
                int offset = index * NSpectrumSamples;
                DCHECK_LT(offset, soa->nAlloc);
                for (int i = 0; i < NSpectrumSamples; ++i)
                    soa->ptr1[offset + i] = s[i];
            }
        }

        static constexpr int n4 = (NSpectrumSamples + 3) / 4;
        SOA<SampledSpectrum> *soa;
        int index;
    };

    PBRT_CPU_GPU
    GetSetIndirector operator[](int i) { return GetSetIndirector{this, i}; }

    // TODO: get rid of these
    PBRT_CPU_GPU
    SampledSpectrum Load(int i) const { return (*this)[i]; }
    PBRT_CPU_GPU
    void Store(int i, const SampledSpectrum &s) { (*this)[i] = s; }

  private:
    // number of float4s needed per SampledSpectrum
    static constexpr int n4 = (NSpectrumSamples + 3) / 4;

    int nAlloc;
    Float4 * PBRT_RESTRICT ptr4 = nullptr;
    Float * PBRT_RESTRICT ptr1 = nullptr;
};

template <>
struct SOA<SampledWavelengths> {
    SOA() = default;
    SOA(int size, Allocator alloc) {
        if constexpr ((NSpectrumSamples % 4) == 0) {
            nAlloc = n4 * size;
            lambda4 = alloc.allocate_object<Float4>(nAlloc);
            pdf4 = alloc.allocate_object<Float4>(nAlloc);
        } else {
            nAlloc = size * NSpectrumSamples;
            lambda1 = alloc.allocate_object<Float>(nAlloc);
            pdf1 = alloc.allocate_object<Float>(nAlloc);
        }
    }
    SOA &operator=(const SOA& s) {
        nAlloc = s.nAlloc;
        lambda4 = s.lambda4;
        pdf4 = s.pdf4;
        lambda1 = s.lambda1;
        pdf1 = s.pdf1;
        return *this;
    }

    PBRT_CPU_GPU
    SampledWavelengths operator[](int i) const {
        SampledWavelengths l;
        if constexpr ((NSpectrumSamples % 4) == 0) {
            int offset = n4 * i;
            for (int i = 0; i < n4; ++i, ++offset) {
                DCHECK_LT(offset, nAlloc);
                Float4 l4 = Load4(lambda4 + offset);
                Float4 p4 = Load4(pdf4 + offset);
                for (int j = 0; j < 4; ++j) {
                    l.lambda[4 * i + j] = l4.v[j];
                    l.pdf[4 * i + j] = p4.v[j];
                }
            }
        } else {
            int offset = NSpectrumSamples * i;
            for (int i = 0; i < NSpectrumSamples; ++i)
                l.lambda[i] = lambda1[offset + i];
            for (int i = 0; i < NSpectrumSamples; ++i)
                l.pdf[i] = pdf1[offset + i];
        }
        return l;
    }

    struct GetSetIndirector {
        PBRT_CPU_GPU
        operator SampledWavelengths() const {
            return soa->Load(index);  //  (*(const SOA<SampledWavelengths> *)soa)[index];
        }
        PBRT_CPU_GPU
        void operator=(const SampledWavelengths &s) {
            if constexpr ((NSpectrumSamples % 4) == 0) {
                int offset = n4 * index;
                for (int i = 0; i < n4; ++i, ++offset) {
                    Store4(soa->lambda4 + offset, {s.lambda[4 * i], s.lambda[4 * i + 1],
                                                   s.lambda[4 * i + 2], s.lambda[4 * i + 3]});
                    Store4(soa->pdf4 + offset, {s.pdf[4 * i], s.pdf[4 * i + 1],
                                                s.pdf[4 * i + 2], s.pdf[4 * i + 3]});
                }
            } else {
                int offset = index * NSpectrumSamples;
                for (int i = 0; i < NSpectrumSamples; ++i)
                    soa->lambda1[offset + i] = s.lambda[i];
                for (int i = 0; i < NSpectrumSamples; ++i)
                    soa->pdf1[offset + i] = s.pdf[i];
            }
        }

        static constexpr int n4 = (NSpectrumSamples + 3) / 4;
        SOA<SampledWavelengths> *soa;
        int index;
    };
    PBRT_CPU_GPU
    GetSetIndirector operator[](int i) { return GetSetIndirector{this, i}; }

    PBRT_CPU_GPU
    SampledWavelengths Load(int i) const { return (*this)[i]; }
    PBRT_CPU_GPU
    void Store(int i, const SampledWavelengths &wl) { (*this)[i] = wl; }

  private:
    static constexpr int n4 = (NSpectrumSamples + 3) / 4;

    int nAlloc;
    Float4 * PBRT_RESTRICT lambda4 = nullptr;
    Float4 * PBRT_RESTRICT pdf4 = nullptr;
    Float * PBRT_RESTRICT lambda1 = nullptr;
    Float * PBRT_RESTRICT pdf1 = nullptr;
};

#include "pbrt_soa.h"

}  // namespace pbrt

#endif  // PBRT_UTIL_SOA_H
