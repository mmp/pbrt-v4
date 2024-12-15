// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_MEDIA_H
#define PBRT_MEDIA_H

#include <pbrt/pbrt.h>

#include <pbrt/base/medium.h>
#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/textures.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace pbrt {

// Media Function Declarations
bool GetMediumScatteringProperties(const std::string &name, Spectrum *sigma_a,
                                   Spectrum *sigma_s, Allocator alloc);

// HGPhaseFunction Definition
class HGPhaseFunction {
  public:
    // HGPhaseFunction Public Methods
    HGPhaseFunction() = default;
    PBRT_CPU_GPU
    HGPhaseFunction(Float g) : g(g) {}

    PBRT_CPU_GPU
    Float p(Vector3f wo, Vector3f wi) const { return HenyeyGreenstein(Dot(wo, wi), g); }

    PBRT_CPU_GPU
    pstd::optional<PhaseFunctionSample> Sample_p(Vector3f wo, Point2f u) const {
        Float pdf;
        Vector3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{pdf, wi, pdf};
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f wo, Vector3f wi) const { return p(wo, wi); }

    static const char *Name() { return "Henyey-Greenstein"; }

    std::string ToString() const;

  private:
    // HGPhaseFunction Private Members
    Float g;
};

// MediumProperties Definition
struct MediumProperties {
    SampledSpectrum sigma_a, sigma_s;
    PhaseFunction phase;
    SampledSpectrum Le;
};

// HomogeneousMajorantIterator Definition
class HomogeneousMajorantIterator {
  public:
    // HomogeneousMajorantIterator Public Methods
    PBRT_CPU_GPU
    HomogeneousMajorantIterator() : called(true) {}
    PBRT_CPU_GPU
    HomogeneousMajorantIterator(Float tMin, Float tMax, SampledSpectrum sigma_maj)
        : seg{tMin, tMax, sigma_maj}, called(false) {}

    PBRT_CPU_GPU
    pstd::optional<RayMajorantSegment> Next() {
        if (called)
            return {};
        called = true;
        return seg;
    }

    std::string ToString() const;

  private:
    RayMajorantSegment seg;
    bool called;
};

// MajorantGrid Definition
struct MajorantGrid {
    // MajorantGrid Public Methods
    MajorantGrid() = default;
    MajorantGrid(Bounds3f bounds, Point3i res, Allocator alloc)
        : bounds(bounds), voxels(res.x * res.y * res.z, alloc), res(res) {}

    PBRT_CPU_GPU
    Float Lookup(int x, int y, int z) const {
        DCHECK(x >= 0 && x < res.x && y >= 0 && y < res.y && z >= 0 && z < res.z);
        return voxels[x + res.x * (y + res.y * z)];
    }
    PBRT_CPU_GPU
    void Set(int x, int y, int z, Float v) {
        DCHECK(x >= 0 && x < res.x && y >= 0 && y < res.y && z >= 0 && z < res.z);
        voxels[x + res.x * (y + res.y * z)] = v;
    }

    PBRT_CPU_GPU
    Bounds3f VoxelBounds(int x, int y, int z) const {
        Point3f p0(Float(x) / res.x, Float(y) / res.y, Float(z) / res.z);
        Point3f p1(Float(x + 1) / res.x, Float(y + 1) / res.y, Float(z + 1) / res.z);
        return Bounds3f(p0, p1);
    }

    // MajorantGrid Public Members
    Bounds3f bounds;
    pstd::vector<Float> voxels;
    Point3i res;
};

// DDAMajorantIterator Definition
class DDAMajorantIterator {
  public:
    // DDAMajorantIterator Public Methods
    DDAMajorantIterator() = default;
    PBRT_CPU_GPU
    DDAMajorantIterator(Ray ray, Float tMin, Float tMax, const MajorantGrid *grid,
                        SampledSpectrum sigma_t)
        : tMin(tMin), tMax(tMax), grid(grid), sigma_t(sigma_t) {
        // Set up 3D DDA for ray through the majorant grid
        Vector3f diag = grid->bounds.Diagonal();
        Ray rayGrid(Point3f(grid->bounds.Offset(ray.o)),
                    Vector3f(ray.d.x / diag.x, ray.d.y / diag.y, ray.d.z / diag.z));
        Point3f gridIntersect = rayGrid(tMin);
        for (int axis = 0; axis < 3; ++axis) {
            // Initialize ray stepping parameters for _axis_
            // Compute current voxel for axis and handle negative zero direction
            voxel[axis] =
                Clamp(gridIntersect[axis] * grid->res[axis], 0, grid->res[axis] - 1);
            deltaT[axis] = 1 / (std::abs(rayGrid.d[axis]) * grid->res[axis]);
            if (rayGrid.d[axis] == -0.f)
                rayGrid.d[axis] = 0.f;

            if (rayGrid.d[axis] >= 0) {
                // Handle ray with positive direction for voxel stepping
                Float nextVoxelPos = Float(voxel[axis] + 1) / grid->res[axis];
                nextCrossingT[axis] =
                    tMin + (nextVoxelPos - gridIntersect[axis]) / rayGrid.d[axis];
                step[axis] = 1;
                voxelLimit[axis] = grid->res[axis];

            } else {
                // Handle ray with negative direction for voxel stepping
                Float nextVoxelPos = Float(voxel[axis]) / grid->res[axis];
                nextCrossingT[axis] =
                    tMin + (nextVoxelPos - gridIntersect[axis]) / rayGrid.d[axis];
                step[axis] = -1;
                voxelLimit[axis] = -1;
            }
        }
    }

    PBRT_CPU_GPU
    pstd::optional<RayMajorantSegment> Next() {
        if (tMin >= tMax)
            return {};
        // Find _stepAxis_ for stepping to next voxel and exit point _tVoxelExit_
        int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
                   ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
                   ((nextCrossingT[1] < nextCrossingT[2]));
        const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        int stepAxis = cmpToAxis[bits];
        Float tVoxelExit = std::min(tMax, nextCrossingT[stepAxis]);

        // Get _maxDensity_ for current voxel and initialize _RayMajorantSegment_, _seg_
        SampledSpectrum sigma_maj = sigma_t * grid->Lookup(voxel[0], voxel[1], voxel[2]);
        RayMajorantSegment seg{tMin, tVoxelExit, sigma_maj};

        // Advance to next voxel in maximum density grid
        tMin = tVoxelExit;
        if (nextCrossingT[stepAxis] > tMax)
            tMin = tMax;
        voxel[stepAxis] += step[stepAxis];
        if (voxel[stepAxis] == voxelLimit[stepAxis])
            tMin = tMax;
        nextCrossingT[stepAxis] += deltaT[stepAxis];

        return seg;
    }

    std::string ToString() const;

  private:
    // DDAMajorantIterator Private Members
    SampledSpectrum sigma_t;
    Float tMin = Infinity, tMax = -Infinity;
    const MajorantGrid *grid;
    Float nextCrossingT[3], deltaT[3];
    int step[3], voxelLimit[3], voxel[3];
};

// HomogeneousMedium Definition
class HomogeneousMedium {
  public:
    // HomogeneousMedium Public Type Definitions
    using MajorantIterator = HomogeneousMajorantIterator;

    // HomogeneousMedium Public Methods
    HomogeneousMedium(Spectrum sigma_a, Spectrum sigma_s, Float sigmaScale, Spectrum Le,
                      Float LeScale, Float g, Allocator alloc)
        : sigma_a_spec(sigma_a, alloc),
          sigma_s_spec(sigma_s, alloc),
          Le_spec(Le, alloc),
          phase(g) {
        sigma_a_spec.Scale(sigmaScale);
        sigma_s_spec.Scale(sigmaScale);
        Le_spec.Scale(LeScale);
    }

    static HomogeneousMedium *Create(const ParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum Le = Le_spec.Sample(lambda);
        return MediumProperties{sigma_a, sigma_s, &phase, Le};
    }

    PBRT_CPU_GPU
    HomogeneousMajorantIterator SampleRay(Ray ray, Float tMax,
                                          const SampledWavelengths &lambda) const {
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        return HomogeneousMajorantIterator(0, tMax, sigma_a + sigma_s);
    }

    std::string ToString() const;

  private:
    // HomogeneousMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec, Le_spec;
    HGPhaseFunction phase;
};

// GridMedium Definition
class GridMedium {
  public:
    // GridMedium Public Type Definitions
    using MajorantIterator = DDAMajorantIterator;

    // GridMedium Public Methods
    GridMedium(const Bounds3f &bounds, const Transform &renderFromMedium,
               Spectrum sigma_a, Spectrum sigma_s, Float sigmaScale, Float g,
               SampledGrid<Float> density, pstd::optional<SampledGrid<Float>> temperature,
               Float temperatureScale, Float temperatureOffset,
               Spectrum Le, SampledGrid<Float> LeScale, Allocator alloc);

    static GridMedium *Create(const ParameterDictionary &parameters,
                              const Transform &renderFromMedium, const FileLoc *loc,
                              Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    bool IsEmissive() const { return isEmissive; }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        // Sample spectra for grid medium $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);

        // Scale scattering coefficients by medium density at _p_
        p = renderFromMedium.ApplyInverse(p);
        p = Point3f(bounds.Offset(p));
        Float d = densityGrid.Lookup(p);
        sigma_a *= d;
        sigma_s *= d;

        // Compute grid emission _Le_ at _p_
        SampledSpectrum Le(0.f);
        if (isEmissive) {
            Float scale = LeScale.Lookup(p);
            if (scale > 0) {
                // Compute emitted radiance using _temperatureGrid_ or _Le_spec_
                if (temperatureGrid) {
                    Float temp = temperatureGrid->Lookup(p);
                    // Added after book publication: optionally offset and scale
                    // temperature based on user-supplied parameters. (Match
                    // NanoVDBMedium functionality.)
                    temp = (temp - temperatureOffset) * temperatureScale;
                    if (temp > 100.f)
                        Le = scale * BlackbodySpectrum(temp).Sample(lambda);
                } else
                    Le = scale * Le_spec.Sample(lambda);
            }
        }

        return MediumProperties{sigma_a, sigma_s, &phase, Le};
    }

    PBRT_CPU_GPU
    DDAMajorantIterator SampleRay(Ray ray, Float raytMax,
                                  const SampledWavelengths &lambda) const {
        // Transform ray to medium's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return {};
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid medium $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);

        SampledSpectrum sigma_t = sigma_a + sigma_s;
        return DDAMajorantIterator(ray, tMin, tMax, &majorantGrid, sigma_t);
    }

  private:
    // GridMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    SampledGrid<Float> densityGrid;
    HGPhaseFunction phase;
    pstd::optional<SampledGrid<Float>> temperatureGrid;
    DenselySampledSpectrum Le_spec;
    SampledGrid<Float> LeScale;
    bool isEmissive;
    Float temperatureScale, temperatureOffset;
    MajorantGrid majorantGrid;
};

// RGBGridMedium Definition
class RGBGridMedium {
  public:
    // RGBGridMedium Public Type Definitions
    using MajorantIterator = DDAMajorantIterator;

    // RGBGridMedium Public Methods
    RGBGridMedium(const Bounds3f &bounds, const Transform &renderFromMedium, Float g,
                  pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_a,
                  pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_s,
                  Float sigmaScale, pstd::optional<SampledGrid<RGBIlluminantSpectrum>> Le,
                  Float LeScale, Allocator alloc);

    static RGBGridMedium *Create(const ParameterDictionary &parameters,
                                 const Transform &renderFromMedium, const FileLoc *loc,
                                 Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    bool IsEmissive() const { return LeGrid && LeScale > 0; }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        p = renderFromMedium.ApplyInverse(p);
        p = Point3f(bounds.Offset(p));
        // Compute $\sigmaa$ and $\sigmas$ for _RGBGridMedium_
        auto convert = [=] PBRT_CPU_GPU(RGBUnboundedSpectrum s) {
            return s.Sample(lambda);
        };
        SampledSpectrum sigma_a =
            sigmaScale *
            (sigma_aGrid ? sigma_aGrid->Lookup(p, convert) : SampledSpectrum(1.f));
        SampledSpectrum sigma_s =
            sigmaScale *
            (sigma_sGrid ? sigma_sGrid->Lookup(p, convert) : SampledSpectrum(1.f));

        // Find emitted radiance _Le_ for _RGBGridMedium_
        SampledSpectrum Le(0.f);
        if (LeGrid && LeScale > 0) {
            auto convert = [=] PBRT_CPU_GPU(RGBIlluminantSpectrum s) {
                return s.Sample(lambda);
            };
            Le = LeScale * LeGrid->Lookup(p, convert);
        }

        return MediumProperties{sigma_a, sigma_s, &phase, Le};
    }

    PBRT_CPU_GPU
    DDAMajorantIterator SampleRay(Ray ray, Float raytMax,
                                  const SampledWavelengths &lambda) const {
        // Transform ray to medium's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return {};
        DCHECK_LE(tMax, raytMax);

        SampledSpectrum sigma_t(1);
        return DDAMajorantIterator(ray, tMin, tMax, &majorantGrid, sigma_t);
    }

  private:
    // RGBGridMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    pstd::optional<SampledGrid<RGBIlluminantSpectrum>> LeGrid;
    Float LeScale;
    HGPhaseFunction phase;
    pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_aGrid, sigma_sGrid;
    Float sigmaScale;
    MajorantGrid majorantGrid;
};

// CloudMedium Definition
class CloudMedium {
  public:
    // CloudMedium Public Type Definitions
    using MajorantIterator = HomogeneousMajorantIterator;

    // CloudMedium Public Methods
    static CloudMedium *Create(const ParameterDictionary &parameters,
                               const Transform &renderFromMedium, const FileLoc *loc,
                               Allocator alloc);

    std::string ToString() const {
        return StringPrintf("[ CloudMedium bounds: %s renderFromMedium: %s phase: %s "
                            "sigma_a_spec: %s sigma_s_spec: %s density: %f wispiness: %f "
                            "frequency: %f ]",
                            bounds, renderFromMedium, phase, sigma_a_spec, sigma_s_spec,
                            density, wispiness, frequency);
    }

    CloudMedium(const Bounds3f &bounds, const Transform &renderFromMedium,
                Spectrum sigma_a, Spectrum sigma_s, Float g, Float density,
                Float wispiness, Float frequency, Allocator alloc)
        : bounds(bounds),
          renderFromMedium(renderFromMedium),
          sigma_a_spec(sigma_a, alloc),
          sigma_s_spec(sigma_s, alloc),
          phase(g),
          density(density),
          wispiness(wispiness),
          frequency(frequency) {}

    PBRT_CPU_GPU
    bool IsEmissive() const { return false; }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        // Compute sampled spectra for cloud $\sigmaa$ and $\sigmas$ at _p_
        Float density = Density(renderFromMedium.ApplyInverse(p));
        SampledSpectrum sigma_a = density * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = density * sigma_s_spec.Sample(lambda);

        return MediumProperties{sigma_a, sigma_s, &phase, SampledSpectrum(0.f)};
    }

    PBRT_CPU_GPU
    HomogeneousMajorantIterator SampleRay(Ray ray, Float raytMax,
                                          const SampledWavelengths &lambda) const {
        // Transform ray to medium's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return {};
        DCHECK_LE(tMax, raytMax);

        // Compute $\sigmat$ bound for cloud medium and initialize majorant iterator
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        return HomogeneousMajorantIterator(tMin, tMax, sigma_t);
    }

  private:
    // CloudMedium Private Methods
    PBRT_CPU_GPU
    Float Density(Point3f p) const {
        Point3f pp = frequency * p;
        if (wispiness > 0) {
            // Perturb cloud lookup point _pp_ using noise
            Float vomega = 0.05f * wispiness, vlambda = 10.f;
            for (int i = 0; i < 2; ++i) {
                pp += vomega * DNoise(vlambda * pp);
                vomega *= 0.5f;
                vlambda *= 1.99f;
            }
        }
        // Sum scales of noise to approximate cloud density
        Float d = 0;
        Float omega = 0.5f, lambda = 1.f;
        for (int i = 0; i < 5; ++i) {
            d += omega * Noise(lambda * pp);
            omega *= 0.5f;
            lambda *= 1.99f;
        }

        // Model decrease in density with altitude and return final cloud density
        d = Clamp((1 - p.y) * 4.5f * density * d, 0, 1);
        d += 2 * std::max<Float>(0, 0.5f - p.y);
        return Clamp(d, 0, 1);
    }

    // CloudMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    HGPhaseFunction phase;
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    Float density, wispiness, frequency;
};

// NanoVDBMedium Definition
// NanoVDBBuffer Definition
class NanoVDBBuffer {
  public:
    static inline void ptrAssert(void *ptr, const char *msg, const char *file, int line,
                                 bool abort = true) {
        if (abort)
            LOG_FATAL("%p: %s (%s:%d)", ptr, msg, file, line);
        else
            LOG_ERROR("%p: %s (%s:%d)", ptr, msg, file, line);
    }

    NanoVDBBuffer() = default;
    NanoVDBBuffer(Allocator alloc) : alloc(alloc) {}
    NanoVDBBuffer(size_t size, Allocator alloc = {}) : alloc(alloc) { init(size); }
    NanoVDBBuffer(const NanoVDBBuffer &) = delete;
    NanoVDBBuffer(NanoVDBBuffer &&other) noexcept
        : alloc(std::move(other.alloc)),
          bytesAllocated(other.bytesAllocated),
          ptr(other.ptr) {
        other.bytesAllocated = 0;
        other.ptr = nullptr;
    }
    NanoVDBBuffer &operator=(const NanoVDBBuffer &) = delete;
    NanoVDBBuffer &operator=(NanoVDBBuffer &&other) noexcept {
        // Note, this isn't how std containers work, but it's expedient for
        // our purposes here...
        clear();
        // operator= was deleted? Fine.
        new (&alloc) Allocator(other.alloc.resource());
        bytesAllocated = other.bytesAllocated;
        ptr = other.ptr;
        other.bytesAllocated = 0;
        other.ptr = nullptr;
        return *this;
    }
    ~NanoVDBBuffer() { clear(); }

    void init(uint64_t size) {
        if (size == bytesAllocated)
            return;
        if (bytesAllocated > 0)
            clear();
        if (size == 0)
            return;
        bytesAllocated = size;
        ptr = (uint8_t *)alloc.allocate_bytes(bytesAllocated, 128);
    }

    const uint8_t *data() const { return ptr; }
    uint8_t *data() { return ptr; }
    uint64_t size() const { return bytesAllocated; }
    bool empty() const { return size() == 0; }

    void clear() {
        alloc.deallocate_bytes(ptr, bytesAllocated, 128);
        bytesAllocated = 0;
        ptr = nullptr;
    }

    static NanoVDBBuffer create(uint64_t size, const NanoVDBBuffer *context = nullptr) {
        return NanoVDBBuffer(size, context ? context->GetAllocator() : Allocator());
    }

    Allocator GetAllocator() const { return alloc; }

  private:
    Allocator alloc;
    size_t bytesAllocated = 0;
    uint8_t *ptr = nullptr;
};

class NanoVDBMedium {
  public:
    using MajorantIterator = DDAMajorantIterator;
    // NanoVDBMedium Public Methods
    static NanoVDBMedium *Create(const ParameterDictionary &parameters,
                                 const Transform &renderFromMedium, const FileLoc *loc,
                                 Allocator alloc);

    std::string ToString() const;

    NanoVDBMedium(const Transform &renderFromMedium, Spectrum sigma_a, Spectrum sigma_s,
                  Float sigmaScale, Float g, nanovdb::GridHandle<NanoVDBBuffer> dg,
                  nanovdb::GridHandle<NanoVDBBuffer> tg, Float LeScale,
                  Float temperatureOffset, Float temperatureScale, Allocator alloc);

    PBRT_CPU_GPU
    bool IsEmissive() const { return temperatureFloatGrid && LeScale > 0; }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);

        // Scale scattering coefficients by medium density at _p_
        p = renderFromMedium.ApplyInverse(p);

        nanovdb::Vec3<float> pIndex =
            densityFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));
        using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        Float d = Sampler(densityFloatGrid->tree())(pIndex);

        return MediumProperties{sigma_a * d, sigma_s * d, &phase, Le(p, lambda)};
    }

    PBRT_CPU_GPU
    DDAMajorantIterator SampleRay(Ray ray, Float raytMax,
                                  const SampledWavelengths &lambda) const {
        // Transform ray to medium's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return {};
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);

        SampledSpectrum sigma_t = sigma_a + sigma_s;
        return DDAMajorantIterator(ray, tMin, tMax, &majorantGrid, sigma_t);
    }

  private:
    // NanoVDBMedium Private Methods
    PBRT_CPU_GPU
    SampledSpectrum Le(Point3f p, const SampledWavelengths &lambda) const {
        if (!temperatureFloatGrid)
            return SampledSpectrum(0.f);
        nanovdb::Vec3<float> pIndex =
            temperatureFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));
        using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        Float temp = Sampler(temperatureFloatGrid->tree())(pIndex);
        temp = (temp - temperatureOffset) * temperatureScale;
        if (temp <= 100.f)
            return SampledSpectrum(0.f);
        return LeScale * BlackbodySpectrum(temp).Sample(lambda);
    }

    // NanoVDBMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    HGPhaseFunction phase;
    MajorantGrid majorantGrid;
    nanovdb::GridHandle<NanoVDBBuffer> densityGrid;
    nanovdb::GridHandle<NanoVDBBuffer> temperatureGrid;
    const nanovdb::FloatGrid *densityFloatGrid = nullptr;
    const nanovdb::FloatGrid *temperatureFloatGrid = nullptr;
    Float LeScale, temperatureOffset, temperatureScale;
};

PBRT_CPU_GPU inline Float PhaseFunction::p(Vector3f wo, Vector3f wi) const {
    auto p = [&](auto ptr) { return ptr->p(wo, wi); };
    return Dispatch(p);
}

PBRT_CPU_GPU inline pstd::optional<PhaseFunctionSample> PhaseFunction::Sample_p(Vector3f wo,
                                                                   Point2f u) const {
    auto sample = [&](auto ptr) { return ptr->Sample_p(wo, u); };
    return Dispatch(sample);
}

PBRT_CPU_GPU inline Float PhaseFunction::PDF(Vector3f wo, Vector3f wi) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(wo, wi); };
    return Dispatch(pdf);
}

PBRT_CPU_GPU inline pstd::optional<RayMajorantSegment> RayMajorantIterator::Next() {
    auto next = [](auto ptr) { return ptr->Next(); };
    return Dispatch(next);
}

PBRT_CPU_GPU inline MediumProperties Medium::SamplePoint(Point3f p,
                                            const SampledWavelengths &lambda) const {
    auto sample = [&](auto ptr) { return ptr->SamplePoint(p, lambda); };
    return Dispatch(sample);
}

// Medium Sampling Function Definitions
inline RayMajorantIterator Medium::SampleRay(Ray ray, Float tMax,
                                             const SampledWavelengths &lambda,
                                             ScratchBuffer &buf) const {
    // Explicit capture to work around MSVC weirdness; it doesn't see |buf| otherwise...
    auto sample = [ray, tMax, lambda, &buf](auto medium) {
        // Return _RayMajorantIterator_ for medium's majorant iterator
        using ConcreteMedium = typename std::remove_reference_t<decltype(*medium)>;
        using Iter = typename ConcreteMedium::MajorantIterator;
        Iter *iter = (Iter *)buf.Alloc(sizeof(Iter), alignof(Iter));
        *iter = medium->SampleRay(ray, tMax, lambda);
        return RayMajorantIterator(iter);
    };
    return DispatchCPU(sample);
}

template <typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj(Ray ray, Float tMax, Float u, RNG &rng,
                                         const SampledWavelengths &lambda, F callback) {
    auto sample = [&](auto medium) {
        using M = typename std::remove_reference_t<decltype(*medium)>;
        return SampleT_maj<M>(ray, tMax, u, rng, lambda, callback);
    };
    return ray.medium.Dispatch(sample);
}

template <typename ConcreteMedium, typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj(Ray ray, Float tMax, Float u, RNG &rng,
                                         const SampledWavelengths &lambda, F callback) {
    // Normalize ray direction and update _tMax_ accordingly
    tMax *= Length(ray.d);
    ray.d = Normalize(ray.d);

    // Initialize _MajorantIterator_ for ray majorant sampling
    ConcreteMedium *medium = ray.medium.Cast<ConcreteMedium>();
    typename ConcreteMedium::MajorantIterator iter = medium->SampleRay(ray, tMax, lambda);

    // Generate ray majorant samples until termination
    SampledSpectrum T_maj(1.f);
    bool done = false;
    while (!done) {
        // Get next majorant segment from iterator and sample it
        pstd::optional<RayMajorantSegment> seg = iter.Next();
        if (!seg)
            return T_maj;
        // Handle zero-valued majorant for current segment
        if (seg->sigma_maj[0] == 0) {
            Float dt = seg->tMax - seg->tMin;
            // Handle infinite _dt_ for ray majorant segment
            if (IsInf(dt))
                dt = std::numeric_limits<Float>::max();

            T_maj *= FastExp(-dt * seg->sigma_maj);
            continue;
        }

        // Generate samples along current majorant segment
        Float tMin = seg->tMin;
        while (true) {
            // Try to generate sample along current majorant segment
            Float t = tMin + SampleExponential(u, seg->sigma_maj[0]);
            PBRT_DBG("Sampled t = %f from tMin %f u %f sigma_maj[0] %f\n", t, tMin, u,
                     seg->sigma_maj[0]);
            u = rng.Uniform<Float>();
            if (t < seg->tMax) {
                // Call callback function for sample within segment
                PBRT_DBG("t < seg->tMax\n");
                T_maj *= FastExp(-(t - tMin) * seg->sigma_maj);
                MediumProperties mp = medium->SamplePoint(ray(t), lambda);
                if (!callback(ray(t), mp, seg->sigma_maj, T_maj)) {
                    // Returning out of doubly-nested while loop is not as good perf. wise
                    // on the GPU vs using "done" here.
                    done = true;
                    break;
                }
                T_maj = SampledSpectrum(1.f);
                tMin = t;

            } else {
                // Handle sample past end of majorant segment
                Float dt = seg->tMax - tMin;
                // Handle infinite _dt_ for ray majorant segment
                if (IsInf(dt))
                    dt = std::numeric_limits<Float>::max();

                T_maj *= FastExp(-dt * seg->sigma_maj);
                PBRT_DBG("Past end, added dt %f * maj[0] %f\n", dt, seg->sigma_maj[0]);
                break;
            }
        }
    }
    return SampledSpectrum(1.f);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_H
