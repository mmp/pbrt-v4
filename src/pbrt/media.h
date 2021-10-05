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
#ifdef PBRT_BUILD_GPU_RENDERER
#include <nanovdb/util/CudaDeviceBuffer.h>
#endif  // PBRT_BUILD_GPU_RENDERER

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
    HomogeneousMajorantIterator() = default;
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
    bool called = true;
};

// DDAMajorantIterator Definition
class DDAMajorantIterator {
  public:
    // DDAMajorantIterator Public Methods
    DDAMajorantIterator() = default;
    PBRT_CPU_GPU
    DDAMajorantIterator(Ray ray, Bounds3f bounds, SampledSpectrum sigma_t, Float tMin,
                        Float tMax, const pstd::vector<Float> *grid, Point3i res)
        : sigma_t(sigma_t), tMin(tMin), tMax(tMax), grid(grid), res(res) {
        // Set up 3D DDA for ray through the majorant grid
        Vector3f diag = bounds.Diagonal();
        Ray rayGrid(Point3f(bounds.Offset(ray.o)),
                    Vector3f(ray.d.x / diag.x, ray.d.y / diag.y, ray.d.z / diag.z));
        Point3f gridIntersect = rayGrid(tMin);
        for (int axis = 0; axis < 3; ++axis) {
            // Initialize ray stepping parameters for _axis_
            // Compute current voxel for axis and handle negative zero direction
            voxel[axis] = Clamp(gridIntersect[axis] * res[axis], 0, res[axis] - 1);
            deltaT[axis] = 1 / (std::abs(rayGrid.d[axis]) * res[axis]);
            if (rayGrid.d[axis] == -0.f)
                rayGrid.d[axis] = 0.f;

            if (rayGrid.d[axis] >= 0) {
                // Handle ray with positive direction for voxel stepping
                Float nextVoxelPos = Float(voxel[axis] + 1) / res[axis];
                nextCrossingT[axis] =
                    tMin + (nextVoxelPos - gridIntersect[axis]) / rayGrid.d[axis];
                step[axis] = 1;
                voxelLimit[axis] = res[axis];

            } else {
                // Handle ray with negative direction for voxel stepping
                Float nextVoxelPos = Float(voxel[axis]) / res[axis];
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
        int offset = voxel[0] + res.x * (voxel[1] + res.y * voxel[2]);
        Float maxDensity = (*grid)[offset];
        SampledSpectrum sigma_maj = sigma_t * maxDensity;
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
    const pstd::vector<Float> *grid;
    Point3i res;
    Float nextCrossingT[3], deltaT[3];
    int step[3], voxelLimit[3], voxel[3];
};

// HomogeneousMedium Definition
class HomogeneousMedium {
  public:
    // HomogeneousMedium Public Type Definitions
    using MajorantIterator = HomogeneousMajorantIterator;

    // HomogeneousMedium Public Methods
    HomogeneousMedium(Spectrum sigma_a, Spectrum sigma_s, Float sigScale, Spectrum Le,
                      Float LeScale, Float g, Allocator alloc)
        : sigma_a_spec(sigma_a, alloc),
          sigma_s_spec(sigma_s, alloc),
          Le_spec(Le, alloc),
          phase(g) {
        sigma_a_spec.Scale(sigScale);
        sigma_s_spec.Scale(sigScale);
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
    void SampleRay(Ray ray, Float tMax, const SampledWavelengths &lambda,
                   HomogeneousMajorantIterator *iter) const {
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        *iter = HomogeneousMajorantIterator(0, tMax, sigma_a + sigma_s);
    }

    std::string ToString() const;

  private:
    // HomogeneousMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec, Le_spec;
    HGPhaseFunction phase;
};

// CuboidMedium Definition
template <typename CuboidProvider>
class CuboidMedium {
  public:
    // CuboidMedium Public Type Definitions
    using MajorantIterator = DDAMajorantIterator;

    // CuboidMedium Public Methods
    CuboidMedium(const CuboidProvider *provider, Spectrum sigma_a, Spectrum sigma_s,
                 Float sigScale, Float g, const Transform &renderFromMedium,
                 Allocator alloc)
        : provider(provider),
          bounds(provider->Bounds()),
          sigma_a_spec(sigma_a, alloc),
          sigma_s_spec(sigma_s, alloc),
          sigScale(sigScale),
          phase(g),
          renderFromMedium(renderFromMedium),
          maxDensityGrid(alloc) {
        // Initialize _maxDensityGrid_
        maxDensityGrid = provider->GetMaxDensityGrid(alloc, &gridResolution);
    }

    std::string ToString() const {
        return StringPrintf("[ CuboidMedium provider: %s bounds: %s "
                            "sigma_a_spec: %s sigma_s_spec: %s sigScale: %f phase: %s "
                            "maxDensityGrid: %s gridResolution: %s ]",
                            *provider, bounds, sigma_a_spec, sigma_s_spec, sigScale,
                            phase, maxDensityGrid, gridResolution);
    }

    PBRT_CPU_GPU
    bool IsEmissive() const { return provider->IsEmissive(); }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);

        // Scale scattering coefficients by medium density at _p_
        p = renderFromMedium.ApplyInverse(p);
        MediumDensity d = provider->Density(p, lambda);
        sigma_a *= d.sigma_a;
        sigma_s *= d.sigma_s;

        return MediumProperties{sigma_a, sigma_s, &phase, provider->Le(p, lambda)};
    }

    PBRT_CPU_GPU
    void SampleRay(Ray ray, Float raytMax, const SampledWavelengths &lambda,
                   DDAMajorantIterator *iter) const {
        // Transform ray to grid density's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax)) {
            *iter = DDAMajorantIterator();
            return;
        }
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);

        // Initialize majorant iterator for _CuboidMedium_
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        *iter = DDAMajorantIterator(ray, bounds, sigma_t, tMin, tMax, &maxDensityGrid,
                                    gridResolution);
    }

    static CuboidMedium<CuboidProvider> *Create(const CuboidProvider *provider,
                                                const ParameterDictionary &parameters,
                                                const Transform &renderFromMedium,
                                                const FileLoc *loc, Allocator alloc) {
        Spectrum sig_a = nullptr, sig_s = nullptr;
        std::string preset = parameters.GetOneString("preset", "");
        if (!preset.empty()) {
            if (!GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc))
                Warning(loc, "Material preset \"%s\" not found.", preset);
        }

        if (!sig_a) {
            sig_a = parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::Unbounded,
                                              alloc);
            if (!sig_a)
                sig_a = alloc.new_object<ConstantSpectrum>(1.f);
        }
        if (!sig_s) {
            sig_s = parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::Unbounded,
                                              alloc);
            if (!sig_s)
                sig_s = alloc.new_object<ConstantSpectrum>(1.f);
        }

        Float sigScale = parameters.GetOneFloat("scale", 1.f);

        Float g = parameters.GetOneFloat("g", 0.0f);

        return alloc.new_object<CuboidMedium<CuboidProvider>>(
            provider, sig_a, sig_s, sigScale, g, renderFromMedium, alloc);
    }

  private:
    // CuboidMedium Private Members
    const CuboidProvider *provider;
    Bounds3f bounds;
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    Float sigScale;
    HGPhaseFunction phase;
    Transform renderFromMedium;
    pstd::vector<Float> maxDensityGrid;
    Point3i gridResolution;
};

// GridMedium Definition
class GridMedium {
  public:
    using MajorantIterator = DDAMajorantIterator;

    // GridMedium Public Methods
    GridMedium(const Bounds3f &bounds, const Transform &renderFromMedium,
               Spectrum sigma_a, Spectrum sigma_s, Float sigScale, Float g,
               SampledGrid<Float> density,
               pstd::optional<SampledGrid<Float>> temperature, Spectrum Le,
               SampledGrid<Float> LeScale, Allocator alloc);

    static GridMedium *Create(const ParameterDictionary &parameters,
                              const Transform &renderFromMedium,
                              const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    const Bounds3f &Bounds() const { return bounds; }

    PBRT_CPU_GPU
    bool IsEmissive() const {
        return isEmissive;
    }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);

        // Scale scattering coefficients by medium density at _p_
        p = renderFromMedium.ApplyInverse(p);
        p = Point3f(bounds.Offset(p));
        Float d = densityGrid.Lookup(p);

        return MediumProperties{sigma_a * d, sigma_s * d, &phase, Le(p, lambda)};
    }

    PBRT_CPU_GPU
    void SampleRay(Ray ray, Float raytMax, const SampledWavelengths &lambda,
                   DDAMajorantIterator *iter) const {
        // Transform ray to grid density's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax)) {
            *iter = DDAMajorantIterator();
            return;
        }
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);

        // Initialize majorant iterator for _CuboidMedium_
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        *iter = DDAMajorantIterator(ray, bounds, sigma_t, tMin, tMax, &maxDensityGrid,
                                    maxDensityGridRes);
    }

  private:
    // GridMedium Private Methods
    PBRT_CPU_GPU
    SampledSpectrum Le(Point3f p, const SampledWavelengths &lambda) const {
        if (!isEmissive)
            return SampledSpectrum(0.f);

        if (temperatureGrid) {
            Float temp = temperatureGrid->Lookup(p);
            if (temp <= 100.f)
                return SampledSpectrum(0.f);
            return LeScale.Lookup(p) * BlackbodySpectrum(temp).Sample(lambda);
        }
        return Le_spec.Sample(lambda) * LeScale.Lookup(p);
    }

    // GridMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    Float sigScale;
    HGPhaseFunction phase;
    Point3i maxDensityGridRes;
    pstd::vector<Float> maxDensityGrid;
    SampledGrid<Float> densityGrid;
    pstd::optional<SampledGrid<Float>> temperatureGrid;
    DenselySampledSpectrum Le_spec;
    SampledGrid<Float> LeScale;
    bool isEmissive;
};

// RGBGridMedium Definition
class RGBGridMedium {
  public:
    using MajorantIterator = DDAMajorantIterator;
    // RGBGridMedium Public Methods
    RGBGridMedium(const Bounds3f &bounds, const Transform &renderFromMedium,
                  Float g,
                  pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_a,
                  pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_s,
                  Float sigScale,
                  pstd::optional<SampledGrid<RGBUnboundedSpectrum>> Le,
                  Float LeScale, Allocator alloc);

    static RGBGridMedium *Create(const ParameterDictionary &parameters,
                                 const Transform &renderFromMedium,
                                 const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    const Bounds3f &Bounds() const { return bounds; }

    PBRT_CPU_GPU
    bool IsEmissive() const { return LeGrid && LeScale > 0.0f; }


    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        p = renderFromMedium.ApplyInverse(p);
        Point3f pp = Point3f(bounds.Offset(p));
        auto convert = [=] PBRT_CPU_GPU(RGBUnboundedSpectrum s) {
            return s.Sample(lambda);
        };
        SampledSpectrum sigma_a = sigma_aGrid ? sigma_aGrid->Lookup(pp, convert) :
            SampledSpectrum(1.f);
        SampledSpectrum sigma_s = sigma_sGrid ? sigma_sGrid->Lookup(pp, convert) :
            SampledSpectrum(1.f);

        return MediumProperties{sigma_a * sigScale, sigma_s * sigScale, &phase,
                Le(pp, lambda)};
    }

    PBRT_CPU_GPU
    void SampleRay(Ray ray, Float raytMax, const SampledWavelengths &lambda,
                   DDAMajorantIterator *iter) const {
        // Transform ray to grid density's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax)) {
            *iter = DDAMajorantIterator();
            return;
        }
        DCHECK_LE(tMax, raytMax);

        // Initialize majorant iterator for _CuboidMedium_
        SampledSpectrum sigma_t(sigScale);
        *iter = DDAMajorantIterator(ray, bounds, sigma_t, tMin, tMax, &maxDensityGrid,
                                    maxDensityGridRes);
    }

    PBRT_CPU_GPU
    SampledSpectrum Le(Point3f p, const SampledWavelengths &lambda) const {
        if (!LeGrid)
            return SampledSpectrum(0.f);

        auto convert = [lambda] PBRT_CPU_GPU(RGBUnboundedSpectrum s) {
            return s.Sample(lambda);
        };
        return LeScale * LeGrid->Lookup(p, convert);
    }

  private:
    // RGBGridMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    HGPhaseFunction phase;
    Point3i maxDensityGridRes;
    pstd::vector<Float> maxDensityGrid;
    SampledGrid<Float> densityGrid;
    Float sigScale;
    pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_aGrid, sigma_sGrid, LeGrid;
    Float LeScale;
};

// CloudMedium Definition
class CloudMedium {
  public:
    using MajorantIterator = HomogeneousMajorantIterator;

    // CloudMediumProvider Public Methods
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
                Spectrum sigma_a, Spectrum sigma_s, Float g, Float density, Float wispiness,
                Float frequency, Allocator alloc)
        : bounds(bounds), sigma_a_spec(sigma_a, alloc), sigma_s_spec(sigma_s,alloc), phase(g),
          density(density), wispiness(wispiness), frequency(frequency), renderFromMedium(renderFromMedium) {}

    PBRT_CPU_GPU
    bool IsEmissive() const { return false; }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);

        // Scale scattering coefficients by medium density at _p_
        p = renderFromMedium.ApplyInverse(p);
        Float density = Density(p);

        return MediumProperties{sigma_a * density, sigma_s * density, &phase,
                                SampledSpectrum(0.f)};
    }

    PBRT_CPU_GPU
    void SampleRay(Ray ray, Float raytMax, const SampledWavelengths &lambda,
                   HomogeneousMajorantIterator *iter) const {
        // Transform ray to grid density's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax)) {
            *iter = HomogeneousMajorantIterator();
            return;
        }
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        *iter = HomogeneousMajorantIterator(tMin, tMax, sigma_t);
    }

  private:
    // CloudMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    HGPhaseFunction phase;
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    Float density, wispiness, frequency;

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
                                 const Transform &renderFromMedium,
                                 const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    NanoVDBMedium(const Transform &renderFromMedium, Spectrum sigma_a, Spectrum sigma_s,
                  Float sigScale, Float g, nanovdb::GridHandle<NanoVDBBuffer> dg,
                  nanovdb::GridHandle<NanoVDBBuffer> tg, Float LeScale,
                  Float temperatureCutoff, Float temperatureScale, Allocator alloc);

    PBRT_CPU_GPU
    bool IsEmissive() const { return temperatureFloatGrid && LeScale > 0; }

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const {
        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);

        // Scale scattering coefficients by medium density at _p_
        p = renderFromMedium.ApplyInverse(p);

        nanovdb::Vec3<float> pIndex =
            densityFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));
        using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        Float d = Sampler(densityFloatGrid->tree())(pIndex);

        return MediumProperties{sigma_a * d, sigma_s * d, &phase, Le(p, lambda)};
    }

    PBRT_CPU_GPU
    void SampleRay(Ray ray, Float raytMax, const SampledWavelengths &lambda,
                   DDAMajorantIterator *iter) const {
        // Transform ray to grid density's space and compute bounds overlap
        ray = renderFromMedium.ApplyInverse(ray, &raytMax);
        Float tMin, tMax;
        if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax)) {
            *iter = DDAMajorantIterator();
            return;
        }
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid $\sigmaa$ and $\sigmas$
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);

        // Initialize majorant iterator for _CuboidMedium_
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        *iter = DDAMajorantIterator(ray, bounds, sigma_t, tMin, tMax, &maxDensityGrid,
                                    maxDensityGridRes);
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
        temp = (temp - temperatureCutoff) * temperatureScale;
        if (temp <= 100.f)
            return SampledSpectrum(0.f);
        return LeScale * BlackbodySpectrum(temp).Sample(lambda);
    }

    // NanoVDBMedium Private Members
    Bounds3f bounds;
    Transform renderFromMedium;
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    Float sigScale;
    HGPhaseFunction phase;
    Point3i maxDensityGridRes;
    pstd::vector<Float> maxDensityGrid;
    nanovdb::GridHandle<NanoVDBBuffer> densityGrid;
    nanovdb::GridHandle<NanoVDBBuffer> temperatureGrid;
    const nanovdb::FloatGrid *densityFloatGrid = nullptr;
    const nanovdb::FloatGrid *temperatureFloatGrid = nullptr;
    Float LeScale, temperatureCutoff, temperatureScale;
};

inline Float PhaseFunction::p(Vector3f wo, Vector3f wi) const {
    auto p = [&](auto ptr) { return ptr->p(wo, wi); };
    return Dispatch(p);
}

inline pstd::optional<PhaseFunctionSample> PhaseFunction::Sample_p(Vector3f wo,
                                                                   Point2f u) const {
    auto sample = [&](auto ptr) { return ptr->Sample_p(wo, u); };
    return Dispatch(sample);
}

inline Float PhaseFunction::PDF(Vector3f wo, Vector3f wi) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(wo, wi); };
    return Dispatch(pdf);
}

inline pstd::optional<RayMajorantSegment> RayMajorantIterator::Next() {
    auto next = [](auto ptr) { return ptr->Next(); };
    return Dispatch(next);
}

inline MediumProperties Medium::SamplePoint(Point3f p,
                                            const SampledWavelengths &lambda) const {
    auto sample = [&](auto ptr) { return ptr->SamplePoint(p, lambda); };
    return Dispatch(sample);
}

// Medium Sampling Function Definitions
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
    typename ConcreteMedium::MajorantIterator iter;
    medium->SampleRay(ray, tMax, lambda, &iter);

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

inline RayMajorantIterator Medium::SampleRay(Ray ray, Float tMax,
                                             const SampledWavelengths &lambda,
                                             ScratchBuffer &buf) const {
    // Explicit capture to work around MSVC weirdness; it doesn't see |buf| otherwise...
    auto sample = [ray, tMax, lambda, &buf](auto medium) -> RayMajorantIterator {
        using Medium = typename std::remove_reference_t<decltype(*medium)>;
        using Iter = typename Medium::MajorantIterator;
        Iter *iter = (Iter *)buf.Alloc(sizeof(Iter), alignof(Iter));
        medium->SampleRay(ray, tMax, lambda, iter);
        return iter;
    };
    return DispatchCPU(sample);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_H
