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
    HomogeneousMajorantIterator(Float tMax, SampledSpectrum sigma_maj)
        : seg{Float(0), tMax, sigma_maj}, called(false) {}

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
        *iter = HomogeneousMajorantIterator(tMax, sigma_a + sigma_s);
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

// UniformGridMediumProvider Definition
class UniformGridMediumProvider {
  public:
    // UniformGridMediumProvider Public Methods
    UniformGridMediumProvider(const Bounds3f &bounds,
                              pstd::optional<SampledGrid<Float>> density,
                              pstd::optional<SampledGrid<Float>> sigma_a,
                              pstd::optional<SampledGrid<Float>> sigma_s,
                              pstd::optional<SampledGrid<RGBUnboundedSpectrum>> rgb,
                              Spectrum Le, SampledGrid<Float> LeScale, Allocator alloc);

    static UniformGridMediumProvider *Create(const ParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    const Bounds3f &Bounds() const { return bounds; }

    PBRT_CPU_GPU
    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    PBRT_CPU_GPU
    SampledSpectrum Le(Point3f p, const SampledWavelengths &lambda) const {
        Point3f pp = Point3f(bounds.Offset(p));
        return Le_spec.Sample(lambda) * LeScale.Lookup(pp);
    }

    PBRT_CPU_GPU
    MediumDensity Density(Point3f p, const SampledWavelengths &lambda) const {
        Point3f pp = Point3f(bounds.Offset(p));
        if (densityGrid)
            return MediumDensity(densityGrid->Lookup(pp));
        else if (sigma_aGrid)
            return MediumDensity(SampledSpectrum(sigma_aGrid->Lookup(pp)),
                                 SampledSpectrum(sigma_sGrid->Lookup(pp)));
        else {
            // Return _SampledSpectrum_ density from _rgb_
            auto convert = [=] PBRT_CPU_GPU(RGBUnboundedSpectrum s) {
                return s.Sample(lambda);
            };
            SampledSpectrum d = rgbGrid->Lookup(pp, convert);
            return MediumDensity(d, d);
        }
    }

    pstd::vector<Float> GetMaxDensityGrid(Allocator alloc, Point3i *res) const {
        *res = Point3i(16, 16, 16);
        pstd::vector<Float> maxGrid(res->x * res->y * res->z, Float(0), alloc);
        // Compute maximum density for each _maxGrid_ cell
        int offset = 0;
        for (Float z = 0; z < res->z; ++z)
            for (Float y = 0; y < res->y; ++y)
                for (Float x = 0; x < res->x; ++x) {
                    Bounds3f bounds(
                        Point3f(x / res->x, y / res->y, z / res->z),
                        Point3f((x + 1) / res->x, (y + 1) / res->y, (z + 1) / res->z));
                    // Set current _maxGrid_ entry for maximum density over _bounds_
                    if (densityGrid)
                        maxGrid[offset++] = densityGrid->MaxValue(bounds);
                    else if (sigma_aGrid)
                        maxGrid[offset++] =
                            std::max<Float>(sigma_aGrid->MaxValue(bounds), sigma_sGrid->MaxValue(bounds));
                    else {
                        auto max = [] PBRT_CPU_GPU(RGBUnboundedSpectrum s) {
                            return s.MaxValue();
                        };
                        maxGrid[offset++] = rgbGrid->MaxValue(bounds, max);
                    }
                }

        return maxGrid;
    }

  private:
    // UniformGridMediumProvider Private Members
    Bounds3f bounds;
    pstd::optional<SampledGrid<Float>> densityGrid;
    pstd::optional<SampledGrid<Float>> sigma_aGrid, sigma_sGrid;
    pstd::optional<SampledGrid<RGBUnboundedSpectrum>> rgbGrid;
    DenselySampledSpectrum Le_spec;
    SampledGrid<Float> LeScale;
};

// CloudMediumProvider Definition
class CloudMediumProvider {
  public:
    // CloudMediumProvider Public Methods
    static CloudMediumProvider *Create(const ParameterDictionary &parameters,
                                       const FileLoc *loc, Allocator alloc);

    std::string ToString() const {
        return StringPrintf("[ CloudMediumProvider bounds: %s density: %f "
                            "wispiness: %f frequency: %f ]",
                            bounds, density, wispiness, frequency);
    }

    CloudMediumProvider(const Bounds3f &bounds, Float density, Float wispiness,
                        Float frequency)
        : bounds(bounds), density(density), wispiness(wispiness), frequency(frequency) {}

    PBRT_CPU_GPU
    const Bounds3f &Bounds() const { return bounds; }

    PBRT_CPU_GPU
    bool IsEmissive() const { return false; }

    PBRT_CPU_GPU
    SampledSpectrum Le(Point3f p, const SampledWavelengths &lambda) const {
        return SampledSpectrum(0.f);
    }

    PBRT_CPU_GPU
    MediumDensity Density(Point3f p, const SampledWavelengths &) const {
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
        return MediumDensity(Clamp(d, 0, 1));
    }

    pstd::vector<Float> GetMaxDensityGrid(Allocator alloc, Point3i *res) const {
        *res = Point3i(1, 1, 1);
        return pstd::vector<Float>(1, 1.f, alloc);
    }

  private:
    // CloudMediumProvider Private Members
    Bounds3f bounds;
    Float density, wispiness, frequency;
};

// NanoVDBMediumProvider Definition
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

class NanoVDBMediumProvider {
  public:
    // NanoVDBMediumProvider Public Methods
    static NanoVDBMediumProvider *Create(const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc);

    std::string ToString() const {
        return StringPrintf("[ NanoVDBMediumProvider bounds: %s LeScale: %f "
                            "temperatureCutoff: %f temperatureScale: %f (grids elided) ]",
                            bounds, LeScale, temperatureCutoff, temperatureScale);
    }

    NanoVDBMediumProvider(nanovdb::GridHandle<NanoVDBBuffer> dg,
                          nanovdb::GridHandle<NanoVDBBuffer> tg, Float LeScale,
                          Float temperatureCutoff, Float temperatureScale)
        : densityGrid(std::move(dg)),
          temperatureGrid(std::move(tg)),
          LeScale(LeScale),
          temperatureCutoff(temperatureCutoff),
          temperatureScale(temperatureScale) {
        densityFloatGrid = densityGrid.grid<float>();

        nanovdb::BBox<nanovdb::Vec3R> bbox = densityFloatGrid->worldBBox();
        bounds = Bounds3f(Point3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                          Point3f(bbox.max()[0], bbox.max()[1], bbox.max()[2]));

        if (temperatureGrid) {
            temperatureFloatGrid = temperatureGrid.grid<float>();
            float minTemperature, maxTemperature;
            temperatureFloatGrid->tree().extrema(minTemperature, maxTemperature);
            LOG_VERBOSE("Max temperature: %f", maxTemperature);

            nanovdb::BBox<nanovdb::Vec3R> bbox = temperatureFloatGrid->worldBBox();
            bounds = Union(
                bounds, Bounds3f(Point3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                                 Point3f(bbox.max()[0], bbox.max()[1], bbox.max()[2])));
        }
    }

    PBRT_CPU_GPU
    const Bounds3f &Bounds() const { return bounds; }

    PBRT_CPU_GPU
    bool IsEmissive() const { return temperatureFloatGrid && LeScale > 0; }

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

    pstd::vector<Float> GetMaxDensityGrid(Allocator alloc, Point3i *res) const {
#if 0
    // For debugging: single, medium-wide majorant...
    *res = Point3i(1, 1, 1);
    Float minDensity, maxDensity;
    densityFloatGrid->tree().extrema(minDensity, maxDensity);
    return pstd::vector<Float>(1, maxDensity, alloc);
#else
        *res = Point3i(64, 64, 64);

        LOG_VERBOSE("Starting nanovdb grid GetMaxDensityGrid()");

        pstd::vector<Float> maxGrid(res->x * res->y * res->z, 0.f, alloc);

        ParallelFor(0, maxGrid.size(), [&](size_t index) {
            // Indices into maxGrid
            int x = index % res->x;
            int y = (index / res->x) % res->y;
            int z = index / (res->x * res->y);
            CHECK_EQ(index, x + res->x * (y + res->y * z));

            // World (aka medium) space bounds of this max grid cell
            Bounds3f wb(bounds.Lerp(Point3f(Float(x) / res->x, Float(y) / res->y,
                                            Float(z) / res->z)),
                        bounds.Lerp(Point3f(Float(x + 1) / res->x, Float(y + 1) / res->y,
                                            Float(z + 1) / res->z)));

            // Compute corresponding NanoVDB index-space bounds in floating-point.
            nanovdb::Vec3R i0 = densityFloatGrid->worldToIndexF(
                nanovdb::Vec3R(wb.pMin.x, wb.pMin.y, wb.pMin.z));
            nanovdb::Vec3R i1 = densityFloatGrid->worldToIndexF(
                nanovdb::Vec3R(wb.pMax.x, wb.pMax.y, wb.pMax.z));

            // Now find integer index-space bounds, accounting for both
            // filtering and the overall index bounding box.
            auto bbox = densityFloatGrid->indexBBox();
            Float delta = 1.f;  // Filter slop
            int nx0 = std::max(int(i0[0] - delta), bbox.min()[0]);
            int nx1 = std::min(int(i1[0] + delta), bbox.max()[0]);
            int ny0 = std::max(int(i0[1] - delta), bbox.min()[1]);
            int ny1 = std::min(int(i1[1] + delta), bbox.max()[1]);
            int nz0 = std::max(int(i0[2] - delta), bbox.min()[2]);
            int nz1 = std::min(int(i1[2] + delta), bbox.max()[2]);

            float maxValue = 0;
            auto accessor = densityFloatGrid->getAccessor();
            // Apparently nanovdb integer bounding boxes are inclusive on
            // the upper end...
            for (int nz = nz0; nz <= nz1; ++nz)
                for (int ny = ny0; ny <= ny1; ++ny)
                    for (int nx = nx0; nx <= nx1; ++nx)
                        maxValue = std::max(maxValue, accessor.getValue({nx, ny, nz}));

            // Only write into maxGrid once when we're done to minimize
            // cache thrashing..
            maxGrid[index] = maxValue;
        });

        LOG_VERBOSE("Finished nanovdb grid GetMaxDensityGrid()");
        return maxGrid;
#endif
    }

    PBRT_CPU_GPU
    MediumDensity Density(Point3f p, const SampledWavelengths &lambda) const {
        nanovdb::Vec3<float> pIndex =
            densityFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));
        using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        Float density = Sampler(densityFloatGrid->tree())(pIndex);
        return MediumDensity(density);
    }

  private:
    // NanoVDBMediumProvider Private Members
    Bounds3f bounds;
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
