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

#include <memory>
#include <vector>

namespace pbrt {

// Media Function Declarations
bool GetMediumScatteringProperties(const std::string &name, SpectrumHandle *sigma_a,
                                   SpectrumHandle *sigma_s, Allocator alloc);

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

    std::string ToString() const;

  private:
    // HGPhaseFunction Private Members
    Float g;
};

// MediumSample Definition
struct MediumSample {
    // MediumSample Public Methods
    MediumSample() = default;
    PBRT_CPU_GPU
    explicit MediumSample(const SampledSpectrum &Tmaj) : Tmaj(Tmaj) {}
    PBRT_CPU_GPU
    MediumSample(const MediumInteraction &intr, const SampledSpectrum &Tmaj)
        : intr(intr), Tmaj(Tmaj) {}

    std::string ToString() const;

    pstd::optional<MediumInteraction> intr;
    SampledSpectrum Tmaj = SampledSpectrum(1.f);
};

// HomogeneousMedium Definition
class HomogeneousMedium {
  public:
    // HomogeneousMedium Public Methods
    HomogeneousMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, Float sigScale,
                      SpectrumHandle Le, Float g, Allocator alloc)
        : sigma_a_spec(sigma_a, alloc),
          sigma_s_spec(sigma_s, alloc),
          sigScale(sigScale),
          Le_spec(Le, alloc),
          phase(g) {}

    static HomogeneousMedium *Create(const ParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    template <typename F>
    PBRT_CPU_GPU void SampleTmaj(Ray ray, Float tMax, RNG &rng,
                                 const SampledWavelengths &lambda, F callback) const {
        // Normalize ray direction for homogeneous medium sampling
        tMax *= Length(ray.d);
        ray.d = Normalize(ray.d);

        // Compute _SampledSpectrum_ scattering properties for medium
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        SampledSpectrum sigma_maj = sigma_t;

        // Sample exponential function to find _t_ for scattering event
        if (sigma_maj[0] == 0)
            return;
        Float u = rng.Uniform<Float>();
        Float t = SampleExponential(u, sigma_maj[0]);

        if (t >= tMax) {
            // Return transmittance to medium exit point
            callback(MediumSample(FastExp(-tMax * sigma_maj)));

        } else {
            // Report scattering event in homogeneous medium
            SampledSpectrum Tmaj = FastExp(-t * sigma_maj);
            SampledSpectrum Le = Le_spec.Sample(lambda);
            MediumInteraction intr(ray(t), -ray.d, ray.time, sigma_a, sigma_s, sigma_maj,
                                   Le, this, &phase);
            callback(MediumSample(intr, Tmaj));
        }
    }

    std::string ToString() const;

  private:
    // HomogeneousMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec, Le_spec;
    Float sigScale;
    HGPhaseFunction phase;
};

// CuboidMedium Definition
template <typename Provider>
class CuboidMedium {
  public:
    // CuboidMedium Public Methods
    CuboidMedium(const Provider *provider, SpectrumHandle sigma_a, SpectrumHandle sigma_s,
                 Float sigScale, Float g, const Transform &renderFromMedium,
                 Allocator alloc)
        : provider(provider),
          mediumBounds(provider->Bounds()),
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
        return StringPrintf("[ CuboidMedium provider: %s mediumBounds: %s "
                            "sigma_a_spec: %s sigma_s_spec: %s sigScale: %f phase: %s "
                            "maxDensityGrid: %s gridResolution: %s ]",
                            *provider, mediumBounds, sigma_a_spec, sigma_s_spec, sigScale,
                            phase, maxDensityGrid, gridResolution);
    }

    bool IsEmissive() const { return provider->IsEmissive(); }

    template <typename F>
    PBRT_CPU_GPU void SampleTmaj(Ray rRender, Float raytMax, RNG &rng,
                                 const SampledWavelengths &lambda, F callback) const {
        // Transform ray to grid density's space and compute bounds overlap
        Ray ray = renderFromMedium.ApplyInverse(rRender, &raytMax);
        raytMax *= Length(ray.d);
        ray.d = Normalize(ray.d);
        Float tMin, tMax;
        if (!mediumBounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return;
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid medium scattering
        SampledSpectrum sigma_a = sigScale * sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigScale * sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;

        // Set up 3D DDA for ray through grid
        Vector3f diag = mediumBounds.Diagonal();
        Ray rayGrid(Point3f(mediumBounds.Offset(ray.o)),
                    Vector3f(ray.d.x / diag.x, ray.d.y / diag.y, ray.d.z / diag.z));
        Point3f gridIntersect = rayGrid(tMin);
        Float nextCrossingT[3], deltaT[3];
        int step[3], voxelLimit[3], voxel[3];
        for (int axis = 0; axis < 3; ++axis) {
            // Initialize ray stepping parameters for axis
            // Compute current voxel for axis and handle negative zero direction
            voxel[axis] = Clamp(gridIntersect[axis] * gridResolution[axis], 0,
                                gridResolution[axis] - 1);
            if (rayGrid.d[axis] == -0.f)
                rayGrid.d[axis] = 0.f;

            if (rayGrid.d[axis] >= 0) {
                // Handle ray with positive direction for voxel stepping
                Float nextVoxelPos = Float(voxel[axis] + 1) / gridResolution[axis];
                nextCrossingT[axis] =
                    tMin + (nextVoxelPos - gridIntersect[axis]) / rayGrid.d[axis];
                deltaT[axis] = 1 / (rayGrid.d[axis] * gridResolution[axis]);
                step[axis] = 1;
                voxelLimit[axis] = gridResolution[axis];

            } else {
                // Handle ray with negative direction for voxel stepping
                Float nextVoxelPos = Float(voxel[axis]) / gridResolution[axis];
                nextCrossingT[axis] =
                    tMin + (nextVoxelPos - gridIntersect[axis]) / rayGrid.d[axis];
                deltaT[axis] = -1 / (rayGrid.d[axis] * gridResolution[axis]);
                step[axis] = -1;
                voxelLimit[axis] = -1;
            }
        }

        // Walk ray through maximum density grid and sample scattering
        Float t0 = tMin;
        while (true) {
            // Find _stepAxis_ for stepping to next voxel and exit point _t1_
            int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
                       ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
                       ((nextCrossingT[1] < nextCrossingT[2]));
            const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
            int stepAxis = cmpToAxis[bits];
            Float t1 = nextCrossingT[stepAxis];

            // Sample volume scattering in current voxel
            // Get _maxDensity_ for current voxel and compute _sigma_maj_
            int offset =
                voxel[0] + gridResolution.x * (voxel[1] + gridResolution.y * voxel[2]);
            Float maxDensity = maxDensityGrid[offset];
            SampledSpectrum sigma_maj(sigma_t * maxDensity);

            if (sigma_maj[0] > 0) {
                while (true) {
                    // Sample medium in current voxel
                    // Sample _t_ for scattering event and check validity
                    Float u = rng.Uniform<Float>();
                    Float t = t0 + SampleExponential(u, sigma_maj[0]);
                    if (t >= t1)
                        break;

                    if (t < tMax) {
                        // Compute medium properties at sampled point in grid
                        SampledSpectrum Tmaj = FastExp(-sigma_maj * (t - t0));
                        Point3f p = ray(t);
                        SampledSpectrum d = provider->Density(p, lambda);
                        SampledSpectrum Le = provider->Le(p, lambda);
                        SampledSpectrum sigmap_a = sigma_a * d, sigmap_s = sigma_s * d;

                        // Report scattering event in grid to callback function
                        Point3f pRender = renderFromMedium(p);
                        MediumInteraction intr(pRender, -Normalize(rRender.d),
                                               rRender.time, sigmap_a, sigmap_s,
                                               sigma_maj, Le, this, &phase);
                        if (!callback(MediumSample(intr, Tmaj)))
                            return;
                    }
                    // Update _t0_ after medium interaction
                    t0 = t;
                }
            }

            // Advance to next voxel in maximum density grid
            if (nextCrossingT[stepAxis] > tMax)
                return;
            voxel[stepAxis] += step[stepAxis];
            if (voxel[stepAxis] == voxelLimit[stepAxis])
                return;
            nextCrossingT[stepAxis] += deltaT[stepAxis];
            t0 = t1;
        }
    }

    static CuboidMedium<Provider> *Create(const Provider *provider,
                                          const ParameterDictionary &parameters,
                                          const Transform &renderFromMedium,
                                          const FileLoc *loc, Allocator alloc) {
        SpectrumHandle sig_a = nullptr, sig_s = nullptr;
        std::string preset = parameters.GetOneString("preset", "");
        if (!preset.empty()) {
            if (!GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc))
                Warning(loc, "Material preset \"%s\" not found.", preset);
        }

        if (sig_a == nullptr) {
            sig_a = parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::General,
                                              alloc);
            if (sig_a == nullptr)
                sig_a = alloc.new_object<ConstantSpectrum>(1.f);
        }
        if (sig_s == nullptr) {
            sig_s = parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::General,
                                              alloc);
            if (sig_s == nullptr)
                sig_s = alloc.new_object<ConstantSpectrum>(1.f);
        }

        Float sigScale = parameters.GetOneFloat("scale", 1.f);

        Float g = parameters.GetOneFloat("g", 0.0f);

        return alloc.new_object<CuboidMedium<Provider>>(provider, sig_a, sig_s, sigScale,
                                                        g, renderFromMedium, alloc);
    }

  private:
    // CuboidMedium Private Members
    const Provider *provider;
    Bounds3f mediumBounds;
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
                              pstd::optional<SampledGrid<Float>> densityGrid,
                              pstd::optional<SampledGrid<RGB>> rgbDensityGrid,
                              const RGBColorSpace *colorSpace, SpectrumHandle Le,
                              SampledGrid<Float> LeScaleGrid, Allocator alloc);

    static UniformGridMediumProvider *Create(const ParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    const Bounds3f &Bounds() const { return bounds; }

    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    PBRT_CPU_GPU
    SampledSpectrum Le(Point3f p, const SampledWavelengths &lambda) const {
        Point3f pp = Point3f(bounds.Offset(p));
        return Le_spec.Sample(lambda) * LeScaleGrid.Lookup(pp);
    }

    PBRT_CPU_GPU
    SampledSpectrum Density(const Point3f &p, const SampledWavelengths &lambda) const {
        Point3f pp = Point3f(bounds.Offset(p));
        if (densityGrid)
            return SampledSpectrum(densityGrid->Lookup(pp));
        else {
            RGB rgb = rgbDensityGrid->Lookup(pp);
            return RGBSpectrum(*colorSpace, rgb).Sample(lambda);
        }
    }

    pstd::vector<Float> GetMaxDensityGrid(Allocator alloc, Point3i *res) const {
        // Set _gridResolution_ and allocate _maxGrid_
        *res = Point3i(4, 4, 4);
        pstd::vector<Float> maxGrid(alloc);
        maxGrid.resize(res->x * res->y * res->z);

        // Define _getMaxDensity_ lambda
        auto getMaxDensity = [&](const Bounds3f &bounds) -> Float {
            if (densityGrid)
                return densityGrid->MaximumValue(bounds);
            else {
                // Compute maximum density of RGB density over _bounds_
                int nx = rgbDensityGrid->xSize();
                int ny = rgbDensityGrid->ySize();
                int nz = rgbDensityGrid->zSize();
                Point3f ps[2] = {
                    Point3f(bounds.pMin.x * nx - .5f, bounds.pMin.y * ny - .5f,
                            bounds.pMin.z * nz - .5f),
                    Point3f(bounds.pMax.x * nx - .5f, bounds.pMax.y * ny - .5f,
                            bounds.pMax.z * nz - .5f)};
                Point3i pi[2] = {Max(Point3i(Floor(ps[0])), Point3i(0, 0, 0)),
                                 Min(Point3i(Floor(ps[1])) + Vector3i(1, 1, 1),
                                     Point3i(nx - 1, ny - 1, nz - 1))};

                Float maxDensity = 0;
                for (int z = pi[0].z; z <= pi[1].z; ++z)
                    for (int y = pi[0].y; y <= pi[1].y; ++y)
                        for (int x = pi[0].x; x <= pi[1].x; ++x) {
                            RGB rgb = rgbDensityGrid->Lookup(Point3i(x, y, z));

                            Float maxComponent = std::max({rgb.r, rgb.g, rgb.b});
                            if (maxComponent == 0)
                                continue;

                            RGBSpectrum spec(*colorSpace, rgb);
                            maxDensity = std::max(maxDensity, spec.MaxValue());
                        }

                return maxDensity * 1.025f;
            }
        };

        // Compute maximum density for each _maxGrid_ cell
        int offset = 0;
        for (Float z = 0; z < res->z; ++z)
            for (Float y = 0; y < res->y; ++y)
                for (Float x = 0; x < res->x; ++x) {
                    Bounds3f bounds(
                        Point3f(x / res->x, y / res->y, z / res->z),
                        Point3f((x + 1) / res->x, (y + 1) / res->y, (z + 1) / res->z));
                    maxGrid[offset++] = getMaxDensity(bounds);
                }

        return maxGrid;
    }

  private:
    // UniformGridMediumProvider Private Members
    Bounds3f bounds;
    pstd::optional<SampledGrid<Float>> densityGrid;
    pstd::optional<SampledGrid<RGB>> rgbDensityGrid;
    const RGBColorSpace *colorSpace;
    DenselySampledSpectrum Le_spec;
    SampledGrid<Float> LeScaleGrid;
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
    SampledSpectrum Density(Point3f p, const SampledWavelengths &) const {
        Point3f pp = frequency * p;
        if (wispiness > 0) {
            // Perturb cloud lookup point _pp_ using noise
            Float vomega = .05f * wispiness, vlambda = 10.f;
            for (int i = 0; i < 2; ++i) {
                pp += vomega * DNoise(vlambda * pp);
                vomega *= 0.5f;
                vlambda *= 1.99f;
            }
        }
        // Sum scales of noise to approximate cloud density
        Float d = 0;
        Float omega = .5, lambda = 1.f;
        for (int i = 0; i < 5; ++i) {
            d += omega * Noise(lambda * pp);
            omega *= 0.5f;
            lambda *= 1.99f;
        }

        // Model decrease in density with altitude and return final cloud density
        d = Clamp((1 - p.y) * 4.5f * density * d, 0, 1);
        d += 2 * std::max<Float>(0, .5f - p.y);
        return SampledSpectrum(Clamp(d, 0, 1));
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
        LOG_VERBOSE("this %p alloc ptr %p bytes %d", this, ptr, bytesAllocated);
    }

    const uint8_t *data() const { return ptr; }
    uint8_t *data() { return ptr; }
    uint64_t size() const { return bytesAllocated; }
    bool empty() const { return size() == 0; }

    void clear() {
        LOG_VERBOSE("this %p clear ptr %p bytes %d", this, ptr, bytesAllocated);
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

    NanoVDBMediumProvider(const Bounds3f &bounds, nanovdb::GridHandle<NanoVDBBuffer> dg,
                          nanovdb::GridHandle<NanoVDBBuffer> tg, Float LeScale,
                          Float temperatureCutoff, Float temperatureScale)
        : bounds(bounds),
          densityGrid(std::move(dg)),
          temperatureGrid(std::move(tg)),
          LeScale(LeScale),
          temperatureCutoff(temperatureCutoff),
          temperatureScale(temperatureScale) {
        densityFloatGrid = densityGrid.grid<float>();
        if (temperatureGrid) {
            temperatureFloatGrid = temperatureGrid.grid<float>();
            Float minTemperature, maxTemperature;
            temperatureFloatGrid->tree().extrema(minTemperature, maxTemperature);
            LOG_VERBOSE("Max temperature: %f", maxTemperature);
        }
    }

    PBRT_CPU_GPU
    const Bounds3f &Bounds() const { return bounds; }

    bool IsEmissive() const { return temperatureFloatGrid != nullptr && LeScale > 0; }

    PBRT_CPU_GPU
    SampledSpectrum Le(const Point3f &p, const SampledWavelengths &lambda) const {
        if (!temperatureFloatGrid)
            return SampledSpectrum(0.f);
        nanovdb::Vec3<float> pIndex =
            densityFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));
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

            Float maxValue = 0;
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
    SampledSpectrum Density(const Point3f &p, const SampledWavelengths &lambda) const {
        nanovdb::Vec3<float> pIndex =
            densityFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));
        using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        Float density = Sampler(densityFloatGrid->tree())(pIndex);
        return SampledSpectrum(density);
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

inline Float PhaseFunctionHandle::p(Vector3f wo, Vector3f wi) const {
    auto p = [&](auto ptr) { return ptr->p(wo, wi); };
    return Dispatch(p);
}

inline pstd::optional<PhaseFunctionSample> PhaseFunctionHandle::Sample_p(
    Vector3f wo, Point2f u) const {
    auto sample = [&](auto ptr) { return ptr->Sample_p(wo, u); };
    return Dispatch(sample);
}

inline Float PhaseFunctionHandle::PDF(Vector3f wo, Vector3f wi) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(wo, wi); };
    return Dispatch(pdf);
}

template <typename F>
void MediumHandle::SampleTmaj(Ray ray, Float tMax, RNG &rng,
                              const SampledWavelengths &lambda, F func) const {
    auto sampletn = [&](auto ptr) { ptr->SampleTmaj(ray, tMax, rng, lambda, func); };
    Dispatch(sampletn);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_H
