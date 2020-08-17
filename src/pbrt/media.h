// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_MEDIA_H
#define PBRT_MEDIA_H

#include <pbrt/pbrt.h>

#include <pbrt/base/medium.h>
#include <pbrt/interaction.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>

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
    Float p(const Vector3f &wo, const Vector3f &wi) const {
        return HenyeyGreenstein(Dot(wo, wi), g);
    }

    PBRT_CPU_GPU
    PhaseFunctionSample Sample_p(const Vector3f &wo, const Point2f &u) const {
        Float pdf;
        Vector3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{pdf, wi, pdf};
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi) const { return p(wo, wi); }

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
    HomogeneousMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, SpectrumHandle Le,
                      Float g, Allocator alloc)
        : sigma_a_spec(sigma_a, alloc),
          sigma_s_spec(sigma_s, alloc),
          Le_spec(Le, alloc),
          phase(g) {}

    static HomogeneousMedium *Create(const ParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    template <typename F>
    PBRT_CPU_GPU void SampleTmaj(const Ray &ray, Float tMax, RNG &rng,
                                 const SampledWavelengths &lambda, F callback) const {
        // Compute normalized ray in medium, _rayp_
        tMax *= Length(ray.d);
        Ray rayp(ray.o, Normalize(ray.d));

        // Compute _SampledSpectrum_ scattering properties for medium
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        SampledSpectrum sigma_maj = sigma_t;

        // Sample exponential funciton to find _t_ for scattering event
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
            MediumInteraction intr(rayp(t), -rayp.d, ray.time, sigma_a, sigma_s,
                                   sigma_maj, Le, this, &phase);
            callback(MediumSample(intr, Tmaj));
        }
    }

    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    std::string ToString() const;

  private:
    // HomogeneousMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec, Le_spec;
    HGPhaseFunction phase;
};

// GridDensityMedium Definition
class GridDensityMedium {
  public:
    // GridDensityMedium Public Methods
    GridDensityMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, SpectrumHandle Le,
                      Float g, const Transform &renderFromMedium,
                      pstd::optional<SampledGrid<Float>> densityGrid,
                      pstd::optional<SampledGrid<RGB>> rgbDensityGrid,
                      const RGBColorSpace *colorSpace, SampledGrid<Float> LeScaleGrid,
                      Allocator alloc);

    static GridDensityMedium *Create(const ParameterDictionary &parameters,
                                     const Transform &renderFromMedium,
                                     const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    template <typename F>
    PBRT_CPU_GPU void SampleTmaj(const Ray &rRender, Float raytMax, RNG &rng,
                                 const SampledWavelengths &lambda, F callback) const {
        // Transform ray to grid density's space and compute bounds overlap
        raytMax *= Length(rRender.d);
        Ray ray = mediumFromRender(Ray(rRender.o, Normalize(rRender.d)), &raytMax);
        const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
        Float tMin, tMax;
        if (!b.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return;
        DCHECK_LE(tMax, raytMax);

        // Sample spectra for grid medium scattering
        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;

        // Set up 3D DDA for ray through grid
        Point3f gridIntersect = ray(tMin);
        float nextCrossingT[3], deltaT[3];
        int step[3], voxelLimit[3], voxel[3];
        for (int axis = 0; axis < 3; ++axis) {
            // Initialize ray stepping parameters for axis
            // Handle negative zero ray direction
            if (ray.d[axis] == -0.f)
                ray.d[axis] = 0.f;

            // Compute current voxel for axis
            voxel[axis] =
                Clamp(gridIntersect[axis] * maxDGridRes[axis], 0, maxDGridRes[axis] - 1);

            if (ray.d[axis] >= 0) {
                // Handle ray with positive direction for voxel stepping
                nextCrossingT[axis] = tMin + (Float(voxel[axis] + 1) / maxDGridRes[axis] -
                                              gridIntersect[axis]) /
                                                 ray.d[axis];
                deltaT[axis] = 1 / (ray.d[axis] * maxDGridRes[axis]);
                step[axis] = 1;
                voxelLimit[axis] = maxDGridRes[axis];

            } else {
                // Handle ray with negative direction for voxel stepping
                nextCrossingT[axis] = tMin + (Float(voxel[axis]) / maxDGridRes[axis] -
                                              gridIntersect[axis]) /
                                                 ray.d[axis];
                deltaT[axis] = -1 / (ray.d[axis] * maxDGridRes[axis]);
                step[axis] = -1;
                voxelLimit[axis] = -1;
            }
        }

        // Walk ray through maximum density grid and sample scattering
        Float t0 = tMin, u = rng.Uniform<Float>();
        while (true) {
            // Find _stepAxis_ for stepping to next voxel and exit point _t1_
            int bits = ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
                       ((nextCrossingT[0] < nextCrossingT[2]) << 1) +
                       ((nextCrossingT[1] < nextCrossingT[2]));
            const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
            int stepAxis = cmpToAxis[bits];
            Float t1 = nextCrossingT[stepAxis];

            // Sample volume scattering in current voxel
            // Get _maxDensity_ for current voxel and compute _sigma\_maj_
            int offset = voxel[0] + maxDGridRes.x * (voxel[1] + maxDGridRes.y * voxel[2]);
            Float maxDensity = maxDensityGrid[offset];
            SampledSpectrum sigma_maj(sigma_t * maxDensity);

            if (sigma_maj[0] > 0) {
                while (true) {
                    // Sample medium in current voxel
                    // Compute _uEnd_ for exiting voxel and continue if no valid event
                    Float uEnd = InvertExponentialSample(t1 - t0, sigma_maj[0]);
                    if (u >= uEnd) {
                        u = (u - uEnd) / (1 - uEnd);
                        goto advance;
                    }

                    // Sample _t_ for scattering event and check validity
                    Float t = t0 + SampleExponential(u, sigma_maj[0]);
                    if (t >= tMax) {
                        callback(MediumSample(SampledSpectrum(1.f)));
                        return;
                    }

                    // Report scattering event in grid to callback function
                    Point3f p = ray(t);
                    SampledSpectrum Tmaj = FastExp(-sigma_maj * (t - t0));
                    // Compute _density_ at sampled point in grid
                    SampledSpectrum density;
                    if (densityGrid)
                        density = SampledSpectrum(densityGrid->Lookup(p));
                    else {
                        RGB rgb = rgbDensityGrid->Lookup(p);
                        density = RGBSpectrum(*colorSpace, rgb).Sample(lambda);
                    }

                    MediumInteraction intr(renderFromMedium(p), -Normalize(rRender.d),
                                           rRender.time, sigma_a * density,
                                           sigma_s * density, sigma_maj, Le(p, lambda),
                                           this, &phase);
                    if (!callback(MediumSample(intr, Tmaj)))
                        return;

                    // Update _u_ and _t0_ after grid medium event
                    u = rng.Uniform<Float>();
                    t0 = t;
                }
            }

        advance:
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

  private:
    // GridDensityMedium Private Methods
    PBRT_CPU_GPU
    SampledSpectrum Le(const Point3f &p, const SampledWavelengths &lambda) const {
        return Le_spec.Sample(lambda) * LeScaleGrid.Lookup(p);
    }

    // GridDensityMedium Private Members
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    HGPhaseFunction phase;
    Transform mediumFromRender, renderFromMedium;
    pstd::optional<SampledGrid<Float>> densityGrid;
    pstd::optional<SampledGrid<RGB>> rgbDensityGrid;
    const RGBColorSpace *colorSpace;
    DenselySampledSpectrum Le_spec;
    SampledGrid<Float> LeScaleGrid;
    pstd::vector<Float> maxDensityGrid;
    Point3i maxDGridRes;
};

inline Float PhaseFunctionHandle::p(const Vector3f &wo, const Vector3f &wi) const {
    auto p = [&](auto ptr) { return ptr->p(wo, wi); };
    return Dispatch(p);
}

inline PhaseFunctionSample PhaseFunctionHandle::Sample_p(const Vector3f &wo,
                                                         const Point2f &u) const {
    auto sample = [&](auto ptr) { return ptr->Sample_p(wo, u); };
    return Dispatch(sample);
}

inline Float PhaseFunctionHandle::PDF(const Vector3f &wo, const Vector3f &wi) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(wo, wi); };
    return Dispatch(pdf);
}

template <typename F>
void MediumHandle::SampleTmaj(const Ray &ray, Float tMax, RNG &rng,
                              const SampledWavelengths &lambda, F func) const {
    auto sampletn = [&](auto ptr) { ptr->SampleTmaj(ray, tMax, rng, lambda, func); };
    Dispatch(sampletn);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_H
