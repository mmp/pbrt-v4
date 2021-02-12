// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_LIGHT_H
#define PBRT_BASE_LIGHT_H

#include <pbrt/pbrt.h>

#include <pbrt/base/medium.h>
#include <pbrt/base/shape.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

// LightType Definition
enum class LightType { DeltaPosition, DeltaDirection, Area, Infinite };

// LightSamplingMode Definition
enum class LightSamplingMode { WithMIS, WithoutMIS };

class PointLight;
class DistantLight;
class ProjectionLight;
class GoniometricLight;
class DiffuseAreaLight;
class UniformInfiniteLight;
class ImageInfiniteLight;
class PortalImageInfiniteLight;
class SpotLight;

class LightSampleContext;
class LightBounds;
class CompactLightBounds;
struct LightLiSample;
struct LightLeSample;

// Light Definition
class Light : public TaggedPointer<  // Light Source Types
                  PointLight, DistantLight, ProjectionLight, GoniometricLight, SpotLight,
                  DiffuseAreaLight, UniformInfiniteLight, ImageInfiniteLight,
                  PortalImageInfiniteLight

                  > {
  public:
    // Light Interface
    using TaggedPointer::TaggedPointer;

    static Light Create(const std::string &name, const ParameterDictionary &parameters,
                        const Transform &renderFromLight,
                        const CameraTransform &cameraTransform, Medium outsideMedium,
                        const FileLoc *loc, Allocator alloc);
    static Light CreateArea(const std::string &name,
                            const ParameterDictionary &parameters,
                            const Transform &renderFromLight,
                            const MediumInterface &mediumInterface, const Shape shape,
                            const FileLoc *loc, Allocator alloc);

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU inline LightType Type() const;

    PBRT_CPU_GPU inline pstd::optional<LightLiSample> SampleLi(
        LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
        LightSamplingMode mode = LightSamplingMode::WithoutMIS) const;

    PBRT_CPU_GPU inline Float PDF_Li(
        LightSampleContext ctx, Vector3f wi,
        LightSamplingMode mode = LightSamplingMode::WithoutMIS) const;

    std::string ToString() const;

    // AreaLights only
    PBRT_CPU_GPU inline SampledSpectrum L(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                                          const SampledWavelengths &lambda) const;

    // InfiniteLights only
    PBRT_CPU_GPU inline SampledSpectrum Le(const Ray &ray,
                                           const SampledWavelengths &lambda) const;

    void Preprocess(const Bounds3f &sceneBounds);

    pstd::optional<LightBounds> Bounds() const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;

    PBRT_CPU_GPU
    void PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const;

    // AreaLights only
    PBRT_CPU_GPU
    void PDF_Le(const Interaction &intr, Vector3f &w, Float *pdfPos, Float *pdfDir) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHT_H
