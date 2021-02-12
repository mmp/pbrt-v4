// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_CAMERA_H
#define PBRT_BASE_CAMERA_H

#include <pbrt/pbrt.h>

#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// Camera Declarations
struct CameraRay;
struct CameraRayDifferential;
struct CameraWiSample;

struct CameraSample;
class CameraTransform;

class PerspectiveCamera;
class OrthographicCamera;
class SphericalCamera;
class RealisticCamera;

// Camera Definition
class Camera : public TaggedPointer<PerspectiveCamera, OrthographicCamera,
                                    SphericalCamera, RealisticCamera> {
  public:
    // Camera Interface
    using TaggedPointer::TaggedPointer;

    static Camera Create(const std::string &name, const ParameterDictionary &parameters,
                         Medium medium, const CameraTransform &cameraTransform, Film film,
                         const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU inline pstd::optional<CameraRay> GenerateRay(
        CameraSample sample, SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraSample sample, SampledWavelengths &lambda) const;

    PBRT_CPU_GPU inline Film GetFilm() const;

    PBRT_CPU_GPU inline Float SampleTime(Float u) const;

    void InitMetadata(ImageMetadata *metadata) const;

    PBRT_CPU_GPU inline const CameraTransform &GetCameraTransform() const;

    PBRT_CPU_GPU
    void ApproximatedPdxy(SurfaceInteraction &si, int samplesPerPixel) const;

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const;

    PBRT_CPU_GPU
    void PDF_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> SampleWi(const Interaction &ref, Point2f u,
                                            SampledWavelengths &lambda) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_CAMERA_H
