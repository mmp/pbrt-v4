// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_FILM_H
#define PBRT_BASE_FILM_H

#include <pbrt/pbrt.h>

#include <pbrt/base/filter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

class VisibleSurface;
class RGBFilm;
class GBufferFilm;
class PixelSensor;

// Film Definition
class Film : public TaggedPointer<RGBFilm, GBufferFilm> {
  public:
    // Film Interface
    PBRT_CPU_GPU inline void AddSample(Point2i pFilm, SampledSpectrum L,
                                       const SampledWavelengths &lambda,
                                       const VisibleSurface *visibleSurface,
                                       Float weight);

    PBRT_CPU_GPU inline Bounds2f SampleBounds() const;

    PBRT_CPU_GPU
    bool UsesVisibleSurface() const;

    PBRT_CPU_GPU
    void AddSplat(Point2f p, SampledSpectrum v, const SampledWavelengths &lambda);

    PBRT_CPU_GPU inline SampledWavelengths SampleWavelengths(Float u) const;

    PBRT_CPU_GPU inline Point2i FullResolution() const;
    PBRT_CPU_GPU inline Bounds2i PixelBounds() const;
    PBRT_CPU_GPU inline Float Diagonal() const;

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);

    PBRT_CPU_GPU inline RGB ToOutputRGB(SampledSpectrum L,
                                        const SampledWavelengths &lambda) const;

    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);
    PBRT_CPU_GPU
    RGB GetPixelRGB(const Point2i &p, Float splatScale = 1) const;

    PBRT_CPU_GPU inline Filter GetFilter() const;
    PBRT_CPU_GPU inline const PixelSensor *GetPixelSensor() const;
    std::string GetFilename() const;

    using TaggedPointer::TaggedPointer;

    static Film Create(const std::string &name, const ParameterDictionary &parameters,
                       Float exposureTime, Filter filter, const FileLoc *loc,
                       Allocator alloc);

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_FILM_H
