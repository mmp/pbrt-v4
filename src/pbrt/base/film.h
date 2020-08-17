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

// FilmHandle Definition
class FilmHandle : public TaggedPointer<RGBFilm, GBufferFilm> {
  public:
    // Film Interface
    PBRT_CPU_GPU inline SampledWavelengths SampleWavelengths(Float u) const;

    PBRT_CPU_GPU inline void AddSample(const Point2i &pFilm, SampledSpectrum L,
                                       const SampledWavelengths &lambda,
                                       const VisibleSurface *visibleSurface,
                                       Float weight);

    PBRT_CPU_GPU
    bool UsesVisibleSurface() const;

    PBRT_CPU_GPU
    void AddSplat(const Point2f &p, SampledSpectrum v, const SampledWavelengths &lambda);

    PBRT_CPU_GPU inline Point2i FullResolution() const;
    PBRT_CPU_GPU inline Float Diagonal() const;
    PBRT_CPU_GPU inline Bounds2i PixelBounds() const;

    PBRT_CPU_GPU
    RGB GetPixelRGB(const Point2i &p, Float splatScale = 1) const;
    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    using TaggedPointer::TaggedPointer;

    static FilmHandle Create(const std::string &name,
                             const ParameterDictionary &parameters, const FileLoc *loc,
                             FilterHandle filter, Allocator alloc);

    PBRT_CPU_GPU inline FilterHandle GetFilter() const;

    std::string GetFilename() const;

    std::string ToString() const;

    PBRT_CPU_GPU inline Bounds2f SampleBounds() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_FILM_H
