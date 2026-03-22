// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_MIPMAP_H
#define PBRT_UTIL_MIPMAP_H

#include <pbrt/pbrt.h>

#include <pbrt/util/image.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <memory>
#include <string>
#include <vector>

namespace pbrt {

// FilterFunction Definition
enum class FilterFunction { Point, Bilinear, Trilinear, EWA };

inline pstd::optional<FilterFunction> ParseFilter(const std::string &f) {
    if (f == "ewa" || f == "EWA")
        return FilterFunction::EWA;
    else if (f == "trilinear")
        return FilterFunction::Trilinear;
    else if (f == "bilinear")
        return FilterFunction::Bilinear;
    else if (f == "point")
        return FilterFunction::Point;
    else
        return {};
}

std::string ToString(FilterFunction f);

// Per-file mip downsize steps for --skipmip (each step is one 2× box-filter halving before the
// mip pyramid). When --skipmip is off, always 0. When on, uses preprocess overrides if set,
// otherwise kDefaultImageTextureSkipMipLevelsWhenSkipMipEnabled in mipmap.cpp.
void ClearImageTextureMipDownsizeOverrides();
void SetImageTextureMipDownsizeOverrideForFile(const std::string &resolvedFilename, int steps);
int ImageTextureMipDownsizeStepsForFile(const std::string &resolvedFilename);

// Continuous EWA LOD matching MIPMap::Filter (EWA branch): level 0 is finest. Returns a value
// >= 0; use with pyramidLevels from MipmapPyramidLevelsForImageResolution.
Float EWAContinuousLOD(Vector2f dst0, Vector2f dst1, Float maxAnisotropy, int pyramidLevels);

// Continuous LOD for imagemap filtering (EWA or non-EWA mip selection in MIPMap::Filter).
Float ImageTextureContinuousLOD(FilterFunction filter, Vector2f dst0, Vector2f dst1,
                                Float maxAnisotropy, int pyramidLevels);

// Pyramid level count after Image::GeneratePyramid's power-of-two resize (matches runtime).
int MipmapPyramidLevelsForImageResolution(Point2i resolution);

// MIPMapFilterOptions Definition
struct MIPMapFilterOptions {
    FilterFunction filter = FilterFunction::EWA;
    Float maxAnisotropy = 8.f;
    bool operator<(MIPMapFilterOptions o) const {
        return std::tie(filter, maxAnisotropy) < std::tie(o.filter, o.maxAnisotropy);
    }
    std::string ToString() const;
};

// MIPMap Definition
class MIPMap {
  public:
    // MIPMap Public Methods
    MIPMap(Image image, const RGBColorSpace *colorSpace, WrapMode wrapMode, Allocator alloc,
           const MIPMapFilterOptions &options, int baseMipDownsizeSteps);
    static MIPMap *CreateFromFile(const std::string &filename,
                                  const MIPMapFilterOptions &options, WrapMode wrapMode,
                                  ColorEncoding encoding, Allocator alloc,
                                  int baseMipDownsizeSteps);

    template <typename T>
    T Filter(Point2f st, Vector2f dstdx, Vector2f dstdy) const;

    std::string ToString() const;

    Point2i LevelResolution(int level) const {
        CHECK(level >= 0 && level < pyramid.size());
        return pyramid[level].Resolution();
    }
    int Levels() const { return int(pyramid.size()); }
    const RGBColorSpace *GetRGBColorSpace() const { return colorSpace; }
    const Image &GetLevel(int level) const { return pyramid[level]; }

    int64_t TotalBytesUsed() const;

  private:
    // MIPMap Private Methods
    template <typename T>
    T Texel(int level, Point2i st) const;
    template <typename T>
    T Bilerp(int level, Point2f st) const;
    template <typename T>
    T EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const;

    // MIPMap Private Members
    pstd::vector<Image> pyramid;
    const RGBColorSpace *colorSpace;
    WrapMode wrapMode;
    MIPMapFilterOptions options;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MIPMAP_H
