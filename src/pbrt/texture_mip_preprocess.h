// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_TEXTURE_MIP_PREPROCESS_H
#define PBRT_TEXTURE_MIP_PREPROCESS_H

#include <pbrt/pbrt.h>

#include <pbrt/util/mipmap.h>
#include <pbrt/util/vecmath.h>

#include <string>
#include <vector>

namespace pbrt {

class BasicScene;
class Camera;

// One shaded triangle in render space with parametric UVs (matches TriangleMesh usage).
struct ImageTextureMeshTriangle {
    Point3f p0, p1, p2;
    Point2f uv0, uv1, uv2;
};

// Geometry using one named imagemap on diffuse materials (UV mapping only for now).
struct ImageTextureGeometryUse {
    std::string resolvedImageFilename;
    std::string geometryDebugLabel;
    std::vector<ImageTextureMeshTriangle> triangles;
    Float su = 1, sv = 1, du = 0, dv = 0;
    Float maxAnisotropy = 8.f;
    FilterFunction filter = FilterFunction::Bilinear;
};

// Per-geometry safe downsizes = floor(min primary-ray continuous LOD); texture override =
// min over geometries. Primary visibility only; UV imagemap on reflectance for diffuse /
// coateddiffuse / diffusetransmission and mix thereof; trianglemesh + plymesh; includes
// ObjectInstance placements (transformed into render space).
int ComputeImageTextureSafeDownsizesFromPreprocess(
    const Camera &camera, int samplesPerPixel,
    const std::vector<ImageTextureGeometryUse> &usesForTexture, int mipmapPyramidLevels,
    Allocator alloc);

// Clears prior overrides, then assigns per-file safe downsizes before image loads.
// No-op when --skipmip is off (aside from clearing stale overrides).
void RunImageTextureMipPreprocess(BasicScene &scene, const Camera &camera,
                                   int samplesPerPixel);

}  // namespace pbrt

#endif  // PBRT_TEXTURE_MIP_PREPROCESS_H
