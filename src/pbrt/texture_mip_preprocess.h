// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_TEXTURE_MIP_PREPROCESS_H
#define PBRT_TEXTURE_MIP_PREPROCESS_H

#include <pbrt/pbrt.h>

#include <pbrt/util/vecmath.h>

#include <string>
#include <vector>

namespace pbrt {

class BasicScene;
class Camera;

// Geometry that contributes to mip-level bounds for one image texture. A single file may
// appear in several uses (different meshes / instances / materials); the texture-wide
// safe downsizes count (box-filter halvings before the mip pyramid) is the minimum over
// those uses so no shared image is downsampled more than the tightest geometry allows.
// Future SIMD paths can batch many positions per use (e.g. SoA layout) while still
// treating uses as separate buckets when UV layouts or mappings differ.
struct ImageTextureGeometryUse {
    std::string resolvedImageFilename;
    // Identifies this mesh / instance / material use for debug logs (shape name, index, …).
    std::string geometryDebugLabel;
    // Triangle corner positions in render space (stub; extend with UVs, normals, mapping).
    std::vector<Point3f> positionsRenderSpace;
};

// Per-geometry safe downsizes are combined with min (shared texture = tightest geometry
// wins). Stub ignores camera/geometry data (same units as --skipmip).
int ComputeImageTextureSafeDownsizesFromPreprocess(
    const Camera &camera, const std::vector<ImageTextureGeometryUse> &usesForTexture);

// Clears prior overrides, then assigns per-file safe downsizes before image loads.
// No-op when --skipmip is off (aside from clearing stale overrides).
void RunImageTextureMipPreprocess(BasicScene &scene, const Camera &camera);

}  // namespace pbrt

#endif  // PBRT_TEXTURE_MIP_PREPROCESS_H
