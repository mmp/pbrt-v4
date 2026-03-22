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
// appear in several uses (different meshes / instances / materials); merge or scan them
// together when computing a conservative maximum. Future SIMD paths can batch many
// positions per use (e.g. SoA layout) while still treating uses as separate buckets when
// UV layouts or mappings differ.
struct ImageTextureGeometryUse {
    std::string resolvedImageFilename;
    // Triangle corner positions in render space (stub; extend with UVs, normals, mapping).
    std::vector<Point3f> positionsRenderSpace;
};

// Placeholder for ray-differential-style analysis: returns box-filter halvings to apply
// before building the mip pyramid for this texture (same units as --skipmip).
int ComputeImageTextureMipDownsizeStepsFromPreprocess(
    const Camera &camera, const std::vector<ImageTextureGeometryUse> &usesForTexture);

// Clears prior overrides, then assigns per-file mip downsize steps before image loads.
// No-op when --skipmip is off (aside from clearing stale overrides).
void RunImageTextureMipPreprocess(BasicScene &scene, const Camera &camera);

}  // namespace pbrt

#endif  // PBRT_TEXTURE_MIP_PREPROCESS_H
