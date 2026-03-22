// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/texture_mip_preprocess.h>

#include <pbrt/cameras.h>
#include <pbrt/options.h>
#include <pbrt/scene.h>
#include <pbrt/util/mipmap.h>

namespace pbrt {

int ComputeImageTextureMipDownsizeStepsFromPreprocess(
    const Camera &camera, const std::vector<ImageTextureGeometryUse> &usesForTexture) {
    (void)camera;
    (void)usesForTexture;
    return 1;
}

void RunImageTextureMipPreprocess(BasicScene &scene, const Camera &camera) {
    ClearImageTextureMipDownsizeOverrides();
    if (!Options || !Options->skipMipImageTextures)
        return;

    std::vector<std::string> files = scene.CollectResolvedImageTextureFilenames();
    for (const std::string &fn : files) {
        std::vector<ImageTextureGeometryUse> uses;
        ImageTextureGeometryUse u;
        u.resolvedImageFilename = fn;
        uses.push_back(std::move(u));
        int steps = ComputeImageTextureMipDownsizeStepsFromPreprocess(camera, uses);
        SetImageTextureMipDownsizeOverrideForFile(fn, steps);
    }
}

}  // namespace pbrt
