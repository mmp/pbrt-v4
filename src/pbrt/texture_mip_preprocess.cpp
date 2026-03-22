// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/texture_mip_preprocess.h>

#include <pbrt/cameras.h>
#include <pbrt/options.h>
#include <pbrt/scene.h>
#include <pbrt/util/mipmap.h>
#include <pbrt/util/print.h>

#include <algorithm>
#include <cctype>
#include <limits>

namespace pbrt {

// Log path relative to a .../pbrt-v4-scenes/ root when present (any slash style, case fold).
static std::string ShortScenePathForMipLog(const std::string &fullPath) {
    static constexpr char kMarker[] = "pbrt-v4-scenes";
    const size_t n = fullPath.size(), m = sizeof(kMarker) - 1;
    for (size_t i = 0; i + m <= n; ++i) {
        bool match = true;
        for (size_t j = 0; j < m; ++j) {
            if (std::tolower(static_cast<unsigned char>(fullPath[i + j])) !=
                std::tolower(static_cast<unsigned char>(kMarker[j]))) {
                match = false;
                break;
            }
        }
        if (match) {
            size_t after = i + m;
            while (after < n && (fullPath[after] == '/' || fullPath[after] == '\\'))
                ++after;
            return fullPath.substr(after);
        }
    }
    return fullPath;
}

int ComputeImageTextureSafeDownsizesFromPreprocess(
    const Camera &camera, const std::vector<ImageTextureGeometryUse> &usesForTexture) {
    (void)camera;

    if (usesForTexture.empty())
        return 0;

    // Hardcoded per (texture, geometry) pair until ray-differential logic exists.
    constexpr int kStubPairSafeDownsizes = 1;

    const std::string texLog = ShortScenePathForMipLog(usesForTexture[0].resolvedImageFilename);
    Printf("[mip preprocess] texture \"%s\"\n", texLog);

    // Minimum over geometries: a shared texture must not be downsampled more than the
    // most restrictive use allows (avoid blurring where one surface still needs fine mips).
    int textureMinSafeDownsizes = std::numeric_limits<int>::max();
    for (const ImageTextureGeometryUse &use : usesForTexture) {
        const std::string &geom =
            use.geometryDebugLabel.empty() ? std::string("(no label)") : use.geometryDebugLabel;
        const int pairSafeDownsizes = kStubPairSafeDownsizes;
        Printf("  geometry \"%s\" -> safe downsizes %d\n", geom, pairSafeDownsizes);
        textureMinSafeDownsizes = std::min(textureMinSafeDownsizes, pairSafeDownsizes);
    }

    return std::max(0, textureMinSafeDownsizes);
}

void RunImageTextureMipPreprocess(BasicScene &scene, const Camera &camera) {
    ClearImageTextureMipDownsizeOverrides();
    if (!Options || !Options->skipMipImageTextures)
        return;

    std::vector<std::string> files = scene.CollectResolvedImageTextureFilenames();
    for (const std::string &fn : files) {
        std::vector<ImageTextureGeometryUse> uses;
        // Stub: two placeholder geometries per file so logs show one line per (texture, geometry).
        for (int i = 0; i < 2; ++i) {
            ImageTextureGeometryUse u;
            u.resolvedImageFilename = fn;
            u.geometryDebugLabel = StringPrintf("stub_geometry_%d", i);
            uses.push_back(std::move(u));
        }
        int safeDownsizes = ComputeImageTextureSafeDownsizesFromPreprocess(camera, uses);
        Printf("  final safe downsizes %d\n", safeDownsizes);
        SetImageTextureMipDownsizeOverrideForFile(fn, safeDownsizes);
    }
}

}  // namespace pbrt
