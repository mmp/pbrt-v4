// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/texture_mip_preprocess.h>

#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/options.h>
#include <pbrt/scene.h>
#include <pbrt/textures.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/math.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/mipmap.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/spectrum.h>

#include <pbrt/util/transform.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_set>

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

// Declarations vs. CollectResolvedImageTextureFilenames may differ only by slash style or case
// on Windows; treat those as the same file.
static bool ResolvedImageTexturePathsEqual(const std::string &a, const std::string &b) {
    if (a == b)
        return true;
#ifdef PBRT_IS_WINDOWS
    std::string na = a, nb = b;
    for (char &c : na)
        if (c == '/')
            c = '\\';
    for (char &c : nb)
        if (c == '/')
            c = '\\';
    return _stricmp(na.c_str(), nb.c_str()) == 0;
#else
    return false;
#endif
}

static Transform InstanceWorldFromObject(const InstanceSceneEntity &inst) {
    if (inst.renderFromInstanceAnim)
        return inst.renderFromInstanceAnim->Interpolate(0.5f);
    if (inst.renderFromInstance)
        return *inst.renderFromInstance;
    return Transform();
}

static void ExtractTrianglesFromShape(const ShapeSceneEntity &sh, const Transform &worldFromShape,
                                      std::vector<ImageTextureMeshTriangle> *out) {
    std::vector<int> vi = sh.parameters.GetIntArray("indices");
    std::vector<Point3f> P = sh.parameters.GetPoint3fArray("P");
    std::vector<Point2f> uvs = sh.parameters.GetPoint2fArray("uv");
    if (P.empty())
        return;
    if (vi.empty() && P.size() == 3)
        vi = {0, 1, 2};
    if (vi.empty() || (vi.size() % 3) != 0)
        return;

    const Transform &xf = worldFromShape;
    const bool haveUvPerVertex = (uvs.size() == P.size());

    for (size_t t = 0; t + 2 < vi.size(); t += 3) {
        int i0 = vi[t], i1 = vi[t + 1], i2 = vi[t + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0 || size_t(i0) >= P.size() || size_t(i1) >= P.size() ||
            size_t(i2) >= P.size())
            continue;
        ImageTextureMeshTriangle tri;
        tri.p0 = xf(P[i0]);
        tri.p1 = xf(P[i1]);
        tri.p2 = xf(P[i2]);
        if (haveUvPerVertex) {
            tri.uv0 = uvs[i0];
            tri.uv1 = uvs[i1];
            tri.uv2 = uvs[i2];
        } else {
            tri.uv0 = Point2f(0, 0);
            tri.uv1 = Point2f(1, 0);
            tri.uv2 = Point2f(1, 1);
        }
        out->push_back(tri);
    }
}

// Inline materials use SceneEntity::name; named materials store the plugin in "string type"
// (see BasicScene::CreateMaterials).
static std::string MaterialPluginType(const SceneEntity &mtl) {
    std::string t = mtl.parameters.GetOneString("type", "");
    if (!t.empty())
        return t;
    return std::string(mtl.name);
}

// Spectrum imagemap names wired to "texture reflectance" on diffuse-like BxDFs, including via
// "mix" of named materials (book scenes use this).
static void CollectReflectanceSpectrumTextureNames(const BasicScene &scene, const SceneEntity &mtl,
                                                   std::unordered_set<std::string> *mixVisit,
                                                   std::vector<std::string> *out) {
    const std::string matType = MaterialPluginType(mtl);
    if (matType == "diffuse" || matType == "coateddiffuse" || matType == "diffusetransmission") {
        std::string ref = mtl.parameters.GetTexture("reflectance");
        if (!ref.empty())
            out->push_back(std::move(ref));
        return;
    }
    if (matType != "mix")
        return;

    std::vector<std::string> subNames = mtl.parameters.GetStringArray("materials");
    if (subNames.size() != 2)
        return;
    for (const std::string &sub : subNames) {
        if (!mixVisit->insert(sub).second)
            continue;
        SceneEntity subEnt;
        if (scene.LookupNamedMaterial(sub, &subEnt))
            CollectReflectanceSpectrumTextureNames(scene, subEnt, mixVisit, out);
        mixVisit->erase(sub);
    }
}

static void ExtractTrianglesFromPlyMeshShape(const ShapeSceneEntity &sh,
                                             const Transform &worldFromShape,
                                             std::vector<ImageTextureMeshTriangle> *out) {
    std::string plyFile = ResolveFilename(sh.parameters.GetOneString("filename", ""));
    if (plyFile.empty())
        return;
    TriQuadMesh mesh = TriQuadMesh::ReadPLY(plyFile);
    mesh.ConvertToOnlyTriangles();
    if (mesh.triIndices.empty())
        return;

    const Transform &xf = worldFromShape;
    const bool haveUvPerVertex =
        !mesh.uv.empty() && mesh.uv.size() == mesh.p.size();

    for (size_t t = 0; t + 2 < mesh.triIndices.size(); t += 3) {
        int i0 = mesh.triIndices[t], i1 = mesh.triIndices[t + 1], i2 = mesh.triIndices[t + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0 || size_t(i0) >= mesh.p.size() ||
            size_t(i1) >= mesh.p.size() || size_t(i2) >= mesh.p.size())
            continue;
        ImageTextureMeshTriangle tri;
        tri.p0 = xf(mesh.p[i0]);
        tri.p1 = xf(mesh.p[i1]);
        tri.p2 = xf(mesh.p[i2]);
        if (haveUvPerVertex) {
            tri.uv0 = mesh.uv[i0];
            tri.uv1 = mesh.uv[i1];
            tri.uv2 = mesh.uv[i2];
        } else {
            tri.uv0 = Point2f(0, 0);
            tri.uv1 = Point2f(1, 0);
            tri.uv2 = Point2f(1, 1);
        }
        out->push_back(tri);
    }
}

static void FillTrianglesForMeshShape(const ShapeSceneEntity &sh, const Transform &worldFromShape,
                                      std::vector<ImageTextureMeshTriangle> *tris) {
    if (sh.name == "trianglemesh")
        ExtractTrianglesFromShape(sh, worldFromShape, tris);
    else if (sh.name == "plymesh")
        ExtractTrianglesFromPlyMeshShape(sh, worldFromShape, tris);
}

static void AppendReflectanceImagemapUsesForMeshShape(
    const BasicScene &scene, const ShapeSceneEntity &sh, const Transform &worldFromShape,
    const std::string &geometryDebugLabel, const std::string &targetFilename,
    const std::vector<std::pair<std::string, SpectrumImagemapDeclarationInfo>> &texForFile,
    std::vector<ImageTextureGeometryUse> *uses) {
    if (sh.name != "trianglemesh" && sh.name != "plymesh")
        return;

    SceneEntity mtl;
    if (!scene.LookupShapeMaterial(sh, &mtl))
        return;

    std::vector<std::string> refTexNames;
    std::unordered_set<std::string> mixVisit;
    CollectReflectanceSpectrumTextureNames(scene, mtl, &mixVisit, &refTexNames);
    if (refTexNames.empty())
        return;

    std::unordered_set<std::string> dedup;
    for (const std::string &refTex : refTexNames) {
        if (!dedup.insert(refTex).second)
            continue;

        const SpectrumImagemapDeclarationInfo *info = nullptr;
        for (const auto &tf : texForFile) {
            if (tf.first == refTex) {
                info = &tf.second;
                break;
            }
        }
        if (!info)
            continue;

        pstd::optional<FilterFunction> ff = ParseFilter(info->filter);
        FilterFunction filter = ff.value_or(FilterFunction::Bilinear);

        ImageTextureGeometryUse use;
        use.resolvedImageFilename = targetFilename;
        use.geometryDebugLabel = geometryDebugLabel;
        use.su = info->su;
        use.sv = info->sv;
        use.du = info->du;
        use.dv = info->dv;
        use.maxAnisotropy = info->maxAnisotropy;
        use.filter = filter;
        FillTrianglesForMeshShape(sh, worldFromShape, &use.triangles);
        if (!use.triangles.empty())
            uses->push_back(std::move(use));
    }
}

static std::vector<ImageTextureGeometryUse> GatherImageTextureUsesForFile(
    const BasicScene &scene, const std::string &targetFilename,
    const std::map<std::string, SpectrumImagemapDeclarationInfo> &decls) {
    std::vector<ImageTextureGeometryUse> uses;

    std::vector<std::pair<std::string, SpectrumImagemapDeclarationInfo>> texForFile;
    for (const auto &kv : decls)
        if (ResolvedImageTexturePathsEqual(kv.second.resolvedFilename, targetFilename))
            texForFile.push_back(kv);

    for (size_t si = 0; si < scene.shapes.size(); ++si) {
        const ShapeSceneEntity &sh = scene.shapes[si];
        Transform worldFromShape =
            sh.renderFromObject ? *sh.renderFromObject : Transform();
        AppendReflectanceImagemapUsesForMeshShape(
            scene, sh, worldFromShape,
            StringPrintf("shape[%zu]_%s", si, std::string(sh.name).c_str()), targetFilename,
            texForFile, &uses);
    }

    for (size_t ii = 0; ii < scene.instances.size(); ++ii) {
        const InstanceSceneEntity &inst = scene.instances[ii];
        auto defIt = scene.instanceDefinitions.find(inst.name);
        if (defIt == scene.instanceDefinitions.end() || defIt->second == nullptr)
            continue;
        const InstanceDefinitionSceneEntity &def = *defIt->second;
        Transform worldFromObject = InstanceWorldFromObject(inst);
        for (size_t si = 0; si < def.shapes.size(); ++si) {
            const ShapeSceneEntity &sh = def.shapes[si];
            Transform shapeLocalFromMesh =
                sh.renderFromObject ? *sh.renderFromObject : Transform();
            Transform worldFromShape = worldFromObject * shapeLocalFromMesh;
            AppendReflectanceImagemapUsesForMeshShape(
                scene, sh, worldFromShape,
                StringPrintf("instance[%zu]_def_%s_shape[%zu]_%s", ii,
                             std::string(inst.name).c_str(), si, std::string(sh.name).c_str()),
                targetFilename, texForFile, &uses);
        }
    }

    return uses;
}

static void MultiplyMatrixPointHomogeneous(const SquareMatrix<4> &m, Point3f p, Float *ox,
                                           Float *oy, Float *ow) {
    Float x = p.x, y = p.y, z = p.z;
    *ox = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    *oy = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    *ow = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
}

// One 2×2 inverse for u/w, v/w, and 1/w on the same screen-space triangle.
static bool AffineTripleScalarGradients2D(Float x0, Float y0, Float x1, Float y1, Float x2,
                                          Float y2, Float fU0, Float fU1, Float fU2, Float *gxU,
                                          Float *gyU, Float fV0, Float fV1, Float fV2, Float *gxV,
                                          Float *gyV, Float fI0, Float fI1, Float fI2, Float *gxI,
                                          Float *gyI) {
    Float s0x = x1 - x0, s0y = y1 - y0;
    Float s1x = x2 - x0, s1y = y2 - y0;
    Float det = s0x * s1y - s0y * s1x;
    if (std::abs(det) < 1e-20f)
        return false;
    Float inv00 = s1y / det, inv01 = -s0y / det, inv10 = -s1x / det, inv11 = s0x / det;
    auto grad = [&](Float a0, Float a1, Float a2, Float *gx, Float *gy) {
        Float d0 = a1 - a0, d1 = a2 - a0;
        *gx = d0 * inv00 + d1 * inv10;
        *gy = d0 * inv01 + d1 * inv11;
    };
    grad(fU0, fU1, fU2, gxU, gyU);
    grad(fV0, fV1, fV2, gxV, gyV);
    grad(fI0, fI1, fI2, gxI, gyI);
    return true;
}

static Float MinPrimaryContinuousLodForUse(const Camera &camera, int samplesPerPixel,
                                           const ImageTextureGeometryUse &use,
                                           int pyramidLevels, Allocator alloc) {
    (void)alloc;
    if (!Options || Options->disableTextureFiltering || use.triangles.empty())
        return 0;

    const PerspectiveCamera *persp = camera.Cast<PerspectiveCamera>();
    const OrthographicCamera *ortho = camera.Cast<OrthographicCamera>();
    const ProjectiveCamera *proj = persp ? static_cast<const ProjectiveCamera *>(persp)
                                         : ortho ? static_cast<const ProjectiveCamera *>(ortho)
                                                  : nullptr;
    if (!proj)
        return 0;

    Transform rasterFromCamera = proj->GetRasterFromCameraTransform();
    const SquareMatrix<4> &M = rasterFromCamera.GetMatrix();
    Transform cameraFromRender = proj->GetCameraTransform().CameraFromRender(
        proj->SampleTime(0.5f));
    Bounds2i pb = proj->GetFilm().PixelBounds();

    Float sppScale = 1.f;
    if (!Options->disablePixelJitter)
        sppScale = std::max<Float>(0.125f, 1.f / std::sqrt(Float(samplesPerPixel)));

    constexpr Float kMinW = 1e-6f;

    auto lodForTriangle = [&](const ImageTextureMeshTriangle &tri) -> Float {
        Point3f pc0 = cameraFromRender(tri.p0);
        Point3f pc1 = cameraFromRender(tri.p1);
        Point3f pc2 = cameraFromRender(tri.p2);

        Float qx0, qy0, qw0, qx1, qy1, qw1, qx2, qy2, qw2;
        MultiplyMatrixPointHomogeneous(M, pc0, &qx0, &qy0, &qw0);
        MultiplyMatrixPointHomogeneous(M, pc1, &qx1, &qy1, &qw1);
        MultiplyMatrixPointHomogeneous(M, pc2, &qx2, &qy2, &qw2);

        if (qw0 <= kMinW || qw1 <= kMinW || qw2 <= kMinW)
            return Infinity;

        Float xr0 = qx0 / qw0, yr0 = qy0 / qw0;
        Float xr1 = qx1 / qw1, yr1 = qy1 / qw1;
        Float xr2 = qx2 / qw2, yr2 = qy2 / qw2;

        Float triMinX = std::min({xr0, xr1, xr2});
        Float triMaxX = std::max({xr0, xr1, xr2});
        Float triMinY = std::min({yr0, yr1, yr2});
        Float triMaxY = std::max({yr0, yr1, yr2});
        if (triMaxX < Float(pb.pMin.x) || triMinX >= Float(pb.pMax.x) ||
            triMaxY < Float(pb.pMin.y) || triMinY >= Float(pb.pMax.y))
            return Infinity;

        Float u0 = tri.uv0.x, v0 = tri.uv0.y;
        Float u1 = tri.uv1.x, v1 = tri.uv1.y;
        Float u2 = tri.uv2.x, v2 = tri.uv2.y;

        Float Iw0 = 1.f / qw0, Iw1 = 1.f / qw1, Iw2 = 1.f / qw2;
        Float Uow0 = u0 * Iw0, Uow1 = u1 * Iw1, Uow2 = u2 * Iw2;
        Float Vow0 = v0 * Iw0, Vow1 = v1 * Iw1, Vow2 = v2 * Iw2;

        Float dUow_dx, dUow_dy, dVow_dx, dVow_dy, dIw_dx, dIw_dy;
        if (!AffineTripleScalarGradients2D(xr0, yr0, xr1, yr1, xr2, yr2, Uow0, Uow1, Uow2,
                                           &dUow_dx, &dUow_dy, Vow0, Vow1, Vow2, &dVow_dx,
                                           &dVow_dy, Iw0, Iw1, Iw2, &dIw_dx, &dIw_dy))
            return Infinity;

        Float inv_w = (Iw0 + Iw1 + Iw2) / 3.f;
        if (!(inv_w >= kMinW) || !IsFinite(inv_w))
            return Infinity;

        Float sumUow = Uow0 + Uow1 + Uow2;
        Float sumVow = Vow0 + Vow1 + Vow2;
        Float sumIw = Iw0 + Iw1 + Iw2;
        if (sumIw <= kMinW)
            return Infinity;
        Float u = sumUow / sumIw;
        Float v = sumVow / sumIw;

        Float dudx = (dUow_dx - u * dIw_dx) / inv_w;
        Float dvdx = (dVow_dx - v * dIw_dx) / inv_w;
        Float dudy = (dUow_dy - u * dIw_dy) / inv_w;
        Float dvdy = (dVow_dy - v * dIw_dy) / inv_w;

        dudx *= sppScale;
        dvdx *= sppScale;
        dudy *= sppScale;
        dvdy *= sppScale;

        if (!IsFinite(dudx) || !IsFinite(dvdx) || !IsFinite(dudy) || !IsFinite(dvdy))
            return Infinity;

        Float dsdx = use.su * dudx, dsdy = use.su * dudy;
        Float dtdx = use.sv * dvdx, dtdy = use.sv * dvdy;

        return ImageTextureContinuousLOD(use.filter, Vector2f(dsdx, dtdx),
                                         Vector2f(dsdy, dtdy), use.maxAnisotropy, pyramidLevels);
    };

    Float minLod = Infinity;
    for (const ImageTextureMeshTriangle &tri : use.triangles) {
        Float lod = lodForTriangle(tri);
        if (IsFinite(lod))
            minLod = std::min(minLod, lod);
    }

    if (!IsFinite(minLod))
        return 0;
    return minLod;
}

int ComputeImageTextureSafeDownsizesFromPreprocess(
    const Camera &camera, int samplesPerPixel,
    const std::vector<ImageTextureGeometryUse> &usesForTexture, int mipmapPyramidLevels,
    Allocator alloc) {
    if (usesForTexture.empty())
        return 0;

    const std::string texLog =
        ShortScenePathForMipLog(usesForTexture[0].resolvedImageFilename);
    Printf("[mip preprocess] texture \"%s\"\n", texLog.c_str());

    int textureMinSafeDownsizes = std::numeric_limits<int>::max();
    for (size_t ui = 0; ui < usesForTexture.size(); ++ui) {
        const ImageTextureGeometryUse &use = usesForTexture[ui];
        Float minLod =
            MinPrimaryContinuousLodForUse(camera, samplesPerPixel, use, mipmapPyramidLevels, alloc);
        Float lodClamped = std::max<Float>(0, minLod);
        int pairSafe = (int)std::floor(lodClamped + 1e-5f);
        if (mipmapPyramidLevels > 0)
            pairSafe = std::min(pairSafe, mipmapPyramidLevels - 1);
        pairSafe = std::max(0, pairSafe);

        const std::string &geom =
            use.geometryDebugLabel.empty() ? std::string("(no label)") : use.geometryDebugLabel;
        Printf("  geometry \"%s\" -> safe downsizes %d (min primary LOD %.4f)\n", geom, pairSafe,
               minLod);

        textureMinSafeDownsizes = std::min(textureMinSafeDownsizes, pairSafe);
        if (textureMinSafeDownsizes == 0) {
            size_t remaining = usesForTexture.size() - ui - 1;
            if (remaining > 0)
                Printf("  (... skipping %zu more geometries; cannot increase safe downsizes above 0)\n",
                       remaining);
            break;
        }
    }

    if (textureMinSafeDownsizes == std::numeric_limits<int>::max())
        textureMinSafeDownsizes = 0;
    return std::max(0, textureMinSafeDownsizes);
}

void RunImageTextureMipPreprocess(BasicScene &scene, const Camera &camera,
                                   int samplesPerPixel) {
    ClearImageTextureMipDownsizeOverrides();
    if (!Options || !Options->skipMipImageTextures)
        return;

    Timer preprocessTimer;

    std::map<std::string, SpectrumImagemapDeclarationInfo> decls = scene.SpectrumImagemapDeclarations();
    std::vector<std::string> files = scene.CollectResolvedImageTextureFilenames();
    Allocator alloc;

    for (const std::string &fn : files) {
        std::vector<ImageTextureGeometryUse> uses =
            GatherImageTextureUsesForFile(scene, fn, decls);

        Point2i res = Image::ReadResolution(fn);
        int pyramidLevels = MipmapPyramidLevelsForImageResolution(res);

        int safeDownsizes = 0;
        if (uses.empty()) {
            Printf("[mip preprocess] texture \"%s\"\n", ShortScenePathForMipLog(fn));
            Printf("  (no trianglemesh/plymesh reflectance imagemap uses found; safe downsizes 0)\n");
        } else {
            safeDownsizes = ComputeImageTextureSafeDownsizesFromPreprocess(
                camera, samplesPerPixel, uses, pyramidLevels, alloc);
        }

        Printf("  final safe downsizes %d\n", safeDownsizes);
        SetImageTextureMipDownsizeOverrideForFile(fn, safeDownsizes);
    }

    Printf("[mip preprocess] wall time %.3f s\n", preprocessTimer.ElapsedSeconds());
}

}  // namespace pbrt
