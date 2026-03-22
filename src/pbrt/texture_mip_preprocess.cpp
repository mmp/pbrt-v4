// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/texture_mip_preprocess.h>

#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/options.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/colorspace.h>
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

static Float MinPrimaryContinuousLodForUse(const Camera &camera, int samplesPerPixel,
                                           const ImageTextureGeometryUse &use,
                                           int pyramidLevels, Allocator alloc) {
    if (!Options || Options->disableTextureFiltering || use.triangles.empty())
        return 0;

    std::vector<int> indices;
    std::vector<Point3f> p;
    std::vector<Point2f> uv;
    int vBase = 0;
    for (const ImageTextureMeshTriangle &t : use.triangles) {
        p.push_back(t.p0);
        p.push_back(t.p1);
        p.push_back(t.p2);
        uv.push_back(t.uv0);
        uv.push_back(t.uv1);
        uv.push_back(t.uv2);
        indices.push_back(vBase);
        indices.push_back(vBase + 1);
        indices.push_back(vBase + 2);
        vBase += 3;
    }

    const TriangleMesh *mesh = alloc.new_object<TriangleMesh>(
        Transform(), false, std::move(indices), std::move(p), std::vector<Vector3f>{},
        std::vector<Normal3f>{}, std::move(uv), std::vector<int>{}, alloc);

    Bounds2i pb = camera.GetFilm().PixelBounds();
    int fw = pb.pMax.x - pb.pMin.x, fh = pb.pMax.y - pb.pMin.y;
    int stride = std::max(1, std::max(fw, fh) / 64);

    Float minLod = Infinity;
    SampledWavelengths lambda = SampledWavelengths::SampleUniform(0.5f);
    UVMapping uvm(use.su, use.sv, use.du, use.dv);

    for (int py = pb.pMin.y; py < pb.pMax.y; py += stride) {
        for (int px = pb.pMin.x; px < pb.pMax.x; px += stride) {
            CameraSample cs;
            cs.pFilm = Point2f(Float(px) + 0.5f, Float(py) + 0.5f);
            cs.time = 0.5f;
            cs.pLens = Point2f(0.5f, 0.5f);
            cs.filterWeight = 1;
            pstd::optional<CameraRayDifferential> crd =
                camera.GenerateRayDifferential(cs, lambda);
            if (!crd)
                continue;
            RayDifferential ray = crd->ray;
            Float scale = std::max<Float>(0.125f, 1.f / std::sqrt(Float(samplesPerPixel)));
            if (!Options->disablePixelJitter)
                ray.ScaleDifferentials(scale);

            Float tBest = Infinity;
            int bestTri = -1;
            pstd::optional<TriangleIntersection> bestIsct;
            for (int ti = 0; ti < mesh->nTriangles; ++ti) {
                const int *vv = &mesh->vertexIndices[3 * ti];
                Point3f p0 = mesh->p[vv[0]], p1 = mesh->p[vv[1]], p2 = mesh->p[vv[2]];
                auto isct = IntersectTriangle(ray, tBest, p0, p1, p2);
                if (isct && isct->t < tBest) {
                    tBest = isct->t;
                    bestTri = ti;
                    bestIsct = isct;
                }
            }
            if (!bestIsct || bestTri < 0)
                continue;

            SurfaceInteraction si = Triangle::InteractionFromIntersection(
                mesh, bestTri, *bestIsct, ray.time, -ray.d);
            si.ComputeDifferentials(ray, camera, samplesPerPixel);
            TextureEvalContext ctx(si);
            TexCoord2D tc = uvm.Map(ctx);
            Float lod = ImageTextureContinuousLOD(use.filter, Vector2f(tc.dsdx, tc.dtdx),
                                                  Vector2f(tc.dsdy, tc.dtdy), use.maxAnisotropy,
                                                  pyramidLevels);
            minLod = std::min(minLod, lod);
        }
    }

    if (!IsFinite(minLod))
        return 0;
    return minLod;
}

int ComputeImageTextureSafeDownsizesFromPreprocess(
    const Camera &camera, int samplesPerPixel,
    const std::vector<ImageTextureGeometryUse> &usesForTexture, int mipmapPyramidLevels,
    Allocator alloc) {
    (void)camera;

    if (usesForTexture.empty())
        return 0;

    const std::string texLog =
        ShortScenePathForMipLog(usesForTexture[0].resolvedImageFilename);
    Printf("[mip preprocess] texture \"%s\"\n", texLog);

    int textureMinSafeDownsizes = std::numeric_limits<int>::max();
    for (const ImageTextureGeometryUse &use : usesForTexture) {
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

        const char *defaultEncoding = HasExtension(fn, "png") ? "sRGB" : "linear";
        ColorEncoding encoding = ColorEncoding::Get(defaultEncoding, alloc);
        ImageAndMetadata imMeta = Image::Read(fn, alloc, encoding);
        Point2i res = imMeta.image.Resolution();
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
