// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/parsedscene.h>

#include <pbrt/cpu/aggregates.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/memory.h>
#endif // PBRT_BUILD_GPU_RENDERER
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/shapes.h>
#include <pbrt/util/args.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/file.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>

#include <iostream>
#include <mutex>

namespace pbrt {

template <typename T, typename U>
static std::string ToString(const std::map<T, U> &m) {
    std::string s = "[ ";
    for (const auto &iter : m)
        s += StringPrintf("%s:%s ", iter.first, iter.second);
    s += "]";
    return s;
}

template <typename T, typename U>
static std::string ToString(const std::pair<T, U> &p) {
    return StringPrintf("[ std::pair first: %s second: %s ]", p.first, p.second);
}

std::string ParsedScene::ToString() const {
    return StringPrintf("[ ParsedScene camera: %s film: %s sampler: %s integrator: %s "
                        "filter: %s accelerator: %s namedMaterials: %s materials: %s "
                        "media: %s floatTextures: %s spectrumTextures: %s "
                        "instanceDefinitions: %s lights: %s "
                        "shapes: %s instances: %s ]",
                        camera, film, sampler, integrator, filter, accelerator,
                        namedMaterials, materials, media, floatTextures, spectrumTextures,
                        instanceDefinitions, lights, shapes, instances);
}

ParsedScene::GraphicsState::GraphicsState() {
    currentMaterialIndex = 0;
}

// API State Macros
#define VERIFY_OPTIONS(func)                                   \
    if (currentBlock == BlockState::WorldBlock) {              \
        ErrorExit(&loc,                                        \
                  "Options cannot be set inside world block; " \
                  "\"%s\" is not allowed.",                    \
                  func);                                       \
        return;                                                \
    } else /* swallow trailing semicolon */
#define VERIFY_WORLD(func)                                         \
    if (currentBlock == BlockState::OptionsBlock) {                \
        ErrorExit(&loc,                                            \
                  "Scene description must be inside world block; " \
                  "\"%s\" is not allowed.",                        \
                  func);                                           \
        return;                                                    \
    } else /* swallow trailing semicolon */

#define FOR_ACTIVE_TRANSFORMS(expr)                         \
    for (int i = 0; i < MaxTransforms; ++i)                 \
        if (graphicsState.activeTransformBits & (1 << i)) { \
            expr                                            \
        }

STAT_MEMORY_COUNTER("Memory/TransformCache", transformCacheBytes);
STAT_PERCENT("Geometry/TransformCache hits", nTransformCacheHits, nTransformCacheLookups);

// TransformCache Method Definitions
TransformCache::TransformCache()
#ifdef PBRT_BUILD_GPU_RENDERER
    : bufferResource(Options->useGPU ? gpuMemoryAllocator.resource()
                     : Allocator().resource()),
#else
      : bufferResource(Allocator().resource()),
#endif
      alloc(&bufferResource) {}

const Transform *TransformCache::Lookup(const Transform &t) {
    ++nTransformCacheLookups;

    if (!hashTable.empty()) {
        size_t offset = t.Hash() % hashTable.bucket_count();
        for (auto iter = hashTable.begin(offset); iter != hashTable.end(offset); ++iter) {
            if (**iter == t) {
                ++nTransformCacheHits;
                return *iter;
            }
        }
    }
    Transform *tptr = alloc.new_object<Transform>(t);
    transformCacheBytes += sizeof(Transform);
    hashTable.insert(tptr);
    return tptr;
}

TransformCache::~TransformCache() {
    for (const auto &iter : hashTable) {
        Transform *tptr = iter;
        alloc.delete_object(tptr);
    }
}

STAT_COUNTER("Scene/Object instances created", nObjectInstancesCreated);
STAT_COUNTER("Scene/Object instances used", nObjectInstancesUsed);

// ParsedScene Method Definitions
ParsedScene::ParsedScene() {
    // Set scene defaults
    camera.name = "perspective";
    sampler.name = "zsobol";
    filter.name = "gaussian";
    integrator.name = "volpath";

    ParameterDictionary dict({}, RGBColorSpace::sRGB);
    materials.push_back(SceneEntity("diffuse", dict, {}));
    accelerator.name = "bvh";
    film.name = "rgb";
    film.parameters = ParameterDictionary({}, RGBColorSpace::sRGB);
}

void ParsedScene::ReverseOrientation(FileLoc loc) {
    VERIFY_WORLD("ReverseOrientation");
    graphicsState.reverseOrientation = !graphicsState.reverseOrientation;
}

void ParsedScene::ColorSpace(const std::string &name, FileLoc loc) {
    if (const RGBColorSpace *cs = RGBColorSpace::GetNamed(name))
        graphicsState.colorSpace = cs;
    else
        Error(&loc, "%s: color space unknown", name);
}

void ParsedScene::Identity(FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = pbrt::Transform();)
}

void ParsedScene::Translate(Float dx, Float dy, Float dz, FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = graphicsState.ctm[i] *
                                                 pbrt::Translate(Vector3f(dx, dy, dz));)
}

void ParsedScene::CoordinateSystem(const std::string &name, FileLoc loc) {
    namedCoordinateSystems[name] = graphicsState.ctm;
}

void ParsedScene::CoordSysTransform(const std::string &name, FileLoc loc) {
    if (namedCoordinateSystems.find(name) != namedCoordinateSystems.end())
        graphicsState.ctm = namedCoordinateSystems[name];
    else
        Warning(&loc, "Couldn't find named coordinate system \"%s\"", name);
}

void ParsedScene::Camera(const std::string &name, ParsedParameterVector params,
                         FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);

    VERIFY_OPTIONS("Camera");

    TransformSet cameraFromWorld = graphicsState.ctm;
    TransformSet worldFromCamera = Inverse(graphicsState.ctm);
    namedCoordinateSystems["camera"] = Inverse(cameraFromWorld);

    CameraTransform cameraTransform(
        AnimatedTransform(worldFromCamera[0], graphicsState.transformStartTime,
                          worldFromCamera[1], graphicsState.transformEndTime));
    renderFromWorld = cameraTransform.RenderFromWorld();

    camera = CameraSceneEntity(name, std::move(dict), loc, cameraTransform,
                               graphicsState.currentOutsideMedium);
}

void ParsedScene::AttributeBegin(FileLoc loc) {
    VERIFY_WORLD("AttributeBegin");
    pushedGraphicsStates.push_back(graphicsState);
    pushStack.push_back(std::make_pair('a', loc));
}

void ParsedScene::AttributeEnd(FileLoc loc) {
    VERIFY_WORLD("AttributeEnd");
    if (pushedGraphicsStates.empty()) {
        Error(&loc, "Unmatched AttributeEnd encountered. Ignoring it.");
        return;
    }
    // NOTE: must keep the following consistent with code in ObjectEnd
    graphicsState = std::move(pushedGraphicsStates.back());
    pushedGraphicsStates.pop_back();

    if (pushStack.back().first == 'o')
        ErrorExitDeferred(&loc,
                          "Mismatched nesting: open ObjectBegin from %s at AttributeEnd",
                          pushStack.back().second);
    else
        CHECK_EQ(pushStack.back().first, 'a');
    pushStack.pop_back();
}

void ParsedScene::Attribute(const std::string &target, ParsedParameterVector attrib,
                            FileLoc loc) {
    ParsedParameterVector *currentAttributes = nullptr;
    if (target == "shape") {
        currentAttributes = &graphicsState.shapeAttributes;
    } else if (target == "light") {
        currentAttributes = &graphicsState.lightAttributes;
    } else if (target == "material") {
        currentAttributes = &graphicsState.materialAttributes;
    } else if (target == "medium") {
        currentAttributes = &graphicsState.mediumAttributes;
    } else if (target == "texture") {
        currentAttributes = &graphicsState.textureAttributes;
    } else {
        ErrorExitDeferred(
            &loc,
            "Unknown attribute target \"%s\". Must be \"shape\", \"light\", "
            "\"material\", \"medium\", or \"texture\".",
            target);
        return;
    }

    // Note that we hold on to the current color space and associate it
    // with the parameters...
    for (ParsedParameter *p : attrib) {
        p->mayBeUnused = true;
        p->colorSpace = graphicsState.colorSpace;
        currentAttributes->push_back(p);
    }
}

void ParsedScene::WorldBegin(FileLoc loc) {
    VERIFY_OPTIONS("WorldBegin");
    currentBlock = BlockState::WorldBlock;
    for (int i = 0; i < MaxTransforms; ++i)
        graphicsState.ctm[i] = pbrt::Transform();
    graphicsState.activeTransformBits = AllTransformsBits;
    namedCoordinateSystems["world"] = graphicsState.ctm;
}

void ParsedScene::LightSource(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    VERIFY_WORLD("LightSource");
    ParameterDictionary dict(std::move(params), graphicsState.lightAttributes,
                             graphicsState.colorSpace);
    lights.push_back(LightSceneEntity(name, std::move(dict), loc, RenderFromObject(),
                                      graphicsState.currentOutsideMedium));
}

void ParsedScene::Shape(const std::string &name, ParsedParameterVector params,
                        FileLoc loc) {
    VERIFY_WORLD("Shape");

    ParameterDictionary dict(std::move(params), graphicsState.shapeAttributes,
                             graphicsState.colorSpace);

    int areaLightIndex = -1;
    if (!graphicsState.areaLightName.empty()) {
        areaLights.push_back(SceneEntity(graphicsState.areaLightName,
                                         graphicsState.areaLightParams,
                                         graphicsState.areaLightLoc));
        areaLightIndex = areaLights.size() - 1;
    }

    if (CTMIsAnimated()) {
        std::vector<AnimatedShapeSceneEntity> *as = &animatedShapes;
        if (currentInstance != nullptr) {
            if (!graphicsState.areaLightName.empty())
                Warning(&loc, "Area lights not supported with object instancing");
            as = &currentInstance->animatedShapes;
        }

        AnimatedTransform renderFromShape = RenderFromObject();
        const class Transform *identity = transformCache.Lookup(pbrt::Transform());

        as->push_back(AnimatedShapeSceneEntity(
            {name, std::move(dict), loc, renderFromShape, identity,
             graphicsState.reverseOrientation, graphicsState.currentMaterialIndex,
             graphicsState.currentMaterialName, areaLightIndex,
             graphicsState.currentInsideMedium, graphicsState.currentOutsideMedium}));
    } else {
        std::vector<ShapeSceneEntity> *s = &shapes;
        if (currentInstance != nullptr) {
            if (!graphicsState.areaLightName.empty())
                Warning(&loc, "Area lights not supported with object instancing");
            s = &currentInstance->shapes;
        }

        const class Transform *renderFromObject =
            transformCache.Lookup(RenderFromObject(0));
        const class Transform *objectFromRender =
            transformCache.Lookup(Inverse(*renderFromObject));

        s->push_back(ShapeSceneEntity(
            {name, std::move(dict), loc, renderFromObject, objectFromRender,
             graphicsState.reverseOrientation, graphicsState.currentMaterialIndex,
             graphicsState.currentMaterialName, areaLightIndex,
             graphicsState.currentInsideMedium, graphicsState.currentOutsideMedium}));
    }
}

void ParsedScene::ObjectBegin(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("ObjectBegin");
    pushedGraphicsStates.push_back(graphicsState);

    pushStack.push_back(std::make_pair('o', loc));

    if (currentInstance != nullptr) {
        ErrorExitDeferred(&loc, "ObjectBegin called inside of instance definition");
        return;
    }

    if (instanceDefinitions.find(name) != instanceDefinitions.end()) {
        ErrorExitDeferred(&loc, "%s: trying to redefine an object instance", name);
        return;
    }

    instanceDefinitions[name] = InstanceDefinitionSceneEntity(name, loc);
    // This should be safe since we're not adding anything to
    // instanceDefinitions until after ObjectEnd... (Still makes me
    // nervous.)
    currentInstance = &instanceDefinitions[name];
}

void ParsedScene::ObjectEnd(FileLoc loc) {
    VERIFY_WORLD("ObjectEnd");
    if (currentInstance == nullptr) {
        ErrorExitDeferred(&loc, "ObjectEnd called outside of instance definition");
        return;
    }
    currentInstance = nullptr;

    // NOTE: Must keep the following consistent with AttributeEnd
    graphicsState = std::move(pushedGraphicsStates.back());
    pushedGraphicsStates.pop_back();

    ++nObjectInstancesCreated;

    if (pushStack.back().first == 'a')
        ErrorExitDeferred(&loc,
                          "Mismatched nesting: open AttributeBegin from %s at ObjectEnd",
                          pushStack.back().second);
    else
        CHECK_EQ(pushStack.back().first, 'o');
    pushStack.pop_back();
}

void ParsedScene::ObjectInstance(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("ObjectInstance");

    if (currentInstance != nullptr) {
        ErrorExitDeferred(&loc,
                          "ObjectInstance can't be called inside instance definition");
        return;
    }

    class Transform worldFromRender = Inverse(renderFromWorld);

    if (CTMIsAnimated()) {
        AnimatedTransform animatedRenderFromInstance(
            RenderFromObject(0) * worldFromRender, graphicsState.transformStartTime,
            RenderFromObject(1) * worldFromRender, graphicsState.transformEndTime);

        instances.push_back(InstanceSceneEntity(name, loc, animatedRenderFromInstance));
    } else {
        const class Transform *renderFromInstance =
            transformCache.Lookup(RenderFromObject(0) * worldFromRender);

        instances.push_back(InstanceSceneEntity(name, loc, renderFromInstance));
    }
}

void ParsedScene::EndOfFiles() {
    if (currentBlock != BlockState::WorldBlock)
        ErrorExitDeferred("End of files before \"WorldBegin\".");

    // Ensure there are no pushed graphics states
    while (!pushedGraphicsStates.empty()) {
        ErrorExitDeferred("Missing end to AttributeBegin");
        pushedGraphicsStates.pop_back();
    }

    if (errorExit)
        ErrorExit("Fatal errors during scene construction");

    // LOG_VERBOSE messages about any unused textures..
    std::set<std::string> unusedFloatTextures, unusedSpectrumTextures;
    for (const auto f : floatTextures) {
        CHECK(unusedFloatTextures.find(f.first) == unusedFloatTextures.end());
        unusedFloatTextures.insert(f.first);
    }
    for (const auto s : spectrumTextures) {
        CHECK(unusedSpectrumTextures.find(s.first) == unusedSpectrumTextures.end());
        unusedSpectrumTextures.insert(s.first);
    }

    auto checkVec = [&](const ParsedParameterVector &vec) {
        for (const ParsedParameter *p : vec) {
            if (p->type == "texture") {
                CHECK(!p->strings.empty());
                if (auto iter = unusedFloatTextures.find(p->strings[0]);
                    iter != unusedFloatTextures.end())
                    unusedFloatTextures.erase(iter);
                else if (auto iter = unusedSpectrumTextures.find(p->strings[0]);
                         iter != unusedSpectrumTextures.end())
                    unusedSpectrumTextures.erase(iter);
            }
        }
    };

    // Walk through everything that uses textures..
    for (const auto &nm : namedMaterials)
        checkVec(nm.second.parameters.GetParameterVector());
    for (const auto &m : materials)
        checkVec(m.parameters.GetParameterVector());
    for (const auto &ft : floatTextures)
        checkVec(ft.second.parameters.GetParameterVector());
    for (const auto &st : spectrumTextures)
        checkVec(st.second.parameters.GetParameterVector());
    for (const auto &s : shapes)
        checkVec(s.parameters.GetParameterVector());
    for (const auto &as : animatedShapes)
        checkVec(as.parameters.GetParameterVector());
    for (const auto &id : instanceDefinitions) {
        for (const auto &s : id.second.shapes)
            checkVec(s.parameters.GetParameterVector());
        for (const auto &as : id.second.animatedShapes)
            checkVec(as.parameters.GetParameterVector());
    }

    LOG_VERBOSE("Scene stats: %d shapes, %d animated shapes, %d instance definitions, "
                "%d instance uses, %d lights, %d float textures, %d spectrum textures, "
                "%d named materials, %d materials",
                shapes.size(), animatedShapes.size(), instanceDefinitions.size(),
                instances.size(), lights.size(), floatTextures.size(),
                spectrumTextures.size(), namedMaterials.size(), materials.size());

    // And complain about what's left.
    for (const std::string &s : unusedFloatTextures)
        LOG_VERBOSE("%s: float texture unused in scene", s);
    for (const std::string &s : unusedSpectrumTextures)
        LOG_VERBOSE("%s: spectrum texture unused in scene", s);
}

ParsedScene *ParsedScene::CopyForImport() {
    ParsedScene *importScene = new ParsedScene;
    importScene->renderFromWorld = renderFromWorld;
    importScene->graphicsState = graphicsState;
    importScene->currentBlock = currentBlock;
    if (currentInstance) {
        importScene->currentInstance = new InstanceDefinitionSceneEntity;
        importScene->currentInstance->name = currentInstance->name;
        importScene->currentInstance->loc = currentInstance->loc;
    }
    return importScene;
}

void ParsedScene::MergeImported(ParsedScene *importScene) {
    while (!importScene->pushedGraphicsStates.empty()) {
        ErrorExitDeferred("Missing end to AttributeBegin");
        importScene->pushedGraphicsStates.pop_back();
    }

    errorExit |= importScene->errorExit;

    // Reindex materials in shapes in importScene
    size_t materialBase = materials.size(), lightBase = lights.size();
    auto reindex = [materialBase, lightBase](auto &map) {
        for (auto &shape : map) {
            if (shape.materialName.empty())
                shape.materialIndex += materialBase;
            if (shape.lightIndex >= 0)
                shape.lightIndex += lightBase;
        }
    };
    reindex(importScene->shapes);
    reindex(importScene->animatedShapes);
    for (auto &inst : importScene->instanceDefinitions) {
        reindex(inst.second.shapes);
        reindex(inst.second.animatedShapes);
    }

    auto mergeVector = [](auto &base, auto &imported) {
        if (base.empty())
            base = std::move(imported);
        else {
            base.reserve(base.size() + imported.size());
            std::move(std::begin(imported), std::end(imported), std::back_inserter(base));
            imported.clear();
            imported.shrink_to_fit();
        }
    };

    if (importScene->currentInstance) {
        reindex(importScene->currentInstance->shapes);
        reindex(importScene->currentInstance->animatedShapes);
        bool found = false;
        for (auto &def : instanceDefinitions) {
            if (def.second.name == importScene->currentInstance->name) {
                found = true;
                mergeVector(def.second.shapes, importScene->currentInstance->shapes);
                mergeVector(def.second.animatedShapes,
                            importScene->currentInstance->animatedShapes);
                delete importScene->currentInstance;
                importScene->currentInstance = nullptr;
                break;
            }
        }
        CHECK(found);
    }

    auto mergeMap = [this](auto &base, auto &imported, const char *name) {
        for (const auto &item : imported) {
            if (base.find(item.first) != base.end())
                ErrorExitDeferred(&item.second.loc, "%s: multiply-defined %s.",
                                  item.first, name);
            base[item.first] = std::move(item.second);
        }
        imported.clear();
    };
    mergeMap(instanceDefinitions, importScene->instanceDefinitions, "object instance");
    // mergeMap(namedCoordinateSystems, importScene->namedCoordinateSystems, "named
    // coordinate system");

    auto mergeSet = [this](auto &base, auto &imported, const char *name) {
        for (const auto &item : imported) {
            if (base.find(item) != base.end())
                ErrorExitDeferred("%s: multiply-defined %s.", item, name);
            base.insert(std::move(item));
        }
        imported.clear();
    };
    mergeSet(namedMaterialNames, importScene->namedMaterialNames, "named material");
    mergeSet(floatTextureNames, importScene->floatTextureNames, "texture");
    mergeSet(spectrumTextureNames, importScene->spectrumTextureNames, "texture");

    mergeVector(materials, importScene->materials);
    mergeVector(namedMaterials, importScene->namedMaterials);
    mergeVector(floatTextures, importScene->floatTextures);
    mergeVector(spectrumTextures, importScene->spectrumTextures);
    mergeVector(lights, importScene->lights);
    mergeVector(shapes, importScene->shapes);
    mergeVector(animatedShapes, importScene->animatedShapes);
    mergeVector(instances, importScene->instances);
}

void ParsedScene::Option(const std::string &name, const std::string &value, FileLoc loc) {
    std::string nName = normalizeArg(name);

    if (nName == "disablepixeljitter") {
        if (value == "true")
            Options->disablePixelJitter = true;
        else if (value == "false")
            Options->disablePixelJitter = false;
        else
            ErrorExitDeferred(&loc, "%s: expected \"true\" or \"false\" for option value",
                              value);
    } else if (nName == "disablewavelengthjitter") {
        if (value == "true")
            Options->disableWavelengthJitter = true;
        else if (value == "false")
            Options->disableWavelengthJitter = false;
        else
            ErrorExitDeferred(&loc, "%s: expected \"true\" or \"false\" for option value",
                              value);
    } else if (nName == "msereferenceimage") {
        if (value.size() < 3 || value.front() != '"' || value.back() != '"')
            ErrorExitDeferred(&loc, "%s: expected quoted string for option value", value);
        Options->mseReferenceImage = value.substr(1, value.size() - 2);
    } else if (nName == "msereferenceout") {
        if (value.size() < 3 || value.front() != '"' || value.back() != '"')
            ErrorExitDeferred(&loc, "%s: expected quoted string for option value", value);
        Options->mseReferenceOutput = value.substr(1, value.size() - 2);
    } else if (nName == "seed") {
        Options->seed = std::atoi(value.c_str());
    } else if (nName == "forcediffuse") {
        if (value == "true")
            Options->forceDiffuse = true;
        else if (value == "false")
            Options->forceDiffuse = false;
        else
            ErrorExitDeferred(&loc, "%s: expected \"true\" or \"false\" for option value",
                              value);
    } else if (nName == "pixelstats") {
        if (value == "true")
            Options->recordPixelStatistics = true;
        else if (value == "false")
            Options->recordPixelStatistics = false;
        else
            ErrorExitDeferred(&loc, "%s: expected \"true\" or \"false\" for option value",
                              value);
    } else
        ErrorExitDeferred(&loc, "%s: unknown option", name);
}

void ParsedScene::Transform(Float tr[16], FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = Transpose(
                              pbrt::Transform(SquareMatrix<4>(pstd::MakeSpan(tr, 16))));)
}

void ParsedScene::ConcatTransform(Float tr[16], FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(
        graphicsState.ctm[i] =
            graphicsState.ctm[i] *
            Transpose(pbrt::Transform(SquareMatrix<4>(pstd::MakeSpan(tr, 16))));)
}

void ParsedScene::Rotate(Float angle, Float dx, Float dy, Float dz, FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] =
                              graphicsState.ctm[i] *
                              pbrt::Rotate(angle, Vector3f(dx, dy, dz));)
}

void ParsedScene::Scale(Float sx, Float sy, Float sz, FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] =
                              graphicsState.ctm[i] * pbrt::Scale(sx, sy, sz);)
}

void ParsedScene::LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                         Float ux, Float uy, Float uz, FileLoc loc) {
    class Transform lookAt =
        pbrt::LookAt(Point3f(ex, ey, ez), Point3f(lx, ly, lz), Vector3f(ux, uy, uz));
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = graphicsState.ctm[i] * lookAt;);
}

void ParsedScene::ActiveTransformAll(FileLoc loc) {
    graphicsState.activeTransformBits = AllTransformsBits;
}

void ParsedScene::ActiveTransformEndTime(FileLoc loc) {
    graphicsState.activeTransformBits = EndTransformBits;
}

void ParsedScene::ActiveTransformStartTime(FileLoc loc) {
    graphicsState.activeTransformBits = StartTransformBits;
}

void ParsedScene::TransformTimes(Float start, Float end, FileLoc loc) {
    VERIFY_OPTIONS("TransformTimes");
    graphicsState.transformStartTime = start;
    graphicsState.transformEndTime = end;
}

void ParsedScene::PixelFilter(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("PixelFilter");
    filter = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::Film(const std::string &type, ParsedParameterVector params,
                       FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("Film");
    film = SceneEntity(type, std::move(dict), loc);
}

void ParsedScene::Sampler(const std::string &name, ParsedParameterVector params,
                          FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);

    VERIFY_OPTIONS("Sampler");
    sampler = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::Accelerator(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("Accelerator");
    accelerator = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::Integrator(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);

    VERIFY_OPTIONS("Integrator");
    integrator = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                                  FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.mediumAttributes,
                             graphicsState.colorSpace);

    if (media.find(name) != media.end()) {
        ErrorExitDeferred(&loc, "Named medium \"%s\" redefined.", name);
        return;
    }

    media[name] = TransformedSceneEntity(name, std::move(dict), loc, RenderFromObject());
}

void ParsedScene::MediumInterface(const std::string &insideName,
                                  const std::string &outsideName, FileLoc loc) {
    graphicsState.currentInsideMedium = insideName;
    graphicsState.currentOutsideMedium = outsideName;
}

void ParsedScene::Texture(const std::string &name, const std::string &type,
                          const std::string &texname, ParsedParameterVector params,
                          FileLoc loc) {
    VERIFY_WORLD("Texture");

    ParameterDictionary dict(std::move(params), graphicsState.textureAttributes,
                             graphicsState.colorSpace);

    if (type != "float" && type != "spectrum") {
        ErrorExitDeferred(
            &loc, "%s: texture type unknown. Must be \"float\" or \"spectrum\".", type);
        return;
    }

    std::set<std::string> &names =
        (type == "float") ? floatTextureNames : spectrumTextureNames;
    if (names.find(name) != names.end()) {
        ErrorExitDeferred(&loc, "Redefining texture \"%s\".", name);
        return;
    }
    names.insert(name);

    std::vector<std::pair<std::string, TextureSceneEntity>> &textures =
        (type == "float") ? floatTextures : spectrumTextures;

    textures.push_back(std::make_pair(
        name, TextureSceneEntity(texname, std::move(dict), loc, RenderFromObject())));
}

void ParsedScene::Material(const std::string &name, ParsedParameterVector params,
                           FileLoc loc) {
    VERIFY_WORLD("Material");
    ParameterDictionary dict(std::move(params), graphicsState.materialAttributes,
                             graphicsState.colorSpace);
    materials.push_back(SceneEntity(name, std::move(dict), loc));
    graphicsState.currentMaterialIndex = materials.size() - 1;
    graphicsState.currentMaterialName.clear();
}

void ParsedScene::MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    VERIFY_WORLD("MakeNamedMaterial");

    ParameterDictionary dict(std::move(params), graphicsState.materialAttributes,
                             graphicsState.colorSpace);

    if (namedMaterialNames.find(name) != namedMaterialNames.end()) {
        ErrorExitDeferred(&loc, "%s: named material redefined.", name);
        return;
    }
    namedMaterialNames.insert(name);

    namedMaterials.push_back(std::make_pair(name, SceneEntity("", std::move(dict), loc)));
}

void ParsedScene::NamedMaterial(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("NamedMaterial");
    graphicsState.currentMaterialName = name;
    graphicsState.currentMaterialIndex = -1;
}

void ParsedScene::AreaLightSource(const std::string &name, ParsedParameterVector params,
                                  FileLoc loc) {
    VERIFY_WORLD("AreaLightSource");
    graphicsState.areaLightName = name;
    graphicsState.areaLightParams = ParameterDictionary(
        std::move(params), graphicsState.lightAttributes, graphicsState.colorSpace);
    graphicsState.areaLightLoc = loc;
}

void ParsedScene::CreateMaterials(
    /*const*/ NamedTextures &textures, Allocator alloc,
    std::map<std::string, pbrt::Material> *namedMaterialsOut,
    std::vector<pbrt::Material> *materialsOut) const {
    // First, load all of the normal maps in parallel.
    std::set<std::string> normalMapFilenames;
    for (const auto &nm : namedMaterials) {
        std::string fn = nm.second.parameters.GetOneString("normalmap", "");
        if (!fn.empty())
            normalMapFilenames.insert(fn);
    }
    for (const auto &mtl : materials) {
        std::string fn = mtl.parameters.GetOneString("normalmap", "");
        if (!fn.empty())
            normalMapFilenames.insert(fn);
    }

    std::vector<std::string> normalMapFilenameVector;
    std::copy(normalMapFilenames.begin(), normalMapFilenames.end(),
              std::back_inserter(normalMapFilenameVector));

    LOG_VERBOSE("Reading %d normal maps in parallel", normalMapFilenameVector.size());
    std::map<std::string, Image *> normalMapCache;
    std::mutex mutex;
    ParallelFor(0, normalMapFilenameVector.size(), [&](int64_t index) {
        std::string filename = normalMapFilenameVector[index];
        ImageAndMetadata immeta =
            Image::Read(filename, Allocator(), ColorEncoding::Linear);
        Image &image = immeta.image;
        ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
        if (!rgbDesc)
            ErrorExitDeferred("%s: normal map image must contain R, G, and B channels",
                              filename);
        Image *normalMap = alloc.new_object<Image>(alloc);
        *normalMap = image.SelectChannels(rgbDesc);

        mutex.lock();
        normalMapCache[filename] = normalMap;
        mutex.unlock();
    });
    LOG_VERBOSE("Done reading normal maps");

    // Named materials
    for (const auto &nm : namedMaterials) {
        const std::string &name = nm.first;
        const SceneEntity &mtl = nm.second;
        if (namedMaterialsOut->find(name) != namedMaterialsOut->end()) {
            ErrorExitDeferred(&mtl.loc, "%s: trying to redefine named material.", name);
            continue;
        }

        std::string type = mtl.parameters.GetOneString("type", "");
        if (type.empty()) {
            ErrorExitDeferred(&mtl.loc,
                              "%s: \"string type\" not provided in named material's "
                              "parameters.",
                              name);
            continue;
        }

        std::string fn = nm.second.parameters.GetOneString("normalmap", "");
        Image *normalMap = !fn.empty() ? normalMapCache[fn] : nullptr;

        TextureParameterDictionary texDict(&mtl.parameters, &textures);
        class Material m = Material::Create(type, texDict, normalMap, *namedMaterialsOut,
                                            &mtl.loc, alloc);
        (*namedMaterialsOut)[name] = m;
    }

    // Regular materials
    materialsOut->reserve(materials.size());
    for (const auto &mtl : materials) {
        std::string fn = mtl.parameters.GetOneString("normalmap", "");
        Image *normalMap = !fn.empty() ? normalMapCache[fn] : nullptr;

        TextureParameterDictionary texDict(&mtl.parameters, &textures);
        class Material m = Material::Create(mtl.name, texDict, normalMap,
                                            *namedMaterialsOut, &mtl.loc, alloc);
        materialsOut->push_back(m);
    }
}

NamedTextures ParsedScene::CreateTextures(Allocator alloc, bool gpu) const {
    NamedTextures textures;

    std::set<std::string> seenFloatTextureFilenames, seenSpectrumTextureFilenames;
    std::vector<size_t> parallelFloatTextures, serialFloatTextures;
    std::vector<size_t> parallelSpectrumTextures, serialSpectrumTextures;

    // Figure out which textures to load in parallel
    // Need to be careful since two textures can use the same image file;
    // we only want to load it once in that case...
    int nMissingTextures = 0;
    for (size_t i = 0; i < floatTextures.size(); ++i) {
        const auto &tex = floatTextures[i];

        if (tex.second.renderFromObject.IsAnimated())
            Warning(&tex.second.loc,
                    "Animated world to texture transforms are not supported. "
                    "Using start transform.");

        if (tex.second.texName != "imagemap" && tex.second.texName != "ptex") {
            serialFloatTextures.push_back(i);
            continue;
        }

        std::string filename =
            ResolveFilename(tex.second.parameters.GetOneString("filename", ""));
        if (filename.empty())
            continue;
        if (!FileExists(filename)) {
            Error(&tex.second.loc, "%s: file not found.", filename);
            ++nMissingTextures;
        }

        if (seenFloatTextureFilenames.find(filename) == seenFloatTextureFilenames.end()) {
            seenFloatTextureFilenames.insert(filename);
            parallelFloatTextures.push_back(i);
        } else
            serialFloatTextures.push_back(i);
    }
    for (size_t i = 0; i < spectrumTextures.size(); ++i) {
        const auto &tex = spectrumTextures[i];

        if (tex.second.renderFromObject.IsAnimated())
            Warning(&tex.second.loc,
                    "Animated world to texture transforms are not supported. "
                    "Using start transform.");

        if (tex.second.texName != "imagemap" && tex.second.texName != "ptex") {
            serialSpectrumTextures.push_back(i);
            continue;
        }

        std::string filename =
            ResolveFilename(tex.second.parameters.GetOneString("filename", ""));
        if (filename.empty())
            continue;
        if (!FileExists(filename)) {
            Error(&tex.second.loc, "%s: file not found.", filename);
            ++nMissingTextures;
        }

        if (seenSpectrumTextureFilenames.find(filename) ==
            seenSpectrumTextureFilenames.end()) {
            seenSpectrumTextureFilenames.insert(filename);
            parallelSpectrumTextures.push_back(i);
        } else
            serialSpectrumTextures.push_back(i);
    }

    if (nMissingTextures > 0)
        ErrorExit("%d missing textures", nMissingTextures);

    LOG_VERBOSE("Loading %d,%d textures in parallel, %d,%d serially",
                parallelFloatTextures.size(), parallelSpectrumTextures.size(),
                serialFloatTextures.size(), serialSpectrumTextures.size());

    // Load textures in parallel
    std::mutex mutex;

    ParallelFor(0, parallelFloatTextures.size(), [&](int64_t i) {
        const auto &tex = floatTextures[parallelFloatTextures[i]];

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        // Pass nullptr for the textures, since they shouldn't be accessed
        // anyway.
        TextureParameterDictionary texDict(&tex.second.parameters, nullptr);
        FloatTexture t = FloatTexture::Create(tex.second.texName, renderFromTexture,
                                              texDict, &tex.second.loc, alloc, gpu);
        std::lock_guard<std::mutex> lock(mutex);
        textures.floatTextures[tex.first] = t;
    });

    ParallelFor(0, parallelSpectrumTextures.size(), [&](int64_t i) {
        const auto &tex = spectrumTextures[parallelSpectrumTextures[i]];

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        // nullptr for the textures, as above.
        TextureParameterDictionary texDict(&tex.second.parameters, nullptr);
        SpectrumTexture albedoTex =
            SpectrumTexture::Create(tex.second.texName, renderFromTexture, texDict,
                                    SpectrumType::Albedo, &tex.second.loc, alloc, gpu);
        // These should be fast since they should hit the texture cache
        SpectrumTexture unboundedTex =
            SpectrumTexture::Create(tex.second.texName, renderFromTexture, texDict,
                                    SpectrumType::Unbounded, &tex.second.loc, alloc, gpu);
        SpectrumTexture illumTex = SpectrumTexture::Create(
            tex.second.texName, renderFromTexture, texDict, SpectrumType::Illuminant,
            &tex.second.loc, alloc, gpu);

        std::lock_guard<std::mutex> lock(mutex);
        textures.albedoSpectrumTextures[tex.first] = albedoTex;
        textures.unboundedSpectrumTextures[tex.first] = unboundedTex;
        textures.illuminantSpectrumTextures[tex.first] = illumTex;
    });

    LOG_VERBOSE("Loading serial textures");
    // And do the rest serially
    for (size_t index : serialFloatTextures) {
        const auto &tex = floatTextures[index];

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        TextureParameterDictionary texDict(&tex.second.parameters, &textures);
        FloatTexture t = FloatTexture::Create(tex.second.texName, renderFromTexture,
                                              texDict, &tex.second.loc, alloc, gpu);
        textures.floatTextures[tex.first] = t;
    }
    for (size_t index : serialSpectrumTextures) {
        const auto &tex = spectrumTextures[index];

        if (tex.second.renderFromObject.IsAnimated())
            Warning(&tex.second.loc, "Animated world to texture transform not supported. "
                                     "Using start transform.");

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        TextureParameterDictionary texDict(&tex.second.parameters, &textures);
        SpectrumTexture albedoTex =
            SpectrumTexture::Create(tex.second.texName, renderFromTexture, texDict,
                                    SpectrumType::Albedo, &tex.second.loc, alloc, gpu);
        SpectrumTexture unboundedTex =
            SpectrumTexture::Create(tex.second.texName, renderFromTexture, texDict,
                                    SpectrumType::Unbounded, &tex.second.loc, alloc, gpu);
        SpectrumTexture illumTex = SpectrumTexture::Create(
            tex.second.texName, renderFromTexture, texDict, SpectrumType::Illuminant,
            &tex.second.loc, alloc, gpu);

        textures.albedoSpectrumTextures[tex.first] = albedoTex;
        textures.unboundedSpectrumTextures[tex.first] = unboundedTex;
        textures.illuminantSpectrumTextures[tex.first] = illumTex;
    }

    LOG_VERBOSE("Done creating textures");
    return textures;
}

std::map<std::string, Medium> ParsedScene::CreateMedia(Allocator alloc) const {
    std::map<std::string, Medium> mediaMap;

    for (const auto &m : media) {
        std::string type = m.second.parameters.GetOneString("type", "");
        if (type.empty())
            ErrorExit(&m.second.loc, "No parameter string \"type\" found for medium.");

        if (m.second.renderFromObject.IsAnimated())
            Warning(&m.second.loc,
                    "Animated transformation provided for medium. Only the "
                    "start transform will be used.");
        Medium medium = Medium::Create(type, m.second.parameters,
                                       m.second.renderFromObject.startTransform,
                                       &m.second.loc, alloc);
        mediaMap[m.first] = medium;
    }

    return mediaMap;
}

std::vector<Light>
ParsedScene::CreateLights(Allocator alloc,
                          const std::map<std::string, Medium> &media, const NamedTextures &textures,
                          std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights) {
    auto findMedium = [&media](const std::string &s, const FileLoc *loc) -> Medium {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        return iter->second;
    };

    auto getAlphaTexture = [&](const ParameterDictionary &parameters,
                               const FileLoc *loc) -> FloatTexture {
        std::string alphaTexName = parameters.GetTexture("alpha");
        if (!alphaTexName.empty()) {
            if (auto iter = textures.floatTextures.find(alphaTexName);
                iter != textures.floatTextures.end())
                return iter->second;
            else
                ErrorExit(loc, "%s: couldn't find float texture for \"alpha\" parameter.",
                          alphaTexName);
        } else if (Float alpha = parameters.GetOneFloat("alpha", 1.f); alpha < 1.f)
            return alloc.new_object<FloatConstantTexture>(alpha);
        else
            return nullptr;
    };

    // Lights (area lights will be done later, with shapes...)
    LOG_VERBOSE("Starting lights");
    std::vector<Light> lights;
    lights.reserve(lights.size() + areaLights.size());
    for (const auto &light : this->lights) {
        Medium outsideMedium = findMedium(light.medium, &light.loc);
        if (light.renderFromObject.IsAnimated())
            Warning(&light.loc,
                    "Animated lights aren't supported. Using the start transform.");
        Light l = Light::Create(
            light.name, light.parameters, light.renderFromObject.startTransform,
            camera.cameraTransform, outsideMedium, &light.loc, alloc);

        lights.push_back(l);
    }

    // Area Lights
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto &sh = shapes[i];

        if (sh.lightIndex == -1)
            continue;

        std::string materialName;
        if (!sh.materialName.empty()) {
            auto iter = std::find_if(namedMaterials.begin(), namedMaterials.end(),
                                     [&](auto iter) { return iter.first == sh.materialName; });
            if (iter == namedMaterials.end())
                ErrorExit(&sh.loc, "%s: no named material defined.", sh.materialName);
            CHECK(iter->second.parameters.GetStringArray("type").size() > 0);
            materialName = iter->second.parameters.GetOneString("type", "");
        } else {
            CHECK_LT(sh.materialIndex, materials.size());
            materialName = materials[sh.materialIndex].name;
        }
        if (materialName == "interface" || materialName == "none" || materialName == "") {
            Warning(&sh.loc, "Ignoring area light specification for shape "
                    "with \"interface\" material.");
            continue;
        }

        pstd::vector<pbrt::Shape> shapeObjects =
            Shape::Create(sh.name, sh.renderFromObject, sh.objectFromRender,
                          sh.reverseOrientation, sh.parameters, &sh.loc, alloc);

        FloatTexture alphaTex = getAlphaTexture(sh.parameters, &sh.loc);

        pbrt::MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                                 findMedium(sh.outsideMedium, &sh.loc));

        pstd::vector<Light> *shapeLights = new pstd::vector<Light>;
        const auto &areaLightEntity = areaLights[sh.lightIndex];
        for (pbrt::Shape ps : shapeObjects) {
            Light area = Light::CreateArea(areaLightEntity.name,
                                           areaLightEntity.parameters,
                                           *sh.renderFromObject, mi, ps,
                                           alphaTex, &areaLightEntity.loc, alloc);
            if (area) {
                lights.push_back(area);
                shapeLights->push_back(area);
            }
        }

        shapeIndexToAreaLights[i] = shapeLights;
    }

    LOG_VERBOSE("Finished Lights");
    return lights;
}

ParsedScene::Scene
ParsedScene::CreateAggregate(Allocator alloc, NamedTextures &textures,
                             const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
                             const std::map<std::string, Medium> &media) {
    // Materials
    LOG_VERBOSE("Starting materials");
    std::map<std::string, pbrt::Material> namedMaterials;
    std::vector<pbrt::Material> materials;
    CreateMaterials(textures, alloc, &namedMaterials, &materials);
    LOG_VERBOSE("Finished materials");

    auto findMedium = [&media](const std::string &s, const FileLoc *loc) -> Medium {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        return iter->second;
    };

    // Primitives
    auto getAlphaTexture = [&](const ParameterDictionary &parameters,
                               const FileLoc *loc) -> FloatTexture {
        std::string alphaTexName = parameters.GetTexture("alpha");
        if (!alphaTexName.empty()) {
            if (textures.floatTextures.find(alphaTexName) != textures.floatTextures.end())
                return textures.floatTextures[alphaTexName];
            else
                ErrorExit(loc, "%s: couldn't find float texture for \"alpha\" parameter.",
                          alphaTexName);
        } else if (Float alpha = parameters.GetOneFloat("alpha", 1.f); alpha < 1.f)
            return alloc.new_object<FloatConstantTexture>(alpha);
        else
            return nullptr;
    };

    // Non-animated shapes
    auto CreatePrimitivesForShapes =
        [&](std::vector<ShapeSceneEntity> &shapes) -> std::vector<Primitive> {
        // Parallelize Shape::Create calls, which will in turn
        // parallelize PLY file loading, etc...
        pstd::vector<pstd::vector<pbrt::Shape>> shapeVectors(shapes.size());
        ParallelFor(0, shapes.size(), [&](int64_t i) {
            const auto &sh = shapes[i];
            shapeVectors[i] =
                Shape::Create(sh.name, sh.renderFromObject, sh.objectFromRender,
                              sh.reverseOrientation, sh.parameters, &sh.loc, alloc);
        });

        std::vector<Primitive> primitives;
        for (size_t i = 0; i < shapes.size(); ++i) {
            auto &sh = shapes[i];
            pstd::vector<pbrt::Shape> &shapes = shapeVectors[i];
            if (shapes.empty())
                continue;

            FloatTexture alphaTex = getAlphaTexture(sh.parameters, &sh.loc);
            sh.parameters.ReportUnused();  // do now so can grab alpha...

            pbrt::Material mtl = nullptr;
            if (!sh.materialName.empty()) {
                auto iter = namedMaterials.find(sh.materialName);
                if (iter == namedMaterials.end())
                    ErrorExit(&sh.loc, "%s: no named material defined.", sh.materialName);
                mtl = iter->second;
            } else {
                CHECK_LT(sh.materialIndex, materials.size());
                mtl = materials[sh.materialIndex];
            }

            pbrt::MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                                     findMedium(sh.outsideMedium, &sh.loc));

            auto iter = shapeIndexToAreaLights.find(i);
            for (size_t j = 0; j < shapes.size(); ++j) {
                // Possibly create area light for shape
                Light area = nullptr;
                // Will not be present in the map if it has an "interface"
                // material...
                if (sh.lightIndex != -1 && iter != shapeIndexToAreaLights.end())
                    area = (*iter->second)[j];

                if (area == nullptr && !mi.IsMediumTransition() && !alphaTex)
                    primitives.push_back(new SimplePrimitive(shapes[j], mtl));
                else
                    primitives.push_back(
                        new GeometricPrimitive(shapes[j], mtl, area, mi, alphaTex));
            }
            sh.parameters.FreeParameters();
            sh = ShapeSceneEntity();
        }
        return primitives;
    };

    LOG_VERBOSE("Starting shapes");
    std::vector<Primitive> primitives = CreatePrimitivesForShapes(shapes);

    shapes.clear();
    shapes.shrink_to_fit();

    // Animated shapes
    auto CreatePrimitivesForAnimatedShapes =
        [&](std::vector<AnimatedShapeSceneEntity> &shapes) -> std::vector<Primitive> {
        std::vector<Primitive> primitives;
        primitives.reserve(shapes.size());

        for (auto &sh : shapes) {
            pstd::vector<pbrt::Shape> shapes =
                Shape::Create(sh.name, sh.identity, sh.identity, sh.reverseOrientation,
                              sh.parameters, &sh.loc, alloc);
            if (shapes.empty())
                continue;

            FloatTexture alphaTex = getAlphaTexture(sh.parameters, &sh.loc);
            sh.parameters.ReportUnused();  // do now so can grab alpha...

            // Create initial shape or shapes for animated shape

            pbrt::Material mtl = nullptr;
            if (!sh.materialName.empty()) {
                auto iter = namedMaterials.find(sh.materialName);
                if (iter == namedMaterials.end())
                    ErrorExit(&sh.loc, "%s: no named material defined.", sh.materialName);
                mtl = iter->second;
            } else {
                CHECK_LT(sh.materialIndex, materials.size());
                mtl = materials[sh.materialIndex];
            }

            pbrt::MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                                     findMedium(sh.outsideMedium, &sh.loc));

            std::vector<Primitive> prims;
            for (auto &s : shapes) {
                if (sh.lightIndex != -1) {
                    CHECK(sh.renderFromObject.IsAnimated());
                    ErrorExit(&sh.loc, "Animated area lights are not supported.");
                }

                if (!mi.IsMediumTransition() && !alphaTex)
                    prims.push_back(new SimplePrimitive(s, mtl));
                else
                    prims.push_back(new GeometricPrimitive(s, mtl, nullptr /* area light */, mi, alphaTex));
            }

            // TODO: could try to be greedy or even segment them according
            // to same sh.renderFromObject...

            // Create single _Primitive_ for _prims_
            if (prims.size() > 1) {
                Primitive bvh = new BVHAggregate(std::move(prims));
                prims.clear();
                prims.push_back(bvh);
            }
            primitives.push_back(new AnimatedPrimitive(prims[0], sh.renderFromObject));

            sh.parameters.FreeParameters();
            sh = AnimatedShapeSceneEntity();
        }
        return primitives;
    };
    std::vector<Primitive> animatedPrimitives =
        CreatePrimitivesForAnimatedShapes(animatedShapes);
    primitives.insert(primitives.end(), animatedPrimitives.begin(),
                      animatedPrimitives.end());

    animatedShapes.clear();
    animatedShapes.shrink_to_fit();
    LOG_VERBOSE("Finished shapes");

    // Instance definitions
    LOG_VERBOSE("Starting instances");
    std::map<std::string, Primitive> instanceDefinitions;
    std::mutex instanceDefinitionsMutex;
    std::vector<std::map<std::string, InstanceDefinitionSceneEntity>::iterator>
        instanceDefinitionIterators;
    for (auto iter = this->instanceDefinitions.begin();
         iter != this->instanceDefinitions.end(); ++iter)
        instanceDefinitionIterators.push_back(iter);
    ParallelFor(0, instanceDefinitionIterators.size(), [&](int64_t i) {
        auto &inst = *instanceDefinitionIterators[i];

        std::vector<Primitive> instancePrimitives =
            CreatePrimitivesForShapes(inst.second.shapes);
        std::vector<Primitive> movingInstancePrimitives =
            CreatePrimitivesForAnimatedShapes(inst.second.animatedShapes);
        instancePrimitives.insert(instancePrimitives.end(),
                                  movingInstancePrimitives.begin(),
                                  movingInstancePrimitives.end());

        if (instancePrimitives.size() > 1) {
            Primitive bvh = new BVHAggregate(std::move(instancePrimitives));
            instancePrimitives.clear();
            instancePrimitives.push_back(bvh);
        }

        std::lock_guard<std::mutex> lock(instanceDefinitionsMutex);
        if (instancePrimitives.empty())
            instanceDefinitions[inst.first] = nullptr;
        else
            instanceDefinitions[inst.first] = instancePrimitives[0];

        inst.second = InstanceDefinitionSceneEntity();
    });

    this->instanceDefinitions.clear();

    // Instances
    for (const auto &inst : instances) {
        auto iter = instanceDefinitions.find(inst.name);
        if (iter == instanceDefinitions.end())
            ErrorExit(&inst.loc, "%s: object instance not defined", inst.name);

        if (iter->second == nullptr)
            // empty instance
            continue;

        if (inst.renderFromInstance)
            primitives.push_back(
                new TransformedPrimitive(iter->second, inst.renderFromInstance));
        else {
            primitives.push_back(
                new AnimatedPrimitive(iter->second, *inst.renderFromInstanceAnim));
            delete inst.renderFromInstanceAnim;
        }
    }

    instances.clear();
    instances.shrink_to_fit();
    LOG_VERBOSE("Finished instances");

    // Accelerator
    LOG_VERBOSE("Starting top-level accelerator");
    Primitive accel = nullptr;
    if (!primitives.empty())
        accel = CreateAccelerator(accelerator.name, std::move(primitives),
                                  accelerator.parameters);
    LOG_VERBOSE("Finished top-level accelerator");

    return Scene{accel, materials, namedMaterials};
}

// FormattingScene Method Definitions
FormattingScene::~FormattingScene() {
    if (errorExit)
        ErrorExit("Fatal errors during scene updating.");
}

void FormattingScene::Option(const std::string &name, const std::string &value,
                             FileLoc loc) {
    std::string nName = normalizeArg(name);
    if (nName == "msereferenceimage" || nName == "msereferenceout")
        Printf("%sOption \"%s\" \"%s\"\n", indent(), name, value);
    else
        Printf("%sOption \"%s\" %s\n", indent(), name, value);
}

void FormattingScene::Identity(FileLoc loc) {
    Printf("%sIdentity\n", indent());
}

void FormattingScene::Translate(Float dx, Float dy, Float dz, FileLoc loc) {
    Printf("%sTranslate %f %f %f\n", indent(), dx, dy, dz);
}

void FormattingScene::Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc) {
    Printf("%sRotate %f %f %f %f\n", indent(), angle, ax, ay, az);
}

void FormattingScene::Scale(Float sx, Float sy, Float sz, FileLoc loc) {
    Printf("%sScale %f %f %f\n", indent(), sx, sy, sz);
}

void FormattingScene::LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                             Float ux, Float uy, Float uz, FileLoc loc) {
    Printf("%sLookAt %f %f %f\n%s    %f %f %f\n%s    %f %f %f\n", indent(), ex, ey, ez,
           indent(), lx, ly, lz, indent(), ux, uy, uz);
}

void FormattingScene::ConcatTransform(Float transform[16], FileLoc loc) {
    Printf("%sConcatTransform [ ", indent());
    for (int i = 0; i < 16; ++i)
        Printf("%f ", transform[i]);
    Printf(" ]\n");
}

void FormattingScene::Transform(Float transform[16], FileLoc loc) {
    Printf("%sTransform [ ", indent());
    for (int i = 0; i < 16; ++i)
        Printf("%f ", transform[i]);
    Printf(" ]\n");
}

void FormattingScene::CoordinateSystem(const std::string &name, FileLoc loc) {
    Printf("%sCoordinateSystem \"%s\"\n", indent(), name);
}

void FormattingScene::CoordSysTransform(const std::string &name, FileLoc loc) {
    Printf("%sCoordSysTransform \"%s\"\n", indent(), name);
}

void FormattingScene::ActiveTransformAll(FileLoc loc) {
    Printf("%sActiveTransform All\n", indent());
}

void FormattingScene::ActiveTransformEndTime(FileLoc loc) {
    Printf("%sActiveTransform EndTime\n", indent());
}

void FormattingScene::ActiveTransformStartTime(FileLoc loc) {
    Printf("%sActiveTransform StartTime\n", indent());
}

void FormattingScene::TransformTimes(Float start, Float end, FileLoc loc) {
    Printf("%sTransformTimes %f %f\n", indent(), start, end);
}

void FormattingScene::ColorSpace(const std::string &n, FileLoc loc) {
    Printf("%sColorSpace \"%s\"\n", indent(), n);
}

void FormattingScene::PixelFilter(const std::string &name, ParsedParameterVector params,
                                  FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    std::string extra;
    if (upgrade) {
        std::vector<Float> xr = dict.GetFloatArray("xwidth");
        if (xr.size() == 1) {
            dict.RemoveFloat("xwidth");
            extra += StringPrintf("%s\"float xradius\" [ %f ]\n", indent(1), xr[0]);
        }
        std::vector<Float> yr = dict.GetFloatArray("ywidth");
        if (yr.size() == 1) {
            dict.RemoveFloat("ywidth");
            extra += StringPrintf("%s\"float yradius\" [ %f ]\n", indent(1), yr[0]);
        }

        if (name == "gaussian") {
            std::vector<Float> alpha = dict.GetFloatArray("alpha");
            if (alpha.size() == 1) {
                dict.RemoveFloat("alpha");
                extra += StringPrintf("%s\"float sigma\" [ %f ]\n", indent(1),
                                      1 / std::sqrt(2 * alpha[0]));
            }
        }
    }

    Printf("%sPixelFilter \"%s\"\n", indent(), name);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingScene::Film(const std::string &type, ParsedParameterVector params,
                           FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    std::string extra;
    if (upgrade) {
        std::vector<Float> m = dict.GetFloatArray("maxsampleluminance");
        if (!m.empty()) {
            dict.RemoveFloat("maxsampleluminance");
            extra +=
                StringPrintf("%s\"float maxcomponentvalue\" [ %f ]\n", indent(1), m[0]);
        }
        std::vector<Float> s = dict.GetFloatArray("scale");
        if (!s.empty()) {
            dict.RemoveFloat("scale");
            extra += StringPrintf("%s\"float iso\" [ %f ]\n", indent(1), 100 * s[0]);
        }
    }

    if (upgrade && type == "image")
        Printf("%sFilm \"rgb\"\n", indent());
    else
        Printf("%sFilm \"%s\"\n", indent(), type);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingScene::Sampler(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    if (upgrade) {
        if (name == "lowdiscrepancy" || name == "02sequence")
            Printf("%sSampler \"paddedsobol\"\n", indent());
        else if (name == "maxmindist")
            Printf("%sSampler \"pmj02bn\"\n", indent());
        else
            Printf("%sSampler \"%s\"\n", indent(), name);
    } else
        Printf("%sSampler \"%s\"\n", indent(), name);
    std::cout << dict.ToParameterList(catIndentCount);
}

void FormattingScene::Accelerator(const std::string &name, ParsedParameterVector params,
                                  FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    Printf("%sAccelerator \"%s\"\n%s", indent(), name,
           dict.ToParameterList(catIndentCount));
}

void FormattingScene::Integrator(const std::string &name, ParsedParameterVector params,
                                 FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    std::string extra;
    if (upgrade) {
        dict.RemoveFloat("rrthreshold");

        if (name == "sppm") {
            dict.RemoveInt("imagewritefrequency");

            std::vector<int> iterations = dict.GetIntArray("numiterations");
            if (!iterations.empty()) {
                dict.RemoveInt("numiterations");
                Warning(
                    &loc,
                    "The SPPM integrator no longer takes a \"numiterations\" parameter. "
                    "This value is now set via the Sampler's number of pixel samples.");
            }
        }
        std::string lss = dict.GetOneString("lightsamplestrategy", "");
        if (lss == "spatial") {
            dict.RemoveString("lightsamplestrategy");
            extra += indent(1) + "\"string lightsamplestrategy\" \"bvh\"\n";
        }
    }

    if (upgrade && name == "directlighting") {
        Printf("%sIntegrator \"path\"\n", indent());
        extra += indent(1) + "\"integer maxdepth\" [ 1 ]\n";
    } else
        Printf("%sIntegrator \"%s\"\n", indent(), name);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingScene::Camera(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    if (upgrade && name == "environment")
        Printf("%sCamera \"spherical\" \"string mapping\" \"equirectangular\"\n",
               indent());
    else
        Printf("%sCamera \"%s\"\n", indent(), name);
    if (upgrade && name == "realistic")
        dict.RemoveBool("simpleweighting");

    std::cout << dict.ToParameterList(catIndentCount);
}

void FormattingScene::MakeNamedMedium(const std::string &name,
                                      ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    if (upgrade && name == "heterogeneous")
        Printf("%sMakeNamedMedium \"%s\"\n%s\n", indent(), "uniformgrid",
               dict.ToParameterList(catIndentCount));
    else
        Printf("%sMakeNamedMedium \"%s\"\n%s\n", indent(), name,
               dict.ToParameterList(catIndentCount));
}

void FormattingScene::MediumInterface(const std::string &insideName,
                                      const std::string &outsideName, FileLoc loc) {
    Printf("%sMediumInterface \"%s\" \"%s\"\n", indent(), insideName, outsideName);
}

void FormattingScene::WorldBegin(FileLoc loc) {
    Printf("\n\nWorldBegin\n\n");
}

void FormattingScene::AttributeBegin(FileLoc loc) {
    Printf("\n%sAttributeBegin\n", indent());
    catIndentCount += 4;
}

void FormattingScene::AttributeEnd(FileLoc loc) {
    catIndentCount -= 4;
    Printf("%sAttributeEnd\n", indent());
}

void FormattingScene::Attribute(const std::string &target, ParsedParameterVector params,
                                FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    Printf("%sAttribute \"%s\" ", indent(), target);
    if (params.size() == 1)
        // just one; put it on the same line
        std::cout << dict.ToParameterList(0) << '\n';
    else
        std::cout << '\n' << dict.ToParameterList(catIndentCount);
}

void FormattingScene::TransformBegin(FileLoc loc) {
    Printf("%sTransformBegin\n", indent());
    catIndentCount += 4;
}

void FormattingScene::TransformEnd(FileLoc loc) {
    catIndentCount -= 4;
    Printf("%sTransformEnd\n", indent());
}

void FormattingScene::Texture(const std::string &name, const std::string &type,
                              const std::string &texname, ParsedParameterVector params,
                              FileLoc loc) {
    if (upgrade) {
        if (definedTextures.find(name) != definedTextures.end()) {
            static int count = 0;
            Warning(&loc, "%s: renaming multiply-defined texture", name);
            definedTextures[name] = StringPrintf("%s-renamed-%d", name, count++);
        } else
            definedTextures[name] = name;
    }

    if (upgrade && texname == "scale") {
        // This is easier to do in the raw ParsedParameterVector...
        if (type == "float") {
            for (ParsedParameter *p : params) {
                if (p->name == "tex1")
                    p->name = "tex";
                if (p->name == "tex2")
                    p->name = "scale";
            }
        } else {
            // more subtle: rename one of them as float, but need one of them
            // to be an RGB and spectrally constant...
            bool foundRGB = false, foundTexture = false;
            for (ParsedParameter *p : params) {
                if (p->name != "tex1" && p->name != "tex2")
                    continue;

                if (p->type == "rgb") {
                    if (foundRGB) {
                        ErrorExitDeferred(
                            &p->loc,
                            "Two \"rgb\" textures found for \"scale\" "
                            "texture \"%s\". Please manually edit the file to "
                            "upgrade.",
                            name);
                        return;
                    }
                    if (p->floats.size() != 3) {
                        ErrorExitDeferred(
                            &p->loc, "Didn't find 3 values for \"rgb\" \"%s\".", p->name);
                        return;
                    }
                    if (p->floats[0] != p->floats[1] || p->floats[1] != p->floats[2]) {
                        ErrorExitDeferred(&p->loc,
                                          "Non-constant \"rgb\" value found for "
                                          "\"scale\" texture parameter \"%s\". Please "
                                          "manually "
                                          "edit the file to upgrade.",
                                          p->name);
                        return;
                    }

                    foundRGB = true;
                    p->type = "float";
                    p->name = "scale";
                    p->floats.resize(1);
                } else {
                    if (foundTexture) {
                        ErrorExitDeferred(
                            &p->loc,
                            "Two textures found for \"scale\" "
                            "texture \"%s\". Please manually edit the file to "
                            "upgrade.",
                            name);
                        return;
                    }
                    p->name = "tex";
                    foundTexture = true;
                }
            }
        }
    }

    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    if (upgrade)
        dict.RenameUsedTextures(definedTextures);

    std::string extra;
    if (upgrade) {
        if (texname == "imagemap") {
            std::vector<uint8_t> tri = dict.GetBoolArray("trilinear");
            if (tri.size() == 1) {
                dict.RemoveBool("trilinear");
                extra += indent(1) + "\"string filter\" ";
                extra += tri[0] != 0u ? "\"trilinear\"\n" : "\"bilinear\"\n";
            }
        }

        if (texname == "imagemap" || texname == "ptex") {
            Float gamma = dict.GetOneFloat("gamma", 0);
            if (gamma != 0) {
                dict.RemoveFloat("gamma");
                extra +=
                    indent(1) + StringPrintf("\"string encoding \" \"gamma %f\" ", gamma);
            } else {
                std::vector<uint8_t> gamma = dict.GetBoolArray("gamma");
                if (gamma.size() == 1) {
                    dict.RemoveBool("gamma");
                    extra += indent(1) + "\"string encoding\" ";
                    extra += gamma[0] != 0u ? "\"sRGB\"\n" : "\"linear\"\n";
                }
            }
        }
    }

    if (upgrade) {
        if (type == "color")
            Printf("%sTexture \"%s\" \"spectrum\" \"%s\"\n", indent(),
                   definedTextures[name], texname);
        else
            Printf("%sTexture \"%s\" \"%s\" \"%s\"\n", indent(), definedTextures[name],
                   type, texname);
    } else
        Printf("%sTexture \"%s\" \"%s\" \"%s\"\n", indent(), name, type, texname);

    std::cout << extra << dict.ToParameterList(catIndentCount);
}

std::string FormattingScene::upgradeMaterialIndex(const std::string &name,
                                                  ParameterDictionary *dict,
                                                  FileLoc loc) const {
    if (name != "glass" && name != "uber")
        return "";

    std::string tex = dict->GetTexture("index");
    if (!tex.empty()) {
        if (!dict->GetTexture("eta").empty()) {
            ErrorExitDeferred(
                &loc, R"(Material "%s" has both "index" and "eta" parameters.)", name);
            return "";
        }

        dict->RemoveTexture("index");
        return indent(1) + StringPrintf("\"texture eta\" \"%s\"\n", tex);
    } else {
        auto index = dict->GetFloatArray("index");
        if (index.empty())
            return "";
        if (index.size() != 1) {
            ErrorExitDeferred(&loc, "Multiple values provided for \"index\" parameter.");
            return "";
        }
        if (!dict->GetFloatArray("eta").empty()) {
            ErrorExitDeferred(
                &loc, R"(Material "%s" has both "index" and "eta" parameters.)", name);
            return "";
        }

        Float value = index[0];
        dict->RemoveFloat("index");
        return indent(1) + StringPrintf("\"float eta\" [ %f ]\n", value);
    }
}

std::string FormattingScene::upgradeMaterial(std::string *name, ParameterDictionary *dict,
                                             FileLoc loc) const {
    std::string extra = upgradeMaterialIndex(*name, dict, loc);

    dict->RenameParameter("bumpmap", "displacement");

    auto removeParamSilentIfConstant = [&](const char *paramName, Float value) {
        pstd::optional<RGB> rgb = dict->GetOneRGB(paramName);
        bool matches = (rgb && rgb->r == value && rgb->g == value && rgb->b == value);

        if (!matches &&
            !dict->GetSpectrumArray(paramName, SpectrumType::Unbounded, {}).empty())
            Warning(&loc,
                    "Parameter is being removed when converting "
                    "to \"%s\" material: %s",
                    *name, dict->ToParameterDefinition(paramName));
        dict->RemoveSpectrum(paramName);
        dict->RemoveTexture(paramName);
        return matches;
    };

    if (*name == "uber") {
        *name = "coateddiffuse";
        if (removeParamSilentIfConstant("Ks", 0)) {
            *name = "diffuse";
            dict->RemoveFloat("eta");
            dict->RemoveFloat("roughness");
        }
        removeParamSilentIfConstant("Kr", 0);
        removeParamSilentIfConstant("Kt", 0);
        dict->RenameParameter("Kd", "reflectance");

        if (!dict->GetTexture("opacity").empty()) {
            ErrorExitDeferred(&loc, "Non-opaque \"opacity\" texture in \"uber\" "
                                    "material not supported "
                                    "in pbrt-v4. Please edit the file manually.");
            return "";
        }

        if (dict->GetSpectrumArray("opacity", SpectrumType::Unbounded, {}).empty())
            return "";

        pstd::optional<RGB> opacity = dict->GetOneRGB("opacity");
        if (opacity && opacity->r == 1 && opacity->g == 1 && opacity->b == 1) {
            dict->RemoveSpectrum("opacity");
            return "";
        }

        ErrorExitDeferred(&loc, "A non-opaque \"opacity\" in the \"uber\" "
                                "material is not supported "
                                "in pbrt-v4. Please edit the file manually.");
    } else if (*name == "mix") {
        // Convert the amount to a scalar
        pstd::optional<RGB> rgb = dict->GetOneRGB("amount");
        if (rgb) {
            if (rgb->r == rgb->g && rgb->g == rgb->b)
                extra += indent(1) + StringPrintf("\"float amount\" [ %f ]\n", rgb->r);
            else {
                Float avg = (rgb->r + rgb->g + rgb->b) / 3;
                Warning(&loc, "Changing RGB \"amount\" (%f, %f, %f) to scalar average %f",
                        rgb->r, rgb->g, rgb->b, avg);
                extra += indent(1) + StringPrintf("\"float amount\" [ %f ]\n", avg);
            }
        } else if (dict->GetSpectrumArray("amount", SpectrumType::Unbounded, {}).size() >
                   0)
            ErrorExitDeferred(
                &loc, "Unable to update non-RGB spectrum \"amount\" to a scalar: %s",
                dict->ToParameterDefinition("amount"));

        dict->RemoveSpectrum("amount");

        // And rename...
        std::string m1 = dict->GetOneString("namedmaterial1", "");
        if (m1.empty())
            ErrorExitDeferred(
                &loc, "Didn't find \"namedmaterial1\" parameter for \"mix\" material.");
        dict->RemoveString("namedmaterial1");

        std::string m2 = dict->GetOneString("namedmaterial2", "");
        if (m2.empty())
            ErrorExitDeferred(
                &loc, "Didn't find \"namedmaterial1\" parameter for \"mix\" material.");
        dict->RemoveString("namedmaterial2");

        // Note: swapped order vs pbrt-v3!
        extra +=
            indent(1) + StringPrintf("\"string materials\" [ \"%s\" \"%s\" ]\n", m2, m1);
    } else if (*name == "substrate") {
        *name = "coateddiffuse";
        removeParamSilentIfConstant("Ks", 1);
        dict->RenameParameter("Kd", "reflectance");
    } else if (*name == "glass") {
        *name = "dielectric";
        removeParamSilentIfConstant("Kr", 1);
        removeParamSilentIfConstant("Kt", 1);
    } else if (*name == "plastic") {
        *name = "coateddiffuse";
        if (removeParamSilentIfConstant("Ks", 0)) {
            *name = "diffuse";
            dict->RemoveFloat("roughness");
            dict->RemoveFloat("eta");
        }
        dict->RenameParameter("Kd", "reflectance");
    } else if (*name == "fourier")
        Warning(&loc, "\"fourier\" material is no longer supported. (But there "
                      "is \"measured\"!)");
    else if (*name == "kdsubsurface") {
        *name = "subsurface";
        dict->RenameParameter("Kd", "reflectance");
    } else if (*name == "matte") {
        *name = "diffuse";
        dict->RenameParameter("Kd", "reflectance");
    } else if (*name == "metal") {
        *name = "conductor";
        removeParamSilentIfConstant("Kr", 1);
    } else if (*name == "translucent") {
        *name = "diffusetransmission";

        dict->RenameParameter("Kd", "transmittance");

        removeParamSilentIfConstant("reflect", 0);
        removeParamSilentIfConstant("transmit", 1);

        removeParamSilentIfConstant("Ks", 0);
        dict->RemoveFloat("roughness");
    } else if (*name == "mirror") {
        *name = "conductor";
        extra += indent(1) + "\"float roughness\" [ 0 ]\n";
        extra += indent(1) + "\"spectrum eta\" [ \"metal-Ag-eta\" ]\n";
        extra += indent(1) + "\"spectrum k\" [ \"metal-Ag-k\" ]\n";

        removeParamSilentIfConstant("Kr", 0);
    } else if (*name == "disney") {
        *name = "diffuse";
        dict->RenameParameter("color", "reflectance");
    } else if (name->empty() || *name == "none")
        *name = "interface";

    return extra;
}

void FormattingScene::Material(const std::string &name, ParsedParameterVector params,
                               FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    if (upgrade)
        dict.RenameUsedTextures(definedTextures);

    std::string extra;
    std::string newName = name;
    if (upgrade)
        extra = upgradeMaterial(&newName, &dict, loc);

#if 0
    // Hack for landscape upgrade
    if (upgrade && name == "mix") {
        ParameterDictionary dict(params, RGBColorSpace::sRGB);
        const ParameterDictionary &d1 = namedMaterialDictionaries[dict.GetOneString("namedmaterial1", "")];
        const ParameterDictionary &d2 = namedMaterialDictionaries[dict.GetOneString("namedmaterial2", "")];

        if (!d1.GetTexture("reflectance").empty() &&
            !d2.GetTexture("transmittance").empty()) {
            Printf("%sMaterial \"diffusetransmission\"\n", indent());
            Printf("%s\"texture reflectance\" \"%s\"\n", indent(1), d1.GetTexture("reflectance"));
            Printf("%s\"texture transmittance\" \"%s\"\n", indent(1), d2.GetTexture("transmittance"));

            if (!d1.GetTexture("displacement").empty())
                Printf("%s\"texture displacement\" \"%s\"\n", indent(1), d1.GetTexture("displacement"));
            else if (!d2.GetTexture("displacement").empty())
                Printf("%s\"texture displacement\" \"%s\"\n", indent(1), d2.GetTexture("displacement"));

            Printf("%s\"float scale\" 0.5\n", indent(1));

            return;
        }
    }
#endif

    Printf("%sMaterial \"%s\"\n", indent(), newName);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingScene::MakeNamedMaterial(const std::string &name,
                                        ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);

    if (upgrade) {
        dict.RenameUsedTextures(definedTextures);

        if (definedNamedMaterials.find(name) != definedNamedMaterials.end()) {
            static int count = 0;
            Warning(&loc, "%s: renaming multiply-defined named material", name);
            definedNamedMaterials[name] = StringPrintf("%s-renamed-%d", name, count++);
        } else
            definedNamedMaterials[name] = name;
        Printf("%sMakeNamedMaterial \"%s\"\n", indent(), definedNamedMaterials[name]);
    } else
        Printf("%sMakeNamedMaterial \"%s\"\n", indent(), name);

    std::string extra;
    if (upgrade) {
        std::string matName = dict.GetOneString("type", "");
        extra = upgradeMaterial(&matName, &dict, loc);
        dict.RemoveString("type");
        extra = indent(1) + StringPrintf("\"string type\" [ \"%s\" ]\n", matName) + extra;
    }
    std::cout << extra << dict.ToParameterList(catIndentCount);

    if (upgrade)
        namedMaterialDictionaries[definedNamedMaterials[name]] = std::move(dict);
}

void FormattingScene::NamedMaterial(const std::string &name, FileLoc loc) {
    Printf("%sNamedMaterial \"%s\"\n", indent(), name);
}

static bool upgradeRGBToScale(ParameterDictionary *dict, const char *name,
                              Float *totalScale) {
    std::vector<Spectrum> s = dict->GetSpectrumArray(name, SpectrumType::Unbounded, {});
    if (s.empty())
        return true;

    pstd::optional<RGB> rgb = dict->GetOneRGB(name);
    if (!rgb || rgb->r != rgb->g || rgb->g != rgb->b)
        return false;

    *totalScale *= rgb->r;
    dict->RemoveSpectrum(name);
    return true;
}

static std::string upgradeMapname(const FormattingScene &scene,
                                  ParameterDictionary *dict) {
    std::string n = dict->GetOneString("mapname", "");
    if (n.empty())
        return "";

    dict->RemoveString("mapname");
    return scene.indent(1) + StringPrintf("\"string filename\" \"%s\"\n", n);
}

void FormattingScene::LightSource(const std::string &name, ParsedParameterVector params,
                                  FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);

    Printf("%sLightSource \"%s\"\n", indent(), name);

    std::string extra;
    if (upgrade) {
        Float totalScale = 1;
        if (!upgradeRGBToScale(&dict, "scale", &totalScale)) {
            ErrorExitDeferred(dict.loc("scale"),
                              "In pbrt-v4, \"scale\" is a \"float\" parameter "
                              "to light sources. "
                              "Please modify your scene file manually.");
            return;
        }
        dict.RemoveInt("samples");

        if (dict.GetOneString("mapname", "").empty() == false) {
            if (name == "infinite" && !upgradeRGBToScale(&dict, "L", &totalScale)) {
                ErrorExitDeferred(
                    dict.loc("L"),
                    "Non-constant \"L\" is no longer supported with "
                    "\"mapname\" for "
                    "the \"infinite\" light source. Please upgrade your scene "
                    "file manually.");
                return;
            } else if (name == "projection" &&
                       !upgradeRGBToScale(&dict, "I", &totalScale)) {
                ErrorExitDeferred(
                    dict.loc("I"),
                    "\"I\" is no longer supported with \"mapname\" for "
                    "the \"projection\" light source. Please upgrade your scene "
                    "file manually.");
                return;
            }
        }

        totalScale *= dict.UpgradeBlackbody("I");
        totalScale *= dict.UpgradeBlackbody("L");

        // Do this after we've handled infinite "L" with a map, since
        // it removes the "mapname" parameter from the dictionary.
        extra += upgradeMapname(*this, &dict);

        if (totalScale != 1) {
            totalScale *= dict.GetOneFloat("scale", 1.f);
            dict.RemoveFloat("scale");
            Printf("%s\"float scale\" [%f]\n", indent(1), totalScale);
        }
    }

    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingScene::AreaLightSource(const std::string &name,
                                      ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    std::string extra;
    Float totalScale = 1;
    if (upgrade) {
        if (!upgradeRGBToScale(&dict, "scale", &totalScale)) {
            ErrorExitDeferred(dict.loc("scale"),
                              "In pbrt-v4, \"scale\" is a \"float\" parameter "
                              "to light sources. "
                              "Please modify your scene file manually.");
            return;
        }

        totalScale *= dict.UpgradeBlackbody("L");

        if (name == "area")
            Printf("%sAreaLightSource \"diffuse\"\n", indent());
        else
            Printf("%sAreaLightSource \"%s\"\n", indent(), name);
        dict.RemoveInt("nsamples");
    } else
        Printf("%sAreaLightSource \"%s\"\n", indent(), name);

    if (totalScale != 1)
        Printf("%s\"float scale\" [%f]\n", indent(1), totalScale);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

static std::string upgradeTriMeshUVs(const FormattingScene &scene,
                                     ParameterDictionary *dict) {
    std::vector<Point2f> uv = dict->GetPoint2fArray("st");
    if (!uv.empty())
        dict->RemovePoint2f("st");
    else {
        auto upgradeFloatArray = [&](const char *name) {
            std::vector<Float> fuv = dict->GetFloatArray(name);
            if (fuv.empty())
                return;

            std::vector<Point2f> tempUVs;
            tempUVs.reserve(fuv.size() / 2);
            for (size_t i = 0; i < fuv.size() / 2; ++i)
                tempUVs.push_back(Point2f(fuv[2 * i], fuv[2 * i + 1]));
            dict->RemoveFloat(name);
            uv = tempUVs;
        };
        upgradeFloatArray("uv");
        upgradeFloatArray("st");
    }

    if (uv.empty())
        return "";

    std::string s = scene.indent(1) + "\"point2 uv\" [ ";
    for (size_t i = 0; i < uv.size(); ++i) {
        s += StringPrintf("%f %f ", uv[i][0], uv[i][1]);
        if ((i + 1) % 4 == 0) {
            s += "\n";
            s += scene.indent(2);
        }
    }
    s += "]\n";
    return s;
}

void FormattingScene::Shape(const std::string &name, ParsedParameterVector params,
                            FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);

    if (toPly && name == "trianglemesh") {
        std::vector<int> vi = dict.GetIntArray("indices");

        if (vi.size() < 500) {
            // It's a small mesh; don't bother with a PLY file after all.
            Printf("%sShape \"%s\"\n", indent(), name);
            std::cout << dict.ToParameterList(catIndentCount);
        } else {
            static int count = 1;
            const char *plyPrefix =
                getenv("PLY_PREFIX") != nullptr ? getenv("PLY_PREFIX") : "mesh";
            std::string fn = StringPrintf("%s_%05d.ply", plyPrefix, count++);

            class Transform identity;
            const TriangleMesh *mesh =
                Triangle::CreateMesh(&identity, false, dict, &loc, Allocator());
            if (!mesh->WritePLY(fn))
                ErrorExit(&loc, "%s: unable to write PLY file.", fn);

            dict.RemoveInt("indices");
            dict.RemovePoint3f("P");
            dict.RemovePoint2f("uv");
            dict.RemoveNormal3f("N");
            dict.RemoveVector3f("S");
            dict.RemoveInt("faceIndices");

            Printf("%sShape \"plymesh\" \"string filename\" \"%s\"\n", indent(), fn);
            std::cout << dict.ToParameterList(catIndentCount);
        }
        return;
    }

    Printf("%sShape \"%s\"\n", indent(), name);

    if (upgrade) {
        if (name == "trianglemesh") {
            // Remove indices if they're [0 1 2] and we have a single triangle
            auto indices = dict.GetIntArray("indices");
            if (indices.size() == 3 && dict.GetPoint3fArray("P").size() == 3 &&
                indices[0] == 0 && indices[1] == 1 && indices[2] == 2)
                dict.RemoveInt("indices");
        }

        if (name == "bilinearmesh") {
            // Remove indices if they're [0 1 2 3] and we have a single blp
            auto indices = dict.GetIntArray("indices");
            if (indices.size() == 4 && dict.GetPoint3fArray("P").size() == 4 &&
                indices[0] == 0 && indices[1] == 1 && indices[2] == 2 && indices[3] == 3)
                dict.RemoveInt("indices");
        }

        if (name == "loopsubdiv") {
            auto levels = dict.GetIntArray("nlevels");
            if (!levels.empty()) {
                Printf("%s\"integer levels\" [ %d ]\n", indent(1), levels[0]);
                dict.RemoveInt("nlevels");
            }
        }
        if (name == "trianglemesh" || name == "plymesh") {
            dict.RemoveBool("discarddegenerateUVs");
            dict.RemoveTexture("shadowalpha");
        }

        if (name == "trianglemesh") {
            std::string extra = upgradeTriMeshUVs(*this, &dict);
            std::cout << extra;
        }

        dict.RenameParameter("Kd", "reflectance");
    }

    std::cout << dict.ToParameterList(catIndentCount);
}

void FormattingScene::ReverseOrientation(FileLoc loc) {
    Printf("%sReverseOrientation\n", indent());
}

void FormattingScene::ObjectBegin(const std::string &name, FileLoc loc) {
    if (upgrade) {
        if (definedObjectInstances.find(name) != definedObjectInstances.end()) {
            static int count = 0;
            Warning(&loc, "%s: renaming multiply-defined object instance", name);
            definedObjectInstances[name] = StringPrintf("%s-renamed-%d", name, count++);
        } else
            definedObjectInstances[name] = name;
        Printf("%sObjectBegin \"%s\"\n", indent(), definedObjectInstances[name]);
    } else
        Printf("%sObjectBegin \"%s\"\n", indent(), name);
}

void FormattingScene::ObjectEnd(FileLoc loc) {
    Printf("%sObjectEnd\n", indent());
}

void FormattingScene::ObjectInstance(const std::string &name, FileLoc loc) {
    if (upgrade) {
        if (definedObjectInstances.find(name) == definedObjectInstances.end())
            // this is legit if we're upgrading multiple files separately...
            Printf("%sObjectInstance \"%s\"\n", indent(), name);
        else
            // use the most recent renaming of it
            Printf("%sObjectInstance \"%s\"\n", indent(), definedObjectInstances[name]);
    } else
        Printf("%sObjectInstance \"%s\"\n", indent(), name);
}

void FormattingScene::EndOfFiles() {}

}  // namespace pbrt
