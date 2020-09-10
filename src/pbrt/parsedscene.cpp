// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/parsedscene.h>

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

// ParsedScene::GraphicsState Definition
struct ParsedScene::GraphicsState {
    // Graphics State Methods
    GraphicsState();

    // Graphics State
    std::string currentInsideMedium, currentOutsideMedium;

    int currentMaterialIndex = 0;
    std::string currentMaterialName;

    std::string areaLightName;
    ParameterDictionary areaLightParams;
    FileLoc areaLightLoc;

    const RGBColorSpace *colorSpace = RGBColorSpace::sRGB;

    bool reverseOrientation = false;

    ParsedParameterVector shapeAttributes;
    ParsedParameterVector lightAttributes;
    ParsedParameterVector materialAttributes;
    ParsedParameterVector mediumAttributes;
    ParsedParameterVector textureAttributes;
};

ParsedScene::GraphicsState::GraphicsState() {
    currentMaterialIndex = 0;
}

// API State Macros
#define VERIFY_INITIALIZED(func)                             \
    if (currentApiState == APIState::Uninitialized) {        \
        Error(&loc,                                          \
              "pbrtInit() must be before calling \"%s()\". " \
              "Ignoring.",                                   \
              func);                                         \
        return;                                              \
    } else /* swallow trailing semicolon */

#define FOR_ACTIVE_TRANSFORMS(expr)           \
    for (int i = 0; i < MaxTransforms; ++i)   \
        if (activeTransformBits & (1 << i)) { \
            expr                              \
        }

#define WARN_IF_ANIMATED_TRANSFORM(func)                                 \
    do {                                                                 \
        if (curTransform.IsAnimated())                                   \
            Warning(&loc,                                                \
                    "Animated transformations set; ignoring for \"%s\" " \
                    "and using the start transform only",                \
                    func);                                               \
    } while (false) /* swallow trailing semicolon */

#define VERIFY_OPTIONS(func)                               \
    VERIFY_INITIALIZED(func);                              \
    if (currentApiState == APIState::WorldBlock) {         \
        Error(&loc,                                        \
              "Options cannot be set inside world block; " \
              "\"%s\" not allowed.  Ignoring.",            \
              func);                                       \
        return;                                            \
    } else /* swallow trailing semicolon */
#define VERIFY_WORLD(func)                                     \
    VERIFY_INITIALIZED(func);                                  \
    if (currentApiState == APIState::OptionsBlock) {           \
        Error(&loc,                                            \
              "Scene description must be inside world block; " \
              "\"%s\" not allowed. Ignoring.",                 \
              func);                                           \
        return;                                                \
    } else /* swallow trailing semicolon */

STAT_MEMORY_COUNTER("Memory/TransformCache", transformCacheBytes);
STAT_PERCENT("Geometry/TransformCache hits", nTransformCacheHits, nTransformCacheLookups);

// TransformCache Method Definitions
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
    // Allocate _graphicsState_
    graphicsState = new GraphicsState;

    // Set scene defaults
    ParameterDictionary dict({}, RGBColorSpace::sRGB);
    materials.push_back(SceneEntity("diffuse", dict, {}));

    accelerator.name = "bvh";

    camera.name = "perspective";

    sampler.name = "pmj02bn";

    filter.name = "gaussian";

    film.name = "rgb";
    film.parameters = ParameterDictionary({}, RGBColorSpace::sRGB);

    integrator.name = "volpath";
}

void ParsedScene::Identity(FileLoc loc) {
    VERIFY_INITIALIZED("Identity");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = pbrt::Transform();)
}

void ParsedScene::Translate(Float dx, Float dy, Float dz, FileLoc loc) {
    VERIFY_INITIALIZED("Translate");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] =
                              curTransform[i] * pbrt::Translate(Vector3f(dx, dy, dz));)
}

void ParsedScene::CoordinateSystem(const std::string &name, FileLoc loc) {
    VERIFY_INITIALIZED("CoordinateSystem");
    namedCoordinateSystems[name] = curTransform;
}

void ParsedScene::CoordSysTransform(const std::string &name, FileLoc loc) {
    VERIFY_INITIALIZED("CoordSysTransform");
    if (namedCoordinateSystems.find(name) != namedCoordinateSystems.end())
        curTransform = namedCoordinateSystems[name];
    else
        Warning(&loc, "Couldn't find named coordinate system \"%s\"", name);
}

void ParsedScene::WorldBegin(FileLoc loc) {
    VERIFY_OPTIONS("WorldBegin");
    currentApiState = APIState::WorldBlock;
    for (int i = 0; i < MaxTransforms; ++i)
        curTransform[i] = pbrt::Transform();
    activeTransformBits = AllTransformsBits;
    namedCoordinateSystems["world"] = curTransform;
}

void ParsedScene::AttributeBegin(FileLoc loc) {
    VERIFY_WORLD("AttributeBegin");

    pushedGraphicsStates.push_back(*graphicsState);

    pushedTransforms.push_back(curTransform);
    pushedActiveTransformBits.push_back(activeTransformBits);

    pushStack.push_back(std::make_pair('a', loc));
}

void ParsedScene::AttributeEnd(FileLoc loc) {
    VERIFY_WORLD("AttributeEnd");
    if (pushedGraphicsStates.empty()) {
        Error(&loc, "Unmatched AttributeEnd encountered. Ignoring it.");
        return;
    }

    // NOTE: must keep the following consistent with code in ObjectEnd
    *graphicsState = std::move(pushedGraphicsStates.back());
    pushedGraphicsStates.pop_back();

    curTransform = pushedTransforms.back();
    pushedTransforms.pop_back();
    activeTransformBits = pushedActiveTransformBits.back();
    pushedActiveTransformBits.pop_back();

    if (pushStack.back().first == 't')
        ErrorExitDeferred(
            &loc, "Mismatched nesting: open TransformBegin from %s at AttributeEnd",
            pushStack.back().second);
    else if (pushStack.back().first == 'o')
        ErrorExitDeferred(&loc,
                          "Mismatched nesting: open ObjectBegin from %s at AttributeEnd",
                          pushStack.back().second);
    else
        CHECK_EQ(pushStack.back().first, 'a');
    pushStack.pop_back();
}

void ParsedScene::Attribute(const std::string &target, ParsedParameterVector attrib,
                            FileLoc loc) {
    VERIFY_INITIALIZED("Attribute");

    ParsedParameterVector *currentAttributes = nullptr;
    if (target == "shape") {
        currentAttributes = &graphicsState->shapeAttributes;
    } else if (target == "light") {
        currentAttributes = &graphicsState->lightAttributes;
    } else if (target == "material") {
        currentAttributes = &graphicsState->materialAttributes;
    } else if (target == "medium") {
        currentAttributes = &graphicsState->mediumAttributes;
    } else if (target == "texture") {
        currentAttributes = &graphicsState->textureAttributes;
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
        p->colorSpace = graphicsState->colorSpace;
        currentAttributes->push_back(p);
    }
}

void ParsedScene::LightSource(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    VERIFY_WORLD("LightSource");
    ParameterDictionary dict(std::move(params), graphicsState->lightAttributes,
                             graphicsState->colorSpace);
    AnimatedTransform renderFromLight(GetCTM(0), transformStartTime, GetCTM(1),
                                      transformEndTime);

    lights.push_back(LightSceneEntity(name, std::move(dict), loc, renderFromLight,
                                      graphicsState->currentOutsideMedium));
}

void ParsedScene::Shape(const std::string &name, ParsedParameterVector params,
                        FileLoc loc) {
    VERIFY_WORLD("Shape");

    ParameterDictionary dict(std::move(params), graphicsState->shapeAttributes,
                             graphicsState->colorSpace);

    int areaLightIndex = -1;
    if (!graphicsState->areaLightName.empty()) {
        areaLights.push_back(SceneEntity(graphicsState->areaLightName,
                                         graphicsState->areaLightParams,
                                         graphicsState->areaLightLoc));
        areaLightIndex = areaLights.size() - 1;
    }

    if (CTMIsAnimated()) {
        std::vector<AnimatedShapeSceneEntity> *as = &animatedShapes;
        if (currentInstance != nullptr) {
            if (!graphicsState->areaLightName.empty())
                Warning(&loc, "Area lights not supported with object instancing");
            as = &currentInstance->animatedShapes;
        }

        AnimatedTransform renderFromShape(GetCTM(0), transformStartTime, GetCTM(1),
                                          transformEndTime);
        const class Transform *identity = transformCache.Lookup(pbrt::Transform());

        as->push_back(AnimatedShapeSceneEntity(
            {name, std::move(dict), loc, renderFromShape, identity,
             graphicsState->reverseOrientation, graphicsState->currentMaterialIndex,
             graphicsState->currentMaterialName, areaLightIndex,
             graphicsState->currentInsideMedium, graphicsState->currentOutsideMedium}));
    } else {
        std::vector<ShapeSceneEntity> *s = &shapes;
        if (currentInstance != nullptr) {
            if (!graphicsState->areaLightName.empty())
                Warning(&loc, "Area lights not supported with object instancing");
            s = &currentInstance->shapes;
        }

        const class Transform *renderFromObject = transformCache.Lookup(GetCTM(0));
        const class Transform *objectFromRender =
            transformCache.Lookup(Inverse(*renderFromObject));

        s->push_back(ShapeSceneEntity(
            {name, std::move(dict), loc, renderFromObject, objectFromRender,
             graphicsState->reverseOrientation, graphicsState->currentMaterialIndex,
             graphicsState->currentMaterialName, areaLightIndex,
             graphicsState->currentInsideMedium, graphicsState->currentOutsideMedium}));
    }
}

void ParsedScene::ObjectBegin(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("ObjectBegin");
    pushedGraphicsStates.push_back(*graphicsState);
    pushedTransforms.push_back(curTransform);
    pushedActiveTransformBits.push_back(activeTransformBits);

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
    *graphicsState = std::move(pushedGraphicsStates.back());
    pushedGraphicsStates.pop_back();

    curTransform = pushedTransforms.back();
    pushedTransforms.pop_back();
    activeTransformBits = pushedActiveTransformBits.back();
    pushedActiveTransformBits.pop_back();

    ++nObjectInstancesCreated;

    if (pushStack.back().first == 't')
        ErrorExitDeferred(&loc,
                          "Mismatched nesting: open TransformBegin from %s at ObjectEnd",
                          pushStack.back().second);
    else if (pushStack.back().first == 'a')
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
            GetCTM(0) * worldFromRender, transformStartTime, GetCTM(1) * worldFromRender,
            transformEndTime);

        instances.push_back(
            InstanceSceneEntity(name, loc, animatedRenderFromInstance, nullptr));
    } else {
        const class Transform *renderFromInstance =
            transformCache.Lookup(GetCTM(0) * worldFromRender);

        instances.push_back(
            InstanceSceneEntity(name, loc, AnimatedTransform(), renderFromInstance));
    }
}

void ParsedScene::EndOfFiles() {
    if (currentApiState != APIState::WorldBlock)
        ErrorExitDeferred("End of files before \"WorldBegin\".");

    // Ensure there are no pushed graphics states
    while (!pushedGraphicsStates.empty()) {
        ErrorExitDeferred("Missing end to AttributeBegin");
        pushedGraphicsStates.pop_back();
        pushedTransforms.pop_back();
    }
    while (!pushedTransforms.empty()) {
        ErrorExitDeferred("Missing end to TransformBegin");
        pushedTransforms.pop_back();
    }

    if (errorExit)
        ErrorExit("Fatal errors during scene construction");
}

ParsedScene::~ParsedScene() {
    delete graphicsState;
}

void ParsedScene::Option(const std::string &name, const std::string &value, FileLoc loc) {
    std::string nName = normalizeArg(name);

    VERIFY_INITIALIZED("Option");
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
    VERIFY_INITIALIZED("Transform");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = Transpose(
                              pbrt::Transform(SquareMatrix<4>(pstd::MakeSpan(tr, 16))));)
}

void ParsedScene::ConcatTransform(Float tr[16], FileLoc loc) {
    VERIFY_INITIALIZED("ConcatTransform");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] =
                              curTransform[i] * Transpose(pbrt::Transform(SquareMatrix<4>(
                                                    pstd::MakeSpan(tr, 16))));)
}

void ParsedScene::Rotate(Float angle, Float dx, Float dy, Float dz, FileLoc loc) {
    VERIFY_INITIALIZED("Rotate");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = curTransform[i] *
                                            pbrt::Rotate(angle, Vector3f(dx, dy, dz));)
}

void ParsedScene::Scale(Float sx, Float sy, Float sz, FileLoc loc) {
    VERIFY_INITIALIZED("Scale");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = curTransform[i] * pbrt::Scale(sx, sy, sz);)
}

void ParsedScene::LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                         Float ux, Float uy, Float uz, FileLoc loc) {
    VERIFY_INITIALIZED("LookAt");
    class Transform lookAt =
        pbrt::LookAt(Point3f(ex, ey, ez), Point3f(lx, ly, lz), Vector3f(ux, uy, uz));
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = curTransform[i] * lookAt;);
}

void ParsedScene::ActiveTransformAll(FileLoc loc) {
    activeTransformBits = AllTransformsBits;
}

void ParsedScene::ActiveTransformEndTime(FileLoc loc) {
    activeTransformBits = EndTransformBits;
}

void ParsedScene::ActiveTransformStartTime(FileLoc loc) {
    activeTransformBits = StartTransformBits;
}

void ParsedScene::TransformTimes(Float start, Float end, FileLoc loc) {
    VERIFY_OPTIONS("TransformTimes");
    transformStartTime = start;
    transformEndTime = end;
}

void ParsedScene::ColorSpace(const std::string &n, FileLoc loc) {
    VERIFY_INITIALIZED("RGBColorSpace");
    if (const RGBColorSpace *cs = RGBColorSpace::GetNamed(n))
        graphicsState->colorSpace = cs;
    else
        Error(&loc, "%s: color space unknown", n);
}

void ParsedScene::PixelFilter(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState->colorSpace);
    VERIFY_OPTIONS("PixelFilter");
    filter = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::Film(const std::string &type, ParsedParameterVector params,
                       FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState->colorSpace);
    VERIFY_OPTIONS("Film");
    film = SceneEntity(type, std::move(dict), loc);
}

void ParsedScene::Sampler(const std::string &name, ParsedParameterVector params,
                          FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState->colorSpace);

    VERIFY_OPTIONS("Sampler");
    sampler = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::Accelerator(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState->colorSpace);
    VERIFY_OPTIONS("Accelerator");
    accelerator = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::Integrator(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState->colorSpace);

    VERIFY_OPTIONS("Integrator");
    integrator = SceneEntity(name, std::move(dict), loc);
}

void ParsedScene::Camera(const std::string &name, ParsedParameterVector params,
                         FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState->colorSpace);

    VERIFY_OPTIONS("Camera");

    TransformSet cameraFromWorld = curTransform;
    TransformSet worldFromCamera = Inverse(curTransform);
    namedCoordinateSystems["camera"] = Inverse(cameraFromWorld);

    CameraTransform cameraTransform(AnimatedTransform(
        worldFromCamera[0], transformStartTime, worldFromCamera[1], transformEndTime));
    renderFromWorld = cameraTransform.RenderFromWorld();

    camera = CameraSceneEntity(name, std::move(dict), loc, cameraTransform,
                               graphicsState->currentOutsideMedium);
}

void ParsedScene::MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                                  FileLoc loc) {
    VERIFY_INITIALIZED("MakeNamedMedium");
    WARN_IF_ANIMATED_TRANSFORM("MakeNamedMedium");
    ParameterDictionary dict(std::move(params), graphicsState->mediumAttributes,
                             graphicsState->colorSpace);

    if (media.find(name) != media.end()) {
        ErrorExitDeferred(&loc, "Named medium \"%s\" redefined.", name);
        return;
    }

    AnimatedTransform renderFromMedium(GetCTM(0), transformStartTime, GetCTM(1),
                                       transformEndTime);

    media[name] = TransformedSceneEntity(name, std::move(dict), loc, renderFromMedium);
}

void ParsedScene::MediumInterface(const std::string &insideName,
                                  const std::string &outsideName, FileLoc loc) {
    VERIFY_INITIALIZED("MediumInterface");
    graphicsState->currentInsideMedium = insideName;
    graphicsState->currentOutsideMedium = outsideName;
}

void ParsedScene::TransformBegin(FileLoc loc) {
    VERIFY_WORLD("TransformBegin");
    pushedTransforms.push_back(curTransform);
    pushedActiveTransformBits.push_back(activeTransformBits);
    pushStack.push_back(std::make_pair('t', loc));
}

void ParsedScene::TransformEnd(FileLoc loc) {
    VERIFY_WORLD("TransformEnd");
    if (pushedTransforms.empty()) {
        Error(&loc, "Unmatched TransformEnd encountered. Ignoring it.");
        return;
    }
    curTransform = pushedTransforms.back();
    pushedTransforms.pop_back();
    activeTransformBits = pushedActiveTransformBits.back();
    pushedActiveTransformBits.pop_back();

    if (pushStack.back().first == 'a')
        ErrorExitDeferred(
            &loc, "Mismatched nesting: open AttributeBegin from %s at TransformEnd",
            pushStack.back().second);
    else if (pushStack.back().first == 'o')
        ErrorExitDeferred(&loc,
                          "Mismatched nesting: open ObjectBegin from %s at TransformEnd",
                          pushStack.back().second);
    else
        CHECK_EQ(pushStack.back().first, 't');
    pushStack.pop_back();
}

void ParsedScene::Texture(const std::string &name, const std::string &type,
                          const std::string &texname, ParsedParameterVector params,
                          FileLoc loc) {
    VERIFY_WORLD("Texture");

    ParameterDictionary dict(std::move(params), graphicsState->textureAttributes,
                             graphicsState->colorSpace);

    AnimatedTransform renderFromTexture(GetCTM(0), transformStartTime, GetCTM(1),
                                        transformEndTime);

    if (type != "float" && type != "spectrum") {
        ErrorExitDeferred(
            &loc, "%s: texture type unknown. Must be \"float\" or \"spectrum\".", type);
        return;
    }

    std::vector<std::pair<std::string, TextureSceneEntity>> &textures =
        (type == "float") ? floatTextures : spectrumTextures;

    for (const auto &tex : textures)
        if (tex.first == name) {
            ErrorExitDeferred(&loc, "Redefining texture \"%s\".", name);
            return;
        }

    textures.push_back(std::make_pair(
        name, TextureSceneEntity(texname, std::move(dict), loc, renderFromTexture)));
}

void ParsedScene::Material(const std::string &name, ParsedParameterVector params,
                           FileLoc loc) {
    VERIFY_WORLD("Material");
    ParameterDictionary dict(std::move(params), graphicsState->materialAttributes,
                             graphicsState->colorSpace);
    materials.push_back(SceneEntity(name, std::move(dict), loc));
    graphicsState->currentMaterialIndex = materials.size() - 1;
    graphicsState->currentMaterialName.clear();
}

void ParsedScene::MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    VERIFY_WORLD("MakeNamedMaterial");

    ParameterDictionary dict(std::move(params), graphicsState->materialAttributes,
                             graphicsState->colorSpace);

    // Note: O(n). FIXME?
    for (const auto &nm : namedMaterials)
        if (nm.first == name) {
            ErrorExitDeferred(&loc, "%s: named material redefined.", name);
            return;
        }

    namedMaterials.push_back(std::make_pair(name, SceneEntity("", std::move(dict), loc)));
}

void ParsedScene::NamedMaterial(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("NamedMaterial");
    graphicsState->currentMaterialName = name;
    graphicsState->currentMaterialIndex = -1;
}

void ParsedScene::AreaLightSource(const std::string &name, ParsedParameterVector params,
                                  FileLoc loc) {
    VERIFY_WORLD("AreaLightSource");
    graphicsState->areaLightName = name;
    graphicsState->areaLightParams = ParameterDictionary(
        std::move(params), graphicsState->lightAttributes, graphicsState->colorSpace);
    graphicsState->areaLightLoc = loc;
}

void ParsedScene::ReverseOrientation(FileLoc loc) {
    VERIFY_WORLD("ReverseOrientation");
    graphicsState->reverseOrientation = !graphicsState->reverseOrientation;
}

void ParsedScene::CreateMaterials(
    /*const*/ std::map<std::string, FloatTextureHandle> &floatTextures,
    /*const*/ std::map<std::string, SpectrumTextureHandle> &spectrumTextures,
    Allocator alloc, std::map<std::string, MaterialHandle> *namedMaterialsOut,
    std::vector<MaterialHandle> *materialsOut) const {
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
        TextureParameterDictionary texDict(&mtl.parameters, &floatTextures,
                                           &spectrumTextures);
        MaterialHandle m =
            MaterialHandle::Create(type, texDict, *namedMaterialsOut, &mtl.loc, alloc);
        (*namedMaterialsOut)[name] = m;
    }

    // Regular materials
    materialsOut->reserve(materials.size());
    for (const auto &mtl : materials) {
        TextureParameterDictionary texDict(&mtl.parameters, &floatTextures,
                                           &spectrumTextures);
        MaterialHandle m = MaterialHandle::Create(mtl.name, texDict, *namedMaterialsOut,
                                                  &mtl.loc, alloc);
        materialsOut->push_back(m);
    }
}

void ParsedScene::CreateTextures(
    std::map<std::string, FloatTextureHandle> *floatTextureMap,
    std::map<std::string, SpectrumTextureHandle> *spectrumTextureMap, Allocator alloc,
    bool gpu) const {
    std::set<std::string> seenFloatTextureFilenames, seenSpectrumTextureFilenames;
    std::vector<size_t> parallelFloatTextures, serialFloatTextures;
    std::vector<size_t> parallelSpectrumTextures, serialSpectrumTextures;

    // Figure out which textures to load in parallel
    // Need to be careful since two textures can use the same image file;
    // we only want to load it once in that case...
    for (size_t i = 0; i < floatTextures.size(); ++i) {
        const auto &tex = floatTextures[i];

        if (tex.second.renderFromObject.IsAnimated())
            Warning(&tex.second.loc,
                    "Animated world to texture transforms are not supported. "
                    "Using start transform.");

        if (tex.second.texName != "imagemap") {
            serialFloatTextures.push_back(i);
            continue;
        }

        std::string filename =
            ResolveFilename(tex.second.parameters.GetOneString("filename", ""));
        if (filename.empty())
            continue;

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

        if (tex.second.texName != "imagemap") {
            serialSpectrumTextures.push_back(i);
            continue;
        }

        std::string filename =
            ResolveFilename(tex.second.parameters.GetOneString("filename", ""));
        if (filename.empty())
            continue;

        if (seenSpectrumTextureFilenames.find(filename) ==
            seenSpectrumTextureFilenames.end()) {
            seenSpectrumTextureFilenames.insert(filename);
            parallelSpectrumTextures.push_back(i);
        } else
            serialSpectrumTextures.push_back(i);
    }

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
        TextureParameterDictionary texDict(&tex.second.parameters, nullptr, nullptr);
        FloatTextureHandle t = FloatTextureHandle::Create(
            tex.second.texName, renderFromTexture, texDict, &tex.second.loc, alloc, gpu);
        std::lock_guard<std::mutex> lock(mutex);
        (*floatTextureMap)[tex.first] = t;
    });

    ParallelFor(0, parallelSpectrumTextures.size(), [&](int64_t i) {
        const auto &tex = spectrumTextures[parallelSpectrumTextures[i]];

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        // nullptr for the textures, as above.
        TextureParameterDictionary texDict(&tex.second.parameters, nullptr, nullptr);
        SpectrumTextureHandle t = SpectrumTextureHandle::Create(
            tex.second.texName, renderFromTexture, texDict, &tex.second.loc, alloc, gpu);
        std::lock_guard<std::mutex> lock(mutex);
        (*spectrumTextureMap)[tex.first] = t;
    });

    LOG_VERBOSE("Loading serial textures");
    // And do the rest serially
    for (size_t index : serialFloatTextures) {
        const auto &tex = floatTextures[index];

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        TextureParameterDictionary texDict(&tex.second.parameters, floatTextureMap,
                                           spectrumTextureMap);
        FloatTextureHandle t = FloatTextureHandle::Create(
            tex.second.texName, renderFromTexture, texDict, &tex.second.loc, alloc, gpu);
        (*floatTextureMap)[tex.first] = t;
    }
    for (size_t index : serialSpectrumTextures) {
        const auto &tex = spectrumTextures[index];

        if (tex.second.renderFromObject.IsAnimated())
            Warning(&tex.second.loc, "Animated world to texture transform not supported. "
                                     "Using start transform.");

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        TextureParameterDictionary texDict(&tex.second.parameters, floatTextureMap,
                                           spectrumTextureMap);
        SpectrumTextureHandle t = SpectrumTextureHandle::Create(
            tex.second.texName, renderFromTexture, texDict, &tex.second.loc, alloc, gpu);
        (*spectrumTextureMap)[tex.first] = t;
    }

    LOG_VERBOSE("Done creating textures");
}

std::map<std::string, MediumHandle> ParsedScene::CreateMedia(Allocator alloc) const {
    std::map<std::string, MediumHandle> mediaMap;

    for (const auto &m : media) {
        std::string type = m.second.parameters.GetOneString("type", "");
        if (type.empty())
            ErrorExit(&m.second.loc, "No parameter string \"type\" found for medium.");

        if (m.second.renderFromObject.IsAnimated())
            Warning(&m.second.loc,
                    "Animated transformation provided for medium. Only the "
                    "start transform will be used.");
        MediumHandle medium = MediumHandle::Create(
            type, m.second.parameters, m.second.renderFromObject.startTransform,
            &m.second.loc, alloc);
        mediaMap[m.first] = medium;
    }

    return mediaMap;
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
        if (name == "sppm") {
            dict.RemoveInt("imagewritefrequency");

            std::vector<int> iterations = dict.GetIntArray("numiterations");
            if (!iterations.empty()) {
                dict.RemoveInt("numiterations");
                extra += indent(1) +
                         StringPrintf("\"integer iterations\" [ %d ]\n", iterations[0]);
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
                    if (p->numbers.size() != 3) {
                        ErrorExitDeferred(
                            &p->loc, "Didn't find 3 values for \"rgb\" \"%s\".", p->name);
                        return;
                    }
                    if (p->numbers[0] != p->numbers[1] ||
                        p->numbers[1] != p->numbers[2]) {
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
                    p->numbers.resize(1);
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
            !dict->GetSpectrumArray(paramName, SpectrumType::Reflectance, {}).empty())
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

        if (dict->GetSpectrumArray("opacity", SpectrumType::Reflectance, {}).empty())
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
        } else if (dict->GetSpectrumArray("amount", SpectrumType::General, {}).size() > 0)
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
    }

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
    std::vector<SpectrumHandle> s =
        dict->GetSpectrumArray(name, SpectrumType::General, {});
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
        dict.RemoveInt("nsamples");

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
