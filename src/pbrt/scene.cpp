// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/scene.h>

#include <pbrt/cpu/aggregates.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/memory.h>
#endif  // PBRT_BUILD_GPU_RENDERER
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

InternCache<std::string> SceneEntity::internedStrings(Allocator{});

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

SceneProcessor::~SceneProcessor() {}

std::string SceneStateManager::ToString() const {
    return StringPrintf(
        "[ SceneStateManager camera: %s film: %s sampler: %s integrator: %s "
        "filter: %s accelerator: %s ]",
        camera, film, sampler, integrator, filter, accelerator);
}

SceneStateManager::GraphicsState::GraphicsState() {
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

STAT_COUNTER("Scene/Object instances created", nObjectInstancesCreated);
STAT_COUNTER("Scene/Object instances used", nObjectInstancesUsed);

// SceneStateManager Method Definitions
SceneStateManager::SceneStateManager(SceneProcessor *sceneProcessor)
    : sceneProcessor(sceneProcessor),
      transformCache(Options->useGPU ? Allocator(&CUDATrackedMemoryResource::singleton) :
                                       Allocator()) {
    // Set scene defaults
    camera.name = "perspective";
    sampler.name = "zsobol";
    filter.name = "gaussian";
    integrator.name = "volpath";

    ParameterDictionary dict({}, RGBColorSpace::sRGB);
    currentMaterialIndex = sceneProcessor->AddMaterial(SceneEntity("diffuse", dict, {}));
    accelerator.name = "bvh";
    film.name = "rgb";
    film.parameters = ParameterDictionary({}, RGBColorSpace::sRGB);
}

void SceneStateManager::ReverseOrientation(FileLoc loc) {
    VERIFY_WORLD("ReverseOrientation");
    graphicsState.reverseOrientation = !graphicsState.reverseOrientation;
}

void SceneStateManager::ColorSpace(const std::string &name, FileLoc loc) {
    if (const RGBColorSpace *cs = RGBColorSpace::GetNamed(name))
        graphicsState.colorSpace = cs;
    else
        Error(&loc, "%s: color space unknown", name);
}

void SceneStateManager::Identity(FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = pbrt::Transform();)
}

void SceneStateManager::Translate(Float dx, Float dy, Float dz, FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = graphicsState.ctm[i] *
                                                 pbrt::Translate(Vector3f(dx, dy, dz));)
}

void SceneStateManager::CoordinateSystem(const std::string &name, FileLoc loc) {
    namedCoordinateSystems[name] = graphicsState.ctm;
}

void SceneStateManager::CoordSysTransform(const std::string &name, FileLoc loc) {
    if (namedCoordinateSystems.find(name) != namedCoordinateSystems.end())
        graphicsState.ctm = namedCoordinateSystems[name];
    else
        Warning(&loc, "Couldn't find named coordinate system \"%s\"", name);
}

void SceneStateManager::Camera(const std::string &name, ParsedParameterVector params,
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

void SceneStateManager::AttributeBegin(FileLoc loc) {
    VERIFY_WORLD("AttributeBegin");
    pushedGraphicsStates.push_back(graphicsState);
    pushStack.push_back(std::make_pair('a', loc));
}

void SceneStateManager::AttributeEnd(FileLoc loc) {
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

void SceneStateManager::Attribute(const std::string &target, ParsedParameterVector attrib,
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

void SceneStateManager::WorldBegin(FileLoc loc) {
    VERIFY_OPTIONS("WorldBegin");
    currentBlock = BlockState::WorldBlock;
    for (int i = 0; i < MaxTransforms; ++i)
        graphicsState.ctm[i] = pbrt::Transform();
    graphicsState.activeTransformBits = AllTransformsBits;
    namedCoordinateSystems["world"] = graphicsState.ctm;

    // Pass these along now
    sceneProcessor->SetFilm(std::move(film));
    sceneProcessor->SetSampler(std::move(sampler));
    sceneProcessor->SetIntegrator(std::move(integrator));
    sceneProcessor->SetFilter(std::move(filter));
    sceneProcessor->SetAccelerator(std::move(accelerator));
    sceneProcessor->SetCamera(std::move(camera));
}

void SceneStateManager::LightSource(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    VERIFY_WORLD("LightSource");
    ParameterDictionary dict(std::move(params), graphicsState.lightAttributes,
                             graphicsState.colorSpace);
    sceneProcessor->AddLight(LightSceneEntity(name, std::move(dict), loc,
                                              RenderFromObject(),
                                              graphicsState.currentOutsideMedium));
}

void SceneStateManager::Shape(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    VERIFY_WORLD("Shape");

    ParameterDictionary dict(std::move(params), graphicsState.shapeAttributes,
                             graphicsState.colorSpace);

    int areaLightIndex = -1;
    if (!graphicsState.areaLightName.empty()) {
        areaLightIndex = sceneProcessor->AddAreaLight(
            SceneEntity(graphicsState.areaLightName, graphicsState.areaLightParams,
                        graphicsState.areaLightLoc));
        if (activeInstanceDefinition)
            Warning(&loc, "Area lights not supported with object instancing");
    }

    if (CTMIsAnimated()) {
        AnimatedTransform renderFromShape = RenderFromObject();
        const class Transform *identity = transformCache.Lookup(pbrt::Transform());

        AnimatedShapeSceneEntity entity(
            {name, std::move(dict), loc, renderFromShape, identity,
             graphicsState.reverseOrientation, graphicsState.currentMaterialIndex,
             graphicsState.currentMaterialName, areaLightIndex,
             graphicsState.currentInsideMedium, graphicsState.currentOutsideMedium});

        if (activeInstanceDefinition)
            activeInstanceDefinition->entity.animatedShapes.push_back(std::move(entity));
        else
            sceneProcessor->AddAnimatedShape(std::move(entity));
    } else {
        const class Transform *renderFromObject =
            transformCache.Lookup(RenderFromObject(0));
        const class Transform *objectFromRender =
            transformCache.Lookup(Inverse(*renderFromObject));

        ShapeSceneEntity entity(
            {name, std::move(dict), loc, renderFromObject, objectFromRender,
             graphicsState.reverseOrientation, graphicsState.currentMaterialIndex,
             graphicsState.currentMaterialName, areaLightIndex,
             graphicsState.currentInsideMedium, graphicsState.currentOutsideMedium});
        if (activeInstanceDefinition)
            activeInstanceDefinition->entity.shapes.push_back(std::move(entity));
        else
            shapes.push_back(std::move(entity));
    }
}

void SceneStateManager::ObjectBegin(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("ObjectBegin");
    pushedGraphicsStates.push_back(graphicsState);

    pushStack.push_back(std::make_pair('o', loc));

    if (activeInstanceDefinition) {
        ErrorExitDeferred(&loc, "ObjectBegin called inside of instance definition");
        return;
    }

    if (instanceNames.find(name) != instanceNames.end()) {
        ErrorExitDeferred(&loc, "%s: trying to redefine an object instance", name);
        return;
    }
    instanceNames.insert(name);

    activeInstanceDefinition = new ActiveInstanceDefinition(name, loc);
}

void SceneStateManager::ObjectEnd(FileLoc loc) {
    VERIFY_WORLD("ObjectEnd");
    if (!activeInstanceDefinition) {
        ErrorExitDeferred(&loc, "ObjectEnd called outside of instance definition");
        return;
    }
    if (activeInstanceDefinition->parent) {
        ErrorExitDeferred(&loc, "ObjectEnd called inside Import for instance definition");
        return;
    }

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

    // Otherwise it will be taken care of in MergeImported()
    if (--activeInstanceDefinition->activeImports == 0) {
        sceneProcessor->AddInstanceDefinition(
            std::move(activeInstanceDefinition->entity));
        delete activeInstanceDefinition;
    }

    activeInstanceDefinition = nullptr;
}

void SceneStateManager::ObjectInstance(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("ObjectInstance");

    if (activeInstanceDefinition) {
        ErrorExitDeferred(&loc,
                          "ObjectInstance can't be called inside instance definition");
        return;
    }

    class Transform worldFromRender = Inverse(renderFromWorld);

    if (CTMIsAnimated()) {
        AnimatedTransform animatedRenderFromInstance(
            RenderFromObject(0) * worldFromRender, graphicsState.transformStartTime,
            RenderFromObject(1) * worldFromRender, graphicsState.transformEndTime);

        instanceUses.push_back(
            InstanceSceneEntity(name, loc, animatedRenderFromInstance));
    } else {
        const class Transform *renderFromInstance =
            transformCache.Lookup(RenderFromObject(0) * worldFromRender);

        instanceUses.push_back(InstanceSceneEntity(name, loc, renderFromInstance));
    }
}

void SceneStateManager::EndOfFiles() {
    if (currentBlock != BlockState::WorldBlock)
        ErrorExitDeferred("End of files before \"WorldBegin\".");

    // Ensure there are no pushed graphics states
    while (!pushedGraphicsStates.empty()) {
        ErrorExitDeferred("Missing end to AttributeBegin");
        pushedGraphicsStates.pop_back();
    }

    if (errorExit)
        ErrorExit("Fatal errors during scene construction");

    if (!shapes.empty())
        sceneProcessor->AddShapes(shapes);
    if (!instanceUses.empty())
        sceneProcessor->AddInstanceUses(instanceUses);

    sceneProcessor->Done();
}

SceneStateManager *SceneStateManager::CopyForImport() {
    SceneStateManager *importScene = new SceneStateManager(sceneProcessor);
    importScene->renderFromWorld = renderFromWorld;
    importScene->graphicsState = graphicsState;
    importScene->currentBlock = currentBlock;
    if (activeInstanceDefinition) {
        importScene->activeInstanceDefinition = new ActiveInstanceDefinition(
            activeInstanceDefinition->entity.name, activeInstanceDefinition->entity.loc);

        // In case of nested imports, go up to the true root parent since
        // that's where we need to merge our shapes and that's where the
        // refcount is.
        ActiveInstanceDefinition *parent = activeInstanceDefinition;
        while (parent->parent)
            parent = parent->parent;
        importScene->activeInstanceDefinition->parent = parent;
        ++parent->activeImports;
    }
    return importScene;
}

void SceneStateManager::MergeImported(SceneStateManager *imported) {
    while (!imported->pushedGraphicsStates.empty()) {
        ErrorExitDeferred("Missing end to AttributeBegin");
        imported->pushedGraphicsStates.pop_back();
    }

    errorExit |= imported->errorExit;

    if (!imported->shapes.empty())
        sceneProcessor->AddShapes(imported->shapes);
    if (!imported->instanceUses.empty())
        sceneProcessor->AddInstanceUses(imported->instanceUses);

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
    if (imported->activeInstanceDefinition) {
        ActiveInstanceDefinition *current = imported->activeInstanceDefinition;
        ActiveInstanceDefinition *parent = current->parent;
        CHECK(parent != nullptr);

        std::lock_guard<std::mutex> lock(parent->mutex);
        mergeVector(parent->entity.shapes, current->entity.shapes);
        mergeVector(parent->entity.animatedShapes, current->entity.animatedShapes);

        delete current;

        if (--parent->activeImports == 0)
            sceneProcessor->AddInstanceDefinition(std::move(parent->entity));

        parent->mutex.unlock();
    }

    auto mergeSet = [this](auto &base, auto &imported, const char *name) {
        for (const auto &item : imported) {
            if (base.find(item) != base.end())
                ErrorExitDeferred("%s: multiply-defined %s.", item, name);
            base.insert(std::move(item));
        }
        imported.clear();
    };
    mergeSet(namedMaterialNames, imported->namedMaterialNames, "named material");
    mergeSet(floatTextureNames, imported->floatTextureNames, "texture");
    mergeSet(spectrumTextureNames, imported->spectrumTextureNames, "texture");
}

void SceneStateManager::Option(const std::string &name, const std::string &value,
                               FileLoc loc) {
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

void SceneStateManager::Transform(Float tr[16], FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = Transpose(
                              pbrt::Transform(SquareMatrix<4>(pstd::MakeSpan(tr, 16))));)
}

void SceneStateManager::ConcatTransform(Float tr[16], FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(
        graphicsState.ctm[i] =
            graphicsState.ctm[i] *
            Transpose(pbrt::Transform(SquareMatrix<4>(pstd::MakeSpan(tr, 16))));)
}

void SceneStateManager::Rotate(Float angle, Float dx, Float dy, Float dz, FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] =
                              graphicsState.ctm[i] *
                              pbrt::Rotate(angle, Vector3f(dx, dy, dz));)
}

void SceneStateManager::Scale(Float sx, Float sy, Float sz, FileLoc loc) {
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] =
                              graphicsState.ctm[i] * pbrt::Scale(sx, sy, sz);)
}

void SceneStateManager::LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                               Float ux, Float uy, Float uz, FileLoc loc) {
    class Transform lookAt =
        pbrt::LookAt(Point3f(ex, ey, ez), Point3f(lx, ly, lz), Vector3f(ux, uy, uz));
    FOR_ACTIVE_TRANSFORMS(graphicsState.ctm[i] = graphicsState.ctm[i] * lookAt;);
}

void SceneStateManager::ActiveTransformAll(FileLoc loc) {
    graphicsState.activeTransformBits = AllTransformsBits;
}

void SceneStateManager::ActiveTransformEndTime(FileLoc loc) {
    graphicsState.activeTransformBits = EndTransformBits;
}

void SceneStateManager::ActiveTransformStartTime(FileLoc loc) {
    graphicsState.activeTransformBits = StartTransformBits;
}

void SceneStateManager::TransformTimes(Float start, Float end, FileLoc loc) {
    VERIFY_OPTIONS("TransformTimes");
    graphicsState.transformStartTime = start;
    graphicsState.transformEndTime = end;
}

void SceneStateManager::PixelFilter(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("PixelFilter");
    filter = SceneEntity(name, std::move(dict), loc);
}

void SceneStateManager::Film(const std::string &type, ParsedParameterVector params,
                             FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("Film");
    film = SceneEntity(type, std::move(dict), loc);
}

void SceneStateManager::Sampler(const std::string &name, ParsedParameterVector params,
                                FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);

    VERIFY_OPTIONS("Sampler");
    sampler = SceneEntity(name, std::move(dict), loc);
}

void SceneStateManager::Accelerator(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("Accelerator");
    accelerator = SceneEntity(name, std::move(dict), loc);
}

void SceneStateManager::Integrator(const std::string &name, ParsedParameterVector params,
                                   FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);

    VERIFY_OPTIONS("Integrator");
    integrator = SceneEntity(name, std::move(dict), loc);
}

void SceneStateManager::MakeNamedMedium(const std::string &name,
                                        ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.mediumAttributes,
                             graphicsState.colorSpace);

    if (mediumNames.find(name) != mediumNames.end()) {
        ErrorExitDeferred(&loc, "Named medium \"%s\" redefined.", name);
        return;
    }

    mediumNames.insert(name);

    sceneProcessor->AddMedium(
        TransformedSceneEntity(name, std::move(dict), loc, RenderFromObject()));
}

void SceneStateManager::MediumInterface(const std::string &insideName,
                                        const std::string &outsideName, FileLoc loc) {
    graphicsState.currentInsideMedium = insideName;
    graphicsState.currentOutsideMedium = outsideName;
}

void SceneStateManager::Texture(const std::string &name, const std::string &type,
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

    if (type == "float")
        sceneProcessor->AddFloatTexture(
            name, TextureSceneEntity(texname, std::move(dict), loc, RenderFromObject()));
    else
        sceneProcessor->AddSpectrumTexture(
            name, TextureSceneEntity(texname, std::move(dict), loc, RenderFromObject()));
}

void SceneStateManager::Material(const std::string &name, ParsedParameterVector params,
                                 FileLoc loc) {
    VERIFY_WORLD("Material");

    ParameterDictionary dict(std::move(params), graphicsState.materialAttributes,
                             graphicsState.colorSpace);

    graphicsState.currentMaterialIndex =
        sceneProcessor->AddMaterial(SceneEntity(name, std::move(dict), loc));
    graphicsState.currentMaterialName.clear();
}

void SceneStateManager::MakeNamedMaterial(const std::string &name,
                                          ParsedParameterVector params, FileLoc loc) {
    VERIFY_WORLD("MakeNamedMaterial");

    ParameterDictionary dict(std::move(params), graphicsState.materialAttributes,
                             graphicsState.colorSpace);

    if (namedMaterialNames.find(name) != namedMaterialNames.end()) {
        ErrorExitDeferred(&loc, "%s: named material redefined.", name);
        return;
    }
    namedMaterialNames.insert(name);

    sceneProcessor->AddNamedMaterial(name, SceneEntity("", std::move(dict), loc));
}

void SceneStateManager::NamedMaterial(const std::string &name, FileLoc loc) {
    VERIFY_WORLD("NamedMaterial");
    graphicsState.currentMaterialName = name;
    graphicsState.currentMaterialIndex = -1;
}

void SceneStateManager::AreaLightSource(const std::string &name,
                                        ParsedParameterVector params, FileLoc loc) {
    VERIFY_WORLD("AreaLightSource");
    graphicsState.areaLightName = name;
    graphicsState.areaLightParams = ParameterDictionary(
        std::move(params), graphicsState.lightAttributes, graphicsState.colorSpace);
    graphicsState.areaLightLoc = loc;
}

// ParsedScene Method Definitions
ParsedScene::ParsedScene()
    : threadAllocators([]() {
          pstd::pmr::memory_resource *baseResource = pstd::pmr::get_default_resource();
#ifdef PBRT_BUILD_GPU_RENDERER
          if (Options->useGPU)
              baseResource = &CUDATrackedMemoryResource::singleton;
#endif
          pstd::pmr::monotonic_buffer_resource *resource =
              new pstd::pmr::monotonic_buffer_resource(1024 * 1024, baseResource);
          return Allocator(resource);
      }) {
}

void ParsedScene::SetFilm(SceneEntity film) {
    this->film = std::move(film);
}

void ParsedScene::SetSampler(SceneEntity sampler) {
    this->sampler = std::move(sampler);
}

void ParsedScene::SetIntegrator(SceneEntity integrator) {
    this->integrator = std::move(integrator);
}

void ParsedScene::SetFilter(SceneEntity filter) {
    this->filter = std::move(filter);
}

void ParsedScene::SetAccelerator(SceneEntity accelerator) {
    this->accelerator = std::move(accelerator);
}

void ParsedScene::SetCamera(CameraSceneEntity camera) {
    this->camera = std::move(camera);
}

void ParsedScene::AddNamedMaterial(std::string name, SceneEntity material) {
    std::lock_guard<std::mutex> lock(materialMutex);
    startLoadingNormalMaps(material.parameters);
    namedMaterials.push_back(std::make_pair(std::move(name), std::move(material)));
}

int ParsedScene::AddMaterial(SceneEntity material) {
    std::lock_guard<std::mutex> lock(materialMutex);
    materials.push_back(std::move(material));
    startLoadingNormalMaps(material.parameters);
    return materials.size() - 1;
}

void ParsedScene::startLoadingNormalMaps(const ParameterDictionary &parameters) {
    std::string filename = parameters.GetOneString("normalmap", "");
    if (filename.empty())
        return;

    // Overload materialMutex, which we already hold, for the futures...
    if (normalMapFutures.find(filename) != normalMapFutures.end())
        // It's already in flight.
        return;

    auto create = [=](std::string filename) {
        Allocator alloc = threadAllocators.Get();
        ImageAndMetadata immeta =
            Image::Read(filename, Allocator(), ColorEncoding::Linear);
        Image &image = immeta.image;
        ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
        if (!rgbDesc)
            ErrorExit("%s: normal map image must contain R, G, and B channels", filename);
        Image *normalMap = alloc.new_object<Image>(alloc);
        *normalMap = image.SelectChannels(rgbDesc);

        return normalMap;
    };
    normalMapFutures[filename] = RunAsync(create, filename);
}

void ParsedScene::AddMedium(TransformedSceneEntity medium) {
    std::lock_guard<std::mutex> lock(mediaMutex);

    auto create = [=]() {
        std::string type = medium.parameters.GetOneString("type", "");
        if (type.empty())
            ErrorExit(&medium.loc, "No parameter string \"type\" found for medium.");

        if (medium.renderFromObject.IsAnimated())
            Warning(&medium.loc, "Animated transformation provided for medium. Only the "
                                 "start transform will be used.");
        return Medium::Create(type, medium.parameters,
                              medium.renderFromObject.startTransform, &medium.loc,
                              threadAllocators.Get());
    };

    mediaFutures[medium.name] = RunAsync(create);
}

void ParsedScene::AddFloatTexture(std::string name, TextureSceneEntity texture) {
    if (texture.renderFromObject.IsAnimated())
        Warning(&texture.loc, "Animated world to texture transforms are not supported. "
                              "Using start transform.");

    std::lock_guard<std::mutex> lock(textureMutex);
    if (texture.texName != "imagemap" && texture.texName != "ptex") {
        serialFloatTextures.push_back(
            std::make_pair(std::move(name), std::move(texture)));
        return;
    }

    std::string filename =
        ResolveFilename(texture.parameters.GetOneString("filename", ""));
    if (filename.empty()) {
        Error(&texture.loc, "\"string filename\" not provided for image texture.");
        ++nMissingTextures;
        return;
    }
    if (!FileExists(filename)) {
        Error(&texture.loc, "%s: file not found.", filename);
        ++nMissingTextures;
        return;
    }

    if (loadingTextureFilenames.find(filename) != loadingTextureFilenames.end()) {
        serialFloatTextures.push_back(
            std::make_pair(std::move(name), std::move(texture)));
        return;
    }
    loadingTextureFilenames.insert(filename);

    auto create = [=](TextureSceneEntity texture) {
        Allocator alloc = threadAllocators.Get();

        pbrt::Transform renderFromTexture = texture.renderFromObject.startTransform;
        // Pass nullptr for the textures, since they shouldn't be accessed
        // anyway.
        TextureParameterDictionary texDict(&texture.parameters, nullptr);
        return FloatTexture::Create(texture.texName, renderFromTexture, texDict,
                                    &texture.loc, alloc, Options->useGPU);
    };
    floatTextureFutures[name] = RunAsync(create, texture);
}

void ParsedScene::AddSpectrumTexture(std::string name, TextureSceneEntity texture) {
    std::lock_guard<std::mutex> lock(textureMutex);

    if (texture.texName != "imagemap" && texture.texName != "ptex") {
        serialSpectrumTextures.push_back(
            std::make_pair(std::move(name), std::move(texture)));
        return;
    }

    std::string filename =
        ResolveFilename(texture.parameters.GetOneString("filename", ""));
    if (filename.empty()) {
        Error(&texture.loc, "\"string filename\" not provided for image texture.");
        ++nMissingTextures;
        return;
    }
    if (!FileExists(filename)) {
        Error(&texture.loc, "%s: file not found.", filename);
        ++nMissingTextures;
        return;
    }

    if (loadingTextureFilenames.find(filename) != loadingTextureFilenames.end()) {
        serialSpectrumTextures.push_back(
            std::make_pair(std::move(name), std::move(texture)));
        return;
    }
    loadingTextureFilenames.insert(filename);

    asyncSpectrumTextures.push_back(std::make_pair(name, texture));

    auto create = [=](TextureSceneEntity texture) {
        Allocator alloc = threadAllocators.Get();

        pbrt::Transform renderFromTexture = texture.renderFromObject.startTransform;
        // nullptr for the textures, as with float textures.
        TextureParameterDictionary texDict(&texture.parameters, nullptr);
        // Only create SpectrumType::Albedo for now; will get the other two
        // types in CreateTextures().
        return SpectrumTexture::Create(texture.texName, renderFromTexture, texDict,
                                       SpectrumType::Albedo, &texture.loc, alloc,
                                       Options->useGPU);
    };
    spectrumTextureFutures[name] = RunAsync(create, texture);
}

void ParsedScene::AddLight(LightSceneEntity light) {
    std::lock_guard<std::mutex> lock(lightMutex);
    if (!light.medium.empty()) {
        // If the light has a medium associated with it, punt for now since
        // the Medium may not yet be initialized; these lights will be
        // taken care of when CreateLights() is called.  At the cost of
        // some complexity, we could check and see if it's already in the
        // medium map and wait for its in-flight future if it's not yet
        // ready, though the most important case here is probably infinite
        // image lights and those can't have media associated with them
        // anyway...
        lightsWithMedia.push_back(std::move(light));
        return;
    }

    if (light.renderFromObject.IsAnimated())
        Warning(&light.loc,
                "Animated lights aren't supported. Using the start transform.");

    auto create = [this](LightSceneEntity light) {
        return Light::Create(light.name, light.parameters,
                             light.renderFromObject.startTransform,
                             camera.cameraTransform, nullptr /* Medium */, &light.loc,
                             threadAllocators.Get());
    };
    lightFutures.push_back(RunAsync(create, light));
}

int ParsedScene::AddAreaLight(SceneEntity light) {
    std::lock_guard<std::mutex> lock(areaLightMutex);
    areaLights.push_back(std::move(light));
    return areaLights.size() - 1;
}

void ParsedScene::AddShapes(pstd::span<ShapeSceneEntity> s) {
    std::lock_guard<std::mutex> lock(shapeMutex);
    std::move(std::begin(s), std::end(s), std::back_inserter(shapes));
}

void ParsedScene::AddAnimatedShape(AnimatedShapeSceneEntity shape) {
    std::lock_guard<std::mutex> lock(animatedShapeMutex);
    animatedShapes.push_back(std::move(shape));
}

void ParsedScene::AddInstanceDefinition(InstanceDefinitionSceneEntity instance) {
    InstanceDefinitionSceneEntity *def =
        new InstanceDefinitionSceneEntity(std::move(instance));

    std::lock_guard<std::mutex> lock(instanceDefinitionMutex);
    instanceDefinitions[def->name] = def;
}

void ParsedScene::AddInstanceUses(pstd::span<InstanceSceneEntity> in) {
    std::lock_guard<std::mutex> lock(instanceUseMutex);
    std::move(std::begin(in), std::end(in), std::back_inserter(instances));
}

void ParsedScene::Done() {
#if 0
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
        for (const auto &s : id.second->shapes)
            checkVec(s.parameters.GetParameterVector());
        for (const auto &as : id.second->animatedShapes)
            checkVec(as.parameters.GetParameterVector());
    }

    LOG_VERBOSE("Scene stats: %d shapes, %d animated shapes, %d instance definitions, "
                "%d instance uses, %d float textures, %d spectrum textures, "
                "%d named materials, %d materials",
                shapes.size(), animatedShapes.size(), instanceDefinitions.size(),
                instances.size(), floatTextures.size(),
                spectrumTextures.size(), namedMaterials.size(), materials.size());

    // And complain about what's left.
    for (const std::string &s : unusedFloatTextures)
        LOG_VERBOSE("%s: float texture unused in scene", s);
    for (const std::string &s : unusedSpectrumTextures)
        LOG_VERBOSE("%s: spectrum texture unused in scene", s);
#endif
}

void ParsedScene::CreateMaterials(
    const NamedTextures &textures, ThreadLocal<Allocator> &threadAllocators,
    std::map<std::string, pbrt::Material> *namedMaterialsOut,
    std::vector<pbrt::Material> *materialsOut) {
    LOG_VERBOSE("Starting to consume normal map futures");
    for (auto &fut : normalMapFutures) {
        CHECK(normalMaps.find(fut.first) == normalMaps.end());
        normalMaps[fut.first] = fut.second.Get();
    }
    normalMapFutures.clear();
    LOG_VERBOSE("Finished consuming normal map futures");

    // Named materials
    for (const auto &nm : namedMaterials) {
        const std::string &name = nm.first;
        const SceneEntity &mtl = nm.second;
        Allocator alloc = threadAllocators.Get();

        if (namedMaterialsOut->find(name) != namedMaterialsOut->end()) {
            ErrorExit(&mtl.loc, "%s: trying to redefine named material.", name);
            continue;
        }

        std::string type = mtl.parameters.GetOneString("type", "");
        if (type.empty()) {
            ErrorExit(&mtl.loc,
                      "%s: \"string type\" not provided in named material's parameters.",
                      name);
            continue;
        }

        std::string fn = nm.second.parameters.GetOneString("normalmap", "");
        Image *normalMap = !fn.empty() ? normalMaps[fn] : nullptr;

        TextureParameterDictionary texDict(&mtl.parameters, &textures);
        class Material m = Material::Create(type, texDict, normalMap, *namedMaterialsOut,
                                            &mtl.loc, alloc);
        (*namedMaterialsOut)[name] = m;
    }

    // Regular materials
    materialsOut->reserve(materials.size());
    for (const auto &mtl : materials) {
        Allocator alloc = threadAllocators.Get();
        std::string fn = mtl.parameters.GetOneString("normalmap", "");
        Image *normalMap = !fn.empty() ? normalMaps[fn] : nullptr;

        TextureParameterDictionary texDict(&mtl.parameters, &textures);
        class Material m = Material::Create(mtl.name, texDict, normalMap,
                                            *namedMaterialsOut, &mtl.loc, alloc);
        materialsOut->push_back(m);
    }
}

NamedTextures ParsedScene::CreateTextures() {
    NamedTextures textures;

    if (nMissingTextures > 0)
        ErrorExit("%d missing textures", nMissingTextures);

    // Consume futures
    LOG_VERBOSE("Starting to consume texture futures");
    for (auto &tex : floatTextureFutures)
        textures.floatTextures[tex.first] = tex.second.Get();
    for (auto &tex : spectrumTextureFutures)
        textures.albedoSpectrumTextures[tex.first] = tex.second.Get();
    LOG_VERBOSE("Finished consuming texture futures");

    LOG_VERBOSE("Starting to create remaining textures");
    Allocator alloc = threadAllocators.Get();
    // Create the other SpectrumTypes for the spectrum textures.
    for (const auto &tex : asyncSpectrumTextures) {
        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        // These are all image textures, so nullptr is fine for the
        // textures, as earlier.
        TextureParameterDictionary texDict(&tex.second.parameters, nullptr);

        // These should be fast since they should hit the texture cache
        SpectrumTexture unboundedTex = SpectrumTexture::Create(
            tex.second.texName, renderFromTexture, texDict, SpectrumType::Unbounded,
            &tex.second.loc, alloc, Options->useGPU);
        SpectrumTexture illumTex = SpectrumTexture::Create(
            tex.second.texName, renderFromTexture, texDict, SpectrumType::Illuminant,
            &tex.second.loc, alloc, Options->useGPU);

        textures.unboundedSpectrumTextures[tex.first] = unboundedTex;
        textures.illuminantSpectrumTextures[tex.first] = illumTex;
    }

    // And do the rest serially
    for (auto &tex : serialFloatTextures) {
        Allocator alloc = threadAllocators.Get();

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        TextureParameterDictionary texDict(&tex.second.parameters, &textures);
        FloatTexture t =
            FloatTexture::Create(tex.second.texName, renderFromTexture, texDict,
                                 &tex.second.loc, alloc, Options->useGPU);
        textures.floatTextures[tex.first] = t;
    }

    for (auto &tex : serialSpectrumTextures) {
        Allocator alloc = threadAllocators.Get();

        if (tex.second.renderFromObject.IsAnimated())
            Warning(&tex.second.loc, "Animated world to texture transform not supported. "
                                     "Using start transform.");

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        TextureParameterDictionary texDict(&tex.second.parameters, &textures);
        SpectrumTexture albedoTex = SpectrumTexture::Create(
            tex.second.texName, renderFromTexture, texDict, SpectrumType::Albedo,
            &tex.second.loc, alloc, Options->useGPU);
        SpectrumTexture unboundedTex = SpectrumTexture::Create(
            tex.second.texName, renderFromTexture, texDict, SpectrumType::Unbounded,
            &tex.second.loc, alloc, Options->useGPU);
        SpectrumTexture illumTex = SpectrumTexture::Create(
            tex.second.texName, renderFromTexture, texDict, SpectrumType::Illuminant,
            &tex.second.loc, alloc, Options->useGPU);

        textures.albedoSpectrumTextures[tex.first] = albedoTex;
        textures.unboundedSpectrumTextures[tex.first] = unboundedTex;
        textures.illuminantSpectrumTextures[tex.first] = illumTex;
    }

    LOG_VERBOSE("Done creating textures");
    return textures;
}

std::map<std::string, Medium> ParsedScene::CreateMedia() {
    std::lock_guard<std::mutex> lock(mediaMutex);

    LOG_VERBOSE("Consume media futures start");
    for (auto &m : mediaFutures) {
        CHECK(mediaMap.find(m.first) == mediaMap.end());
        mediaMap[m.first] = m.second.Get();
    }

    mediaFutures.clear();
    LOG_VERBOSE("Consume media futures finished");

    return mediaMap;
}

std::vector<Light> ParsedScene::CreateLights(
    const NamedTextures &textures,
    std::map<int, pstd::vector<Light> *> *shapeIndexToAreaLights) {
    // Ensure that media are all ready
    (void)CreateMedia();

    auto findMedium = [this](const std::string &s, const FileLoc *loc) -> Medium {
        if (s.empty())
            return nullptr;

        auto iter = mediaMap.find(s);
        if (iter == mediaMap.end())
            ErrorExit(loc, "%s: medium not defined", s);
        return iter->second;
    };

    Allocator alloc = threadAllocators.Get();

    auto getAlphaTexture = [&](const ParameterDictionary &parameters,
                               const FileLoc *loc) -> FloatTexture {
        std::string alphaTexName = parameters.GetTexture("alpha");
        if (!alphaTexName.empty()) {
            if (auto iter = textures.floatTextures.find(alphaTexName);
                iter != textures.floatTextures.end()) {
                if (Options->useGPU &&
                    !BasicTextureEvaluator().CanEvaluate({iter->second}, {}))
                    // A warning will be issued elsewhere...
                    return nullptr;
                return iter->second;
            } else
                ErrorExit(loc, "%s: couldn't find float texture for \"alpha\" parameter.",
                          alphaTexName);
        } else if (Float alpha = parameters.GetOneFloat("alpha", 1.f); alpha < 1.f)
            return alloc.new_object<FloatConstantTexture>(alpha);
        else
            return nullptr;
    };

    LOG_VERBOSE("Starting non-future lights");
    std::vector<Light> lights;
    // Lights with media (punted in AddLight() earlier.)
    for (const auto &light : lightsWithMedia) {
        Medium outsideMedium = findMedium(light.medium, &light.loc);
        if (light.renderFromObject.IsAnimated())
            Warning(&light.loc,
                    "Animated lights aren't supported. Using the start transform.");
        Light l = Light::Create(light.name, light.parameters,
                                light.renderFromObject.startTransform,
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
            auto iter =
                std::find_if(namedMaterials.begin(), namedMaterials.end(),
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

        pstd::vector<pbrt::Shape> shapeObjects = Shape::Create(
            sh.name, sh.renderFromObject, sh.objectFromRender, sh.reverseOrientation,
            sh.parameters, textures.floatTextures, &sh.loc, alloc);

        FloatTexture alphaTex = getAlphaTexture(sh.parameters, &sh.loc);

        pbrt::MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                                 findMedium(sh.outsideMedium, &sh.loc));

        pstd::vector<Light> *shapeLights = new pstd::vector<Light>(alloc);
        const auto &areaLightEntity = areaLights[sh.lightIndex];
        for (pbrt::Shape ps : shapeObjects) {
            Light area = Light::CreateArea(
                areaLightEntity.name, areaLightEntity.parameters, *sh.renderFromObject,
                mi, ps, alphaTex, &areaLightEntity.loc, alloc);
            if (area) {
                lights.push_back(area);
                shapeLights->push_back(area);
            }
        }

        (*shapeIndexToAreaLights)[i] = shapeLights;
    }

    LOG_VERBOSE("Finished non-future lights");

    LOG_VERBOSE("Starting to consume non-area light futures");
    for (auto &fut : lightFutures)
        lights.push_back(fut.Get());
    LOG_VERBOSE("Finished consuming non-area light futures");

    return lights;
}

Primitive ParsedScene::CreateAggregate(
    const NamedTextures &textures,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    const std::map<std::string, Medium> &media,
    const std::map<std::string, pbrt::Material> &namedMaterials,
    const std::vector<pbrt::Material> &materials) {
    Allocator alloc;
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

    // Non-animated shapes
    auto CreatePrimitivesForShapes =
        [&](std::vector<ShapeSceneEntity> &shapes) -> std::vector<Primitive> {
        // Parallelize Shape::Create calls, which will in turn
        // parallelize PLY file loading, etc...
        pstd::vector<pstd::vector<pbrt::Shape>> shapeVectors(shapes.size());
        ParallelFor(0, shapes.size(), [&](int64_t i) {
            const auto &sh = shapes[i];
            shapeVectors[i] = Shape::Create(
                sh.name, sh.renderFromObject, sh.objectFromRender, sh.reverseOrientation,
                sh.parameters, textures.floatTextures, &sh.loc, alloc);
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

                if (!area && !mi.IsMediumTransition() && !alphaTex)
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
                              sh.parameters, textures.floatTextures, &sh.loc, alloc);
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
                    prims.push_back(new GeometricPrimitive(
                        s, mtl, nullptr /* area light */, mi, alphaTex));
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
    std::vector<std::map<std::string, InstanceDefinitionSceneEntity *>::iterator>
        instanceDefinitionIterators;
    for (auto iter = this->instanceDefinitions.begin();
         iter != this->instanceDefinitions.end(); ++iter)
        instanceDefinitionIterators.push_back(iter);
    ParallelFor(0, instanceDefinitionIterators.size(), [&](int64_t i) {
        auto &inst = *instanceDefinitionIterators[i];

        std::vector<Primitive> instancePrimitives =
            CreatePrimitivesForShapes(inst.second->shapes);
        std::vector<Primitive> movingInstancePrimitives =
            CreatePrimitivesForAnimatedShapes(inst.second->animatedShapes);
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

        delete inst.second;
        inst.second = nullptr;
    });

    this->instanceDefinitions.clear();

    // Instances
    for (const auto &inst : instances) {
        auto iter = instanceDefinitions.find(*inst.name);
        if (iter == instanceDefinitions.end())
            ErrorExit(&inst.loc, "%s: object instance not defined", inst.name);

        if (!iter->second)
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
    Primitive aggregate = nullptr;
    LOG_VERBOSE("Starting top-level accelerator");
    if (!primitives.empty())
        aggregate = CreateAccelerator(accelerator.name, std::move(primitives),
                                      accelerator.parameters);
    LOG_VERBOSE("Finished top-level accelerator");
    return aggregate;
}

}  // namespace pbrt
