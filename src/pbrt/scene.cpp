// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/scene.h>

#include <pbrt/cpu/aggregates.h>
#include <pbrt/cpu/integrators.h>
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
#include <pbrt/util/string.h>
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

std::string BasicSceneBuilder::ToString() const {
    return StringPrintf(
        "[ BasicSceneBuilder camera: %s film: %s sampler: %s integrator: %s "
        "filter: %s accelerator: %s ]",
        camera, film, sampler, integrator, filter, accelerator);
}

BasicSceneBuilder::GraphicsState::GraphicsState() {
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

STAT_COUNTER("Scene/Object instances created", nObjectInstancesCreated);
STAT_COUNTER("Scene/Object instances used", nObjectInstancesUsed);

// BasicSceneBuilder Method Definitions
BasicSceneBuilder::BasicSceneBuilder(BasicScene *scene)
    : scene(scene)
#ifdef PBRT_BUILD_GPU_RENDERER
      ,
      transformCache(Options->useGPU ? Allocator(&CUDATrackedMemoryResource::singleton)
                                     : Allocator())
#endif
{
    // Set scene defaults
    camera.name = SceneEntity::internedStrings.Lookup("perspective");
    sampler.name = SceneEntity::internedStrings.Lookup("zsobol");
    filter.name = SceneEntity::internedStrings.Lookup("gaussian");
    integrator.name = SceneEntity::internedStrings.Lookup("volpath");
    accelerator.name = SceneEntity::internedStrings.Lookup("bvh");

    film.name = SceneEntity::internedStrings.Lookup("rgb");
    film.parameters = ParameterDictionary({}, RGBColorSpace::sRGB);

    ParameterDictionary dict({}, RGBColorSpace::sRGB);
    currentMaterialIndex = scene->AddMaterial(SceneEntity("diffuse", dict, {}));
}

void BasicSceneBuilder::ReverseOrientation(FileLoc loc) {
    VERIFY_WORLD("ReverseOrientation");
    graphicsState.reverseOrientation = !graphicsState.reverseOrientation;
}

void BasicSceneBuilder::ColorSpace(const std::string &name, FileLoc loc) {
    if (const RGBColorSpace *cs = RGBColorSpace::GetNamed(name))
        graphicsState.colorSpace = cs;
    else
        Error(&loc, "%s: color space unknown", name);
}

void BasicSceneBuilder::Identity(FileLoc loc) {
    graphicsState.ForActiveTransforms([](auto t) { return pbrt::Transform(); });
}

void BasicSceneBuilder::Translate(Float dx, Float dy, Float dz, FileLoc loc) {
    graphicsState.ForActiveTransforms(
        [=](auto t) { return t * pbrt::Translate(Vector3f(dx, dy, dz)); });
}

void BasicSceneBuilder::CoordinateSystem(const std::string &origName, FileLoc loc) {
    std::string name = NormalizeUTF8(origName);
    namedCoordinateSystems[name] = graphicsState.ctm;
}

void BasicSceneBuilder::CoordSysTransform(const std::string &origName, FileLoc loc) {
    std::string name = NormalizeUTF8(origName);
    if (namedCoordinateSystems.find(name) != namedCoordinateSystems.end())
        graphicsState.ctm = namedCoordinateSystems[name];
    else
        Warning(&loc, "Couldn't find named coordinate system \"%s\"", name);
}

void BasicSceneBuilder::Camera(const std::string &name, ParsedParameterVector params,
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

void BasicSceneBuilder::AttributeBegin(FileLoc loc) {
    VERIFY_WORLD("AttributeBegin");
    pushedGraphicsStates.push_back(graphicsState);
    pushStack.push_back(std::make_pair('a', loc));
}

void BasicSceneBuilder::AttributeEnd(FileLoc loc) {
    VERIFY_WORLD("AttributeEnd");
    // Issue error on unmatched _AttributeEnd_
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

void BasicSceneBuilder::Attribute(const std::string &target, ParsedParameterVector attrib,
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

void BasicSceneBuilder::Sampler(const std::string &name, ParsedParameterVector params,
                                FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("Sampler");
    sampler = SceneEntity(name, std::move(dict), loc);
}

void BasicSceneBuilder::WorldBegin(FileLoc loc) {
    VERIFY_OPTIONS("WorldBegin");
    // Reset graphics state for _WorldBegin_
    currentBlock = BlockState::WorldBlock;
    for (int i = 0; i < MaxTransforms; ++i)
        graphicsState.ctm[i] = pbrt::Transform();
    graphicsState.activeTransformBits = AllTransformsBits;
    namedCoordinateSystems["world"] = graphicsState.ctm;

    // Pass pre-_WorldBegin_ entities to _scene_
    scene->SetOptions(filter, film, camera, sampler, integrator, accelerator);
}

void BasicSceneBuilder::MakeNamedMedium(const std::string &origName,
                                        ParsedParameterVector params, FileLoc loc) {
    std::string name = NormalizeUTF8(origName);
    // Issue error if medium _name_ is multiply defined
    if (mediumNames.find(name) != mediumNames.end()) {
        ErrorExitDeferred(&loc, "Named medium \"%s\" redefined.", name);
        return;
    }
    mediumNames.insert(name);

    // Create _ParameterDictionary_ for medium and call _AddMedium()_
    ParameterDictionary dict(std::move(params), graphicsState.mediumAttributes,
                             graphicsState.colorSpace);
    scene->AddMedium(MediumSceneEntity(name, std::move(dict), loc, RenderFromObject()));
}

void BasicSceneBuilder::LightSource(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    VERIFY_WORLD("LightSource");
    ParameterDictionary dict(std::move(params), graphicsState.lightAttributes,
                             graphicsState.colorSpace);
    scene->AddLight(LightSceneEntity(name, std::move(dict), loc, RenderFromObject(),
                                     graphicsState.currentOutsideMedium));
}

void BasicSceneBuilder::Shape(const std::string &name, ParsedParameterVector params,
                              FileLoc loc) {
    VERIFY_WORLD("Shape");

    ParameterDictionary dict(std::move(params), graphicsState.shapeAttributes,
                             graphicsState.colorSpace);

    int areaLightIndex = -1;
    if (!graphicsState.areaLightName.empty()) {
        areaLightIndex = scene->AddAreaLight(SceneEntity(graphicsState.areaLightName,
                                                         graphicsState.areaLightParams,
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
            scene->AddAnimatedShape(std::move(entity));
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

void BasicSceneBuilder::ObjectBegin(const std::string &origName, FileLoc loc) {
    std::string name = NormalizeUTF8(origName);

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

void BasicSceneBuilder::ObjectEnd(FileLoc loc) {
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
        scene->AddInstanceDefinition(std::move(activeInstanceDefinition->entity));
        delete activeInstanceDefinition;
    }

    activeInstanceDefinition = nullptr;
}

void BasicSceneBuilder::ObjectInstance(const std::string &origName, FileLoc loc) {
    std::string name = NormalizeUTF8(origName);
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

        // For very small changes, animatedRenderFromInstance may have both
        // xforms equal even if CTMIsAnimated() has returned true. Fall
        // through to create a regular non-animated instance in that case.
        if (animatedRenderFromInstance.IsAnimated()) {
            instanceUses.push_back(
                InstanceSceneEntity(name, loc, animatedRenderFromInstance));
            return;
        }
    }

    const class Transform *renderFromInstance =
        transformCache.Lookup(RenderFromObject(0) * worldFromRender);
    instanceUses.push_back(InstanceSceneEntity(name, loc, renderFromInstance));
}

void BasicSceneBuilder::EndOfFiles() {
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
        scene->AddShapes(shapes);
    if (!instanceUses.empty())
        scene->AddInstanceUses(instanceUses);

    scene->Done();
}

BasicSceneBuilder *BasicSceneBuilder::CopyForImport() {
    BasicSceneBuilder *importBuilder = new BasicSceneBuilder(scene);
    importBuilder->renderFromWorld = renderFromWorld;
    importBuilder->graphicsState = graphicsState;
    importBuilder->currentBlock = currentBlock;
    if (activeInstanceDefinition) {
        importBuilder->activeInstanceDefinition = new ActiveInstanceDefinition(
            activeInstanceDefinition->entity.name, activeInstanceDefinition->entity.loc);

        // In case of nested imports, go up to the true root parent since
        // that's where we need to merge our shapes and that's where the
        // refcount is.
        ActiveInstanceDefinition *parent = activeInstanceDefinition;
        while (parent->parent)
            parent = parent->parent;
        importBuilder->activeInstanceDefinition->parent = parent;
        ++parent->activeImports;
    }
    return importBuilder;
}

void BasicSceneBuilder::MergeImported(BasicSceneBuilder *imported) {
    while (!imported->pushedGraphicsStates.empty()) {
        ErrorExitDeferred("Missing end to AttributeBegin");
        imported->pushedGraphicsStates.pop_back();
    }

    errorExit |= imported->errorExit;

    if (!imported->shapes.empty())
        scene->AddShapes(imported->shapes);
    if (!imported->instanceUses.empty())
        scene->AddInstanceUses(imported->instanceUses);

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
            scene->AddInstanceDefinition(std::move(parent->entity));

        parent->mutex.unlock();
    }

    auto mergeSet = [this](auto &base, auto &imported, const char *name) {
        for (const auto &item : imported) {
            if (base.find(item) != base.end())
                ErrorExitDeferred("%s: multiply defined %s.", item, name);
            base.insert(std::move(item));
        }
        imported.clear();
    };
    mergeSet(namedMaterialNames, imported->namedMaterialNames, "named material");
    mergeSet(floatTextureNames, imported->floatTextureNames, "texture");
    mergeSet(spectrumTextureNames, imported->spectrumTextureNames, "texture");
}

void BasicSceneBuilder::Option(const std::string &name, const std::string &value,
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
    } else if (nName == "disabletexturefiltering") {
        if (value == "true")
            Options->disableTextureFiltering = true;
        else if (value == "false")
            Options->disableTextureFiltering = false;
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
    } else if (nName == "displacementedgescale") {
        if (!Atof(value, &Options->displacementEdgeScale))
            ErrorExitDeferred(&loc, "%s: expected floating-point option value", value);
    } else if (nName == "msereferenceimage") {
        if (value.size() < 3 || value.front() != '"' || value.back() != '"')
            ErrorExitDeferred(&loc, "%s: expected quoted string for option value", value);
        Options->mseReferenceImage = value.substr(1, value.size() - 2);
    } else if (nName == "msereferenceout") {
        if (value.size() < 3 || value.front() != '"' || value.back() != '"')
            ErrorExitDeferred(&loc, "%s: expected quoted string for option value", value);
        Options->mseReferenceOutput = value.substr(1, value.size() - 2);
    } else if (nName == "rendercoordsys") {
        if (value.size() < 3 || value.front() != '"' || value.back() != '"')
            ErrorExitDeferred(&loc, "%s: expected quoted string for option value", value);
        std::string renderCoordSys = value.substr(1, value.size() - 2);
        if (renderCoordSys == "camera")
            Options->renderingSpace = RenderingCoordinateSystem::Camera;
        else if (renderCoordSys == "cameraworld")
            Options->renderingSpace = RenderingCoordinateSystem::CameraWorld;
        else if (renderCoordSys == "world")
            Options->renderingSpace = RenderingCoordinateSystem::World;
        else
            ErrorExit("%s: unknown rendering coordinate system.", renderCoordSys);
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
    } else if (nName == "wavefront") {
        if (value == "true")
            Options->wavefront = true;
        else if (value == "false")
            Options->wavefront = false;
        else
            ErrorExitDeferred(&loc, "%s: expected \"true\" or \"false\" for option value",
                              value);
    } else
        ErrorExitDeferred(&loc, "%s: unknown option", name);

#ifdef PBRT_BUILD_GPU_RENDERER
    CopyOptionsToGPU();
#endif  // PBRT_BUILD_GPU_RENDERER
}

void BasicSceneBuilder::Transform(Float tr[16], FileLoc loc) {
    graphicsState.ForActiveTransforms([=](auto t) {
        return Transpose(pbrt::Transform(SquareMatrix<4>(pstd::MakeSpan(tr, 16))));
    });
}

void BasicSceneBuilder::ConcatTransform(Float tr[16], FileLoc loc) {
    graphicsState.ForActiveTransforms([=](auto t) {
        return t * Transpose(pbrt::Transform(SquareMatrix<4>(pstd::MakeSpan(tr, 16))));
    });
}

void BasicSceneBuilder::Rotate(Float angle, Float dx, Float dy, Float dz, FileLoc loc) {
    graphicsState.ForActiveTransforms(
        [=](auto t) { return t * pbrt::Rotate(angle, Vector3f(dx, dy, dz)); });
}

void BasicSceneBuilder::Scale(Float sx, Float sy, Float sz, FileLoc loc) {
    graphicsState.ForActiveTransforms(
        [=](auto t) { return t * pbrt::Scale(sx, sy, sz); });
}

void BasicSceneBuilder::LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                               Float ux, Float uy, Float uz, FileLoc loc) {
    class Transform lookAt =
        pbrt::LookAt(Point3f(ex, ey, ez), Point3f(lx, ly, lz), Vector3f(ux, uy, uz));
    graphicsState.ForActiveTransforms([=](auto t) { return t * lookAt; });
}

void BasicSceneBuilder::ActiveTransformAll(FileLoc loc) {
    graphicsState.activeTransformBits = AllTransformsBits;
}

void BasicSceneBuilder::ActiveTransformEndTime(FileLoc loc) {
    graphicsState.activeTransformBits = EndTransformBits;
}

void BasicSceneBuilder::ActiveTransformStartTime(FileLoc loc) {
    graphicsState.activeTransformBits = StartTransformBits;
}

void BasicSceneBuilder::TransformTimes(Float start, Float end, FileLoc loc) {
    VERIFY_OPTIONS("TransformTimes");
    graphicsState.transformStartTime = start;
    graphicsState.transformEndTime = end;
}

void BasicSceneBuilder::PixelFilter(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("PixelFilter");
    filter = SceneEntity(name, std::move(dict), loc);
}

void BasicSceneBuilder::Film(const std::string &type, ParsedParameterVector params,
                             FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("Film");
    film = SceneEntity(type, std::move(dict), loc);
}

void BasicSceneBuilder::Accelerator(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);
    VERIFY_OPTIONS("Accelerator");
    accelerator = SceneEntity(name, std::move(dict), loc);
}

void BasicSceneBuilder::Integrator(const std::string &name, ParsedParameterVector params,
                                   FileLoc loc) {
    ParameterDictionary dict(std::move(params), graphicsState.colorSpace);

    VERIFY_OPTIONS("Integrator");
    integrator = SceneEntity(name, std::move(dict), loc);
}

void BasicSceneBuilder::MediumInterface(const std::string &origInsideName,
                                        const std::string &origOutsideName, FileLoc loc) {
    std::string insideName = NormalizeUTF8(origInsideName);
    std::string outsideName = NormalizeUTF8(origOutsideName);

    graphicsState.currentInsideMedium = insideName;
    graphicsState.currentOutsideMedium = outsideName;
}

void BasicSceneBuilder::Texture(const std::string &origName, const std::string &type,
                                const std::string &texname, ParsedParameterVector params,
                                FileLoc loc) {
    std::string name = NormalizeUTF8(origName);
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
        scene->AddFloatTexture(
            name, TextureSceneEntity(texname, std::move(dict), loc, RenderFromObject()));
    else
        scene->AddSpectrumTexture(
            name, TextureSceneEntity(texname, std::move(dict), loc, RenderFromObject()));
}

void BasicSceneBuilder::Material(const std::string &name, ParsedParameterVector params,
                                 FileLoc loc) {
    VERIFY_WORLD("Material");

    ParameterDictionary dict(std::move(params), graphicsState.materialAttributes,
                             graphicsState.colorSpace);

    graphicsState.currentMaterialIndex =
        scene->AddMaterial(SceneEntity(name, std::move(dict), loc));
    graphicsState.currentMaterialName.clear();
}

void BasicSceneBuilder::MakeNamedMaterial(const std::string &origName,
                                          ParsedParameterVector params, FileLoc loc) {
    std::string name = NormalizeUTF8(origName);
    VERIFY_WORLD("MakeNamedMaterial");

    ParameterDictionary dict(std::move(params), graphicsState.materialAttributes,
                             graphicsState.colorSpace);

    if (namedMaterialNames.find(name) != namedMaterialNames.end()) {
        ErrorExitDeferred(&loc, "%s: named material redefined.", name);
        return;
    }
    namedMaterialNames.insert(name);

    scene->AddNamedMaterial(name, SceneEntity("", std::move(dict), loc));
}

void BasicSceneBuilder::NamedMaterial(const std::string &origName, FileLoc loc) {
    std::string name = NormalizeUTF8(origName);
    VERIFY_WORLD("NamedMaterial");
    graphicsState.currentMaterialName = name;
    graphicsState.currentMaterialIndex = -1;
}

void BasicSceneBuilder::AreaLightSource(const std::string &name,
                                        ParsedParameterVector params, FileLoc loc) {
    VERIFY_WORLD("AreaLightSource");
    graphicsState.areaLightName = name;
    graphicsState.areaLightParams = ParameterDictionary(
        std::move(params), graphicsState.lightAttributes, graphicsState.colorSpace);
    graphicsState.areaLightLoc = loc;
}

// BasicScene Method Definitions
void BasicScene::SetOptions(SceneEntity filter, SceneEntity film,
                            CameraSceneEntity camera, SceneEntity sampler,
                            SceneEntity integ, SceneEntity accel) {
    // Store information for specified integrator and accelerator
    filmColorSpace = film.parameters.ColorSpace();
    integrator = integ;
    accelerator = accel;

    // Immediately create filter and film
    LOG_VERBOSE("Starting to create filter and film");
    Allocator alloc = threadAllocators.Get();
    Filter filt = Filter::Create(filter.name, filter.parameters, &filter.loc, alloc);

    // It's a little ugly to poke into the camera's parameters here, but we
    // have this circular dependency that Camera::Create() expects a
    // Film, yet now the film needs to know the exposure time from
    // the camera....
    Float exposureTime = camera.parameters.GetOneFloat("shutterclose", 1.f) -
                         camera.parameters.GetOneFloat("shutteropen", 0.f);
    if (exposureTime <= 0)
        ErrorExit(&camera.loc,
                  "The specified camera shutter times imply that the shutter "
                  "does not open.  A black image will result.");

    this->film = Film::Create(film.name, film.parameters, exposureTime,
                              camera.cameraTransform, filt, &film.loc, alloc);
    LOG_VERBOSE("Finished creating filter and film");

    // Enqueue asynchronous job to create sampler
    samplerJob = RunAsync([sampler, this]() {
        LOG_VERBOSE("Starting to create sampler");
        Allocator alloc = threadAllocators.Get();
        Point2i res = this->film.FullResolution();
        return Sampler::Create(sampler.name, sampler.parameters, res, &sampler.loc,
                               alloc);
    });

    // Enqueue asynchronous job to create camera
    cameraJob = RunAsync([camera, this]() {
        LOG_VERBOSE("Starting to create camera");
        Allocator alloc = threadAllocators.Get();
        Medium cameraMedium = GetMedium(camera.medium, &camera.loc);

        Camera c = Camera::Create(camera.name, camera.parameters, cameraMedium,
                                  camera.cameraTransform, this->film, &camera.loc, alloc);
        LOG_VERBOSE("Finished creating camera");
        return c;
    });
}

void BasicScene::AddMedium(MediumSceneEntity medium) {
    // Define _create_ lambda function for _Medium_ creation
    auto create = [medium, this]() {
        std::string type = medium.parameters.GetOneString("type", "");
        // Check for missing medium ``type'' or animated medium transform
        if (type.empty())
            ErrorExit(&medium.loc, "No parameter \"string type\" found for medium.");
        if (medium.renderFromObject.IsAnimated())
            Warning(&medium.loc, "Animated transformation provided for medium. Only the "
                                 "start transform will be used.");

        return Medium::Create(type, medium.parameters,
                              medium.renderFromObject.startTransform, &medium.loc,
                              threadAllocators.Get());
    };

    std::lock_guard<std::mutex> lock(mediaMutex);
    mediumJobs[medium.name] = RunAsync(create);
}

Medium BasicScene::GetMedium(const std::string &name, const FileLoc *loc) {
    if (name.empty())
        return nullptr;

    mediaMutex.lock();
    while (true) {
        if (auto iter = mediaMap.find(name); iter != mediaMap.end()) {
            Medium m = iter->second;
            mediaMutex.unlock();
            return m;
        } else {
            auto fiter = mediumJobs.find(name);
            if (fiter == mediumJobs.end())
                ErrorExit(loc, "%s: medium is not defined.", name);

            pstd::optional<Medium> m = fiter->second->TryGetResult(&mediaMutex);
            if (m) {
                mediaMap[name] = *m;
                mediumJobs.erase(fiter);
                mediaMutex.unlock();
                return *m;
            }
        }
    }
}

std::map<std::string, Medium> BasicScene::CreateMedia() {
    mediaMutex.lock();
    if (!mediumJobs.empty()) {
        // Consume results for asynchronously-created _Medium_ objects
        LOG_VERBOSE("Consume media futures start");
        for (auto &m : mediumJobs) {
            while (mediaMap.find(m.first) == mediaMap.end()) {
                pstd::optional<Medium> med = m.second->TryGetResult(&mediaMutex);
                if (med)
                    mediaMap[m.first] = *med;
            }
        }
        LOG_VERBOSE("Consume media futures finished");
        mediumJobs.clear();
    }
    mediaMutex.unlock();
    return mediaMap;
}

std::unique_ptr<Integrator> BasicScene::CreateIntegrator(
    Camera camera, Sampler sampler, Primitive accel, std::vector<Light> lights) const {
    return Integrator::Create(integrator.name, integrator.parameters, camera, sampler,
                              accel, lights, filmColorSpace, &integrator.loc);
}

BasicScene::BasicScene()
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

void BasicScene::AddNamedMaterial(std::string name, SceneEntity material) {
    std::lock_guard<std::mutex> lock(materialMutex);
    startLoadingNormalMaps(material.parameters);
    namedMaterials.push_back(std::make_pair(std::move(name), std::move(material)));
}

int BasicScene::AddMaterial(SceneEntity material) {
    std::lock_guard<std::mutex> lock(materialMutex);
    startLoadingNormalMaps(material.parameters);
    materials.push_back(std::move(material));
    return int(materials.size() - 1);
}

void BasicScene::startLoadingNormalMaps(const ParameterDictionary &parameters) {
    std::string filename = ResolveFilename(parameters.GetOneString("normalmap", ""));
    if (filename.empty())
        return;

    // Overload materialMutex, which we already hold, for the futures...
    if (normalMapJobs.find(filename) != normalMapJobs.end())
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
    normalMapJobs[filename] = RunAsync(create, filename);
}

void BasicScene::AddFloatTexture(std::string name, TextureSceneEntity texture) {
    if (texture.renderFromObject.IsAnimated())
        Warning(&texture.loc, "Animated world to texture transforms are not supported. "
                              "Using start transform.");

    std::lock_guard<std::mutex> lock(textureMutex);
    if (texture.name != "imagemap" && texture.name != "ptex") {
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
        return FloatTexture::Create(texture.name, renderFromTexture, texDict,
                                    &texture.loc, alloc, Options->useGPU);
    };
    floatTextureJobs[name] = RunAsync(create, texture);
}

void BasicScene::AddSpectrumTexture(std::string name, TextureSceneEntity texture) {
    std::lock_guard<std::mutex> lock(textureMutex);

    if (texture.name != "imagemap" && texture.name != "ptex") {
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
        return SpectrumTexture::Create(texture.name, renderFromTexture, texDict,
                                       SpectrumType::Albedo, &texture.loc, alloc,
                                       Options->useGPU);
    };
    spectrumTextureJobs[name] = RunAsync(create, texture);
}

void BasicScene::AddLight(LightSceneEntity light) {
    Medium lightMedium = GetMedium(light.medium, &light.loc);
    std::lock_guard<std::mutex> lock(lightMutex);

    if (light.renderFromObject.IsAnimated())
        Warning(&light.loc,
                "Animated lights aren't supported. Using the start transform.");

    auto create = [this, light, lightMedium]() {
        return Light::Create(light.name, light.parameters,
                             light.renderFromObject.startTransform,
                             GetCamera().GetCameraTransform(), lightMedium, &light.loc,
                             threadAllocators.Get());
    };
    lightJobs.push_back(RunAsync(create));
}

int BasicScene::AddAreaLight(SceneEntity light) {
    std::lock_guard<std::mutex> lock(areaLightMutex);
    areaLights.push_back(std::move(light));
    return areaLights.size() - 1;
}

void BasicScene::AddShapes(pstd::span<ShapeSceneEntity> s) {
    std::lock_guard<std::mutex> lock(shapeMutex);
    std::move(std::begin(s), std::end(s), std::back_inserter(shapes));
}

void BasicScene::AddAnimatedShape(AnimatedShapeSceneEntity shape) {
    std::lock_guard<std::mutex> lock(animatedShapeMutex);
    animatedShapes.push_back(std::move(shape));
}

void BasicScene::AddInstanceDefinition(InstanceDefinitionSceneEntity instance) {
    InstanceDefinitionSceneEntity *def =
        new InstanceDefinitionSceneEntity(std::move(instance));

    std::lock_guard<std::mutex> lock(instanceDefinitionMutex);
    instanceDefinitions[def->name] = def;
}

void BasicScene::AddInstanceUses(pstd::span<InstanceSceneEntity> in) {
    std::lock_guard<std::mutex> lock(instanceUseMutex);
    std::move(std::begin(in), std::end(in), std::back_inserter(instances));
}

void BasicScene::Done() {
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

void BasicScene::CreateMaterials(const NamedTextures &textures,
                                 std::map<std::string, pbrt::Material> *namedMaterialsOut,
                                 std::vector<pbrt::Material> *materialsOut) {
    LOG_VERBOSE("Starting to consume %d normal map futures", normalMapJobs.size());
    std::lock_guard<std::mutex> lock(materialMutex);
    for (auto &job : normalMapJobs) {
        CHECK(normalMaps.find(job.first) == normalMaps.end());
        normalMaps[job.first] = job.second->GetResult();
    }
    normalMapJobs.clear();
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

        std::string fn =
            ResolveFilename(nm.second.parameters.GetOneString("normalmap", ""));
        Image *normalMap = nullptr;
        if (!fn.empty()) {
            CHECK(normalMaps.find(fn) != normalMaps.end());
            normalMap = normalMaps[fn];
        }

        TextureParameterDictionary texDict(&mtl.parameters, &textures);
        class Material m = Material::Create(type, texDict, normalMap, *namedMaterialsOut,
                                            &mtl.loc, alloc);
        (*namedMaterialsOut)[name] = m;
    }

    // Regular materials
    materialsOut->reserve(materials.size());
    for (const auto &mtl : materials) {
        Allocator alloc = threadAllocators.Get();
        std::string fn = ResolveFilename(mtl.parameters.GetOneString("normalmap", ""));
        Image *normalMap = nullptr;
        if (!fn.empty()) {
            CHECK(normalMaps.find(fn) != normalMaps.end());
            normalMap = normalMaps[fn];
        }

        TextureParameterDictionary texDict(&mtl.parameters, &textures);
        class Material m = Material::Create(mtl.name, texDict, normalMap,
                                            *namedMaterialsOut, &mtl.loc, alloc);
        materialsOut->push_back(m);
    }
}

NamedTextures BasicScene::CreateTextures() {
    NamedTextures textures;

    if (nMissingTextures > 0)
        ErrorExit("%d missing textures", nMissingTextures);

    // Consume futures
    LOG_VERBOSE("Starting to consume texture futures");
    // The lock shouldn't be necessary since only the main thread should be
    // active when CreateTextures() is called, but valgrind doesn't know
    // that...
    textureMutex.lock();
    for (auto &tex : floatTextureJobs)
        textures.floatTextures[tex.first] = tex.second->GetResult();
    floatTextureJobs.clear();
    for (auto &tex : spectrumTextureJobs)
        textures.albedoSpectrumTextures[tex.first] = tex.second->GetResult();
    spectrumTextureJobs.clear();
    textureMutex.unlock();
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
            tex.second.name, renderFromTexture, texDict, SpectrumType::Unbounded,
            &tex.second.loc, alloc, Options->useGPU);
        SpectrumTexture illumTex = SpectrumTexture::Create(
            tex.second.name, renderFromTexture, texDict, SpectrumType::Illuminant,
            &tex.second.loc, alloc, Options->useGPU);

        textures.unboundedSpectrumTextures[tex.first] = unboundedTex;
        textures.illuminantSpectrumTextures[tex.first] = illumTex;
    }

    // And do the rest serially
    for (auto &tex : serialFloatTextures) {
        Allocator alloc = threadAllocators.Get();

        pbrt::Transform renderFromTexture = tex.second.renderFromObject.startTransform;
        TextureParameterDictionary texDict(&tex.second.parameters, &textures);
        FloatTexture t = FloatTexture::Create(tex.second.name, renderFromTexture, texDict,
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
            tex.second.name, renderFromTexture, texDict, SpectrumType::Albedo,
            &tex.second.loc, alloc, Options->useGPU);
        SpectrumTexture unboundedTex = SpectrumTexture::Create(
            tex.second.name, renderFromTexture, texDict, SpectrumType::Unbounded,
            &tex.second.loc, alloc, Options->useGPU);
        SpectrumTexture illumTex = SpectrumTexture::Create(
            tex.second.name, renderFromTexture, texDict, SpectrumType::Illuminant,
            &tex.second.loc, alloc, Options->useGPU);

        textures.albedoSpectrumTextures[tex.first] = albedoTex;
        textures.unboundedSpectrumTextures[tex.first] = unboundedTex;
        textures.illuminantSpectrumTextures[tex.first] = illumTex;
    }

    LOG_VERBOSE("Done creating textures");
    return textures;
}

std::vector<Light> BasicScene::CreateLights(
    const NamedTextures &textures,
    std::map<int, pstd::vector<Light> *> *shapeIndexToAreaLights) {
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

    LOG_VERBOSE("Starting area lights");
    std::vector<Light> lights;
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

    LOG_VERBOSE("Finished area lights");

    LOG_VERBOSE("Starting to consume non-area light futures");
    std::lock_guard<std::mutex> lock(lightMutex);
    for (auto &job : lightJobs)
        lights.push_back(job->GetResult());
    LOG_VERBOSE("Finished consuming non-area light futures");

    return lights;
}

Primitive BasicScene::CreateAggregate(
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
    std::map<InternedString, Primitive> instanceDefinitions;
    std::mutex instanceDefinitionsMutex;
    std::vector<std::map<InternedString, InstanceDefinitionSceneEntity *>::iterator>
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
        auto iter = instanceDefinitions.find(inst.name);
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
