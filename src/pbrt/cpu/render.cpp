// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/render.h>

#include <pbrt/cameras.h>
#include <pbrt/cpu/aggregates.h>
#include <pbrt/cpu/integrators.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/parsedscene.h>
#include <pbrt/samplers.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/colorspace.h>

namespace pbrt {

void CPURender(ParsedScene &parsedScene) {
    Allocator alloc;

    // Create media first (so have them for the camera...)
    std::map<std::string, Medium> media = parsedScene.CreateMedia(alloc);

    bool haveScatteringMedia = false;
    auto findMedium = [&media, &haveScatteringMedia](const std::string &s,
                                                     const FileLoc *loc) -> Medium {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        haveScatteringMedia = true;
        return iter->second;
    };

    // Filter
    Filter filter = Filter::Create(parsedScene.filter.name, parsedScene.filter.parameters,
                                   &parsedScene.filter.loc, alloc);

    // Film
    // It's a little ugly to poke into the camera's parameters here, but we
    // have this circular dependency that Camera::Create() expects a
    // Film, yet now the film needs to know the exposure time from
    // the camera....
    Float exposureTime = parsedScene.camera.parameters.GetOneFloat("shutterclose", 1.f) -
                         parsedScene.camera.parameters.GetOneFloat("shutteropen", 0.f);
    if (exposureTime <= 0)
        ErrorExit(&parsedScene.camera.loc,
                  "The specified camera shutter times imply that the shutter "
                  "does not open.  A black image will result.");
    Film film = Film::Create(parsedScene.film.name, parsedScene.film.parameters,
                             exposureTime, filter, &parsedScene.film.loc, alloc);

    // Camera
    Medium cameraMedium = findMedium(parsedScene.camera.medium, &parsedScene.camera.loc);
    Camera camera = Camera::Create(parsedScene.camera.name, parsedScene.camera.parameters,
                                   cameraMedium, parsedScene.camera.cameraTransform, film,
                                   &parsedScene.camera.loc, alloc);

    // Create _Sampler_ for rendering
    Point2i fullImageResolution = camera.GetFilm().FullResolution();
    Sampler sampler =
        Sampler::Create(parsedScene.sampler.name, parsedScene.sampler.parameters,
                        fullImageResolution, &parsedScene.sampler.loc, alloc);

    // Textures
    NamedTextures textures = parsedScene.CreateTextures(alloc, false);

    // Materials
    std::map<std::string, pbrt::Material> namedMaterials;
    std::vector<pbrt::Material> materials;
    parsedScene.CreateMaterials(textures, alloc, &namedMaterials, &materials);
    bool haveSubsurface = false;
    for (const auto &mtl : parsedScene.materials)
        if (mtl.name == "subsurface")
            haveSubsurface = true;
    for (const auto &namedMtl : parsedScene.namedMaterials)
        if (namedMtl.second.name == "subsurface")
            haveSubsurface = true;

    // Lights (area lights will be done later, with shapes...)
    std::vector<Light> lights;
    std::mutex lightsMutex;
    lights.reserve(parsedScene.lights.size() + parsedScene.areaLights.size());
    for (const auto &light : parsedScene.lights) {
        Medium outsideMedium = findMedium(light.medium, &light.loc);
        if (light.renderFromObject.IsAnimated())
            Warning(&light.loc,
                    "Animated lights aren't supported. Using the start transform.");
        Light l = Light::Create(
            light.name, light.parameters, light.renderFromObject.startTransform,
            parsedScene.camera.cameraTransform, outsideMedium, &light.loc, alloc);
        // No need to hold the mutex here
        lights.push_back(l);
    }

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
        [&](const std::vector<ShapeSceneEntity> &shapes) -> std::vector<Primitive> {
        // Parallelize Shape::Create calls, which will in turn
        // parallelize PLY file loading, etc...
        pstd::vector<pstd::vector<Shape>> shapeVectors(shapes.size());
        ParallelFor(0, shapes.size(), [&](int64_t i) {
            const auto &sh = shapes[i];
            shapeVectors[i] =
                Shape::Create(sh.name, sh.renderFromObject, sh.objectFromRender,
                              sh.reverseOrientation, sh.parameters, &sh.loc, alloc);
        });

        std::vector<Primitive> primitives;
        for (size_t i = 0; i < shapes.size(); ++i) {
            const auto &sh = shapes[i];
            pstd::vector<Shape> &shapes = shapeVectors[i];
            if (shapes.empty())
                continue;

            FloatTexture alphaTex = getAlphaTexture(sh.parameters, &sh.loc);
            sh.parameters.ReportUnused();  // do now so can grab alpha...

            Material mtl = nullptr;
            if (!sh.materialName.empty()) {
                auto iter = namedMaterials.find(sh.materialName);
                if (iter == namedMaterials.end())
                    ErrorExit(&sh.loc, "%s: no named material defined.", sh.materialName);
                mtl = iter->second;
            } else {
                CHECK_LT(sh.materialIndex, materials.size());
                mtl = materials[sh.materialIndex];
            }

            MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                               findMedium(sh.outsideMedium, &sh.loc));

            for (auto &s : shapes) {
                // Possibly create area light for shape
                Light area = nullptr;
                if (sh.lightIndex != -1) {
                    CHECK_LT(sh.lightIndex, parsedScene.areaLights.size());
                    const auto &areaLightEntity = parsedScene.areaLights[sh.lightIndex];

                    area = Light::CreateArea(
                        areaLightEntity.name, areaLightEntity.parameters,
                        *sh.renderFromObject, mi, s, &areaLightEntity.loc, Allocator{});
                    if (area) {
                        std::lock_guard<std::mutex> lock(lightsMutex);
                        lights.push_back(area);
                    }
                }
                if (area == nullptr && !mi.IsMediumTransition() && !alphaTex)
                    primitives.push_back(new SimplePrimitive(s, mtl));
                else
                    primitives.push_back(
                        new GeometricPrimitive(s, mtl, area, mi, alphaTex));
            }
        }
        return primitives;
    };

    std::vector<Primitive> primitives = CreatePrimitivesForShapes(parsedScene.shapes);

    // Animated shapes
    auto CreatePrimitivesForAnimatedShapes =
        [&](const std::vector<AnimatedShapeSceneEntity> &shapes)
        -> std::vector<Primitive> {
        std::vector<Primitive> primitives;
        primitives.reserve(shapes.size());

        for (const auto &sh : shapes) {
            pstd::vector<Shape> shapes =
                Shape::Create(sh.name, sh.identity, sh.identity, sh.reverseOrientation,
                              sh.parameters, &sh.loc, alloc);
            if (shapes.empty())
                continue;

            FloatTexture alphaTex = getAlphaTexture(sh.parameters, &sh.loc);
            sh.parameters.ReportUnused();  // do now so can grab alpha...

            // Create initial shape or shapes for animated shape

            Material mtl = nullptr;
            if (!sh.materialName.empty()) {
                auto iter = namedMaterials.find(sh.materialName);
                if (iter == namedMaterials.end())
                    ErrorExit(&sh.loc, "%s: no named material defined.", sh.materialName);
                mtl = iter->second;
            } else {
                CHECK_LT(sh.materialIndex, materials.size());
                mtl = materials[sh.materialIndex];
            }

            MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                               findMedium(sh.outsideMedium, &sh.loc));

            std::vector<Primitive> prims;
            for (auto &s : shapes) {
                // Possibly create area light for shape
                Light area = nullptr;
                if (sh.lightIndex != -1) {
                    CHECK_LT(sh.lightIndex, parsedScene.areaLights.size());
                    const auto &areaLightEntity = parsedScene.areaLights[sh.lightIndex];

                    // TODO: shouldn't this always be true if we got here?
                    if (sh.renderFromObject.IsAnimated())
                        ErrorExit(&sh.loc, "Animated area lights are not supported.");

                    area = Light::CreateArea(
                        areaLightEntity.name, areaLightEntity.parameters,
                        sh.renderFromObject.startTransform, mi, s, &sh.loc, Allocator{});
                    if (area)
                        lights.push_back(area);
                }
                if (area == nullptr && !mi.IsMediumTransition() && !alphaTex)
                    prims.push_back(new SimplePrimitive(s, mtl));
                else
                    prims.push_back(new GeometricPrimitive(s, mtl, area, mi, alphaTex));
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
        }
        return primitives;
    };
    std::vector<Primitive> animatedPrimitives =
        CreatePrimitivesForAnimatedShapes(parsedScene.animatedShapes);
    primitives.insert(primitives.end(), animatedPrimitives.begin(),
                      animatedPrimitives.end());

    // Instance definitions
    std::map<std::string, Primitive> instanceDefinitions;
    std::mutex instanceDefinitionsMutex;
    std::vector<std::map<std::string, InstanceDefinitionSceneEntity>::iterator>
        instanceDefinitionIterators;
    for (auto iter = parsedScene.instanceDefinitions.begin();
         iter != parsedScene.instanceDefinitions.end(); ++iter)
        instanceDefinitionIterators.push_back(iter);
    ParallelFor(0, instanceDefinitionIterators.size(), [&](int64_t i) {
        const auto &inst = *instanceDefinitionIterators[i];

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
    });

    // Instances
    for (const auto &inst : parsedScene.instances) {
        auto iter = instanceDefinitions.find(inst.name);
        if (iter == instanceDefinitions.end())
            ErrorExit(&inst.loc, "%s: object instance not defined", inst.name);

        if (iter->second == nullptr)
            // empty instance
            continue;

        if (inst.renderFromInstance)
            primitives.push_back(
                new TransformedPrimitive(iter->second, inst.renderFromInstance));
        else
            primitives.push_back(
                new AnimatedPrimitive(iter->second, inst.renderFromInstanceAnim));
    }

    // Accelerator
    Primitive accel = nullptr;
    if (!primitives.empty())
        accel = CreateAccelerator(parsedScene.accelerator.name, std::move(primitives),
                                  parsedScene.accelerator.parameters);

    // Integrator
    const RGBColorSpace *integratorColorSpace = parsedScene.film.parameters.ColorSpace();
    std::unique_ptr<Integrator> integrator(Integrator::Create(
        parsedScene.integrator.name, parsedScene.integrator.parameters, camera, sampler,
        accel, lights, integratorColorSpace, &parsedScene.integrator.loc));

    // Helpful warnings
    if (haveScatteringMedia && parsedScene.integrator.name != "volpath" &&
        parsedScene.integrator.name != "simplevolpath" &&
        parsedScene.integrator.name != "bdpt" && parsedScene.integrator.name != "mlt")
        Warning("Scene has scattering media but \"%s\" integrator doesn't support "
                "volume scattering. Consider using \"volpath\", \"simplevolpath\", "
                "\"bdpt\", or \"mlt\".",
                parsedScene.integrator.name);

    bool haveLights = !lights.empty();
    for (const auto &m : media)
        haveLights |= m.second.IsEmissive();

    if (!haveLights && parsedScene.integrator.name != "ambientocclusion" &&
        parsedScene.integrator.name != "aov")
        Warning("No light sources defined in scene; rendering a black image.");

    if (parsedScene.film.name == "gbuffer" && !(parsedScene.integrator.name == "path" ||
                                                parsedScene.integrator.name == "volpath"))
        Warning(&parsedScene.film.loc,
                "GBufferFilm is not supported by the \"%s\" integrator. The channels "
                "other than R, G, B will be zero.",
                parsedScene.integrator.name);

    if (haveSubsurface && parsedScene.integrator.name != "volpath")
        Warning("Some objects in the scene have subsurface scattering, which is "
                "not supported by the %s integrator. Use the \"volpath\" integrator "
                "to render them correctly.",
                parsedScene.integrator.name);

    LOG_VERBOSE("Memory used after scene creation: %d", GetCurrentRSS());

    if (Options->pixelMaterial) {
        SampledWavelengths lambda = SampledWavelengths::SampleUniform(0.5f);

        CameraSample cs;
        cs.pFilm = *Options->pixelMaterial + Vector2f(0.5f, 0.5f);
        cs.time = 0.5f;
        cs.pLens = Point2f(0.5f, 0.5f);
        cs.filterWeight = 1;
        pstd::optional<CameraRay> cr = camera.GenerateRay(cs, lambda);
        if (!cr)
            ErrorExit("Unable to generate camera ray for specified pixel.");

        pstd::optional<ShapeIntersection> isect = accel.Intersect(cr->ray, Infinity);
        if (!isect)
            ErrorExit("No geometry visible at specified pixel.");

        const SurfaceInteraction &intr = isect->intr;
        if (!intr.material)
            ErrorExit("No material at intersection point.");

        Transform worldFromRender = camera.GetCameraTransform().WorldFromRender();
        Printf("World-space p: %s\n", worldFromRender(intr.p()));
        Printf("World-space n: %s\n", worldFromRender(intr.n));
        Printf("World-space ns: %s\n", worldFromRender(intr.shading.n));

        for (const auto &mtl : namedMaterials)
            if (mtl.second == intr.material) {
                Printf("Named material: %s\n", mtl.first);
                return;
            }

        // If we didn't find a named material, dump out the whole thing.
        Printf("%s\n\n", intr.material.ToString());

        return;
    }

    // Render!
    integrator->Render();

    LOG_VERBOSE("Memory used after rendering: %s", GetCurrentRSS());

    PtexTextureBase::ReportStats();
    ImageTextureBase::ClearCache();
    FreeBufferCaches();
}

}  // namespace pbrt
