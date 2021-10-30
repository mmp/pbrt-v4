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
#include <pbrt/samplers.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/parallel.h>

namespace pbrt {

void RenderCPU(BasicScene &parsedScene) {
    Allocator alloc;
    ThreadLocal<Allocator> threadAllocators([]() { return Allocator(); });

    // Create media first (so have them for the camera...)
    std::map<std::string, Medium> media = parsedScene.CreateMedia();

    // Textures
    LOG_VERBOSE("Starting textures");
    NamedTextures textures = parsedScene.CreateTextures();
    LOG_VERBOSE("Finished textures");

    // Lights
    std::map<int, pstd::vector<Light> *> shapeIndexToAreaLights;
    std::vector<Light> lights =
        parsedScene.CreateLights(textures, &shapeIndexToAreaLights);

    LOG_VERBOSE("Starting materials");
    std::map<std::string, pbrt::Material> namedMaterials;
    std::vector<pbrt::Material> materials;
    parsedScene.CreateMaterials(textures, &namedMaterials, &materials);
    LOG_VERBOSE("Finished materials");

    Primitive accel = parsedScene.CreateAggregate(textures, shapeIndexToAreaLights, media,
                                                  namedMaterials, materials);

    Camera camera = parsedScene.GetCamera();
    Film film = camera.GetFilm();
    Sampler sampler = parsedScene.GetSampler();

    // Integrator
    LOG_VERBOSE("Starting to create integrator");
    std::unique_ptr<Integrator> integrator(
        parsedScene.CreateIntegrator(camera, sampler, accel, lights));
    LOG_VERBOSE("Finished creating integrator");

    // Helpful warnings
    bool haveScatteringMedia = false;
    for (const auto &sh : parsedScene.shapes)
        if (!sh.insideMedium.empty() || !sh.outsideMedium.empty())
            haveScatteringMedia = true;
    for (const auto &sh : parsedScene.animatedShapes)
        if (!sh.insideMedium.empty() || !sh.outsideMedium.empty())
            haveScatteringMedia = true;

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

    if (film.Is<GBufferFilm>() && !(parsedScene.integrator.name == "path" ||
                                    parsedScene.integrator.name == "volpath"))
        Warning("GBufferFilm is not supported by the \"%s\" integrator. The channels "
                "other than R, G, B will be zero.",
                parsedScene.integrator.name);

    bool haveSubsurface = false;
    for (pbrt::Material mtl : materials)
        haveSubsurface |= mtl && mtl.HasSubsurfaceScattering();
    for (const auto &namedMtl : namedMaterials)
        haveSubsurface |= namedMtl.second && namedMtl.second.HasSubsurfaceScattering();

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

        int depth = 1;
        Ray ray = cr->ray;
        while (true) {
            pstd::optional<ShapeIntersection> isect = accel.Intersect(ray, Infinity);
            if (!isect) {
                if (depth == 1)
                    ErrorExit("No geometry visible at specified pixel.");
                else
                    break;
            }

            const SurfaceInteraction &intr = isect->intr;
            if (!intr.material)
                Warning("Ignoring \"interface\" material at intersection.");
            else {
                Transform worldFromRender = camera.GetCameraTransform().WorldFromRender();
                Printf("Intersection depth %d\n", depth);
                Printf("World-space p: %s\n", worldFromRender(intr.p()));
                Printf("World-space n: %s\n", worldFromRender(intr.n));
                Printf("World-space ns: %s\n", worldFromRender(intr.shading.n));
                Printf("Distance from camera: %f\n", Distance(intr.p(), cr->ray.o));

                bool isNamed = false;
                for (const auto &mtl : namedMaterials)
                    if (mtl.second == intr.material) {
                        Printf("Named material: %s\n\n", mtl.first);
                        isNamed = true;
                        break;
                    }
                if (!isNamed)
                    // If we didn't find a named material, dump out the whole thing.
                    Printf("%s\n\n", intr.material.ToString());

                ++depth;
                ray = intr.SpawnRay(ray.d);
            }
        }

        return;
    }

    // Render!
    integrator->Render();

    LOG_VERBOSE("Memory used after rendering: %s", GetCurrentRSS());

    PtexTextureBase::ReportStats();
    ImageTextureBase::ClearCache();
}

}  // namespace pbrt
