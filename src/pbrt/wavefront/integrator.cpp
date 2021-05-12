// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/integrator.h>

#include <pbrt/base/medium.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/aggregate.h>
#include <pbrt/gpu/memory.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/display.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/string.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/wavefront/aggregate.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <map>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // PBRT_BUILD_GPU_RENDERER

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Wavefront integrator pixel state", pathIntegratorBytes);

WavefrontPathIntegrator::WavefrontPathIntegrator(Allocator alloc, ParsedScene &scene) {
    // Allocate all of the data structures that represent the scene...
    std::map<std::string, Medium> media = scene.CreateMedia(alloc);

    haveMedia = false;
    // Check the shapes...
    for (const auto &shape : scene.shapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;
    for (const auto &shape : scene.animatedShapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;

    auto findMedium = [&](const std::string &s, const FileLoc *loc) -> Medium {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        haveMedia = true;
        return iter->second;
    };

    filter = Filter::Create(scene.filter.name, scene.filter.parameters, &scene.filter.loc,
                            alloc);

    Float exposureTime = scene.camera.parameters.GetOneFloat("shutterclose", 1.f) -
                         scene.camera.parameters.GetOneFloat("shutteropen", 0.f);
    if (exposureTime <= 0)
        ErrorExit(&scene.camera.loc,
                  "The specified camera shutter times imply that the shutter "
                  "does not open.  A black image will result.");

    film = Film::Create(scene.film.name, scene.film.parameters, exposureTime, filter,
                        &scene.film.loc, alloc);
    initializeVisibleSurface = film.UsesVisibleSurface();

    sampler = Sampler::Create(scene.sampler.name, scene.sampler.parameters,
                              film.FullResolution(), &scene.sampler.loc, alloc);
    samplesPerPixel = sampler.SamplesPerPixel();

    Medium cameraMedium = findMedium(scene.camera.medium, &scene.camera.loc);
    camera = Camera::Create(scene.camera.name, scene.camera.parameters, cameraMedium,
                            scene.camera.cameraTransform, film, &scene.camera.loc, alloc);

    // Textures
    LOG_VERBOSE("Starting to create textures");
    NamedTextures textures = scene.CreateTextures(alloc, Options->useGPU);
    LOG_VERBOSE("Done creating textures");

    pstd::vector<Light> allLights;

    envLights = alloc.new_object<pstd::vector<Light>>(alloc);
    for (const auto &light : scene.lights) {
        Medium outsideMedium = findMedium(light.medium, &light.loc);
        if (light.renderFromObject.IsAnimated())
            Warning(&light.loc,
                    "Animated lights aren't supported. Using the start transform.");

        Light l = Light::Create(
            light.name, light.parameters, light.renderFromObject.startTransform,
            scene.camera.cameraTransform, outsideMedium, &light.loc, alloc);

        if (l.Is<UniformInfiniteLight>() || l.Is<ImageInfiniteLight>() ||
            l.Is<PortalImageInfiniteLight>())
            envLights->push_back(l);

        allLights.push_back(l);
    }

    // Area lights...
    std::map<int, pstd::vector<Light> *> shapeIndexToAreaLights;
    for (size_t i = 0; i < scene.shapes.size(); ++i) {
        const auto &shape = scene.shapes[i];
        if (shape.lightIndex == -1)
            continue;

        auto isInterface = [&]() {
            std::string materialName;
            if (shape.materialIndex != -1)
                materialName = scene.materials[shape.materialIndex].name;
            else {
                for (auto iter = scene.namedMaterials.begin();
                     iter != scene.namedMaterials.end(); ++iter)
                    if (iter->first == shape.materialName) {
                        materialName = iter->second.parameters.GetOneString("type", "");
                        break;
                    }
            }
            return (materialName == "interface" || materialName == "none" ||
                    materialName.empty());
        };
        if (isInterface())
            continue;

        CHECK_LT(shape.lightIndex, scene.areaLights.size());
        const auto &areaLightEntity = scene.areaLights[shape.lightIndex];
        AnimatedTransform renderFromLight(*shape.renderFromObject);

        pstd::vector<Shape> shapes =
            Shape::Create(shape.name, shape.renderFromObject, shape.objectFromRender,
                          shape.reverseOrientation, shape.parameters, &shape.loc, alloc);

        if (shapes.empty())
            continue;

        Medium outsideMedium = findMedium(shape.outsideMedium, &shape.loc);

        FloatTexture alphaTex;
        std::string alphaTexName = shape.parameters.GetTexture("alpha");
        if (!alphaTexName.empty()) {
            if (textures.floatTextures.find(alphaTexName) !=
                textures.floatTextures.end()) {
                alphaTex = textures.floatTextures[alphaTexName];
                if (!BasicTextureEvaluator().CanEvaluate({alphaTex}, {}))
                    // A warning will be issued elsewhere...
                    alphaTex = nullptr;
            } else
                ErrorExit(&shape.loc,
                          "%s: couldn't find float texture for \"alpha\" parameter.",
                          alphaTexName);
        } else if (Float alpha = shape.parameters.GetOneFloat("alpha", 1.f); alpha < 1.f)
            alphaTex = alloc.new_object<FloatConstantTexture>(alpha);

        pstd::vector<Light> *lightsForShape =
            alloc.new_object<pstd::vector<Light>>(alloc);
        for (Shape sh : shapes) {
            if (renderFromLight.IsAnimated())
                ErrorExit(&shape.loc, "Animated lights are not supported.");
            DiffuseAreaLight *area = DiffuseAreaLight::Create(
                renderFromLight.startTransform, outsideMedium, areaLightEntity.parameters,
                areaLightEntity.parameters.ColorSpace(), &areaLightEntity.loc, alloc, sh,
                alphaTex);
            allLights.push_back(area);
            lightsForShape->push_back(area);
        }
        shapeIndexToAreaLights[i] = lightsForShape;
    }

    haveBasicEvalMaterial.fill(false);
    haveUniversalEvalMaterial.fill(false);
    haveSubsurface = false;
    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        aggregate = new OptiXAggregate(scene, alloc, textures, shapeIndexToAreaLights,
                                       media, &haveBasicEvalMaterial,
                                       &haveUniversalEvalMaterial, &haveSubsurface);
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
    } else
        aggregate = new CPUAggregate(scene, alloc, textures, shapeIndexToAreaLights,
                                     media, &haveBasicEvalMaterial,
                                     &haveUniversalEvalMaterial, &haveSubsurface);

    // Preprocess the light sources
    for (Light light : allLights)
        light.Preprocess(aggregate->Bounds());

    bool haveLights = !allLights.empty();
    for (const auto &m : media)
        haveLights |= m.second.IsEmissive();
    if (!haveLights)
        ErrorExit("No light sources specified");

    std::string lightSamplerName =
        scene.integrator.parameters.GetOneString("lightsampler", "bvh");
    if (allLights.size() == 1)
        lightSamplerName = "uniform";
    lightSampler = LightSampler::Create(lightSamplerName, allLights, alloc);

    if (scene.integrator.name != "path" && scene.integrator.name != "volpath")
        Warning(&scene.integrator.loc,
                "Ignoring specified integrator \"%s\": the wavefront integrator "
                "always uses a \"volpath\" integrator.",
                scene.integrator.name);

    // Integrator parameters
    regularize = scene.integrator.parameters.GetOneBool("regularize", false);
    maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    // Warn about unsupported stuff...
    if (Options->forceDiffuse)
        Warning("The wavefront integrator does not support --force-diffuse.");
    if (Options->writePartialImages)
        Warning("The wavefront integrator does not support --write-partial-images.");
    if (Options->recordPixelStatistics)
        Warning("The wavefront integrator does not support --pixelstats.");
    if (!Options->mseReferenceImage.empty())
        Warning("The wavefront integrator does not support --mse-reference-image.");
    if (!Options->mseReferenceOutput.empty())
        Warning("The wavefront integrator does not support --mse-reference-out.");

        ///////////////////////////////////////////////////////////////////////////
        // Allocate storage for all of the queues/buffers...

#ifdef PBRT_BUILD_GPU_RENDERER
    CUDATrackedMemoryResource *mr =
        dynamic_cast<CUDATrackedMemoryResource *>(gpuMemoryAllocator.resource());
    CHECK(mr != nullptr);
    size_t startSize = mr->BytesAllocated();
#endif  // PBRT_BUILD_GPU_RENDERER

    // Compute number of scanlines to render per pass
    Vector2i resolution = film.PixelBounds().Diagonal();
    // TODO: make this configurable. Base it on the amount of GPU memory?
    int maxSamples = 1024 * 1024;
    scanlinesPerPass = std::max(1, maxSamples / resolution.x);
    int nPasses = (resolution.y + scanlinesPerPass - 1) / scanlinesPerPass;
    scanlinesPerPass = (resolution.y + nPasses - 1) / nPasses;
    maxQueueSize = resolution.x * scanlinesPerPass;
    LOG_VERBOSE("Will render in %d passes %d scanlines per pass\n", nPasses,
                scanlinesPerPass);

    pixelSampleState = SOA<PixelSampleState>(maxQueueSize, alloc);

    rayQueues[0] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
    rayQueues[1] = alloc.new_object<RayQueue>(maxQueueSize, alloc);

    shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize, alloc);

    if (haveSubsurface) {
        bssrdfEvalQueue =
            alloc.new_object<GetBSSRDFAndProbeRayQueue>(maxQueueSize, alloc);
        subsurfaceScatterQueue =
            alloc.new_object<SubsurfaceScatterQueue>(maxQueueSize, alloc);
    }

    if (envLights->size())
        escapedRayQueue = alloc.new_object<EscapedRayQueue>(maxQueueSize, alloc);
    hitAreaLightQueue = alloc.new_object<HitAreaLightQueue>(maxQueueSize, alloc);

    basicEvalMaterialQueue = alloc.new_object<MaterialEvalQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveBasicEvalMaterial[1], haveBasicEvalMaterial.size() - 1));
    universalEvalMaterialQueue = alloc.new_object<MaterialEvalQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveUniversalEvalMaterial[1],
                            haveUniversalEvalMaterial.size() - 1));

    if (haveMedia) {
        mediumSampleQueue = alloc.new_object<MediumSampleQueue>(maxQueueSize, alloc);

        // TODO: in the presence of multiple PhaseFunction implementations,
        // it could be worthwhile to see which are present in the scene and
        // then initialize havePhase accordingly...
        pstd::array<bool, PhaseFunction::NumTags()> havePhase;
        havePhase.fill(true);
        mediumScatterQueue =
            alloc.new_object<MediumScatterQueue>(maxQueueSize, alloc, havePhase);
    }

    stats = alloc.new_object<Stats>(maxDepth, alloc);

#ifdef PBRT_BUILD_GPU_RENDERER
    size_t endSize = mr->BytesAllocated();
    pathIntegratorBytes += endSize - startSize;
#endif  // PBRT_BUILD_GPU_RENDERER
}

// WavefrontPathIntegrator Method Definitions
Float WavefrontPathIntegrator::Render() {
    // Prefetch allocations to GPU memory
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        int deviceIndex;
        CUDA_CHECK(cudaGetDevice(&deviceIndex));
        int hasConcurrentManagedAccess;
        CUDA_CHECK(cudaDeviceGetAttribute(&hasConcurrentManagedAccess,
                                          cudaDevAttrConcurrentManagedAccess,
                                          deviceIndex));

        // Copy all of the scene data structures over to GPU memory.  This
        // ensures that there isn't a big performance hitch for the first batch
        // of rays as that stuff is copied over on demand.
        if (hasConcurrentManagedAccess) {
            // Set things up so that we can still have read from the
            // WavefrontPathIntegrator struct on the CPU without hurting
            // performance. (This makes it possible to use the values of things
            // like WavefrontPathIntegrator::haveSubsurface to conditionally launch
            // kernels according to what's in the scene...)
            CUDA_CHECK(cudaMemAdvise(this, sizeof(*this), cudaMemAdviseSetReadMostly,
                                     /* ignored argument */ 0));
            CUDA_CHECK(cudaMemAdvise(this, sizeof(*this),
                                     cudaMemAdviseSetPreferredLocation, deviceIndex));

            // Copy all of the scene data structures over to GPU memory.  This
            // ensures that there isn't a big performance hitch for the first batch
            // of rays as that stuff is copied over on demand.
            CUDATrackedMemoryResource *mr =
                dynamic_cast<CUDATrackedMemoryResource *>(gpuMemoryAllocator.resource());
            CHECK(mr != nullptr);
            mr->PrefetchToGPU();
        } else {
            // TODO: on systems with basic unified memory, just launching a
            // kernel should cause everything to be copied over. Is an empty
            // kernel sufficient?
        }
    }
#endif  // PBRT_BUILD_GPU_RENDERER

    Timer timer;
    Vector2i resolution = film.PixelBounds().Diagonal();
    Bounds2i pixelBounds = film.PixelBounds();
    // Launch thread to copy image for display server, if enabled
    RGB *displayRGB = nullptr, *displayRGBHost = nullptr;
    std::atomic<bool> exitCopyThread{false};
    std::thread copyThread;

    if (!Options->displayServer.empty()) {
#ifdef PBRT_BUILD_GPU_RENDERER
        if (Options->useGPU) {
            // Allocate staging memory on the GPU to store the current WIP
            // image.
            CUDA_CHECK(
                cudaMalloc(&displayRGB, resolution.x * resolution.y * sizeof(RGB)));
            CUDA_CHECK(
                cudaMemset(displayRGB, 0, resolution.x * resolution.y * sizeof(RGB)));

            // Host-side memory for the WIP Image.  We'll just let this leak so
            // that the lambda passed to DisplayDynamic below doesn't access
            // freed memory after Render() returns...
            displayRGBHost = new RGB[resolution.x * resolution.y];

            copyThread = std::thread([&]() {
                GPURegisterThread("DISPLAY_SERVER_COPY_THREAD");

                // Copy back to the CPU using a separate stream so that we can
                // periodically but asynchronously pick up the latest results
                // from the GPU.
                cudaStream_t memcpyStream;
                CUDA_CHECK(cudaStreamCreate(&memcpyStream));
                GPUNameStream(memcpyStream, "DISPLAY_SERVER_COPY_STREAM");

                // Copy back to the host from the GPU buffer, without any
                // synthronization.
                while (!exitCopyThread) {
                    CUDA_CHECK(cudaMemcpyAsync(displayRGBHost, displayRGB,
                                               resolution.x * resolution.y * sizeof(RGB),
                                               cudaMemcpyDeviceToHost, memcpyStream));
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));

                    CUDA_CHECK(cudaStreamSynchronize(memcpyStream));
                }

                // Copy one more time to get the final image before exiting.
                CUDA_CHECK(cudaMemcpy(displayRGBHost, displayRGB,
                                      resolution.x * resolution.y * sizeof(RGB),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaDeviceSynchronize());
            });

            // Now on the CPU side, give the display system a lambda that
            // copies values from |displayRGBHost| into its buffers used for
            // sending messages to the display program (i.e., tev).
            DisplayDynamic(film.GetFilename(), {resolution.x, resolution.y},
                           {"R", "G", "B"},
                           [resolution, displayRGBHost](
                               Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                               int index = 0;
                               for (Point2i p : b) {
                                   RGB rgb = displayRGBHost[p.x + p.y * resolution.x];
                                   displayValue[0][index] = rgb.r;
                                   displayValue[1][index] = rgb.g;
                                   displayValue[2][index] = rgb.b;
                                   ++index;
                               }
                           });
        } else
#endif  // PBRT_BUILD_GPU_RENDERER
            DisplayDynamic(
                film.GetFilename(), Point2i(pixelBounds.Diagonal()), {"R", "G", "B"},
                [pixelBounds, this](Bounds2i b,
                                    pstd::span<pstd::span<Float>> displayValue) {
                    int index = 0;
                    for (Point2i p : b) {
                        RGB rgb =
                            film.GetPixelRGB(pixelBounds.pMin + p, 1.f /* splat scale */);
                        for (int c = 0; c < 3; ++c)
                            displayValue[c][index] = rgb[c];
                        ++index;
                    }
                });
    }

    int firstSampleIndex = 0, lastSampleIndex = samplesPerPixel;
    // Update sample index range based on debug start, if provided
    if (!Options->debugStart.empty()) {
        std::vector<int> values = SplitStringToInts(Options->debugStart, ',');
        if (values.size() != 2)
            ErrorExit("Expected two integer values for --debugstart.");

        firstSampleIndex = values[0];
        lastSampleIndex = firstSampleIndex + values[1];
    }

    ProgressReporter progress(lastSampleIndex - firstSampleIndex, "Rendering",
                              Options->quiet, Options->useGPU);
    for (int sampleIndex = firstSampleIndex; sampleIndex < lastSampleIndex;
         ++sampleIndex) {
        // Render image for sample _sampleIndex_
        LOG_VERBOSE("Starting to submit work for sample %d", sampleIndex);
        for (int y0 = pixelBounds.pMin.y; y0 < pixelBounds.pMax.y;
             y0 += scanlinesPerPass) {
            // Generate camera rays for current scanline range
            RayQueue *cameraRayQueue = CurrentRayQueue(0);
            Do(
                "Reset ray queue", PBRT_CPU_GPU_LAMBDA() {
                    PBRT_DBG("Starting scanlines at y0 = %d, sample %d / %d\n", y0,
                             sampleIndex, spp);
                    cameraRayQueue->Reset();
                });
            GenerateCameraRays(y0, sampleIndex);
            Do(
                "Update camera ray stats",
                PBRT_CPU_GPU_LAMBDA() { stats->cameraRays += cameraRayQueue->Size(); });

            // Trace rays and estimate radiance up to maximum ray depth
            for (int depth = 0; true; ++depth) {
                // Reset queues before tracing rays
                RayQueue *nextQueue = NextRayQueue(depth);
                Do(
                    "Reset queues before tracing rays", PBRT_CPU_GPU_LAMBDA() {
                        nextQueue->Reset();
                        // Reset queues before tracing next batch of rays
                        if (mediumSampleQueue)
                            mediumSampleQueue->Reset();
                        if (mediumScatterQueue)
                            mediumScatterQueue->Reset();

                        if (escapedRayQueue)
                            escapedRayQueue->Reset();
                        hitAreaLightQueue->Reset();

                        basicEvalMaterialQueue->Reset();
                        universalEvalMaterialQueue->Reset();

                        if (bssrdfEvalQueue)
                            bssrdfEvalQueue->Reset();
                        if (subsurfaceScatterQueue)
                            subsurfaceScatterQueue->Reset();
                    });

                // Follow active ray paths and accumulate radiance estimates
                GenerateRaySamples(depth, sampleIndex);
                // Find closest intersections along active rays
                aggregate->IntersectClosest(maxQueueSize, escapedRayQueue,
                                            hitAreaLightQueue, basicEvalMaterialQueue,
                                            universalEvalMaterialQueue, mediumSampleQueue,
                                            CurrentRayQueue(depth), NextRayQueue(depth));

                if (depth > 0) {
                    // As above, with the indexing...
                    RayQueue *statsQueue = CurrentRayQueue(depth);
                    Do(
                        "Update indirect ray stats", PBRT_CPU_GPU_LAMBDA() {
                            stats->indirectRays[depth] += statsQueue->Size();
                        });
                }
                if (haveMedia)
                    SampleMediumInteraction(depth);
                if (escapedRayQueue)
                    HandleEscapedRays(depth);
                HandleRayFoundEmission(depth);
                if (depth == maxDepth)
                    break;
                EvaluateMaterialsAndBSDFs(depth);
                // Do immediately so that we have space for shadow rays for subsurface..
                TraceShadowRays(depth);
                if (haveSubsurface)
                    SampleSubsurface(depth);
            }

            UpdateFilm();
            // Copy updated film pixels to buffer for display
#ifdef PBRT_BUILD_GPU_RENDERER
            if (Options->useGPU && !Options->displayServer.empty())
                GPUParallelFor(
                    "Update Display RGB Buffer", maxQueueSize,
                    PBRT_CPU_GPU_LAMBDA(int pixelIndex) {
                        Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
                        if (!InsideExclusive(pPixel, film.PixelBounds()))
                            return;

                        Point2i p(pPixel - film.PixelBounds().pMin);
                        displayRGB[p.x + p.y * resolution.x] = film.GetPixelRGB(pPixel);
                    });
#endif  //  PBRT_BUILD_GPU_RENDERER
        }

        progress.Update();
    }
    progress.Done();
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU)
        GPUWait();
#endif  // PBRT_BUILD_GPU_RENDERER
    Float seconds = timer.ElapsedSeconds();
    // Shut down display server thread, if active
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        // Wait until rendering is all done before we start to shut down the
        // display stuff..
        if (!Options->displayServer.empty()) {
            exitCopyThread = true;
            copyThread.join();
        }

        // Another synchronization to make sure no kernels are running on the
        // GPU so that we can safely access unified memory from the CPU.
        GPUWait();
    }
#endif  // PBRT_BUILD_GPU_RENDERER

    return seconds;
}

void WavefrontPathIntegrator::HandleEscapedRays(int depth) {
    ForAllQueued(
        "Handle escaped rays", escapedRayQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const EscapedRayWorkItem w) {
            // Update pixel radiance for escaped ray
            SampledSpectrum L(0.f);
            for (const auto &light : *envLights) {
                if (SampledSpectrum Le = light.Le(Ray(w.rayo, w.rayd), w.lambda); Le) {
                    // Compute path radiance contribution from infinite light
                    PBRT_DBG("L %f %f %f %f T_hat %f %f %f %f Le %f %f %f %f", L[0], L[1],
                             L[2], L[3], w.T_hat[0], w.T_hat[1], w.T_hat[2], w.T_hat[3],
                             Le[0], Le[1], Le[2], Le[3]);
                    PBRT_DBG("pdf uni %f %f %f %f pdf nee %f %f %f %f", w.uniPathPDF[0],
                             w.uniPathPDF[1], w.uniPathPDF[2], w.uniPathPDF[3],
                             w.lightPathPDF[0], w.lightPathPDF[1], w.lightPathPDF[2],
                             w.lightPathPDF[3]);

                    if (depth == 0 || w.specularBounce) {
                        L += w.T_hat * Le / w.uniPathPDF.Average();
                    } else {
                        // Compute MIS-weighted radiance contribution from infinite light
                        LightSampleContext ctx = w.prevIntrCtx;
                        Float lightChoicePDF = lightSampler.PDF(ctx, light);
                        SampledSpectrum lightPathPDF =
                            w.lightPathPDF * lightChoicePDF *
                            light.PDF_Li(ctx, w.rayd, LightSamplingMode::WithMIS);
                        L += w.T_hat * Le / (w.uniPathPDF + lightPathPDF).Average();
                    }
                }
            }
            if (L) {
                PBRT_DBG("Added L %f %f %f %f for escaped ray pixel index %d\n", L[0],
                         L[1], L[2], L[3], w.pixelIndex);

                L += pixelSampleState.L[w.pixelIndex];
                pixelSampleState.L[w.pixelIndex] = L;
            }
        });
}

void WavefrontPathIntegrator::HandleRayFoundEmission(int depth) {
    ForAllQueued(
        "Handle emitters hit by indirect rays", hitAreaLightQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const HitAreaLightWorkItem w) {
            // Find emitted radiance from surface that ray hit
            SampledSpectrum Le = w.areaLight.L(w.p, w.n, w.uv, w.wo, w.lambda);
            if (!Le)
                return;
            PBRT_DBG("Got Le %f %f %f %f from hit area light at depth %d\n", Le[0], Le[1],
                     Le[2], Le[3], depth);

            // Compute area light's weighted radiance contribution to the path
            SampledSpectrum L(0.f);
            if (depth == 0 || w.isSpecularBounce) {
                L = w.T_hat * Le / w.uniPathPDF.Average();
            } else {
                // Compute MIS-weighted radiance contribution from area light
                Vector3f wi = -w.wo;
                LightSampleContext ctx = w.prevIntrCtx;
                Float lightChoicePDF = lightSampler.PDF(ctx, w.areaLight);
                Float lightPDF = lightChoicePDF *
                                 w.areaLight.PDF_Li(ctx, wi, LightSamplingMode::WithMIS);

                SampledSpectrum uniPathPDF = w.uniPathPDF;
                SampledSpectrum lightPathPDF = w.lightPathPDF * lightPDF;
                L = w.T_hat * Le / (uniPathPDF + lightPathPDF).Average();
            }

            PBRT_DBG("Added L %f %f %f %f for pixel index %d\n", L[0], L[1], L[2], L[3],
                     w.pixelIndex);

            // Update _L_ in _PixelSampleState_ for area light's radiance
            L += pixelSampleState.L[w.pixelIndex];
            pixelSampleState.L[w.pixelIndex] = L;
        });
}

void WavefrontPathIntegrator::TraceShadowRays(int depth) {
    if (haveMedia)
        aggregate->IntersectShadowTr(maxQueueSize, shadowRayQueue, &pixelSampleState);
    else
        aggregate->IntersectShadow(maxQueueSize, shadowRayQueue, &pixelSampleState);
    // Reset shadow ray queue
    Do(
        "Reset shadowRayQueue", PBRT_CPU_GPU_LAMBDA() {
            stats->shadowRays[depth] += shadowRayQueue->Size();
            shadowRayQueue->Reset();
        });
}

WavefrontPathIntegrator::Stats::Stats(int maxDepth, Allocator alloc)
    : indirectRays(maxDepth + 1, alloc), shadowRays(maxDepth, alloc) {}

std::string WavefrontPathIntegrator::Stats::Print() const {
    std::string s;
    s += StringPrintf("    %-42s               %12" PRIu64 "\n", "Camera rays",
                      cameraRays);
    for (int i = 1; i < indirectRays.size(); ++i)
        s += StringPrintf("    %-42s               %12" PRIu64 "\n",
                          StringPrintf("Indirect rays, depth %-3d", i), indirectRays[i]);
    for (int i = 0; i < shadowRays.size(); ++i)
        s += StringPrintf("    %-42s               %12" PRIu64 "\n",
                          StringPrintf("Shadow rays, depth %-3d", i), shadowRays[i]);
    return s;
}

}  // namespace pbrt
