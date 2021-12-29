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

static void updateMaterialNeeds(
    Material m, pstd::array<bool, Material::NumTags()> *haveBasicEvalMaterial,
    pstd::array<bool, Material::NumTags()> *haveUniversalEvalMaterial,
    bool *haveSubsurface, bool *haveMedia) {
    if (!m)
        return;

    if (MixMaterial *mix = m.CastOrNullptr<MixMaterial>(); mix) {
        // This is a somewhat odd place for this check, but it's convenient...
        if (!m.CanEvaluateTextures(BasicTextureEvaluator()))
            ErrorExit("\"mix\" material has a texture that can't be evaluated with the "
                      "BasicTextureEvaluator, which is all that is currently supported "
                      "int the wavefront renderer--sorry! %s",
                      *mix);

        updateMaterialNeeds(mix->GetMaterial(0), haveBasicEvalMaterial,
                            haveUniversalEvalMaterial, haveSubsurface, haveMedia);
        updateMaterialNeeds(mix->GetMaterial(1), haveBasicEvalMaterial,
                            haveUniversalEvalMaterial, haveSubsurface, haveMedia);
        return;
    }

    *haveSubsurface |= m.HasSubsurfaceScattering();
    *haveMedia |= (m == nullptr);  // interface material

    FloatTexture displace = m.GetDisplacement();
    if (m.CanEvaluateTextures(BasicTextureEvaluator()) &&
        (!displace || BasicTextureEvaluator().CanEvaluate({displace}, {})))
        (*haveBasicEvalMaterial)[m.Tag()] = true;
    else
        (*haveUniversalEvalMaterial)[m.Tag()] = true;
}

WavefrontPathIntegrator::WavefrontPathIntegrator(
    pstd::pmr::memory_resource *memoryResource, BasicScene &scene)
    : memoryResource(memoryResource) {
    ThreadLocal<Allocator> threadAllocators(
        [memoryResource]() { return Allocator(memoryResource); });

    Allocator alloc = threadAllocators.Get();

    // Allocate all of the data structures that represent the scene...
    std::map<std::string, Medium> media = scene.CreateMedia();

    // "haveMedia" is a bit of a misnomer in that determines both whether
    // queues are allocated for the medium sampling kernels and they are
    // launched as well as whether the ray marching shadow ray kernel is
    // launched... Thus, it will be true if there actually are no media,
    // but some "interface" materials are present in the scene.
    haveMedia = false;
    // Check the shapes and instance definitions...
    for (const auto &shape : scene.shapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;
    for (const auto &shape : scene.animatedShapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;
    for (const auto &instanceDefinition: scene.instanceDefinitions) {
        for (const auto &shape : instanceDefinition.second->shapes)
            if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
                haveMedia = true;
        for (const auto &shape : instanceDefinition.second->animatedShapes)
            if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
                haveMedia = true;
    }

    // Textures
    LOG_VERBOSE("Starting to create textures");
    NamedTextures textures = scene.CreateTextures();
    LOG_VERBOSE("Done creating textures");

    LOG_VERBOSE("Starting to create lights");
    pstd::vector<Light> allLights;
    std::map<int, pstd::vector<Light> *> shapeIndexToAreaLights;

    infiniteLights = alloc.new_object<pstd::vector<Light>>(alloc);

    for (Light l : scene.CreateLights(textures, &shapeIndexToAreaLights)) {
        if (l.Is<UniformInfiniteLight>() || l.Is<ImageInfiniteLight>() ||
            l.Is<PortalImageInfiniteLight>())
            infiniteLights->push_back(l);

        allLights.push_back(l);
    }
    LOG_VERBOSE("Done creating lights");

    LOG_VERBOSE("Starting to create materials");
    std::map<std::string, pbrt::Material> namedMaterials;
    std::vector<pbrt::Material> materials;
    scene.CreateMaterials(textures, &namedMaterials, &materials);

    haveBasicEvalMaterial.fill(false);
    haveUniversalEvalMaterial.fill(false);
    haveSubsurface = false;
    for (Material m : materials)
        updateMaterialNeeds(m, &haveBasicEvalMaterial, &haveUniversalEvalMaterial,
                            &haveSubsurface, &haveMedia);
    for (const auto &m : namedMaterials)
        updateMaterialNeeds(m.second, &haveBasicEvalMaterial, &haveUniversalEvalMaterial,
                            &haveSubsurface, &haveMedia);
    LOG_VERBOSE("Finished creating materials");

    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        CUDATrackedMemoryResource *mr =
            dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
        CHECK(mr);
        aggregate = new OptiXAggregate(scene, mr, textures, shapeIndexToAreaLights, media,
                                       namedMaterials, materials);
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
    } else
        aggregate = new CPUAggregate(scene, textures, shapeIndexToAreaLights, media,
                                     namedMaterials, materials);

    // Preprocess the light sources
    for (Light light : allLights)
        light.Preprocess(aggregate->Bounds());

    bool haveLights = !allLights.empty();
    for (const auto &m : media)
        haveLights |= m.second.IsEmissive();
    if (!haveLights)
        ErrorExit("No light sources specified");

    LOG_VERBOSE("Starting to create light sampler");
    std::string lightSamplerName =
        scene.integrator.parameters.GetOneString("lightsampler", "bvh");
    if (allLights.size() == 1)
        lightSamplerName = "uniform";
    lightSampler = LightSampler::Create(lightSamplerName, allLights, alloc);
    LOG_VERBOSE("Finished creating light sampler");

    if (scene.integrator.name != "path" && scene.integrator.name != "volpath")
        Warning(&scene.integrator.loc,
                "Ignoring specified integrator \"%s\": the wavefront integrator "
                "always uses a \"volpath\" integrator.",
                scene.integrator.name);

    // Integrator parameters
    regularize = scene.integrator.parameters.GetOneBool("regularize", false);
    maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    camera = scene.GetCamera();
    film = camera.GetFilm();
    filter = film.GetFilter();
    sampler = scene.GetSampler();

    initializeVisibleSurface = film.UsesVisibleSurface();
    samplesPerPixel = sampler.SamplesPerPixel();

    // Warn about unsupported stuff...
    if (Options->forceDiffuse)
        ErrorExit("The wavefront integrator does not support --force-diffuse.");
    if (Options->writePartialImages)
        Warning("The wavefront integrator does not support --write-partial-images.");
    if (Options->recordPixelStatistics)
        ErrorExit("The wavefront integrator does not support --pixelstats.");
    if (!Options->mseReferenceImage.empty())
        ErrorExit("The wavefront integrator does not support --mse-reference-image.");
    if (!Options->mseReferenceOutput.empty())
        ErrorExit("The wavefront integrator does not support --mse-reference-out.");

        ///////////////////////////////////////////////////////////////////////////
        // Allocate storage for all of the queues/buffers...

#ifdef PBRT_BUILD_GPU_RENDERER
    size_t startSize = 0;
    if (Options->useGPU) {
        CUDATrackedMemoryResource *mr =
            dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
        CHECK(mr);
        startSize = mr->BytesAllocated();
    }
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

    if (infiniteLights->size())
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
    if (Options->useGPU) {
        CUDATrackedMemoryResource *mr =
            dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
        CHECK(mr);
        size_t endSize = mr->BytesAllocated();
        pathIntegratorBytes += endSize - startSize;
    }
#endif  // PBRT_BUILD_GPU_RENDERER
}

// WavefrontPathIntegrator Method Definitions
Float WavefrontPathIntegrator::Render() {
    Bounds2i pixelBounds = film.PixelBounds();
    Vector2i resolution = pixelBounds.Diagonal();
    Timer timer;
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
                dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
            CHECK(mr);
            mr->PrefetchToGPU();
        } else {
            // TODO: on systems with basic unified memory, just launching a
            // kernel should cause everything to be copied over. Is an empty
            // kernel sufficient?
        }
    }
#endif  // PBRT_BUILD_GPU_RENDERER

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

    // Loop over sample indices and evaluate pixel samples
    int firstSampleIndex = 0, lastSampleIndex = samplesPerPixel;
    // Update sample index range based on debug start, if provided
    if (!Options->debugStart.empty()) {
        std::vector<int> values = SplitStringToInts(Options->debugStart, ',');
        if (values.size() != 1 && values.size() != 2)
            ErrorExit("Expected either one or two integer values for --debugstart.");

        firstSampleIndex = values[0];
        if (values.size() == 2)
            lastSampleIndex = firstSampleIndex + values[1];
        else
            lastSampleIndex = firstSampleIndex + 1;
    }

    ProgressReporter progress(lastSampleIndex - firstSampleIndex, "Rendering",
                              Options->quiet, Options->useGPU);
    for (int sampleIndex = firstSampleIndex; sampleIndex < lastSampleIndex;
         ++sampleIndex) {
        // Attempt to work around issue #145.
#if !(defined(PBRT_IS_WINDOWS) && defined(PBRT_BUILD_GPU_RENDERER) && \
      __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 1)
        CheckCallbackScope _([&]() {
            return StringPrintf("Wavefront rendering failed at sample %d. Debug with "
                                "\"--debugstart %d\"\n",
                                sampleIndex, sampleIndex);
        });
#endif

        // Render image for sample _sampleIndex_
        LOG_VERBOSE("Starting to submit work for sample %d", sampleIndex);
        for (int y0 = pixelBounds.pMin.y; y0 < pixelBounds.pMax.y;
             y0 += scanlinesPerPass) {
            // Generate camera rays for current scanline range
            RayQueue *cameraRayQueue = CurrentRayQueue(0);
            Do(
                "Reset ray queue", PBRT_CPU_GPU_LAMBDA() {
                    PBRT_DBG("Starting scanlines at y0 = %d, sample %d / %d\n", y0,
                             sampleIndex, samplesPerPixel);
                    cameraRayQueue->Reset();
                });
            GenerateCameraRays(y0, sampleIndex);
            Do(
                "Update camera ray stats",
                PBRT_CPU_GPU_LAMBDA() { stats->cameraRays += cameraRayQueue->Size(); });

            // Trace rays and estimate radiance up to maximum ray depth
            for (int wavefrontDepth = 0; true; ++wavefrontDepth) {
                // Reset queues before tracing rays
                RayQueue *nextQueue = NextRayQueue(wavefrontDepth);
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
                GenerateRaySamples(wavefrontDepth, sampleIndex);

                // Find closest intersections along active rays
                aggregate->IntersectClosest(
                    maxQueueSize, CurrentRayQueue(wavefrontDepth), escapedRayQueue,
                    hitAreaLightQueue, basicEvalMaterialQueue, universalEvalMaterialQueue,
                    mediumSampleQueue, NextRayQueue(wavefrontDepth));

                if (wavefrontDepth > 0) {
                    // As above, with the indexing...
                    RayQueue *statsQueue = CurrentRayQueue(wavefrontDepth);
                    Do(
                        "Update indirect ray stats", PBRT_CPU_GPU_LAMBDA() {
                            stats->indirectRays[wavefrontDepth] += statsQueue->Size();
                        });
                }

                SampleMediumInteraction(wavefrontDepth);

                HandleEscapedRays();

                HandleEmissiveIntersection();

                if (wavefrontDepth == maxDepth)
                    break;

                EvaluateMaterialsAndBSDFs(wavefrontDepth);

                // Do immediately so that we have space for shadow rays for subsurface..
                TraceShadowRays(wavefrontDepth);

                SampleSubsurface(wavefrontDepth);
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

void WavefrontPathIntegrator::HandleEscapedRays() {
    if (!escapedRayQueue)
        return;
    ForAllQueued(
        "Handle escaped rays", escapedRayQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const EscapedRayWorkItem w) {
            // Compute weighted radiance for escaped ray
            SampledSpectrum L(0.f);
            for (const auto &light : *infiniteLights) {
                if (SampledSpectrum Le = light.Le(Ray(w.rayo, w.rayd), w.lambda); Le) {
                    // Compute path radiance contribution from infinite light
                    PBRT_DBG("L %f %f %f %f beta %f %f %f %f Le %f %f %f %f", L[0], L[1],
                             L[2], L[3], w.beta[0], w.beta[1], w.beta[2], w.beta[3],
                             Le[0], Le[1], Le[2], Le[3]);
                    PBRT_DBG("pdf uni %f %f %f %f pdf nee %f %f %f %f", w.inv_w_u[0],
                             w.inv_w_u[1], w.inv_w_u[2], w.inv_w_u[3], w.inv_w_l[0],
                             w.inv_w_l[1], w.inv_w_l[2], w.inv_w_l[3]);

                    if (w.depth == 0 || w.specularBounce) {
                        L += w.beta * Le / w.inv_w_u.Average();
                    } else {
                        // Compute MIS-weighted radiance contribution from infinite light
                        LightSampleContext ctx = w.prevIntrCtx;
                        Float lightChoicePDF = lightSampler.PMF(ctx, light);
                        SampledSpectrum inv_w_l =
                            w.inv_w_l * lightChoicePDF * light.PDF_Li(ctx, w.rayd, true);
                        L += w.beta * Le / (w.inv_w_u + inv_w_l).Average();
                    }
                }
            }

            // Update pixel radiance if ray's radiance is nonzero
            if (L) {
                PBRT_DBG("Added L %f %f %f %f for escaped ray pixel index %d\n", L[0],
                         L[1], L[2], L[3], w.pixelIndex);

                L += pixelSampleState.L[w.pixelIndex];
                pixelSampleState.L[w.pixelIndex] = L;
            }
        });
}

void WavefrontPathIntegrator::HandleEmissiveIntersection() {
    ForAllQueued(
        "Handle emitters hit by indirect rays", hitAreaLightQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const HitAreaLightWorkItem w) {
            // Find emitted radiance from surface that ray hit
            SampledSpectrum Le = w.areaLight.L(w.p, w.n, w.uv, w.wo, w.lambda);
            if (!Le)
                return;
            PBRT_DBG("Got Le %f %f %f %f from hit area light at depth %d\n", Le[0], Le[1],
                     Le[2], Le[3], w.depth);

            // Compute area light's weighted radiance contribution to the path
            SampledSpectrum L(0.f);
            if (w.depth == 0 || w.specularBounce) {
                L = w.beta * Le / w.inv_w_u.Average();
            } else {
                // Compute MIS-weighted radiance contribution from area light
                Vector3f wi = -w.wo;
                LightSampleContext ctx = w.prevIntrCtx;
                Float lightChoicePDF = lightSampler.PMF(ctx, w.areaLight);
                Float lightPDF = lightChoicePDF * w.areaLight.PDF_Li(ctx, wi, true);

                SampledSpectrum inv_w_u = w.inv_w_u;
                SampledSpectrum inv_w_l = w.inv_w_l * lightPDF;
                L = w.beta * Le / (inv_w_u + inv_w_l).Average();
            }

            PBRT_DBG("Added L %f %f %f %f for pixel index %d\n", L[0], L[1], L[2], L[3],
                     w.pixelIndex);

            // Update _L_ in _PixelSampleState_ for area light's radiance
            L += pixelSampleState.L[w.pixelIndex];
            pixelSampleState.L[w.pixelIndex] = L;
        });
}

void WavefrontPathIntegrator::TraceShadowRays(int wavefrontDepth) {
    if (haveMedia)
        aggregate->IntersectShadowTr(maxQueueSize, shadowRayQueue, &pixelSampleState);
    else
        aggregate->IntersectShadow(maxQueueSize, shadowRayQueue, &pixelSampleState);
    // Reset shadow ray queue
    Do(
        "Reset shadowRayQueue", PBRT_CPU_GPU_LAMBDA() {
            stats->shadowRays[wavefrontDepth] += shadowRayQueue->Size();
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
