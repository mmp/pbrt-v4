// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/pathintegrator.h>

#include <pbrt/base/medium.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/optix.h>
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

#include <atomic>
#include <cstring>
#include <iostream>
#include <map>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#ifdef NVTX
#ifdef PBRT_IS_WINDOWS
#include <windows.h>
#else
#include <sys/syscall.h>
#endif  // PBRT_IS_WINDOWS
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#endif

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/GPU path integrator pixel state", pathIntegratorBytes);

GPUPathIntegrator::GPUPathIntegrator(Allocator alloc, const ParsedScene &scene) {
    // Allocate all of the data structures that represent the scene...
    std::map<std::string, MediumHandle> media = scene.CreateMedia(alloc);

    haveMedia = false;
    // Check the shapes...
    for (const auto &shape : scene.shapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;
    for (const auto &shape : scene.animatedShapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;

    auto findMedium = [&](const std::string &s, const FileLoc *loc) -> MediumHandle {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        haveMedia = true;
        return iter->second;
    };

    filter = FilterHandle::Create(scene.filter.name, scene.filter.parameters,
                                  &scene.filter.loc, alloc);

    Float exposureTime = scene.camera.parameters.GetOneFloat("shutterclose", 1.f) -
                         scene.camera.parameters.GetOneFloat("shutteropen", 0.f);
    if (exposureTime <= 0)
        ErrorExit(&scene.camera.loc,
                  "The specified camera shutter times imply that the shutter "
                  "does not open.  A black image will result.");

    film = FilmHandle::Create(scene.film.name, scene.film.parameters, exposureTime,
                              filter, &scene.film.loc, alloc);
    initializeVisibleSurface = film.UsesVisibleSurface();

    sampler = SamplerHandle::Create(scene.sampler.name, scene.sampler.parameters,
                                    film.FullResolution(), &scene.sampler.loc, alloc);

    MediumHandle cameraMedium = findMedium(scene.camera.medium, &scene.camera.loc);
    camera = CameraHandle::Create(scene.camera.name, scene.camera.parameters,
                                  cameraMedium, scene.camera.cameraTransform, film,
                                  &scene.camera.loc, alloc);

    pstd::vector<LightHandle> allLights;

    for (const auto &light : scene.lights) {
        MediumHandle outsideMedium = findMedium(light.medium, &light.loc);
        if (light.renderFromObject.IsAnimated())
            Warning(&light.loc,
                    "Animated lights aren't supported. Using the start transform.");

        LightHandle l = LightHandle::Create(
            light.name, light.parameters, light.renderFromObject.startTransform,
            scene.camera.cameraTransform, outsideMedium, &light.loc, alloc);

        if (l.Is<UniformInfiniteLight>() || l.Is<ImageInfiniteLight>() ||
            l.Is<PortalImageInfiniteLight>()) {
            if (envLight)
                Warning(&light.loc,
                        "Multiple infinite lights specified. Using this one.");
            envLight = l;
        }

        allLights.push_back(l);
    }

    // Area lights...
    std::map<int, pstd::vector<LightHandle> *> shapeIndexToAreaLights;
    for (size_t i = 0; i < scene.shapes.size(); ++i) {
        const auto &shape = scene.shapes[i];
        if (shape.lightIndex == -1)
            continue;
        CHECK_LT(shape.lightIndex, scene.areaLights.size());
        const auto &areaLightEntity = scene.areaLights[shape.lightIndex];
        AnimatedTransform renderFromLight(*shape.renderFromObject);

        pstd::vector<ShapeHandle> shapeHandles = ShapeHandle::Create(
            shape.name, shape.renderFromObject, shape.objectFromRender,
            shape.reverseOrientation, shape.parameters, &shape.loc, alloc);

        if (shapeHandles.empty())
            continue;

        MediumHandle outsideMedium = findMedium(shape.outsideMedium, &shape.loc);

        pstd::vector<LightHandle> *lightsForShape =
            alloc.new_object<pstd::vector<LightHandle>>(alloc);
        for (ShapeHandle sh : shapeHandles) {
            if (renderFromLight.IsAnimated())
                ErrorExit(&shape.loc, "Animated lights are not supported.");
            DiffuseAreaLight *area = DiffuseAreaLight::Create(
                renderFromLight.startTransform, outsideMedium, areaLightEntity.parameters,
                areaLightEntity.parameters.ColorSpace(), &areaLightEntity.loc, alloc, sh);
            allLights.push_back(area);
            lightsForShape->push_back(area);
        }
        shapeIndexToAreaLights[i] = lightsForShape;
    }

    haveBasicEvalMaterial.fill(false);
    haveUniversalEvalMaterial.fill(false);
    haveSubsurface = false;
    accel = new GPUAccel(scene, alloc, nullptr /* cuda stream */, shapeIndexToAreaLights,
                         media, &haveBasicEvalMaterial, &haveUniversalEvalMaterial,
                         &haveSubsurface);

    // Preprocess the light sources
    for (LightHandle light : allLights)
        light.Preprocess(accel->Bounds());

    bool haveLights = !allLights.empty();
    for (const auto &m : media)
        haveLights |= m.second.IsEmissive();
    if (!haveLights)
        ErrorExit("No light sources specified");

    std::string lightSamplerName =
        scene.integrator.parameters.GetOneString("lightsampler", "bvh");
    if (allLights.size() == 1)
        lightSamplerName = "uniform";
    lightSampler = LightSamplerHandle::Create(lightSamplerName, allLights, alloc);

    if (scene.integrator.name != "path" && scene.integrator.name != "volpath")
        Warning(&scene.integrator.loc,
                "The GPU renderer always uses a \"volpath\" integrator.");

    // Integrator parameters
    regularize = scene.integrator.parameters.GetOneBool("regularize", false);
    maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    ///////////////////////////////////////////////////////////////////////////
    // Allocate storage for all of the queues/buffers...

    CUDATrackedMemoryResource *mr =
        dynamic_cast<CUDATrackedMemoryResource *>(gpuMemoryAllocator.resource());
    CHECK(mr != nullptr);
    size_t startSize = mr->BytesAllocated();

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

    if (envLight)
        escapedRayQueue = alloc.new_object<EscapedRayQueue>(maxQueueSize, alloc);
    hitAreaLightQueue = alloc.new_object<HitAreaLightQueue>(maxQueueSize, alloc);

    basicEvalMaterialQueue = alloc.new_object<MaterialEvalQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveBasicEvalMaterial[1], haveBasicEvalMaterial.size() - 1));
    universalEvalMaterialQueue = alloc.new_object<MaterialEvalQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveUniversalEvalMaterial[1],
                            haveUniversalEvalMaterial.size() - 1));

    // Always allocate this, even if no media
    mediumTransitionQueue = alloc.new_object<MediumTransitionQueue>(maxQueueSize, alloc);
    if (haveMedia) {
        mediumSampleQueue = alloc.new_object<MediumSampleQueue>(maxQueueSize, alloc);
        mediumScatterQueue = alloc.new_object<MediumScatterQueue>(maxQueueSize, alloc);
    }

    stats = alloc.new_object<Stats>(maxDepth, alloc);

    size_t endSize = mr->BytesAllocated();
    pathIntegratorBytes += endSize - startSize;
}

// GPUPathIntegrator Method Definitions
void GPUPathIntegrator::Render() {
    Vector2i resolution = film.PixelBounds().Diagonal();
    int spp = sampler.SamplesPerPixel();
    // Launch thread to copy image for display server, if enabled
    RGB *displayRGB = nullptr, *displayRGBHost = nullptr;
    std::atomic<bool> exitCopyThread{false};
    std::thread copyThread;

    if (!Options->displayServer.empty()) {
        // Allocate staging memory on the GPU to store the current WIP
        // image.
        CUDA_CHECK(cudaMalloc(&displayRGB, resolution.x * resolution.y * sizeof(RGB)));
        CUDA_CHECK(cudaMemset(displayRGB, 0, resolution.x * resolution.y * sizeof(RGB)));

        // Host-side memory for the WIP Image.  We'll just let this leak so
        // that the lambda passed to DisplayDynamic below doesn't access
        // freed memory after Render() returns...
        displayRGBHost = new RGB[resolution.x * resolution.y];

        copyThread = std::thread([&]() {
#ifdef NVTX
#ifdef PBRT_IS_WINDOWS
            nvtxNameOsThread(GetCurrentThreadId(), "DISPLAY_SERVER_COPY_THREAD");
#else
            nvtxNameOsThread(syscall(SYS_gettid), "DISPLAY_SERVER_COPY_THREAD");
#endif
#endif
            // Copy back to the CPU using a separate stream so that we can
            // periodically but asynchronously pick up the latest results
            // from the GPU.
            cudaStream_t memcpyStream;
            CUDA_CHECK(cudaStreamCreate(&memcpyStream));
#ifdef NVTX
            nvtxNameCuStream(memcpyStream, "DISPLAY_SERVER_COPY_STREAM");
#endif

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
        DisplayDynamic(film.GetFilename(), {resolution.x, resolution.y}, {"R", "G", "B"},
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
    }

    int firstSampleIndex = 0, lastSampleIndex = spp;
    // Update sample index range based on debug start, if provided
    if (!Options->debugStart.empty()) {
        std::vector<int> values = SplitStringToInts(Options->debugStart, ',');
        if (values.size() != 2)
            ErrorExit("Expected two integer values for --debugstart.");

        firstSampleIndex = values[0];
        lastSampleIndex = firstSampleIndex + values[1];
    }

    ProgressReporter progress(lastSampleIndex - firstSampleIndex, "Rendering",
                              Options->quiet, true /* GPU */);
    for (int sampleIndex = firstSampleIndex; sampleIndex < lastSampleIndex;
         ++sampleIndex) {
        // Render image for sample _sampleIndex_
        LOG_VERBOSE("Starting to submit work for sample %d", sampleIndex);
        Bounds2i pixelBounds = film.PixelBounds();
        for (int y0 = pixelBounds.pMin.y; y0 < pixelBounds.pMax.y;
             y0 += scanlinesPerPass) {
            // Generate camera rays for current scanline range
            RayQueue *cameraRayQueue = CurrentRayQueue(0);
            GPUDo(
                "Reset ray queue", PBRT_GPU_LAMBDA() {
                    PBRT_DBG("Starting scanlines at y0 = %d, sample %d / %d\n", y0,
                             sampleIndex, spp);
                    cameraRayQueue->Reset();
                });
            GenerateCameraRays(y0, sampleIndex);
            GPUDo(
                "Update camera ray stats",
                PBRT_GPU_LAMBDA() { stats->cameraRays += cameraRayQueue->Size(); });

            // Trace rays and estimate radiance up to maximum ray depth
            for (int depth = 0; true; ++depth) {
                // Reset ray queues before tracing rays
                RayQueue *nextQueue = NextRayQueue(depth);
                GPUDo(
                    "Reset queues before tracing rays", PBRT_GPU_LAMBDA() {
                        nextQueue->Reset();
                        // Reset queues before tracing next batch of rays
                        mediumTransitionQueue->Reset();
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
                accel->IntersectClosest(maxQueueSize, escapedRayQueue, hitAreaLightQueue,
                                        basicEvalMaterialQueue,
                                        universalEvalMaterialQueue, mediumTransitionQueue,
                                        mediumSampleQueue, CurrentRayQueue(depth));

                if (depth > 0) {
                    // As above, with the indexing...
                    RayQueue *statsQueue = CurrentRayQueue(depth);
                    GPUDo(
                        "Update indirect ray stats", PBRT_GPU_LAMBDA() {
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
                HandleMediumTransitions(depth);
                if (haveSubsurface)
                    SampleSubsurface(depth);
            }

            UpdateFilm();
            // Copy updated film pixels to buffer for display
            if (!Options->displayServer.empty())
                GPUParallelFor(
                    "Update Display RGB Buffer", maxQueueSize,
                    PBRT_GPU_LAMBDA(int pixelIndex) {
                        Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
                        if (!InsideExclusive(pPixel, film.PixelBounds()))
                            return;

                        Point2i p(pPixel - film.PixelBounds().pMin);
                        displayRGB[p.x + p.y * resolution.x] = film.GetPixelRGB(pPixel);
                    });
        }

        progress.Update();
    }
    progress.Done();
    GPUWait();
    // Shut down display server thread, if active
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

void GPUPathIntegrator::HandleEscapedRays(int depth) {
    ForAllQueued(
        "Handle escaped rays", escapedRayQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(const EscapedRayWorkItem er, int index) {
            // Update pixel radiance for escaped ray
            Ray ray(er.rayo, er.rayd);
            SampledSpectrum Le = envLight.Le(ray, er.lambda);
            if (!Le)
                return;

            SampledSpectrum L(0.f);

            PBRT_DBG("L %f %f %f %f beta %f %f %f %f Le %f %f %f %f", L[0], L[1], L[2],
                     L[3], er.beta[0], er.beta[1], er.beta[2], er.beta[3], Le[0], Le[1],
                     Le[2], Le[3]);
            PBRT_DBG("pdf uni %f %f %f %f pdf nee %f %f %f %f", er.uniPathPDF[0],
                     er.uniPathPDF[1], er.uniPathPDF[2], er.uniPathPDF[3],
                     er.lightPathPDF[0], er.lightPathPDF[1], er.lightPathPDF[2],
                     er.lightPathPDF[3]);

            if (depth == 0 || er.specularBounce) {
                L = er.beta * Le / er.uniPathPDF.Average();
            } else {
                LightSampleContext ctx = er.prevIntrCtx;

                Float lightChoicePDF = lightSampler.PDF(ctx, envLight);
                Float lightPDF = lightChoicePDF *
                                 envLight.PDF_Li(ctx, ray.d, LightSamplingMode::WithMIS);

                SampledSpectrum uniPathPDF = er.uniPathPDF;
                SampledSpectrum lightPathPDF = er.lightPathPDF * lightPDF;

                L = er.beta * Le / (uniPathPDF + lightPathPDF).Average();
            }
            L = SafeDiv(L, er.lambda.PDF());

            PBRT_DBG("Added L %f %f %f %f for escaped ray pixel index %d\n", L[0], L[1],
                     L[2], L[3], er.pixelIndex);

            L += pixelSampleState.L[er.pixelIndex];
            pixelSampleState.L[er.pixelIndex] = L;
        });
}

void GPUPathIntegrator::HandleRayFoundEmission(int depth) {
    ForAllQueued(
        "Handle emitters hit by indirect rays", hitAreaLightQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(const HitAreaLightWorkItem he, int index) {
            LightHandle areaLight = he.areaLight;
            SampledSpectrum Le = areaLight.L(he.p, he.n, he.uv, he.wo, he.lambda);
            if (!Le)
                return;

            PBRT_DBG("Got Le %f %f %f %f from hit area light at depth %d\n", Le[0], Le[1],
                     Le[2], Le[3], depth);

            SampledSpectrum L(0.f);

            if (depth == 0 || he.isSpecularBounce) {
                L = he.beta * Le / he.uniPathPDF.Average();
            } else {
                Vector3f wi = -he.wo;

                LightSampleContext ctx = he.prevIntrCtx;

                Float lightChoicePDF = lightSampler.PDF(ctx, areaLight);
                Float lightPDF = lightChoicePDF *
                                 areaLight.PDF_Li(ctx, wi, LightSamplingMode::WithMIS);

                SampledSpectrum uniPathPDF = he.uniPathPDF;
                SampledSpectrum lightPathPDF = he.lightPathPDF * lightPDF;

                L = he.beta * Le / (uniPathPDF + lightPathPDF).Average();
            }
            L = SafeDiv(L, he.lambda.PDF());

            PBRT_DBG("Added L %f %f %f %f for pixel index %d\n", L[0], L[1], L[2], L[3],
                     he.pixelIndex);

            L += pixelSampleState.L[he.pixelIndex];

            pixelSampleState.L[he.pixelIndex] = L;
        });
}

void GPUPathIntegrator::TraceShadowRays(int depth) {
    if (haveMedia)
        accel->IntersectShadowTr(maxQueueSize, shadowRayQueue);
    else
        accel->IntersectShadow(maxQueueSize, shadowRayQueue);

    // Add contribution if light was visible
    ForAllQueued(
        "Incorporate shadow ray contribution", shadowRayQueue, maxQueueSize,
        PBRT_GPU_LAMBDA(const ShadowRayWorkItem sr, int index) {
            if (!sr.Ld)
                return;

            SampledSpectrum Lpixel = pixelSampleState.L[sr.pixelIndex];

            PBRT_DBG("Adding shadow ray Ld %f %f %f %f at pixel index %d \n", sr.Ld[0],
                     sr.Ld[1], sr.Ld[2], sr.Ld[3], sr.pixelIndex);

            pixelSampleState.L[sr.pixelIndex] = Lpixel + sr.Ld;
        });

    GPUDo(
        "Reset shadowRayQueue", PBRT_GPU_LAMBDA() {
            stats->shadowRays[depth] += shadowRayQueue->Size();
            shadowRayQueue->Reset();
        });
}

void GPURender(ParsedScene &scene) {
#ifdef PBRT_IS_WINDOWS
    // NOTE: on Windows, where only basic unified memory is supported, the
    // GPUPathIntegrator itself is *not* allocated using the unified memory
    // allocator so that the CPU can access the values of its members
    // (e.g. maxDepth) concurrently while the GPU is rendering.  In turn,
    // the lambda capture for GPU kernels has to capture *this by value (see
    // the definition of PBRT_GPU_LAMBDA in gpulaunch.h.).
    GPUPathIntegrator *integrator = new GPUPathIntegrator(gpuMemoryAllocator, scene);
#else
    // With more capable unified memory, the GPUPathIntegrator can live in
    // unified memory and some cudaMemAdvise calls, to come shortly, let us
    // have fast read-only access to it on the CPU.
    GPUPathIntegrator *integrator =
        gpuMemoryAllocator.new_object<GPUPathIntegrator>(gpuMemoryAllocator, scene);
#endif

    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));
    int hasConcurrentManagedAccess;
    CUDA_CHECK(cudaDeviceGetAttribute(&hasConcurrentManagedAccess,
                                      cudaDevAttrConcurrentManagedAccess, deviceIndex));

    // Copy all of the scene data structures over to GPU memory.  This
    // ensures that there isn't a big performance hitch for the first batch
    // of rays as that stuff is copied over on demand.
    if (hasConcurrentManagedAccess) {
        // Set things up so that we can still have read from the
        // GPUPathIntegrator struct on the CPU without hurting
        // performance. (This makes it possible to use the values of things
        // like GPUPathIntegrator::haveSubsurface to conditionally launch
        // kernels according to what's in the scene...)
        CUDA_CHECK(cudaMemAdvise(integrator, sizeof(*integrator),
                                 cudaMemAdviseSetReadMostly, /* ignored argument */ 0));
        CUDA_CHECK(cudaMemAdvise(integrator, sizeof(*integrator),
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

    ///////////////////////////////////////////////////////////////////////////
    // Render!
    Timer timer;
    integrator->Render();

    LOG_VERBOSE("Total rendering time: %.3f s", timer.ElapsedSeconds());

    CUDA_CHECK(cudaProfilerStop());

    if (!Options->quiet) {
        ReportKernelStats();

        Printf("GPU Statistics:\n");
        Printf("%s\n", integrator->stats->Print());
    }

    std::vector<GPULogItem> logs = ReadGPULogs();
    for (const auto &item : logs)
        Log(item.level, item.file, item.line, item.message);

    ImageMetadata metadata;
    metadata.samplesPerPixel = integrator->sampler.SamplesPerPixel();
    integrator->camera.InitMetadata(&metadata);
    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    metadata.samplesPerPixel = integrator->sampler.SamplesPerPixel();
    integrator->film.WriteImage(metadata);
}

GPUPathIntegrator::Stats::Stats(int maxDepth, Allocator alloc)
    : indirectRays(maxDepth + 1, alloc), shadowRays(maxDepth, alloc) {}

std::string GPUPathIntegrator::Stats::Print() const {
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
