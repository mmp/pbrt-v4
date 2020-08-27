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
#include <pbrt/util/stats.h>
#include <pbrt/util/string.h>
#include <pbrt/util/taggedptr.h>

#include <cstring>
#include <iostream>
#include <map>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda/std/atomic>

#ifdef NVTX
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

    film = FilmHandle::Create(scene.film.name, scene.film.parameters, &scene.film.loc,
                              filter, alloc);
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
                Warning(&shape.loc,
                        "Animated lights aren't supported. Using the start transform.");
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

    // Integrator parameters
    regularize = scene.integrator.parameters.GetOneBool("regularize", false);
    maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    ///////////////////////////////////////////////////////////////////////////
    // Allocate storage for all of the queues/buffers...

    CUDATrackedMemoryResource *mr =
        dynamic_cast<CUDATrackedMemoryResource *>(gpuMemoryAllocator.resource());
    CHECK(mr != nullptr);
    size_t startSize = mr->BytesAllocated();

    // Compute number of scanlines to render per pass.
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

void GPUPathIntegrator::TraceShadowRays(int depth) {
    std::pair<cudaEvent_t, cudaEvent_t> events;
    if (haveMedia)
        accel->IntersectShadowTr(maxQueueSize, shadowRayQueue);
    else
        accel->IntersectShadow(maxQueueSize, shadowRayQueue);

    // Add contribution if light was visible
    ForAllQueued("Incorporate shadow ray contribution", shadowRayQueue, maxQueueSize,
                 [=] PBRT_GPU(const ShadowRayWorkItem sr, int index) {
                     if (!sr.Ld)
                         return;

                     SampledSpectrum Lpixel = pixelSampleState.L[sr.pixelIndex];

                     PBRT_DBG("Adding shadow ray Ld %f %f %f %f at pixel index %d \n",
                         sr.Ld[0], sr.Ld[1], sr.Ld[2], sr.Ld[3], sr.pixelIndex);

                     pixelSampleState.L[sr.pixelIndex] = Lpixel + sr.Ld;
                 });

    GPUDo("Reset shadowRayQueue", [=] PBRT_GPU() {
        stats->shadowRays[depth] += shadowRayQueue->Size();
        shadowRayQueue->Reset();
    });
}

void GPUPathIntegrator::Render(ImageMetadata *metadata) {
    Vector2i resolution = film.PixelBounds().Diagonal();
    int spp = sampler.SamplesPerPixel();

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
            nvtxNameOsThread(syscall(SYS_gettid), "DISPLAY_SERVER_COPY_THREAD");
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
    if (!Options->debugStart.empty()) {
        pstd::optional<std::vector<int>> values =
            SplitStringToInts(Options->debugStart, ',');
        if (!values || values->size() != 2)
            ErrorExit("Expected two integer values for --debugstart.");

        firstSampleIndex = (*values)[0];
        lastSampleIndex = firstSampleIndex + (*values)[1];
    }

    ProgressReporter progress(lastSampleIndex - firstSampleIndex, "Rendering",
                              Options->quiet, true /* GPU */);

    for (int sampleIndex = firstSampleIndex; sampleIndex < lastSampleIndex; ++sampleIndex) {
        LOG_VERBOSE("Starting to submit work for sample %d", sampleIndex);

        for (int y0 = 0; y0 < resolution.y; y0 += scanlinesPerPass) {
            GPUDo("Reset ray queue", [=] PBRT_GPU() {
                PBRT_DBG("Starting scanlines at y0 = %d, sample %d / %d\n", y0, sampleIndex,
                    spp);
                rayQueues[0]->Reset();
            });

            GenerateCameraRays(y0, sampleIndex);

            GPUDo("Update camera ray stats",
                  [=] PBRT_GPU() { stats->cameraRays += rayQueues[0]->Size(); });

            for (int depth = 0; true; ++depth) {
                GenerateRaySamples(depth, sampleIndex);

                GPUDo("Reset queues before tracing rays", [=] PBRT_GPU() {
                    hitAreaLightQueue->Reset();
                    if (escapedRayQueue)
                        escapedRayQueue->Reset();

                    basicEvalMaterialQueue->Reset();
                    universalEvalMaterialQueue->Reset();

                    if (bssrdfEvalQueue)
                        bssrdfEvalQueue->Reset();
                    if (subsurfaceScatterQueue)
                        subsurfaceScatterQueue->Reset();

                    mediumTransitionQueue->Reset();
                    if (mediumSampleQueue)
                        mediumSampleQueue->Reset();
                    if (mediumScatterQueue)
                        mediumScatterQueue->Reset();

                    rayQueues[(depth + 1) & 1]->Reset();
                });

                accel->IntersectClosest(
                    maxQueueSize, escapedRayQueue, hitAreaLightQueue,
                    basicEvalMaterialQueue, universalEvalMaterialQueue,
                    mediumTransitionQueue, mediumSampleQueue, rayQueues[depth & 1]);

                if (depth > 0)
                    GPUDo("Update indirect ray stats", [=] PBRT_GPU() {
                        stats->indirectRays[depth] += rayQueues[depth & 1]->Size();
                    });

                if (haveMedia)
                    SampleMediumInteraction(depth);

                if (escapedRayQueue)
                    HandleEscapedRays(depth);

                HandleRayFoundEmission(depth);

                if (depth == maxDepth)
                    break;

                EvaluateMaterialsAndBSDFs(depth);

                // Do immediately so that we have space for shadow rays for
                // subsurface..
                TraceShadowRays(depth);

                HandleMediumTransitions(depth);

                if (haveSubsurface)
                    SampleSubsurface(depth);
            }

            UpdateFilm();

            if (!Options->displayServer.empty())
                GPUParallelFor("Update Display RGB Buffer", maxQueueSize,
                               [=] PBRT_GPU(int pixelIndex) {
                                   Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
                                   if (!InsideExclusive(pPixel, film.PixelBounds()))
                                       return;

                                   Point2i p(pPixel - film.PixelBounds().pMin);
                                   displayRGB[p.x + p.y * resolution.x] =
                                       film.GetPixelRGB(pPixel);
                               });
        }

        progress.Update();
    }
    progress.Done();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Wait until rendering is all done before we start to shut down the
    // display stuff..
    if (!Options->displayServer.empty()) {
        exitCopyThread = true;
        copyThread.join();
    }

    metadata->samplesPerPixel = sampler.SamplesPerPixel();
    camera.InitMetadata(metadata);
}

void GPUPathIntegrator::HandleEscapedRays(int depth) {
    ForAllQueued("Handle escaped rays", escapedRayQueue, maxQueueSize,
                 [=] PBRT_GPU(const EscapedRayWorkItem er, int index) {
                     Ray ray(er.rayo, er.rayd);
                     SampledSpectrum Le = envLight.Le(ray, er.lambda);
                     if (!Le)
                         return;

                     SampledSpectrum L = pixelSampleState.L[er.pixelIndex];

                     PBRT_DBG("L %f %f %f %f beta %f %f %f %f Le %f %f %f %f",
                         L[0], L[1], L[2], L[3], er.beta[0], er.beta[1], er.beta[2],
                         er.beta[3], Le[0], Le[1], Le[2], Le[3]);
                     PBRT_DBG("pdf uni %f %f %f %f pdf nee %f %f %f %f",
                         er.pdfUni[0], er.pdfUni[1], er.pdfUni[2], er.pdfUni[3],
                         er.pdfNEE[0], er.pdfNEE[1], er.pdfNEE[2], er.pdfNEE[3]);

                     if (depth == 0 || er.specularBounce) {
                         L += er.beta * Le / er.pdfUni.Average();
                     } else {
                         Float time = 0;  // FIXME
                         LightSampleContext ctx(er.piPrev, er.nPrev, er.nsPrev);

                         Float lightChoicePDF = lightSampler.PDF(ctx, envLight);
                         Float lightPDF =
                             lightChoicePDF *
                             envLight.PDF_Li(ctx, ray.d, LightSamplingMode::WithMIS);

                         SampledSpectrum pdfUni = er.pdfUni;
                         SampledSpectrum pdfNEE = er.pdfNEE * lightPDF;

                         L += er.beta * Le / (pdfUni + pdfNEE).Average();
                     }

                     PBRT_DBG("Added L %f %f %f %f for escaped ray pixel index %d\n", L[0],
                         L[1], L[2], L[3], er.pixelIndex);
                     pixelSampleState.L[er.pixelIndex] = L;
                 });
}

void GPUPathIntegrator::HandleRayFoundEmission(int depth) {
    ForAllQueued(
        "Handle emitters hit by indirect rays", hitAreaLightQueue, maxQueueSize,
        [=] PBRT_GPU(const HitAreaLightWorkItem he, int index) {
            LightHandle areaLight = he.areaLight;
            SampledSpectrum Le = areaLight.L(he.p, he.n, he.uv, he.wo, he.lambda);
            if (!Le)
                return;

            PBRT_DBG("Got Le %f %f %f %f from hit area light at depth %d\n", Le[0], Le[1],
                Le[2], Le[3], depth);

            SampledSpectrum L = pixelSampleState.L[he.pixelIndex];

            if (depth == 0 || he.isSpecularBounce) {
                L += he.beta * Le / he.pdfUni.Average();
            } else {
                Vector3f wi = he.rayd;

                LightSampleContext ctx(he.piPrev, he.nPrev, he.nsPrev);

                Float lightChoicePDF = lightSampler.PDF(ctx, areaLight);
                Float lightPDF = lightChoicePDF *
                                 areaLight.PDF_Li(ctx, wi, LightSamplingMode::WithMIS);

                SampledSpectrum pdfUni = he.pdfUni;
                SampledSpectrum pdfNEE = he.pdfNEE * lightPDF;

                L += he.beta * Le / (pdfUni + pdfNEE).Average();
            }

            PBRT_DBG("Added L %f %f %f %f for pixel index %d\n", L[0], L[1], L[2], L[3],
                he.pixelIndex);
            pixelSampleState.L[he.pixelIndex] = L;
        });
}

void GPURender(ParsedScene &scene) {
    GPUPathIntegrator *integrator =
        gpuMemoryAllocator.new_object<GPUPathIntegrator>(gpuMemoryAllocator, scene);

    // Set things up so that we can still have read from the
    // GPUPathIntegrator struct on the CPU without hurting
    // performance. (This makes it possible to use the values of things
    // like GPUPathIntegrator::haveSubsurface to conditionally launch
    // kernels according to what's in the scene...)
    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));
    CUDA_CHECK(
        cudaMemAdvise(integrator, sizeof(*integrator), cudaMemAdviseSetReadMostly, 0));
    CUDA_CHECK(cudaMemAdvise(integrator, sizeof(*integrator),
                             cudaMemAdviseSetPreferredLocation, deviceIndex));

    // Copy all of the scene data structures over to GPU memory.  This
    // ensures that there isn't a big performance hitch for the first batch
    // of rays as that stuff is copied over on demand.
    CUDATrackedMemoryResource *mr =
        dynamic_cast<CUDATrackedMemoryResource *>(gpuMemoryAllocator.resource());
    CHECK(mr != nullptr);
    mr->PrefetchToGPU();

    ///////////////////////////////////////////////////////////////////////////
    // Render!
    Timer timer;
    ImageMetadata metadata;
    integrator->Render(&metadata);

    LOG_VERBOSE("Total rendering time: %.3f s", timer.ElapsedSeconds());

    CUDA_CHECK(cudaProfilerStop());

    if (!Options->quiet) {
        ReportKernelStats();

        Printf("GPU Statistics:\n");
        Printf("%s\n", integrator->stats->Print());
    }

    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    metadata.samplesPerPixel = integrator->sampler.SamplesPerPixel();

    std::vector<GPULogItem> logs = ReadGPULogs();
    for (const auto &item : logs)
        Log(item.level, item.file, item.line, item.message);

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
