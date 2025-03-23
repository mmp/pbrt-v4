// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/wavefront.h>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/memory.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/wavefront/integrator.h>

namespace pbrt {

void RenderWavefront(BasicScene &scene) {
    WavefrontPathIntegrator *integrator = nullptr;

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
#ifdef PBRT_IS_WINDOWS
        // NOTE: on Windows, where only basic unified memory is supported, the
        // WavefrontPathIntegrator itself is *not* allocated using the unified
        // memory allocator so that the CPU can access the values of its
        // members (e.g. maxDepth) concurrently while the GPU is rendering.  In
        // turn, the lambda capture for GPU kernels has to capture *this by
        // value (see the definition of PBRT_CPU_GPU_LAMBDA in pbrt/pbrt.h.).
        integrator =
            new WavefrontPathIntegrator(&CUDATrackedMemoryResource::singleton, scene);
#else
        // With more capable unified memory, the WavefrontPathIntegrator can live in
        // unified memory.  Some cudaMemAdvise calls, to come shortly, let us
        // have fast read-only access to it on the CPU.
        Allocator alloc(&CUDATrackedMemoryResource::singleton);
        integrator = alloc.new_object<WavefrontPathIntegrator>(
            &CUDATrackedMemoryResource::singleton, scene);
#endif
    } else
#endif  // PBRT_BUILD_GPU_RENDERER
        integrator =
            new WavefrontPathIntegrator(pstd::pmr::get_default_resource(), scene);

    ///////////////////////////////////////////////////////////////////////////
    // Render!
    Float seconds = integrator->Render();

    LOG_VERBOSE("Total rendering time: %.3f s", seconds);

    if (Options->printStatistics) {
#ifdef PBRT_BUILD_GPU_RENDERER
        if (Options->useGPU)
            ReportKernelStats();
#endif  // PBRT_BUILD_GPU_RENDERER

        Printf("Wavefront integrator statistics:\n");
        Printf("%s\n", integrator->stats->Print());
    }

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        std::vector<GPULogItem> logs = ReadGPULogs();
        for (const auto &item : logs)
            Log(item.level, item.file, item.line, item.message);
    }
#endif  // PBRT_BUILD_GPU_RENDERER

    ImageMetadata metadata;
    integrator->camera.InitMetadata(&metadata);
    metadata.renderTimeSeconds = seconds;
    metadata.samplesPerPixel = integrator->sampler.SamplesPerPixel();
    // multithreading somehow hangs in HIP
#ifdef __HIPCC__
    if (Options->useGPU) DisableThreadPool();
#endif
    integrator->film.WriteImage(metadata);
#ifdef __HIPCC__
    if (Options->useGPU) ReenableThreadPool();
#endif
}

}  // namespace pbrt
