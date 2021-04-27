// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/gpu/init.h>
#include <pbrt/options.h>
#include <pbrt/shapes.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/display.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

#include <stdlib.h>

namespace pbrt {

// API Function Definitions
void InitPBRT(const PBRTOptions &opt) {
    Options = new PBRTOptions(opt);
    // API Initialization

#ifdef PBRT_IS_WINDOWS
    if (Options->useGPU && Options->gpuDevice &&
        getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
        // Limit CUDA to considering only a single GPU on Windows.  pbrt
        // only uses a single GPU anyway, and if there are multiple GPUs
        // plugged in with different architectures, pbrt's use of unified
        // memory causes a performance hit.  We set this early, before CUDA
        // gets going...
        std::string env = StringPrintf("CUDA_VISIBLE_DEVICES=%d", *Options->gpuDevice);
        _putenv(env.c_str());
        // Now CUDA should only see a single device, so tell it that zero
        // is the one to use.
        *Options->gpuDevice = 0;
    }
#endif  // PBRT_IS_WINDOWS

    if (Options->quiet)
        SuppressErrorMessages();

    InitLogging(opt.logLevel, opt.logFile, Options->useGPU);

    // General \pbrt Initialization
    int nThreads = Options->nThreads != 0 ? Options->nThreads : AvailableCores();
    ParallelInit(nThreads);  // Threads must be launched before the
                             // profiler is initialized.

    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        GPUInit();

        CUDA_CHECK(cudaMemcpyToSymbol(OptionsGPU, Options, sizeof(OptionsGPU)));

        ColorEncoding::Init(gpuMemoryAllocator);
        Spectra::Init(gpuMemoryAllocator);
        RGBToSpectrumTable::Init(gpuMemoryAllocator);

        RGBColorSpace::Init(gpuMemoryAllocator);
        InitBufferCaches(gpuMemoryAllocator);
        Triangle::Init(gpuMemoryAllocator);
        BilinearPatch::Init(gpuMemoryAllocator);
#else
        LOG_FATAL("Options::useGPU set with non-GPU build");
#endif
    } else {
        ColorEncoding::Init(Allocator{});
        // Before RGBColorSpace::Init!
        Spectra::Init(Allocator{});
        RGBToSpectrumTable::Init(Allocator{});

        RGBColorSpace::Init(Allocator{});
        InitBufferCaches({});
        Triangle::Init({});
        BilinearPatch::Init({});
    }

    if (!Options->displayServer.empty())
        ConnectToDisplayServer(Options->displayServer);
}

void CleanupPBRT() {
    ForEachThread(ReportThreadStats);

    if (Options->recordPixelStatistics)
        StatsWritePixelImages();

    if (Options->printStatistics) {
        PrintStats(stdout);
        ClearStats();
    }
    if (PrintCheckRare(stdout))
        ErrorExit("CHECK_RARE failures");

    if (!Options->displayServer.empty())
        DisconnectFromDisplayServer();

    // API Cleanup
    ParallelCleanup();

    // CO    delete Options;
    Options = nullptr;
}

}  // namespace pbrt
