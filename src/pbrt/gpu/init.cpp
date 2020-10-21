// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/init.h>

#include <pbrt/options.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/print.h>

#include <cuda.h>

#ifdef NVTX
#include <nvtx3/nvToolsExtCuda.h>
#endif

namespace pbrt {

void GPUInit() {
    cudaFree(nullptr);

    int driverVersion;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    int runtimeVersion;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    auto versionToString = [](int version) {
        int major = version / 1000;
        int minor = (version - major * 1000) / 10;
        return StringPrintf("%d.%d", major, minor);
    };
    LOG_VERBOSE("GPU CUDA driver %s, CUDA runtime %s", versionToString(driverVersion),
                versionToString(runtimeVersion));

    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    cudaDeviceProp firstDeviceProperties;
    std::string devices;
    for (int i = 0; i < nDevices; ++i) {
        cudaDeviceProp deviceProperties;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, i));
        if (i == 0)
            firstDeviceProperties = deviceProperties;
        CHECK(deviceProperties.canMapHostMemory);

        std::string deviceString = StringPrintf(
            "CUDA device %d (%s) with %f MiB, %d SMs running at %f MHz "
            "with shader model %d.%d",
            i, deviceProperties.name, deviceProperties.totalGlobalMem / (1024. * 1024.),
            deviceProperties.multiProcessorCount, deviceProperties.clockRate / 1000.,
            deviceProperties.major, deviceProperties.minor);
        LOG_VERBOSE("%s", deviceString);
        devices += deviceString + "\n";

#ifdef PBRT_IS_WINDOWS
        if (deviceProperties.major != firstDeviceProperties.major)
            ErrorExit("Found multiple GPUs with different shader models.\n"
                      "On Windows, this unfortunately causes a significant slowdown with pbrt.\n"
                      "Please select a single GPU and use the --gpu-device command line option to specify it.\n"
                      "Found devices:\n%s", devices);
#endif
    }

    int device = Options->gpuDevice ? *Options->gpuDevice : 0;
    LOG_VERBOSE("Selecting GPU device %d", device);
#ifdef NVTX
    nvtxNameCuDevice(device, "PBRT_GPU");
#endif
    CUDA_CHECK(cudaSetDevice(device));

    int hasUnifiedAddressing;
    CUDA_CHECK(cudaDeviceGetAttribute(&hasUnifiedAddressing, cudaDevAttrUnifiedAddressing,
                                      device));
    if (!hasUnifiedAddressing)
        LOG_FATAL("The selected GPU device (%d) does not support unified addressing.",
                  device);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
    size_t stackSize;
    CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
    LOG_VERBOSE("Reset stack size to %d", stackSize);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32 * 1024 * 1024));

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
}

void GPUThreadInit() {
    if (!Options->useGPU)
        return;
    int device = Options->gpuDevice ? *Options->gpuDevice : 0;
    LOG_VERBOSE("Selecting GPU device %d", device);
    CUDA_CHECK(cudaSetDevice(device));
}

}  // namespace pbrt
