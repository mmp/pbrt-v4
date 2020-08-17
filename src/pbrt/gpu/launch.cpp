// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/launch.h>

#include <pbrt/util/print.h>

#include <map>
#include <vector>

namespace pbrt {

static std::vector<std::type_index> gpuKernelLaunchOrder;
static std::map<std::type_index, GPUKernelStats> gpuKernels;

GPUKernelStats &GetGPUKernelStats(std::type_index typeIndex, const char *description) {
    auto iter = gpuKernels.find(typeIndex);
    if (iter != gpuKernels.end()) {
        CHECK_EQ(iter->second.description, std::string(description));
        return iter->second;
    }

    gpuKernelLaunchOrder.push_back(typeIndex);
    gpuKernels[typeIndex] = GPUKernelStats(description);
    return gpuKernels.find(typeIndex)->second;
}

void ReportKernelStats() {
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute total milliseconds over all kernels and launches
    float totalms = 0.f;
    for (const auto kernelTypeId : gpuKernelLaunchOrder) {
        const GPUKernelStats &stats = gpuKernels[kernelTypeId];
        for (const auto &launch : stats.launchEvents) {
            cudaEventSynchronize(launch.second);
            float ms = 0;
            cudaEventElapsedTime(&ms, launch.first, launch.second);
            totalms += ms;
        }
    }

    printf("GPU Kernel Profile:\n");
    int otherLaunches = 0;
    float otherms = 0;
    const float otherCutoff = 0.001f * totalms;
    for (const auto kernelTypeId : gpuKernelLaunchOrder) {
        float summs = 0.f, minms = 1e30, maxms = 0;
        const GPUKernelStats &stats = gpuKernels[kernelTypeId];
        for (const auto &launch : stats.launchEvents) {
            float ms = 0;
            cudaEventElapsedTime(&ms, launch.first, launch.second);
            summs += ms;
            minms = std::min(minms, ms);
            maxms = std::max(maxms, ms);
        }

        if (summs > otherCutoff)
            Printf("  %-49s %5d launches %9.2f ms / %5.1f%s (avg %6.3f, min "
                   "%6.3f, max %7.3f)\n",
                   stats.description, stats.launchEvents.size(), summs,
                   100.f * summs / totalms, "%", summs / stats.launchEvents.size(), minms,
                   maxms);
        else {
            otherms += summs;
            otherLaunches += stats.launchEvents.size();
        }
    }
    Printf("  %-49s %5d launches %9.2f ms / %5.1f%s (avg %6.3f)\n", "Other",
           otherLaunches, otherms, 100.f * otherms / totalms, "%",
           otherms / otherLaunches);
    Printf("\nTotal GPU time: %9.2f ms\n", totalms);
    Printf("\n");
}

}  // namespace pbrt
