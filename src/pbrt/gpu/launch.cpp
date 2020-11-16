// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/gpu/launch.h>

#include <pbrt/util/print.h>

#include <algorithm>
#include <vector>

namespace pbrt {

struct KernelStats {
    KernelStats(const char *description)
        : description(description) { }

    std::string description;
    int numLaunches = 0;
    float sumMS = 0, minMS = 0, maxMS = 0;
};

struct ProfilerEvent {
    ProfilerEvent() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    void Sync() {
        CHECK(active);
        CUDA_CHECK(cudaEventSynchronize(start));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        ++stats->numLaunches;
        if (stats->numLaunches == 1)
            stats->sumMS = stats->minMS = stats->maxMS = ms;
        else {
            stats->sumMS += ms;
            stats->minMS = std::min(stats->minMS, ms);
            stats->maxMS = std::max(stats->maxMS, ms);
        }

        active = false;
    }

    cudaEvent_t start, stop;
    bool active = false;
    KernelStats *stats = nullptr;
};

// Store pointers so that reallocs don't mess up held KernelStats pointers
// in ProfilerEvent..
static std::vector<KernelStats *> kernelStats;

// Ring buffer
static std::vector<ProfilerEvent> eventPool;
static size_t eventPoolOffset = 0;

std::pair<cudaEvent_t, cudaEvent_t> GetProfilerEvents(const char *description) {
    if (eventPool.empty())
        eventPool.resize(1024);  // how many? This is probably more than we need...

    if (eventPoolOffset == eventPool.size())
        eventPoolOffset = 0;

    ProfilerEvent &pe = eventPool[eventPoolOffset++];
    if (pe.active)
        pe.Sync();

    pe.active = true;
    pe.stats = nullptr;

    for (size_t i = 0; i < kernelStats.size(); ++i) {
        if (kernelStats[i]->description == description) {
            pe.stats = kernelStats[i];
            break;
        }
    }
    if (!pe.stats) {
        kernelStats.push_back(new KernelStats(description));
        pe.stats = kernelStats.back();
    }

    return {pe.start, pe.stop};
}

void ReportKernelStats() {
    CUDA_CHECK(cudaDeviceSynchronize());

    // Drain active profiler events
    for (size_t i = 0; i < eventPool.size(); ++i)
        if (eventPool[i].active)
            eventPool[i].Sync();

    // Compute total milliseconds over all kernels and launches
    float totalMS = 0;
    for (size_t i = 0; i < kernelStats.size(); ++i)
        totalMS += kernelStats[i]->sumMS;

    printf("GPU Kernel Profile:\n");
    int otherLaunches = 0;
    float otherMS = 0;
    const float otherCutoff = 0.001f * totalMS;
    for (size_t i = 0; i < kernelStats.size(); ++i) {
        KernelStats *stats = kernelStats[i];
        if (stats->sumMS > otherCutoff)
            Printf("  %-49s %5d launches %9.2f ms / %5.1f%s (avg %6.3f, min "
                   "%6.3f, max %7.3f)\n",
                   stats->description, stats->numLaunches, stats->sumMS,
                   100.f * stats->sumMS / totalMS, "%",
                   stats->sumMS / stats->numLaunches, stats->minMS,
                   stats->maxMS);
        else {
            otherMS += stats->sumMS;
            otherLaunches += stats->numLaunches;
        }
    }
    Printf("  %-49s %5d launches %9.2f ms / %5.1f%s (avg %6.3f)\n", "Other",
           otherLaunches, otherMS, 100.f * otherMS / totalMS, "%",
           otherMS / otherLaunches);
    Printf("\nTotal GPU time: %9.2f ms\n", totalMS);
    Printf("\n");
}

void GPUWait() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace pbrt
