// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_PROGRESSREPORTER_H
#define PBRT_UTIL_PROGRESSREPORTER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#ifdef PBRT_BUILD_GPU_RENDERER
#if defined(__HIPCC__)
#include <pbrt/util/hip_aliases.h>
#else
#include <cuda_runtime.h>
#endif
#include <vector>
#endif

namespace pbrt {

// Timer Definition
class Timer {
  public:
    Timer() { start = clock::now(); }
    double ElapsedSeconds() const {
        clock::time_point now = clock::now();
        int64_t elapseduS =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        return elapseduS / 1000000.;
    }

    std::string ToString() const;

  private:
    using clock = std::chrono::steady_clock;
    clock::time_point start;
};

// ProgressReporter Definition
class ProgressReporter {
  public:
    // ProgressReporter Public Methods
    ProgressReporter() : quiet(true) {}
    ProgressReporter(int64_t totalWork, std::string title, bool quiet, bool gpu = false);

    ~ProgressReporter();

    void Update(int64_t num = 1);
    void Done();
    double ElapsedSeconds() const;

    std::string ToString() const;

  private:
    // ProgressReporter Private Methods
    void printBar();

    // ProgressReporter Private Members
    int64_t totalWork;
    std::string title;
    bool quiet;
    Timer timer;
    std::atomic<int64_t> workDone;
    std::atomic<bool> exitThread;
    std::thread updateThread;
    pstd::optional<float> finishTime;

#ifdef PBRT_BUILD_GPU_RENDERER
    std::vector<cudaEvent_t> gpuEvents;
    std::atomic<size_t> gpuEventsLaunchedOffset;
    int gpuEventsFinishedOffset;
#endif
};
// ProgressReporter Inline Method Definitions
inline double ProgressReporter::ElapsedSeconds() const {
    return finishTime ? *finishTime : timer.ElapsedSeconds();
}

inline void ProgressReporter::Update(int64_t num) {
#ifdef PBRT_BUILD_GPU_RENDERER
    if (gpuEvents.size() > 0) {
        if (gpuEventsLaunchedOffset + num <= gpuEvents.size()) {
            while (num-- > 0) {
                CHECK_EQ(cudaEventRecord(gpuEvents[gpuEventsLaunchedOffset]),
                         cudaSuccess);
                ++gpuEventsLaunchedOffset;
            }
        }
        return;
    }
#endif
    if (num == 0 || quiet)
        return;
    workDone += num;
}

}  // namespace pbrt

#endif  // PBRT_UTIL_PROGRESSREPORTER_H
