// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/progressreporter.h>

#include <pbrt/util/check.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>

#include <cerrno>
#include <cstdio>
#include <memory>

#ifdef PBRT_IS_WINDOWS
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif  // !PBRT_IS_WINDOWS

#ifdef NVTX
#ifndef PBRT_IS_WINDOWS
#include <sys/syscall.h>
#endif
#include "nvtx3/nvToolsExtCuda.h"
#endif

namespace pbrt {

static int TerminalWidth();

std::string Timer::ToString() const {
    return StringPrintf(
        "[ Timer start(ns): %d ]",
        std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch())
            .count());
}

// ProgressReporter Method Definitions
ProgressReporter::ProgressReporter(int64_t totalWork, const std::string &title,
                                   bool quiet, bool gpu)
    : totalWork(std::max<int64_t>(1, totalWork)), title(title), quiet(quiet) {
    workDone = 0;
    exitThread = false;

#ifdef PBRT_BUILD_GPU_RENDERER
    if (gpu) {
        gpuEventsLaunchedOffset = 0;
        gpuEventsFinishedOffset = 0;

        gpuEvents.resize(totalWork);
        for (cudaEvent_t &event : gpuEvents)
            CUDA_CHECK(cudaEventCreate(&event));
    }
#else
    CHECK(gpu == false);
#endif

    // Launch thread to periodically update progress bar
    if (!quiet)
        launchThread();
}

void ProgressReporter::launchThread() {
    Barrier *barrier = new Barrier(2);
    updateThread = std::thread([this, barrier]() {
#if NVTX
#ifdef PBRT_IS_WINDOWS
        nvtxNameOsThread(GetCurrentThreadId(), "PBRT_PROGRESS_BAR");
#else
        nvtxNameOsThread(syscall(SYS_gettid), "PBRT_PROGRESS_BAR");
#endif
#endif
        if (barrier->Block())
            delete barrier;
        printBar();
    });
    // Wait for the thread to get past the ProfilerWorkerThreadInit()
    // call.
    if (barrier->Block())
        delete barrier;
}

ProgressReporter::~ProgressReporter() {
    Done();
}

void ProgressReporter::printBar() {
    int barLength = TerminalWidth() - 28;
    int totalPlusses = std::max<int>(2, barLength - title.size());
    int plussesPrinted = 0;

    // Initialize progress string
    const int bufLen = title.size() + totalPlusses + 64;
    std::unique_ptr<char[]> buf = std::make_unique<char[]>(bufLen);
    snprintf(buf.get(), bufLen, "\r%s: [", title.c_str());
    char *curSpace = buf.get() + strlen(buf.get());
    char *s = curSpace;
    for (int i = 0; i < totalPlusses; ++i)
        *s++ = ' ';
    *s++ = ']';
    *s++ = ' ';
    *s++ = '\0';
    fputs(buf.get(), stdout);
    fflush(stdout);

#ifdef PBRT_BUILD_GPU_RENDERER
    std::chrono::milliseconds sleepDuration(gpuEvents.size() ? 50 : 250);
#else
    std::chrono::milliseconds sleepDuration(250);
#endif

    int iterCount = 0;
    bool reallyExit = false;  // make sure we do one more go-round to get the final report
    while (!reallyExit) {
        if (exitThread)
            reallyExit = true;
        else
            std::this_thread::sleep_for(sleepDuration);

        // Periodically increase sleepDuration to reduce overhead of
        // updates.
        ++iterCount;
        if (iterCount == 10)
            // Up to 0.5s after ~2.5s elapsed
            sleepDuration *= 2;
        else if (iterCount == 70)
            // Up to 1s after an additional ~30s have elapsed.
            sleepDuration *= 2;
        else if (iterCount == 520)
            // After 15m, jump up to 5s intervals
            sleepDuration *= 5;

#ifdef PBRT_BUILD_GPU_RENDERER
        if (gpuEvents.size()) {
            while (gpuEventsFinishedOffset < gpuEventsLaunchedOffset) {
                cudaError_t err = cudaEventQuery(gpuEvents[gpuEventsFinishedOffset]);
                if (err == cudaSuccess)
                    ++gpuEventsFinishedOffset;
                else if (err == cudaErrorNotReady)
                    break;
                else
                    LOG_FATAL("CUDA error: %s", cudaGetErrorString(err));
            }
            workDone = gpuEventsFinishedOffset;
        }
#endif

        Float percentDone = Float(workDone) / Float(totalWork);
        int plussesNeeded = std::round(totalPlusses * percentDone);
        while (plussesPrinted < plussesNeeded) {
            *curSpace++ = '+';
            ++plussesPrinted;
        }
        fputs(buf.get(), stdout);

        // Update elapsed time and estimated time to completion
        Float elapsed = ElapsedSeconds();
        Float estRemaining = elapsed / percentDone - elapsed;
        if (percentDone == 1.f)
            printf(" (%.1fs)       ", elapsed);
        else if (!std::isinf(estRemaining))
            printf(" (%.1fs|%.1fs)  ", elapsed, std::max<Float>(0, estRemaining));
        else
            printf(" (%.1fs|?s)  ", elapsed);
        fflush(stdout);
    }
}

void ProgressReporter::Done() {
    if (!quiet) {
#ifdef PBRT_BUILD_GPU_RENDERER
        if (gpuEvents.size()) {
            while (gpuEventsFinishedOffset < gpuEventsLaunchedOffset) {
                cudaError_t err =
                    cudaEventSynchronize(gpuEvents[gpuEventsFinishedOffset]);
                if (err != cudaSuccess)
                    LOG_FATAL("CUDA error: %s", cudaGetErrorString(err));
            }
            workDone = gpuEvents.size();
        }
#endif

        // Only let one thread shut things down.
        bool fa = false;
        if (exitThread.compare_exchange_strong(fa, true)) {
            workDone = totalWork;
            exitThread = true;
            if (updateThread.joinable())
                updateThread.join();
            printf("\n");
        }
    }
}

std::string ProgressReporter::ToString() const {
    return StringPrintf("[ ProgressReporter totalWork: %d title: %s "
                        "timer: %s workDone: %d exitThread: %s",
                        totalWork, title, timer, workDone, exitThread);
}

static int TerminalWidth() {
#ifdef PBRT_IS_WINDOWS
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || !h) {
        fprintf(stderr, "GetStdHandle() call failed");
        return 80;
    }
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo = {0};
    GetConsoleScreenBufferInfo(h, &bufferInfo);
    return bufferInfo.dwSize.X;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) < 0) {
        // ENOTTY is fine and expected, e.g. if output is being piped to a file.
        if (errno != ENOTTY) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr, "Error in ioctl() in TerminalWidth(): %d\n", errno);
            }
        }
        return 80;
    }
    return w.ws_col;
#endif  // PBRT_IS_WINDOWS
}

}  // namespace pbrt
