// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/log.h>

#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/parallel.h>

#include <stdio.h>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#ifdef PBRT_IS_OSX
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifdef PBRT_IS_LINUX
#include <sys/types.h>
#include <unistd.h>
#endif
#ifdef PBRT_IS_WINDOWS
#include <windows.h>
#endif

namespace pbrt {

namespace {

std::string TimeNow() {
    std::time_t t = std::time(NULL);
    std::tm tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d.%H%M%S");
    return ss.str();
}

#define LOG_BASE_FMT "%d.%03d %s"
#ifdef PBRT_IS_WINDOWS
#define LOG_BASE_ARGS GetCurrentThreadId(), ThreadIndex, TimeNow().c_str()
#else
#define LOG_BASE_ARGS getpid(), ThreadIndex, TimeNow().c_str()
#endif

}  // namespace

LogLevel LOGGING_LogLevel;

#ifdef PBRT_BUILD_GPU_RENDERER
__constant__ LogLevel LOGGING_LogLevelGPU;

#define MAX_LOG_ITEMS 1024
PBRT_GPU GPULogItem rawLogItems[MAX_LOG_ITEMS];
PBRT_GPU int nRawLogItems;
#endif  // PBRT_BUILD_GPU_RENDERER

void InitLogging(LogLevel level, bool useGPU) {
    LOGGING_LogLevel = level;

    if (level == LogLevel::Invalid)
        ErrorExit("Invalid --log-level specified.");

#ifdef PBRT_BUILD_GPU_RENDERER
    if (useGPU)
        CUDA_CHECK(cudaMemcpyToSymbol(LOGGING_LogLevelGPU, &LOGGING_LogLevel,
                                      sizeof(LOGGING_LogLevel)));
#endif
}

#ifdef PBRT_BUILD_GPU_RENDERER
std::vector<GPULogItem> ReadGPULogs() {
    CUDA_CHECK(cudaDeviceSynchronize());
    int nItems;
    CUDA_CHECK(cudaMemcpyFromSymbol(&nItems, nRawLogItems, sizeof(nItems)));

    nItems = std::min(nItems, MAX_LOG_ITEMS);
    std::vector<GPULogItem> items(nItems);
    CUDA_CHECK(cudaMemcpyFromSymbol(items.data(), rawLogItems,
                                    nItems * sizeof(GPULogItem), 0,
                                    cudaMemcpyDeviceToHost));

    return items;
}
#endif

LogLevel LogLevelFromString(const std::string &s) {
    if (s == "verbose")
        return LogLevel::Verbose;
    else if (s == "error")
        return LogLevel::Error;
    else if (s == "fatal")
        return LogLevel::Fatal;
    return LogLevel::Invalid;
}

std::string ToString(LogLevel level) {
    switch (level) {
    case LogLevel::Verbose:
        return "VERBOSE";
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Fatal:
        return "FATAL";
    default:
        return "UNKNOWN";
    }
}

void Log(LogLevel level, const char *file, int line, const char *s) {
#ifdef PBRT_IS_GPU_CODE
    auto strlen = [](const char *ptr) {
        int len = 0;
        while (*ptr) {
            ++len;
            ++ptr;
        }
        return len;
    };

    // Grab a slot
    int offset = atomicAdd(&nRawLogItems, 1);
    GPULogItem &item = rawLogItems[offset % MAX_LOG_ITEMS];
    item.level = level;

    // If the file name is too long to fit in GPULogItem.file, then copy
    // the trailing bits.
    int len = strlen(file);
    if (len + 1 > sizeof(item.file)) {
        int start = len - sizeof(item.file) + 1;
        if (start < 0)
            start = 0;
        for (int i = start; i < len; ++i)
            item.file[i - start] = file[i];
        item.file[len - start] = '\0';

        // Now clobber the start with "..." to show it was truncated
        item.file[0] = item.file[1] = item.file[2] = '.';
    } else {
        for (int i = 0; i < len; ++i)
            item.file[i] = file[i];
        item.file[len] = '\0';
    }

    item.line = line;

    // Copy as much of the message as we can...
    int i;
    for (i = 0; i < sizeof(item.message) - 1 && *s; ++i, ++s)
        item.message[i] = *s;
    item.message[i] = '\0';
#else
    int len = strlen(s);
    if (len == 0)
        return;
    fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s %s\n", LOG_BASE_ARGS, file, line,
            ToString(level).c_str(), s);
#endif
}

void LogFatal(LogLevel level, const char *file, int line, const char *s) {
#ifdef PBRT_IS_GPU_CODE
    Log(LogLevel::Fatal, file, line, s);
    __threadfence();
    asm("trap;");
#else
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s %s\n", LOG_BASE_ARGS, file, line,
            ToString(level).c_str(), s);

    CheckCallbackScope::Fail();
    abort();
#endif
}

}  // namespace pbrt
