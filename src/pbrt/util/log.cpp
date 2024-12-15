// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/log.h>

#include <pbrt/options.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/string.h>

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#ifdef PBRT_IS_OSX
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif
#ifdef PBRT_IS_LINUX
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif
#ifdef PBRT_IS_WINDOWS
#include <windows.h>
#endif
#if defined(PBRT_USE_NVML)
#include <nvml.h>
#endif

namespace pbrt {

namespace {

float ElapsedSeconds() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    using clock = std::chrono::steady_clock;
    static clock::time_point start = clock::now();

    clock::time_point now = clock::now();
    int64_t elapseduS =
        std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
    return elapseduS / 1000000.;
}

uint32_t GetThreadIndex() {
#ifdef PBRT_IS_LINUX
    unsigned int tid = syscall(SYS_gettid);
    return tid;
#elif defined(PBRT_IS_WINDOWS)
    return GetCurrentThreadId();
#elif defined(PBRT_IS_OSX)
    uint64_t tid;
    CHECK_EQ(pthread_threadid_np(pthread_self(), &tid), 0);
    return tid;
#else
#error "Need to define GetThreadIndex() for system"
#endif
}

#define LOG_BASE_FMT "tid %03d @ %9.3fs"
#define LOG_BASE_ARGS GetThreadIndex(), ElapsedSeconds()

}  // namespace

namespace logging {

LogLevel logLevel = LogLevel::Error;
FILE *logFile;

}  // namespace logging

#ifdef PBRT_BUILD_GPU_RENDERER
__constant__ LogLevel LOGGING_LogLevelGPU;

#define MAX_LOG_ITEMS 1024
PBRT_GPU GPULogItem rawLogItems[MAX_LOG_ITEMS];
PBRT_GPU int nRawLogItems;
#endif  // PBRT_BUILD_GPU_RENDERER

static std::atomic<bool> shutdownLogUtilization;
static std::thread logUtilizationThread;

void InitLogging(LogLevel level, std::string logFile, bool logUtilization, bool useGPU) {
    logging::logLevel = level;
    if (!logFile.empty()) {
        logging::logFile = FOpenWrite(logFile);
        if (!logging::logFile)
            ErrorExit("%s: %s", logFile, ErrorString());
        logging::logLevel = LogLevel::Verbose;
    }

#ifdef PBRT_IS_WINDOWS
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD consoleMode;
    GetConsoleMode(hStdout, &consoleMode);
    consoleMode |= 0xc;  // virtual terminal processing, disable newline auto return
    SetConsoleMode(hStdout, consoleMode);
#endif  // PBRT_IS_WINDOWS

    if (level == LogLevel::Invalid)
        ErrorExit("Invalid --log-level specified.");

#ifdef PBRT_BUILD_GPU_RENDERER
    if (useGPU)
        CUDA_CHECK(cudaMemcpyToSymbol(LOGGING_LogLevelGPU, &logging::logLevel,
                                      sizeof(logging::logLevel)));
#endif

    if (logUtilization) {
        shutdownLogUtilization = false;
        auto sleepms = [](int64_t ms) {
#if defined(PBRT_IS_LINUX) || defined(PBRT_IS_OSX)
            struct timespec rec, rem;
            rec.tv_sec = 0;
            rec.tv_nsec = ms * 1000000;
            nanosleep(&rec, &rem);
#elif defined(PBRT_IS_WINDOWS)
            Sleep(ms);
#else
#error "Need to implement sleepms() for current platform"
#endif
        };

        struct Usage {
            // CPU usage in ticks
            int64_t user = 0, nice = 0, system = 0, idle = 0;
            // I/O in bytes: requested by process
            int64_t readRequest = 0, writeRequest = 0;
            // I/O in bytes: how much actually came from/went to storage
            int64_t readActual = 0, writeActual = 0;
        };
        // Return overall CPU usage in ticks
        auto getCPUUsage = [&]() -> Usage {
            Usage usage;
#ifdef PBRT_IS_LINUX
            std::ifstream stat("/proc/stat");
            CHECK((bool)stat);

            std::string line;
            while (std::getline(stat, line)) {
                if (line.compare(0, 4, "cpu ") != 0)
                    continue;

                std::string_view tail(line.c_str() + 5);
                std::vector<int64_t> values = SplitStringToInt64s(tail, ' ');

                CHECK_GE(values.size(), 4);
                usage.user = values[0];
                usage.nice = values[1];
                usage.system = values[2];
                usage.idle = values[3];

                break;
            }

            std::ifstream io("/proc/self/io");
            CHECK((bool)io);
            while (std::getline(io, line)) {
                if (line.compare(0, 7, "rchar: ") == 0)
                    CHECK(Atoi(line.data() + 7, &usage.readRequest));
                if (line.compare(0, 7, "wchar: ") == 0)
                    CHECK(Atoi(line.data() + 7, &usage.writeRequest));
                if (line.compare(0, 12, "read_bytes: ") == 0)
                    CHECK(Atoi(line.data() + 12, &usage.readActual));
                if (line.compare(0, 13, "write_bytes: ") == 0)
                    CHECK(Atoi(line.data() + 13, &usage.writeActual));
            }
#elif defined(PBRT_IS_OSX)
            // possibly useful:
            // https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
#elif defined(PBRT_IS_WINDOWS)
            FILETIME idleTime, kernelTime, userTime;
            if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
                auto FileTimeToInt64 = [](const FILETIME &ft) {
                    return (((unsigned long long)(ft.dwHighDateTime)) << 32) |
                           ((unsigned long long)ft.dwLowDateTime);
                };
                usage.idle = FileTimeToInt64(idleTime);
                usage.system = FileTimeToInt64(kernelTime);
                usage.nice = 0;
                usage.user = FileTimeToInt64(userTime);
            }

            IO_COUNTERS ioCounters;
            if (GetProcessIoCounters(GetCurrentProcess(), &ioCounters)) {
                // Windows doesn't seem to distinguish between I/O that hit
                // the buffer cache and I/O that didn't...
                usage.readRequest = ioCounters.ReadTransferCount;
                usage.writeRequest = ioCounters.WriteTransferCount;
            }
#else
#error "Need to implement getCPUUsage for current platform"
#endif
            return usage;
        };

        logUtilizationThread = std::thread([&]() {
            Usage prevUsage = getCPUUsage();

#ifdef PBRT_IS_WINDOWS
            // It's necessary to increase this thread's priority a bit since otherwise
            // the worker threads during rendering take all the cycles and it only
            // runs once a second or so.
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif

#ifdef PBRT_USE_NVML
            // NOTE: per https://stackoverflow.com/a/64610732 apparently a
            // call to LoadLibraryW(L"nvapi64.dll") will be a good idea on
            // windows...
            if (nvmlInit_v2() != NVML_SUCCESS)
                LOG_ERROR("Unable to initialize NVML");

            int deviceIndex = Options->gpuDevice ? *Options->gpuDevice : 0;
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex_v2(deviceIndex, &device) != NVML_SUCCESS)
                LOG_ERROR("Unable to get NVML device");
#endif

            while (!shutdownLogUtilization) {
                sleepms(100);

                Usage currentUsage = getCPUUsage();

                // Report activity since last logging event. A value of 1
                // represents all cores running at 100% utilization.
                int64_t delta =
                    (currentUsage.user + currentUsage.nice + currentUsage.system +
                     currentUsage.idle) -
                    (prevUsage.user + prevUsage.nice + prevUsage.system + prevUsage.idle);
                LOG_VERBOSE("CPU: Memory used %d MB. "
                            "Core activity: %.4f user %.4f nice %.4f system %.4f idle",
                            GetCurrentRSS() / (1024 * 1024),
                            double(currentUsage.user - prevUsage.user) / delta,
                            double(currentUsage.nice - prevUsage.nice) / delta,
                            double(currentUsage.system - prevUsage.system) / delta,
                            double(currentUsage.idle - prevUsage.idle) / delta);
                LOG_VERBOSE(
                    "IO: read request %d read actual %d write request %d write actual %d",
                    currentUsage.readRequest - prevUsage.readRequest,
                    currentUsage.readActual - prevUsage.readActual,
                    currentUsage.writeRequest - prevUsage.writeRequest,
                    currentUsage.writeActual - prevUsage.writeActual);

                prevUsage = currentUsage;

#ifdef PBRT_USE_NVML
                nvmlMemory_t info;
                nvmlUtilization_t utilization;
                if (nvmlDeviceGetMemoryInfo(device, &info) == NVML_SUCCESS &&
                    nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS)
                    LOG_VERBOSE(
                        "GPU: Memory used %d MB. SM activity: %d memory activity: %d",
                        info.used / (1024 * 1024), utilization.gpu, utilization.memory);
#endif
            }
#ifdef PBRT_USE_NVML
            nvmlShutdown();
#endif
        });
    }
}

void ShutdownLogging() {
    if (logUtilizationThread.get_id() != std::thread::id()) {
        shutdownLogUtilization = true;
        logUtilizationThread.join();
    }
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

PBRT_CPU_GPU void Log(LogLevel level, const char *file, int line, const char *s) {
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
    std::string levelString = (level == LogLevel::Verbose) ? "" : (ToString(level) + " ");

    // cut off everything up to pbrt/
    const char *fileStart = strstr(file, "pbrt/");
    std::string shortfile(fileStart ? (fileStart + 5) : file);

    if (logging::logFile) {
        fprintf(logging::logFile, "[ " LOG_BASE_FMT " %s:%d ] %s%s\n", LOG_BASE_ARGS,
                shortfile.c_str(), line, levelString.c_str(), s);
        fflush(logging::logFile);
    } else
        fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s%s\n", LOG_BASE_ARGS,
                shortfile.c_str(), line, levelString.c_str(), s);
#endif
}

#if defined(__NVCC__) || defined(__HIPCC__)
// warning #1305-D: function declared with "noreturn" does return
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 1305
#else
#pragma diag_suppress 1305
#endif
#endif

PBRT_CPU_GPU void LogFatal(LogLevel level, const char *file, int line, const char *s) {
#ifdef PBRT_IS_GPU_CODE
    Log(LogLevel::Fatal, file, line, s);
    __threadfence();
    asm("trap;");
#else
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // cut off everything up to pbrt/
    const char *fileStart = strstr(file, "pbrt/");
    std::string shortfile(fileStart ? (fileStart + 5) : file);
    fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s %s\n", LOG_BASE_ARGS,
            shortfile.c_str(), line, ToString(level).c_str(), s);

    CheckCallbackScope::Fail();
    abort();
#endif
}

}  // namespace pbrt
