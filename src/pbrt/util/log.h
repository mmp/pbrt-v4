// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_LOG_H
#define PBRT_UTIL_LOG_H

#include <pbrt/pbrt.h>

#include <string>
#include <vector>

namespace pbrt {

// LogLevel Definition
enum class LogLevel { Verbose, Error, Fatal, Invalid };

std::string ToString(LogLevel level);
LogLevel LogLevelFromString(const std::string &s);

void ShutdownLogging();
void InitLogging(LogLevel level, std::string logFile, bool logUtilization, bool useGPU);

#ifdef PBRT_BUILD_GPU_RENDERER

struct GPULogItem {
    LogLevel level;
    char file[64];
    int line;
    char message[128];
};

std::vector<GPULogItem> ReadGPULogs();

#endif

// LogLevel Global Variable Declaration
namespace logging {
extern LogLevel logLevel;
extern FILE *logFile;
}  // namespace logging

// Logging Function Declarations
PBRT_CPU_GPU
void Log(LogLevel level, const char *file, int line, const char *s);

template <typename... Args>
PBRT_CPU_GPU inline void LogFatal(LogLevel level, const char *file, int line,
                                  const char *fmt, Args &&...args);

#ifndef __HIPCC__
PBRT_CPU_GPU void LogFatal(LogLevel level, const char *file, int line, const char *s);

template <typename... Args>
PBRT_CPU_GPU inline void Log(LogLevel level, const char *file, int line, const char *fmt,
                             Args &&...args);
#endif

#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x

#ifdef PBRT_IS_GPU_CODE

extern __constant__ LogLevel LOGGING_LogLevelGPU;

// printing may cause hang in the device code
#ifdef __HIP_DEVICE_COMPILE__

#define LOG_VERBOSE(...) \
    do {                 \
    } while (false) /* swallow semicolon */

#define LOG_ERROR(...) \
    do {               \
    } while (false) /* swallow semicolon */

#define LOG_FATAL(...) \
    do {               \
    } while (false) /* swallow semicolon */

#else

#define LOG_VERBOSE(...)                               \
    (pbrt::LogLevel::Verbose >= LOGGING_LogLevelGPU && \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                               \
    (pbrt::LogLevel::Error >= LOGGING_LogLevelGPU && \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...) \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#endif

#else

// Logging Macros
#define LOG_VERBOSE(...)                             \
    (pbrt::LogLevel::Verbose >= logging::logLevel && \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                                   \
    (pbrt::LogLevel::Error >= pbrt::logging::logLevel && \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...) \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#endif

}  // namespace pbrt

#include <pbrt/util/print.h>

namespace pbrt {

template <typename... Args>
PBRT_CPU_GPU inline void Log(LogLevel level, const char *file, int line, const char *fmt,
                             Args &&...args) {
#if defined(PBRT_IS_GPU_CODE)
    Log(level, file, line, fmt);  // just the format string #yolo
#else
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    Log(level, file, line, s.c_str());
#endif
}

template <typename... Args>
PBRT_CPU_GPU inline void LogFatal(LogLevel level, const char *file, int line,
                                  const char *fmt, Args &&...args) {
#if defined(PBRT_IS_GPU_CODE) || defined(__HIPCC__)
    LogFatal(level, file, line, fmt);  // just the format string #yolo
#else
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    LogFatal(level, file, line, s.c_str());
#endif
}

}  // namespace pbrt

#endif  // PBRT_UTIL_LOG_H
