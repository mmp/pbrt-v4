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

void InitLogging(LogLevel level, bool useGPU);

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
extern LogLevel LOGGING_LogLevel;

// Logging Function Declarations
PBRT_CPU_GPU
void Log(LogLevel level, const char *file, int line, const char *s);

PBRT_CPU_GPU [[noreturn]] void LogFatal(LogLevel level, const char *file, int line,
                                        const char *s);

template <typename... Args>
PBRT_CPU_GPU inline void Log(LogLevel level, const char *file, int line, const char *fmt,
                             Args &&...args);

template <typename... Args>
PBRT_CPU_GPU [[noreturn]] inline void LogFatal(LogLevel level, const char *file, int line,
                                               const char *fmt, Args &&...args);

#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x

#ifdef PBRT_IS_GPU_CODE

extern __constant__ LogLevel LOGGING_LogLevelGPU;

#define LOG_VERBOSE(...)                               \
    (pbrt::LogLevel::Verbose >= LOGGING_LogLevelGPU && \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                               \
    (pbrt::LogLevel::Error >= LOGGING_LogLevelGPU && \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...) \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#else

// Logging Macros
#define LOG_VERBOSE(...)                            \
    (pbrt::LogLevel::Verbose >= LOGGING_LogLevel && \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                            \
    (pbrt::LogLevel::Error >= LOGGING_LogLevel && \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...) \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#endif

}  // namespace pbrt

#include <pbrt/util/print.h>

namespace pbrt {

template <typename... Args>
inline void Log(LogLevel level, const char *file, int line, const char *fmt,
                Args &&...args) {
#ifdef PBRT_IS_GPU_CODE
    Log(level, file, line, fmt);  // just the format string #yolo
#else
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    Log(level, file, line, s.c_str());
#endif
}

template <typename... Args>
inline void LogFatal(LogLevel level, const char *file, int line, const char *fmt,
                     Args &&...args) {
#ifdef PBRT_IS_GPU_CODE
    LogFatal(level, file, line, fmt);  // just the format string #yolo
#else
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    LogFatal(level, file, line, s.c_str());
#endif
}

}  // namespace pbrt

#endif  // PBRT_UTIL_LOG_H
