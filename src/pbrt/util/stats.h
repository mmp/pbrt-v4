// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_STATS_H
#define PBRT_UTIL_STATS_H

#include <pbrt/pbrt.h>

#include <cstdio>
#include <limits>
#include <string>

namespace pbrt {

class StatsAccumulator;
class PixelStatsAccumulator;
// StatRegisterer Definition
class StatRegisterer {
  public:
    // StatRegisterer Public Methods
    using AccumFunc = void (*)(StatsAccumulator &);
    using PixelAccumFunc = void (*)(const Point2i &p, int counterIndex,
                                    PixelStatsAccumulator &);
    StatRegisterer(AccumFunc func, PixelAccumFunc = {});

    static void CallCallbacks(StatsAccumulator &accum);
    static void CallPixelCallbacks(const Point2i &p, PixelStatsAccumulator &accum);
};

void StatsEnablePixelStats(const Bounds2i &b, const std::string &baseName);
void StatsReportPixelStart(const Point2i &p);
void StatsReportPixelEnd(const Point2i &p);

void PrintStats(FILE *dest);
void StatsWritePixelImages();
bool PrintCheckRare(FILE *dest);
void ClearStats();
void ReportThreadStats();

// StatsAccumulator Definition
class StatsAccumulator {
  public:
    // StatsAccumulator Public Methods
    StatsAccumulator();

    void ReportCounter(const char *name, int64_t val);
    void ReportMemoryCounter(const char *name, int64_t val);
    void ReportPercentage(const char *name, int64_t num, int64_t denom);
    void ReportRatio(const char *name, int64_t num, int64_t denom);
    void ReportRareCheck(const char *condition, Float maxFrequency, int64_t numTrue,
                         int64_t total);

    void ReportIntDistribution(const char *name, int64_t sum, int64_t count, int64_t min,
                               int64_t max);
    void ReportFloatDistribution(const char *name, double sum, int64_t count, double min,
                                 double max);

    void AccumulatePixelStats(const PixelStatsAccumulator &accum);
    void WritePixelImages() const;

    void Print(FILE *file);
    bool PrintCheckRare(FILE *dest);
    void Clear();

  private:
    // StatsAccumulator Private Data
    struct Stats;
    Stats *stats = nullptr;
};

// PixelStatsAccumulator Definition
class PixelStatsAccumulator {
  public:
    PixelStatsAccumulator();

    void ReportPixelMS(const Point2i &p, float ms);
    void ReportCounter(const Point2i &p, int counterIndex, const char *name, int64_t val);
    void ReportRatio(const Point2i &p, int counterIndex, const char *name, int64_t num,
                     int64_t denom);

  private:
    friend class StatsAccumulator;
    struct PixelStats;
    PixelStats *stats = nullptr;
};

// Statistics Macros
#define STAT_COUNTER(title, var)                                       \
    static thread_local int64_t var;                                   \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) { \
        accum.ReportCounter(title, var);                               \
        var = 0;                                                       \
    });

#define STAT_PIXEL_COUNTER(title, var)                                         \
    static thread_local int64_t var, var##Sum;                                 \
    static StatRegisterer STATS_REG##var(                                      \
        [](StatsAccumulator &accum) {                                          \
            /* report sum, since if disabled, it all just goes into var... */  \
            accum.ReportCounter(title, var + var##Sum);                        \
            var##Sum = 0;                                                      \
            var = 0;                                                           \
        },                                                                     \
        [](const Point2i &p, int counterIndex, PixelStatsAccumulator &accum) { \
            accum.ReportCounter(p, counterIndex, title, var);                  \
            var##Sum += var;                                                   \
            var = 0;                                                           \
        });

#define STAT_MEMORY_COUNTER(title, var)                                \
    static thread_local int64_t var;                                   \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) { \
        accum.ReportMemoryCounter(title, var);                         \
        var = 0;                                                       \
    });

#define STAT_INT_DISTRIBUTION(title, var)                                             \
    static thread_local int64_t var##sum;                                             \
    static thread_local int64_t var##count;                                           \
    static thread_local int64_t var##min(std::numeric_limits<int64_t>::max());        \
    static thread_local int64_t var##max(std::numeric_limits<int64_t>::lowest());     \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) {                \
        accum.ReportIntDistribution(title, var##sum, var##count, var##min, var##max); \
        var##sum = 0;                                                                 \
        var##count = 0;                                                               \
        var##min = int64_t(std::numeric_limits<int64_t>::max());                      \
        var##max = int64_t(std::numeric_limits<int64_t>::lowest());                   \
    });

#define STAT_FLOAT_DISTRIBUTION(title, var)                                             \
    static thread_local StatCounter<double> var##sum;                                   \
    static thread_local int64_t var##count;                                             \
    static thread_local StatCounter<double> var##min(                                   \
        std::numeric_limits<double>::max());                                            \
    static thread_local StatCounter<double> var##max(                                   \
        std::numeric_limits<double>::lowest());                                         \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) {                  \
        accum.ReportFloatDistribution(title, var##sum, var##count, var##min, var##max); \
        var##sum = 0;                                                                   \
        var##count = 0;                                                                 \
        var##min = StatCounter<double>(std::numeric_limits<double>::max());             \
        var##max = StatCounter<double>(std::numeric_limits<double>::lowest());          \
    });

#define STAT_PERCENT(title, numVar, denomVar)                             \
    static thread_local int64_t numVar, denomVar;                         \
    static StatRegisterer STATS_REG##numVar([](StatsAccumulator &accum) { \
        accum.ReportPercentage(title, numVar, denomVar);                  \
        numVar = 0;                                                       \
        denomVar = 0;                                                     \
    });

#define STAT_RATIO(title, numVar, denomVar)                               \
    static thread_local int64_t numVar, denomVar;                         \
    static StatRegisterer STATS_REG##numVar([](StatsAccumulator &accum) { \
        accum.ReportRatio(title, numVar, denomVar);                       \
        numVar = 0;                                                       \
        denomVar = 0;                                                     \
    });

#define STAT_PIXEL_RATIO(title, numVar, denomVar)                                     \
    static thread_local int64_t numVar, numVar##Sum, denomVar, denomVar##Sum;         \
    static StatRegisterer STATS_REG##numVar##denomVar(                                \
        [](StatsAccumulator &accum) {                                                 \
            /* report sum, since if disabled, it all just goes into var... */         \
            accum.ReportRatio(title, numVar + numVar##Sum, denomVar + denomVar##Sum); \
            numVar = 0;                                                               \
            numVar##Sum = 0;                                                          \
            denomVar = 0;                                                             \
            denomVar##Sum = 0;                                                        \
        },                                                                            \
        [](const Point2i &p, int counterIndex, PixelStatsAccumulator &accum) {        \
            accum.ReportRatio(p, counterIndex, title, numVar, denomVar);              \
            numVar##Sum += numVar;                                                    \
            denomVar##Sum += denomVar;                                                \
            numVar = 0;                                                               \
            denomVar = 0;                                                             \
        });

#define ReportValue(var, value)                           \
    do {                                                  \
        var##sum += value;                                \
        var##count += 1;                                  \
        var##min = (value < var##min) ? value : var##min; \
        var##max = (value > var##max) ? value : var##max; \
    } while (0)

}  // namespace pbrt

#endif  // PBRT_UTIL_STATS_H
