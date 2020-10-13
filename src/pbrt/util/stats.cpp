// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/stats.h>

#include <pbrt/util/check.h>
#include <pbrt/util/image.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/string.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <functional>
#include <map>
#include <mutex>
#include <string>

namespace pbrt {

// ThreadStatsState Definition
struct ThreadStatsState {
    Point2i p;
    bool active;
    std::chrono::steady_clock::time_point start;
    PixelStatsAccumulator accum;
};

// Statistics Local Variables
static std::vector<StatRegisterer::AccumFunc> *statFuncs;

bool pixelStatsEnabled = false;
static thread_local ThreadStatsState threadStatsState;

static std::vector<StatRegisterer::PixelAccumFunc> *pixelStatFuncs;

static StatsAccumulator statsAccumulator;

static Bounds2i imageBounds;
std::string pixelStatsBaseName;

// Statistics Function Definitions
void ReportThreadStats() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    StatRegisterer::CallCallbacks(statsAccumulator);
    if (pixelStatsEnabled) {
        statsAccumulator.AccumulatePixelStats(threadStatsState.accum);
        threadStatsState.accum = PixelStatsAccumulator();
    }
}

void StatsReportPixelStart(const Point2i &p) {
    if (!pixelStatsEnabled)
        return;
    CHECK(threadStatsState.active == false);
    threadStatsState.active = true;
    threadStatsState.p = p;
    threadStatsState.start = std::chrono::steady_clock::now();
}

void StatsReportPixelEnd(const Point2i &p) {
    if (!pixelStatsEnabled)
        return;

    ThreadStatsState &tss = threadStatsState;
    CHECK(tss.active == true && tss.p == p);
    tss.active = false;

    auto elapsed = std::chrono::steady_clock::now() - tss.start;
    float deltaMS =
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 1e6f;
    tss.accum.ReportPixelMS(p, deltaMS);

    StatRegisterer::CallPixelCallbacks(p, tss.accum);
}

void StatsEnablePixelStats(const Bounds2i &b, const std::string &baseName) {
    pixelStatsEnabled = true;
    imageBounds = b;
    pixelStatsBaseName = baseName;
}

// StatRegisterer Method Definitions
void StatRegisterer::CallCallbacks(StatsAccumulator &accum) {
    for (AccumFunc func : *statFuncs)
        func(accum);
}

StatRegisterer::StatRegisterer(AccumFunc func, PixelAccumFunc pfunc) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (statFuncs == nullptr)
        statFuncs = new std::vector<AccumFunc>;
    if (func)
        statFuncs->push_back(func);

    if (pixelStatFuncs == nullptr)
        pixelStatFuncs = new std::vector<PixelAccumFunc>;
    if (pfunc)
        pixelStatFuncs->push_back(pfunc);
}

void StatRegisterer::CallPixelCallbacks(const Point2i &p, PixelStatsAccumulator &accum) {
    for (size_t i = 0; i < pixelStatFuncs->size(); ++i)
        (*pixelStatFuncs)[i](p, i, accum);
}

// PixelStatsAccumulator::PixelStats Definition
struct PixelStatsAccumulator::PixelStats {
    Image time;
    std::vector<std::string> counterNames;
    std::vector<Image> counterImages;
    std::vector<std::string> ratioNames;
    std::vector<Image> ratioImages;
};

PixelStatsAccumulator::PixelStatsAccumulator() {
    stats = new PixelStats;
}

// PixelStatsAccumulator Method Definitions
void PixelStatsAccumulator::ReportPixelMS(const Point2i &p, float ms) {
    Point2i res = Point2i(imageBounds.Diagonal());
    if (stats->time.Resolution() != res)
        stats->time = Image(PixelFormat::Float, res, {"ms"});

    Point2i pp = Point2i(p - imageBounds.pMin);
    stats->time.SetChannel(pp, 0, stats->time.GetChannel(pp, 0) + ms);
}

void PixelStatsAccumulator::ReportCounter(const Point2i &p, int statIndex,
                                          const char *name, int64_t val) {
    if (statIndex >= stats->counterImages.size()) {
        stats->counterImages.resize(statIndex + 1);
        stats->counterNames.resize(statIndex + 1);
        stats->counterNames[statIndex] = name;
    }

    Image &im = stats->counterImages[statIndex];
    Point2i res = Point2i(imageBounds.Diagonal());
    if (im.Resolution() != res)
        im = Image(PixelFormat::Float, res, {"count"});

    Point2i pp = Point2i(p - imageBounds.pMin);
    im.SetChannel(pp, 0, im.GetChannel(pp, 0) + val);
}

void PixelStatsAccumulator::ReportRatio(const Point2i &p, int statIndex, const char *name,
                                        int64_t num, int64_t denom) {
    if (statIndex >= stats->ratioImages.size()) {
        stats->ratioImages.resize(statIndex + 1);
        stats->ratioNames.resize(statIndex + 1);
        stats->ratioNames[statIndex] = name;
    }

    Image &im = stats->ratioImages[statIndex];
    Point2i res = Point2i(imageBounds.Diagonal());
    if (im.Resolution() != res)
        im = Image(PixelFormat::Float, res, {"numerator", "denominator", "ratio"});

    Point2i pp = Point2i(p - imageBounds.pMin);
    im.SetChannel(pp, 0, im.GetChannel(pp, 0) + num);
    im.SetChannel(pp, 1, im.GetChannel(pp, 1) + denom);
    if (im.GetChannel(pp, 0) == 0)
        im.SetChannel(pp, 2, 0);
    else
        im.SetChannel(pp, 2, im.GetChannel(pp, 0) / im.GetChannel(pp, 1));
}

// StatsAccumulator::Stats Definition
struct StatsAccumulator::Stats {
    std::map<std::string, int64_t> counters;
    std::map<std::string, int64_t> memoryCounters;
    template <typename T>
    struct Distribution {
        int64_t count = 0;
        T min = std::numeric_limits<T>::max();
        T max = std::numeric_limits<T>::lowest();
        T sum = 0;
    };
    std::map<std::string, Distribution<int64_t>> intDistributions;
    std::map<std::string, Distribution<double>> floatDistributions;
    std::map<std::string, std::pair<int64_t, int64_t>> percentages;
    std::map<std::string, std::pair<int64_t, int64_t>> ratios;
    struct RareCheck {
        RareCheck(Float f = 0) : maxFrequency(f) {}
        Float maxFrequency;
        int64_t numTrue = 0, total = 0;
    };
    std::map<std::string, RareCheck> rareChecks;

    Image pixelTime;
    std::vector<std::string> pixelCounterNames;
    std::vector<Image> pixelCounterImages;
    std::vector<std::string> pixelRatioNames;
    std::vector<Image> pixelRatioImages;
};

// StatsAccumulator Method Definitions
void StatsAccumulator::ReportCounter(const char *name, int64_t val) {
    stats->counters[name] += val;
}

StatsAccumulator::StatsAccumulator() {
    stats = new Stats;
}

void StatsAccumulator::ReportMemoryCounter(const char *name, int64_t val) {
    stats->memoryCounters[name] += val;
}

void StatsAccumulator::ReportPercentage(const char *name, int64_t num, int64_t denom) {
    stats->percentages[name].first += num;
    stats->percentages[name].second += denom;
}

void StatsAccumulator::ReportRatio(const char *name, int64_t num, int64_t denom) {
    stats->ratios[name].first += num;
    stats->ratios[name].second += denom;
}

void StatsAccumulator::ReportRareCheck(const char *condition, Float maxFrequency,
                                       int64_t numTrue, int64_t total) {
    if (stats->rareChecks.find(condition) == stats->rareChecks.end())
        stats->rareChecks[condition] = Stats::RareCheck(maxFrequency);
    Stats::RareCheck &rc = stats->rareChecks[condition];
    rc.numTrue += numTrue;
    rc.total += total;
}

void StatsAccumulator::ReportIntDistribution(const char *name, int64_t sum, int64_t count,
                                             int64_t min, int64_t max) {
    Stats::Distribution<int64_t> &distrib = stats->intDistributions[name];
    distrib.sum += sum;
    distrib.count += count;
    distrib.min = std::min(distrib.min, min);
    distrib.max = std::max(distrib.max, max);
}

void StatsAccumulator::ReportFloatDistribution(const char *name, double sum,
                                               int64_t count, double min, double max) {
    Stats::Distribution<double> &distrib = stats->floatDistributions[name];
    distrib.sum += sum;
    distrib.count += count;
    distrib.min = std::min(distrib.min, min);
    distrib.max = std::max(distrib.max, max);
}

void StatsAccumulator::AccumulatePixelStats(const PixelStatsAccumulator &accum) {
    Point2i res = Point2i(imageBounds.Diagonal());
    if (stats->pixelTime.Resolution() == Point2i(0, 0))
        stats->pixelTime = Image(PixelFormat::Float, res, {"ms"});
    else
        CHECK_EQ(stats->pixelTime.Resolution(), res);
    CHECK_EQ(stats->pixelTime.Resolution(), accum.stats->time.Resolution());

    for (int y = 0; y < stats->pixelTime.Resolution().y; ++y)
        for (int x = 0; x < stats->pixelTime.Resolution().x; ++x)
            stats->pixelTime.SetChannel({x, y}, 0,
                                        (stats->pixelTime.GetChannel({x, y}, 0) +
                                         accum.stats->time.GetChannel({x, y}, 0)));

    if (stats->pixelCounterImages.size() < accum.stats->counterImages.size()) {
        stats->pixelCounterImages.resize(accum.stats->counterImages.size());
        stats->pixelCounterNames.resize(accum.stats->counterNames.size());
    }
    if (stats->pixelRatioImages.size() < accum.stats->ratioImages.size()) {
        stats->pixelRatioImages.resize(accum.stats->ratioImages.size());
        stats->pixelRatioNames.resize(accum.stats->ratioNames.size());
    }

    for (size_t i = 0; i < accum.stats->counterImages.size(); ++i) {
        if (stats->pixelCounterNames[i].empty())
            stats->pixelCounterNames[i] = accum.stats->counterNames[i];

        const Image &threadImage = accum.stats->counterImages[i];
        if (threadImage.Resolution() == Point2i(0, 0))
            continue;
        Image &accumImage = stats->pixelCounterImages[i];
        if (accumImage.Resolution() == Point2i(0, 0))
            accumImage = Image(PixelFormat::Float, threadImage.Resolution(), {"count"});
        for (int y = 0; y < threadImage.Resolution().y; ++y)
            for (int x = 0; x < threadImage.Resolution().x; ++x)
                accumImage.SetChannel({x, y}, 0,
                                      (accumImage.GetChannel({x, y}, 0) +
                                       threadImage.GetChannel({x, y}, 0)));
    }
    for (size_t i = 0; i < accum.stats->ratioImages.size(); ++i) {
        if (stats->pixelRatioNames[i].empty())
            stats->pixelRatioNames[i] = accum.stats->ratioNames[i];

        const Image &threadImage = accum.stats->ratioImages[i];
        if (threadImage.Resolution() == Point2i(0, 0))
            continue;
        Image &accumImage = stats->pixelRatioImages[i];
        if (accumImage.Resolution() == Point2i(0, 0))
            accumImage = Image(PixelFormat::Float, threadImage.Resolution(),
                               {"numerator", "denominator", "ratio"});
        for (int y = 0; y < threadImage.Resolution().y; ++y)
            for (int x = 0; x < threadImage.Resolution().x; ++x) {
                accumImage.SetChannel({x, y}, 0,
                                      (accumImage.GetChannel({x, y}, 0) +
                                       threadImage.GetChannel({x, y}, 0)));
                accumImage.SetChannel({x, y}, 1,
                                      (accumImage.GetChannel({x, y}, 1) +
                                       threadImage.GetChannel({x, y}, 1)));
                if (accumImage.GetChannel({x, y}, 0) == 0)
                    accumImage.SetChannel({x, y}, 2, 0.f);
                else
                    accumImage.SetChannel({x, y}, 2,
                                          (accumImage.GetChannel({x, y}, 0) /
                                           accumImage.GetChannel({x, y}, 1)));
            }
    }
}

void PrintStats(FILE *dest) {
    statsAccumulator.Print(dest);
}

bool PrintCheckRare(FILE *dest) {
    return statsAccumulator.PrintCheckRare(dest);
}

void ClearStats() {
    statsAccumulator.Clear();
}

static void getCategoryAndTitle(const std::string &str, std::string *category,
                                std::string *title) {
    std::vector<std::string> comps = SplitString(str, '/');
    if (comps.size() == 1)
        *title = comps[0];
    else {
        CHECK_EQ(comps.size(), 2);
        *category = comps[0];
        *title = comps[1];
    }
}

void StatsAccumulator::Print(FILE *dest) {
    fprintf(dest, "Statistics:\n");
    std::map<std::string, std::vector<std::string>> toPrint;

    for (auto &counter : stats->counters) {
        if (counter.second == 0)
            continue;
        std::string category, title;
        getCategoryAndTitle(counter.first, &category, &title);
        toPrint[category].push_back(
            StringPrintf("%-42s               %12" PRIu64, title, counter.second));
    }

    size_t totalMemoryReported = 0;
    auto printBytes = [](size_t bytes) -> std::string {
        float kb = (double)bytes / 1024.;
        if (std::abs(kb) < 1024.)
            return StringPrintf("%9.2f kB", kb);

        float mib = kb / 1024.;
        if (std::abs(mib) < 1024.)
            return StringPrintf("%9.2f MiB", mib);

        float gib = mib / 1024.;
        return StringPrintf("%9.2f GiB", gib);
    };

    for (auto &counter : stats->memoryCounters) {
        if (counter.second == 0)
            continue;
        totalMemoryReported += counter.second;

        std::string category, title;
        getCategoryAndTitle(counter.first, &category, &title);
        toPrint[category].push_back(
            StringPrintf("%-42s                  %s", title, printBytes(counter.second)));
    }
    int64_t unreportedBytes = GetCurrentRSS() - totalMemoryReported;
    if (unreportedBytes > 0)
        toPrint["Memory"].push_back(StringPrintf("%-42s                  %s",
                                                 "Unreported / unused",
                                                 printBytes(unreportedBytes)));

    for (auto &distrib : stats->intDistributions) {
        const std::string &name = distrib.first;
        if (distrib.second.count == 0)
            continue;
        std::string category, title;
        getCategoryAndTitle(name, &category, &title);
        double avg = (double)distrib.second.sum / (double)distrib.second.count;
        toPrint[category].push_back(StringPrintf(
            "%-42s                      %.3f avg [range %" PRIu64 " - %" PRIu64 "]",
            title, avg, distrib.second.min, distrib.second.max));
    }
    for (auto &distrib : stats->floatDistributions) {
        const std::string &name = distrib.first;
        if (distrib.second.count == 0)
            continue;
        std::string category, title;
        getCategoryAndTitle(name, &category, &title);
        double avg = (double)distrib.second.sum / (double)distrib.second.count;
        toPrint[category].push_back(
            StringPrintf("%-42s                      %.3f avg [range %f - %f]", title,
                         avg, distrib.second.min, distrib.second.max));
    }
    for (auto &percentage : stats->percentages) {
        if (percentage.second.second == 0)
            continue;
        int64_t num = percentage.second.first;
        int64_t denom = percentage.second.second;
        std::string category, title;
        getCategoryAndTitle(percentage.first, &category, &title);
        toPrint[category].push_back(
            StringPrintf("%-42s%12" PRIu64 " / %12" PRIu64 " (%.2f%%)", title, num, denom,
                         (100.f * num) / denom));
    }
    for (auto &ratio : stats->ratios) {
        if (ratio.second.second == 0)
            continue;
        int64_t num = ratio.second.first;
        int64_t denom = ratio.second.second;
        std::string category, title;
        getCategoryAndTitle(ratio.first, &category, &title);
        toPrint[category].push_back(
            StringPrintf("%-42s%12" PRIu64 " / %12" PRIu64 " (%.2fx)", title, num, denom,
                         (double)num / (double)denom));
    }

    for (auto &categories : toPrint) {
        fprintf(dest, "  %s\n", categories.first.c_str());
        for (auto &item : categories.second)
            fprintf(dest, "    %s\n", item.c_str());
    }
}

void StatsWritePixelImages() {
    statsAccumulator.WritePixelImages();
}

void StatsAccumulator::WritePixelImages() const {
    // FIXME: do this where?
    CHECK(stats->pixelTime.Write(pixelStatsBaseName + "-time.exr"));

    for (size_t i = 0; i < stats->pixelCounterImages.size(); ++i) {
        std::string n = pixelStatsBaseName + "-" + stats->pixelCounterNames[i] + ".exr";
        for (size_t j = 0; j < n.size(); ++j)
            if (n[j] == '/')
                n[j] = '_';

        auto AllZero = [](const Image &im) {
            for (int y = 0; y < im.Resolution().y; ++y)
                for (int x = 0; x < im.Resolution().x; ++x)
                    if (im.GetChannel({x, y}, 0) != 0)
                        return false;
            return true;
        };
        if (!AllZero(stats->pixelCounterImages[i]))
            CHECK(stats->pixelCounterImages[i].Write(n));
    }

    for (size_t i = 0; i < stats->pixelRatioImages.size(); ++i) {
        std::string n = pixelStatsBaseName + "-" + stats->pixelRatioNames[i] + ".exr";
        for (size_t j = 0; j < n.size(); ++j)
            if (n[j] == '/')
                n[j] = '_';

        auto AllZero = [](const Image &im) {
            for (int y = 0; y < im.Resolution().y; ++y)
                for (int x = 0; x < im.Resolution().x; ++x)
                    if (im.GetChannels({x, y}).MaxValue() != 0)
                        return false;
            return true;
        };
        if (!AllZero(stats->pixelRatioImages[i]))
            CHECK(stats->pixelRatioImages[i].Write(n));
    }
}

bool StatsAccumulator::PrintCheckRare(FILE *dest) {
    bool anyFailed = false;
    for (const auto &iter : stats->rareChecks) {
        const Stats::RareCheck &rc = iter.second;
        Float trueFreq = double(rc.numTrue) / double(rc.total);
        Float varianceEstimate = 1 / double(rc.total - 1) * trueFreq * (1 - trueFreq);
        if (trueFreq - 2 * varianceEstimate >= rc.maxFrequency) {
            fprintf(dest,
                    "%s @ %.9g failures was %fx over limit %.9g (%" PRId64
                    " samples, sigma est %.9g)\n",
                    iter.first.c_str(), trueFreq, trueFreq / rc.maxFrequency,
                    rc.maxFrequency, rc.total, varianceEstimate);
            anyFailed = true;
        }
    }
    return anyFailed;
}

void StatsAccumulator::Clear() {
    stats->counters.clear();
    stats->memoryCounters.clear();
    stats->intDistributions.clear();
    stats->floatDistributions.clear();
    stats->percentages.clear();
    stats->ratios.clear();
}

}  // namespace pbrt
