// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_SAMPLERS_H
#define PBRT_SAMPLERS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/sampler.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pmj02tables.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/vecmath.h>

#include <limits>
#include <memory>
#include <string>

namespace pbrt {

// HaltonSampler Definition
class HaltonSampler {
  public:
    // HaltonSampler Public Methods
    HaltonSampler(int samplesPerPixel, const Point2i &fullResolution,
                  pstd::vector<DigitPermutation> *digitPermutations = nullptr,
                  Allocator alloc = {});

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "HaltonSampler"; }
    static HaltonSampler *Create(const ParameterDictionary &parameters,
                                 const Point2i &fullResolution, const FileLoc *loc,
                                 Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index, int dim) {
        if (p != pixel)
            haltonPixelIndexer.SetPixel(p);
        haltonPixelIndexer.SetPixelSample(index);

        pixel = p;
        sampleIndex = index;
        dimension = dim;
    }

    PBRT_CPU_GPU
    Float Get1D() {
        if (dimension >= PrimeTableSize)
            dimension = 2;
        int dim = dimension++;
        return ScrambledRadicalInverse(dim, haltonPixelIndexer.SampleIndex(),
                                       (*digitPermutations)[dim]);
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        if (dimension == 0) {
            dimension += 2;
            return haltonPixelIndexer.SampleFirst2D();
        } else {
            if (dimension + 1 >= PrimeTableSize)
                dimension = 2;

            int dim = dimension;
            dimension += 2;
            return {ScrambledRadicalInverse(dim, haltonPixelIndexer.SampleIndex(),
                                            (*digitPermutations)[dim]),
                    ScrambledRadicalInverse(dim + 1, haltonPixelIndexer.SampleIndex(),
                                            (*digitPermutations)[dim + 1])};
        }
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // HaltonSampler Private Members
    pstd::vector<DigitPermutation> *digitPermutations;
    int samplesPerPixel;
    HaltonPixelIndexer haltonPixelIndexer;
    Point2i pixel =
        Point2i(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
    int sampleIndex = 0;
    int dimension = 0;
};

// PaddedSobolSampler Definition
class PaddedSobolSampler {
  public:
    // PaddedSobolSampler Public Methods
    PaddedSobolSampler(int samplesPerPixel, RandomizeStrategy randomizeStrategy);

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "PaddedSobolSampler"; }
    static PaddedSobolSampler *Create(const ParameterDictionary &parameters,
                                      const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index, int dim) {
        pixel = p;
        sampleIndex = index;
        dimension = dim;
    }

    PBRT_CPU_GPU
    Float Get1D() {
        // Get permuted index for current pixel sample
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);

        int dim = dimension++;
        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
            // Return 1D sample randomized with Cranley-Patterson rotation
            return SampleGeneratorMatrix(
                CVanDerCorput, index,
                CranleyPattersonRotator(BlueNoise(dim, pixel.x, pixel.y)));

        else
            return generateSample(CVanDerCorput, index, hash >> 32);
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        // Get permuted index for current pixel sample
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);

        int dim = dimension;
        dimension += 2;
        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
            // Return 2D sample randomized with Cranley-Patterson rotation
            return {SampleGeneratorMatrix(
                        CSobol[0], index,
                        CranleyPattersonRotator(BlueNoise(dim, pixel.x, pixel.y))),
                    SampleGeneratorMatrix(
                        CSobol[1], index,
                        CranleyPattersonRotator(BlueNoise(dim + 1, pixel.x, pixel.y)))};

        else
            return {generateSample(CSobol[0], index, hash >> 8),
                    generateSample(CSobol[1], index, hash >> 32)};
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // PaddedSobolSampler Private Methods
    PBRT_CPU_GPU
    Float generateSample(pstd::span<const uint32_t> C, uint32_t a, uint32_t hash) const {
        switch (randomizeStrategy) {
        case RandomizeStrategy::None:
            return SampleGeneratorMatrix(C, a, NoRandomizer());
        case RandomizeStrategy::Xor:
            return SampleGeneratorMatrix(C, a, XORScrambler(hash));
        case RandomizeStrategy::Owen:
            return SampleGeneratorMatrix(C, a, OwenScrambler(hash));
        default:
            LOG_FATAL("Unhandled randomization strategy");
            return {};
        }
    }

    // PaddedSobolSampler Private Members
    int samplesPerPixel;
    RandomizeStrategy randomizeStrategy;
    Point2i pixel;
    int sampleIndex = 0;
    int dimension = 0;
};

// PMJ02BNSampler Definition
class PMJ02BNSampler {
  public:
    // PMJ02BNSampler Public Methods
    PMJ02BNSampler(int samplesPerPixel, Allocator alloc = {});

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "PMJ02BNSampler"; }
    static PMJ02BNSampler *Create(const ParameterDictionary &parameters,
                                  const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index, int dim) {
        pixel = p;
        sampleIndex = index;
        dimension = dim;
    }

    PBRT_CPU_GPU
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);

        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);
        Float cpOffset = BlueNoise(dimension, pixel.x, pixel.y);
        Float u = (index + cpOffset) / samplesPerPixel;
        if (u >= 1)
            u -= 1;
        ++dimension;
        return std::min(u, OneMinusEpsilon);
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        // Don't start permuting until the second time through: when we
        // permute, that breaks the progressive part of the pattern and in
        // turn, convergence is similar to random until the very end. This way,
        // we generally do well for intermediate images as well.
        int index = sampleIndex;
        int pmjInstance = dimension;
        if (pmjInstance >= nPMJ02bnSets) {
            uint64_t hash =
                MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                        ((uint64_t)dimension << 16) ^ GetOptions().seed);
            index = PermutationElement(sampleIndex, samplesPerPixel, hash);
        }

        if (dimension == 0) {
            // special case the pixel sample
            int offset = pixelSampleOffset(Point2i(pixel));
            dimension += 2;
            return (*pixelSamples)[offset + index];
        } else {
            Vector2f cpOffset(BlueNoise(dimension, pixel.x, pixel.y),
                              BlueNoise(dimension + 1, pixel.x, pixel.y));
            Point2f u = GetPMJ02BNSample(pmjInstance % nPMJ02bnSets, index) + cpOffset;
            if (u.x >= 1)
                u.x -= 1;
            if (u.y >= 1)
                u.y -= 1;
            dimension += 2;
            return {std::min(u.x, OneMinusEpsilon), std::min(u.y, OneMinusEpsilon)};
        }
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // PMJ02BNSampler Private Methods
    PBRT_CPU_GPU
    int pixelSampleOffset(Point2i p) const {
        DCHECK(p.x >= 0 && p.y >= 0);
        int px = p.x % pixelTileSize, py = p.y % pixelTileSize;
        return (px + py * pixelTileSize) * samplesPerPixel;
    }

    // PMJ02BNSampler Private Members
    Point2i pixel;
    int sampleIndex = 0;
    int dimension = 0;

    int samplesPerPixel;
    int pixelTileSize;
    pstd::vector<Point2f> *pixelSamples;
};

// RandomSampler Definition
class RandomSampler {
  public:
    // RandomSampler Public Methods
    RandomSampler(int samplesPerPixel, int seed = 0)
        : samplesPerPixel(samplesPerPixel), seed(seed) {}

    static RandomSampler *Create(const ParameterDictionary &parameters,
                                 const FileLoc *loc, Allocator alloc);
    PBRT_CPU_GPU
    static constexpr const char *Name() { return "RandomSampler"; }

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int pixelSample, int dimension) {
        rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
        rng.Advance(pixelSample * 65536 + dimension * 1024);
    }

    PBRT_CPU_GPU
    Float Get1D() { return rng.Uniform<Float>(); }
    PBRT_CPU_GPU
    Point2f Get2D() { return {rng.Uniform<Float>(), rng.Uniform<Float>()}; }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // RandomSampler Private Members
    int samplesPerPixel, seed;
    RNG rng;
};

// SobolSampler Definition
class SobolSampler {
  public:
    // SobolSampler Public Methods
    SobolSampler(int spp, const Point2i &fullResolution,
                 RandomizeStrategy randomizeStrategy)
        : samplesPerPixel(RoundUpPow2(spp)), randomizeStrategy(randomizeStrategy) {
        if (!IsPowerOf2(spp))
            Warning("Non power-of-two sample count rounded up to %d "
                    "for SobolSampler.",
                    samplesPerPixel);
        resolution = RoundUpPow2(std::max(fullResolution.x, fullResolution.y));
    }

    PBRT_CPU_GPU
    static constexpr const char *Name() { return "SobolSampler"; }
    static SobolSampler *Create(const ParameterDictionary &parameters,
                                const Point2i &fullResolution, const FileLoc *loc,
                                Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int sampleIndex, int dim) {
        pixel = p;
        dimension = dim;
        sequenceIndex = SobolIntervalToIndex(Log2Int(resolution), sampleIndex, pixel);
    }

    PBRT_CPU_GPU
    Float Get1D() {
        if (dimension >= NSobolDimensions)
            dimension = 2;
        return sampleDimension(dimension++);
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        if (dimension + 1 >= NSobolDimensions)
            dimension = 2;

        Point2f u(sampleDimension(dimension), sampleDimension(dimension + 1));
        if (dimension == 0) {
            // Remap Sobol$'$ dimensions used for pixel samples
            for (int dim = 0; dim < 2; ++dim) {
                u[dim] = u[dim] * resolution;
                CHECK_RARE(1e-7, u[dim] - pixel[dim] < 0);
                CHECK_RARE(1e-7, u[dim] - pixel[dim] > 1);
                u[dim] = Clamp(u[dim] - pixel[dim], (Float)0, OneMinusEpsilon);
            }
        }
        dimension += 2;
        return u;
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // SobolSampler Private Methods
    PBRT_CPU_GPU
    Float sampleDimension(int dimension) const {
        if (dimension < 2 || randomizeStrategy == RandomizeStrategy::None)
            return SobolSample(sequenceIndex, dimension, NoRandomizer());

        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson) {
            uint32_t hash = MixBits(dimension);
            return SobolSample(sequenceIndex, dimension, CranleyPattersonRotator(hash));
        } else if (randomizeStrategy == RandomizeStrategy::Xor) {
            // Only use the dimension! (Want the same scrambling over all
            // pixels).
            uint32_t hash = MixBits(dimension);
            return SobolSample(sequenceIndex, dimension, XORScrambler(hash));
        } else {
            DCHECK(randomizeStrategy == RandomizeStrategy::Owen);
            uint32_t seed = MixBits(dimension);  // Only dimension!
            return SobolSample(sequenceIndex, dimension, OwenScrambler(seed));
        }
    }

    // SobolSampler Private Members
    int samplesPerPixel;
    int resolution;
    RandomizeStrategy randomizeStrategy;
    Point2i pixel;
    int dimension = 0;
    int64_t sequenceIndex;
};

// StratifiedSampler Definition
class StratifiedSampler {
  public:
    // StratifiedSampler Public Methods
    StratifiedSampler(int xPixelSamples, int yPixelSamples, bool jitter, int seed = 0)
        : xPixelSamples(xPixelSamples),
          yPixelSamples(yPixelSamples),
          jitter(jitter),
          seed(seed) {}

    static StratifiedSampler *Create(const ParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);
    PBRT_CPU_GPU
    static constexpr const char *Name() { return "StratifiedSampler"; }

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return xPixelSamples * yPixelSamples; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index, int dim) {
        pixel = p;
        sampleIndex = index;
        dimension = dim;
        rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
        rng.Advance(sampleIndex * 65536 + dimension * 1024);
    }

    PBRT_CPU_GPU
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        ++dimension;

        int stratum = PermutationElement(sampleIndex, SamplesPerPixel(), hash);
        Float delta = jitter ? rng.Uniform<Float>() : 0.5f;
        return (stratum + delta) / SamplesPerPixel();
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        dimension += 2;

        int stratum = PermutationElement(sampleIndex, SamplesPerPixel(), hash);
        int x = stratum % xPixelSamples;
        int y = stratum / xPixelSamples;
        Float dx = jitter ? rng.Uniform<Float>() : 0.5f;
        Float dy = jitter ? rng.Uniform<Float>() : 0.5f;
        return {(x + dx) / xPixelSamples, (y + dy) / yPixelSamples};
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // StratifiedSampler Private Members
    int xPixelSamples, yPixelSamples;
    bool jitter;
    int seed;
    RNG rng;
    Point2i pixel;
    int sampleIndex = 0;
    int dimension = 0;
};

// MLTSampler Definition
class MLTSampler {
  public:
    // MLTSampler Public Methods
    MLTSampler(int mutationsPerPixel, int rngSequenceIndex, Float sigma,
               Float largeStepProbability, int streamCount)
        : mutationsPerPixel(mutationsPerPixel),
          rng(rngSequenceIndex),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          streamCount(streamCount) {}

    PBRT_CPU_GPU
    void StartIteration();

    PBRT_CPU_GPU
    void Reject();

    PBRT_CPU_GPU
    void StartStream(int index);

    PBRT_CPU_GPU
    int GetNextIndex() { return streamIndex + streamCount * sampleIndex++; }

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return mutationsPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int sampleIndex, int dim) {
        rng.SetSequence(p.x + p.y * 65536);
        rng.Advance(sampleIndex * 65536 + dim * 8192);
    }

    PBRT_CPU_GPU
    Float Get1D();

    PBRT_CPU_GPU
    Point2f Get2D();

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);

    PBRT_CPU_GPU
    void Accept();

    std::string DumpState() const;

    std::string ToString() const {
        return StringPrintf(
            "[ MLTSampler rng: %s sigma: %f largeStepProbability: %f "
            "streamCount: %d X: %s currentIteration: %d largeStep: %s "
            "lastLargeStepIteration: %d streamIndex: %d sampleIndex: %d ] ",
            rng, sigma, largeStepProbability, streamCount, X, currentIteration, largeStep,
            lastLargeStepIteration, streamIndex, sampleIndex);
    }

  protected:
    // MLTSampler Private Declarations
    struct PrimarySample {
        Float value = 0;
        // PrimarySample Public Methods
        PBRT_CPU_GPU
        void Backup() {
            valueBackup = value;
            modifyBackup = lastModificationIteration;
        }
        PBRT_CPU_GPU
        void Restore() {
            value = valueBackup;
            lastModificationIteration = modifyBackup;
        }

        std::string ToString() const {
            return StringPrintf("[ PrimarySample lastModificationIteration: %d "
                                "valueBackup: %f modifyBackup: %d ]",
                                lastModificationIteration, valueBackup, modifyBackup);
        }

        // PrimarySample Public Members
        int64_t lastModificationIteration = 0;
        Float valueBackup = 0;
        int64_t modifyBackup = 0;
    };

    // MLTSampler Private Methods
    PBRT_CPU_GPU
    void EnsureReady(int index);

    // MLTSampler Private Members
    int mutationsPerPixel;
    RNG rng;
    const Float sigma, largeStepProbability;
    const int streamCount;
    pstd::vector<PrimarySample> X;
    int64_t currentIteration = 0;
    bool largeStep = true;
    int64_t lastLargeStepIteration = 0;
    int streamIndex, sampleIndex;
};

class DebugMLTSampler : public MLTSampler {
  public:
    static DebugMLTSampler Create(pstd::span<const std::string> state,
                                  int nSampleStreams);

    PBRT_CPU_GPU
    Float Get1D() {
        int index = GetNextIndex();
        CHECK_LT(index, u.size());
#ifdef PBRT_IS_GPU_CODE
        return 0;
#else
        return u[index];
#endif
    }

    PBRT_CPU_GPU
    Point2f Get2D() { return {Get1D(), Get1D()}; }

    std::string ToString() const {
        return StringPrintf("[ DebugMLTSampler %s u: %s ]",
                            ((const MLTSampler *)this)->ToString(), u);
    }

  private:
    DebugMLTSampler(int nSampleStreams) : MLTSampler(1, 0, 0.5, 0.5, nSampleStreams) {}

    std::vector<Float> u;
};

inline void SamplerHandle::StartPixelSample(const Point2i &p, int sampleIndex,
                                            int dimension) {
    auto start = [&](auto ptr) {
        return ptr->StartPixelSample(p, sampleIndex, dimension);
    };
    return Dispatch(start);
}

inline int SamplerHandle::SamplesPerPixel() const {
    auto spp = [&](auto ptr) { return ptr->SamplesPerPixel(); };
    return Dispatch(spp);
}

inline Float SamplerHandle::Get1D() {
    auto get = [&](auto ptr) { return ptr->Get1D(); };
    return Dispatch(get);
}

inline Point2f SamplerHandle::Get2D() {
    auto get = [&](auto ptr) { return ptr->Get2D(); };
    return Dispatch(get);
}

// Sampler Inline Functions
template <typename Sampler>
inline PBRT_CPU_GPU CameraSample GetCameraSample(Sampler sampler, const Point2i &pPixel,
                                                 FilterHandle filter) {
    FilterSample fs = filter.Sample(sampler.Get2D());
    if (GetOptions().disablePixelJitter) {
        fs.p = Point2f(0, 0);
        fs.weight = 1;
    }

    CameraSample cs;
    cs.pFilm = pPixel + fs.p + Vector2f(0.5, 0.5);
    cs.time = sampler.Get1D();
    cs.pLens = sampler.Get2D();
    cs.weight = fs.weight;
    return cs;
}

}  // namespace pbrt

#endif  // PBRT_SAMPLERS_H
