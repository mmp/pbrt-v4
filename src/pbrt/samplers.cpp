// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/samplers.h>

#include <pbrt/cameras.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/string.hpp>

#include <string>

namespace pbrt {

Sampler Sampler::Clone(Allocator alloc) {
    auto clone = [&](auto ptr) { return ptr->Clone(alloc); };
    return DispatchCPU(clone);
}

std::string Sampler::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

// HaltonSampler Method Definitions
HaltonSampler::HaltonSampler(int samplesPerPixel, Point2i fullRes,
                             RandomizeStrategy randomize, int seed, Allocator alloc)
    : samplesPerPixel(samplesPerPixel), randomize(randomize) {
    if (randomize == RandomizeStrategy::PermuteDigits)
        digitPermutations = ComputeRadicalInversePermutations(seed, alloc);
    // Find radical inverse base scales and exponents that cover sampling area
    for (int i = 0; i < 2; ++i) {
        int base = (i == 0) ? 2 : 3;
        int scale = 1, exp = 0;
        while (scale < std::min(fullRes[i], MaxHaltonResolution)) {
            scale *= base;
            ++exp;
        }
        baseScales[i] = scale;
        baseExponents[i] = exp;
    }

    // Compute multiplicative inverses for _baseScales_
    multInverse[0] = multiplicativeInverse(baseScales[1], baseScales[0]);
    multInverse[1] = multiplicativeInverse(baseScales[0], baseScales[1]);
}

Sampler HaltonSampler::Clone(Allocator alloc) {
    return alloc.new_object<HaltonSampler>(*this);
}

std::string HaltonSampler::ToString() const {
    return StringPrintf("[ HaltonSampler randomize: %s digitPermutations: %p "
                        "haltonIndex: %d dimension: %d samplesPerPixel: %d "
                        "baseScales: %s baseExponents: %s multInverse: [ %d %d ] ]",
                        randomize, digitPermutations, haltonIndex, dimension,
                        samplesPerPixel, baseScales, baseExponents, multInverse[0],
                        multInverse[1]);
}

HaltonSampler *HaltonSampler::Create(const ParameterDictionary &parameters,
                                     Point2i fullResolution, const FileLoc *loc,
                                     Allocator alloc) {
    int nsamp = parameters.GetOneInt("pixelsamples", 16);
    if (Options->pixelSamples)
        nsamp = *Options->pixelSamples;
    int seed = parameters.GetOneInt("seed", Options->seed);
    if (Options->quickRender)
        nsamp = 1;

    RandomizeStrategy randomizer;
    std::string s = parameters.GetOneString("randomization", "permutedigits");
    if (s == "none")
        randomizer = RandomizeStrategy::None;
    else if (s == "permutedigits")
        randomizer = RandomizeStrategy::PermuteDigits;
    else if (s == "fastowen")
        ErrorExit("%s: \"fastowen\" randomization not supported by Halton sampler.");
    else if (s == "owen")
        randomizer = RandomizeStrategy::Owen;
    else
        ErrorExit(loc, "%s: unknown randomization strategy given to HaltonSampler", s);

    return alloc.new_object<HaltonSampler>(nsamp, fullResolution, randomizer, seed,
                                           alloc);
}

Sampler SobolSampler::Clone(Allocator alloc) {
    return alloc.new_object<SobolSampler>(*this);
}

std::string PaddedSobolSampler::ToString() const {
    return StringPrintf("[ PaddedSobolSampler pixel: %s sampleIndex: %d dimension: %d "
                        "samplesPerPixel: %d seed: %d randomize: %s ]",
                        pixel, sampleIndex, dimension, samplesPerPixel, seed, randomize);
}

Sampler PaddedSobolSampler::Clone(Allocator alloc) {
    return alloc.new_object<PaddedSobolSampler>(*this);
}

PaddedSobolSampler *PaddedSobolSampler::Create(const ParameterDictionary &parameters,
                                               const FileLoc *loc, Allocator alloc) {
    int nsamp = parameters.GetOneInt("pixelsamples", 16);
    if (Options->pixelSamples)
        nsamp = *Options->pixelSamples;
    if (Options->quickRender)
        nsamp = 1;
    int seed = parameters.GetOneInt("seed", Options->seed);

    RandomizeStrategy randomizer;
    std::string s = parameters.GetOneString("randomization", "fastowen");
    if (s == "none")
        randomizer = RandomizeStrategy::None;
    else if (s == "permutedigits")
        randomizer = RandomizeStrategy::PermuteDigits;
    else if (s == "fastowen")
        randomizer = RandomizeStrategy::FastOwen;
    else if (s == "owen")
        randomizer = RandomizeStrategy::Owen;
    else
        ErrorExit(loc, "%s: unknown randomization strategy given to PaddedSobolSampler",
                  s);

    return alloc.new_object<PaddedSobolSampler>(nsamp, randomizer, seed);
}

// ZSobolSampler Method Definitions
Sampler ZSobolSampler::Clone(Allocator alloc) {
    return alloc.new_object<ZSobolSampler>(*this);
}

std::string ZSobolSampler::ToString() const {
    return StringPrintf("[ ZSobolSampler randomize: %s log2SamplesPerPixel: %d "
                        " seed: %d nBase4Digits: %d mortonIndex: %d dimension: %d ]",
                        randomize, log2SamplesPerPixel, seed, nBase4Digits, mortonIndex,
                        dimension);
}

ZSobolSampler *ZSobolSampler::Create(const ParameterDictionary &parameters,
                                     Point2i fullResolution, const FileLoc *loc,
                                     Allocator alloc) {
    int nsamp = parameters.GetOneInt("pixelsamples", 16);
    if (Options->pixelSamples)
        nsamp = *Options->pixelSamples;
    if (Options->quickRender)
        nsamp = 1;
    int seed = parameters.GetOneInt("seed", Options->seed);

    RandomizeStrategy randomizer;
    std::string s = parameters.GetOneString("randomization", "fastowen");
    if (s == "none")
        randomizer = RandomizeStrategy::None;
    else if (s == "permutedigits")
        randomizer = RandomizeStrategy::PermuteDigits;
    else if (s == "fastowen")
        randomizer = RandomizeStrategy::FastOwen;
    else if (s == "owen")
        randomizer = RandomizeStrategy::Owen;
    else
        ErrorExit(loc, "%s: unknown randomization strategy given to ZSobolSampler", s);

    return alloc.new_object<ZSobolSampler>(nsamp, fullResolution, randomizer, seed);
}

// PMJ02BNSampler Method Definitions
PMJ02BNSampler::PMJ02BNSampler(int samplesPerPixel, int seed, Allocator alloc)
    : samplesPerPixel(samplesPerPixel), seed(seed) {
    if (!IsPowerOf4(samplesPerPixel))
        Warning("PMJ02BNSampler results are best with power-of-4 samples per "
                "pixel (1, 4, 16, 64, ...)");
    // Get sorted pmj02bn samples for pixel samples
    if (samplesPerPixel > nPMJ02bnSamples)
        Error("PMJ02BNSampler only supports up to %d samples per pixel", nPMJ02bnSamples);
    // Compute _pixelTileSize_ for pmj02bn pixel samples and allocate _pixelSamples_
    pixelTileSize =
        1 << (Log4Int(nPMJ02bnSamples) - Log4Int(RoundUpPow4(samplesPerPixel)));
    int nPixelSamples = pixelTileSize * pixelTileSize * samplesPerPixel;
    pixelSamples = alloc.new_object<pstd::vector<Point2f>>(nPixelSamples, alloc);

    // Loop over pmj02bn samples and associate them with their pixels
    std::vector<int> nStored(pixelTileSize * pixelTileSize, 0);
    for (int i = 0; i < nPMJ02bnSamples; ++i) {
        Point2f p = GetPMJ02BNSample(0, i);
        p *= pixelTileSize;
        int pixelOffset = int(p.x) + int(p.y) * pixelTileSize;
        if (nStored[pixelOffset] == samplesPerPixel) {
            CHECK(!IsPowerOf4(samplesPerPixel));
            continue;
        }
        int sampleOffset = pixelOffset * samplesPerPixel + nStored[pixelOffset];
        CHECK((*pixelSamples)[sampleOffset] == Point2f(0, 0));
        (*pixelSamples)[sampleOffset] = Point2f(p - Floor(p));
        ++nStored[pixelOffset];
    }

    for (int i = 0; i < nStored.size(); ++i)
        CHECK_EQ(nStored[i], samplesPerPixel);
    for (int c : nStored)
        DCHECK_EQ(c, samplesPerPixel);
}

PMJ02BNSampler *PMJ02BNSampler::Create(const ParameterDictionary &parameters,
                                       const FileLoc *loc, Allocator alloc) {
    int nsamp = parameters.GetOneInt("pixelsamples", 16);
    if (Options->pixelSamples)
        nsamp = *Options->pixelSamples;
    if (Options->quickRender)
        nsamp = 1;
    int seed = parameters.GetOneInt("seed", Options->seed);
    return alloc.new_object<PMJ02BNSampler>(nsamp, seed, alloc);
}

Sampler PMJ02BNSampler::Clone(Allocator alloc) {
    return alloc.new_object<PMJ02BNSampler>(*this);
}

std::string PMJ02BNSampler::ToString() const {
    return StringPrintf("[ PMJ02BNSampler pixel: %s sampleIndex: %d dimension: %d "
                        "samplesPerPixel: %d pixelTileSize: %d pixelSamples: %p ]",
                        pixel, sampleIndex, dimension, samplesPerPixel, pixelTileSize,
                        pixelSamples);
}

std::string IndependentSampler::ToString() const {
    return StringPrintf("[ IndependentSampler samplesPerPixel: %d seed: %d rng: %s ]",
                        samplesPerPixel, seed, rng);
}

Sampler IndependentSampler::Clone(Allocator alloc) {
    return alloc.new_object<IndependentSampler>(*this);
}

IndependentSampler *IndependentSampler::Create(const ParameterDictionary &parameters,
                                               const FileLoc *loc, Allocator alloc) {
    int ns = parameters.GetOneInt("pixelsamples", 4);
    if (Options->pixelSamples)
        ns = *Options->pixelSamples;
    int seed = parameters.GetOneInt("seed", Options->seed);
    return alloc.new_object<IndependentSampler>(ns, seed);
}

// SobolSampler Method Definitions
std::string SobolSampler::ToString() const {
    return StringPrintf("[ SobolSampler pixel: %s dimension: %d "
                        "samplesPerPixel: %d scale: %d sobolIndex: %d "
                        "seed: %d randomize: %s ]",
                        pixel, dimension, samplesPerPixel, scale, sobolIndex, seed,
                        randomize);
}

SobolSampler *SobolSampler::Create(const ParameterDictionary &parameters,
                                   Point2i fullResolution, const FileLoc *loc,
                                   Allocator alloc) {
    int nsamp = parameters.GetOneInt("pixelsamples", 16);
    if (Options->pixelSamples)
        nsamp = *Options->pixelSamples;
    if (Options->quickRender)
        nsamp = 1;

    RandomizeStrategy randomizer;
    std::string s = parameters.GetOneString("randomization", "fastowen");
    if (s == "none")
        randomizer = RandomizeStrategy::None;
    else if (s == "permutedigits")
        randomizer = RandomizeStrategy::PermuteDigits;
    else if (s == "fastowen")
        randomizer = RandomizeStrategy::FastOwen;
    else if (s == "owen")
        randomizer = RandomizeStrategy::Owen;
    else
        ErrorExit(loc, "%s: unknown randomization strategy given to SobolSampler", s);

    int seed = parameters.GetOneInt("seed", Options->seed);

    return alloc.new_object<SobolSampler>(nsamp, fullResolution, randomizer, seed);
}

// StratifiedSampler Method Definitions
std::string StratifiedSampler::ToString() const {
    return StringPrintf(
        "[ StratifiedSampler pixel: %s sampleIndex: %d dimension: %d "
        "xPixelSamples: %d yPixelSamples: %d jitter: %s seed: %d rng: %s ]",
        pixel, sampleIndex, dimension, xPixelSamples, yPixelSamples, jitter, seed, rng);
}

Sampler StratifiedSampler::Clone(Allocator alloc) {
    return alloc.new_object<StratifiedSampler>(*this);
}

StratifiedSampler *StratifiedSampler::Create(const ParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc) {
    bool jitter = parameters.GetOneBool("jitter", true);
    int xSamples = parameters.GetOneInt("xsamples", 4);
    int ySamples = parameters.GetOneInt("ysamples", 4);
    if (Options->pixelSamples) {
        int nSamples = *Options->pixelSamples;
        int div = std::sqrt(nSamples);
        while (nSamples % div) {
            CHECK_GT(div, 0);
            --div;
        }
        xSamples = nSamples / div;
        ySamples = nSamples / xSamples;
        CHECK_EQ(nSamples, xSamples * ySamples);
        LOG_VERBOSE("xSamples %d ySamples %d", xSamples, ySamples);
    }
    if (Options->quickRender)
        xSamples = ySamples = 1;
    int seed = parameters.GetOneInt("seed", Options->seed);

    return alloc.new_object<StratifiedSampler>(xSamples, ySamples, jitter, seed);
}

// MLTSampler Method Definitions
Float MLTSampler::Get1D() {
    int index = GetNextIndex();
    EnsureReady(index);
    return X[index].value;
}

Point2f MLTSampler::Get2D() {
    return {Get1D(), Get1D()};
}
Point2f MLTSampler::GetPixel2D() {
    return Get2D();
}

Sampler MLTSampler::Clone(Allocator alloc) {
    LOG_FATAL("MLTSampler::Clone() is not implemented");
    return {};
}

void MLTSampler::StartIteration() {
    currentIteration++;
    largeStep = rng.Uniform<Float>() < largeStepProbability;
}

void MLTSampler::Accept() {
    if (largeStep)
        lastLargeStepIteration = currentIteration;
}

void MLTSampler::EnsureReady(int index) {
#ifdef PBRT_IS_GPU_CODE
    LOG_FATAL("MLTSampler not supported on GPU--needs vector resize...");
    return;
#else
    // Enlarge _MLTSampler::X_ if necessary and get current $\VEC{X}_i$
    if (index >= X.size())
        X.resize(index + 1);
    PrimarySample &X_i = X[index];

    // Reset $\VEC{X}_i$ if a large step took place in the meantime
    if (X_i.lastModificationIteration < lastLargeStepIteration) {
        X_i.value = rng.Uniform<Float>();
        X_i.lastModificationIteration = lastLargeStepIteration;
    }

    // Apply remaining sequence of mutations to _sample_
    X_i.Backup();
    if (largeStep)
        X_i.value = rng.Uniform<Float>();
    else {
        int64_t nSmall = currentIteration - X_i.lastModificationIteration;
        // Apply _nSmall_ small step mutations to $\VEC{X}_i$
        Float effSigma = sigma * std::sqrt((Float)nSmall);
        Float delta = SampleNormal(rng.Uniform<Float>(), 0, effSigma);
        X_i.value += delta;
        X_i.value -= pstd::floor(X_i.value);
    }
    X_i.lastModificationIteration = currentIteration;

#endif
}

void MLTSampler::Reject() {
    for (auto &X_i : X)
        if (X_i.lastModificationIteration == currentIteration)
            X_i.Restore();
    --currentIteration;
}

void MLTSampler::StartStream(int index) {
    DCHECK_LT(index, streamCount);
    streamIndex = index;
    sampleIndex = 0;
}

std::string MLTSampler::DumpState() const {
    std::string state;
    for (const PrimarySample &Xi : X)
        state += StringPrintf("%f,", Xi.value);
    state += "0";
    return state;
}

DebugMLTSampler DebugMLTSampler::Create(pstd::span<const std::string> state,
                                        int nSampleStreams) {
    DebugMLTSampler ds(nSampleStreams);
    ds.u.resize(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        if (!Atof(state[i], &ds.u[i]))
            ErrorExit("Invalid value in --debugstate: %s", state[i]);
    }
    return ds;
}

// Sampler Method Definitions
Sampler Sampler::Create(const std::string &name, const ParameterDictionary &parameters,
                        Point2i fullRes, const FileLoc *loc, Allocator alloc) {
    Sampler sampler = nullptr;
    if (name == "zsobol")
        sampler = ZSobolSampler::Create(parameters, fullRes, loc, alloc);
    // Create remainder of _Sampler_ types
    else if (name == "paddedsobol")
        sampler = PaddedSobolSampler::Create(parameters, loc, alloc);
    else if (name == "halton")
        sampler = HaltonSampler::Create(parameters, fullRes, loc, alloc);
    else if (name == "sobol")
        sampler = SobolSampler::Create(parameters, fullRes, loc, alloc);
    else if (name == "pmj02bn")
        sampler = PMJ02BNSampler::Create(parameters, loc, alloc);
    else if (name == "independent")
        sampler = IndependentSampler::Create(parameters, loc, alloc);
    else if (name == "stratified")
        sampler = StratifiedSampler::Create(parameters, loc, alloc);
    else
        ErrorExit(loc, "%s: sampler type unknown.", name);
    if (!sampler)
        ErrorExit(loc, "%s: unable to create sampler.", name);
    parameters.ReportUnused();

    return sampler;
}

}  // namespace pbrt
