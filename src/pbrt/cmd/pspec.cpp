// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// pspec.cpp

// Computes power spectra of a variety point sets used by pbrt's samplers.

#include <pbrt/pbrt.h>

#include <pbrt/base/sampler.h>
#include <pbrt/parser.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/util/args.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/image.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>

#include <string>

using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pspec: %s\n\n", msg.c_str());

    fprintf(stderr,
            R"(usage: pspec <sampler> [<options...>]

Where <sampler> is one of: stdin, halton, random, stratified, pmj02bn, sobol,
                           sobol.xor, sobol.owen, sobol.realowen

Options:
  --npoints <n>        Number of sample points to generate in each set.
                       (Default: 1024).
  --nsets <n>          Number of independent sets of sample points.
                       (Default: 4).
  --outfile <name>     Filename to use for saving the image of the power
                       spectrum. (Default: <sampler>.exr.)
  --resolution <res>   Resolution of image for power spectrum. (Default: 513).
  --seed <s>           Seed to use for randomizing samples. (Default: 0).

)");
}

static std::vector<Point2f> GenerateSamples(std::string samplerName, int nPoints, int iter) {
    std::vector<Point2f> points;
    points.reserve(nPoints);

    if (samplerName == "stdin") {
        for (int i = 0; i < nPoints; ++i) {
            float s[2];
            fread(s, 4, 2, stdin);
            points.push_back(Point2f(s[0], s[1]));
        }
    } else if (samplerName == "halton") {
        RNG rng(Options->seed, iter);
        DigitPermutation perm2(2, rng.Uniform<uint32_t>(), {});
        DigitPermutation perm3(3, rng.Uniform<uint32_t>(), {});

        for (int i = 0; i < nPoints; ++i)
            points.push_back(Point2f(ScrambledRadicalInverse(0, i, perm2),
                                     ScrambledRadicalInverse(1, i, perm3)));
    } else if (samplerName == "sobol.realowen") {
        RNG rng(Options->seed, iter);
        uint32_t r[2] = { rng.Uniform<uint32_t>(), rng.Uniform<uint32_t>() };
        for (int i = 0; i < nPoints; ++i) {
            Point2f u(SobolSample(i, 0, NoRandomizer()), SobolSample(i, 1, NoRandomizer()));
            uint32_t uu[2] = { uint32_t(u[0] * 0x1p32), uint32_t(u[1] * 0x1p32) };

            if (r[0] & 1)
                uu[0] ^= 1u << 31;
            if (r[1] & 1)
                uu[1] ^= 1u << 31;

            for (int b = 1; b < 32; ++b) {
                uint32_t mask = (~0u) << (32-b);
                if (MixBits((uu[0] & mask) ^ r[0]) & 1)
                    uu[0] ^= 1u << (31 - b);
                if (MixBits((uu[1] & mask) ^ r[1]) & 1)
                    uu[1] ^= 1u << (31 - b);
            }

            u[0] = uu[0] * 0x1p-32;
            u[1] = uu[1] * 0x1p-32;

            points.push_back(u);
        }
    } else {
        SamplerHandle sampler = [&]() -> SamplerHandle {
            if (samplerName == "random")
                return new RandomSampler(nPoints, Options->seed);
            else if (samplerName == "stratified") {
                int sqrtSamples = std::sqrt(nPoints);
                nPoints = Sqr(sqrtSamples);
                return new StratifiedSampler(sqrtSamples, sqrtSamples, true, Options->seed);
            } else if (samplerName == "pmj02bn") {
                return new PMJ02BNSampler(nPoints, Options->seed);
            } else if (samplerName == "sobol") {
                return new PaddedSobolSampler(nPoints, RandomizeStrategy::None);
            } else if (samplerName == "sobol.xor") {
                return new PaddedSobolSampler(nPoints, RandomizeStrategy::XOR);
            } else if (samplerName == "sobol.owen") {
                return new PaddedSobolSampler(nPoints, RandomizeStrategy::Owen);
            } else
                ErrorExit("%s: sampler unknown", samplerName);
        }();

        for (int i = 0; i < nPoints; ++i) {
            sampler.StartPixelSample(Point2i(0, 0), i, 0);
            Point2f u = sampler.Get2D();
            points.push_back(u);
        }

        sampler.DispatchCPU([&](auto sampler) { delete sampler; });
    }

    return points;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string samplerName;
    int nPoints = 1024;
    int seed = 0;
    int nSets = 4;
    int res = 513;
    std::string outFilename;

    argv += 1;
    while (*argv != nullptr) {
        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        if (ParseArg(&argv, "npoints", &nPoints, onError) ||
            ParseArg(&argv, "seed", &seed, onError) ||
            ParseArg(&argv, "resolution", &res, onError) ||
            ParseArg(&argv, "outfile", &outFilename, onError) ||
            ParseArg(&argv, "nsets", &nSets, onError))
            ;
        else if (samplerName.empty())
            samplerName = *argv++;
        else {
            usage(StringPrintf("unknown argument \"%s\"", *argv));
            return 1;
        }
    }

    if (samplerName.empty()) {
        usage("Must specify name of sampler.\n");
        return 1;
    }

    if (outFilename.empty())
        outFilename = StringPrintf("%s.exr", samplerName);

    if (!(res & 1))
        ++res;

    PBRTOptions options;
    options.quiet = true;
    InitPBRT(options);

    Allocator alloc;
    RNG rng;
    Image pspec(PixelFormat::Float, {res, res}, {"power"});

    ProgressReporter progress(nSets, "Analyzing", nSets == 1);

    for (int i = 0; i < nSets; ++i) {
        Options->seed = MixBits(seed + i);

        // Generate points
        std::vector<Point2f> points = GenerateSamples(samplerName, nPoints, i);

        // Fourier transform
        ParallelFor(0, res, [&](int y) {
            for (int x = 0; x < res; ++x) {
                Point2f uv(0, 0);
                Float wx = x - res / 2, wy = y - res / 2;
                for (Point2f u : points) {
                    Float exp = -2 * Pi * (wx * u[0] + wy * u[1]);
                    uv[0] += std::cos(exp);
                    uv[1] += std::sin(exp);
                }

                // Update power spectrum
                pspec.SetChannel({x, y}, 0, (pspec.GetChannel({x, y}, 0) +
                                             (Sqr(uv[0]) + Sqr(uv[1])) / (nPoints * nSets)));
            }
        });

        progress.Update();
    }
    progress.Done();

    pspec.Write(outFilename);

    CleanupPBRT();

    return 0;
}
