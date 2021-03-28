// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// pspec.cpp

// Computes power spectra of a variety point sets used by pbrt's samplers.

#include <pbrt/pbrt.h>

#include <pbrt/base/sampler.h>
#include <pbrt/paramdict.h>
#include <pbrt/parser.h>
#include <pbrt/samplers.h>
#include <pbrt/util/args.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/init.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/util/memory.h>
#endif

#include <string>

#ifdef PBRT_BUILD_GPU_RENDERER
namespace pbrt {
void UPSInit(int nPoints);
void UpdatePowerSpectrum(const std::vector<Point2f> &points, Image *pspec);
}  // namespace pbrt
#endif

using namespace pbrt;


static void usage(const std::string &msg = {}) {
    fprintf(stderr, "\n");
    if (!msg.empty()) fprintf(stderr, "pspec: %s\n\n", msg.c_str());

    fprintf(stderr,
            R"(usage: pspec <sampler> [<options...>]

Where <sampler> is one of:
    cwd.pts:         Each file named "pts-*" in the current directory is read to
                     find the sample values for a single point set. Files are
                     plain text and should just be whitespace-separated sample
                     values between 0 and 1.
    grid:            A regular grid of sample points, with floor(sqrt(npoints))
                     samples.
    halton:          The first two dimensions of the Halton sequence.
    halton.owen:     The first two dimensions of the Halton sequence, randomized
                     with Owen scrambling.
    halton.permutedigits:
                     The first two dimensions of the Halton sequence, randomized
                     using random digit permutations.
    independent:     Independent uniform random samples.
    lhs:             Latin hypercube sampling.
    pmj02bn:         Progressive multi-jittered (0,2) blue noise points. (Note:
                     pbrt uses precomputed tables for these and only has five,
                     so nsets > 5 does not make sense in this case.)
    sobol:           The first two dimensions of the Sobol' sequence.
    sobol.fastowen:  The first two dimensions of the Sobol' sequence, randomized
                     using a fast hashing approach that operates on all bits in
                     parallel.
    sobol.owen:      The first two dimensions of the Sobol' sequence, randomized
                     using Owen scrambling.
    sobol.permutedigits:
                     The first two dimensions of the Sobol' sequence, randomized
                     with bitwise permutations.
    sobol.z:         Randomized Morton z-curve Sobol' corresponding to the
                     ZSobolSampler.
    stdin.binary:    Sample values are read from standard input as binary 32-bit
                     floats. Multiple point sets may be provided by providing
                     successive point sets one after the previous.
    stdin.dat:       Sample values are read from standard input as plain text
                     numbers.  Multiple point sets may be provided by separating
                     them with a '#' character.
    stratified:      A grid of stratified sample points, with floor(sqrt(npoints))
                     in each dimension.

Options:
  --npoints <n>        Number of sample points to generate in each set.
                       (Default: 1024).
  --nsets <n>          Number of independent sets of sample points.
                       (Default: 4).
  --outbase <name>     Base filename to use for saving output.  The power
                       spectrum is saved in a filewith the given name and an
                       ".exr" suffix, and the power spectrum is saved with an
                       ".txt" suffix.
                       (Default: <sampler>-<npoints>pts-<nsets>sets.)
  --resolution <res>   Resolution of image for power spectrum. (Default: 1500).

)");
}

static pstd::optional<std::vector<Point2f>>
GenerateSamples(std::string samplerName, int nPoints, int iter) {
    std::vector<Point2f> points;
    points.reserve(nPoints);

    if (samplerName == "stdin.binary") {
        for (int i = 0; i < nPoints; ++i) {
            float s[2];
            size_t n = fread(s, 4, 2, stdin);
            if (n != 2) {
                if (i > 1)
                    ErrorExit("Partial point set provided in standard input: "
                              "have %d points at EOF.", i);
                return {};
            }
            points.push_back(Point2f(s[0], s[1]));
        }
    } else if (samplerName == "stdin.dat") {
        for (int i = 0; i < nPoints; ++i) {
            float s[2];
            int n = scanf("%f %f", &s[0], &s[1]);
            if (n != 2) {
                if (i > 1)
                    ErrorExit("Partial point set provided in standard input: "
                              "have %d points at EOF.", i);
                return {};
            }
            points.push_back(Point2f(s[0], s[1]));
        }
        int ch;
        do {
            ch = getchar();
        } while (ch != '#' && ch != EOF);
    } else if (samplerName == "cwd.pts") {
        static std::vector<std::string> files = MatchingFilenames("pts-");
        static int offset = 0;
        FILE *f = nullptr;
        if (files.empty())
            ErrorExit("No *.dat files found in current directory.");

    retry:
        if (offset == files.size())
            return {};

        f = fopen(files[offset].c_str(), "r");
        if (!f)
            ErrorExit("%s: unable to open file", files[offset]);

        for (int i = 0; i < nPoints; ++i) {
            float s[2];
            int n = fscanf(f, "%f %f", &s[0], &s[1]);
            if (n < 2) {
                Warning("%s: premature EOF. Read %d / %d points. Ignoring file.",
                        files[iter], i, nPoints);
                ++offset;
                fclose(f);
                points.clear();
                goto retry;
            }
            points.push_back(Point2f(s[0], s[1]));
        }
        ++offset;
        fclose(f);
    } else if (samplerName == "grid") {
        int sqrtSamples = std::sqrt(nPoints);
        nPoints = Sqr(sqrtSamples);

        for (int i = 0; i < sqrtSamples; ++i)
            for (int j = 0; j < sqrtSamples; ++j)
                points.push_back(
                    Point2f(Float(i) / sqrtSamples, Float(j) / sqrtSamples));
    } else if (samplerName == "lhs") {
        RNG rng(Options->seed, iter);
        // Sample points along the diagonal
        for (int i = 0; i < nPoints; ++i)
            points.push_back(Point2f((i + rng.Uniform<Float>()) / nPoints,
                                     (i + rng.Uniform<Float>()) / nPoints));
        // Suffle x
        for (int i = 0; i < nPoints; ++i) {
            int other = i + rng.Uniform<uint32_t>(nPoints - i);
            std::swap(points[i].x, points[other].x);
        }
    } else if (samplerName == "halton") {
        for (int i = 0; i < nPoints; ++i)
            points.push_back(
                Point2f(RadicalInverse(0, i), RadicalInverse(1, i)));
    } else if (samplerName == "halton.permutedigits") {
        RNG rng(Options->seed, iter);
        DigitPermutation perm2(2, rng.Uniform<uint32_t>(), {});
        DigitPermutation perm3(3, rng.Uniform<uint32_t>(), {});

        for (int i = 0; i < nPoints; ++i)
            points.push_back(Point2f(ScrambledRadicalInverse(0, i, perm2),
                                     ScrambledRadicalInverse(1, i, perm3)));
    } else if (samplerName == "halton.owen") {
        RNG rng(Options->seed, iter);
        uint32_t r[2] = {rng.Uniform<uint32_t>(), rng.Uniform<uint32_t>()};

        for (int i = 0; i < nPoints; ++i)
            points.push_back(Point2f(SobolSample(i, 0, OwenScrambler(r[0])),
                                     OwenScrambledRadicalInverse(1, i, r[1])));
    } else if (samplerName == "sobol.z") {
        if (!IsPowerOf2(nPoints))
            ErrorExit("Must use power of 2 points for \"sobol.z\".");

        int n = 1 << (Log2Int(nPoints) / 2);
        int spp = nPoints / Sqr(n);
        ZSobolSampler sampler(spp, {n, n}, RandomizeStrategy::Owen, Options->seed);

        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                sampler.StartPixelSample({x, y}, 0, 0);
                for (int s = 0; s < spp; ++s) {
                    Point2f u = sampler.Get2D();
                    points.push_back(Point2f((x + u[0]) / n, (y + u[1]) / n));
                }
            }
    } else {
        Sampler sampler = [&]() -> Sampler {
            if (samplerName == "independent")
                return new IndependentSampler(nPoints, Options->seed);
            else if (samplerName == "stratified") {
                int sqrtSamples = std::sqrt(nPoints);
                nPoints = Sqr(sqrtSamples);
                return new StratifiedSampler(sqrtSamples, sqrtSamples, true,
                                             Options->seed);
            } else if (samplerName == "pmj02bn") {
                return new PMJ02BNSampler(nPoints, Options->seed);
            } else if (samplerName == "sobol") {
                return new PaddedSobolSampler(nPoints, RandomizeStrategy::None, Options->seed);
            } else if (samplerName == "sobol.permutedigits") {
                return new PaddedSobolSampler(nPoints, RandomizeStrategy::PermuteDigits,
                                              Options->seed);
            } else if (samplerName == "sobol.fastowen") {
                return new PaddedSobolSampler(nPoints, RandomizeStrategy::FastOwen,
                                              Options->seed);
            } else if (samplerName == "sobol.owen") {
                return new PaddedSobolSampler(nPoints, RandomizeStrategy::Owen,
                                              Options->seed);
            } else {
                usage(StringPrintf("%s: sampler unknown", samplerName));
                exit(1);
            }
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
    int nSets = 0;
    int res = 1500;
    std::string baseOutFilename;

    argv += 1;
    while (*argv != nullptr) {
        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        if (ParseArg(&argv, "npoints", &nPoints, onError) ||
            ParseArg(&argv, "resolution", &res, onError) ||
            ParseArg(&argv, "outbase", &baseOutFilename, onError) ||
            ParseArg(&argv, "nsets", &nSets, onError))
            ;
        else if (samplerName.empty()) {
            if (*argv[0] == '-') {
                usage();
                return 0;
            }
            samplerName = *argv++;
        } else {
            usage(StringPrintf("unknown argument \"%s\"", *argv));
            return 1;
        }
    }

    if (samplerName.empty()) {
        usage("Must specify name of sampler.\n");
        return 1;
    }

    if (!(res & 1)) ++res;

    PBRTOptions options;
    options.quiet = true;
#ifdef PBRT_BUILD_GPU_RENDERER
    options.useGPU = true;
#endif
    InitPBRT(options);

#ifdef PBRT_BUILD_GPU_RENDERER
    CUDAMemoryResource *memoryResource = new CUDAMemoryResource;
#else
    pstd::pmr::memory_resource *memoryResource =
        pstd::pmr::get_default_resource();
#endif
    Allocator alloc(memoryResource);

    Image *pspec = alloc.new_object<Image>(
        PixelFormat::Float, Point2i(res, res),
        std::vector<std::string>{"power"}, nullptr, alloc);
    ProgressReporter progress(nSets, "Analyzing", nSets == 1, options.useGPU);

#ifdef PBRT_BUILD_GPU_RENDERER
    GPUInit();
    UPSInit(nPoints);
#endif

    int actualNSets = 0;
    while (true) {
        Options->seed = MixBits(actualNSets);

        // Generate points
        pstd::optional<std::vector<Point2f>> points = GenerateSamples(samplerName, nPoints, actualNSets);
        if (!points)
            break;
        ++actualNSets;

#ifdef PBRT_BUILD_GPU_RENDERER
        UpdatePowerSpectrum(*points, pspec);
#else
        // Fourier transform
        ParallelFor(0, res, [&](int y) {
            for (int x = 0; x < res; ++x) {
                Point2f uv(0, 0);
                Float wx = x - res / 2, wy = y - res / 2;
                for (Point2f u : *points) {
                    Float exp = -2 * Pi * (wx * u[0] + wy * u[1]);
                    uv[0] += std::cos(exp);
                    uv[1] += std::sin(exp);
                }

                // Update power spectrum
                pspec->SetChannel(
                    {x, y}, 0,
                    (pspec->GetChannel({x, y}, 0) + (Sqr(uv[0]) + Sqr(uv[1]))));
            }
        });
#endif

        progress.Update();

        if (nSets != 0 && actualNSets == nSets)
            break;
    }

#ifdef PBRT_BUILD_GPU_RENDERER
    GPUWait();
#endif  // PBRT_BUILD_GPU_RENDERER

    ParallelFor(0, res, [&](int y) {
        for (int x = 0; x < res; ++x) {
            // Early float cast so no integer overflow...
            pspec->SetChannel({x, y}, 0,
                              (pspec->GetChannel({x, y}, 0) /
                               (Float(nPoints) * Float(actualNSets))));
        }
    });

    progress.Done();

    if (baseOutFilename.empty())
        baseOutFilename =
            StringPrintf("%s-%dpts-%dsets", samplerName, nPoints, actualNSets);

    pspec->Write(baseOutFilename + ".exr");

    // Compute radial average
    std::vector<Float> sumPower(res / 2, Float(0));
    std::vector<int> nPower(res / 2, 0);
    for (int y = 0; y < res; ++y)
        for (int x = 0; x < res; ++x) {
            int dx = std::abs(x - res / 2), dy = std::abs(y - res / 2);
            if (dx == 0 && dy == 0)
                // skip the central spike
                continue;
            int bucket = std::sqrt(dx*dx + dy*dy);
            if (bucket >= sumPower.size())
                continue;
            sumPower[bucket] += pspec->GetChannel({x, y}, 0);
            ++nPower[bucket];
        }

    FILE *f = fopen((baseOutFilename + ".txt").c_str(), "w");
    if (!f)
        ErrorExit("%s: could not open output file", baseOutFilename + ".txt");
    for (int i = 1; i < res / 2; ++i)
        fprintf(f, "%d %f\n", i, sumPower[i] / nPower[i]);
    fclose(f);

    CleanupPBRT();

    return 0;
}
