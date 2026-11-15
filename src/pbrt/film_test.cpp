// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/pbrt.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/image.h>
#include <pbrt/util/spectrum.h>

using namespace pbrt;

// Radiance arriving at a SpectralFilm pixel may be provided via AddSample()
// (camera paths), AddSplat() (light paths, as in the LightPathIntegrator),
// or a mix of the two (BDPT). Check that the final RGB and spectral bucket
// values are the same regardless of how the radiance is split between them.
// (https://github.com/mmp/pbrt-v4/issues/558)
TEST(SpectralFilm, SplatMatchesSample) {
    const Float lambdaMin = 400, lambdaMax = 700;
    // Radiance that varies linearly with wavelength, so that its average
    // over a bucket is its value at the bucket's center.
    auto Lfunc = [&](Float lambda) {
        return 1 + (lambda - lambdaMin) / (lambdaMax - lambdaMin);
    };

    BoxFilter boxFilter(Vector2f(0.5, 0.5));
    Point2i resolution(1, 1);
    FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution), &boxFilter,
                          1., PixelSensor::CreateDefault(), "test.exr");

    // Returns the final image for a single-pixel film where
    // |sampleFraction| of the radiance is provided via AddSample() and the
    // rest via AddSplat().
    auto render = [&](int nBuckets, Float sampleFraction) {
        SpectralFilm film(fp, lambdaMin, lambdaMax, nBuckets, RGBColorSpace::sRGB,
                          Infinity, false /* writeFP16 */);

        // Stratify the wavelength samples so that every bucket receives the
        // same number of them, evenly spaced over the bucket. Both estimates
        // are then exact (up to floating-point error) for the linear Lfunc.
        int nSamples = 64 * nBuckets * NSpectrumSamples;
        for (int i = 0; i < nSamples; ++i) {
            SampledWavelengths lambda = film.SampleWavelengths((i + 0.5f) / nSamples);
            SampledSpectrum L;
            for (int j = 0; j < NSpectrumSamples; ++j)
                L[j] = Lfunc(lambda[j]);

            if (sampleFraction > 0)
                film.AddSample(Point2i(0, 0), sampleFraction * L, lambda, nullptr, 1);
            if (sampleFraction < 1)
                film.AddSplat(Point2f(0.5, 0.5), (1 - sampleFraction) * L, lambda);
        }

        ImageMetadata metadata;
        return film.GetImage(&metadata, 1.f / nSamples);
    };

    for (int nBuckets : {1, 15, 30, 64}) {
        Image ref = render(nBuckets, 1);

        for (Float sampleFraction : {1.f, 0.5f, 0.f}) {
            Image image = render(nBuckets, sampleFraction);
            ASSERT_EQ(3 + nBuckets, image.NChannels());

            // RGB should match the AddSample()-only film.
            for (int c = 0; c < 3; ++c) {
                Float v = image.GetChannel(Point2i(0, 0), c);
                Float vRef = ref.GetChannel(Point2i(0, 0), c);
                EXPECT_NEAR(vRef, v, 1e-3f * vRef)
                    << "nBuckets " << nBuckets << ", sampleFraction " << sampleFraction
                    << ", RGB channel " << c;
            }

            // Spectral buckets store the bucket's average radiance, scaled
            // by CIE_Y_integral (see SpectralFilm::AddSample()).
            for (int b = 0; b < nBuckets; ++b) {
                Float lambdaCenter = Lerp((b + 0.5f) / nBuckets, lambdaMin, lambdaMax);
                Float expected = CIE_Y_integral * Lfunc(lambdaCenter);
                Float v = image.GetChannel(Point2i(0, 0), 3 + b);
                EXPECT_NEAR(expected, v, 1e-3f * expected)
                    << "nBuckets " << nBuckets << ", sampleFraction " << sampleFraction
                    << ", bucket " << b;
            }
        }
    }
}

// When a path's secondary wavelengths have been terminated (e.g., due to
// dispersion), only lambda[0] carries a valid radiance estimate. Check that
// SpectralFilm ignores the radiance values at the terminated wavelengths,
// including when clamping, and that it weights lambda[0]'s value correctly
// relative to non-terminated samples in the spectral bucket averages.
// (https://github.com/mmp/pbrt-v4/pull/550)
TEST(SpectralFilm, TerminatedSecondaryWavelengths) {
    const Float lambdaMin = 400, lambdaMax = 700;
    // Radiance for samples with and without terminated secondary
    // wavelengths, respectively. Both are linear in wavelength, so that
    // their averages over a bucket are their values at the bucket's center.
    auto t = [&](Float lambda) { return (lambda - lambdaMin) / (lambdaMax - lambdaMin); };
    auto LTerminated = [&](Float lambda) { return 2 + 2 * t(lambda); };
    auto LFull = [&](Float lambda) { return 1 - 0.5f * t(lambda); };
    // Radiance at terminated wavelengths; it should have no effect.
    const Float LIgnored = 1000;

    BoxFilter boxFilter(Vector2f(0.5, 0.5));
    Point2i resolution(1, 1);
    FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution), &boxFilter,
                          1., PixelSensor::CreateDefault(), "test.exr");

    // A finite maxComponentValue that is larger than all of the valid
    // radiance values should not cause any clamping.
    for (Float maxComponentValue : {Float(Infinity), Float(100)}) {
        for (int nBuckets : {1, 15, 30, 64}) {
            SpectralFilm film(fp, lambdaMin, lambdaMax, nBuckets, RGBColorSpace::sRGB,
                              maxComponentValue, false /* writeFP16 */);

            // Provide pairs of samples, one with terminated secondary
            // wavelengths and one without, so that the pixel's radiance is
            // the average of LTerminated and LFull. As in the
            // SplatMatchesSample test, stratify the wavelengths so that each
            // bucket's estimate is exact for linear radiance functions.
            int nSamples = 64 * nBuckets * NSpectrumSamples;
            for (int i = 0; i < nSamples; ++i) {
                SampledWavelengths lambda = film.SampleWavelengths((i + 0.5f) / nSamples);
                SampledSpectrum L;
                for (int j = 0; j < NSpectrumSamples; ++j)
                    L[j] = LFull(lambda[j]);
                film.AddSample(Point2i(0, 0), L, lambda, nullptr, 1);

                lambda.TerminateSecondary();
                L = SampledSpectrum(LIgnored);
                L[0] = LTerminated(lambda[0]);
                film.AddSample(Point2i(0, 0), L, lambda, nullptr, 1);
            }

            ImageMetadata metadata;
            Image image = film.GetImage(&metadata);
            ASSERT_EQ(3 + nBuckets, image.NChannels());

            for (int b = 0; b < nBuckets; ++b) {
                Float lambdaCenter = Lerp((b + 0.5f) / nBuckets, lambdaMin, lambdaMax);
                Float expected = CIE_Y_integral *
                                 (LTerminated(lambdaCenter) + LFull(lambdaCenter)) / 2;
                Float v = image.GetChannel(Point2i(0, 0), 3 + b);
                EXPECT_NEAR(expected, v, 1e-3f * expected)
                    << "nBuckets " << nBuckets << ", maxComponentValue "
                    << maxComponentValue << ", bucket " << b;
            }
        }
    }
}
