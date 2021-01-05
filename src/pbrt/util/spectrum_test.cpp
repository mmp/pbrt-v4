// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

#include <array>

using namespace pbrt;

TEST(Spectrum, Blackbody) {
    // Relative error.
    auto err = [](Float val, Float ref) { return std::abs(val - ref) / ref; };

    // Planck's law.
    // A few values via
    // http://www.spectralcalc.com/blackbody_calculator/blackbody.php
    // lambda, T, expected radiance
    Float v[][3] = {
        {483, 6000, 3.1849e13},
        {600, 6000, 2.86772e13},
        {500, 3700, 1.59845e12},
        {600, 4500, 7.46497e12},
    };
    int n = PBRT_ARRAYSIZE(v);
    for (int i = 0; i < n; ++i) {
        Float lambda = v[i][0], T = v[i][1], LeExpected = v[i][2];
        EXPECT_LT(err(Blackbody(lambda, T), LeExpected), .001);
    }

    // Use Wien's displacement law to compute maximum wavelength for a few
    // temperatures, then confirm that the value returned by Blackbody() is
    // consistent with this.
    for (Float T : {2700, 3000, 4500, 5600, 6000}) {
        Float lambdaMax = 2.8977721e-3 / T * 1e9;
        Float lambda[3] = {Float(.99 * lambdaMax), lambdaMax, Float(1.01 * lambdaMax)};
        EXPECT_LT(Blackbody(lambda[0], T), Blackbody(lambda[1], T));
        EXPECT_GT(Blackbody(lambda[1], T), Blackbody(lambda[2], T));
    }
}

TEST(Spectrum, XYZ) {
    {
        // Make sure the integral of all matching function sample values is
        // basically one in x, y, and z.
        Float xx = 0, yy = 0, zz = 0;
        for (Float lambda = 360; lambda < 831; ++lambda) {
            xx += Spectra::X()(lambda);
            yy += Spectra::Y()(lambda);
            zz += Spectra::Z()(lambda);
        }
        static constexpr Float CIE_Y_integral = 106.856895;
        xx /= CIE_Y_integral;
        yy /= CIE_Y_integral;
        zz /= CIE_Y_integral;
        EXPECT_LT(std::abs(1 - xx), .005) << xx;
        EXPECT_LT(std::abs(1 - yy), .005) << yy;
        EXPECT_LT(std::abs(1 - zz), .005) << zz;
    }
    {
        // Make sure the xyz of a constant spectrum are basically one.
        std::array<Float, 3> xyzSum = {0};
        int n = 100;
        for (Float u : Stratified1D(n)) {
            SampledWavelengths lambda = SampledWavelengths::SampleUniform(u, 360, 830);
            XYZ xyz = SampledSpectrum(1.).ToXYZ(lambda);
            for (int c = 0; c < 3; ++c)
                xyzSum[c] += xyz[c];
        }
        for (int c = 0; c < 3; ++c)
            xyzSum[c] /= n;

        EXPECT_LT(std::abs(1 - xyzSum[0]), .035) << xyzSum[0];
        EXPECT_LT(std::abs(1 - xyzSum[1]), .035) << xyzSum[1];
        EXPECT_LT(std::abs(1 - xyzSum[2]), .035) << xyzSum[2];
    }
}

TEST(Spectrum, MaxValue) {
    EXPECT_EQ(2.5, ConstantSpectrum(2.5).MaxValue());

    EXPECT_EQ(Float(10.1),
              PiecewiseLinearSpectrum(
                  {Float(300), Float(380), Float(510), Float(662), Float(700)},
                  {Float(1.5), Float(2.6), Float(10.1), Float(5.3), Float(7.7)})
                  .MaxValue());

    EXPECT_GT(BlackbodySpectrum(5000).MaxValue(), .9999);
    EXPECT_LT(BlackbodySpectrum(5000).MaxValue(), 1.0001);

    BlackbodySpectrum bb(5000);
    EXPECT_GT(DenselySampledSpectrum(&bb).MaxValue(), .9999);
    EXPECT_LT(DenselySampledSpectrum(&bb).MaxValue(), 1.0001);

    RNG rng;
    for (int i = 0; i < 20; ++i) {
        RGB rgb(rng.Uniform<Float>(), rng.Uniform<Float>(), rng.Uniform<Float>());
        RGBAlbedoSpectrum sr(*RGBColorSpace::sRGB, rgb);
        Float m = sr.MaxValue() * 1.00001f;
        for (Float lambda = 360; lambda < 830; lambda += .92)
            EXPECT_LE(sr(lambda), m);

        RGBUnboundedSpectrum su(*RGBColorSpace::sRGB, 10 * rgb);
        m = su.MaxValue() * 1.00001f * 10.f;
        for (Float lambda = 360; lambda < 830; lambda += .92)
            EXPECT_LE(su(lambda), m);

        RGBIlluminantSpectrum si(*RGBColorSpace::sRGB, rgb);
        m = si.MaxValue() * 1.00001f;
        for (Float lambda = 360; lambda < 830; lambda += .92)
            EXPECT_LE(si(lambda), m);
    }
}

TEST(Spectrum, SamplingPdfY) {
    // Make sure we can integrate the y matching curve correctly
    Float ysum = 0;
    int n = 1000;
    for (Float u : Stratified1D(n)) {
        Float lambda = SampleXYZMatching(u);
        Float pdf = XYZMatchingPDF(lambda);
        if (pdf > 0)
            ysum += Spectra::Y()(lambda) / pdf;
    }
    Float yint = ysum / n;

    EXPECT_LT(std::abs((yint - CIE_Y_integral) / CIE_Y_integral), 1e-3)
        << yint << " vs. " << CIE_Y_integral;
}

TEST(Spectrum, SamplingPdfXYZ) {
    // Make sure we can integrate the sum of the x+y+z matching curves correctly
    Float impSum = 0, unifSum = 0;
    int n = 10000;
    for (Float u : Stratified1D(n)) {
        {
            // Uniform
            Float lambda = Lerp(u, Lambda_min, Lambda_max);
            Float pdf = 1. / (Lambda_max - Lambda_min);
            unifSum +=
                (Spectra::X()(lambda) + Spectra::Y()(lambda) + Spectra::Z()(lambda)) /
                pdf;
        }

        Float lambda = SampleXYZMatching(u);
        Float pdf = XYZMatchingPDF(lambda);
        if (pdf > 0)
            impSum +=
                (Spectra::X()(lambda) + Spectra::Y()(lambda) + Spectra::Z()(lambda)) /
                pdf;
    }
    Float impInt = impSum / n, unifInt = unifSum / n;

    EXPECT_LT(std::abs((impInt - unifInt) / unifInt), 1e-3)
        << impInt << " vs. " << unifInt;
}
