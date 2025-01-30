// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// PhysLight code contributed by Anders Langlands and Luca Fascione
// Copyright (c) 2020, Weta Digital, Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <pbrt/film.h>

#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/gui.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/transform.h>

#include <algorithm>
#include <cstring>

namespace pbrt {

PBRT_CPU_GPU void Film::AddSplat(Point2f p, SampledSpectrum v, const SampledWavelengths &lambda) {
    auto splat = [&](auto ptr) { return ptr->AddSplat(p, v, lambda); };
    return Dispatch(splat);
}

void Film::WriteImage(ImageMetadata metadata, Float splatScale) {
    auto write = [&](auto ptr) { return ptr->WriteImage(metadata, splatScale); };
    return DispatchCPU(write);
}

Image Film::GetImage(ImageMetadata *metadata, Float splatScale) {
    auto get = [&](auto ptr) { return ptr->GetImage(metadata, splatScale); };
    return DispatchCPU(get);
}

std::string Film::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

std::string Film::GetFilename() const {
    auto get = [&](auto ptr) { return ptr->GetFilename(); };
    return DispatchCPU(get);
}

// FilmBaseParameters Method Definitions
FilmBaseParameters::FilmBaseParameters(const ParameterDictionary &parameters,
                                       Filter filter, const PixelSensor *sensor,
                                       const FileLoc *loc)
    : filter(filter), sensor(sensor) {
    filename = parameters.GetOneString("filename", "");
    if (!Options->imageFile.empty()) {
        if (!filename.empty())
            Warning(loc,
                    "Output filename supplied on command line, \"%s\" will "
                    "override "
                    "filename provided in scene description file, \"%s\".",
                    Options->imageFile, filename);
        filename = Options->imageFile;
    } else if (filename.empty())
        filename = "pbrt.exr";

    if (Options->fullscreen) {
        fullResolution = GUI::GetResolution();

        // Omit unused parameter error
        auto unusedX = parameters.GetOneInt("xresolution", 1280);
        auto unusedY = parameters.GetOneInt("yresolution", 720);
    } else {
        fullResolution = Point2i(parameters.GetOneInt("xresolution", 1280),
                                 parameters.GetOneInt("yresolution", 720));
    }
    if (Options->quickRender) {
        fullResolution.x = std::max(1, fullResolution.x / 4);
        fullResolution.y = std::max(1, fullResolution.y / 4);
    }

    pixelBounds = Bounds2i(Point2i(0, 0), fullResolution);
    std::vector<int> pb = parameters.GetIntArray("pixelbounds");
    if (Options->pixelBounds) {
        Bounds2i newBounds = *Options->pixelBounds;
        if (Intersect(newBounds, pixelBounds) != newBounds)
            Warning(loc, "Supplied pixel bounds extend beyond image "
                         "resolution. Clamping.");
        pixelBounds = Intersect(newBounds, pixelBounds);

        if (!pb.empty())
            Warning(loc, "Both pixel bounds and crop window were specified. Using the "
                         "crop window.");
    } else if (!pb.empty()) {
        if (pb.size() != 4)
            Error(loc, "%d values supplied for \"pixelbounds\". Expected 4.",
                  int(pb.size()));
        else {
            Bounds2i newBounds = Bounds2i({pb[0], pb[2]}, {pb[1], pb[3]});
            if (Intersect(newBounds, pixelBounds) != newBounds)
                Warning(loc, "Supplied pixel bounds extend beyond image "
                             "resolution. Clamping.");
            pixelBounds = Intersect(newBounds, pixelBounds);
        }
    }

    std::vector<Float> cr = parameters.GetFloatArray("cropwindow");
    if (Options->cropWindow) {
        Bounds2f crop = *Options->cropWindow;
        if (Intersect(crop, Bounds2f(Point2f(0, 0), Point2f(1, 1))) != crop) {
            Error(loc,
                  "Film crop window %s is not in [0,1] range; did you "
                  "mean to use \"pixelbounds\"? Clamping to valid range.",
                  crop);
            crop = Intersect(crop, Bounds2f(Point2f(0, 0), Point2f(1, 1)));
        }

        // Compute film image bounds
        pixelBounds = Bounds2i(Point2i(pstd::ceil(fullResolution.x * crop.pMin.x),
                                       pstd::ceil(fullResolution.y * crop.pMin.y)),
                               Point2i(pstd::ceil(fullResolution.x * crop.pMax.x),
                                       pstd::ceil(fullResolution.y * crop.pMax.y)));

        if (!cr.empty())
            Warning(loc, "Crop window supplied on command line will override "
                         "crop window specified with Film.");
        if (Options->pixelBounds || !pb.empty())
            Warning(loc, "Both pixel bounds and crop window were specified. Using the "
                         "crop window.");
    } else if (!cr.empty()) {
        if (Options->pixelBounds)
            Warning(loc, "Ignoring \"cropwindow\" since pixel bounds were specified "
                         "on the command line.");
        else if (cr.size() == 4) {
            if (!pb.empty())
                Warning(loc, "Both pixel bounds and crop window were "
                             "specified. Using the "
                             "crop window.");

            Bounds2f crop;
            crop.pMin.x = Clamp(std::min(cr[0], cr[1]), 0.f, 1.f);
            crop.pMax.x = Clamp(std::max(cr[0], cr[1]), 0.f, 1.f);
            crop.pMin.y = Clamp(std::min(cr[2], cr[3]), 0.f, 1.f);
            crop.pMax.y = Clamp(std::max(cr[2], cr[3]), 0.f, 1.f);

            // Compute film image bounds
            pixelBounds = Bounds2i(Point2i(pstd::ceil(fullResolution.x * crop.pMin.x),
                                           pstd::ceil(fullResolution.y * crop.pMin.y)),
                                   Point2i(pstd::ceil(fullResolution.x * crop.pMax.x),
                                           pstd::ceil(fullResolution.y * crop.pMax.y)));
        } else
            Error(loc, "%d values supplied for \"cropwindow\". Expected 4.",
                  (int)cr.size());
    }

    if (pixelBounds.IsEmpty())
        ErrorExit(loc, "Degenerate pixel bounds provided to film: %s.", pixelBounds);

    diagonal = parameters.GetOneFloat("diagonal", 35.);
}

// FilmBase Method Definitions
PBRT_CPU_GPU Bounds2f FilmBase::SampleBounds() const {
    Vector2f radius = filter.Radius();
    return Bounds2f(pixelBounds.pMin - radius + Vector2f(0.5f, 0.5f),
                    pixelBounds.pMax + radius - Vector2f(0.5f, 0.5f));
}

std::string FilmBase::BaseToString() const {
    return StringPrintf("fullResolution: %s diagonal: %f filter: %s filename: %s "
                        "pixelBounds: %s",
                        fullResolution, diagonal, filter, filename, pixelBounds);
}

// VisibleSurface Method Definitions
PBRT_CPU_GPU VisibleSurface::VisibleSurface(const SurfaceInteraction &si, SampledSpectrum albedo,
                               const SampledWavelengths &lambda)
    : albedo(albedo) {
    set = true;
    // Initialize geometric _VisibleSurface_ members
    p = si.p();
    Vector3f wo = si.wo;
    n = FaceForward(si.n, wo);
    ns = FaceForward(si.shading.n, wo);
    uv = si.uv;
    time = si.time;
    dpdx = si.dpdx;
    dpdy = si.dpdy;
}

std::string VisibleSurface::ToString() const {
    return StringPrintf("[ VisibleSurface set: %s p: %s n: %s ns: %s dpdx: %f dpdy: %f "
                        "time: %f albedo: %s ]",
                        set, p, n, ns, dpdx, dpdy, time, albedo);
}

// PixelSensor Method Definitions
PixelSensor *PixelSensor::Create(const ParameterDictionary &parameters,
                                 const RGBColorSpace *colorSpace, Float exposureTime,
                                 const FileLoc *loc, Allocator alloc) {
    // Imaging ratio parameters
    // The defaults here represent a "passthrough" setup such that the imaging
    // ratio will be exactly 1. This is a useful default since scenes that
    // weren't authored with a physical camera in mind will render as expected.
    Float ISO = parameters.GetOneFloat("iso", 100.);
    Float whiteBalanceTemp = parameters.GetOneFloat("whitebalance", 0);

    std::string sensorName = parameters.GetOneString("sensor", "cie1931");

    // Pass through 0 for cie1931 if it's unspecified so that it doesn't do
    // any white balancing. For actual sensors, 6500 is the default...
    if (sensorName != "cie1931" && whiteBalanceTemp == 0)
        whiteBalanceTemp = 6500;

    // Note: in the talk we mention using 312.5 for historical reasons. The
    // choice of 100 here just means that the other parameters make nice
    // "round" numbers like 1 and 100.
    Float imagingRatio = exposureTime * ISO / 100;

    DenselySampledSpectrum dIllum =
        Spectra::D(whiteBalanceTemp == 0 ? 6500.f : whiteBalanceTemp, alloc);
    Spectrum sensorIllum = whiteBalanceTemp != 0 ? &dIllum : nullptr;

    if (sensorName == "cie1931") {
        return alloc.new_object<PixelSensor>(colorSpace, sensorIllum, imagingRatio,
                                             alloc);
    } else {
        Spectrum r = GetNamedSpectrum(sensorName + "_r");
        Spectrum g = GetNamedSpectrum(sensorName + "_g");
        Spectrum b = GetNamedSpectrum(sensorName + "_b");

        if (!r || !g || !b)
            ErrorExit(loc, "%s: unknown sensor type", sensorName);

        return alloc.new_object<PixelSensor>(r, g, b, colorSpace, sensorIllum,
                                             imagingRatio, alloc);
    }
}

PixelSensor *PixelSensor::CreateDefault(Allocator alloc) {
    return Create(ParameterDictionary(), RGBColorSpace::sRGB, 1.0, nullptr, alloc);
}

// Swatch reflectances are taken from Danny Pascale's Macbeth chart measurements
// BabelColor ColorChecker data: Copyright (c) 2004-2012 Danny Pascale
// (www.babelcolor.com); used by permission.
// http://www.babelcolor.com/index_htm_files/ColorChecker_RGB_and_spectra.zip
Spectrum PixelSensor::swatchReflectances[nSwatchReflectances]{
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.055, 390.0, 0.058, 400.0, 0.061, 410.0, 0.062, 420.0, 0.062, 430.0,
         0.062, 440.0, 0.062, 450.0, 0.062, 460.0, 0.062, 470.0, 0.062, 480.0, 0.062,
         490.0, 0.063, 500.0, 0.065, 510.0, 0.070, 520.0, 0.076, 530.0, 0.079, 540.0,
         0.081, 550.0, 0.084, 560.0, 0.091, 570.0, 0.103, 580.0, 0.119, 590.0, 0.134,
         600.0, 0.143, 610.0, 0.147, 620.0, 0.151, 630.0, 0.158, 640.0, 0.168, 650.0,
         0.179, 660.0, 0.188, 670.0, 0.190, 680.0, 0.186, 690.0, 0.181, 700.0, 0.182,
         710.0, 0.187, 720.0, 0.196, 730.0, 0.209},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.117, 390.0, 0.143, 400.0, 0.175, 410.0, 0.191, 420.0, 0.196, 430.0,
         0.199, 440.0, 0.204, 450.0, 0.213, 460.0, 0.228, 470.0, 0.251, 480.0, 0.280,
         490.0, 0.309, 500.0, 0.329, 510.0, 0.333, 520.0, 0.315, 530.0, 0.286, 540.0,
         0.273, 550.0, 0.276, 560.0, 0.277, 570.0, 0.289, 580.0, 0.339, 590.0, 0.420,
         600.0, 0.488, 610.0, 0.525, 620.0, 0.546, 630.0, 0.562, 640.0, 0.578, 650.0,
         0.595, 660.0, 0.612, 670.0, 0.625, 680.0, 0.638, 690.0, 0.656, 700.0, 0.678,
         710.0, 0.700, 720.0, 0.717, 730.0, 0.734},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.130, 390.0, 0.177, 400.0, 0.251, 410.0, 0.306, 420.0, 0.324, 430.0,
         0.330, 440.0, 0.333, 450.0, 0.331, 460.0, 0.323, 470.0, 0.311, 480.0, 0.298,
         490.0, 0.285, 500.0, 0.269, 510.0, 0.250, 520.0, 0.231, 530.0, 0.214, 540.0,
         0.199, 550.0, 0.185, 560.0, 0.169, 570.0, 0.157, 580.0, 0.149, 590.0, 0.145,
         600.0, 0.142, 610.0, 0.141, 620.0, 0.141, 630.0, 0.141, 640.0, 0.143, 650.0,
         0.147, 660.0, 0.152, 670.0, 0.154, 680.0, 0.150, 690.0, 0.144, 700.0, 0.136,
         710.0, 0.132, 720.0, 0.135, 730.0, 0.147},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.051, 390.0, 0.054, 400.0, 0.056, 410.0, 0.057, 420.0, 0.058, 430.0,
         0.059, 440.0, 0.060, 450.0, 0.061, 460.0, 0.062, 470.0, 0.063, 480.0, 0.065,
         490.0, 0.067, 500.0, 0.075, 510.0, 0.101, 520.0, 0.145, 530.0, 0.178, 540.0,
         0.184, 550.0, 0.170, 560.0, 0.149, 570.0, 0.133, 580.0, 0.122, 590.0, 0.115,
         600.0, 0.109, 610.0, 0.105, 620.0, 0.104, 630.0, 0.106, 640.0, 0.109, 650.0,
         0.112, 660.0, 0.114, 670.0, 0.114, 680.0, 0.112, 690.0, 0.112, 700.0, 0.115,
         710.0, 0.120, 720.0, 0.125, 730.0, 0.130},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.144, 390.0, 0.198, 400.0, 0.294, 410.0, 0.375, 420.0, 0.408, 430.0,
         0.421, 440.0, 0.426, 450.0, 0.426, 460.0, 0.419, 470.0, 0.403, 480.0, 0.379,
         490.0, 0.346, 500.0, 0.311, 510.0, 0.281, 520.0, 0.254, 530.0, 0.229, 540.0,
         0.214, 550.0, 0.208, 560.0, 0.202, 570.0, 0.194, 580.0, 0.193, 590.0, 0.200,
         600.0, 0.214, 610.0, 0.230, 620.0, 0.241, 630.0, 0.254, 640.0, 0.279, 650.0,
         0.313, 660.0, 0.348, 670.0, 0.366, 680.0, 0.366, 690.0, 0.359, 700.0, 0.358,
         710.0, 0.365, 720.0, 0.377, 730.0, 0.398},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.136, 390.0, 0.179, 400.0, 0.247, 410.0, 0.297, 420.0, 0.320, 430.0,
         0.337, 440.0, 0.355, 450.0, 0.381, 460.0, 0.419, 470.0, 0.466, 480.0, 0.510,
         490.0, 0.546, 500.0, 0.567, 510.0, 0.574, 520.0, 0.569, 530.0, 0.551, 540.0,
         0.524, 550.0, 0.488, 560.0, 0.445, 570.0, 0.400, 580.0, 0.350, 590.0, 0.299,
         600.0, 0.252, 610.0, 0.221, 620.0, 0.204, 630.0, 0.196, 640.0, 0.191, 650.0,
         0.188, 660.0, 0.191, 670.0, 0.199, 680.0, 0.212, 690.0, 0.223, 700.0, 0.232,
         710.0, 0.233, 720.0, 0.229, 730.0, 0.229},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.054, 390.0, 0.054, 400.0, 0.053, 410.0, 0.054, 420.0, 0.054, 430.0,
         0.055, 440.0, 0.055, 450.0, 0.055, 460.0, 0.056, 470.0, 0.057, 480.0, 0.058,
         490.0, 0.061, 500.0, 0.068, 510.0, 0.089, 520.0, 0.125, 530.0, 0.154, 540.0,
         0.174, 550.0, 0.199, 560.0, 0.248, 570.0, 0.335, 580.0, 0.444, 590.0, 0.538,
         600.0, 0.587, 610.0, 0.595, 620.0, 0.591, 630.0, 0.587, 640.0, 0.584, 650.0,
         0.584, 660.0, 0.590, 670.0, 0.603, 680.0, 0.620, 690.0, 0.639, 700.0, 0.655,
         710.0, 0.663, 720.0, 0.663, 730.0, 0.667},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.122, 390.0, 0.164, 400.0, 0.229, 410.0, 0.286, 420.0, 0.327, 430.0,
         0.361, 440.0, 0.388, 450.0, 0.400, 460.0, 0.392, 470.0, 0.362, 480.0, 0.316,
         490.0, 0.260, 500.0, 0.209, 510.0, 0.168, 520.0, 0.138, 530.0, 0.117, 540.0,
         0.104, 550.0, 0.096, 560.0, 0.090, 570.0, 0.086, 580.0, 0.084, 590.0, 0.084,
         600.0, 0.084, 610.0, 0.084, 620.0, 0.084, 630.0, 0.085, 640.0, 0.090, 650.0,
         0.098, 660.0, 0.109, 670.0, 0.123, 680.0, 0.143, 690.0, 0.169, 700.0, 0.205,
         710.0, 0.244, 720.0, 0.287, 730.0, 0.332},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.096, 390.0, 0.115, 400.0, 0.131, 410.0, 0.135, 420.0, 0.133, 430.0,
         0.132, 440.0, 0.130, 450.0, 0.128, 460.0, 0.125, 470.0, 0.120, 480.0, 0.115,
         490.0, 0.110, 500.0, 0.105, 510.0, 0.100, 520.0, 0.095, 530.0, 0.093, 540.0,
         0.092, 550.0, 0.093, 560.0, 0.096, 570.0, 0.108, 580.0, 0.156, 590.0, 0.265,
         600.0, 0.399, 610.0, 0.500, 620.0, 0.556, 630.0, 0.579, 640.0, 0.588, 650.0,
         0.591, 660.0, 0.593, 670.0, 0.594, 680.0, 0.598, 690.0, 0.602, 700.0, 0.607,
         710.0, 0.609, 720.0, 0.609, 730.0, 0.610},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.092, 390.0, 0.116, 400.0, 0.146, 410.0, 0.169, 420.0, 0.178, 430.0,
         0.173, 440.0, 0.158, 450.0, 0.139, 460.0, 0.119, 470.0, 0.101, 480.0, 0.087,
         490.0, 0.075, 500.0, 0.066, 510.0, 0.060, 520.0, 0.056, 530.0, 0.053, 540.0,
         0.051, 550.0, 0.051, 560.0, 0.052, 570.0, 0.052, 580.0, 0.051, 590.0, 0.052,
         600.0, 0.058, 610.0, 0.073, 620.0, 0.096, 630.0, 0.119, 640.0, 0.141, 650.0,
         0.166, 660.0, 0.194, 670.0, 0.227, 680.0, 0.265, 690.0, 0.309, 700.0, 0.355,
         710.0, 0.396, 720.0, 0.436, 730.0, 0.478},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.061, 390.0, 0.061, 400.0, 0.062, 410.0, 0.063, 420.0, 0.064, 430.0,
         0.066, 440.0, 0.069, 450.0, 0.075, 460.0, 0.085, 470.0, 0.105, 480.0, 0.139,
         490.0, 0.192, 500.0, 0.271, 510.0, 0.376, 520.0, 0.476, 530.0, 0.531, 540.0,
         0.549, 550.0, 0.546, 560.0, 0.528, 570.0, 0.504, 580.0, 0.471, 590.0, 0.428,
         600.0, 0.381, 610.0, 0.347, 620.0, 0.327, 630.0, 0.318, 640.0, 0.312, 650.0,
         0.310, 660.0, 0.314, 670.0, 0.327, 680.0, 0.345, 690.0, 0.363, 700.0, 0.376,
         710.0, 0.381, 720.0, 0.378, 730.0, 0.379},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.063, 390.0, 0.063, 400.0, 0.063, 410.0, 0.064, 420.0, 0.064, 430.0,
         0.064, 440.0, 0.065, 450.0, 0.066, 460.0, 0.067, 470.0, 0.068, 480.0, 0.071,
         490.0, 0.076, 500.0, 0.087, 510.0, 0.125, 520.0, 0.206, 530.0, 0.305, 540.0,
         0.383, 550.0, 0.431, 560.0, 0.469, 570.0, 0.518, 580.0, 0.568, 590.0, 0.607,
         600.0, 0.628, 610.0, 0.637, 620.0, 0.640, 630.0, 0.642, 640.0, 0.645, 650.0,
         0.648, 660.0, 0.651, 670.0, 0.653, 680.0, 0.657, 690.0, 0.664, 700.0, 0.673,
         710.0, 0.680, 720.0, 0.684, 730.0, 0.688},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.066, 390.0, 0.079, 400.0, 0.102, 410.0, 0.146, 420.0, 0.200, 430.0,
         0.244, 440.0, 0.282, 450.0, 0.309, 460.0, 0.308, 470.0, 0.278, 480.0, 0.231,
         490.0, 0.178, 500.0, 0.130, 510.0, 0.094, 520.0, 0.070, 530.0, 0.054, 540.0,
         0.046, 550.0, 0.042, 560.0, 0.039, 570.0, 0.038, 580.0, 0.038, 590.0, 0.038,
         600.0, 0.038, 610.0, 0.039, 620.0, 0.039, 630.0, 0.040, 640.0, 0.041, 650.0,
         0.042, 660.0, 0.044, 670.0, 0.045, 680.0, 0.046, 690.0, 0.046, 700.0, 0.048,
         710.0, 0.052, 720.0, 0.057, 730.0, 0.065},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.052, 390.0, 0.053, 400.0, 0.054, 410.0, 0.055, 420.0, 0.057, 430.0,
         0.059, 440.0, 0.061, 450.0, 0.066, 460.0, 0.075, 470.0, 0.093, 480.0, 0.125,
         490.0, 0.178, 500.0, 0.246, 510.0, 0.307, 520.0, 0.337, 530.0, 0.334, 540.0,
         0.317, 550.0, 0.293, 560.0, 0.262, 570.0, 0.230, 580.0, 0.198, 590.0, 0.165,
         600.0, 0.135, 610.0, 0.115, 620.0, 0.104, 630.0, 0.098, 640.0, 0.094, 650.0,
         0.092, 660.0, 0.093, 670.0, 0.097, 680.0, 0.102, 690.0, 0.108, 700.0, 0.113,
         710.0, 0.115, 720.0, 0.114, 730.0, 0.114},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.050, 390.0, 0.049, 400.0, 0.048, 410.0, 0.047, 420.0, 0.047, 430.0,
         0.047, 440.0, 0.047, 450.0, 0.047, 460.0, 0.046, 470.0, 0.045, 480.0, 0.044,
         490.0, 0.044, 500.0, 0.045, 510.0, 0.046, 520.0, 0.047, 530.0, 0.048, 540.0,
         0.049, 550.0, 0.050, 560.0, 0.054, 570.0, 0.060, 580.0, 0.072, 590.0, 0.104,
         600.0, 0.178, 610.0, 0.312, 620.0, 0.467, 630.0, 0.581, 640.0, 0.644, 650.0,
         0.675, 660.0, 0.690, 670.0, 0.698, 680.0, 0.706, 690.0, 0.715, 700.0, 0.724,
         710.0, 0.730, 720.0, 0.734, 730.0, 0.738},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.058, 390.0, 0.054, 400.0, 0.052, 410.0, 0.052, 420.0, 0.053, 430.0,
         0.054, 440.0, 0.056, 450.0, 0.059, 460.0, 0.067, 470.0, 0.081, 480.0, 0.107,
         490.0, 0.152, 500.0, 0.225, 510.0, 0.336, 520.0, 0.462, 530.0, 0.559, 540.0,
         0.616, 550.0, 0.650, 560.0, 0.672, 570.0, 0.694, 580.0, 0.710, 590.0, 0.723,
         600.0, 0.731, 610.0, 0.739, 620.0, 0.746, 630.0, 0.752, 640.0, 0.758, 650.0,
         0.764, 660.0, 0.769, 670.0, 0.771, 680.0, 0.776, 690.0, 0.782, 700.0, 0.790,
         710.0, 0.796, 720.0, 0.799, 730.0, 0.804},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.145, 390.0, 0.195, 400.0, 0.283, 410.0, 0.346, 420.0, 0.362, 430.0,
         0.354, 440.0, 0.334, 450.0, 0.306, 460.0, 0.276, 470.0, 0.248, 480.0, 0.218,
         490.0, 0.190, 500.0, 0.168, 510.0, 0.149, 520.0, 0.127, 530.0, 0.107, 540.0,
         0.100, 550.0, 0.102, 560.0, 0.104, 570.0, 0.109, 580.0, 0.137, 590.0, 0.200,
         600.0, 0.290, 610.0, 0.400, 620.0, 0.516, 630.0, 0.615, 640.0, 0.687, 650.0,
         0.732, 660.0, 0.760, 670.0, 0.774, 680.0, 0.783, 690.0, 0.793, 700.0, 0.803,
         710.0, 0.812, 720.0, 0.817, 730.0, 0.825},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.108, 390.0, 0.141, 400.0, 0.192, 410.0, 0.236, 420.0, 0.261, 430.0,
         0.286, 440.0, 0.317, 450.0, 0.353, 460.0, 0.390, 470.0, 0.426, 480.0, 0.446,
         490.0, 0.444, 500.0, 0.423, 510.0, 0.385, 520.0, 0.337, 530.0, 0.283, 540.0,
         0.231, 550.0, 0.185, 560.0, 0.146, 570.0, 0.118, 580.0, 0.101, 590.0, 0.090,
         600.0, 0.082, 610.0, 0.076, 620.0, 0.074, 630.0, 0.073, 640.0, 0.073, 650.0,
         0.074, 660.0, 0.076, 670.0, 0.077, 680.0, 0.076, 690.0, 0.075, 700.0, 0.073,
         710.0, 0.072, 720.0, 0.074, 730.0, 0.079},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.189, 390.0, 0.255, 400.0, 0.423, 410.0, 0.660, 420.0, 0.811, 430.0,
         0.862, 440.0, 0.877, 450.0, 0.884, 460.0, 0.891, 470.0, 0.896, 480.0, 0.899,
         490.0, 0.904, 500.0, 0.907, 510.0, 0.909, 520.0, 0.911, 530.0, 0.910, 540.0,
         0.911, 550.0, 0.914, 560.0, 0.913, 570.0, 0.916, 580.0, 0.915, 590.0, 0.916,
         600.0, 0.914, 610.0, 0.915, 620.0, 0.918, 630.0, 0.919, 640.0, 0.921, 650.0,
         0.923, 660.0, 0.924, 670.0, 0.922, 680.0, 0.922, 690.0, 0.925, 700.0, 0.927,
         710.0, 0.930, 720.0, 0.930, 730.0, 0.933},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.171, 390.0, 0.232, 400.0, 0.365, 410.0, 0.507, 420.0, 0.567, 430.0,
         0.583, 440.0, 0.588, 450.0, 0.590, 460.0, 0.591, 470.0, 0.590, 480.0, 0.588,
         490.0, 0.588, 500.0, 0.589, 510.0, 0.589, 520.0, 0.591, 530.0, 0.590, 540.0,
         0.590, 550.0, 0.590, 560.0, 0.589, 570.0, 0.591, 580.0, 0.590, 590.0, 0.590,
         600.0, 0.587, 610.0, 0.585, 620.0, 0.583, 630.0, 0.580, 640.0, 0.578, 650.0,
         0.576, 660.0, 0.574, 670.0, 0.572, 680.0, 0.571, 690.0, 0.569, 700.0, 0.568,
         710.0, 0.568, 720.0, 0.566, 730.0, 0.566},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.144, 390.0, 0.192, 400.0, 0.272, 410.0, 0.331, 420.0, 0.350, 430.0,
         0.357, 440.0, 0.361, 450.0, 0.363, 460.0, 0.363, 470.0, 0.361, 480.0, 0.359,
         490.0, 0.358, 500.0, 0.358, 510.0, 0.359, 520.0, 0.360, 530.0, 0.360, 540.0,
         0.361, 550.0, 0.361, 560.0, 0.360, 570.0, 0.362, 580.0, 0.362, 590.0, 0.361,
         600.0, 0.359, 610.0, 0.358, 620.0, 0.355, 630.0, 0.352, 640.0, 0.350, 650.0,
         0.348, 660.0, 0.345, 670.0, 0.343, 680.0, 0.340, 690.0, 0.338, 700.0, 0.335,
         710.0, 0.334, 720.0, 0.332, 730.0, 0.331},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.105, 390.0, 0.131, 400.0, 0.163, 410.0, 0.180, 420.0, 0.186, 430.0,
         0.190, 440.0, 0.193, 450.0, 0.194, 460.0, 0.194, 470.0, 0.192, 480.0, 0.191,
         490.0, 0.191, 500.0, 0.191, 510.0, 0.192, 520.0, 0.192, 530.0, 0.192, 540.0,
         0.192, 550.0, 0.192, 560.0, 0.192, 570.0, 0.193, 580.0, 0.192, 590.0, 0.192,
         600.0, 0.191, 610.0, 0.189, 620.0, 0.188, 630.0, 0.186, 640.0, 0.184, 650.0,
         0.182, 660.0, 0.181, 670.0, 0.179, 680.0, 0.178, 690.0, 0.176, 700.0, 0.174,
         710.0, 0.173, 720.0, 0.172, 730.0, 0.171},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.068, 390.0, 0.077, 400.0, 0.084, 410.0, 0.087, 420.0, 0.089, 430.0,
         0.090, 440.0, 0.092, 450.0, 0.092, 460.0, 0.091, 470.0, 0.090, 480.0, 0.090,
         490.0, 0.090, 500.0, 0.090, 510.0, 0.090, 520.0, 0.090, 530.0, 0.090, 540.0,
         0.090, 550.0, 0.090, 560.0, 0.090, 570.0, 0.090, 580.0, 0.090, 590.0, 0.089,
         600.0, 0.089, 610.0, 0.088, 620.0, 0.087, 630.0, 0.086, 640.0, 0.086, 650.0,
         0.085, 660.0, 0.084, 670.0, 0.084, 680.0, 0.083, 690.0, 0.083, 700.0, 0.082,
         710.0, 0.081, 720.0, 0.081, 730.0, 0.081},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.0, 0.031, 390.0, 0.032, 400.0, 0.032, 410.0, 0.033, 420.0, 0.033, 430.0,
         0.033, 440.0, 0.033, 450.0, 0.033, 460.0, 0.032, 470.0, 0.032, 480.0, 0.032,
         490.0, 0.032, 500.0, 0.032, 510.0, 0.032, 520.0, 0.032, 530.0, 0.032, 540.0,
         0.032, 550.0, 0.032, 560.0, 0.032, 570.0, 0.032, 580.0, 0.032, 590.0, 0.032,
         600.0, 0.032, 610.0, 0.032, 620.0, 0.032, 630.0, 0.032, 640.0, 0.032, 650.0,
         0.032, 660.0, 0.032, 670.0, 0.032, 680.0, 0.032, 690.0, 0.032, 700.0, 0.032,
         710.0, 0.032, 720.0, 0.032, 730.0, 0.033},
        false, Allocator())};

STAT_MEMORY_COUNTER("Memory/Film pixels", filmPixelMemory);

// RGBFilm Method Definitions
RGBFilm::RGBFilm(FilmBaseParameters p, const RGBColorSpace *colorSpace,
                 Float maxComponentValue, bool writeFP16, Allocator alloc)
    : FilmBase(p),
      pixels(p.pixelBounds, alloc),
      colorSpace(colorSpace),
      maxComponentValue(maxComponentValue),
      writeFP16(writeFP16) {
    filterIntegral = filter.Integral();
    CHECK(!pixelBounds.IsEmpty());
    CHECK(colorSpace);
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
    // Compute _outputRGBFromSensorRGB_ matrix
    outputRGBFromSensorRGB = colorSpace->RGBFromXYZ * sensor->XYZFromSensorRGB;
}

PBRT_CPU_GPU void RGBFilm::AddSplat(Point2f p, SampledSpectrum L, const SampledWavelengths &lambda) {
    CHECK(!L.HasNaNs());
    // Convert sample radiance to _PixelSensor_ RGB
    RGB rgb = sensor->ToSensorRGB(L, lambda);

    // Optionally clamp sensor RGB value
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue)
        rgb *= maxComponentValue / m;

    // Compute bounds of affected pixels for splat, _splatBounds_
    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Vector2f radius = filter.Radius();
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - radius)),
                         Point2i(Floor(pDiscrete + radius)) + Vector2i(1, 1));
    splatBounds = Intersect(splatBounds, pixelBounds);

    for (Point2i pi : splatBounds) {
        // Evaluate filter at _pi_ and add splat contribution
        Float wt = filter.Evaluate(Point2f(p - pi - Vector2f(0.5, 0.5)));
        if (wt != 0) {
            Pixel &pixel = pixels[pi];
            for (int i = 0; i < 3; ++i)
                pixel.rgbSplat[i].Add(wt * rgb[i]);
        }
    }
}

void RGBFilm::WriteImage(ImageMetadata metadata, Float splatScale) {
    Image image = GetImage(&metadata, splatScale);
    LOG_VERBOSE("Writing image %s with bounds %s", filename, pixelBounds);
    image.Write(filename, metadata);
}

Image RGBFilm::GetImage(ImageMetadata *metadata, Float splatScale) {
    // Convert image to RGB and compute final pixel values
    LOG_VERBOSE("Converting image to RGB and computing final weighted pixel values");
    PixelFormat format = writeFP16 ? PixelFormat::Half : PixelFormat::Float;
    Image image(format, Point2i(pixelBounds.Diagonal()), {"R", "G", "B"});

    std::atomic<int> nClamped{0};
    ParallelFor2D(pixelBounds, [&](Point2i p) {
        RGB rgb = GetPixelRGB(p, splatScale);

        if (writeFP16 && std::max({rgb.r, rgb.g, rgb.b}) > 65504) {
            if (rgb.r > 65504)
                rgb.r = 65504;
            if (rgb.g > 65504)
                rgb.g = 65504;
            if (rgb.b > 65504)
                rgb.b = 65504;
            ++nClamped;
        }

        Point2i pOffset(p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, {rgb[0], rgb[1], rgb[2]});
    });

    if (nClamped.load() > 0)
        Warning("%d pixel values clamped to maximum fp16 value.", nClamped.load());

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;
    metadata->colorSpace = colorSpace;

    return image;
}

std::string RGBFilm::ToString() const {
    return StringPrintf(
        "[ RGBFilm %s colorSpace: %s maxComponentValue: %f writeFP16: %s ]",
        BaseToString(), *colorSpace, maxComponentValue, writeFP16);
}

RGBFilm *RGBFilm::Create(const ParameterDictionary &parameters, Float exposureTime,
                         Filter filter, const RGBColorSpace *colorSpace,
                         const FileLoc *loc, Allocator alloc) {
    Float maxComponentValue = parameters.GetOneFloat("maxcomponentvalue", Infinity);
    bool writeFP16 = parameters.GetOneBool("savefp16", true);

    PixelSensor *sensor =
        PixelSensor::Create(parameters, colorSpace, exposureTime, loc, alloc);
    FilmBaseParameters filmBaseParameters(parameters, filter, sensor, loc);

    return alloc.new_object<RGBFilm>(filmBaseParameters, colorSpace, maxComponentValue,
                                     writeFP16, alloc);
}

// GBufferFilm Method Definitions
PBRT_CPU_GPU void GBufferFilm::AddSample(Point2i pFilm, SampledSpectrum L,
                            const SampledWavelengths &lambda,
                            const VisibleSurface *visibleSurface, Float weight) {
    RGB rgb = sensor->ToSensorRGB(L, lambda);
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue)
        rgb *= maxComponentValue / m;

    Pixel &p = pixels[pFilm];
    if (visibleSurface && *visibleSurface) {
        p.gBufferWeightSum += weight;

        // Update variance estimates.
        for (int c = 0; c < 3; ++c)
            p.rgbVariance[c].Add(rgb[c]);

        if (applyInverse) {
            p.pSum += weight * outputFromRender.ApplyInverse(visibleSurface->p,
                                                             visibleSurface->time);
            p.nSum += weight * outputFromRender.ApplyInverse(visibleSurface->n,
                                                             visibleSurface->time);
            p.nsSum += weight * outputFromRender.ApplyInverse(visibleSurface->ns,
                                                              visibleSurface->time);
            p.dzdxSum +=
                weight *
                outputFromRender.ApplyInverse(visibleSurface->dpdx, visibleSurface->time)
                    .z;
            p.dzdySum +=
                weight *
                outputFromRender.ApplyInverse(visibleSurface->dpdy, visibleSurface->time)
                    .z;
        } else {
            p.pSum += weight * outputFromRender(visibleSurface->p, visibleSurface->time);
            p.nSum += weight * outputFromRender(visibleSurface->n, visibleSurface->time);
            p.nsSum +=
                weight * outputFromRender(visibleSurface->ns, visibleSurface->time);
            p.dzdxSum +=
                weight * outputFromRender(visibleSurface->dpdx, visibleSurface->time).z;
            p.dzdySum +=
                weight * outputFromRender(visibleSurface->dpdy, visibleSurface->time).z;
        }
        p.uvSum += weight * visibleSurface->uv;

        SampledSpectrum albedo =
            visibleSurface->albedo * colorSpace->illuminant.Sample(lambda);
        RGB albedoRGB = albedo.ToRGB(lambda, *colorSpace);
        for (int c = 0; c < 3; ++c)
            p.rgbAlbedoSum[c] += weight * albedoRGB[c];
    }

    for (int c = 0; c < 3; ++c)
        p.rgbSum[c] += rgb[c] * weight;
    p.weightSum += weight;
}

GBufferFilm::GBufferFilm(FilmBaseParameters p, const AnimatedTransform &outputFromRender,
                         bool applyInverse, const RGBColorSpace *colorSpace,
                         Float maxComponentValue, bool writeFP16, Allocator alloc)
    : FilmBase(p),
      outputFromRender(outputFromRender),
      applyInverse(applyInverse),
      pixels(pixelBounds, alloc),
      colorSpace(colorSpace),
      maxComponentValue(maxComponentValue),
      writeFP16(writeFP16),
      filterIntegral(filter.Integral()) {
    CHECK(!pixelBounds.IsEmpty());
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
    outputRGBFromSensorRGB = colorSpace->RGBFromXYZ * sensor->XYZFromSensorRGB;
}

PBRT_CPU_GPU void GBufferFilm::AddSplat(Point2f p, SampledSpectrum v,
                           const SampledWavelengths &lambda) {
    // NOTE: same code as RGBFilm::AddSplat()...
    CHECK(!v.HasNaNs());
    RGB rgb = sensor->ToSensorRGB(v, lambda);
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue)
        rgb *= maxComponentValue / m;

    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - filter.Radius())),
                         Point2i(Floor(pDiscrete + filter.Radius())) + Vector2i(1, 1));
    splatBounds = Intersect(splatBounds, pixelBounds);
    for (Point2i pi : splatBounds) {
        Float wt = filter.Evaluate(Point2f(p - pi - Vector2f(0.5, 0.5)));
        if (wt != 0) {
            Pixel &pixel = pixels[pi];
            for (int i = 0; i < 3; ++i)
                pixel.rgbSplat[i].Add(wt * rgb[i]);
        }
    }
}

void GBufferFilm::WriteImage(ImageMetadata metadata, Float splatScale) {
    Image image = GetImage(&metadata, splatScale);
    LOG_VERBOSE("Writing image %s with bounds %s", filename, pixelBounds);
    image.Write(filename, metadata);
}

Image GBufferFilm::GetImage(ImageMetadata *metadata, Float splatScale) {
    // Convert image to RGB and compute final pixel values
    LOG_VERBOSE("Converting image to RGB and computing final weighted pixel values");
    PixelFormat format = writeFP16 ? PixelFormat::Half : PixelFormat::Float;
    Image image(format, Point2i(pixelBounds.Diagonal()),
                {"R",
                 "G",
                 "B",
                 "Albedo.R",
                 "Albedo.G",
                 "Albedo.B",
                 "P.X",
                 "P.Y",
                 "P.Z",
                 "dzdx",
                 "dzdy",
                 "N.X",
                 "N.Y",
                 "N.Z",
                 "Ns.X",
                 "Ns.Y",
                 "Ns.Z",
                 "u",
                 "v",
                 "Variance.R",
                 "Variance.G",
                 "Variance.B",
                 "RelativeVariance.R",
                 "RelativeVariance.G",
                 "RelativeVariance.B"});

    ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
    ImageChannelDesc pDesc = image.GetChannelDesc({"P.X", "P.Y", "P.Z"});
    ImageChannelDesc dzDesc = image.GetChannelDesc({"dzdx", "dzdy"});
    ImageChannelDesc nDesc = image.GetChannelDesc({"N.X", "N.Y", "N.Z"});
    ImageChannelDesc nsDesc = image.GetChannelDesc({"Ns.X", "Ns.Y", "Ns.Z"});
    ImageChannelDesc uvDesc = image.GetChannelDesc({"u", "v"});
    ImageChannelDesc albedoRgbDesc =
        image.GetChannelDesc({"Albedo.R", "Albedo.G", "Albedo.B"});
    ImageChannelDesc varianceDesc =
        image.GetChannelDesc({"Variance.R", "Variance.G", "Variance.B"});
    ImageChannelDesc relVarianceDesc = image.GetChannelDesc(
        {"RelativeVariance.R", "RelativeVariance.G", "RelativeVariance.B"});

    std::atomic<int> nClamped{0};
    ParallelFor2D(pixelBounds, [&](Point2i p) {
        Pixel &pixel = pixels[p];
        RGB rgb(pixel.rgbSum[0], pixel.rgbSum[1], pixel.rgbSum[2]);
        RGB albedoRgb(pixel.rgbAlbedoSum[0], pixel.rgbAlbedoSum[1],
                      pixel.rgbAlbedoSum[2]);

        // Normalize pixel with weight sum
        Float weightSum = pixel.weightSum, gBufferWeightSum = pixel.gBufferWeightSum;
        Point3f pt = pixel.pSum;
        Point2f uv = pixel.uvSum;
        Float dzdx = pixel.dzdxSum, dzdy = pixel.dzdySum;
        if (weightSum != 0) {
            rgb /= weightSum;
            albedoRgb /= weightSum;
        }
        if (gBufferWeightSum != 0) {
            pt /= gBufferWeightSum;
            uv /= gBufferWeightSum;
            dzdx /= gBufferWeightSum;
            dzdy /= gBufferWeightSum;
        }

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            rgb[c] += splatScale * pixel.rgbSplat[c] / filterIntegral;

        rgb = outputRGBFromSensorRGB * rgb;

        if (writeFP16 && std::max({rgb.r, rgb.g, rgb.b}) > 65504) {
            if (rgb.r > 65504)
                rgb.r = 65504;
            if (rgb.g > 65504)
                rgb.g = 65504;
            if (rgb.b > 65504)
                rgb.b = 65504;
            ++nClamped;
        }

        Point2i pOffset(p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, rgbDesc, {rgb[0], rgb[1], rgb[2]});
        image.SetChannels(pOffset, albedoRgbDesc,
                          {albedoRgb[0], albedoRgb[1], albedoRgb[2]});

        Normal3f n =
            LengthSquared(pixel.nSum) > 0 ? Normalize(pixel.nSum) : Normal3f(0, 0, 0);
        Normal3f ns =
            LengthSquared(pixel.nsSum) > 0 ? Normalize(pixel.nsSum) : Normal3f(0, 0, 0);
        image.SetChannels(pOffset, pDesc, {pt.x, pt.y, pt.z});
        image.SetChannels(pOffset, dzDesc, {std::abs(dzdx), std::abs(dzdy)});
        image.SetChannels(pOffset, nDesc, {n.x, n.y, n.z});
        image.SetChannels(pOffset, nsDesc, {ns.x, ns.y, ns.z});
        image.SetChannels(pOffset, uvDesc, {uv[0], uv[1]});
        image.SetChannels(
            pOffset, varianceDesc,
            {pixel.rgbVariance[0].Variance(), pixel.rgbVariance[1].Variance(),
             pixel.rgbVariance[2].Variance()});
        image.SetChannels(pOffset, relVarianceDesc,
                          {pixel.rgbVariance[0].RelativeVariance(),
                           pixel.rgbVariance[1].RelativeVariance(),
                           pixel.rgbVariance[2].RelativeVariance()});
    });

    if (nClamped.load() > 0)
        Warning("%d pixel values clamped to maximum fp16 value.", nClamped.load());

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;
    metadata->colorSpace = colorSpace;

    return image;
}

std::string GBufferFilm::ToString() const {
    return StringPrintf("[ GBufferFilm %s outputFromRender: %s applyInverse: %s "
                        "colorSpace: %s maxComponentValue: %f writeFP16: %s ]",
                        BaseToString(), outputFromRender, applyInverse, *colorSpace,
                        maxComponentValue, writeFP16);
}

GBufferFilm *GBufferFilm::Create(const ParameterDictionary &parameters,
                                 Float exposureTime,
                                 const CameraTransform &cameraTransform, Filter filter,
                                 const RGBColorSpace *colorSpace, const FileLoc *loc,
                                 Allocator alloc) {
    Float maxComponentValue = parameters.GetOneFloat("maxcomponentvalue", Infinity);
    bool writeFP16 = parameters.GetOneBool("savefp16", true);

    PixelSensor *sensor =
        PixelSensor::Create(parameters, colorSpace, exposureTime, loc, alloc);

    FilmBaseParameters filmBaseParameters(parameters, filter, sensor, loc);

    if (!HasExtension(filmBaseParameters.filename, "exr"))
        ErrorExit(loc, "%s: EXR is the only format supported by the GBufferFilm.",
                  filmBaseParameters.filename);

    std::string coordinateSystem = parameters.GetOneString("coordinatesystem", "camera");
    AnimatedTransform outputFromRender;
    bool applyInverse = false;
    if (coordinateSystem == "camera") {
        outputFromRender = cameraTransform.RenderFromCamera();
        applyInverse = true;
    } else if (coordinateSystem == "world")
        outputFromRender = AnimatedTransform(cameraTransform.WorldFromRender());
    else
        ErrorExit(loc,
                  "%s: unknown coordinate system for GBufferFilm. (Expecting \"camera\" "
                  "or \"world\".)",
                  coordinateSystem);

    return alloc.new_object<GBufferFilm>(filmBaseParameters, outputFromRender,
                                         applyInverse, colorSpace, maxComponentValue,
                                         writeFP16, alloc);
}

// SpectralFilm Method Definitions
SpectralFilm::SpectralFilm(FilmBaseParameters p, Float lambdaMin, Float lambdaMax,
                           int nBuckets, const RGBColorSpace *colorSpace,
                           Float maxComponentValue, bool writeFP16, Allocator alloc)
    : FilmBase(p),
      colorSpace(colorSpace),
      lambdaMin(lambdaMin),
      lambdaMax(lambdaMax),
      nBuckets(nBuckets),
      maxComponentValue(maxComponentValue),
      writeFP16(writeFP16),
      pixels(p.pixelBounds, alloc) {
    // Compute _outputRGBFromSensorRGB_ matrix
    outputRGBFromSensorRGB = colorSpace->RGBFromXYZ * sensor->XYZFromSensorRGB;

    filterIntegral = filter.Integral();
    CHECK(!pixelBounds.IsEmpty());
    filmPixelMemory +=
        pixelBounds.Area() * (sizeof(Pixel) + 3 * nBuckets * sizeof(double));

    // Allocate memory for the pixel buffers in big arrays. Note that it's
    // wasteful (but convenient) to be storing three pointers in each
    // SpectralFilm::Pixel structure since the addresses could be computed
    // based on the base pointers and pixel coordinates.
    int nPixels = pixelBounds.Area();
    double *bucketWeightBuffer = alloc.allocate_object<double>(2 * nBuckets * nPixels);
    std::memset(bucketWeightBuffer, 0, 2 * nBuckets * nPixels * sizeof(double));
    AtomicDouble *splatBuffer = alloc.allocate_object<AtomicDouble>(nBuckets * nPixels);
    std::memset(splatBuffer, 0, nBuckets * nPixels * sizeof(double));

    for (Point2i p : pixelBounds) {
        Pixel &pixel = pixels[p];
        pixel.bucketSums = bucketWeightBuffer;
        bucketWeightBuffer += nBuckets;
        pixel.weightSums = bucketWeightBuffer;
        bucketWeightBuffer += nBuckets;
        pixel.bucketSplats = splatBuffer;
        splatBuffer += nBuckets;
    }
}

PBRT_CPU_GPU RGB SpectralFilm::GetPixelRGB(Point2i p, Float splatScale) const {
    // Note: this is effectively the same as RGBFilm::GetPixelRGB

    const Pixel &pixel = pixels[p];
    RGB rgb(pixel.rgbSum[0], pixel.rgbSum[1], pixel.rgbSum[2]);
    // Normalize _rgb_ with weight sum
    Float weightSum = pixel.rgbWeightSum;
    if (weightSum != 0)
        rgb /= weightSum;

    // Add splat value at pixel
    for (int c = 0; c < 3; ++c)
        rgb[c] += splatScale * pixel.rgbSplat[c] / filterIntegral;

    // Convert _rgb_ to output RGB color space
    rgb = outputRGBFromSensorRGB * rgb;

    return rgb;
}

PBRT_CPU_GPU void SpectralFilm::AddSplat(Point2f p, SampledSpectrum L,
                            const SampledWavelengths &lambda) {
    // This, too, is similar to RGBFilm::AddSplat(), with additions for
    // spectra.

    CHECK(!L.HasNaNs());

    // Convert sample radiance to _PixelSensor_ RGB
    RGB rgb = sensor->ToSensorRGB(L, lambda);

    // Optionally clamp sensor RGB value
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue)
        rgb *= maxComponentValue / m;

    // Spectral clamping and normalization.
    Float lm = L.MaxComponentValue();
    if (lm > maxComponentValue)
        L *= maxComponentValue / lm;
    L = SafeDiv(L, lambda.PDF()) / NSpectrumSamples;

    // Compute bounds of affected pixels for splat, _splatBounds_
    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Vector2f radius = filter.Radius();
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - radius)),
                         Point2i(Floor(pDiscrete + radius)) + Vector2i(1, 1));
    splatBounds = Intersect(splatBounds, pixelBounds);

    // Splat both RGB and spectral bucket contributions.
    for (Point2i pi : splatBounds) {
        // Evaluate filter at _pi_ and add splat contribution
        Float wt = filter.Evaluate(Point2f(p - pi - Vector2f(0.5, 0.5)));
        if (wt != 0) {
            Pixel &pixel = pixels[pi];

            for (int i = 0; i < 3; ++i)
                pixel.rgbSplat[i].Add(wt * rgb[i]);

            for (int i = 0; i < NSpectrumSamples; ++i) {
                int b = LambdaToBucket(lambda[i]);
                pixel.bucketSplats[b].Add(wt * L[i]);
            }
        }
    }
}

void SpectralFilm::WriteImage(ImageMetadata metadata, Float splatScale) {
    Image image = GetImage(&metadata, splatScale);
    LOG_VERBOSE("Writing image %s with bounds %s", filename, pixelBounds);
    image.Write(filename, metadata);
}

Image SpectralFilm::GetImage(ImageMetadata *metadata, Float splatScale) {
    // Convert image to RGB and compute final pixel values
    LOG_VERBOSE("Computing final weighted pixel values");
    PixelFormat format = writeFP16 ? PixelFormat::Half : PixelFormat::Float;

    std::vector<std::string> imageChannels{{"R", "G", "B"}};
    for (int i = 0; i < nBuckets; ++i) {
        // The OpenEXR spectral layout takes the bucket center (and then
        // determines bucket widths based on the neighbor wavelengths).
        std::string lambda =
            StringPrintf("%.3fnm", Lerp((i + 0.5f) / nBuckets, lambdaMin, lambdaMax));
        // Convert any '.' to ',' in the number since OpenEXR uses '.' for
        // separating layers.
        std::replace(lambda.begin(), lambda.end(), '.', ',');

        imageChannels.push_back("S0." + lambda);
    }
    Image image(format, Point2i(pixelBounds.Diagonal()), imageChannels);

    std::atomic<int> nClamped{0};
    ParallelFor2D(pixelBounds, [&](Point2i p) {
        Pixel &pixel = pixels[p];

        RGB rgb = GetPixelRGB(p, splatScale);

        // Clamp to max representable fp16 to avoid Infs
        if (writeFP16) {
            for (int c = 0; c < 3; ++c) {
                if (rgb[c] > 65504) {
                    rgb[c] = 65504;
                    ++nClamped;
                }
            }
        }

        Point2i pOffset(p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, {rgb[0], rgb[1], rgb[2]});

        // Set spectral channels. Hardcoded assuming that they come
        // immediately after RGB, as is currently specified above.
        for (int i = 0; i < nBuckets; ++i) {
            Float c = 0;
            if (pixel.weightSums[i] > 0) {
                c = pixel.bucketSums[i] / pixel.weightSums[i] +
                    splatScale * pixel.bucketSplats[i] / filterIntegral;
                if (writeFP16 && c > 65504) {
                    c = 65504;
                    ++nClamped;
                }
            }
            image.SetChannel(pOffset, 3 + i, c);
        }
    });

    if (nClamped.load() > 0)
        Warning("%d pixel values clamped to maximum fp16 value.", nClamped.load());

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;
    metadata->colorSpace = colorSpace;
    metadata->strings["spectralLayoutVersion"] = "1.0";
    // FIXME: if the RealisticCamera is being used, then we're actually
    // storing "J.m^-2", but that isn't a supported value for
    // "emissiveUnits" in the spec.
    metadata->strings["emissiveUnits"] = "W.m^-2.sr^-1";

    return image;
}

std::string SpectralFilm::ToString() const {
    return StringPrintf("[ SpectralFilm %s lambdaMin: %f lambdaMax: %f nBuckets: %d "
                        "writeFP16: %s maxComponentValue: %f ]",
                        BaseToString(), lambdaMin, lambdaMax, nBuckets, writeFP16,
                        maxComponentValue);
}

SpectralFilm *SpectralFilm::Create(const ParameterDictionary &parameters,
                                   Float exposureTime, Filter filter,
                                   const RGBColorSpace *colorSpace, const FileLoc *loc,
                                   Allocator alloc) {
    PixelSensor *sensor =
        PixelSensor::Create(parameters, colorSpace, exposureTime, loc, alloc);
    FilmBaseParameters filmBaseParameters(parameters, filter, sensor, loc);
    bool writeFP16 = parameters.GetOneBool("savefp16", true);

    if (!HasExtension(filmBaseParameters.filename, "exr"))
        ErrorExit(loc, "%s: EXR is the only output format supported by the SpectralFilm.",
                  filmBaseParameters.filename);

    int nBuckets = parameters.GetOneInt("nbuckets", 16);
    Float lambdaMin = parameters.GetOneFloat("lambdamin", Lambda_min);
    Float lambdaMax = parameters.GetOneFloat("lambdamax", Lambda_max);
    if (lambdaMin < Lambda_min || lambdaMax > Lambda_max)
        ErrorExit("Unfortunately pbrt must be recompiled to render wavelengths "
                  "beyond the [%f,%f] range ([%f,%f] was specified). Please "
                  "update Lambda_min and/or Lambda_max as necessary in "
                  "src/pbrt/util/spectrum.h and recompile.", Lambda_min, Lambda_max,
                  lambdaMin, lambdaMax);

    Float maxComponentValue = parameters.GetOneFloat("maxcomponentvalue", Infinity);

    return alloc.new_object<SpectralFilm>(filmBaseParameters, lambdaMin, lambdaMax,
                                          nBuckets, colorSpace, maxComponentValue,
                                          writeFP16, alloc);
}

Film Film::Create(const std::string &name, const ParameterDictionary &parameters,
                  Float exposureTime, const CameraTransform &cameraTransform,
                  Filter filter, const FileLoc *loc, Allocator alloc) {
    Film film;
    if (name == "rgb")
        film = RGBFilm::Create(parameters, exposureTime, filter, parameters.ColorSpace(),
                               loc, alloc);
    else if (name == "gbuffer")
        film = GBufferFilm::Create(parameters, exposureTime, cameraTransform, filter,
                                   parameters.ColorSpace(), loc, alloc);
    else if (name == "spectral")
        film = SpectralFilm::Create(parameters, exposureTime, filter,
                                    parameters.ColorSpace(), loc, alloc);
    else
        ErrorExit(loc, "%s: film type unknown.", name);

    if (!film)
        ErrorExit(loc, "%s: unable to create film.", name);

    parameters.ReportUnused();
    return film;
}

}  // namespace pbrt
