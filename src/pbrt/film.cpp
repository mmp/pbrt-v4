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
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/transform.h>

namespace pbrt {

void FilmHandle::AddSplat(const Point2f &p, SampledSpectrum v,
                          const SampledWavelengths &lambda) {
    auto splat = [&](auto ptr) { return ptr->AddSplat(p, v, lambda); };
    return Dispatch(splat);
}

void FilmHandle::WriteImage(ImageMetadata metadata, Float splatScale) {
    auto write = [&](auto ptr) { return ptr->WriteImage(metadata, splatScale); };
    return DispatchCPU(write);
}

Image FilmHandle::GetImage(ImageMetadata *metadata, Float splatScale) {
    auto get = [&](auto ptr) { return ptr->GetImage(metadata, splatScale); };
    return DispatchCPU(get);
}

std::string FilmHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

std::string FilmHandle::GetFilename() const {
    auto get = [&](auto ptr) { return ptr->GetFilename(); };
    return DispatchCPU(get);
}

// FilmBaseParameters Method Definitions
FilmBaseParameters::FilmBaseParameters(const ParameterDictionary &parameters,
                                       FilterHandle filter, const PixelSensor *sensor,
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

    fullResolution = Point2i(parameters.GetOneInt("xresolution", 1280),
                             parameters.GetOneInt("yresolution", 720));
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
        // Compute film image bounds
        pixelBounds = Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                       std::ceil(fullResolution.y * crop.pMin.y)),
                               Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                       std::ceil(fullResolution.y * crop.pMax.y)));

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
            pixelBounds = Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                           std::ceil(fullResolution.y * crop.pMin.y)),
                                   Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                           std::ceil(fullResolution.y * crop.pMax.y)));
        } else
            Error(loc, "%d values supplied for \"cropwindow\". Expected 4.",
                  (int)cr.size());
    }

    if (pixelBounds.IsEmpty())
        ErrorExit(loc, "Degenerate pixel bounds provided to film: %s.", pixelBounds);

    diagonal = parameters.GetOneFloat("diagonal", 35.);
}

// FilmBase Method Definitions
std::string FilmBase::BaseToString() const {
    return StringPrintf("fullResolution: %s diagonal: %f filter: %s filename: %s "
                        "pixelBounds: %s",
                        fullResolution, diagonal, filter, filename, pixelBounds);
}

Bounds2f FilmBase::SampleBounds() const {
    return Bounds2f(Point2f(pixelBounds.pMin) - filter.Radius() + Vector2f(0.5f, 0.5f),
                    Point2f(pixelBounds.pMax) + filter.Radius() - Vector2f(0.5f, 0.5f));
}

// VisibleSurface Method Definitions
VisibleSurface::VisibleSurface(const SurfaceInteraction &si,
                               const CameraTransform &cameraTransform,
                               const SampledSpectrum &albedo,
                               const SampledWavelengths &lambda)
    : albedo(albedo) {
    set = true;
    // Initialize geometric _VisibleSurface_ members
    Transform cameraFromRender = cameraTransform.CameraFromRender(si.time);
    p = cameraFromRender(si.p());
    Vector3f wo = cameraFromRender(si.wo);
    n = FaceForward(cameraFromRender(si.n), wo);
    ns = FaceForward(cameraFromRender(si.shading.n), wo);
    time = si.time;
    dzdx = cameraFromRender(si.dpdx).z;
    dzdy = cameraFromRender(si.dpdy).z;
}

std::string VisibleSurface::ToString() const {
    return StringPrintf("[ VisibleSurface set: %s p: %s n: %s ns: %s dzdx: %f dzdy: %f "
                        "time: %f albedo: %s ]",
                        set, p, n, ns, dzdx, dzdy, time, albedo);
}

// PixelSensor Method Definitions
PixelSensor *PixelSensor::Create(const ParameterDictionary &parameters,
                                 const RGBColorSpace *colorSpace, Float exposureTime,
                                 const FileLoc *loc, Allocator alloc) {
    // Imaging ratio parameters
    // The defaults here represent a "passthrough" setup such that the imaging
    // ratio will be exactly 1. This is a useful default since scenes that
    // weren't authored with a physical camera in mind will render as expected.
    Float fNumber = parameters.GetOneFloat("fnumber", 1.);
    Float ISO = parameters.GetOneFloat("iso", 100.);
    // Note: in the talk we mention using 312.5 for historical reasons. The
    // choice of 100 * Pi here just means that the other parameters make nice
    // "round" numbers like 1 and 100.
    Float C = parameters.GetOneFloat("c", 100.0 * Pi);
    Float whiteBalanceTemp = parameters.GetOneFloat("whitebalance", 0);

    std::string sensorName = parameters.GetOneString("sensor", "cie1931");

    // Pass through 0 for cie1931 if it's unspecified so that it doesn't do
    // any white balancing. For actual sensors, 6500 is the default...
    if (sensorName != "cie1931" && whiteBalanceTemp == 0)
        whiteBalanceTemp = 6500;

    Float imagingRatio = Pi * exposureTime * ISO * K_m / (C * fNumber * fNumber);

    if (sensorName == "cie1931") {
        return alloc.new_object<PixelSensor>(colorSpace, whiteBalanceTemp, imagingRatio,
                                             alloc);
    } else {
        SpectrumHandle r = GetNamedSpectrum(sensorName + "_r");
        SpectrumHandle g = GetNamedSpectrum(sensorName + "_g");
        SpectrumHandle b = GetNamedSpectrum(sensorName + "_b");

        if (!r || !g || !b)
            ErrorExit(loc, "%s: unknown sensor type", sensorName);

        return alloc.new_object<PixelSensor>(r, g, b, colorSpace, whiteBalanceTemp,
                                             imagingRatio, alloc);
    }
}

PixelSensor *PixelSensor::CreateDefault(Allocator alloc) {
    return Create(ParameterDictionary(), RGBColorSpace::sRGB, 1.0, nullptr, alloc);
}

pstd::optional<SquareMatrix<3>> PixelSensor::SolveXYZFromSensorRGB(
    SpectrumHandle sensorIllum, SpectrumHandle outputIllum) const {
    Float rgbCamera[24][3], xyzOutput[24][3];
    // Compute _rgbCamera_ values for training swatches
    RGB outputWhite = IlluminantToSensorRGB(outputIllum);
    for (size_t i = 0; i < swatchReflectances.size(); ++i) {
        RGB rgb = ProjectReflectance<RGB>(swatchReflectances[i], sensorIllum, &r_bar,
                                          &g_bar, &b_bar) /
                  outputWhite;
        for (int c = 0; c < 3; ++c)
            rgbCamera[i][c] = rgb[c];
    }

    // Compute _xyzOutput_ values for training swatches
    Float sensorWhiteG = InnerProduct(sensorIllum, &g_bar);
    Float sensorWhiteY = InnerProduct(sensorIllum, &Spectra::Y());
    for (size_t i = 0; i < swatchReflectances.size(); ++i) {
        SpectrumHandle s = swatchReflectances[i];
        XYZ xyz = ProjectReflectance<XYZ>(s, outputIllum, &Spectra::X(), &Spectra::Y(),
                                          &Spectra::Z()) *
                  (sensorWhiteY / sensorWhiteG);
        for (int c = 0; c < 3; ++c)
            xyzOutput[i][c] = xyz[c];
    }

    return LinearLeastSquares(rgbCamera, xyzOutput, swatchReflectances.size());
}

std::vector<SpectrumHandle> PixelSensor::swatchReflectances{
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.051500, 390.000000, 0.056500, 400.000000, 0.063000,
         410.000000, 0.065000, 420.000000, 0.063000, 430.000000, 0.060500,
         440.000000, 0.058500, 450.000000, 0.057500, 460.000000, 0.057000,
         470.000000, 0.057000, 480.000000, 0.058000, 490.000000, 0.060000,
         500.000000, 0.063000, 510.000000, 0.067500, 520.000000, 0.073000,
         530.000000, 0.076500, 540.000000, 0.078500, 550.000000, 0.081500,
         560.000000, 0.089000, 570.000000, 0.101500, 580.000000, 0.117000,
         590.000000, 0.131500, 600.000000, 0.140500, 610.000000, 0.146500,
         620.000000, 0.152500, 630.000000, 0.160500, 640.000000, 0.170500,
         650.000000, 0.183500, 660.000000, 0.196000, 670.000000, 0.206000,
         680.000000, 0.214000, 690.000000, 0.221000, 700.000000, 0.232000,
         710.000000, 0.246000, 720.000000, 0.265000, 730.000000, 0.290500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.110000, 390.000000, 0.142000, 400.000000, 0.178500,
         410.000000, 0.194000, 420.000000, 0.198500, 430.000000, 0.202000,
         440.000000, 0.208000, 450.000000, 0.218500, 460.000000, 0.234000,
         470.000000, 0.256500, 480.000000, 0.281000, 490.000000, 0.301000,
         500.000000, 0.315000, 510.000000, 0.327000, 520.000000, 0.318500,
         530.000000, 0.292000, 540.000000, 0.282500, 550.000000, 0.288000,
         560.000000, 0.286000, 570.000000, 0.297000, 580.000000, 0.348500,
         590.000000, 0.427500, 600.000000, 0.491000, 610.000000, 0.527500,
         620.000000, 0.548000, 630.000000, 0.563000, 640.000000, 0.576000,
         650.000000, 0.592500, 660.000000, 0.608500, 670.000000, 0.624500,
         680.000000, 0.645000, 690.000000, 0.669000, 700.000000, 0.695500,
         710.000000, 0.722500, 720.000000, 0.739500, 730.000000, 0.758500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.121500, 390.000000, 0.175500, 400.000000, 0.258500,
         410.000000, 0.313000, 420.000000, 0.330000, 430.000000, 0.333500,
         440.000000, 0.334000, 450.000000, 0.331000, 460.000000, 0.322500,
         470.000000, 0.310500, 480.000000, 0.295500, 490.000000, 0.280500,
         500.000000, 0.264500, 510.000000, 0.246500, 520.000000, 0.228000,
         530.000000, 0.211000, 540.000000, 0.198500, 550.000000, 0.188000,
         560.000000, 0.176000, 570.000000, 0.164500, 580.000000, 0.156000,
         590.000000, 0.151000, 600.000000, 0.146000, 610.000000, 0.142500,
         620.000000, 0.139000, 630.000000, 0.135500, 640.000000, 0.133000,
         650.000000, 0.132500, 660.000000, 0.132000, 670.000000, 0.131000,
         680.000000, 0.127500, 690.000000, 0.124000, 700.000000, 0.119500,
         710.000000, 0.117000, 720.000000, 0.118500, 730.000000, 0.124500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.049500, 390.000000, 0.051500, 400.000000, 0.053000,
         410.000000, 0.053000, 420.000000, 0.054000, 430.000000, 0.055500,
         440.000000, 0.057000, 450.000000, 0.059500, 460.000000, 0.061500,
         470.000000, 0.063500, 480.000000, 0.066000, 490.000000, 0.068500,
         500.000000, 0.076500, 510.000000, 0.103500, 520.000000, 0.150000,
         530.000000, 0.179500, 540.000000, 0.180500, 550.000000, 0.163500,
         560.000000, 0.143000, 570.000000, 0.129500, 580.000000, 0.122000,
         590.000000, 0.115000, 600.000000, 0.106500, 610.000000, 0.101500,
         620.000000, 0.101000, 630.000000, 0.103000, 640.000000, 0.103000,
         650.000000, 0.103500, 660.000000, 0.104500, 670.000000, 0.107500,
         680.000000, 0.118500, 690.000000, 0.143000, 700.000000, 0.181000,
         710.000000, 0.215500, 720.000000, 0.234000, 730.000000, 0.244000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.133500, 390.000000, 0.197500, 400.000000, 0.311000,
         410.000000, 0.396500, 420.000000, 0.427000, 430.000000, 0.434500,
         440.000000, 0.435000, 450.000000, 0.430000, 460.000000, 0.420000,
         470.000000, 0.404000, 480.000000, 0.380000, 490.000000, 0.354000,
         500.000000, 0.326500, 510.000000, 0.297500, 520.000000, 0.262500,
         530.000000, 0.230000, 540.000000, 0.212500, 550.000000, 0.208500,
         560.000000, 0.201500, 570.000000, 0.195000, 580.000000, 0.199500,
         590.000000, 0.211500, 600.000000, 0.224500, 610.000000, 0.237500,
         620.000000, 0.242000, 630.000000, 0.250500, 640.000000, 0.274000,
         650.000000, 0.315500, 660.000000, 0.366000, 670.000000, 0.406000,
         680.000000, 0.428000, 690.000000, 0.435000, 700.000000, 0.439000,
         710.000000, 0.444000, 720.000000, 0.449000, 730.000000, 0.460000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.123000, 390.000000, 0.173000, 400.000000, 0.249500,
         410.000000, 0.300000, 420.000000, 0.321000, 430.000000, 0.336500,
         440.000000, 0.354000, 450.000000, 0.378000, 460.000000, 0.413500,
         470.000000, 0.463000, 480.000000, 0.516500, 490.000000, 0.556000,
         500.000000, 0.574500, 510.000000, 0.577000, 520.000000, 0.569000,
         530.000000, 0.550000, 540.000000, 0.521500, 550.000000, 0.484000,
         560.000000, 0.440500, 570.000000, 0.396000, 580.000000, 0.348000,
         590.000000, 0.300500, 600.000000, 0.256000, 610.000000, 0.227500,
         620.000000, 0.212500, 630.000000, 0.205500, 640.000000, 0.200500,
         650.000000, 0.198000, 660.000000, 0.201000, 670.000000, 0.209500,
         680.000000, 0.222500, 690.000000, 0.233500, 700.000000, 0.242000,
         710.000000, 0.241500, 720.000000, 0.236500, 730.000000, 0.237000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.053500, 390.000000, 0.054000, 400.000000, 0.053500,
         410.000000, 0.053500, 420.000000, 0.053000, 430.000000, 0.053500,
         440.000000, 0.053500, 450.000000, 0.053500, 460.000000, 0.054000,
         470.000000, 0.055000, 480.000000, 0.056500, 490.000000, 0.059000,
         500.000000, 0.064500, 510.000000, 0.078500, 520.000000, 0.105500,
         530.000000, 0.137000, 540.000000, 0.172000, 550.000000, 0.213500,
         560.000000, 0.272500, 570.000000, 0.357500, 580.000000, 0.448000,
         590.000000, 0.520500, 600.000000, 0.559500, 610.000000, 0.573500,
         620.000000, 0.578500, 630.000000, 0.582500, 640.000000, 0.585500,
         650.000000, 0.591500, 660.000000, 0.597000, 670.000000, 0.605500,
         680.000000, 0.617500, 690.000000, 0.630500, 700.000000, 0.641500,
         710.000000, 0.648000, 720.000000, 0.648000, 730.000000, 0.652000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.110500, 390.000000, 0.157000, 400.000000, 0.230000,
         410.000000, 0.289500, 420.000000, 0.325500, 430.000000, 0.354500,
         440.000000, 0.380500, 450.000000, 0.393500, 460.000000, 0.383000,
         470.000000, 0.353500, 480.000000, 0.308500, 490.000000, 0.253500,
         500.000000, 0.205500, 510.000000, 0.167500, 520.000000, 0.137500,
         530.000000, 0.116500, 540.000000, 0.105000, 550.000000, 0.097500,
         560.000000, 0.090000, 570.000000, 0.084500, 580.000000, 0.083500,
         590.000000, 0.085000, 600.000000, 0.085500, 610.000000, 0.084500,
         620.000000, 0.084000, 630.000000, 0.086500, 640.000000, 0.094000,
         650.000000, 0.104500, 660.000000, 0.116000, 670.000000, 0.124500,
         680.000000, 0.131500, 690.000000, 0.142000, 700.000000, 0.160500,
         710.000000, 0.182000, 720.000000, 0.207500, 730.000000, 0.235500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.096000, 390.000000, 0.119000, 400.000000, 0.137500,
         410.000000, 0.139500, 420.000000, 0.135500, 430.000000, 0.132000,
         440.000000, 0.130500, 450.000000, 0.128500, 460.000000, 0.125500,
         470.000000, 0.122500, 480.000000, 0.117000, 490.000000, 0.109500,
         500.000000, 0.104000, 510.000000, 0.100000, 520.000000, 0.094500,
         530.000000, 0.091000, 540.000000, 0.092000, 550.000000, 0.097500,
         560.000000, 0.102000, 570.000000, 0.110000, 580.000000, 0.156500,
         590.000000, 0.269500, 600.000000, 0.407000, 610.000000, 0.508500,
         620.000000, 0.561500, 630.000000, 0.585000, 640.000000, 0.594500,
         650.000000, 0.599000, 660.000000, 0.600000, 670.000000, 0.599500,
         680.000000, 0.601500, 690.000000, 0.603500, 700.000000, 0.606500,
         710.000000, 0.605500, 720.000000, 0.604000, 730.000000, 0.603000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.096500, 390.000000, 0.125500, 400.000000, 0.161500,
         410.000000, 0.184000, 420.000000, 0.192000, 430.000000, 0.181500,
         440.000000, 0.163000, 450.000000, 0.141500, 460.000000, 0.119500,
         470.000000, 0.101000, 480.000000, 0.086500, 490.000000, 0.075000,
         500.000000, 0.066500, 510.000000, 0.060500, 520.000000, 0.057000,
         530.000000, 0.053500, 540.000000, 0.051500, 550.000000, 0.052000,
         560.000000, 0.053500, 570.000000, 0.053000, 580.000000, 0.051500,
         590.000000, 0.052500, 600.000000, 0.058500, 610.000000, 0.073500,
         620.000000, 0.097500, 630.000000, 0.122500, 640.000000, 0.145000,
         650.000000, 0.169000, 660.000000, 0.193500, 670.000000, 0.222000,
         680.000000, 0.256500, 690.000000, 0.295500, 700.000000, 0.337000,
         710.000000, 0.375500, 720.000000, 0.412000, 730.000000, 0.450000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.058500, 390.000000, 0.060000, 400.000000, 0.061000,
         410.000000, 0.062000, 420.000000, 0.063000, 430.000000, 0.065000,
         440.000000, 0.068500, 450.000000, 0.075000, 460.000000, 0.085000,
         470.000000, 0.104500, 480.000000, 0.137000, 490.000000, 0.188500,
         500.000000, 0.270000, 510.000000, 0.380000, 520.000000, 0.480000,
         530.000000, 0.532500, 540.000000, 0.547000, 550.000000, 0.539500,
         560.000000, 0.520500, 570.000000, 0.495500, 580.000000, 0.462500,
         590.000000, 0.422000, 600.000000, 0.377500, 610.000000, 0.346500,
         620.000000, 0.329000, 630.000000, 0.321500, 640.000000, 0.316000,
         650.000000, 0.314500, 660.000000, 0.319000, 670.000000, 0.332000,
         680.000000, 0.349500, 690.000000, 0.365500, 700.000000, 0.377500,
         710.000000, 0.380000, 720.000000, 0.375500, 730.000000, 0.377000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.061500, 390.000000, 0.063000, 400.000000, 0.064000,
         410.000000, 0.064000, 420.000000, 0.064000, 430.000000, 0.064000,
         440.000000, 0.065000, 450.000000, 0.066500, 460.000000, 0.068000,
         470.000000, 0.072500, 480.000000, 0.081500, 490.000000, 0.091500,
         500.000000, 0.105000, 510.000000, 0.135500, 520.000000, 0.199500,
         530.000000, 0.289000, 540.000000, 0.378500, 550.000000, 0.443500,
         560.000000, 0.490500, 570.000000, 0.536000, 580.000000, 0.576000,
         590.000000, 0.608000, 600.000000, 0.626000, 610.000000, 0.636000,
         620.000000, 0.642500, 630.000000, 0.648000, 640.000000, 0.653500,
         650.000000, 0.660000, 660.000000, 0.663500, 670.000000, 0.665500,
         680.000000, 0.671000, 690.000000, 0.677000, 700.000000, 0.684500,
         710.000000, 0.689000, 720.000000, 0.691000, 730.000000, 0.694500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.067500, 390.000000, 0.087500, 400.000000, 0.119000,
         410.000000, 0.160500, 420.000000, 0.204000, 430.000000, 0.244000,
         440.000000, 0.286000, 450.000000, 0.322000, 460.000000, 0.323000,
         470.000000, 0.290000, 480.000000, 0.235000, 490.000000, 0.175000,
         500.000000, 0.125000, 510.000000, 0.090000, 520.000000, 0.068000,
         530.000000, 0.054000, 540.000000, 0.047000, 550.000000, 0.043500,
         560.000000, 0.041000, 570.000000, 0.039500, 580.000000, 0.039000,
         590.000000, 0.039000, 600.000000, 0.038500, 610.000000, 0.039500,
         620.000000, 0.039500, 630.000000, 0.040500, 640.000000, 0.041500,
         650.000000, 0.042000, 660.000000, 0.043500, 670.000000, 0.044500,
         680.000000, 0.045000, 690.000000, 0.045500, 700.000000, 0.048000,
         710.000000, 0.051500, 720.000000, 0.056500, 730.000000, 0.064500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.053500, 390.000000, 0.055000, 400.000000, 0.056000,
         410.000000, 0.057000, 420.000000, 0.058000, 430.000000, 0.060500,
         440.000000, 0.063000, 450.000000, 0.068000, 460.000000, 0.076500,
         470.000000, 0.092000, 480.000000, 0.119000, 490.000000, 0.159000,
         500.000000, 0.213000, 510.000000, 0.275500, 520.000000, 0.330500,
         530.000000, 0.348500, 540.000000, 0.336000, 550.000000, 0.308000,
         560.000000, 0.271500, 570.000000, 0.234000, 580.000000, 0.197000,
         590.000000, 0.161500, 600.000000, 0.129500, 610.000000, 0.108000,
         620.000000, 0.096500, 630.000000, 0.091000, 640.000000, 0.087000,
         650.000000, 0.084500, 660.000000, 0.084000, 670.000000, 0.086000,
         680.000000, 0.090000, 690.000000, 0.095000, 700.000000, 0.100500,
         710.000000, 0.102000, 720.000000, 0.102000, 730.000000, 0.101500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.051000, 390.000000, 0.050500, 400.000000, 0.049500,
         410.000000, 0.048500, 420.000000, 0.048000, 430.000000, 0.048000,
         440.000000, 0.048000, 450.000000, 0.048000, 460.000000, 0.047000,
         470.000000, 0.046000, 480.000000, 0.044500, 490.000000, 0.044000,
         500.000000, 0.044500, 510.000000, 0.045000, 520.000000, 0.045500,
         530.000000, 0.046000, 540.000000, 0.047000, 550.000000, 0.048500,
         560.000000, 0.052000, 570.000000, 0.058500, 580.000000, 0.072000,
         590.000000, 0.106500, 600.000000, 0.185000, 610.000000, 0.322000,
         620.000000, 0.476500, 630.000000, 0.589500, 640.000000, 0.649000,
         650.000000, 0.680500, 660.000000, 0.695000, 670.000000, 0.702500,
         680.000000, 0.712000, 690.000000, 0.719500, 700.000000, 0.726500,
         710.000000, 0.730000, 720.000000, 0.730500, 730.000000, 0.733500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.056000, 390.000000, 0.054000, 400.000000, 0.052500,
         410.000000, 0.052500, 420.000000, 0.052500, 430.000000, 0.053500,
         440.000000, 0.054500, 450.000000, 0.057000, 460.000000, 0.063000,
         470.000000, 0.078000, 480.000000, 0.114000, 490.000000, 0.177000,
         500.000000, 0.264000, 510.000000, 0.365000, 520.000000, 0.468500,
         530.000000, 0.551500, 540.000000, 0.606500, 550.000000, 0.640000,
         560.000000, 0.666000, 570.000000, 0.690000, 580.000000, 0.709000,
         590.000000, 0.724500, 600.000000, 0.734000, 610.000000, 0.742500,
         620.000000, 0.749500, 630.000000, 0.756500, 640.000000, 0.763000,
         650.000000, 0.770500, 660.000000, 0.774500, 670.000000, 0.776000,
         680.000000, 0.780500, 690.000000, 0.784500, 700.000000, 0.791000,
         710.000000, 0.794500, 720.000000, 0.794500, 730.000000, 0.798000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.131500, 390.000000, 0.187000, 400.000000, 0.283000,
         410.000000, 0.344500, 420.000000, 0.360500, 430.000000, 0.352000,
         440.000000, 0.330500, 450.000000, 0.302000, 460.000000, 0.271500,
         470.000000, 0.243500, 480.000000, 0.213500, 490.000000, 0.186000,
         500.000000, 0.165500, 510.000000, 0.147500, 520.000000, 0.125500,
         530.000000, 0.106500, 540.000000, 0.101000, 550.000000, 0.104500,
         560.000000, 0.105000, 570.000000, 0.110500, 580.000000, 0.139000,
         590.000000, 0.199000, 600.000000, 0.284500, 610.000000, 0.397000,
         620.000000, 0.519000, 630.000000, 0.621500, 640.000000, 0.691500,
         650.000000, 0.737000, 660.000000, 0.763000, 670.000000, 0.777000,
         680.000000, 0.787000, 690.000000, 0.795500, 700.000000, 0.803500,
         710.000000, 0.809500, 720.000000, 0.812000, 730.000000, 0.819000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.100500, 390.000000, 0.137500, 400.000000, 0.193500,
         410.000000, 0.237000, 420.000000, 0.259500, 430.000000, 0.283500,
         440.000000, 0.316000, 450.000000, 0.352500, 460.000000, 0.390500,
         470.000000, 0.430000, 480.000000, 0.452000, 490.000000, 0.450500,
         500.000000, 0.428000, 510.000000, 0.388500, 520.000000, 0.338000,
         530.000000, 0.282500, 540.000000, 0.229500, 550.000000, 0.182500,
         560.000000, 0.143500, 570.000000, 0.116000, 580.000000, 0.099000,
         590.000000, 0.089000, 600.000000, 0.081000, 610.000000, 0.075500,
         620.000000, 0.073500, 630.000000, 0.073000, 640.000000, 0.073000,
         650.000000, 0.074000, 660.000000, 0.076000, 670.000000, 0.077000,
         680.000000, 0.075500, 690.000000, 0.074500, 700.000000, 0.072500,
         710.000000, 0.071500, 720.000000, 0.074500, 730.000000, 0.080500},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.171000, 390.000000, 0.250000, 400.000000, 0.416000,
         410.000000, 0.665500, 420.000000, 0.825500, 430.000000, 0.870000,
         440.000000, 0.880000, 450.000000, 0.885000, 460.000000, 0.889000,
         470.000000, 0.892000, 480.000000, 0.893500, 490.000000, 0.896000,
         500.000000, 0.897000, 510.000000, 0.898000, 520.000000, 0.899000,
         530.000000, 0.898500, 540.000000, 0.899000, 550.000000, 0.900000,
         560.000000, 0.900000, 570.000000, 0.902000, 580.000000, 0.901000,
         590.000000, 0.901000, 600.000000, 0.900500, 610.000000, 0.902000,
         620.000000, 0.904500, 630.000000, 0.905000, 640.000000, 0.905500,
         650.000000, 0.906000, 660.000000, 0.906500, 670.000000, 0.905000,
         680.000000, 0.905000, 690.000000, 0.906500, 700.000000, 0.907500,
         710.000000, 0.908000, 720.000000, 0.908000, 730.000000, 0.909000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.160500, 390.000000, 0.233500, 400.000000, 0.368500,
         410.000000, 0.518000, 420.000000, 0.573500, 430.000000, 0.584000,
         440.000000, 0.587500, 450.000000, 0.589000, 460.000000, 0.588500,
         470.000000, 0.586500, 480.000000, 0.584500, 490.000000, 0.584000,
         500.000000, 0.584500, 510.000000, 0.584500, 520.000000, 0.586000,
         530.000000, 0.586000, 540.000000, 0.586500, 550.000000, 0.586500,
         560.000000, 0.586500, 570.000000, 0.588500, 580.000000, 0.589000,
         590.000000, 0.589000, 600.000000, 0.587500, 610.000000, 0.585500,
         620.000000, 0.584000, 630.000000, 0.581500, 640.000000, 0.579000,
         650.000000, 0.577000, 660.000000, 0.575000, 670.000000, 0.573000,
         680.000000, 0.571500, 690.000000, 0.569500, 700.000000, 0.568000,
         710.000000, 0.567000, 720.000000, 0.565000, 730.000000, 0.564000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.141000, 390.000000, 0.199000, 400.000000, 0.280500,
         410.000000, 0.338500, 420.000000, 0.353500, 430.000000, 0.358000,
         440.000000, 0.361000, 450.000000, 0.362500, 460.000000, 0.362000,
         470.000000, 0.359500, 480.000000, 0.358000, 490.000000, 0.357000,
         500.000000, 0.357000, 510.000000, 0.357500, 520.000000, 0.358500,
         530.000000, 0.358500, 540.000000, 0.359500, 550.000000, 0.359500,
         560.000000, 0.359500, 570.000000, 0.361000, 580.000000, 0.361500,
         590.000000, 0.361000, 600.000000, 0.359500, 610.000000, 0.358500,
         620.000000, 0.356000, 630.000000, 0.353500, 640.000000, 0.351500,
         650.000000, 0.349500, 660.000000, 0.347000, 670.000000, 0.344500,
         680.000000, 0.342500, 690.000000, 0.340500, 700.000000, 0.338000,
         710.000000, 0.336500, 720.000000, 0.334500, 730.000000, 0.333000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.109000, 390.000000, 0.140500, 400.000000, 0.173000,
         410.000000, 0.189500, 420.000000, 0.194000, 430.000000, 0.196500,
         440.000000, 0.199000, 450.000000, 0.199500, 460.000000, 0.199000,
         470.000000, 0.197500, 480.000000, 0.196500, 490.000000, 0.196500,
         500.000000, 0.196500, 510.000000, 0.197000, 520.000000, 0.197000,
         530.000000, 0.197500, 540.000000, 0.197500, 550.000000, 0.197500,
         560.000000, 0.197500, 570.000000, 0.198500, 580.000000, 0.198500,
         590.000000, 0.198500, 600.000000, 0.197500, 610.000000, 0.196500,
         620.000000, 0.195500, 630.000000, 0.193500, 640.000000, 0.192000,
         650.000000, 0.190000, 660.000000, 0.189000, 670.000000, 0.187500,
         680.000000, 0.186500, 690.000000, 0.185000, 700.000000, 0.183000,
         710.000000, 0.182000, 720.000000, 0.181000, 730.000000, 0.180000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.071000, 390.000000, 0.080500, 400.000000, 0.087500,
         410.000000, 0.090500, 420.000000, 0.091500, 430.000000, 0.092000,
         440.000000, 0.093500, 450.000000, 0.093500, 460.000000, 0.092500,
         470.000000, 0.092000, 480.000000, 0.091500, 490.000000, 0.091500,
         500.000000, 0.091000, 510.000000, 0.091500, 520.000000, 0.091500,
         530.000000, 0.091500, 540.000000, 0.091500, 550.000000, 0.091500,
         560.000000, 0.091500, 570.000000, 0.091500, 580.000000, 0.091500,
         590.000000, 0.091000, 600.000000, 0.090500, 610.000000, 0.090000,
         620.000000, 0.089000, 630.000000, 0.088000, 640.000000, 0.088000,
         650.000000, 0.087000, 660.000000, 0.086500, 670.000000, 0.086000,
         680.000000, 0.085000, 690.000000, 0.085000, 700.000000, 0.084000,
         710.000000, 0.083500, 720.000000, 0.083000, 730.000000, 0.083000},
        false, Allocator()),
    PiecewiseLinearSpectrum::FromInterleaved(
        {380.000000, 0.031500, 390.000000, 0.032500, 400.000000, 0.033500,
         410.000000, 0.034500, 420.000000, 0.034500, 430.000000, 0.034500,
         440.000000, 0.034000, 450.000000, 0.034000, 460.000000, 0.033500,
         470.000000, 0.033500, 480.000000, 0.033000, 490.000000, 0.033000,
         500.000000, 0.033000, 510.000000, 0.033000, 520.000000, 0.033000,
         530.000000, 0.033000, 540.000000, 0.033000, 550.000000, 0.033000,
         560.000000, 0.032500, 570.000000, 0.032500, 580.000000, 0.032500,
         590.000000, 0.032500, 600.000000, 0.032500, 610.000000, 0.032500,
         620.000000, 0.032500, 630.000000, 0.032500, 640.000000, 0.032500,
         650.000000, 0.032500, 660.000000, 0.032500, 670.000000, 0.032500,
         680.000000, 0.032500, 690.000000, 0.032000, 700.000000, 0.032000,
         710.000000, 0.032000, 720.000000, 0.032000, 730.000000, 0.032500},
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
    CHECK(colorSpace != nullptr);
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
    outputRGBFromSensorRGB = colorSpace->RGBFromXYZ * sensor->XYZFromSensorRGB;
}

SampledWavelengths RGBFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleXYZ(u);
}

void RGBFilm::AddSplat(const Point2f &p, SampledSpectrum L,
                       const SampledWavelengths &lambda) {
    CHECK(!L.HasNaNs());
    // Convert sample radiance to _PixelSensor_ RGB
    SampledSpectrum H = L * sensor->ImagingRatio();
    RGB rgb = sensor->ToSensorRGB(H, lambda);

    // Optionally clamp sensor RGB value
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue) {
        H *= maxComponentValue / m;
        rgb *= maxComponentValue / m;
    }

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
                pixel.splatRGB[i].Add(wt * rgb[i]);
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

    ParallelFor2D(pixelBounds, [&](Point2i p) {
        RGB rgb = GetPixelRGB(p, splatScale);

        Point2i pOffset(p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, {rgb[0], rgb[1], rgb[2]});
    });

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
                         FilterHandle filter, const RGBColorSpace *colorSpace,
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
void GBufferFilm::AddSample(const Point2i &pFilm, SampledSpectrum L,
                            const SampledWavelengths &lambda,
                            const VisibleSurface *visibleSurface, Float weight) {
    // First convert to sensor exposure, H, then to camera RGB
    SampledSpectrum H = L * sensor->ImagingRatio();
    RGB rgb = sensor->ToSensorRGB(H, lambda);
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue) {
        H *= maxComponentValue / m;
        rgb *= maxComponentValue / m;
    }

    Pixel &p = pixels[pFilm];
    if (visibleSurface && *visibleSurface) {
        // Update variance estimates.
        for (int c = 0; c < 3; ++c)
            p.varianceEstimator[c].Add(rgb[c]);

        p.pSum += weight * visibleSurface->p;

        p.nSum += weight * visibleSurface->n;
        p.nsSum += weight * visibleSurface->ns;

        p.dzdxSum += weight * visibleSurface->dzdx;
        p.dzdySum += weight * visibleSurface->dzdy;

        SampledSpectrum albedo =
            visibleSurface->albedo * colorSpace->illuminant.Sample(lambda);
        RGB albedoRGB = albedo.ToRGB(lambda, *colorSpace);
        for (int c = 0; c < 3; ++c)
            p.albedoSum[c] += weight * albedoRGB[c];
    }

    for (int c = 0; c < 3; ++c)
        p.rgbSum[c] += rgb[c] * weight;
    p.weightSum += weight;
}

GBufferFilm::GBufferFilm(FilmBaseParameters p, const RGBColorSpace *colorSpace,
                         Float maxComponentValue, bool writeFP16, Allocator alloc)
    : FilmBase(p),
      pixels(pixelBounds, alloc),
      colorSpace(colorSpace),
      maxComponentValue(maxComponentValue),
      writeFP16(writeFP16),
      filterIntegral(filter.Integral()) {
    CHECK(!pixelBounds.IsEmpty());
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
    outputRGBFromSensorRGB = colorSpace->RGBFromXYZ * sensor->XYZFromSensorRGB;
}

SampledWavelengths GBufferFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleXYZ(u);
}

void GBufferFilm::AddSplat(const Point2f &p, SampledSpectrum v,
                           const SampledWavelengths &lambda) {
    // NOTE: same code as RGBFilm::AddSplat()...
    CHECK(!v.HasNaNs());
    // First convert to sensor exposure, H, then to camera RGB
    SampledSpectrum H = v * sensor->ImagingRatio();
    RGB rgb = sensor->ToSensorRGB(H, lambda);
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
                pixel.splatRGB[i].Add(wt * rgb[i]);
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
                 "Px",
                 "Py",
                 "Pz",
                 "dzdx",
                 "dzdy",
                 "Nx",
                 "Ny",
                 "Nz",
                 "Nsx",
                 "Nsy",
                 "Nsz",
                 "Variance.R",
                 "Variance.G",
                 "Variance.B",
                 "RelativeVariance.R",
                 "RelativeVariance.G",
                 "RelativeVariance.B"});

    ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
    ImageChannelDesc pDesc = image.GetChannelDesc({"Px", "Py", "Pz"});
    ImageChannelDesc dzDesc = image.GetChannelDesc({"dzdx", "dzdy"});
    ImageChannelDesc nDesc = image.GetChannelDesc({"Nx", "Ny", "Nz"});
    ImageChannelDesc nsDesc = image.GetChannelDesc({"Nsx", "Nsy", "Nsz"});
    ImageChannelDesc albedoRgbDesc =
        image.GetChannelDesc({"Albedo.R", "Albedo.G", "Albedo.B"});
    ImageChannelDesc varianceDesc =
        image.GetChannelDesc({"Variance.R", "Variance.G", "Variance.B"});
    ImageChannelDesc relVarianceDesc = image.GetChannelDesc(
        {"RelativeVariance.R", "RelativeVariance.G", "RelativeVariance.B"});

    ParallelFor2D(pixelBounds, [&](Point2i p) {
        Pixel &pixel = pixels[p];
        RGB rgb(pixel.rgbSum[0], pixel.rgbSum[1], pixel.rgbSum[2]);
        RGB albedoRgb(pixel.albedoSum[0], pixel.albedoSum[1], pixel.albedoSum[2]);

        // Normalize pixel with weight sum
        Float weightSum = pixel.weightSum;
        Point3f pt = pixel.pSum;
        Float dzdx = pixel.dzdxSum, dzdy = pixel.dzdySum;
        if (weightSum != 0) {
            rgb /= weightSum;
            albedoRgb /= weightSum;
            pt /= weightSum;
            dzdx /= weightSum;
            dzdy /= weightSum;
        }

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            rgb[c] += splatScale * pixel.splatRGB[c] / filterIntegral;

        rgb = outputRGBFromSensorRGB * rgb;

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
        image.SetChannels(
            pOffset, varianceDesc,
            {pixel.varianceEstimator[0].Variance(), pixel.varianceEstimator[1].Variance(),
             pixel.varianceEstimator[2].Variance()});
        image.SetChannels(pOffset, relVarianceDesc,
                          {pixel.varianceEstimator[0].RelativeVariance(),
                           pixel.varianceEstimator[1].RelativeVariance(),
                           pixel.varianceEstimator[2].RelativeVariance()});
    });

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;
    metadata->colorSpace = colorSpace;

    return image;
}

std::string GBufferFilm::ToString() const {
    return StringPrintf("[ GBufferFilm %s colorSpace: %s maxComponentValue: %f "
                        "writeFP16: %s ]",
                        BaseToString(), *colorSpace, maxComponentValue, writeFP16);
}

GBufferFilm *GBufferFilm::Create(const ParameterDictionary &parameters,
                                 Float exposureTime, FilterHandle filter,
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

    return alloc.new_object<GBufferFilm>(filmBaseParameters, colorSpace,
                                         maxComponentValue, writeFP16, alloc);
}

FilmHandle FilmHandle::Create(const std::string &name,
                              const ParameterDictionary &parameters, Float exposureTime,
                              FilterHandle filter, const FileLoc *loc, Allocator alloc) {
    FilmHandle film;
    if (name == "rgb")
        film = RGBFilm::Create(parameters, exposureTime, filter, parameters.ColorSpace(),
                               loc, alloc);
    else if (name == "gbuffer")
        film = GBufferFilm::Create(parameters, exposureTime, filter,
                                   parameters.ColorSpace(), loc, alloc);
    else
        ErrorExit(loc, "%s: film type unknown.", name);

    if (!film)
        ErrorExit(loc, "%s: unable to create film.", name);

    parameters.ReportUnused();
    return film;
}

}  // namespace pbrt
