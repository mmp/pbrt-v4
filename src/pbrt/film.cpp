// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

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
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/transform.h>

namespace pbrt {

// Sensor Method Definitions
Sensor::Sensor(SpectrumHandle r_bar, SpectrumHandle g_bar, SpectrumHandle b_bar,
        const RGBColorSpace* outputColorSpace, const SquareMatrix<3> &XYZFromCameraRGB,
        Float exposureTime, Float fNumber, Float ISO, Float C, const DenselySampledSpectrum& whiteIlluminant,
        Allocator alloc)
    : r_bar(r_bar, alloc),
        g_bar(g_bar, alloc),
        b_bar(b_bar, alloc),
        XYZFromCameraRGB(XYZFromCameraRGB),
        exposureTime(exposureTime),
        fNumber(fNumber),
        ISO(ISO),
        C(C) {
    // Compute white normalization factor for sensor
    RGB white;
    g_integral = 0;
    for (Float l = Lambda_min; l <= Lambda_max; ++l) {
        white.r += r_bar(l) * whiteIlluminant(l);
        white.g += g_bar(l) * whiteIlluminant(l);
        white.b += b_bar(l) * whiteIlluminant(l);
        g_integral += g_bar(l);
    }
    g_integral /= (Lambda_max - Lambda_min + 1);
    Warning("g_integral: %f", g_integral);
    if (XYZFromCameraRGB.IsIdentity()) {
        cameraRGBWhiteNorm = RGB(1, 1, 1);
    } else {
        // Compute RGB of illuminant in sensor's RGB space
        cameraRGBWhiteNorm = RGB(white.g, white.g, white.g) / white;
    }
}

Sensor *Sensor::Create(const std::string &name, const RGBColorSpace *colorSpace,
                       Float exposureTime, Float fNumber, Float ISO, Float C,
                       Float whiteBalanceTemp, const FileLoc *loc, Allocator alloc) {
    if (name == "cie1931") {
        return alloc.new_object<Sensor>(&Spectra::X(), &Spectra::Y(), &Spectra::Z(),
                                        colorSpace, SquareMatrix<3>(),
                                        exposureTime, fNumber, ISO, C, colorSpace->illuminant,
                                        alloc);
    } else {
        /*
        SpectrumHandle r = GetNamedSpectrum(name + "_r");
        SpectrumHandle g = GetNamedSpectrum(name + "_g");
        SpectrumHandle b = GetNamedSpectrum(name + "_b");

        if (!r || !g || !b) {
            ErrorExit(loc, "%s: unknown sensor type", name);
        }

        auto whiteIlluminant = Spectra::D(whiteBalanceTemp);
        SquareMatrix<3> XYZFromCameraRGB = SolveCameraMatrix(r, g, b, whiteIlluminant, colorSpace->illuminant);
        return alloc.new_object<Sensor>(
            r, g, b,
            colorSpace, XYZFromCameraRGB,
            exposureTime, fNumber, ISO, C, colorSpace->illuminant,
            alloc);
            */
    }
}

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

STAT_MEMORY_COUNTER("Memory/Film pixels", filmPixelMemory);

// RGBFilm Method Definitions
RGBFilm::RGBFilm(const Sensor* sensor, const Point2i &resolution, const Bounds2i &pixelBounds,
                 FilterHandle filter, Float diagonal, const std::string &filename,
                 Float scale, const RGBColorSpace *colorSpace, Float maxComponentValue,
                 bool writeFP16, Allocator allocator)
    : FilmBase(resolution, pixelBounds, filter, diagonal, filename),
      sensor(sensor),
      pixels(pixelBounds, allocator),
      scale(scale),
      colorSpace(colorSpace),
      maxComponentValue(maxComponentValue),
      writeFP16(writeFP16) {
    filterIntegral = filter.Integral();
    CHECK(!pixelBounds.IsEmpty());
    CHECK(colorSpace != nullptr);
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
    outputRGBFromCameraRGB = colorSpace->RGBFromXYZ * sensor->XYZFromCameraRGB;
}

SampledWavelengths RGBFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleXYZ(u);
}

void RGBFilm::AddSplat(const Point2f &p, SampledSpectrum v,
                       const SampledWavelengths &lambda) {
    CHECK(!v.HasNaNs());
    // First convert to sensor exposure, H, then to camera RGB
    SampledSpectrum H = v * sensor->ImagingRatio();
    RGB rgb = sensor->ToCameraRGB(H, lambda);
    // Optionally clamp splat sensor RGB value
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue)
        rgb *= maxComponentValue / m;

    // Compute bounds of affected pixels for splat, _splatBounds_
    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - filter.Radius())),
                         Point2i(Floor(pDiscrete + filter.Radius())) + Vector2i(1, 1));
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

    Float varianceSum = 0;
    for (Point2i p : pixelBounds) {
        const Pixel &pixel = pixels[p];
        varianceSum += Float(pixel.varianceEstimator.Variance());
    }
    metadata->estimatedVariance = varianceSum / pixelBounds.Area();

    return image;
}

std::string RGBFilm::ToString() const {
    return StringPrintf("[ RGBFilm %s scale: %f colorSpace: %s maxComponentValue: %f "
                        "writeFP16: %s ]",
                        BaseToString(), scale, *colorSpace, maxComponentValue, writeFP16);
}

RGBFilm *RGBFilm::Create(const ParameterDictionary &parameters, FilterHandle filter,
                         const RGBColorSpace *colorSpace, const FileLoc *loc,
                         Allocator alloc) {
    std::string filename = parameters.GetOneString("filename", "");
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

    Point2i fullResolution(parameters.GetOneInt("xresolution", 1280),
                           parameters.GetOneInt("yresolution", 720));
    if (Options->quickRender) {
        fullResolution.x = std::max(1, fullResolution.x / 4);
        fullResolution.y = std::max(1, fullResolution.y / 4);
    }

    Bounds2i pixelBounds(Point2i(0, 0), fullResolution);
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

    Float scale = parameters.GetOneFloat("scale", 1.);
    Float diagonal = parameters.GetOneFloat("diagonal", 35.);
    Float maxComponentValue = parameters.GetOneFloat("maxcomponentvalue", Infinity);
    bool writeFP16 = parameters.GetOneBool("savefp16", true);

    // Imaging ratio parameters
    // The defaults here represent a "passthrough" setup such that the imaging
    // ratio will be exactly 1. This is a useful default since scenes that
    // weren't authored with a physical camera in mind will render as expected.
    Float exposureTime = parameters.GetOneFloat("exposuretime", 1.);
    Float fNumber = parameters.GetOneFloat("fnumber", 1.);
    Float ISO = parameters.GetOneFloat("iso", 100.);
    // Note: in the talk we mention using 312.5 for historical reasons. The 
    // choice of 100 * Pi here just means that the other parameters make nice
    // "round" numbers like 1 and 100.
    Float C = parameters.GetOneFloat("c", 100.0 * Pi);
    Float whiteBalanceTemp = parameters.GetOneFloat("whitebalance", 6500);

    std::string sensorName = parameters.GetOneString("sensor", "cie1931");
    Sensor *sensor = Sensor::Create(sensorName, colorSpace, exposureTime, 
                                    fNumber, ISO, C, whiteBalanceTemp, loc, alloc);

    return alloc.new_object<RGBFilm>(sensor, fullResolution, pixelBounds, filter, diagonal,
                                     filename, scale, colorSpace, maxComponentValue,
                                     writeFP16, alloc);
}

// GBufferFilm Method Definitions
void GBufferFilm::AddSample(const Point2i &pFilm, SampledSpectrum L,
                            const SampledWavelengths &lambda,
                            const VisibleSurface *visibleSurface, Float weight) {
    RGB rgb = L.ToRGB(lambda, *colorSpace);
    Float m = std::max({rgb.r, rgb.g, rgb.b});
    if (m > maxComponentValue) {
        L *= maxComponentValue / m;
        rgb *= maxComponentValue / m;
    }

    Pixel &p = pixels[pFilm];
    if (visibleSurface && *visibleSurface) {
        // Update variance estimates.
        // TODO: store channels independently?
        p.rgbVarianceEstimator.Add(L.y(lambda));

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

GBufferFilm::GBufferFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
                         FilterHandle filter, Float diagonal, const std::string &filename,
                         Float scale, const RGBColorSpace *colorSpace,
                         Float maxComponentValue, bool writeFP16, Allocator alloc)
    : FilmBase(resolution, pixelBounds, filter, diagonal, filename),
      pixels(pixelBounds, alloc),
      scale(scale),
      colorSpace(colorSpace),
      maxComponentValue(maxComponentValue),
      writeFP16(writeFP16),
      filterIntegral(filter.Integral()) {
    CHECK(!pixelBounds.IsEmpty());
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
}

SampledWavelengths GBufferFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleXYZ(u);
}

void GBufferFilm::AddSplat(const Point2f &p, SampledSpectrum v,
                           const SampledWavelengths &lambda) {
    // NOTE: same code as RGBFilm::AddSplat()...
    CHECK(!v.HasNaNs());
    RGB rgb = v.ToRGB(lambda, *colorSpace);
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
                 "materialId.R",
                 "materialId.G",
                 "materialId.B",
                 "rgbVariance",
                 "rgbRelativeVariance"});

    ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
    ImageChannelDesc pDesc = image.GetChannelDesc({"Px", "Py", "Pz"});
    ImageChannelDesc dzDesc = image.GetChannelDesc({"dzdx", "dzdy"});
    ImageChannelDesc nDesc = image.GetChannelDesc({"Nx", "Ny", "Nz"});
    ImageChannelDesc nsDesc = image.GetChannelDesc({"Nsx", "Nsy", "Nsz"});
    ImageChannelDesc albedoRgbDesc =
        image.GetChannelDesc({"Albedo.R", "Albedo.G", "Albedo.B"});
    ImageChannelDesc varianceDesc =
        image.GetChannelDesc({"rgbVariance", "rgbRelativeVariance"});

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

        rgb *= scale;

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
        image.SetChannels(pOffset, varianceDesc,
                          {pixel.rgbVarianceEstimator.Variance(),
                           pixel.rgbVarianceEstimator.RelativeVariance()});
    });

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;
    metadata->colorSpace = colorSpace;

    Float varianceSum = 0;
    for (Point2i p : pixelBounds) {
        const Pixel &pixel = pixels[p];
        varianceSum += pixel.rgbVarianceEstimator.Variance();
    }
    metadata->estimatedVariance = varianceSum / pixelBounds.Area();

    return image;
}

std::string GBufferFilm::ToString() const {
    return StringPrintf("[ GBufferFilm %s colorSpace: %s maxComponentValue: %f "
                        "writeFP16: %s ]",
                        BaseToString(), *colorSpace, maxComponentValue, writeFP16);
}

GBufferFilm *GBufferFilm::Create(const ParameterDictionary &parameters,
                                 FilterHandle filter, const RGBColorSpace *colorSpace,
                                 const FileLoc *loc, Allocator alloc) {
    std::string filename = parameters.GetOneString("filename", "");
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

    Point2i fullResolution(parameters.GetOneInt("xresolution", 1280),
                           parameters.GetOneInt("yresolution", 720));
    if (Options->quickRender) {
        fullResolution.x = std::max(1, fullResolution.x / 4);
        fullResolution.y = std::max(1, fullResolution.y / 4);
    }

    Bounds2i pixelBounds(Point2i(0, 0), fullResolution);
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

    Float diagonal = parameters.GetOneFloat("diagonal", 35.);
    Float maxComponentValue = parameters.GetOneFloat("maxcomponentvalue", Infinity);
    Float scale = parameters.GetOneFloat("scale", 1.);
    bool writeFP16 = parameters.GetOneBool("savefp16", true);

    return alloc.new_object<GBufferFilm>(fullResolution, pixelBounds, filter, diagonal,
                                         filename, scale, colorSpace, maxComponentValue,
                                         writeFP16, alloc);
}

FilmHandle FilmHandle::Create(const std::string &name,
                              const ParameterDictionary &parameters, const FileLoc *loc,
                              FilterHandle filter, Allocator alloc) {
    FilmHandle film;
    if (name == "rgb")
        film = RGBFilm::Create(parameters, filter, parameters.ColorSpace(), loc, alloc);
    else if (name == "gbuffer")
        film =
            GBufferFilm::Create(parameters, filter, parameters.ColorSpace(), loc, alloc);
    else
        ErrorExit(loc, "%s: film type unknown.", name);

    if (!film)
        ErrorExit(loc, "%s: unable to create film.", name);

    parameters.ReportUnused();
    return film;
}

}  // namespace pbrt
