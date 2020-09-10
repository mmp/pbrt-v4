// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// PhysLight code contributed by Anders Langlands and Luca Fascione
// Copyright (c) 2020, Weta Digital, Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <pbrt/lights.h>

#include <pbrt/cameras.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/shapes.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

namespace pbrt {

STAT_COUNTER("Scene/Lights", numLights);
STAT_COUNTER("Scene/AreaLights", numAreaLights);

// Light Method Definitions
std::string ToString(LightType lf) {
    switch (lf) {
    case LightType::DeltaPosition:
        return "DeltaPosition";
    case LightType::DeltaDirection:
        return "DeltaDirection,";
    case LightType::Area:
        return "Area";
    case LightType::Infinite:
        return "Infinite";
    default:
        LOG_FATAL("Unhandled type");
        return "";
    }
}

LightBase::LightBase(LightType type, const Transform &renderFromLight,
                     const MediumInterface &mediumInterface)
    : type(type), mediumInterface(mediumInterface), renderFromLight(renderFromLight) {
    ++numLights;
}

std::string LightBase::BaseToString() const {
    return StringPrintf("type: %s mediumInterface: %s renderFromLight: %s", type,
                        mediumInterface, renderFromLight);
}

std::string LightBounds::ToString() const {
    return StringPrintf("[ LightBounds b: %s w: %s phi: %f theta_o: %f theta_e: %f "
                        "cosTheta_o: %f cosTheta_e: %f twoSided: %s ]",
                        b, w, phi, theta_o, theta_e, cosTheta_o, cosTheta_e, twoSided);
}

// LightBounds Method Definitions
LightBounds Union(const LightBounds &a, const LightBounds &b) {
    if (a.phi == 0)
        return b;
    if (b.phi == 0)
        return a;
    DirectionCone c =
        Union(DirectionCone(a.w, a.cosTheta_o), DirectionCone(b.w, b.cosTheta_o));
    Float theta_o = SafeACos(c.cosTheta);
    return LightBounds(Union(a.b, b.b), c.w, a.phi + b.phi, theta_o,
                       std::max(a.theta_e, b.theta_e), a.twoSided | b.twoSided);
}

// PointLight Method Definitions
SampledSpectrum PointLight::Phi(const SampledWavelengths &lambda) const {
    return 4 * Pi * scale * I.Sample(lambda);
}

LightBounds PointLight::Bounds() const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    return LightBounds(p, Vector3f(0, 0, 1), 4 * Pi * scale * I.MaxValue(), Pi, Pi / 2,
                       false);
}

LightLeSample PointLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                   SampledWavelengths &lambda, Float time) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Ray ray(p, SampleUniformSphere(u1), time, mediumInterface.outside);
    return LightLeSample(scale * I.Sample(lambda), ray, 1, UniformSpherePDF());
}

void PointLight::PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;
    *pdfDir = UniformSpherePDF();
}

std::string PointLight::ToString() const {
    return StringPrintf("[ PointLight %s I: %s scale: %f ]", BaseToString(), I, scale);
}

PointLight *PointLight::Create(const Transform &renderFromLight, MediumHandle medium,
                               const ParameterDictionary &parameters,
                               const RGBColorSpace *colorSpace, const FileLoc *loc,
                               Allocator alloc) {
    SpectrumHandle I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                                 SpectrumType::General, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float k_e = 4 * Pi;
        sc *= phi_v / k_e;
    }

    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Transform tf = Translate(Vector3f(from.x, from.y, from.z));
    Transform finalRenderFromLight(renderFromLight * tf);

    return alloc.new_object<PointLight>(finalRenderFromLight, medium, I, sc, alloc);
}

// DistantLight Method Definitions
DistantLight::DistantLight(const Transform &renderFromLight, SpectrumHandle Lemit,
                           Float scale, Allocator alloc)
    : LightBase(LightType::DeltaDirection, renderFromLight, MediumInterface()),
      Lemit(Lemit, alloc),
      scale(scale) {}

SampledSpectrum DistantLight::Phi(const SampledWavelengths &lambda) const {
    return scale * Lemit.Sample(lambda) * Pi * sceneRadius * sceneRadius;
}

LightLeSample DistantLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                     SampledWavelengths &lambda, Float time) const {
    // Choose point on disk oriented toward infinite light direction
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    Frame wFrame = Frame::FromZ(w);
    Point2f cd = SampleUniformDiskConcentric(u1);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));

    // Compute _DistantLight_ light ray
    Ray ray(pDisk + sceneRadius * w, -w, time);

    return LightLeSample(scale * Lemit.Sample(lambda), ray,
                         1 / (Pi * sceneRadius * sceneRadius), 1);
}

void DistantLight::PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 1 / (Pi * sceneRadius * sceneRadius);
    *pdfDir = 0;
}

std::string DistantLight::ToString() const {
    return StringPrintf("[ DistantLight %s Lemit: %s scale: %f ]", BaseToString(), Lemit,
                        scale);
}

DistantLight *DistantLight::Create(const Transform &renderFromLight,
                                   const ParameterDictionary &parameters,
                                   const RGBColorSpace *colorSpace, const FileLoc *loc,
                                   Allocator alloc) {
    SpectrumHandle L = parameters.GetOneSpectrum("L", &colorSpace->illuminant,
                                                 SpectrumType::General, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = parameters.GetOnePoint3f("to", Point3f(0, 0, 1));

    Vector3f w = Normalize(from - to);
    Vector3f v1, v2;
    CoordinateSystem(w, &v1, &v2);
    Float m[4][4] = {v1.x, v2.x, w.x, 0, v1.y, v2.y, w.y, 0,
                     v1.z, v2.z, w.z, 0, 0,    0,    0,   1};
    Transform t(m);
    Transform finalRenderFromLight = renderFromLight * t;

    // Scale the light spectrum to be equivalent to 1 nit
    sc /= SpectrumToPhotometric(L);

    // Adjust scale to meet target illuminance value
    // Like for IBLs we measure illuminance as incident on an upward-facing
    // patch.
    Float E_v = parameters.GetOneFloat("illuminance", -1);
    if (E_v > 0) {
        Float k_e = -w.y;
        sc *= E_v / k_e;
    }

    return alloc.new_object<DistantLight>(finalRenderFromLight, L, sc, alloc);
}

STAT_MEMORY_COUNTER("Memory/Light image and distributions", imageBytes);

// ProjectionLight Method Definitions
ProjectionLight::ProjectionLight(const Transform &renderFromLight,
                                 const MediumInterface &mediumInterface, Image im,
                                 const RGBColorSpace *imageColorSpace, Float lscale,
                                 Float fov, Float phi_v, Allocator alloc)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(lscale),
      distrib(alloc) {
    // Initialize ProjectionLight projection matrix
    Float aspect = Float(image.Resolution().x) / Float(image.Resolution().y);
    if (aspect > 1)
        screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
    else
        screenBounds = Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
    hither = 1e-3f;
    Float yon = 1e30f;
    ScreenFromLight = Perspective(fov, hither, yon);
    LightFromScreen = Inverse(ScreenFromLight);

    // Compute cosine of cone surrounding projection directions
    Float opposite = std::tan(Radians(fov) / 2.f);
    // Area of the image on projection plane.
    A = 4 * opposite * opposite * (aspect > 1 ? aspect : 1 / aspect);

    Point3f pCorner(screenBounds.pMax.x, screenBounds.pMax.y, 0);
    Vector3f wCorner = Normalize(Vector3f(LightFromScreen(pCorner)));
    cosTotalWidth = wCorner.z;

    // Compute sampling distribution for _ProjectionLight_
    ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("Image used for ProjectionLight doesn't have R, G, B channels.");
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());
    auto dwdA = [&](const Point2f &p) {
        Vector3f w = Vector3f(LightFromScreen(Point3f(p.x, p.y, 0)));
        w = Normalize(w);
        return Pow<3>(w.z);
    };
    Array2D<Float> d = image.GetSamplingDistribution(dwdA, screenBounds);
    distrib = PiecewiseConstant2D(d, screenBounds);

    // scale radiance to 1 nit
    scale /= SpectrumToPhotometric(&imageColorSpace->illuminant);
    // scale to target photometric power if requested
    if (phi_v > 0) {
        Float sum = 0;
        RGB luminance = imageColorSpace->LuminanceVector();
        for (int v = 0; v < image.Resolution().y; ++v)
            for (int u = 0; u < image.Resolution().x; ++u) {
                Point2f ps = screenBounds.Lerp(
                    {(u + .5f) / image.Resolution().x, (v + .5f) / image.Resolution().y});
                Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));
                w = Normalize(w);
                Float dwdA = Pow<3>(w.z);

                for (int c = 0; c < 3; ++c)
                    sum += image.GetChannel({u, v}, c) * luminance[c] * dwdA;
            }

        scale *= phi_v / (A * sum / (image.Resolution().x * image.Resolution().y));
    }

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

LightLiSample ProjectionLight::SampleLi(LightSampleContext ctx, Point2f u,
                                        SampledWavelengths lambda,
                                        LightSamplingMode mode) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f wi = Normalize(p - ctx.p());
    Vector3f wl = renderFromLight.ApplyInverse(-wi);
    return LightLiSample(this, Projection(wl, lambda) / DistanceSquared(p, ctx.p()), wi,
                         1, Interaction(p, 0 /* time */, &mediumInterface));
}

Float ProjectionLight::PDF_Li(LightSampleContext, Vector3f,
                              LightSamplingMode mode) const {
    return 0.f;
}

LightBounds ProjectionLight::Bounds() const {
#if 0
    // Along the lines of Phi()
    Float sum = 0;
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u) {
            Point2f ps = screenBounds.Lerp({(u + .5f) / image.Resolution().x,
                                            (v + .5f) / image.Resolution().y});
            Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));
            w = Normalize(w);
            Float dwdA = Pow<3>(w.z);
            sum += image.GetChannels({u, v}, rgbChannelDesc).MaxValue() * dwdA;
        }
    Float phi = scale * A * sum / (image.Resolution().x * image.Resolution().y);
#else
    // See comment in SpotLight::Bounds()
    Float sum = 0;
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u)
            sum += std::max({image.GetChannel({u, v}, 0), image.GetChannel({u, v}, 1),
                             image.GetChannel({u, v}, 2)});
    Float phi = scale * sum / (image.Resolution().x * image.Resolution().y);
#endif
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    return LightBounds(p, w, phi, 0.f, std::acos(cosTotalWidth), false);
}

std::string ProjectionLight::ToString() const {
    return StringPrintf("[ ProjectionLight %s scale: %f A: %f cosTotalWidth: %f ]",
                        BaseToString(), scale, A, cosTotalWidth);
}

// Takes wl already in light coordinate system!
SampledSpectrum ProjectionLight::Projection(const Vector3f &wl,
                                            const SampledWavelengths &lambda) const {
    // Discard directions behind projection light
    if (wl.z < hither)
        return SampledSpectrum(0.);

    // Project point onto projection plane and compute RGB
    Point3f ps = ScreenFromLight(Point3f(wl.x, wl.y, wl.z));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds))
        return SampledSpectrum(0.f);
    Point2f st = Point2f(screenBounds.Offset(Point2f(ps.x, ps.y)));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(st, c);

    return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
}

SampledSpectrum ProjectionLight::Phi(const SampledWavelengths &lambda) const {
    SampledSpectrum sum(0.f);
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u) {
            Point2f ps = screenBounds.Lerp(
                {(u + .5f) / image.Resolution().x, (v + .5f) / image.Resolution().y});
            Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));
            w = Normalize(w);
            Float dwdA = Pow<3>(w.z);

            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({u, v}, c);

            SampledSpectrum L = RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);

            sum += L * dwdA;
        }

    return scale * A * sum / (image.Resolution().x * image.Resolution().y);
}

LightLeSample ProjectionLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                        SampledWavelengths &lambda, Float time) const {
    Float pdf;
    Point2f ps = distrib.Sample(u1, &pdf);
    if (pdf == 0)
        return {};

    Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));

    Ray ray = renderFromLight(
        Ray(Point3f(0, 0, 0), Normalize(w), time, mediumInterface.outside));
    Float cosTheta = CosTheta(Normalize(w));
    CHECK_GT(cosTheta, 0);
    Float pdfDir = pdf * screenBounds.Area() / (A * Pow<3>(cosTheta));

    Point2f p = Point2f(screenBounds.Offset(ps));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(p, c);

    SampledSpectrum L = scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);

    return LightLeSample(L, ray, 1, pdfDir);
}

void ProjectionLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;

    Vector3f w = Normalize(renderFromLight.ApplyInverse(ray.d));
    if (w.z < hither) {
        *pdfDir = 0;
        return;
    }
    Point3f ps = ScreenFromLight(Point3f(w));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds)) {
        *pdfDir = 0;
        return;
    }
    *pdfDir = distrib.PDF(Point2f(ps.x, ps.y)) * screenBounds.Area() / (A * Pow<3>(w.z));
}

ProjectionLight *ProjectionLight::Create(const Transform &renderFromLight,
                                         MediumHandle medium,
                                         const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc) {
    Float scale = parameters.GetOneFloat("scale", 1);
    Float power = parameters.GetOneFloat("power", -1);
    Float fov = parameters.GetOneFloat("fov", 90.);

    std::string texname = ResolveFilename(parameters.GetOneString("filename", ""));
    if (texname.empty())
        ErrorExit(loc, "Must provide \"filename\" to \"projection\" light source");

    ImageAndMetadata imageAndMetadata = Image::Read(texname, alloc);
    const RGBColorSpace *colorSpace = imageAndMetadata.metadata.GetColorSpace();

    ImageChannelDesc channelDesc = imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit(loc, "Image provided to \"projection\" light must have R, G, "
                       "and B channels.");
    Image image = imageAndMetadata.image.SelectChannels(channelDesc, alloc);

    Transform flip = Scale(1, -1, 1);
    Transform renderFromLightFlipY = renderFromLight * flip;

    return alloc.new_object<ProjectionLight>(renderFromLightFlipY, medium,
                                             std::move(image), colorSpace, scale, fov,
                                             power, alloc);
}

// GoniometricLight Method Definitions
GoniometricLight::GoniometricLight(const Transform &renderFromLight,
                                   const MediumInterface &mediumInterface,
                                   SpectrumHandle I, Float scale, Image im,
                                   const RGBColorSpace *imageColorSpace, Allocator alloc)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      I(I, alloc),
      scale(scale),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      wrapMode(WrapMode::Repeat, WrapMode::Clamp),
      distrib(alloc) {
    CHECK_EQ(1, image.NChannels());
    // Compute sampling distribution for _GoniometricLight_
    Bounds2f domain(Point2f(0, 0), Point2f(2 * Pi, Pi));
    auto dpdA = [](const Point2f &p) { return std::sin(p.y); };
    Array2D<Float> d = image.GetSamplingDistribution(dpdA, domain);
    distrib = PiecewiseConstant2D(d, domain);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

LightLiSample GoniometricLight::SampleLi(LightSampleContext ctx, Point2f u,
                                         SampledWavelengths lambda,
                                         LightSamplingMode mode) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f wi = Normalize(p - ctx.p());
    SampledSpectrum L =
        Scale(renderFromLight.ApplyInverse(-wi), lambda) / DistanceSquared(p, ctx.p());
    return LightLiSample(this, L, wi, 1, Interaction(p, 0 /* time */, &mediumInterface));
}

Float GoniometricLight::PDF_Li(LightSampleContext, Vector3f,
                               LightSamplingMode mode) const {
    return 0.f;
}

LightBounds GoniometricLight::Bounds() const {
    // Like Phi() method, but compute the weighted max component value of
    // the image map.
    Float weightedMaxImageSum = 0;
    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
        for (int u = 0; u < width; ++u)
            weightedMaxImageSum +=
                sinTheta * image.GetChannels({u, v}, wrapMode).MaxValue();
    }
    Float phi =
        scale * I.MaxValue() * 2 * Pi * Pi * weightedMaxImageSum / (width * height);

    Point3f p = renderFromLight(Point3f(0, 0, 0));
    // Bound it as an isotropic point light.
    return LightBounds(p, Vector3f(0, 0, 1), phi, Pi, Pi / 2, false);
}

SampledSpectrum GoniometricLight::Phi(const SampledWavelengths &lambda) const {
    // integrate over speherical coordinates [0,Pi], [0,2pi]
    Float sumY = 0;
    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
        for (int u = 0; u < width; ++u)
            sumY += sinTheta * image.GetChannels({u, v}, wrapMode).Average();
    }
    return scale * I.Sample(lambda) * 2 * Pi * Pi * sumY / (width * height);
}

LightLeSample GoniometricLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                         SampledWavelengths &lambda, Float time) const {
    Float pdf;
    Point2f uv = distrib.Sample(u1, &pdf);
    Float theta = uv[1], phi = uv[0];
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Vector3f wl = SphericalDirection(sinTheta, cosTheta, phi);
    Float pdfDir = sinTheta == 0 ? 0 : pdf / sinTheta;

    Ray ray = renderFromLight(Ray(Point3f(0, 0, 0), wl, time, mediumInterface.inside));
    return LightLeSample(Scale(wl, lambda), ray, 1, pdfDir);
}

void GoniometricLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0.f;

    Vector3f wl = Normalize(renderFromLight.ApplyInverse(ray.d));
    Float theta = SphericalTheta(wl), phi = SphericalPhi(wl);
    *pdfDir = distrib.PDF(Point2f(phi, theta)) / std::sin(theta);
}

std::string GoniometricLight::ToString() const {
    return StringPrintf("[ GoniometricLight %s I: %s scale: %f ]", BaseToString(), I,
                        scale);
}

GoniometricLight *GoniometricLight::Create(const Transform &renderFromLight,
                                           MediumHandle medium,
                                           const ParameterDictionary &parameters,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc) {
    SpectrumHandle I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                                 SpectrumType::General, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Image image(alloc);
    const RGBColorSpace *imageColorSpace = nullptr;

    std::string texname = ResolveFilename(parameters.GetOneString("filename", ""));
    if (!texname.empty()) {
        ImageAndMetadata imageAndMetadata = Image::Read(texname, alloc);
        ImageChannelDesc rgbDesc = imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
        ImageChannelDesc yDesc = imageAndMetadata.image.GetChannelDesc({"Y"});

        imageColorSpace = imageAndMetadata.metadata.GetColorSpace();

        if (rgbDesc) {
            if (yDesc)
                ErrorExit("%s: has both \"R\", \"G\", and \"B\" or \"Y\" "
                          "channels.",
                          texname);
            image = Image(imageAndMetadata.image.Format(),
                          imageAndMetadata.image.Resolution(), {"Y"},
                          imageAndMetadata.image.Encoding(), alloc);
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x)
                    image.SetChannel(
                        {x, y}, 0,
                        imageAndMetadata.image.GetChannels({x, y}, rgbDesc).Average());
        } else if (yDesc)
            image = imageAndMetadata.image;
        else
            ErrorExit(loc,
                      "%s: has neither \"R\", \"G\", and \"B\" or \"Y\" "
                      "channels.",
                      texname);
    }

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        WrapMode2D wrapMode(WrapMode::Repeat, WrapMode::Clamp);
        // integrate over speherical coordinates [0,Pi], [0,2pi]
        Float sumY = 0;
        int width = image.Resolution().x, height = image.Resolution().y;
        for (int v = 0; v < height; ++v) {
            Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
            for (int u = 0; u < width; ++u)
                sumY += sinTheta * image.GetChannels({u, v}, wrapMode).Average();
        }
        Float k_e = 2 * Pi * Pi * sumY / (width * height);
        sc *= phi_v / k_e;
    }

    const Float swapYZ[4][4] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    Transform t(swapYZ);
    Transform finalRenderFromLight = renderFromLight * t;

    return alloc.new_object<GoniometricLight>(finalRenderFromLight, medium, I, sc,
                                              std::move(image), imageColorSpace, alloc);
}

// DiffuseAreaLight Method Definitions
DiffuseAreaLight::DiffuseAreaLight(const Transform &renderFromLight,
                                   const MediumInterface &mediumInterface,
                                   SpectrumHandle Le, Float scale,
                                   const ShapeHandle shape, Image im,
                                   const RGBColorSpace *imageColorSpace, bool twoSided,
                                   Allocator alloc)
    : LightBase(LightType::Area, renderFromLight, mediumInterface),
      Lemit(Le, alloc),
      scale(scale),
      shape(shape),
      twoSided(twoSided),
      area(shape.Area()),
      imageColorSpace(imageColorSpace),
      image(std::move(im)) {
    ++numAreaLights;

    if (image) {
        ImageChannelDesc desc = image.GetChannelDesc({"R", "G", "B"});
        if (!desc)
            ErrorExit("Image used for DiffuseAreaLight doesn't have R, G, B "
                      "channels.");
        CHECK_EQ(3, desc.size());
        CHECK(desc.IsIdentity());
        CHECK(imageColorSpace != nullptr);
    } else {
        CHECK(Le);
    }

    // Warn if light has transformation with non-uniform scale, though not
    // for Triangles, since this doesn't matter for them.
    // FIXME: is this still true with animated transformations?
    if (renderFromLight.HasScale() && !shape.Is<Triangle>() && !shape.Is<BilinearPatch>())
        Warning("Scaling detected in world to light transformation! "
                "The system has numerous assumptions, implicit and explicit, "
                "that this transform will have no scale factors in it. "
                "Proceed at your own risk; your image may have errors.");
}

SampledSpectrum DiffuseAreaLight::Phi(const SampledWavelengths &lambda) const {
    SampledSpectrum phi(0.f);
    if (image) {
        // Compute average light image emission
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                RGB rgb;
                for (int c = 0; c < 3; ++c)
                    rgb[c] = image.GetChannel({x, y}, c);
                phi += RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
            }
        phi /= image.Resolution().x * image.Resolution().y;

    } else
        phi = Lemit.Sample(lambda);
    return phi * (twoSided ? 2 : 1) * scale * area * Pi;
}

LightBounds DiffuseAreaLight::Bounds() const {
    Float phi = 0;
    if (image) {
        // Assume no distortion in the mapping, FWIW...
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                for (int c = 0; c < 3; ++c)
                    phi += image.GetChannel({x, y}, c);
        phi /= 3 * image.Resolution().x * image.Resolution().y;
    } else
        phi = Lemit.MaxValue();

    phi *= scale * (twoSided ? 2 : 1) * area * Pi;

    // TODO: for animated shapes, we probably need to worry about
    // renderFromLight as in SampleLi().
    DirectionCone nb = shape.NormalBounds();
    return LightBounds(shape.Bounds(), nb.w, phi, SafeACos(nb.cosTheta), Pi / 2,
                       twoSided);
}

LightLeSample DiffuseAreaLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                         SampledWavelengths &lambda, Float time) const {
    // Sample a point on the area light's _Shape_
    Float pdfDir;
    pstd::optional<ShapeSample> ss = shape.Sample(u1);
    if (!ss)
        return {};
    ss->intr.time = time;
    ss->intr.mediumInterface = &mediumInterface;

    // Sample a cosine-weighted outgoing direction _w_ for area light
    Vector3f w;
    if (twoSided) {
        Point2f u = u2;
        // Choose a side to sample and then remap u[0] to [0,1] before
        // applying cosine-weighted hemisphere sampling for the chosen side.
        if (u[0] < .5) {
            u[0] = std::min(u[0] * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u);
        } else {
            u[0] = std::min((u[0] - .5f) * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u);
            w.z *= -1;
        }
        pdfDir = 0.5f * CosineHemispherePDF(std::abs(w.z));
    } else {
        w = SampleCosineHemisphere(u2);
        pdfDir = CosineHemispherePDF(w.z);
    }

    if (pdfDir == 0)
        return {};

    // Return _LightLeSample_ for ray leaving area light
    Frame nFrame = Frame::FromZ(ss->intr.n);
    w = nFrame.FromLocal(w);
    return LightLeSample(L(ss->intr.p(), ss->intr.n, ss->intr.uv, w, lambda),
                         ss->intr.SpawnRay(w), ss->intr, ss->pdf, pdfDir);
}

void DiffuseAreaLight::PDF_Le(const Interaction &intr, Vector3f &w, Float *pdfPos,
                              Float *pdfDir) const {
    CHECK_NE(intr.n, Normal3f(0, 0, 0));
    *pdfPos = shape.PDF(intr);
    *pdfDir = twoSided ? (.5 * CosineHemispherePDF(AbsDot(intr.n, w)))
                       : CosineHemispherePDF(Dot(intr.n, w));
}

std::string DiffuseAreaLight::ToString() const {
    return StringPrintf("[ DiffuseAreaLight %s Lemit: %s scale: %f shape: %s "
                        "twoSided: %s area: %f image: %s ]",
                        BaseToString(), Lemit, scale, shape, twoSided ? "true" : "false",
                        area, image);
}

DiffuseAreaLight *DiffuseAreaLight::Create(const Transform &renderFromLight,
                                           MediumHandle medium,
                                           const ParameterDictionary &parameters,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc,
                                           const ShapeHandle shape) {
    SpectrumHandle L =
        parameters.GetOneSpectrum("L", nullptr, SpectrumType::General, alloc);
    Float scale = parameters.GetOneFloat("scale", 1);
    bool twoSided = parameters.GetOneBool("twosided", false);

    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    Image image;
    const RGBColorSpace *imageColorSpace = nullptr;
    if (!filename.empty()) {
        if (L != nullptr)
            ErrorExit(loc, "Both \"L\" and \"filename\" specified for DiffuseAreaLight.");
        ImageAndMetadata im = Image::Read(filename, alloc);

        ImageChannelDesc channelDesc = im.image.GetChannelDesc({"R", "G", "B"});
        if (!channelDesc)
            ErrorExit(loc,
                      "%s: Image provided to \"diffuse\" area light must have "
                      "R, G, and B channels.",
                      filename);
        image = im.image.SelectChannels(channelDesc, alloc);

        imageColorSpace = im.metadata.GetColorSpace();
    } else if (L == nullptr)
        L = &colorSpace->illuminant;

    // scale so that radiance is equivalent to 1 nit
    scale /= SpectrumToPhotometric(L ? L : &colorSpace->illuminant);

    Float phi_v = parameters.GetOneFloat("power", -1.0f);
    if (phi_v > 0) {
        // k_e is the emissive power of the light as defined by the spectral
        // distribution and texture and is used to normalize the emitted
        // radiance such that the user-defined power will be the actual power
        // emitted by the light.
        Float k_e;
        // Get the appropriate luminance vector from the image colour space
        RGB lum = imageColorSpace->LuminanceVector();
        // we need to know which channels correspond to R, G and B
        // we know that the channelDesc is valid as we would have exited in the
        // block above otherwise
        ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
        if (image) {
            k_e = 0;
            // Assume no distortion in the mapping, FWIW...
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x) {
                    for (int c = 0; c < 3; ++c)
                        k_e += image.GetChannel({x, y}, c) * lum[c];
                }
            k_e /= image.Resolution().x * image.Resolution().y;
        }

        k_e *= (twoSided ? 2 : 1) * shape.Area() * Pi;

        // now multiply up scale to hit the target power
        scale *= phi_v / k_e;
    }

    return alloc.new_object<DiffuseAreaLight>(renderFromLight, medium, L, scale, shape,
                                              std::move(image), imageColorSpace, twoSided,
                                              alloc);
}

// UniformInfiniteLight Method Definitions
UniformInfiniteLight::UniformInfiniteLight(const Transform &renderFromLight,
                                           SpectrumHandle Lemit, Float scale,
                                           Allocator alloc)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      Lemit(Lemit, alloc),
      scale(scale) {}

SampledSpectrum UniformInfiniteLight::Le(const Ray &ray,
                                         const SampledWavelengths &lambda) const {
    return scale * Lemit.Sample(lambda);
}

SampledSpectrum UniformInfiniteLight::Phi(const SampledWavelengths &lambda) const {
    // TODO: is there another Pi or so for the hemisphere?
    // pi r^2 for disk
    // 2pi for cosine-weighted sphere
    return 2 * Pi * Pi * Sqr(sceneRadius) * scale * Lemit.Sample(lambda);
}

LightLiSample UniformInfiniteLight::SampleLi(LightSampleContext ctx, Point2f u,
                                             SampledWavelengths lambda,
                                             LightSamplingMode mode) const {
    Vector3f wi = SampleUniformSphere(u);
    Float pdf = UniformSpherePDF();
    return LightLiSample(
        this, scale * Lemit.Sample(lambda), wi, pdf,
        Interaction(ctx.p() + wi * (2 * sceneRadius), 0 /* time */, &mediumInterface));
}

Float UniformInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                   LightSamplingMode mode) const {
    return UniformSpherePDF();
}

LightLeSample UniformInfiniteLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                             SampledWavelengths &lambda,
                                             Float time) const {
    Vector3f w = SampleUniformSphere(u1);
    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    // Compute probabilities for uniform infinite light
    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
    Float pdfDir = UniformSpherePDF();

    return LightLeSample(scale * Lemit.Sample(lambda), ray, pdfPos, pdfDir);
}

void UniformInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfDir = UniformSpherePDF();
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string UniformInfiniteLight::ToString() const {
    return StringPrintf("[ UniformInfiniteLight %s Lemit: %s ]", BaseToString(), Lemit);
}

// ImageInfiniteLight Method Definitions
ImageInfiniteLight::ImageInfiniteLight(const Transform &renderFromLight, Image im,
                                       const RGBColorSpace *imageColorSpace, Float scale,
                                       const std::string &filename, Allocator alloc)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      filename(filename),
      wrapMode(WrapMode::OctahedralSphere, WrapMode::OctahedralSphere),
      distribution(alloc),
      compensatedDistribution(alloc) {
    // Initialize sampling PDFs for image infinite area light
    ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for ImageInfiniteLight doesn't have R, G, B "
                  "channels.",
                  filename);
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());
    if (image.Resolution().x != image.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an equirect environment map.",
                  filename, image.Resolution().x, image.Resolution().y);
    Array2D<Float> d = image.GetSamplingDistribution();
    Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
    distribution = PiecewiseConstant2D(d, domain, alloc);

    // Initialize compensated PDF for image infinite area light
    Float average = std::accumulate(d.begin(), d.end(), 0.) / d.size();
    for (Float &v : d)
        v = std::max<Float>(v - average, std::min<Float>(.001f * average, v));
    compensatedDistribution = PiecewiseConstant2D(d, domain, alloc);
}

Float ImageInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                 LightSamplingMode mode) const {
    Vector3f wl = renderFromLight.ApplyInverse(w);
    Float pdf = (mode == LightSamplingMode::WithMIS)
                    ? compensatedDistribution.PDF(EqualAreaSphereToSquare(wl))
                    : distribution.PDF(EqualAreaSphereToSquare(wl));
    return pdf / (4 * Pi);
}

SampledSpectrum ImageInfiniteLight::Phi(const SampledWavelengths &lambda) const {
    // We're really computing fluence, then converting to power, for what
    // that's worth..
    SampledSpectrum sumL(0.);

    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({u, v}, c, wrapMode);
            sumL += RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
        }
    }
    // Integrating over the sphere, so 4pi for that.  Then one more for Pi
    // r^2 for the area of the disk receiving illumination...
    return 4 * Pi * Pi * Sqr(sceneRadius) * scale * sumL / (width * height);
}

LightLeSample ImageInfiniteLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                           SampledWavelengths &lambda, Float time) const {
    // Sample infinite light image and compute ray direction _w_
    Float mapPDF;
    Point2f uv = distribution.Sample(u1, &mapPDF);
    Vector3f wl = EqualAreaSquareToSphere(uv);
    Vector3f w = -renderFromLight(wl);

    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    // Compute _ImageInfiniteLight_ ray PDFs
    Float pdfDir = mapPDF / (4 * Pi);
    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));

    return LightLeSample(LookupLe(uv, lambda), ray, pdfPos, pdfDir);
}

void ImageInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    Vector3f wl = -renderFromLight.ApplyInverse(ray.d);
    Float mapPDF = distribution.PDF(EqualAreaSphereToSquare(wl));
    *pdfDir = mapPDF / (4 * Pi);
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string ImageInfiniteLight::ToString() const {
    return StringPrintf("[ ImageInfiniteLight %s filename:%s scale: %f ]", BaseToString(),
                        filename, scale);
}

// PortalImageInfiniteLight Method Definitions
PortalImageInfiniteLight::PortalImageInfiniteLight(
    const Transform &renderFromLight, Image equiAreaImage,
    const RGBColorSpace *imageColorSpace, Float scale, const std::string &filename,
    std::vector<Point3f> p, Allocator alloc)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      image(alloc),
      imageColorSpace(imageColorSpace),
      scale(scale),
      filename(filename),
      distribution(alloc) {
    // Initialize sampling PDFs for infinite area light
    ImageChannelDesc channelDesc = equiAreaImage.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for PortalImageInfiniteLight doesn't have R, "
                  "G, B channels.",
                  filename);
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());

    if (equiAreaImage.Resolution().x != equiAreaImage.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an "
                  "equirect environment map.",
                  filename, equiAreaImage.Resolution().x, equiAreaImage.Resolution().y);

    if (p.size() != 4)
        ErrorExit("Expected 4 vertices for infinite light portal but given %d", p.size());
    for (int i = 0; i < 4; ++i)
        portal[i] = p[i];

    // Compute frame for portal coordinate system
    Vector3f p01 = Normalize(portal[1] - portal[0]);
    Vector3f p12 = Normalize(portal[2] - portal[1]);
    Vector3f p32 = Normalize(portal[2] - portal[3]);
    Vector3f p03 = Normalize(portal[3] - portal[0]);
    // Do opposite edges have the same direction?
    if (std::abs(Dot(p01, p32) - 1) > .001 || std::abs(Dot(p12, p03) - 1) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");
    // Sides perpendicular?
    if (std::abs(Dot(p01, p12)) > .001 || std::abs(Dot(p12, p32)) > .001 ||
        std::abs(Dot(p32, p03)) > .001 || std::abs(Dot(p03, p01)) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");
    portalFrame = Frame::FromXY(p01, p03);

    // Resample environment map into rectified coordinates
    // Resample the latlong map into rectified coordinates
    image = Image(PixelFormat::Float, equiAreaImage.Resolution(), {"R", "G", "B"},
                  equiAreaImage.Encoding(), alloc);
    ParallelFor(0, image.Resolution().y, [&](int y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            // [0,1]^2 image coordinates
            Point2f st((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);

            Vector3f w = RenderFromImage(st);

            w = Normalize(renderFromLight.ApplyInverse(w));

            WrapMode2D equiAreaWrap(WrapMode::OctahedralSphere,
                                    WrapMode::OctahedralSphere);
            Point2f stEqui = EqualAreaSphereToSquare(w);
            for (int c = 0; c < 3; ++c)
                image.SetChannel({x, y}, c,
                                 equiAreaImage.BilerpChannel(stEqui, c, equiAreaWrap));
        }
    });

    // Initialize sampling PDFs for infinite area light
    auto duvdw = [&](const Point2f &p) {
        Float duv_dw;
        (void)RenderFromImage(p, &duv_dw);
        return duv_dw;
    };
    Array2D<Float> d = image.GetSamplingDistribution(duvdw);
    distribution = WindowedPiecewiseConstant2D(d, alloc);
}

SampledSpectrum PortalImageInfiniteLight::Phi(const SampledWavelengths &lambda) const {
    // We're really computing fluence, then converting to power, for what
    // that's worth..
    SampledSpectrum sumL(0.);

    for (int y = 0; y < image.Resolution().y; ++y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({x, y}, c);

            Point2f st((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);
            Float duv_dw;
            (void)RenderFromImage(st, &duv_dw);

            sumL += RGBSpectrum(*imageColorSpace, rgb).Sample(lambda) / duv_dw;
        }
    }

    return scale * Area() * sumL / (image.Resolution().x * image.Resolution().y);
}

SampledSpectrum PortalImageInfiniteLight::Le(const Ray &ray,
                                             const SampledWavelengths &lambda) const {
    // Ignore world to light...
    Vector3f w = Normalize(ray.d);
    Point2f st = ImageFromRender(w);

    if (!Inside(st, ImageBounds(ray.o)))
        return SampledSpectrum(0.f);

    return ImageLookup(st, lambda);
}

SampledSpectrum PortalImageInfiniteLight::ImageLookup(
    const Point2f &st, const SampledWavelengths &lambda) const {
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(st, c);
    return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
}

LightLiSample PortalImageInfiniteLight::SampleLi(LightSampleContext ctx, Point2f u,
                                                 SampledWavelengths lambda,
                                                 LightSamplingMode mode) const {
    Bounds2f b = ImageBounds(ctx.p());

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPDF;
    Point2f uv = distribution.Sample(u, b, &mapPDF);
    if (mapPDF == 0)
        return {};

    // Convert infinite light sample point to direction
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Vector3f wi = RenderFromImage(uv, &duv_dw);
    if (duv_dw == 0)
        return {};

    // Compute PDF for sampled infinite light direction
    Float pdf = mapPDF / duv_dw;
    CHECK(!IsInf(pdf));

    SampledSpectrum L = ImageLookup(uv, lambda);

    return LightLiSample(
        this, L, wi, pdf,
        Interaction(ctx.p() + wi * (2 * sceneRadius), 0 /* time */, &mediumInterface));
}

Float PortalImageInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                       LightSamplingMode mode) const {
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Point2f st = ImageFromRender(w, &duv_dw);
    if (duv_dw == 0)
        return 0;

    Bounds2f b = ImageBounds(ctx.p());
    Float pdf = distribution.PDF(st, b);
    return pdf / duv_dw;
}

LightLeSample PortalImageInfiniteLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                                 SampledWavelengths &lambda,
                                                 Float time) const {
    Float mapPDF;
    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    Point2f uv = distribution.Sample(u1, b, &mapPDF);
    if (mapPDF == 0)
        return {};

    // Convert infinite light sample point to direction
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Vector3f w = -RenderFromImage(uv, &duv_dw);
    if (duv_dw == 0)
        return {};

    // Compute PDF for sampled infinite light direction
    Float pdfDir = mapPDF / duv_dw;

#if 0
    // Just sample within the portal.
    // This works with the light path integrator, but not BDPT :-(
    Point3f p = portal[0] + u2[0] * (portal[1] - portal[0]) +
        u2[1] * (portal[3] - portal[0]);
    // Compute _PortalImageInfiniteLight_ ray PDFs
    Ray ray(p, w, time);

    // Cosine to account for projected area of portal w.r.t. ray direction.
    Normal3f n = Normal3f(portalFrame.z);
    Float pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    SampledSpectrum L = ImageLookup(uv, lambda);

    return LightLeSample(L, ray, pdfPos, pdfDir);
}

void PortalImageInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos,
                                      Float *pdfDir) const {
    // TODO: negate here or???
    Vector3f w = -Normalize(ray.d);
    Float duv_dw;
    Point2f st = ImageFromRender(w, &duv_dw);

    if (duv_dw == 0) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    Float pdf = distribution.PDF(st, b);

#if 0
    Normal3f n = Normal3f(portalFrame.z);
    *pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    *pdfDir = pdf / duv_dw;
}

std::string PortalImageInfiniteLight::ToString() const {
    return StringPrintf("[ PortalImageInfiniteLight %s filename:%s scale: %f portal: %s "
                        " portalFrame: %s ]",
                        BaseToString(), filename, scale, portal, portalFrame);
}

// SpotLight Method Definitions
SpotLight::SpotLight(const Transform &renderFromLight,
                     const MediumInterface &mediumInterface, SpectrumHandle I,
                     Float scale, Float totalWidth, Float falloffStart, Allocator alloc)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      I(I, alloc),
      scale(scale),
      cosFalloffEnd(std::cos(Radians(totalWidth))),
      cosFalloffStart(std::cos(Radians(falloffStart))) {
    CHECK_LE(falloffStart, totalWidth);
}

LightLiSample SpotLight::SampleLi(LightSampleContext ctx, Point2f u,
                                  SampledWavelengths lambda,
                                  LightSamplingMode mode) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f wi = Normalize(p - ctx.p());
    Vector3f wl = Normalize(renderFromLight.ApplyInverse(-wi));
    SampledSpectrum L =
        scale * I.Sample(lambda) * Falloff(wl) / DistanceSquared(p, ctx.p());
    if (!L)
        return {};
    return LightLiSample(this, L, wi, 1, Interaction(p, 0 /* time */, &mediumInterface));
}

Float SpotLight::PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const {
    return 0.f;
}

Float SpotLight::Falloff(const Vector3f &wl) const {
    Float cosTheta = CosTheta(wl);
    if (cosTheta >= cosFalloffStart)
        return 1;
    return SmoothStep(cosTheta, cosFalloffEnd, cosFalloffStart);
}

LightBounds SpotLight::Bounds() const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    // As in Phi()
#if 0
    Float phi = scale * I.MaxValue() * 2 * Pi * ((1 - cosFalloffStart) +
                                          (cosFalloffStart - cosFalloffEnd) / 2);
#else
    // cf. room-subsurf-from-kd.pbrt test: we sorta kinda actually want to
    // compute power as if it was an isotropic light source; the
    // LightBounds geometric terms give zero importance outside the spot
    // light's cone, so inside the cone, it doesn't matter if the overall
    // power is low; it's more accurate to effectively treat it as a point
    // light source.
    Float phi = scale * I.MaxValue() * 4 * Pi;
#endif

    return LightBounds(p, w, phi, 0.f, std::acos(cosFalloffEnd), false);
}

SampledSpectrum SpotLight::Phi(const SampledWavelengths &lambda) const {
    // int_0^start sin theta dtheta = 1 - cosFalloffStart
    // See notes/sample-spotlight.nb for the falloff part:
    // int_start^end smoothstep(cost, end, start) sin theta dtheta =
    //  (cosStart - cosEnd) / 2
    return scale * I.Sample(lambda) * 2 * Pi *
           ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
}

LightLeSample SpotLight::SampleLe(const Point2f &u1, const Point2f &u2,
                                  SampledWavelengths &lambda, Float time) const {
    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};
    Float sectionPDF;
    Vector3f wl;
    int section = SampleDiscrete(p, u2[0], &sectionPDF);
    Float pdfDir;
    if (section == 0) {
        // Sample center cone
        wl = SampleUniformCone(u1, cosFalloffStart);
        pdfDir = sectionPDF * UniformConePDF(cosFalloffStart);
    } else {
        DCHECK_EQ(1, section);

        Float cosTheta = SampleSmoothStep(u1[0], cosFalloffEnd, cosFalloffStart);
        CHECK(cosTheta >= cosFalloffEnd && cosTheta <= cosFalloffStart);
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Float phi = u1[1] * 2 * Pi;
        wl = SphericalDirection(sinTheta, cosTheta, phi);
        pdfDir = sectionPDF * SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) /
                 (2 * Pi);
    }

    Ray ray = renderFromLight(Ray(Point3f(0, 0, 0), wl, time, mediumInterface.outside));
    return LightLeSample(scale * I.Sample(lambda) * Falloff(wl), ray, 1, pdfDir);
}

void SpotLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;

    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};

    Float cosTheta = CosTheta(renderFromLight.ApplyInverse(ray.d));
    if (cosTheta >= cosFalloffStart)
        *pdfDir = UniformConePDF(cosFalloffStart) * p[0] / (p[0] + p[1]);
    else
        *pdfDir = SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) / (2 * Pi) *
                  (p[1] / (p[0] + p[1]));
}

std::string SpotLight::ToString() const {
    return StringPrintf("[ SpotLight %s I: %s cosFalloffStart: %f cosFalloffEnd: %f ]",
                        BaseToString(), I, cosFalloffStart, cosFalloffEnd);
}

SpotLight *SpotLight::Create(const Transform &renderFromLight, MediumHandle medium,
                             const ParameterDictionary &parameters,
                             const RGBColorSpace *colorSpace, const FileLoc *loc,
                             Allocator alloc) {
    SpectrumHandle I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                                 SpectrumType::General, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Float coneangle = parameters.GetOneFloat("coneangle", 30.);
    Float conedelta = parameters.GetOneFloat("conedeltaangle", 5.);
    // Compute spotlight world to light transformation
    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = parameters.GetOnePoint3f("to", Point3f(0, 0, 1));

    Transform dirToZ = (Transform)Frame::FromZ(Normalize(to - from));
    Transform t = Translate(Vector3f(from.x, from.y, from.z)) * Inverse(dirToZ);
    Transform finalRenderFromLight = renderFromLight * t;

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float cosFalloffEnd = std::cos(Radians(coneangle));
        Float cosFalloffStart = std::cos(Radians(coneangle - conedelta));
        Float k_e =
            2 * Pi * ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
        sc *= phi_v / k_e;
    }

    return alloc.new_object<SpotLight>(finalRenderFromLight, medium, I, sc, coneangle,
                                       coneangle - conedelta, alloc);
}

SampledSpectrum LightHandle::Phi(const SampledWavelengths &lambda) const {
    auto phi = [&](auto ptr) { return ptr->Phi(lambda); };
    return DispatchCPU(phi);
}

void LightHandle::Preprocess(const Bounds3f &sceneBounds) {
    auto preprocess = [&](auto ptr) { return ptr->Preprocess(sceneBounds); };
    return DispatchCPU(preprocess);
}

LightLeSample LightHandle::SampleLe(const Point2f &u1, const Point2f &u2,
                                    SampledWavelengths &lambda, Float time) const {
    auto sample = [&](auto ptr) { return ptr->SampleLe(u1, u2, lambda, time); };
    return Dispatch(sample);
}

void LightHandle::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_Le(ray, pdfPos, pdfDir); };
    return Dispatch(pdf);
}

LightBounds LightHandle::Bounds() const {
    auto bounds = [](auto ptr) { return ptr->Bounds(); };
    return DispatchCPU(bounds);
}

std::string LightHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto str = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(str);
}

void LightHandle::PDF_Le(const Interaction &intr, Vector3f &w, Float *pdfPos,
                         Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_Le(intr, w, pdfPos, pdfDir); };
    return Dispatch(pdf);
}

LightHandle LightHandle::Create(const std::string &name,
                                const ParameterDictionary &parameters,
                                const Transform &renderFromLight,
                                const CameraTransform &cameraTransform,
                                MediumHandle outsideMedium, const FileLoc *loc,
                                Allocator alloc) {
    LightHandle light = nullptr;
    if (name == "point")
        light = PointLight::Create(renderFromLight, outsideMedium, parameters,
                                   parameters.ColorSpace(), loc, alloc);
    else if (name == "spot")
        light = SpotLight::Create(renderFromLight, outsideMedium, parameters,
                                  parameters.ColorSpace(), loc, alloc);
    else if (name == "goniometric")
        light = GoniometricLight::Create(renderFromLight, outsideMedium, parameters,
                                         parameters.ColorSpace(), loc, alloc);
    else if (name == "projection")
        light = ProjectionLight::Create(renderFromLight, outsideMedium, parameters, loc,
                                        alloc);
    else if (name == "distant")
        light = DistantLight::Create(renderFromLight, parameters, parameters.ColorSpace(),
                                     loc, alloc);
    else if (name == "infinite") {
        const RGBColorSpace *colorSpace = parameters.ColorSpace();
        std::vector<SpectrumHandle> L =
            parameters.GetSpectrumArray("L", SpectrumType::General, alloc);
        Float scale = parameters.GetOneFloat("scale", 1);
        std::vector<Point3f> portal = parameters.GetPoint3fArray("portal");
        std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
        Float E_v = parameters.GetOneFloat("illuminance", -1);

        if (L.empty() && filename.empty()) {
            // Scale the light spectrum to be equivalent to 1 nit
            scale /= SpectrumToPhotometric(&colorSpace->illuminant);
            if (E_v > 0) {
                // If the scene specifies desired illuminance, first calculate
                // the illuminance from a uniform hemispherical emission
                // of L_v then use this to scale the emission spectrum.
                Float k_e = Pi;
                scale *= E_v / k_e;
            }

            // Default: color space's std illuminant
            light = alloc.new_object<UniformInfiniteLight>(
                renderFromLight, &colorSpace->illuminant, scale, alloc);
        } else if (!L.empty()) {
            if (!filename.empty())
                ErrorExit(loc, "Can't specify both emission \"L\" and "
                               "\"filename\" with InfiniteAreaLight");

            if (!portal.empty())
                ErrorExit(loc, "Portals are not supported for InfiniteAreaLights "
                               "without \"filename\".");

            // Scale the light spectrum to be equivalent to 1 nit
            scale /= SpectrumToPhotometric(L[0]);

            if (E_v > 0) {
                // If the scene specifies desired illuminance, first calculate
                // the illuminance from a uniform hemispherical emission
                // of L_v then use this to scale the emission spectrum.
                Float k_e = Pi;
                scale *= E_v / k_e;
            }

            light = alloc.new_object<UniformInfiniteLight>(renderFromLight, L[0], scale,
                                                           alloc);
        } else {
            ImageAndMetadata imageAndMetadata = Image::Read(filename, alloc);
            const RGBColorSpace *colorSpace = imageAndMetadata.metadata.GetColorSpace();

            ImageChannelDesc channelDesc =
                imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
            if (!channelDesc)
                ErrorExit(loc,
                          "%s: image provided to \"infinite\" light must "
                          "have R, G, and B channels.",
                          filename);

            // Scale the light spectrum to be equivalent to 1 nit
            scale /= SpectrumToPhotometric(&colorSpace->illuminant);

            if (E_v > 0) {
                // Upper hemisphere illuminance calculation for converting map to physical
                // units
                float illuminance = 0;
                const Image &image = imageAndMetadata.image;
                int ye = image.Resolution().y / 2;
                int ys = 0;
                int xs = 0;
                int xe = image.Resolution().x;
                RGB lum = imageAndMetadata.metadata.GetColorSpace()->LuminanceVector();
                for (int y = ys; y < ye; ++y) {
                    float v = (float(y) + 0.5f) / float(image.Resolution().y);
                    float theta = (v - 0.5f) * Pi;
                    float cosTheta = std::cos(theta);
                    float sinTheta = std::sin(theta);
                    for (int x = xs; x < xe; ++x) {
                        ImageChannelValues values = image.GetChannels({x, y});
                        for (int c = 0; c < 3; ++c) {
                            illuminance += values[c] * lum[c] * std::abs(cosTheta) *
                                           std::abs(sinTheta);
                        }
                    }
                }
                illuminance /= float(ye - ys) * float(xe - xs);
                illuminance *= Pi * Pi;

                // scaling factor is just the ratio of the target
                // illuminance and the illuminance of the map multiplied by
                // the illuminant spectrum
                Float k_e = illuminance;
                scale *= E_v / k_e;
            }

            Image image = imageAndMetadata.image.SelectChannels(channelDesc, alloc);

            if (!portal.empty()) {
                for (Point3f &p : portal)
                    p = cameraTransform.RenderFromWorld(p);

                light = alloc.new_object<PortalImageInfiniteLight>(
                    renderFromLight, std::move(image), colorSpace, scale, filename,
                    portal, alloc);
            } else
                light = alloc.new_object<ImageInfiniteLight>(renderFromLight,
                                                             std::move(image), colorSpace,
                                                             scale, filename, alloc);
        }
    } else
        ErrorExit(loc, "%s: light type unknown.", name);

    if (!light)
        ErrorExit(loc, "%s: unable to create light.", name);

    parameters.ReportUnused();
    return light;
}

LightHandle LightHandle::CreateArea(const std::string &name,
                                    const ParameterDictionary &parameters,
                                    const Transform &renderFromLight,
                                    const MediumInterface &mediumInterface,
                                    const ShapeHandle shape, const FileLoc *loc,
                                    Allocator alloc) {
    LightHandle area = nullptr;
    if (name == "diffuse")
        area =
            DiffuseAreaLight::Create(renderFromLight, mediumInterface.outside, parameters,
                                     parameters.ColorSpace(), loc, alloc, shape);
    else
        ErrorExit(loc, "%s: area light type unknown.", name);

    if (!area)
        ErrorExit(loc, "%s: unable to create area light.", name);

    parameters.ReportUnused();
    return area;
}

}  // namespace pbrt
