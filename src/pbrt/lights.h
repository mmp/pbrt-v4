// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_LIGHTS_H
#define PBRT_LIGHTS_H

// PhysLight code contributed by Anders Langlands and Luca Fascione
// Copyright (c) 2020, Weta Digital, Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/medium.h>
#include <pbrt/interaction.h>
#include <pbrt/shapes.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <memory>

namespace pbrt {

std::string ToString(LightType type);

// Light Inline Functions
PBRT_CPU_GPU inline bool IsDeltaLight(LightType type) {
    return (type == LightType::DeltaPosition || type == LightType::DeltaDirection);
}

// LightLiSample Definition
struct LightLiSample {
    // LightLiSample Public Methods
    LightLiSample() = default;
    PBRT_CPU_GPU
    LightLiSample(const SampledSpectrum &L, Vector3f wi, Float pdf,
                  const Interaction &pLight)
        : L(L), wi(wi), pdf(pdf), pLight(pLight) {}

    SampledSpectrum L;
    Vector3f wi;
    Float pdf;
    Interaction pLight;
};

// LightLeSample Definition
struct LightLeSample {
    // LightLeSample Public Methods
    LightLeSample() = default;
    PBRT_CPU_GPU
    LightLeSample(const SampledSpectrum &L, const Ray &ray, Float pdfPos, Float pdfDir)
        : L(L), ray(ray), pdfPos(pdfPos), pdfDir(pdfDir) {}
    PBRT_CPU_GPU
    LightLeSample(const SampledSpectrum &L, const Ray &ray, const Interaction &intr,
                  Float pdfPos, Float pdfDir)
        : L(L), ray(ray), intr(intr), pdfPos(pdfPos), pdfDir(pdfDir) {
        CHECK(this->intr->n != Normal3f(0, 0, 0));
    }

    PBRT_CPU_GPU
    Float AbsCosTheta(const Vector3f &w) const { return intr ? AbsDot(w, intr->n) : 1; }

    // LightLeSample Public Members
    SampledSpectrum L;
    Ray ray;
    pstd::optional<Interaction> intr;
    Float pdfPos = 0, pdfDir = 0;
};

// LightSampleContext Definition
class LightSampleContext {
  public:
    // LightSampleContext Public Methods
    LightSampleContext() = default;
    PBRT_CPU_GPU
    LightSampleContext(const SurfaceInteraction &si)
        : pi(si.pi), n(si.n), ns(si.shading.n) {}
    PBRT_CPU_GPU
    LightSampleContext(const Interaction &intr) : pi(intr.pi) {}
    PBRT_CPU_GPU
    LightSampleContext(Point3fi pi, Normal3f n, Normal3f ns) : pi(pi), n(n), ns(ns) {}

    PBRT_CPU_GPU
    Point3f p() const { return Point3f(pi); }

    // LightSampleContext Public Members
    Point3fi pi;
    Normal3f n, ns;
};

// LightBounds Definition
struct LightBounds {
    // LightBounds Public Methods
    PBRT_CPU_GPU
    operator bool() const { return !b.IsDegenerate(); }

    LightBounds() = default;
    LightBounds(const Bounds3f &b, const Vector3f &w, Float phi, Float theta_o,
                Float theta_e, bool twoSided)
        : b(b),
          w(Normalize(w)),
          phi(phi),
          theta_o(theta_o),
          theta_e(theta_e),
          cosTheta_o(std::cos(theta_o)),
          cosTheta_e(std::cos(theta_e)),
          twoSided(twoSided) {}
    LightBounds(const Point3f &p, const Vector3f &w, Float phi, Float theta_o,
                Float theta_e, bool twoSided)
        : b(p, p),
          w(Normalize(w)),
          phi(phi),
          theta_o(theta_o),
          theta_e(theta_e),
          cosTheta_o(std::cos(theta_o)),
          cosTheta_e(std::cos(theta_e)),
          twoSided(twoSided) {}

    PBRT_CPU_GPU
    Float Importance(Point3f p, Normal3f n) const {
        // Compute clamped squared distance to _intr_
        Point3f pc = Centroid();
        Float d2 = DistanceSquared(p, pc);
        // Don't let d2 get too small if p is inside the bounds.
        d2 = std::max(d2, Length(b.Diagonal()) / 2);

        Vector3f wi = Normalize(p - pc);

        Float cosTheta = Dot(w, wi);
        if (twoSided)
            cosTheta = std::abs(cosTheta);
#if 0
    else if (cosTheta < 0 && cosTheta_o == 1) {
        // Catch the case where the point is outside the bounds and definitely
        // not in the emitted cone even though the conservative theta_u test
        // make suggest it could be.
        // Doesn't seem to make much difference in practice.
        if ((p.x < b.pMin.x || p.x > b.pMax.x) &&
            (p.y < b.pMin.y || p.y > b.pMax.y) &&
            (p.z < b.pMin.z || p.z > b.pMax.z))
            return 0;
    }
#endif

        // FIXME? unstable when cosTheta \approx 1
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);

        // Define sine and cosine clamped subtraction lambdas
        // cos(max(0, a-b))
        auto cosSubClamped = [](Float sinThetaA, Float cosThetaA, Float sinThetaB,
                                Float cosThetaB) -> Float {
            if (cosThetaA > cosThetaB)
                // Handle the max(0, ...)
                return 1;
            return cosThetaA * cosThetaB + sinThetaA * sinThetaB;
        };
        // sin(max(0, a-b))
        auto sinSubClamped = [](Float sinThetaA, Float cosThetaA, Float sinThetaB,
                                Float cosThetaB) -> Float {
            if (cosThetaA > cosThetaB)
                // Handle the max(0, ...)
                return 0;
            return sinThetaA * cosThetaB - cosThetaA * sinThetaB;
        };

        // Compute $\cos \theta_\roman{u}$ for _intr_
        Float cosTheta_u = BoundSubtendedDirections(b, p).cosTheta;
        Float sinTheta_u = SafeSqrt(1 - cosTheta_u * cosTheta_u);

        // Compute $\cos \theta_\roman{p}$ for _intr_ and test against $\cos
        // \theta_\roman{e}$
        // cos(theta_p). Compute in two steps
        Float cosTheta_x = cosSubClamped(
            sinTheta, cosTheta, SafeSqrt(1 - cosTheta_o * cosTheta_o), cosTheta_o);
        Float sinTheta_x = sinSubClamped(
            sinTheta, cosTheta, SafeSqrt(1 - cosTheta_o * cosTheta_o), cosTheta_o);
        Float cosTheta_p = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_u, cosTheta_u);
        if (cosTheta_p <= cosTheta_e)
            return 0;

        Float imp = phi * cosTheta_p / d2;
        DCHECK_GE(imp, -1e-3);

        // Account for $\cos \theta_\roman{i}$ in importance at surfaces
        if (n != Normal3f(0, 0, 0)) {
            // cos(thetap_i) = cos(max(0, theta_i - theta_u))
            // cos (a-b) = cos a cos b + sin a sin b
            Float cosTheta_i = AbsDot(wi, n);
            Float sinTheta_i = SafeSqrt(1 - cosTheta_i * cosTheta_i);
            Float cosThetap_i =
                cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_u, cosTheta_u);
            imp *= cosThetap_i;
        }

        return std::max<Float>(imp, 0);
    }

    PBRT_CPU_GPU
    Point3f Centroid() const { return (b.pMin + b.pMax) / 2; }

    std::string ToString() const;

    // LightBounds Public Members
    Bounds3f b;  // TODO: rename to |bounds|?
    Vector3f w;
    Float phi = 0;
    Float theta_o = 0, theta_e = 0;
    Float cosTheta_o = 1, cosTheta_e = 1;
    bool twoSided = false;
};

LightBounds Union(const LightBounds &a, const LightBounds &b);

// LightBase Definition
class LightBase {
  public:
    // LightBase Public Methods
    LightBase(LightType type, const Transform &renderFromLight,
              const MediumInterface &mediumInterface);

    PBRT_CPU_GPU
    LightType Type() const { return type; }

    PBRT_CPU_GPU
    SampledSpectrum L(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const {
        return SampledSpectrum(0.f);
    }
    PBRT_CPU_GPU
    SampledSpectrum Le(const Ray &, const SampledWavelengths &) const {
        return SampledSpectrum(0.f);
    }

  protected:
    std::string BaseToString() const;
    // LightBase Protected Members
    LightType type;
    Transform renderFromLight;
    MediumInterface mediumInterface;
};

// PointLight Definition
class PointLight : public LightBase {
  public:
    // PointLight Public Methods
    PointLight(Transform renderFromLight, MediumInterface mediumInterface,
               SpectrumHandle I, Float scale, Allocator alloc)
        : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
          I(I, alloc),
          scale(scale) {}

    static PointLight *Create(const Transform &renderFromLight, MediumHandle medium,
                              const ParameterDictionary &parameters,
                              const RGBColorSpace *colorSpace, const FileLoc *loc,
                              Allocator alloc);
    SampledSpectrum Phi(const SampledWavelengths &lambda) const;
    void Preprocess(const Bounds3f &sceneBounds) {}

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    LightBounds Bounds() const;

    std::string ToString() const;

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const {
        Point3f p = renderFromLight(Point3f(0, 0, 0));
        Vector3f wi = Normalize(p - ctx.p());
        SampledSpectrum Li = scale * I.Sample(lambda) / DistanceSquared(p, ctx.p());
        return LightLiSample(Li, wi, 1, Interaction(p, &mediumInterface));
    }

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const { return 0; }

  private:
    // PointLight Private Members
    DenselySampledSpectrum I;
    Float scale;
};

// DistantLight Definition
class DistantLight : public LightBase {
  public:
    // DistantLight Public Methods
    DistantLight(const Transform &renderFromLight, SpectrumHandle Lemit, Float scale,
                 Allocator alloc)
        : LightBase(LightType::DeltaDirection, renderFromLight, MediumInterface()),
          Lemit(Lemit, alloc),
          scale(scale) {}

    static DistantLight *Create(const Transform &renderFromLight,
                                const ParameterDictionary &parameters,
                                const RGBColorSpace *colorSpace, const FileLoc *loc,
                                Allocator alloc);

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const { return 0; }

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    LightBounds Bounds() const { return {}; }

    std::string ToString() const;

    void Preprocess(const Bounds3f &sceneBounds) {
        sceneBounds.BoundingSphere(&sceneCenter, &sceneRadius);
    }

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const {
        Vector3f wi = Normalize(renderFromLight(Vector3f(0, 0, 1)));
        Point3f pOutside = ctx.p() + wi * (2 * sceneRadius);
        return LightLiSample(scale * Lemit.Sample(lambda), wi, 1,
                             Interaction(pOutside, &mediumInterface));
    }

  private:
    // DistantLight Private Members
    DenselySampledSpectrum Lemit;
    Float scale;
    Point3f sceneCenter;
    Float sceneRadius;
};

// ProjectionLight Definition
class ProjectionLight : public LightBase {
  public:
    // ProjectionLight Public Methods
    ProjectionLight(Transform renderFromLight, MediumInterface medium, Image image,
                    const RGBColorSpace *colorSpace, Float scale, Float fov, Float power,
                    Allocator alloc);

    static ProjectionLight *Create(const Transform &renderFromLight, MediumHandle medium,
                                   const ParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {}

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const;
    PBRT_CPU_GPU
    SampledSpectrum I(Vector3f w, const SampledWavelengths &lambda) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    LightBounds Bounds() const;

    std::string ToString() const;

  private:
    // ProjectionLight Private Members
    Image image;
    const RGBColorSpace *imageColorSpace;
    Float scale;
    Bounds2f screenBounds;
    Float hither = 1e-3f;
    Transform screenFromLight, lightFromScreen;
    Float A;
    PiecewiseConstant2D distrib;
};

// GoniometricLight Definition
class GoniometricLight : public LightBase {
  public:
    // GoniometricLight Public Methods
    GoniometricLight(const Transform &renderFromLight,
                     const MediumInterface &mediumInterface, SpectrumHandle I,
                     Float scale, Image image, Allocator alloc);

    static GoniometricLight *Create(const Transform &renderFromLight, MediumHandle medium,
                                    const ParameterDictionary &parameters,
                                    const RGBColorSpace *colorSpace, const FileLoc *loc,
                                    Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {}

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    LightBounds Bounds() const;

    std::string ToString() const;

    PBRT_CPU_GPU
    SampledSpectrum I(Vector3f wl, const SampledWavelengths &lambda) const {
        Point2f uv = EqualAreaSphereToSquare(wl);
        return scale * Iemit.Sample(lambda) * image.LookupNearestChannel(uv, 0);
    }

  private:
    // GoniometricLight Private Members
    DenselySampledSpectrum Iemit;
    Float scale;
    Image image;
    PiecewiseConstant2D distrib;
};

// DiffuseAreaLight Definition
class DiffuseAreaLight : public LightBase {
  public:
    // DiffuseAreaLight Public Methods
    DiffuseAreaLight(const Transform &renderFromLight,
                     const MediumInterface &mediumInterface, SpectrumHandle Le,
                     Float scale, const ShapeHandle shape, Image image,
                     const RGBColorSpace *imageColorSpace, bool twoSided,
                     Allocator alloc);

    static DiffuseAreaLight *Create(const Transform &renderFromLight, MediumHandle medium,
                                    const ParameterDictionary &parameters,
                                    const RGBColorSpace *colorSpace, const FileLoc *loc,
                                    Allocator alloc, const ShapeHandle shape);

    void Preprocess(const Bounds3f &sceneBounds) {}

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const;

    LightBounds Bounds() const;

    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for area lights");
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    SampledSpectrum L(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const {
        if (!twoSided && Dot(n, w) < 0)
            return SampledSpectrum(0.f);
        if (image) {
            // Return _DiffuseAreaLight_ emission using image
            RGB rgb;
            Point2f st = uv;
            st[1] = 1 - st[1];
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.BilerpChannel(st, c);
            return scale *
                   RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb)).Sample(lambda);

        } else
            return scale * Lemit.Sample(lambda);
    }

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext ctx, Vector3f wi, LightSamplingMode) const;

  private:
    // DiffuseAreaLight Private Members
    ShapeHandle shape;
    Float area;
    bool twoSided;
    DenselySampledSpectrum Lemit;
    Float scale;
    Image image;
    const RGBColorSpace *imageColorSpace;
};

// UniformInfiniteLight Definition
class UniformInfiniteLight : public LightBase {
  public:
    // UniformInfiniteLight Public Methods
    UniformInfiniteLight(const Transform &renderFromLight, SpectrumHandle Lemit,
                         Float scale, Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {
        sceneBounds.BoundingSphere(&sceneCenter, &sceneRadius);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const;
    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const;
    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    LightBounds Bounds() const { return {}; }

    std::string ToString() const;

  private:
    // UniformInfiniteLight Private Members
    DenselySampledSpectrum Lemit;
    Float scale;
    Point3f sceneCenter;
    Float sceneRadius;
};

// ImageInfiniteLight Definition
class ImageInfiniteLight : public LightBase {
  public:
    // ImageInfiniteLight Public Methods
    ImageInfiniteLight(Transform renderFromLight, Image image,
                       const RGBColorSpace *imageColorSpace, Float scale,
                       std::string filename, Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {
        sceneBounds.BoundingSphere(&sceneCenter, &sceneRadius);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const {
        Vector3f wl = Normalize(renderFromLight.ApplyInverse(ray.d));
        Point2f uv = EqualAreaSphereToSquare(wl);
        return Le(uv, lambda);
    }

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const {
        // Find $(u,v)$ sample coordinates in infinite light texture
        Float mapPDF;
        Point2f uv = (mode == LightSamplingMode::WithMIS)
                         ? compensatedDistribution.Sample(u, &mapPDF)
                         : distribution.Sample(u, &mapPDF);
        if (mapPDF == 0)
            return {};

        // Convert infinite light sample point to direction
        Vector3f wl = EqualAreaSquareToSphere(uv);
        Vector3f wi = renderFromLight(wl);

        // Compute PDF for sampled infinite light direction
        Float pdf = mapPDF / (4 * Pi);

        // Return radiance value for infinite light direction
        return LightLiSample(
            Le(uv, lambda), wi, pdf,
            Interaction(ctx.p() + wi * (2 * sceneRadius), &mediumInterface));
    }

    LightBounds Bounds() const { return {}; }

  private:
    // ImageInfiniteLight Private Methods
    PBRT_CPU_GPU
    SampledSpectrum Le(Point2f uv, const SampledWavelengths &lambda) const {
        RGB rgb;
        for (int c = 0; c < 3; ++c)
            rgb[c] = image.LookupNearestChannel(uv, c, WrapMode::OctahedralSphere);
        return scale *
               RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb)).Sample(lambda);
    }

    // ImageInfiniteLight Private Members
    Image image;
    const RGBColorSpace *imageColorSpace;
    Float scale;
    Point3f sceneCenter;
    Float sceneRadius;
    PiecewiseConstant2D distribution;
    PiecewiseConstant2D compensatedDistribution;
};

// PortalImageInfiniteLight Definition
class PortalImageInfiniteLight : public LightBase {
  public:
    // PortalImageInfiniteLight Public Methods
    PortalImageInfiniteLight(const Transform &renderFromLight, Image image,
                             const RGBColorSpace *imageColorSpace, Float scale,
                             const std::string &filename, std::vector<Point3f> portal,
                             Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {
        sceneBounds.BoundingSphere(&sceneCenter, &sceneRadius);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    LightBounds Bounds() const { return {}; }

    std::string ToString() const;

  private:
    // PortalImageInfiniteLight Private Methods
    PBRT_CPU_GPU
    SampledSpectrum ImageLookup(Point2f uv, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<Point2f> ImageFromRender(Vector3f wRender,
                                            Float *duv_dw = nullptr) const {
        Vector3f w = portalFrame.ToLocal(wRender);
        if (w.z <= 0)
            return {};
        // Compute Jacobian determinant of mapping $\roman{d}(u,v)/\roman{d}\omega$ if
        // needed
        if (duv_dw)
            *duv_dw = Sqr(Pi) * (1 - Sqr(w.x)) * (1 - Sqr(w.y)) / w.z;

        Float alpha = std::atan2(w.x, w.z), beta = std::atan2(w.y, w.z);
        DCHECK(!IsNaN(alpha + beta));
        return Point2f(Clamp((alpha + Pi / 2) / Pi, 0, 1),
                       Clamp((beta + Pi / 2) / Pi, 0, 1));
    }

    PBRT_CPU_GPU
    Vector3f RenderFromImage(Point2f uv, Float *duv_dw = nullptr) const {
        Float alpha = -Pi / 2 + uv[0] * Pi, beta = -Pi / 2 + uv[1] * Pi;
        Float x = std::tan(alpha), y = std::tan(beta);
        DCHECK(!IsInf(x) && !IsInf(y));
        Vector3f w = Normalize(Vector3f(x, y, 1));
        // Compute Jacobian determinant of mapping $\roman{d}(u,v)/\roman{d}\omega$ if
        // needed
        if (duv_dw)
            *duv_dw = Sqr(Pi) * (1 - Sqr(w.x)) * (1 - Sqr(w.y)) / w.z;

        return portalFrame.FromLocal(w);
    }

    PBRT_CPU_GPU
    pstd::optional<Bounds2f> ImageBounds(const Point3f &p) const {
        pstd::optional<Point2f> p0 = ImageFromRender(Normalize(portal[0] - p));
        pstd::optional<Point2f> p1 = ImageFromRender(Normalize(portal[2] - p));
        if (!p0 || !p1)
            return {};
        return Bounds2f(*p0, *p1);
    }

    PBRT_CPU_GPU
    Float Area() const {
        return Length(portal[1] - portal[0]) * Length(portal[3] - portal[0]);
    }

    // PortalImageInfiniteLight Private Members
    pstd::array<Point3f, 4> portal;
    Frame portalFrame;
    Image image;
    WindowedPiecewiseConstant2D distribution;
    const RGBColorSpace *imageColorSpace;
    Float scale;
    Float sceneRadius;
    std::string filename;
    Point3f sceneCenter;
};

// SpotLight Definition
class SpotLight : public LightBase {
  public:
    // SpotLight Public Methods
    SpotLight(const Transform &renderFromLight, const MediumInterface &m,
              SpectrumHandle I, Float scale, Float totalWidth, Float falloffStart,
              Allocator alloc);

    static SpotLight *Create(const Transform &renderFromLight, MediumHandle medium,
                             const ParameterDictionary &parameters,
                             const RGBColorSpace *colorSpace, const FileLoc *loc,
                             Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {}

    PBRT_CPU_GPU
    SampledSpectrum I(Vector3f w, const SampledWavelengths &) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> SampleLe(Point2f u1, Point2f u2,
                                           SampledWavelengths &lambda, Float time) const;
    PBRT_CPU_GPU
    void PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    void PDF_Le(const Interaction &, Vector3f &w, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Shouldn't be called for non-area lights");
    }

    LightBounds Bounds() const;

    std::string ToString() const;

    PBRT_CPU_GPU
    pstd::optional<LightLiSample> SampleLi(LightSampleContext ctx, Point2f u,
                                           SampledWavelengths lambda,
                                           LightSamplingMode mode) const {
        Point3f p = renderFromLight(Point3f(0, 0, 0));
        Vector3f wi = Normalize(p - ctx.p());
        // Compute incident radiance _Li_ for _SpotLight_
        Vector3f wl = Normalize(renderFromLight.ApplyInverse(-wi));
        SampledSpectrum Li = I(wl, lambda) / DistanceSquared(p, ctx.p());

        if (!Li)
            return {};
        return LightLiSample(Li, wi, 1, Interaction(p, &mediumInterface));
    }

  private:
    // SpotLight Private Members
    DenselySampledSpectrum Iemit;
    Float scale, cosFalloffStart, cosFalloffEnd;
};

inline pstd::optional<LightLiSample> LightHandle::SampleLi(LightSampleContext ctx,
                                                           Point2f u,
                                                           SampledWavelengths lambda,
                                                           LightSamplingMode mode) const {
    auto sample = [&](auto ptr) { return ptr->SampleLi(ctx, u, lambda, mode); };
    return Dispatch(sample);
}

inline Float LightHandle::PDF_Li(LightSampleContext ctx, Vector3f wi,
                                 LightSamplingMode mode) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_Li(ctx, wi, mode); };
    return Dispatch(pdf);
}

inline SampledSpectrum LightHandle::L(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                                      const SampledWavelengths &lambda) const {
    CHECK(Type() == LightType::Area);
    auto l = [&](auto ptr) { return ptr->L(p, n, uv, w, lambda); };
    return Dispatch(l);
}

inline SampledSpectrum LightHandle::Le(const Ray &ray,
                                       const SampledWavelengths &lambda) const {
    auto le = [&](auto ptr) { return ptr->Le(ray, lambda); };
    return Dispatch(le);
}

inline LightType LightHandle::Type() const {
    auto t = [&](auto ptr) { return ptr->Type(); };
    return Dispatch(t);
}

}  // namespace pbrt

#endif  // PBRT_LIGHTS_H
