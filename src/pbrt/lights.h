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
PBRT_CPU_GPU
inline bool IsDeltaLight(LightType type) {
    return (type == LightType::DeltaPosition || type == LightType::DeltaDirection);
}

// LightLiSample Definition
struct LightLiSample {
  public:
    // LightLiSample Public Methods
    PBRT_CPU_GPU
    operator bool() const { return pdf > 0; }

    LightLiSample() = default;
    PBRT_CPU_GPU
    LightLiSample(LightHandle light, const SampledSpectrum &L, const Vector3f &wi,
                  Float pdf, const Interaction &pLight)
        : L(L), wi(wi), pdf(pdf), light(light), pLight(pLight) {}

    SampledSpectrum L;
    Vector3f wi;
    Float pdf = 0;
    LightHandle light;
    Interaction pLight;
};

// LightLeSample Definition
struct LightLeSample {
  public:
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

    PBRT_CPU_GPU
    // FIXME: should this be || or && ?. Review usage...
    operator bool() const { return pdfPos > 0 || pdfDir > 0; }

    SampledSpectrum L;
    Ray ray;
    pstd::optional<Interaction> intr;
    Float pdfPos = 0, pdfDir = 0;
};

// LightSampleContext Definition
class LightSampleContext {
  public:
    LightSampleContext() = default;
    PBRT_CPU_GPU
    LightSampleContext(const SurfaceInteraction &si)
        : pi(si.pi), n(si.n), ns(si.shading.n) {}
    PBRT_CPU_GPU
    LightSampleContext(const Interaction &intr) : pi(intr.pi) {}
    PBRT_CPU_GPU
    LightSampleContext(const Point3fi &pi, const Normal3f &n, const Normal3f &ns)
        : pi(pi), n(n), ns(ns) {}

    PBRT_CPU_GPU
    Point3f p() const { return Point3f(pi); }

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
    LightBase(LightType flags, const Transform &renderFromLight,
              const MediumInterface &mediumInterface);

    PBRT_CPU_GPU
    LightType Type() const { return type; }
    PBRT_CPU_GPU
    SampledSpectrum L(const Point3f &p, const Normal3f &n, const Point2f &uv,
                      const Vector3f &w, const SampledWavelengths &lambda) const {
        return SampledSpectrum(0.f);
    }
    PBRT_CPU_GPU
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const {
        return SampledSpectrum(0.f);
    }

  protected:
    std::string BaseToString() const;
    // LightBase Protected Members
    LightType type;
    MediumInterface mediumInterface;
    Transform renderFromLight;
};

// PointLight Definition
class PointLight : public LightBase {
  public:
    // PointLight Public Methods
    PointLight(const Transform &renderFromLight, const MediumInterface &mediumInterface,
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
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const {
        Point3f p = renderFromLight(Point3f(0, 0, 0));
        Vector3f wi = Normalize(p - ctx.p());
        return LightLiSample(this, scale * I.Sample(lambda) / DistanceSquared(p, ctx.p()),
                             wi, 1, Interaction(p, 0 /* time */, &mediumInterface));
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
    DistantLight(const Transform &renderFromLight, SpectrumHandle L, Float scale,
                 Allocator alloc);

    static DistantLight *Create(const Transform &renderFromLight,
                                const ParameterDictionary &parameters,
                                const RGBColorSpace *colorSpace, const FileLoc *loc,
                                Allocator alloc);

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const { return 0; }

    PBRT_CPU_GPU
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const {
        Vector3f wi = Normalize(renderFromLight(Vector3f(0, 0, 1)));
        Point3f pOutside = ctx.p() + wi * (2 * sceneRadius);
        return LightLiSample(this, scale * Lemit.Sample(lambda), wi, 1,
                             Interaction(pOutside, 0 /* time */, &mediumInterface));
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
    ProjectionLight(const Transform &renderFromLight, const MediumInterface &medium,
                    Image image, const RGBColorSpace *colorSpace, Float scale, Float fov,
                    Float power, Allocator alloc);

    static ProjectionLight *Create(const Transform &renderFromLight, MediumHandle medium,
                                   const ParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {}

    PBRT_CPU_GPU
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const;
    PBRT_CPU_GPU
    SampledSpectrum Projection(const Vector3f &w, const SampledWavelengths &lambda) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    Float hither;
    Transform ScreenFromLight, LightFromScreen;
    Float A, cosTotalWidth;
    PiecewiseConstant2D distrib;
};

// GoniometricLight Definition
class GoniometricLight : public LightBase {
  public:
    // GoniometricLight Public Methods
    GoniometricLight(const Transform &renderFromLight,
                     const MediumInterface &mediumInterface, SpectrumHandle I,
                     Float scale, Image image, const RGBColorSpace *imageColorSpace,
                     Allocator alloc);

    static GoniometricLight *Create(const Transform &renderFromLight, MediumHandle medium,
                                    const ParameterDictionary &parameters,
                                    const RGBColorSpace *colorSpace, const FileLoc *loc,
                                    Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {}

    PBRT_CPU_GPU
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    SampledSpectrum Scale(Vector3f wl, const SampledWavelengths &lambda) const {
        Float theta = SphericalTheta(wl), phi = SphericalPhi(wl);
        Point2f st(phi * Inv2Pi, theta * InvPi);
        return scale * I.Sample(lambda) * image.LookupNearestChannel(st, 0);
    }

  private:
    // GoniometricLight Private Members
    DenselySampledSpectrum I;
    Float scale;
    Image image;
    const RGBColorSpace *imageColorSpace;
    WrapMode2D wrapMode;
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
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    SampledSpectrum L(const Point3f &p, const Normal3f &n, const Point2f &uv,
                      const Vector3f &w, const SampledWavelengths &lambda) const {
        if (!twoSided && Dot(n, w) < 0)
            return SampledSpectrum(0.f);

        if (image) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.BilerpChannel(uv, c);
            return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
        } else
            return scale * Lemit.Sample(lambda);
    }

    PBRT_CPU_GPU
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const {
        // Sample point on shape for _DiffuseAreaLight_
        ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0 /* time */);
        pstd::optional<ShapeSample> ss = shape.Sample(shapeCtx, u);
        if (!ss)
            return {};
        ss->intr.mediumInterface = &mediumInterface;
        DCHECK(!IsNaN(ss->pdf));
        if (ss->pdf == 0 || LengthSquared(ss->intr.p() - ctx.p()) == 0)
            return {};

        // Return _LightLiSample_ for sampled point on shape
        Vector3f wi = Normalize(ss->intr.p() - ctx.p());
        SampledSpectrum Le = L(ss->intr.p(), ss->intr.n, ss->intr.uv, -wi, lambda);
        if (!Le)
            return {};
        return LightLiSample(this, Le, wi, ss->pdf, ss->intr);
    }

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext ctx, Vector3f wi, LightSamplingMode mode) const {
        ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0 /* time */);
        return shape.PDF(shapeCtx, wi);
    }

  private:
    // DiffuseAreaLight Private Members
    DenselySampledSpectrum Lemit;
    Float scale;
    ShapeHandle shape;
    bool twoSided;
    Float area;
    const RGBColorSpace *imageColorSpace;
    Image image;
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
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const;
    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    ImageInfiniteLight(const Transform &renderFromLight, Image image,
                       const RGBColorSpace *imageColorSpace, Float scale,
                       const std::string &filename, Allocator alloc);

    void Preprocess(const Bounds3f &sceneBounds) {
        sceneBounds.BoundingSphere(&sceneCenter, &sceneRadius);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
        Point2f st = EquiAreaSphereToSquare(wl);
        return LookupLe(st, lambda);
    }

    PBRT_CPU_GPU
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const {
        // Find $(u,v)$ sample coordinates in infinite light texture
        Float mapPDF;
        Point2f uv = (mode == LightSamplingMode::WithMIS)
                         ? compensatedDistribution.Sample(u, &mapPDF)
                         : distribution.Sample(u, &mapPDF);
        if (mapPDF == 0)
            return {};

        // Convert infinite light sample point to direction
        Vector3f wl = EquiAreaSquareToSphere(uv);
        Vector3f wi = renderFromLight(wl);

        // Compute PDF for sampled infinite light direction
        Float pdf = mapPDF / (4 * Pi);

        // Return radiance value for infinite light direction
        SampledSpectrum L = LookupLe(uv, lambda);

        return LightLiSample(this, L, wi, pdf,
                             Interaction(ctx.p() + wi * (2 * sceneRadius), 0 /* time */,
                                         &mediumInterface));
    }

    LightBounds Bounds() const { return {}; }

  private:
    // ImageInfiniteLight Private Methods
    PBRT_CPU_GPU
    SampledSpectrum LookupLe(Point2f st, const SampledWavelengths &lambda) const {
        RGB rgb;
        for (int c = 0; c < 3; ++c)
            rgb[c] = image.LookupNearestChannel(st, c, wrapMode);
        return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
    }

    // ImageInfiniteLight Private Members
    std::string filename;
    Image image;
    const RGBColorSpace *imageColorSpace;
    Float scale;
    WrapMode2D wrapMode;
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
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    SampledSpectrum ImageLookup(const Point2f &st,
                                const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Vector3f RenderFromImage(const Point2f &st, Float *duv_dw = nullptr) const {
        Float alpha = -Pi / 2 + st.x * Pi, beta = -Pi / 2 + st.y * Pi;
        Float x = std::tan(alpha), y = std::tan(beta);
        DCHECK(!IsInf(x) && !IsInf(y));
        Vector3f w = Normalize(Vector3f(x, y, -1));

        if (w.z == 0)
            w.z = 1e-5;
        if (duv_dw)
            *duv_dw = Pi * Pi * std::abs((1 - w.y * w.y) * (1 - w.x * w.x) / w.z);
        return portalFrame.FromLocal(w);
    }

    PBRT_CPU_GPU
    Point2f ImageFromRender(const Vector3f &wRender, Float *duv_dw = nullptr) const {
        Vector3f w = portalFrame.ToLocal(wRender);
        if (w.z == 0)
            w.z = 1e-5;
        if (duv_dw)
            *duv_dw = Pi * Pi * std::abs((1 - w.y * w.y) * (1 - w.x * w.x) / w.z);

        Float alpha = std::atan(w.x / -w.z), beta = std::atan(w.y / -w.z);
        DCHECK(!IsNaN(alpha + beta));
        return Point2f(Clamp((alpha + Pi / 2) / Pi, 0, 1),
                       Clamp((beta + Pi / 2) / Pi, 0, 1));
    }

    PBRT_CPU_GPU
    Bounds2f ImageBounds(const Point3f &p) const {
        Point2f p0 = ImageFromRender(Normalize(portal[0] - p));
        Point2f p1 = ImageFromRender(Normalize(portal[2] - p));
        return Bounds2f(p0, p1);
    }

    PBRT_CPU_GPU
    Float Area() const {
        return Length(portal[1] - portal[0]) * Length(portal[3] - portal[0]);
    }

    // PortalImageInfiniteLight Private Members
    std::string filename;
    Image image;
    const RGBColorSpace *imageColorSpace;
    Float scale;
    Frame portalFrame;
    pstd::array<Point3f, 4> portal;
    WindowedPiecewiseConstant2D distribution;
    Point3f sceneCenter;
    Float sceneRadius;
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
    LightLiSample SampleLi(LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
                           LightSamplingMode mode) const;
    PBRT_CPU_GPU
    Float Falloff(const Vector3f &w) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float PDF_Li(LightSampleContext, Vector3f, LightSamplingMode mode) const;

    PBRT_CPU_GPU
    LightLeSample SampleLe(const Point2f &u1, const Point2f &u2,
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
    // SpotLight Private Members
    DenselySampledSpectrum I;
    Float scale;
    Float cosFalloffStart, cosFalloffEnd;
};

inline LightLiSample LightHandle::SampleLi(LightSampleContext ctx, Point2f u,
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

inline SampledSpectrum LightHandle::L(const Point3f &p, const Normal3f &n,
                                      const Point2f &uv, const Vector3f &w,
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
