// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_TEXTURES_H
#define PBRT_TEXTURES_H

#include <pbrt/pbrt.h>

#include <pbrt/base/texture.h>
#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/math.h>
#include <pbrt/util/mipmap.h>
#include <pbrt/util/noise.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <initializer_list>
#include <map>
#include <mutex>
#include <string>

#if defined(__HIPCC__)
#include <pbrt/util/hip_aliases.h>
#endif

namespace pbrt {

// TextureEvalContext Definition
struct TextureEvalContext {
    // TextureEvalContext Public Methods
    TextureEvalContext() = default;
    PBRT_CPU_GPU
    TextureEvalContext(const Interaction &intr) : p(intr.p()), uv(intr.uv) {}
    PBRT_CPU_GPU
    TextureEvalContext(const SurfaceInteraction &si)
        : p(si.p()),
          dpdx(si.dpdx),
          dpdy(si.dpdy),
          n(si.n),
          uv(si.uv),
          dudx(si.dudx),
          dudy(si.dudy),
          dvdx(si.dvdx),
          dvdy(si.dvdy),
          faceIndex(si.faceIndex) {}
    PBRT_CPU_GPU
    TextureEvalContext(Point3f p, Vector3f dpdx, Vector3f dpdy, Normal3f n, Point2f uv,
                       Float dudx, Float dudy, Float dvdx, Float dvdy, int faceIndex)
        : p(p),
          dpdx(dpdx),
          dpdy(dpdy),
          n(n),
          uv(uv),
          dudx(dudx),
          dudy(dudy),
          dvdx(dvdx),
          dvdy(dvdy),
          faceIndex(faceIndex) {}

    std::string ToString() const;

    Point3f p;
    Vector3f dpdx, dpdy;
    Normal3f n;
    Point2f uv;
    Float dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
    int faceIndex = 0;
};

// TexCoord2D Definition
struct TexCoord2D {
    Point2f st;
    Float dsdx, dsdy, dtdx, dtdy;
    std::string ToString() const;
};

// TexCoord3D Definition
struct TexCoord3D {
    Point3f p;
    Vector3f dpdx, dpdy;
    std::string ToString() const;
};

// UVMapping Definition
class UVMapping {
  public:
    // UVMapping Public Methods
    UVMapping(Float su = 1, Float sv = 1, Float du = 0, Float dv = 0)
        : su(su), sv(sv), du(du), dv(dv) {}

    std::string ToString() const;

    PBRT_CPU_GPU
    TexCoord2D Map(TextureEvalContext ctx) const {
        // Compute texture differentials for 2D $(u,v)$ mapping
        Float dsdx = su * ctx.dudx, dsdy = su * ctx.dudy;
        Float dtdx = sv * ctx.dvdx, dtdy = sv * ctx.dvdy;

        Point2f st(su * ctx.uv[0] + du, sv * ctx.uv[1] + dv);
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

  private:
    Float su, sv, du, dv;
};

// SphericalMapping Definition
class SphericalMapping {
  public:
    // SphericalMapping Public Methods
    SphericalMapping(const Transform &textureFromRender)
        : textureFromRender(textureFromRender) {}

    std::string ToString() const;

    PBRT_CPU_GPU
    TexCoord2D Map(TextureEvalContext ctx) const {
        Point3f pt = textureFromRender(ctx.p);
        // Compute $\partial\,s/\partial\,\pt{}$ and $\partial\,t/\partial\,\pt{}$ for
        // spherical mapping
        Float x2y2 = Sqr(pt.x) + Sqr(pt.y);
        Float sqrtx2y2 = std::sqrt(x2y2);
        Vector3f dsdp = Vector3f(-pt.y, pt.x, 0) / (2 * Pi * x2y2);
        Vector3f dtdp =
            1 / (Pi * (x2y2 + Sqr(pt.z))) *
            Vector3f(pt.x * pt.z / sqrtx2y2, pt.y * pt.z / sqrtx2y2, -sqrtx2y2);

        // Compute texture coordinate differentials for spherical mapping
        Vector3f dpdx = textureFromRender(ctx.dpdx);
        Vector3f dpdy = textureFromRender(ctx.dpdy);
        Float dsdx = Dot(dsdp, dpdx), dsdy = Dot(dsdp, dpdy);
        Float dtdx = Dot(dtdp, dpdx), dtdy = Dot(dtdp, dpdy);

        // Return $(s,t)$ texture coordinates and differentials based on spherical mapping
        Vector3f vec = Normalize(pt - Point3f(0, 0, 0));
        Point2f st(SphericalTheta(vec) * InvPi, SphericalPhi(vec) * Inv2Pi);
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

  private:
    // SphericalMapping Private Members
    Transform textureFromRender;
};

// CylindricalMapping Definition
class CylindricalMapping {
  public:
    // CylindricalMapping Public Methods
    CylindricalMapping(const Transform &textureFromRender)
        : textureFromRender(textureFromRender) {}
    std::string ToString() const;

    PBRT_CPU_GPU
    TexCoord2D Map(TextureEvalContext ctx) const {
        Point3f pt = textureFromRender(ctx.p);
        // Compute texture coordinate differentials for cylinder $(u,v)$ mapping
        Float x2y2 = Sqr(pt.x) + Sqr(pt.y);
        Vector3f dsdp = Vector3f(-pt.y, pt.x, 0) / (2 * Pi * x2y2),
                 dtdp = Vector3f(0, 0, 1);
        Vector3f dpdx = textureFromRender(ctx.dpdx), dpdy = textureFromRender(ctx.dpdy);
        Float dsdx = Dot(dsdp, dpdx), dsdy = Dot(dsdp, dpdy);
        Float dtdx = Dot(dtdp, dpdx), dtdy = Dot(dtdp, dpdy);

        Point2f st((Pi + std::atan2(pt.y, pt.x)) * Inv2Pi, pt.z);
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

  private:
    // CylindricalMapping Private Members
    Transform textureFromRender;
};

// PlanarMapping Definition
class PlanarMapping {
  public:
    // PlanarMapping Public Methods
    PlanarMapping(const Transform &textureFromRender, Vector3f vs, Vector3f vt, Float ds,
                  Float dt)
        : textureFromRender(textureFromRender), vs(vs), vt(vt), ds(ds), dt(dt) {}

    PBRT_CPU_GPU
    TexCoord2D Map(TextureEvalContext ctx) const {
        Vector3f vec(textureFromRender(ctx.p));
        // Initialize partial derivatives of planar mapping $(s,t)$ coordinates
        Vector3f dpdx = textureFromRender(ctx.dpdx);
        Vector3f dpdy = textureFromRender(ctx.dpdy);
        Float dsdx = Dot(vs, dpdx), dsdy = Dot(vs, dpdy);
        Float dtdx = Dot(vt, dpdx), dtdy = Dot(vt, dpdy);

        Point2f st(ds + Dot(vec, vs), dt + Dot(vec, vt));
        return TexCoord2D{st, dsdx, dsdy, dtdx, dtdy};
    }

    std::string ToString() const;

  private:
    // PlanarMapping Private Members
    Transform textureFromRender;
    Vector3f vs, vt;
    Float ds, dt;
};

// TextureMapping2D Definition
class TextureMapping2D : public TaggedPointer<UVMapping, SphericalMapping,
                                              CylindricalMapping, PlanarMapping> {
  public:
    // TextureMapping2D Interface
    using TaggedPointer::TaggedPointer;
    PBRT_CPU_GPU
    TextureMapping2D(
        TaggedPointer<UVMapping, SphericalMapping, CylindricalMapping, PlanarMapping> tp)
        : TaggedPointer(tp) {}

    static TextureMapping2D Create(const ParameterDictionary &parameters,
                                   const Transform &renderFromTexture, const FileLoc *loc,
                                   Allocator alloc);

    PBRT_CPU_GPU inline TexCoord2D Map(TextureEvalContext ctx) const;
};

// TextureMapping2D Inline Functions
PBRT_CPU_GPU inline TexCoord2D TextureMapping2D::Map(TextureEvalContext ctx) const {
    auto map = [&](auto ptr) { return ptr->Map(ctx); };
    return Dispatch(map);
}

// PointTransformMapping Definition
class PointTransformMapping {
  public:
    // PointTransformMapping Public Methods
    PointTransformMapping(const Transform &textureFromRender)
        : textureFromRender(textureFromRender) {}

    std::string ToString() const;

    PBRT_CPU_GPU
    TexCoord3D Map(TextureEvalContext ctx) const {
        return TexCoord3D{textureFromRender(ctx.p), textureFromRender(ctx.dpdx),
                          textureFromRender(ctx.dpdy)};
    }

  private:
    Transform textureFromRender;
};

// TextureMapping3D Definition
class TextureMapping3D : public TaggedPointer<PointTransformMapping> {
  public:
    // TextureMapping3D Interface
    using TaggedPointer::TaggedPointer;
    PBRT_CPU_GPU
    TextureMapping3D(TaggedPointer<PointTransformMapping> tp) : TaggedPointer(tp) {}

    static TextureMapping3D Create(const ParameterDictionary &parameters,
                                   const Transform &renderFromTexture, const FileLoc *loc,
                                   Allocator alloc);

    PBRT_CPU_GPU
    TexCoord3D Map(TextureEvalContext ctx) const;
};

PBRT_CPU_GPU inline TexCoord3D TextureMapping3D::Map(TextureEvalContext ctx) const {
    auto map = [&](auto ptr) { return ptr->Map(ctx); };
    return Dispatch(map);
}

// FloatConstantTexture Definition
class FloatConstantTexture {
  public:
    FloatConstantTexture(Float value) : value(value) {}
    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const { return value; }
    // FloatConstantTexture Public Methods
    static FloatConstantTexture *Create(const Transform &renderFromTexture,
                                        const TextureParameterDictionary &parameters,
                                        const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    Float value;
};

// SpectrumConstantTexture Definition
class SpectrumConstantTexture {
  public:
    // SpectrumConstantTexture Public Methods
    SpectrumConstantTexture(Spectrum value) : value(value) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        return value.Sample(lambda);
    }

    static SpectrumConstantTexture *Create(const Transform &renderFromTexture,
                                           const TextureParameterDictionary &parameters,
                                           SpectrumType spectrumType, const FileLoc *loc,
                                           Allocator alloc);
    std::string ToString() const;

  private:
    Spectrum value;
};

// FloatBilerpTexture Definition
class FloatBilerpTexture {
  public:
    FloatBilerpTexture(TextureMapping2D mapping, Float v00, Float v01, Float v10,
                       Float v11)
        : mapping(mapping), v00(v00), v01(v01), v10(v10), v11(v11) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        TexCoord2D c = mapping.Map(ctx);
        return (1 - c.st[0]) * (1 - c.st[1]) * v00 + c.st[0] * (1 - c.st[1]) * v10 +
               (1 - c.st[0]) * c.st[1] * v01 + c.st[0] * c.st[1] * v11;
    }

    static FloatBilerpTexture *Create(const Transform &renderFromTexture,
                                      const TextureParameterDictionary &parameters,
                                      const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // BilerpTexture Private Data
    TextureMapping2D mapping;
    Float v00, v01, v10, v11;
};

// SpectrumBilerpTexture Definition
class SpectrumBilerpTexture {
  public:
    SpectrumBilerpTexture(TextureMapping2D mapping, Spectrum v00, Spectrum v01,
                          Spectrum v10, Spectrum v11)
        : mapping(mapping), v00(v00), v01(v01), v10(v10), v11(v11) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        TexCoord2D c = mapping.Map(ctx);
        return Bilerp({c.st[0], c.st[1]}, {v00.Sample(lambda), v10.Sample(lambda),
                                           v01.Sample(lambda), v11.Sample(lambda)});
    }

    static SpectrumBilerpTexture *Create(const Transform &renderFromTexture,
                                         const TextureParameterDictionary &parameters,
                                         SpectrumType spectrumType, const FileLoc *loc,
                                         Allocator alloc);

    std::string ToString() const;

  private:
    // BilerpTexture Private Data
    TextureMapping2D mapping;
    Spectrum v00, v01, v10, v11;
};

PBRT_CPU_GPU
Float Checkerboard(TextureEvalContext ctx, TextureMapping2D map2D,
                   TextureMapping3D map3D);

class FloatCheckerboardTexture {
  public:
    FloatCheckerboardTexture(TextureMapping2D map2D, TextureMapping3D map3D,
                             FloatTexture tex1, FloatTexture tex2)
        : map2D(map2D), map3D(map3D), tex{tex1, tex2} {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        Float w = Checkerboard(ctx, map2D, map3D);
        Float t0 = 0, t1 = 0;
        if (w != 1)
            t0 = tex[0].Evaluate(ctx);
        if (w != 0)
            t1 = tex[1].Evaluate(ctx);
        return (1 - w) * t0 + w * t1;
    }

    static FloatCheckerboardTexture *Create(const Transform &renderFromTexture,
                                            const TextureParameterDictionary &parameters,
                                            const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping2D map2D;
    TextureMapping3D map3D;
    FloatTexture tex[2];
};

// SpectrumCheckerboardTexture Definition
class SpectrumCheckerboardTexture {
  public:
    // SpectrumCheckerboardTexture Public Methods
    SpectrumCheckerboardTexture(TextureMapping2D map2D, TextureMapping3D map3D,
                                SpectrumTexture tex1, SpectrumTexture tex2)
        : map2D(map2D), map3D(map3D), tex{tex1, tex2} {}

    static SpectrumCheckerboardTexture *Create(
        const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
        SpectrumType spectrumType, const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        Float w = Checkerboard(ctx, map2D, map3D);
        SampledSpectrum t0, t1;
        if (w != 1)
            t0 = tex[0].Evaluate(ctx, lambda);
        if (w != 0)
            t1 = tex[1].Evaluate(ctx, lambda);
        return (1 - w) * t0 + w * t1;
    }

  private:
    // SpectrumCheckerboardTexture Private Members
    TextureMapping2D map2D;
    TextureMapping3D map3D;
    SpectrumTexture tex[2];
};

PBRT_CPU_GPU
bool InsidePolkaDot(Point2f st);

class FloatDotsTexture {
  public:
    FloatDotsTexture(TextureMapping2D mapping, FloatTexture outsideDot,
                     FloatTexture insideDot)
        : mapping(mapping), outsideDot(outsideDot), insideDot(insideDot) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        TexCoord2D c = mapping.Map(ctx);
        return InsidePolkaDot(c.st) ? insideDot.Evaluate(ctx) : outsideDot.Evaluate(ctx);
    }

    static FloatDotsTexture *Create(const Transform &renderFromTexture,
                                    const TextureParameterDictionary &parameters,
                                    const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // DotsTexture Private Data
    TextureMapping2D mapping;
    FloatTexture outsideDot, insideDot;
};

// SpectrumDotsTexture Definition
class SpectrumDotsTexture {
  public:
    // SpectrumDotsTexture Public Methods
    SpectrumDotsTexture(TextureMapping2D mapping, SpectrumTexture outsideDot,
                        SpectrumTexture insideDot)
        : mapping(mapping), outsideDot(outsideDot), insideDot(insideDot) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        TexCoord2D c = mapping.Map(ctx);
        return InsidePolkaDot(c.st) ? insideDot.Evaluate(ctx, lambda)
                                    : outsideDot.Evaluate(ctx, lambda);
    }

    static SpectrumDotsTexture *Create(const Transform &renderFromTexture,
                                       const TextureParameterDictionary &parameters,
                                       SpectrumType spectrumType, const FileLoc *loc,
                                       Allocator alloc);

    std::string ToString() const;

  private:
    // SpectrumDotsTexture Private Members
    TextureMapping2D mapping;
    SpectrumTexture outsideDot, insideDot;
};

// FBmTexture Definition
class FBmTexture {
  public:
    // FBmTexture Public Methods
    FBmTexture(TextureMapping3D mapping, int octaves, Float omega)
        : mapping(mapping), omega(omega), octaves(octaves) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        TexCoord3D c = mapping.Map(ctx);
        return FBm(c.p, c.dpdx, c.dpdy, omega, octaves);
    }

    static FBmTexture *Create(const Transform &renderFromTexture,
                              const TextureParameterDictionary &parameters,
                              const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping3D mapping;
    Float omega;
    int octaves;
};

// TexInfo Definition
struct TexInfo {
    // TexInfo Public Methods
    TexInfo(const std::string &f, MIPMapFilterOptions filterOptions, WrapMode wm,
            ColorEncoding encoding)
        : filename(f), filterOptions(filterOptions), wrapMode(wm), encoding(encoding) {}

    bool operator<(const TexInfo &t) const {
        return std::tie(filename, filterOptions, encoding, wrapMode) <
               std::tie(t.filename, t.filterOptions, t.encoding, t.wrapMode);
    }

    std::string ToString() const;

    std::string filename;
    MIPMapFilterOptions filterOptions;
    WrapMode wrapMode;
    ColorEncoding encoding;
};

// ImageTextureBase Definition
class ImageTextureBase {
  public:
    // ImageTextureBase Public Methods
    ImageTextureBase(TextureMapping2D mapping, std::string filename,
                     MIPMapFilterOptions filterOptions, WrapMode wrapMode, Float scale,
                     bool invert, ColorEncoding encoding, Allocator alloc)
        : mapping(mapping), filename(filename), scale(scale), invert(invert) {
        // Get _MIPMap_ from texture cache if present
        TexInfo texInfo(filename, filterOptions, wrapMode, encoding);
        std::unique_lock<std::mutex> lock(textureCacheMutex);
        if (auto iter = textureCache.find(texInfo); iter != textureCache.end()) {
            mipmap = iter->second;
            return;
        }
        lock.unlock();

        // Create _MIPMap_ for _filename_ and add to texture cache
        mipmap =
            MIPMap::CreateFromFile(filename, filterOptions, wrapMode, encoding, alloc);
        lock.lock();
        // This is actually ok, but if it hits, it means we've wastefully
        // loaded this texture. (Note that in that case, should just return
        // the one that's already in there and not replace it.)
        CHECK(textureCache.find(texInfo) == textureCache.end());
        textureCache[texInfo] = mipmap;
    }

    static void ClearCache() { textureCache.clear(); }

    void MultiplyScale(Float s) { scale *= s; }

  protected:
    // ImageTextureBase Protected Members
    TextureMapping2D mapping;
    std::string filename;
    Float scale;
    bool invert;
    MIPMap *mipmap;

  private:
    // ImageTextureBase Private Members
    static std::mutex textureCacheMutex;
    static std::map<TexInfo, MIPMap *> textureCache;
};

// FloatImageTexture Definition
class FloatImageTexture : public ImageTextureBase {
  public:
    FloatImageTexture(TextureMapping2D m, const std::string &filename,
                      MIPMapFilterOptions filterOptions, WrapMode wm, Float scale,
                      bool invert, ColorEncoding encoding, Allocator alloc)
        : ImageTextureBase(m, filename, filterOptions, wm, scale, invert, encoding,
                           alloc) {}
    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
#ifdef PBRT_IS_GPU_CODE
        CHECK(!"Should not be called in GPU code");
        return 0;
#else
        TexCoord2D c = mapping.Map(ctx);
        // Texture coordinates are (0,0) in the lower left corner, but
        // image coordinates are (0,0) in the upper left.
        c.st[1] = 1 - c.st[1];
        Float v = scale * mipmap->Filter<Float>(c.st, {c.dsdx, c.dtdx}, {c.dsdy, c.dtdy});
        return invert ? std::max<Float>(0, 1 - v) : v;
#endif
    }

    static FloatImageTexture *Create(const Transform &renderFromTexture,
                                     const TextureParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    std::string ToString() const;
};

// SpectrumImageTexture Definition
class SpectrumImageTexture : public ImageTextureBase {
  public:
    // SpectrumImageTexture Public Methods
    SpectrumImageTexture(TextureMapping2D mapping, std::string filename,
                         MIPMapFilterOptions filterOptions, WrapMode wrapMode,
                         Float scale, bool invert, ColorEncoding encoding,
                         SpectrumType spectrumType, Allocator alloc)
        : ImageTextureBase(mapping, filename, filterOptions, wrapMode, scale, invert,
                           encoding, alloc),
          spectrumType(spectrumType) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const;

    static SpectrumImageTexture *Create(const Transform &renderFromTexture,
                                        const TextureParameterDictionary &parameters,
                                        SpectrumType spectrumType, const FileLoc *loc,
                                        Allocator alloc);

    std::string ToString() const;

  private:
    // SpectrumImageTexture Private Members
    SpectrumType spectrumType;
};

#if defined(PBRT_BUILD_GPU_RENDERER) && (defined(__NVCC__) || defined(__HIPCC__))
class GPUSpectrumImageTexture {
  public:
    GPUSpectrumImageTexture(std::string filename, TextureMapping2D mapping,
                            cudaTextureObject_t texObj, Float scale, bool invert,
                            bool isSingleChannel, const RGBColorSpace *colorSpace,
                            SpectrumType spectrumType)
        : mapping(mapping),
          filename(filename),
          texObj(texObj),
          scale(scale),
          invert(invert),
          isSingleChannel(isSingleChannel),
          colorSpace(colorSpace),
          spectrumType(spectrumType) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
#ifndef PBRT_IS_GPU_CODE
        LOG_FATAL("GPUSpectrumImageTexture::Evaluate called from CPU");
        return SampledSpectrum(0);
#else
        // flip y coord since image has (0,0) at upper left, texture at lower
        // left
        TexCoord2D c = mapping.Map(ctx);
        RGB rgb;
        if (isSingleChannel) {
            float tex = scale * tex2DGrad<float>(texObj, c.st[0], 1 - c.st[1],
                                                 make_float2(c.dsdx, c.dsdy),
                                                 make_float2(c.dtdx, c.dtdy));
            rgb = RGB(tex, tex, tex);
        } else {
            float4 tex = tex2DGrad<float4>(texObj, c.st[0], 1 - c.st[1],
                                           make_float2(c.dsdx, c.dsdy),
                                           make_float2(c.dtdx, c.dtdy));
            rgb = scale * RGB(tex.x, tex.y, tex.z);
        }
        if (invert)
            rgb = ClampZero(RGB(1, 1, 1) - rgb);
        if (spectrumType == SpectrumType::Unbounded)
            return RGBUnboundedSpectrum(*colorSpace, rgb).Sample(lambda);
        else if (spectrumType == SpectrumType::Albedo) {
            rgb = Clamp(rgb, 0, 1);
            return RGBAlbedoSpectrum(*colorSpace, rgb).Sample(lambda);
        } else
            return RGBIlluminantSpectrum(*colorSpace, rgb).Sample(lambda);
#endif
    }

    static GPUSpectrumImageTexture *Create(const Transform &renderFromTexture,
                                           const TextureParameterDictionary &parameters,
                                           SpectrumType spectrumType, const FileLoc *loc,
                                           Allocator alloc);

    std::string ToString() const;

    void MultiplyScale(Float s) { scale *= s; }

    TextureMapping2D mapping;
    std::string filename;
    cudaTextureObject_t texObj;
    Float scale;
    bool invert, isSingleChannel;
    const RGBColorSpace *colorSpace;
    SpectrumType spectrumType;
};

class GPUFloatImageTexture {
  public:
    GPUFloatImageTexture(std::string filename, TextureMapping2D mapping,
                         cudaTextureObject_t texObj, Float scale, bool invert)
        : mapping(mapping),
          filename(filename),
          texObj(texObj),
          scale(scale),
          invert(invert) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
#ifndef PBRT_IS_GPU_CODE
        LOG_FATAL("GPUSpectrumImageTexture::Evaluate called from CPU");
        return 0;
#else
        TexCoord2D c = mapping.Map(ctx);
        // flip y coord since image has (0,0) at upper left, texture at lower
        // left
        Float v = scale * tex2DGrad<float>(texObj, c.st[0], 1 - c.st[1],
                                           make_float2(c.dsdx, c.dsdy),
                                           make_float2(c.dtdx, c.dtdy));
        return invert ? std::max<Float>(0, 1 - v) : v;
#endif
    }

    static GPUFloatImageTexture *Create(const Transform &renderFromTexture,
                                        const TextureParameterDictionary &parameters,
                                        const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

    void MultiplyScale(Float s) { scale *= s; }

    TextureMapping2D mapping;
    std::string filename;
    cudaTextureObject_t texObj;
    Float scale;
    bool invert;
};

#else  // PBRT_BUILD_GPU_RENDERER && (__NVCC__ ||  __HIPCC__)

class GPUSpectrumImageTexture {
  public:
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        LOG_FATAL("GPUSpectrumImageTexture::Evaluate called from CPU");
        return SampledSpectrum(0);
    }

    static GPUSpectrumImageTexture *Create(const Transform &renderFromTexture,
                                           const TextureParameterDictionary &parameters,
                                           SpectrumType spectrumType, const FileLoc *loc,
                                           Allocator alloc) {
        LOG_FATAL("GPUSpectrumImageTexture::Create called in non-GPU configuration.");
        return nullptr;
    }

    std::string ToString() const { return "GPUSpectrumImageTexture"; }
};

class GPUFloatImageTexture {
  public:
    Float Evaluate(const TextureEvalContext &) const {
        LOG_FATAL("GPUFloatImageTexture::Evaluate called from CPU");
        return 0;
    }

    static GPUFloatImageTexture *Create(const Transform &renderFromTexture,
                                        const TextureParameterDictionary &parameters,
                                        const FileLoc *loc, Allocator alloc) {
        LOG_FATAL("GPUFloatImageTexture::Create called in non-GPU configuration.");
        return nullptr;
    }

    std::string ToString() const { return "GPUFloatImageTexture"; }
};

#endif  // PBRT_BUILD_GPU_RENDERER && (__NVCC__ ||  __HIPCC__)

// MarbleTexture Definition
class MarbleTexture {
  public:
    // MarbleTexture Public Methods
    MarbleTexture(TextureMapping3D mapping, int octaves, Float omega, Float scale,
                  Float variation)
        : mapping(mapping),
          octaves(octaves),
          omega(omega),
          scale(scale),
          variation(variation) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const;

    static MarbleTexture *Create(const Transform &renderFromTexture,
                                 const TextureParameterDictionary &parameters,
                                 const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // MarbleTexture Private Members
    TextureMapping3D mapping;
    int octaves;
    Float omega, scale, variation;
};

// FloatMixTexture Definition
class FloatMixTexture {
  public:
    // FloatMixTexture Public Methods
    FloatMixTexture(FloatTexture tex1, FloatTexture tex2, FloatTexture amount)
        : tex1(tex1), tex2(tex2), amount(amount) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        Float amt = amount.Evaluate(ctx);
        Float t1 = 0, t2 = 0;
        if (amt != 1)
            t1 = tex1.Evaluate(ctx);
        if (amt != 0)
            t2 = tex2.Evaluate(ctx);
        return (1 - amt) * t1 + amt * t2;
    }

    static FloatMixTexture *Create(const Transform &renderFromTexture,
                                   const TextureParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    FloatTexture tex1, tex2;
    FloatTexture amount;
};

// FloatDirectionMixTexture Definition
class FloatDirectionMixTexture {
  public:
    // FloatDirectionMixTexture Public Methods
    FloatDirectionMixTexture(FloatTexture tex1, FloatTexture tex2, Vector3f dir)
        : tex1(tex1), tex2(tex2), dir(dir) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        Float amt = AbsDot(ctx.n, dir);
        Float t1 = 0, t2 = 0;
        if (amt != 0)
            t1 = tex1.Evaluate(ctx);
        if (amt != 1)
            t2 = tex2.Evaluate(ctx);
        return amt * t1 + (1 - amt) * t2;
    }

    static FloatDirectionMixTexture *Create(const Transform &renderFromTexture,
                                            const TextureParameterDictionary &parameters,
                                            const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // FloatDirectionMixTexture Private Members
    FloatTexture tex1, tex2;
    Vector3f dir;
};

// SpectrumMixTexture Definition
class SpectrumMixTexture {
  public:
    SpectrumMixTexture(SpectrumTexture tex1, SpectrumTexture tex2, FloatTexture amount)
        : tex1(tex1), tex2(tex2), amount(amount) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        Float amt = amount.Evaluate(ctx);
        SampledSpectrum t1, t2;
        if (amt != 1)
            t1 = tex1.Evaluate(ctx, lambda);
        if (amt != 0)
            t2 = tex2.Evaluate(ctx, lambda);
        return (1 - amt) * t1 + amt * t2;
    }

    static SpectrumMixTexture *Create(const Transform &renderFromTexture,
                                      const TextureParameterDictionary &parameters,
                                      SpectrumType spectrumType, const FileLoc *loc,
                                      Allocator alloc);

    std::string ToString() const;

  private:
    SpectrumTexture tex1, tex2;
    FloatTexture amount;
};

// SpectrumDirectionMixTexture Definition
class SpectrumDirectionMixTexture {
  public:
    // SpectrumDirectionMixTexture Public Methods
    SpectrumDirectionMixTexture(SpectrumTexture tex1, SpectrumTexture tex2, Vector3f dir)
        : tex1(tex1), tex2(tex2), dir(dir) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        Float amt = AbsDot(ctx.n, dir);
        SampledSpectrum t1, t2;
        if (amt != 0)
            t1 = tex1.Evaluate(ctx, lambda);
        if (amt != 1)
            t2 = tex2.Evaluate(ctx, lambda);
        return amt * t1 + (1 - amt) * t2;
    }

    static SpectrumDirectionMixTexture *Create(
        const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
        SpectrumType spectrumType, const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // SpectrumDirectionMixTexture Private Members
    SpectrumTexture tex1, tex2;
    Vector3f dir;
};

// PtexTexture Declarations
class PtexTextureBase {
  public:
    PtexTextureBase(const std::string &filename, ColorEncoding encoding, Float scale);

    static void ReportStats();

    int SampleTexture(TextureEvalContext ctx, float *result) const;

  protected:
    std::string BaseToString() const;

  private:
    bool valid;
    std::string filename;
    ColorEncoding encoding;
    Float scale;
};

class FloatPtexTexture : public PtexTextureBase {
  public:
    FloatPtexTexture(const std::string &filename, ColorEncoding encoding, Float scale)
        : PtexTextureBase(filename, encoding, scale) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const;
    static FloatPtexTexture *Create(const Transform &renderFromTexture,
                                    const TextureParameterDictionary &parameters,
                                    const FileLoc *loc, Allocator alloc);
    std::string ToString() const;
};

class SpectrumPtexTexture : public PtexTextureBase {
  public:
    SpectrumPtexTexture(const std::string &filename, ColorEncoding encoding, Float scale,
                        SpectrumType spectrumType)
        : PtexTextureBase(filename, encoding, scale), spectrumType(spectrumType) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const;

    static SpectrumPtexTexture *Create(const Transform &renderFromTexture,
                                       const TextureParameterDictionary &parameters,
                                       SpectrumType spectrumType, const FileLoc *loc,
                                       Allocator alloc);

    std::string ToString() const;

  private:
    SpectrumType spectrumType;
};

class GPUFloatPtexTexture {
  public:
    GPUFloatPtexTexture(const std::string &filename, ColorEncoding encoding, Float scale,
                        Allocator alloc);

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        DCHECK(ctx.faceIndex >= 0 && ctx.faceIndex < faceValues.size());
        return faceValues[ctx.faceIndex];
    }

    static GPUFloatPtexTexture *Create(const Transform &renderFromTexture,
                                       const TextureParameterDictionary &parameters,
                                       const FileLoc *loc, Allocator alloc);
    std::string ToString() const;

  private:
    pstd::vector<Float> faceValues;
};

class GPUSpectrumPtexTexture {
  public:
    GPUSpectrumPtexTexture(const std::string &filename, ColorEncoding encoding,
                           Float scale, SpectrumType spectrumType, Allocator alloc);

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        CHECK(ctx.faceIndex >= 0 && ctx.faceIndex < faceValues.size());

        RGB rgb = faceValues[ctx.faceIndex];
        const RGBColorSpace *sRGB =
#ifdef PBRT_IS_GPU_CODE
            RGBColorSpace_sRGB;
#else
            RGBColorSpace::sRGB;
#endif
        if (spectrumType == SpectrumType::Unbounded)
            return RGBUnboundedSpectrum(*sRGB, rgb).Sample(lambda);
        else if (spectrumType == SpectrumType::Albedo)
            return RGBAlbedoSpectrum(*sRGB, Clamp(rgb, 0, 1)).Sample(lambda);
        else
            return RGBIlluminantSpectrum(*sRGB, rgb).Sample(lambda);
    }

    static GPUSpectrumPtexTexture *Create(const Transform &renderFromTexture,
                                          const TextureParameterDictionary &parameters,
                                          SpectrumType spectrumType, const FileLoc *loc,
                                          Allocator alloc);

    std::string ToString() const;

  private:
    SpectrumType spectrumType;
    pstd::vector<RGB> faceValues;
};

// FloatScaledTexture Definition
class FloatScaledTexture {
  public:
    // FloatScaledTexture Public Methods
    FloatScaledTexture(FloatTexture tex, FloatTexture scale) : tex(tex), scale(scale) {}

    static FloatTexture Create(const Transform &renderFromTexture,
                               const TextureParameterDictionary &parameters,
                               const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        Float sc = scale.Evaluate(ctx);
        if (sc == 0)
            return 0;
        return tex.Evaluate(ctx) * sc;
    }

    std::string ToString() const;

  private:
    FloatTexture tex, scale;
};

// SpectrumScaledTexture Definition
class SpectrumScaledTexture {
  public:
    SpectrumScaledTexture(SpectrumTexture tex, FloatTexture scale)
        : tex(tex), scale(scale) {}

    PBRT_CPU_GPU
    SampledSpectrum Evaluate(TextureEvalContext ctx, SampledWavelengths lambda) const {
        Float sc = scale.Evaluate(ctx);
        if (sc == 0)
            return SampledSpectrum(0.f);
        return tex.Evaluate(ctx, lambda) * sc;
    }

    static SpectrumTexture Create(const Transform &renderFromTexture,
                                  const TextureParameterDictionary &parameters,
                                  SpectrumType spectrumType, const FileLoc *loc,
                                  Allocator alloc);

    std::string ToString() const;

  private:
    SpectrumTexture tex;
    FloatTexture scale;
};

// WindyTexture Definition
class WindyTexture {
  public:
    // WindyTexture Public Methods
    WindyTexture(TextureMapping3D mapping) : mapping(mapping) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        TexCoord3D c = mapping.Map(ctx);
        Float windStrength = FBm(.1f * c.p, .1f * c.dpdx, .1f * c.dpdy, .5, 3);
        Float waveHeight = FBm(c.p, c.dpdx, c.dpdy, .5, 6);
        return std::abs(windStrength) * waveHeight;
    }

    static WindyTexture *Create(const Transform &renderFromTexture,
                                const TextureParameterDictionary &parameters,
                                const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping3D mapping;
};

// WrinkledTexture Definition
class WrinkledTexture {
  public:
    // WrinkledTexture Public Methods
    WrinkledTexture(TextureMapping3D mapping, int octaves, Float omega)
        : mapping(mapping), octaves(octaves), omega(omega) {}

    PBRT_CPU_GPU
    Float Evaluate(TextureEvalContext ctx) const {
        TexCoord3D c = mapping.Map(ctx);
        return Turbulence(c.p, c.dpdx, c.dpdy, omega, octaves);
    }

    static WrinkledTexture *Create(const Transform &renderFromTexture,
                                   const TextureParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // WrinkledTexture Private Data
    TextureMapping3D mapping;
    int octaves;
    Float omega;
};

PBRT_CPU_GPU inline Float FloatTexture::Evaluate(TextureEvalContext ctx) const {
    auto eval = [&](auto ptr) { return ptr->Evaluate(ctx); };
    return Dispatch(eval);
}

PBRT_CPU_GPU inline SampledSpectrum SpectrumTexture::Evaluate(
    TextureEvalContext ctx, SampledWavelengths lambda) const {
    auto eval = [&](auto ptr) { return ptr->Evaluate(ctx, lambda); };
    return Dispatch(eval);
}

// UniversalTextureEvaluator Definition
class UniversalTextureEvaluator {
  public:
    // UniversalTextureEvaluator Public Methods
    PBRT_CPU_GPU
    bool CanEvaluate(std::initializer_list<FloatTexture>,
                     std::initializer_list<SpectrumTexture>) const {
        return true;
    }

    PBRT_CPU_GPU
    Float operator()(FloatTexture tex, TextureEvalContext ctx);

    PBRT_CPU_GPU
    SampledSpectrum operator()(SpectrumTexture tex, TextureEvalContext ctx,
                               SampledWavelengths lambda);
};

// BasicTextureEvaluator Definition
class BasicTextureEvaluator {
  public:
    // BasicTextureEvaluator Public Methods
    PBRT_CPU_GPU
    bool CanEvaluate(std::initializer_list<FloatTexture> ftex,
                     std::initializer_list<SpectrumTexture> stex) const {
        // Return _false_ if any _FloatTexture_s cannot be evaluated
        for (FloatTexture f : ftex)
            if (f && !f.Is<FloatConstantTexture>() && !f.Is<FloatImageTexture>() &&
                !f.Is<GPUFloatPtexTexture>() && !f.Is<GPUFloatImageTexture>())
                return false;

        // Return _false_ if any _SpectrumTexture_s cannot be evaluated
        for (SpectrumTexture s : stex)
            if (s && !s.Is<SpectrumConstantTexture>() && !s.Is<SpectrumImageTexture>() &&
                !s.Is<GPUSpectrumPtexTexture>() && !s.Is<GPUSpectrumImageTexture>())
                return false;

        return true;
    }

    PBRT_CPU_GPU
    Float operator()(FloatTexture tex, TextureEvalContext ctx) {
        if (tex.Is<FloatConstantTexture>())
            return tex.Cast<FloatConstantTexture>()->Evaluate(ctx);
        else if (tex.Is<FloatImageTexture>())
            return tex.Cast<FloatImageTexture>()->Evaluate(ctx);
        else if (tex.Is<GPUFloatImageTexture>())
            return tex.Cast<GPUFloatImageTexture>()->Evaluate(ctx);
        else if (tex.Is<GPUFloatPtexTexture>())
            return tex.Cast<GPUFloatPtexTexture>()->Evaluate(ctx);
        else {
            if (tex)
                LOG_FATAL("BasicTextureEvaluator::operator() called with %s", tex);
            return 0.f;
        }
    }

    PBRT_CPU_GPU
    SampledSpectrum operator()(SpectrumTexture tex, TextureEvalContext ctx,
                               SampledWavelengths lambda) {
        if (tex.Is<SpectrumConstantTexture>())
            return tex.Cast<SpectrumConstantTexture>()->Evaluate(ctx, lambda);
        else if (tex.Is<SpectrumImageTexture>())
            return tex.Cast<SpectrumImageTexture>()->Evaluate(ctx, lambda);
        else if (tex.Is<GPUSpectrumImageTexture>())
            return tex.Cast<GPUSpectrumImageTexture>()->Evaluate(ctx, lambda);
        else if (tex.Is<GPUSpectrumPtexTexture>())
            return tex.Cast<GPUSpectrumPtexTexture>()->Evaluate(ctx, lambda);
        else {
            if (tex)
                LOG_FATAL("BasicTextureEvaluator::operator() called with %s", tex);
            return SampledSpectrum(0.f);
        }
    }
};

}  // namespace pbrt

#endif  // PBRT_TEXTURES_H
