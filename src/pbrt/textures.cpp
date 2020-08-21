// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/textures.h>

#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/stats.h>

#include <mutex>

#include <Ptexture.h>

namespace pbrt {

TextureMapping2DHandle TextureMapping2DHandle::Create(
    const ParameterDictionary &parameters, const Transform &renderFromTexture,
    const FileLoc *loc, Allocator alloc) {
    std::string type = parameters.GetOneString("mapping", "uv");
    if (type == "uv") {
        Float su = parameters.GetOneFloat("uscale", 1.);
        Float sv = parameters.GetOneFloat("vscale", 1.);
        Float du = parameters.GetOneFloat("udelta", 0.);
        Float dv = parameters.GetOneFloat("vdelta", 0.);
        return alloc.new_object<UVMapping2D>(su, sv, du, dv);
    } else if (type == "spherical")
        return alloc.new_object<SphericalMapping2D>(Inverse(renderFromTexture));
    else if (type == "cylindrical")
        return alloc.new_object<CylindricalMapping2D>(Inverse(renderFromTexture));
    else if (type == "planar")
        return alloc.new_object<PlanarMapping2D>(
            parameters.GetOneVector3f("v1", Vector3f(1, 0, 0)),
            parameters.GetOneVector3f("v2", Vector3f(0, 1, 0)),
            parameters.GetOneFloat("udelta", 0.f), parameters.GetOneFloat("vdelta", 0.f));
    else {
        Error(loc, "2D texture mapping \"%s\" unknown", type);
        return alloc.new_object<UVMapping2D>();
    }
}

TextureMapping3DHandle TextureMapping3DHandle::Create(
    const ParameterDictionary &parameters, const Transform &renderFromTexture,
    const FileLoc *loc, Allocator alloc) {
    return alloc.new_object<TransformMapping3D>(Inverse(renderFromTexture));
}

std::string FloatTextureHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto toStr = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(toStr);
}

std::string SpectrumTextureHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto toStr = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(toStr);
}

// Texture Forward Declarations
PBRT_CPU_GPU
inline Float Grad(int x, int y, int z, Float dx, Float dy, Float dz);
PBRT_CPU_GPU
inline Float NoiseWeight(Float t);

// Perlin Noise Data
static constexpr int NoisePermSize = 256;
static PBRT_CONST int NoisePerm[2 * NoisePermSize] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103,
    30, 69, 142,
    // Remainder of the noise permutation table
    8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252,
    219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
    168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229,
    122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
    63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116,
    188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5,
    202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189,
    28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155,
    167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112,
    104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81,
    51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72,
    243, 141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131, 13,
    201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10,
    23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11,
    32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165,
    71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73,
    209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100,
    109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126,
    255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
    213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22,
    39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
    251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14,
    239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45,
    127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78,
    66, 215, 61, 156, 180

};

std::string UVMapping2D::ToString() const {
    return StringPrintf("[ UVMapping2D su: %f sv: %f du: %f dv: %f ]", su, sv, du, dv);
}

std::string SphericalMapping2D::ToString() const {
    return StringPrintf("[ SphericalMapping2D textureFromRender: %s ]",
                        textureFromRender);
}

std::string CylindricalMapping2D::ToString() const {
    return StringPrintf("[ CylindricalMapping2D textureFromRender: %s ]",
                        textureFromRender);
}

std::string PlanarMapping2D::ToString() const {
    return StringPrintf("[ PlanarMapping2D vs: %s vt: %s ds: %f dt: %f]", vs, vt, ds, dt);
}

std::string TransformMapping3D::ToString() const {
    return StringPrintf("[ TransformMapping3D textureFromRender: %s ]",
                        textureFromRender);
}

// Noise Function Definitions
Float Noise(Float x, Float y, Float z) {
    // Compute noise cell coordinates and offsets
    int ix = std::floor(x), iy = std::floor(y), iz = std::floor(z);
    Float dx = x - ix, dy = y - iy, dz = z - iz;

    // Compute gradient weights
    ix &= NoisePermSize - 1;
    iy &= NoisePermSize - 1;
    iz &= NoisePermSize - 1;
    Float w000 = Grad(ix, iy, iz, dx, dy, dz);
    Float w100 = Grad(ix + 1, iy, iz, dx - 1, dy, dz);
    Float w010 = Grad(ix, iy + 1, iz, dx, dy - 1, dz);
    Float w110 = Grad(ix + 1, iy + 1, iz, dx - 1, dy - 1, dz);
    Float w001 = Grad(ix, iy, iz + 1, dx, dy, dz - 1);
    Float w101 = Grad(ix + 1, iy, iz + 1, dx - 1, dy, dz - 1);
    Float w011 = Grad(ix, iy + 1, iz + 1, dx, dy - 1, dz - 1);
    Float w111 = Grad(ix + 1, iy + 1, iz + 1, dx - 1, dy - 1, dz - 1);

    // Compute trilinear interpolation of weights
    Float wx = NoiseWeight(dx), wy = NoiseWeight(dy), wz = NoiseWeight(dz);
    Float x00 = Lerp(wx, w000, w100);
    Float x10 = Lerp(wx, w010, w110);
    Float x01 = Lerp(wx, w001, w101);
    Float x11 = Lerp(wx, w011, w111);
    Float y0 = Lerp(wy, x00, x10);
    Float y1 = Lerp(wy, x01, x11);
    return Lerp(wz, y0, y1);
}

Float Noise(const Point3f &p) {
    return Noise(p.x, p.y, p.z);
}

inline Float Grad(int x, int y, int z, Float dx, Float dy, Float dz) {
    int h = NoisePerm[NoisePerm[NoisePerm[x] + y] + z];
    h &= 15;
    Float u = h < 8 || h == 12 || h == 13 ? dx : dy;
    Float v = h < 4 || h == 12 || h == 13 ? dy : dz;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

inline Float NoiseWeight(Float t) {
    Float t3 = t * t * t;
    Float t4 = t3 * t;
    return 6 * t4 * t - 15 * t4 + 10 * t3;
}

Float FBm(const Point3f &p, const Vector3f &dpdx, const Vector3f &dpdy, Float omega,
          int maxOctaves) {
    // Compute number of octaves for antialiased FBm
    Float len2 = std::max(LengthSquared(dpdx), LengthSquared(dpdy));
    Float n = Clamp(-1 - .5f * Log2(len2), 0, maxOctaves);
    int nInt = std::floor(n);

    // Compute sum of octaves of noise for FBm
    Float sum = 0, lambda = 1, o = 1;
    for (int i = 0; i < nInt; ++i) {
        sum += o * Noise(lambda * p);
        lambda *= 1.99f;
        o *= omega;
    }
    Float nPartial = n - nInt;
    sum += o * SmoothStep(nPartial, .3f, .7f) * Noise(lambda * p);

    return sum;
}

Float Turbulence(const Point3f &p, const Vector3f &dpdx, const Vector3f &dpdy,
                 Float omega, int maxOctaves) {
    // Compute number of octaves for antialiased FBm
    Float len2 = std::max(LengthSquared(dpdx), LengthSquared(dpdy));
    Float n = Clamp(-1 - .5f * Log2(len2), 0, maxOctaves);
    int nInt = std::floor(n);

    // Compute sum of octaves of noise for turbulence
    Float sum = 0, lambda = 1, o = 1;
    for (int i = 0; i < nInt; ++i) {
        sum += o * std::abs(Noise(lambda * p));
        lambda *= 1.99f;
        o *= omega;
    }

    // Account for contributions of clamped octaves in turbulence
    Float nPartial = n - nInt;
    sum += o * Lerp(SmoothStep(nPartial, .3f, .7f), 0.2, std::abs(Noise(lambda * p)));
    for (int i = nInt; i < maxOctaves; ++i) {
        sum += o * 0.2f;
        o *= omega;
    }

    return sum;
}

// ConstantTexture Method Definitions
std::string FloatConstantTexture::ToString() const {
    return StringPrintf("[ FloatConstantTexture value: %f ]", value);
}

FloatConstantTexture *FloatConstantTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    return alloc.new_object<FloatConstantTexture>(parameters.GetOneFloat("value", 1.f));
}

std::string RGBConstantTexture::ToString() const {
    return StringPrintf("[ RGBConstantTexture value: %s ]", value);
}

std::string RGBReflectanceConstantTexture::ToString() const {
    return StringPrintf("[ RGBReflectanceConstantTexture value: %s ]", value);
}

std::string SpectrumConstantTexture::ToString() const {
    return StringPrintf("[ SpectrumConstantTexture value: %s ]", value);
}

SpectrumConstantTexture *SpectrumConstantTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    SpectrumHandle one = alloc.new_object<ConstantSpectrum>(1.);
    SpectrumHandle c =
        parameters.GetOneSpectrum("value", one, SpectrumType::Reflectance, alloc);
    return alloc.new_object<SpectrumConstantTexture>(c);
}

// BilerpTexture Method Definitions
FloatBilerpTexture *FloatBilerpTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    // Initialize 2D texture mapping _map_ from _tp_
    TextureMapping2DHandle map =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

    return alloc.new_object<FloatBilerpTexture>(
        map, parameters.GetOneFloat("v00", 0.f), parameters.GetOneFloat("v01", 1.f),
        parameters.GetOneFloat("v10", 0.f), parameters.GetOneFloat("v11", 1.f));
}

std::string FloatBilerpTexture::ToString() const {
    return StringPrintf(
        "[ FloatBilerpTexture mapping: %s v00: %f v01: %f v10: %f v11: %f ]", mapping,
        v00, v01, v10, v11);
}

SpectrumBilerpTexture *SpectrumBilerpTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    // Initialize 2D texture mapping _map_ from _tp_
    TextureMapping2DHandle map =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

    SpectrumHandle zero = alloc.new_object<ConstantSpectrum>(0.);
    SpectrumHandle one = alloc.new_object<ConstantSpectrum>(1.);

    return alloc.new_object<SpectrumBilerpTexture>(
        map, parameters.GetOneSpectrum("v00", zero, SpectrumType::Reflectance, alloc),
        parameters.GetOneSpectrum("v01", one, SpectrumType::Reflectance, alloc),
        parameters.GetOneSpectrum("v10", zero, SpectrumType::Reflectance, alloc),
        parameters.GetOneSpectrum("v11", one, SpectrumType::Reflectance, alloc));
}

std::string SpectrumBilerpTexture::ToString() const {
    return StringPrintf(
        "[ SpectrumBilerpTexture mapping: %s v00: %s v01: %s v10: %s v11: %s ]", mapping,
        v00, v01, v10, v11);
}

// CheckerboardTexture Function Definitions
pstd::array<Float, 2> Checkerboard(AAMethod aaMethod, TextureEvalContext ctx,
                                   const TextureMapping2DHandle map2D,
                                   const TextureMapping3DHandle map3D) {
    if (map2D) {
        CHECK(!map3D);
        Vector2f dstdx, dstdy;
        Point2f st = map2D.Map(ctx, &dstdx, &dstdy);
        if (aaMethod == AAMethod::None) {
            // Point sample _Checkerboard2DTexture_
            if (((int)std::floor(st[0]) + (int)std::floor(st[1])) % 2 == 0)
                return {Float(1), Float(0)};
            return {Float(0), Float(1)};
        } else {
            // Compute closed-form box-filtered _Checkerboard2DTexture_ value

            // Evaluate single check if filter is entirely inside one of them
            Float ds = std::max(std::abs(dstdx[0]), std::abs(dstdy[0]));
            Float dt = std::max(std::abs(dstdx[1]), std::abs(dstdy[1]));
            Float s0 = st[0] - ds, s1 = st[0] + ds;
            Float t0 = st[1] - dt, t1 = st[1] + dt;
            if (std::floor(s0) == std::floor(s1) && std::floor(t0) == std::floor(t1)) {
                // Point sample _Checkerboard2DTexture_
                if (((int)std::floor(st[0]) + (int)std::floor(st[1])) % 2 == 0)
                    return {Float(1), Float(0)};
                return {Float(0), Float(1)};
            }

            // Apply box filter to checkerboard region
            auto bumpInt = [](Float x) {
                return (int)std::floor(x / 2) +
                       2 * std::max(x / 2 - (int)std::floor(x / 2) - (Float)0.5,
                                    (Float)0);
            };
            Float sint = (bumpInt(s1) - bumpInt(s0)) / (2 * ds);
            Float tint = (bumpInt(t1) - bumpInt(t0)) / (2 * dt);
            Float area2 = sint + tint - 2 * sint * tint;
            if (ds > 1 || dt > 1)
                area2 = .5f;
            return {(1 - area2), area2};
        }
    } else {
        CHECK(map3D);
        Vector3f dpdx, dpdy;
        Point3f p = map3D.Map(ctx, &dpdx, &dpdy);
        if (((int)std::floor(p.x) + (int)std::floor(p.y) + (int)std::floor(p.z)) % 2 == 0)
            return {Float(1), Float(0)};
        return {Float(0), Float(1)};
    }
}

FloatCheckerboardTexture *FloatCheckerboardTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    int dim = parameters.GetOneInt("dimension", 2);
    if (dim != 2 && dim != 3) {
        Error(loc, "%d dimensional checkerboard texture not supported", dim);
        return nullptr;
    }
    FloatTextureHandle tex1 = parameters.GetFloatTexture("tex1", 1.f, alloc);
    FloatTextureHandle tex2 = parameters.GetFloatTexture("tex2", 0.f, alloc);
    if (dim == 2) {
        // Initialize 2D texture mapping _map_ from _tp_
        TextureMapping2DHandle map =
            TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

        // Compute _aaMethod_ for _CheckerboardTexture_
        std::string aa = parameters.GetOneString("aamode", "closedform");
        AAMethod aaMethod;
        if (aa == "none")
            aaMethod = AAMethod::None;
        else if (aa == "closedform")
            aaMethod = AAMethod::ClosedForm;
        else {
            Warning(loc,
                    "Antialiasing mode \"%s\" not understood by "
                    "Checkerboard2DTexture; using \"closedform\"",
                    aa);
            aaMethod = AAMethod::ClosedForm;
        }
        return alloc.new_object<FloatCheckerboardTexture>(map, nullptr, tex1, tex2,
                                                          aaMethod);
    } else {
        // Initialize 3D texture mapping _map_ from _tp_
        TextureMapping3DHandle map =
            TextureMapping3DHandle::Create(parameters, renderFromTexture, loc, alloc);
        return alloc.new_object<FloatCheckerboardTexture>(nullptr, map, tex1, tex2,
                                                          AAMethod::None);
    }
}

std::string FloatCheckerboardTexture::ToString() const {
    return StringPrintf("[ FloatCheckerboardTexture map2D: %s map3D: %s "
                        "tex[0]: %s tex[1]: %s aaMethod: %s]",
                        map2D ? map2D.ToString().c_str() : "(nullptr)",
                        map3D ? map3D.ToString().c_str() : "(nullptr)", tex[0], tex[1],
                        aaMethod == AAMethod::None ? "none" : "closed-form");
}

SpectrumCheckerboardTexture *SpectrumCheckerboardTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    int dim = parameters.GetOneInt("dimension", 2);
    if (dim != 2 && dim != 3) {
        Error(loc, "%d dimensional checkerboard texture not supported", dim);
        return nullptr;
    }

    SpectrumHandle zero = alloc.new_object<ConstantSpectrum>(0.);
    SpectrumHandle one = alloc.new_object<ConstantSpectrum>(1.);

    SpectrumTextureHandle tex1 =
        parameters.GetSpectrumTexture("tex1", one, SpectrumType::Reflectance, alloc);
    SpectrumTextureHandle tex2 =
        parameters.GetSpectrumTexture("tex2", zero, SpectrumType::Reflectance, alloc);
    if (dim == 2) {
        // Initialize 2D texture mapping _map_ from _tp_
        TextureMapping2DHandle map =
            TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

        // Compute _aaMethod_ for _CheckerboardTexture_
        std::string aa = parameters.GetOneString("aamode", "closedform");
        AAMethod aaMethod;
        if (aa == "none")
            aaMethod = AAMethod::None;
        else if (aa == "closedform")
            aaMethod = AAMethod::ClosedForm;
        else {
            Warning(loc,
                    "Antialiasing mode \"%s\" not understood by "
                    "Checkerboard2DTexture; using \"closedform\"",
                    aa);
            aaMethod = AAMethod::ClosedForm;
        }
        return alloc.new_object<SpectrumCheckerboardTexture>(map, nullptr, tex1, tex2,
                                                             aaMethod);
    } else {
        // Initialize 3D texture mapping _map_ from _tp_
        TextureMapping3DHandle map =
            TextureMapping3DHandle::Create(parameters, renderFromTexture, loc, alloc);
        return alloc.new_object<SpectrumCheckerboardTexture>(nullptr, map, tex1, tex2,
                                                             AAMethod::None);
    }
}

std::string SpectrumCheckerboardTexture::ToString() const {
    return StringPrintf("[ SpectrumCheckerboardTexture map2D: %s map3D: %s "
                        "tex[0]: %s tex[1]: %s aaMethod: %s]",
                        map2D ? map2D.ToString().c_str() : "(nullptr)",
                        map3D ? map3D.ToString().c_str() : "(nullptr)", tex[0], tex[1],
                        aaMethod == AAMethod::None ? "none" : "closed-form");
}

// DotsTexture Method Definitions
FloatDotsTexture *FloatDotsTexture::Create(const Transform &renderFromTexture,
                                           const TextureParameterDictionary &parameters,
                                           const FileLoc *loc, Allocator alloc) {
    // Initialize 2D texture mapping _map_ from _tp_
    TextureMapping2DHandle map =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

    return alloc.new_object<FloatDotsTexture>(
        map, parameters.GetFloatTexture("inside", 1.f, alloc),
        parameters.GetFloatTexture("outside", 0.f, alloc));
}

std::string FloatDotsTexture::ToString() const {
    return StringPrintf("[ FloatDotsTexture mapping: %s insideDot: %s outsideDot: %s ]",
                        mapping, insideDot, outsideDot);
}

SpectrumDotsTexture *SpectrumDotsTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    // Initialize 2D texture mapping _map_ from _tp_
    TextureMapping2DHandle map =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);
    SpectrumHandle zero = alloc.new_object<ConstantSpectrum>(0.);
    SpectrumHandle one = alloc.new_object<ConstantSpectrum>(1.);

    return alloc.new_object<SpectrumDotsTexture>(
        map,
        parameters.GetSpectrumTexture("inside", one, SpectrumType::Reflectance, alloc),
        parameters.GetSpectrumTexture("outside", zero, SpectrumType::Reflectance, alloc));
}

std::string SpectrumDotsTexture::ToString() const {
    return StringPrintf(
        "[ SpectrumDotsTexture mapping: %s insideDot: %s outsideDot: %s ]", mapping,
        insideDot, outsideDot);
}

// FBmTexture Method Definitions
FBmTexture *FBmTexture::Create(const Transform &renderFromTexture,
                               const TextureParameterDictionary &parameters,
                               const FileLoc *loc, Allocator alloc) {
    // Initialize 3D texture mapping _map_ from _tp_
    TextureMapping3DHandle map =
        TextureMapping3DHandle::Create(parameters, renderFromTexture, loc, alloc);
    return alloc.new_object<FBmTexture>(map, parameters.GetOneInt("octaves", 8),
                                        parameters.GetOneFloat("roughness", .5f));
}

std::string FBmTexture::ToString() const {
    return StringPrintf("[ FBmTexture mapping: %s omega: %f octaves: %d ]", mapping,
                        omega, octaves);
}

// ImageTextureBase Method Definitions
ImageTextureBase::ImageTextureBase(TextureMapping2DHandle mapping,
                                   const std::string &filename, const std::string &filter,
                                   Float maxAniso, WrapMode wrapMode, Float scale,
                                   ColorEncodingHandle encoding, Allocator alloc)
    : mapping(std::move(mapping)), scale(scale) {
    mipmap = GetTexture(filename, filter, maxAniso, wrapMode, encoding, alloc);
}

MIPMap *ImageTextureBase::GetTexture(const std::string &filename,
                                     const std::string &filter, Float maxAniso,
                                     WrapMode wrap, ColorEncodingHandle encoding,
                                     Allocator alloc) {
    // Return _MIPMap_ from texture cache if present
    TexInfo texInfo(filename, filter, maxAniso, wrap, encoding);
    std::unique_lock<std::mutex> lock(textureCacheMutex);
    if (textureCache.find(texInfo) != textureCache.end())
        return textureCache[texInfo].get();
    else
        lock.unlock();

    // Create _MIPMap_ for _filename_
    MIPMapFilterOptions options;
    options.maxAnisotropy = maxAniso;

    pstd::optional<FilterFunction> ff = ParseFilter(filter);
    if (ff)
        options.filter = *ff;
    else
        Warning("%s: filter function unknown", filter);

    std::unique_ptr<MIPMap> mipmap =
        MIPMap::CreateFromFile(filename, options, wrap, encoding, alloc);
    if (mipmap) {
        lock.lock();
        // This is actually ok, but if it hits, it means we've wastefully
        // loaded this texture. (Note that in that case, should just return
        // the one that's already in there and not replace it.)
        CHECK(textureCache.find(texInfo) == textureCache.end());
        textureCache[texInfo] = std::move(mipmap);

        return textureCache[texInfo].get();
    } else
        return nullptr;
}

// SpectrumImageTexture Method Definitions
SampledSpectrum SpectrumImageTexture::Evaluate(TextureEvalContext ctx,
                                               SampledWavelengths lambda) const {
#ifdef PBRT_IS_GPU_CODE
    assert(!"Should not be called in GPU code");
    return SampledSpectrum(0);
#else
    if (!mipmap)
        return SampledSpectrum(scale);
    Vector2f dstdx, dstdy;
    Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
    // Texture coordinates are (0,0) in the lower left corner, but
    // image coordinates are (0,0) in the upper left.
    st[1] = 1 - st[1];
    RGB rgb = scale * mipmap->Lookup<RGB>(st, dstdx, dstdy);
    const RGBColorSpace *cs = mipmap->GetRGBColorSpace();
    if (cs != nullptr) {
        if (std::max({rgb.r, rgb.g, rgb.b}) > 1)
            return RGBSpectrum(*cs, rgb).Sample(lambda);
        return RGBReflectanceSpectrum(*cs, rgb).Sample(lambda);
    }
    // otherwise it better be a one-channel texture
    CHECK(rgb[0] == rgb[1] && rgb[1] == rgb[2]);
    return SampledSpectrum(rgb[0]);
#endif
}

std::string SpectrumImageTexture::ToString() const {
    return StringPrintf("[ SpectrumImageTexture mapping: %s scale: %f mipmap: %s ]",
                        mapping, scale, *mipmap);
}

std::string FloatImageTexture::ToString() const {
    return StringPrintf("[ FloatImageTexture mapping: %s scale: %f mipmap: %s ]", mapping,
                        scale, *mipmap);
}

std::string TexInfo::ToString() const {
    return StringPrintf("[ TexInfo filename: %s filter: %s maxAniso: %f "
                        "wrapMode: %s encoding: %s ]",
                        filename, filter, maxAniso, wrapMode, encoding);
}

std::mutex ImageTextureBase::textureCacheMutex;
std::map<TexInfo, std::unique_ptr<MIPMap>> ImageTextureBase::textureCache;

FloatImageTexture *FloatImageTexture::Create(const Transform &renderFromTexture,
                                             const TextureParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc) {
    // Initialize 2D texture mapping _map_ from _tp_
    TextureMapping2DHandle map =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

    // Initialize _ImageTexture_ parameters
    Float maxAniso = parameters.GetOneFloat("maxanisotropy", 8.f);
    std::string filter = parameters.GetOneString("filter", "bilinear");
    std::string wrapString = parameters.GetOneString("wrap", "repeat");
    pstd::optional<WrapMode> wrapMode = ParseWrapMode(wrapString.c_str());
    if (!wrapMode)
        ErrorExit("%s: wrap mode unknown", wrapString);
    Float scale = parameters.GetOneFloat("scale", 1.f);
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));

    const char *defaultEncoding = HasExtension(filename, "png") ? "sRGB" : "linear";
    std::string encodingString = parameters.GetOneString("encoding", defaultEncoding);
    ColorEncodingHandle encoding = ColorEncodingHandle::Get(encodingString);

    return alloc.new_object<FloatImageTexture>(map, filename, filter, maxAniso, *wrapMode,
                                               scale, encoding, alloc);
}

SpectrumImageTexture *SpectrumImageTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    // Initialize 2D texture mapping _map_ from _tp_
    TextureMapping2DHandle map =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

    // Initialize _ImageTexture_ parameters
    Float maxAniso = parameters.GetOneFloat("maxanisotropy", 8.f);
    std::string filter = parameters.GetOneString("filter", "bilinear");
    std::string wrapString = parameters.GetOneString("wrap", "repeat");
    pstd::optional<WrapMode> wrapMode = ParseWrapMode(wrapString.c_str());
    if (!wrapMode)
        ErrorExit("%s: wrap mode unknown", wrapString);
    Float scale = parameters.GetOneFloat("scale", 1.f);
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));

    const char *defaultEncoding = HasExtension(filename, "png") ? "sRGB" : "linear";
    std::string encodingString = parameters.GetOneString("encoding", defaultEncoding);
    ColorEncodingHandle encoding = ColorEncodingHandle::Get(encodingString);

    return alloc.new_object<SpectrumImageTexture>(map, filename, filter, maxAniso,
                                                  *wrapMode, scale, encoding, alloc);
}

// MarbleTexture Method Definitions
SampledSpectrum MarbleTexture::Evaluate(TextureEvalContext ctx,
                                        SampledWavelengths lambda) const {
    Vector3f dpdx, dpdy;
    Point3f p = mapping.Map(ctx, &dpdx, &dpdy);
    p *= scale;
    Float marble = p.y + variation * FBm(p, scale * dpdx, scale * dpdy, omega, octaves);
    Float t = .5f + .5f * std::sin(marble);
    // Evaluate marble spline at _t_
    const RGB c[] = {
        {.58f, .58f, .6f}, {.58f, .58f, .6f}, {.58f, .58f, .6f},
        {.5f, .5f, .5f},   {.6f, .59f, .58f}, {.58f, .58f, .6f},
        {.58f, .58f, .6f}, {.2f, .2f, .33f},  {.58f, .58f, .6f},
    };
    int nSeg = PBRT_ARRAYSIZE(c) - 3;
    int first = std::min<int>(std::floor(t * nSeg), nSeg - 1);
    t = (t * nSeg - first);
    // Bezier spline evaluated with de Castilejau's algorithm
    RGB s0 = Lerp(t, c[first], c[first + 1]);
    RGB s1 = Lerp(t, c[first + 1], c[first + 2]);
    RGB s2 = Lerp(t, c[first + 2], c[first + 3]);
    s0 = Lerp(t, s0, s1);
    s1 = Lerp(t, s1, s2);
    // Extra scale of 1.5 to increase variation among colors
    s0 = 1.5f * Lerp(t, s0, s1);
#ifdef PBRT_IS_GPU_CODE
    return RGBReflectanceSpectrum(*RGBColorSpace_sRGB, s0).Sample(lambda);
#else
    return RGBReflectanceSpectrum(*RGBColorSpace::sRGB, s0).Sample(lambda);
#endif
}

std::string MarbleTexture::ToString() const {
    return StringPrintf("[ MarbleTexture mapping: %s octaves: %d omega: %f "
                        "scale: %f variation: %f ]",
                        mapping, octaves, omega, scale, variation);
}

MarbleTexture *MarbleTexture::Create(const Transform &renderFromTexture,
                                     const TextureParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc) {
    // Initialize 3D texture mapping _map_ from _tp_
    TextureMapping3DHandle map =
        TextureMapping3DHandle::Create(parameters, renderFromTexture, loc, alloc);
    return alloc.new_object<MarbleTexture>(
        map, parameters.GetOneInt("octaves", 8), parameters.GetOneFloat("roughness", .5f),
        parameters.GetOneFloat("scale", 1.f), parameters.GetOneFloat("variation", .2f));
}

// MixTexture Method Definitions
std::string FloatMixTexture::ToString() const {
    return StringPrintf("[ FloatMixTexture tex1: %s tex2: %s amount: %s ]", tex1, tex2,
                        amount);
}

std::string SpectrumMixTexture::ToString() const {
    return StringPrintf("[ SpectrumMixTexture tex1: %s tex2: %s amount: %s ]", tex1, tex2,
                        amount);
}

FloatMixTexture *FloatMixTexture::Create(const Transform &renderFromTexture,
                                         const TextureParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc) {
    return alloc.new_object<FloatMixTexture>(
        parameters.GetFloatTexture("tex1", 0.f, alloc),
        parameters.GetFloatTexture("tex2", 1.f, alloc),
        parameters.GetFloatTexture("amount", 0.5f, alloc));
}

SpectrumMixTexture *SpectrumMixTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    SpectrumHandle zero = alloc.new_object<ConstantSpectrum>(0.);
    SpectrumHandle one = alloc.new_object<ConstantSpectrum>(1.);
    return alloc.new_object<SpectrumMixTexture>(
        parameters.GetSpectrumTexture("tex1", zero, SpectrumType::Reflectance, alloc),
        parameters.GetSpectrumTexture("tex2", one, SpectrumType::Reflectance, alloc),
        parameters.GetFloatTexture("amount", 0.5f, alloc));
}

static Ptex::PtexCache *cache;

STAT_COUNTER("Texture/Ptex lookups", nLookups);
STAT_COUNTER("Texture/Ptex files accessed", nFilesAccessed);
STAT_COUNTER("Texture/Ptex block reads", nBlockReads);
STAT_MEMORY_COUNTER("Memory/Ptex peak memory used", peakMemoryUsed);

struct : public PtexErrorHandler {
    void reportError(const char *error) override { Error("%s", error); }
} errorHandler;

// PtexTexture Method Definitions

PtexTextureBase::PtexTextureBase(const std::string &filename,
                                 ColorEncodingHandle encoding)
    : filename(filename), encoding(encoding) {
    if (cache == nullptr) {
        int maxFiles = 100;
        size_t maxMem = 1ull << 32;  // 4GB
        bool premultiply = true;

        cache = Ptex::PtexCache::create(maxFiles, maxMem, premultiply, nullptr,
                                        &errorHandler);
        // TODO? cache->setSearchPath(...);
    }

    // Issue an error if the texture doesn't exist or has an unsupported
    // number of channels.
    valid = false;
    Ptex::String error;
    Ptex::PtexTexture *texture = cache->get(filename.c_str(), error);
    if (texture == nullptr)
        Error("%s", error);
    else {
        if (texture->numChannels() != 1 && texture->numChannels() != 3)
            Error("%s: only one and three channel ptex textures are supported", filename);
        else {
            valid = true;
            LOG_VERBOSE("%s: added ptex texture", filename);
        }
        texture->release();
    }
}

void PtexTextureBase::ReportStats() {
    if (!cache)
        return;

    Ptex::PtexCache::Stats stats;
    cache->getStats(stats);

    nFilesAccessed += stats.filesAccessed;
    nBlockReads += stats.blockReads;
    peakMemoryUsed = std::max(peakMemoryUsed, int64_t(stats.peakMemUsed));
}

int PtexTextureBase::SampleTexture(TextureEvalContext ctx, float result[3]) const {
    if (!valid) {
        result[0] = 0.;
        return 1;
    }

    ++nLookups;
    Ptex::String error;
    Ptex::PtexTexture *texture = cache->get(filename.c_str(), error);
    CHECK(texture != nullptr);
    // TODO: make the filter an option?
    Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
    Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);
    int nc = texture->numChannels();

    int firstChan = 0;
    filter->eval(result, firstChan, nc, ctx.faceIndex, ctx.uv[0], ctx.uv[1], ctx.dudx,
                 ctx.dvdx, ctx.dudy, ctx.dvdy);
    filter->release();
    texture->release();

    if (encoding != ColorEncodingHandle::Linear) {
        // It feels a little dirty to convert to 8-bits to run through the
        // encoding, but it's probably fine.
        uint8_t result8[3];
        for (int i = 0; i < nc; ++i)
            result8[i] = uint8_t(Clamp(result[i] * 255.f + 0.5f, 0, 255));

        // Handle Float == double.
        Float fResult[3];
        encoding.ToLinear(pstd::MakeConstSpan(result8, nc), pstd::MakeSpan(fResult, nc));
        for (int c = 0; c < nc; ++c)
            result[c] = fResult[c];
    }

    return nc;
}

std::string PtexTextureBase::BaseToString() const {
    return StringPrintf("valid: %s filename: %s encoding: %s", valid, filename, encoding);
}

std::string FloatPtexTexture::ToString() const {
    return StringPrintf("[ FloatPtexTexture %s ]", BaseToString());
}

std::string SpectrumPtexTexture::ToString() const {
    return StringPrintf("[ SpectrumPtexTexture %s ]", BaseToString());
}

Float FloatPtexTexture::Evaluate(TextureEvalContext ctx) const {
#ifdef PBRT_IS_GPU_CODE
    LOG_FATAL("Ptex not supported with GPU renderer");
    return 0;
#else
    float result[3];
    int nc = SampleTexture(ctx, result);
    if (nc == 1)
        return result[0];
    DCHECK_EQ(3, nc);
    return (result[0] + result[1] + result[2]) / 3;
#endif
}

SampledSpectrum SpectrumPtexTexture::Evaluate(TextureEvalContext ctx,
                                              SampledWavelengths lambda) const {
#ifdef PBRT_IS_GPU_CODE
    LOG_FATAL("Ptex not supported with GPU renderer");
    return SampledSpectrum(0);
#else
    float result[3];
    int nc = SampleTexture(ctx, result);
    if (nc == 1)
        return SampledSpectrum(result[0]);
    DCHECK_EQ(3, nc);
    return RGBReflectanceSpectrum(*RGBColorSpace::sRGB,
                                  RGB(result[0], result[1], result[2]))
        .Sample(lambda);
#endif
}

FloatPtexTexture *FloatPtexTexture::Create(const Transform &renderFromTexture,
                                           const TextureParameterDictionary &parameters,
                                           const FileLoc *loc, Allocator alloc) {
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    std::string encodingString = parameters.GetOneString("encoding", "gamma 2.2");
    ColorEncodingHandle encoding = ColorEncodingHandle::Get(encodingString);
    return alloc.new_object<FloatPtexTexture>(filename, encoding);
}

SpectrumPtexTexture *SpectrumPtexTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    std::string encodingString = parameters.GetOneString("encoding", "gamma 2.2");
    ColorEncodingHandle encoding = ColorEncodingHandle::Get(encodingString);
    return alloc.new_object<SpectrumPtexTexture>(filename, encoding);
}

// ScaleTexture Method Definitions
std::string FloatScaledTexture::ToString() const {
    return StringPrintf("[ FloatScaledTexture tex: %s scale: %s ]", tex, scale);
}

std::string SpectrumScaledTexture::ToString() const {
    return StringPrintf("[ SpectrumScaledTexture tex: %s scale: %s ]", tex, scale);
}

FloatTextureHandle FloatScaledTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    FloatTextureHandle tex = parameters.GetFloatTexture("tex", 1.f, alloc);
    FloatTextureHandle scale = parameters.GetFloatTexture("scale", 1.f, alloc);

    for (int i = 0; i < 2; ++i) {
        if (FloatConstantTexture *cscale = scale.CastOrNullptr<FloatConstantTexture>()) {
            Float cs = cscale->Evaluate({});
            if (cs == 1) {
                LOG_VERBOSE("Dropping useless scale by 1");
                return tex;
            } else if (FloatImageTexture *image =
                           tex.CastOrNullptr<FloatImageTexture>()) {
                FloatImageTexture *imageCopy =
                    alloc.new_object<FloatImageTexture>(*image);
                LOG_VERBOSE("Flattened scale %f * image texture", cs);
                imageCopy->scale *= cs;
                return imageCopy;
            }
#if defined(PBRT_BUILD_GPU_RENDERER)
            else if (GPUFloatImageTexture *gimage =
                         tex.CastOrNullptr<GPUFloatImageTexture>()) {
                GPUFloatImageTexture *gimageCopy =
                    alloc.new_object<GPUFloatImageTexture>(*gimage);
                LOG_VERBOSE("Flattened scale %f * gpu image texture", cs);
                gimageCopy->scale *= cs;
                return gimageCopy;
            }
#endif
        }
        std::swap(tex, scale);
    }

    std::swap(tex, scale);
    return alloc.new_object<FloatScaledTexture>(tex, scale);
}

SpectrumTextureHandle SpectrumScaledTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    SpectrumHandle one = alloc.new_object<ConstantSpectrum>(1.);
    SpectrumTextureHandle tex =
        parameters.GetSpectrumTexture("tex", one, SpectrumType::Reflectance, alloc);
    FloatTextureHandle scale = parameters.GetFloatTexture("scale", 1.f, alloc);

    if (FloatConstantTexture *cscale = scale.CastOrNullptr<FloatConstantTexture>()) {
        Float cs = cscale->Evaluate({});
        if (cs == 1) {
            LOG_VERBOSE("Dropping useless scale by 1");
            return tex;
        } else if (SpectrumImageTexture *image =
                       tex.CastOrNullptr<SpectrumImageTexture>()) {
            SpectrumImageTexture *imageCopy =
                alloc.new_object<SpectrumImageTexture>(*image);
            LOG_VERBOSE("Flattened scale %f * image texture", cs);
            imageCopy->scale *= cs;
            return imageCopy;
        }
#if defined(PBRT_BUILD_GPU_RENDERER)
        else if (GPUSpectrumImageTexture *gimage =
                     tex.CastOrNullptr<GPUSpectrumImageTexture>()) {
            GPUSpectrumImageTexture *gimageCopy =
                alloc.new_object<GPUSpectrumImageTexture>(*gimage);
            LOG_VERBOSE("Flattened scale %f * gpu image texture", cs);
            gimageCopy->scale *= cs;
            return gimageCopy;
        }
#endif
    }

    return alloc.new_object<SpectrumScaledTexture>(tex, scale);
}

// WindyTexture Method Definitions
std::string WindyTexture::ToString() const {
    return StringPrintf("[ WindyTexture mapping: %s ]", mapping);
}

WindyTexture *WindyTexture::Create(const Transform &renderFromTexture,
                                   const TextureParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc) {
    // Initialize 3D texture mapping _map_ from _tp_
    TextureMapping3DHandle map =
        TextureMapping3DHandle::Create(parameters, renderFromTexture, loc, alloc);
    return alloc.new_object<WindyTexture>(map);
}

// WrinkledTexture Method Definitions
std::string WrinkledTexture::ToString() const {
    return StringPrintf("[ WrinkledTexture mapping: %s octaves: %d "
                        "omega: %f ]",
                        mapping, octaves, omega);
}

WrinkledTexture *WrinkledTexture::Create(const Transform &renderFromTexture,
                                         const TextureParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc) {
    // Initialize 3D texture mapping _map_ from _tp_
    TextureMapping3DHandle map =
        TextureMapping3DHandle::Create(parameters, renderFromTexture, loc, alloc);
    return alloc.new_object<WrinkledTexture>(map, parameters.GetOneInt("octaves", 8),
                                             parameters.GetOneFloat("roughness", .5f));
}

#if defined(PBRT_BUILD_GPU_RENDERER)

struct LuminanceTextureCacheItem {
    cudaArray_t array;
    cudaTextureReadMode readMode;
    bool originallySingleChannel;
};
struct RGBTextureCacheItem {
    cudaArray_t array;
    cudaTextureReadMode readMode;
    const RGBColorSpace *colorSpace;
};

static std::mutex textureCacheMutex;
static std::map<std::string, LuminanceTextureCacheItem> lumTextureCache;
static std::map<std::string, RGBTextureCacheItem> rgbTextureCache;

STAT_MEMORY_COUNTER("Memory/ImageTextures", gpuImageTextureBytes);

static cudaArray_t createSingleChannelTextureArray(const Image &image) {
    CHECK_EQ(1, image.NChannels());
    cudaArray_t texArray;

    cudaChannelFormatDesc channelDesc;
    int pitch;
    switch (image.Format()) {
    case PixelFormat::U256:
        channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        pitch = image.Resolution().x * sizeof(uint8_t);
        break;
    case PixelFormat::Half:
        channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
        pitch = image.Resolution().x * sizeof(Half);
        break;
    case PixelFormat::Float:
        channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        pitch = image.Resolution().x * sizeof(float);
        break;
    }
    gpuImageTextureBytes += pitch * image.Resolution().y;

    CUDA_CHECK(cudaMallocArray(&texArray, &channelDesc, image.Resolution().x,
                               image.Resolution().y));

    CUDA_CHECK(cudaMemcpy2DToArray(texArray, /* offset */ 0, 0, image.RawPointer({0, 0}),
                                   pitch, pitch, image.Resolution().y,
                                   cudaMemcpyHostToDevice));

    return texArray;
}

static cudaTextureAddressMode convertAddressMode(const std::string &mode) {
    if (mode == "repeat")
        return cudaAddressModeWrap;
    else if (mode == "clamp")
        return cudaAddressModeClamp;
    else if (mode == "black")
        return cudaAddressModeBorder;
    else
        ErrorExit("%s: texture wrap mode not supported", mode);
}

GPUSpectrumImageTexture *GPUSpectrumImageTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    /*
      Float maxAniso = parameters.GetOneFloat("maxanisotropy", 8.f);
      std::string filter = parameters.GetOneString("filter", "bilinear");
      const char *defaultEncoding = HasExtension(filename, "png") ? "sRGB" :
      "linear"; std::string encodingString = parameters.GetOneString("encoding",
      defaultEncoding); const ColorEncoding *encoding =
      ColorEncodingHandle::Get(encodingString);
    */

    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));

    // These have to be initialized one way or another in the below
    cudaArray_t texArray;
    cudaTextureReadMode readMode;
    const RGBColorSpace *colorSpace = nullptr;
    bool isSingleChannel = false;

    textureCacheMutex.lock();
    auto rgbIter = rgbTextureCache.find(filename);
    if (rgbIter != rgbTextureCache.end()) {
        LOG_VERBOSE("Found %s in RGB tex array cache!", filename);
        texArray = rgbIter->second.array;
        readMode = rgbIter->second.readMode;
        colorSpace = rgbIter->second.colorSpace;
        textureCacheMutex.unlock();
    } else {
        auto lumIter = lumTextureCache.find(filename);
        // We don't want to take it if it was originally an RGB texture and
        // GPUFloatImageTexture converted it to single channel
        if (lumIter != lumTextureCache.end() && lumIter->second.originallySingleChannel) {
            LOG_VERBOSE("Found %s in luminance tex array cache!", filename);
            texArray = lumIter->second.array;
            readMode = lumIter->second.readMode;
            colorSpace = RGBColorSpace::sRGB;
            textureCacheMutex.unlock();
            isSingleChannel = true;
        } else {
            textureCacheMutex.unlock();

            {
                ImageAndMetadata immeta = Image::Read(filename);
                Image &image = immeta.image;

                readMode = image.Format() == PixelFormat::U256
                               ? cudaReadModeNormalizedFloat
                               : cudaReadModeElementType;
                colorSpace = immeta.metadata.GetColorSpace();

                ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
                if (rgbDesc) {
                    image = image.SelectChannels(rgbDesc);

                    switch (image.Format()) {
                    case PixelFormat::U256: {
                        std::vector<uint8_t> rgba(4 * image.Resolution().x *
                                                  image.Resolution().y);
                        size_t offset = 0;
                        for (int y = 0; y < image.Resolution().y; ++y)
                            for (int x = 0; x < image.Resolution().x; ++x) {
                                for (int c = 0; c < 3; ++c)
                                    rgba[offset++] =
                                        ((uint8_t *)image.RawPointer({x, y}))[c];
                                rgba[offset++] = 255;
                            }

                        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
                            8, 8, 8, 8, cudaChannelFormatKindUnsigned);

                        CUDA_CHECK(cudaMallocArray(&texArray, &channelDesc,
                                                   image.Resolution().x,
                                                   image.Resolution().y));

                        int pitch = image.Resolution().x * 4 * sizeof(uint8_t);
                        gpuImageTextureBytes += pitch * image.Resolution().y;

                        CUDA_CHECK(cudaMemcpy2DToArray(texArray,
                                                       /* offset */ 0, 0, rgba.data(),
                                                       pitch, pitch, image.Resolution().y,
                                                       cudaMemcpyHostToDevice));
                        break;
                    }
                    case PixelFormat::Half: {
                        std::vector<Half> rgba(4 * image.Resolution().x *
                                               image.Resolution().y);

                        size_t offset = 0;
                        for (int y = 0; y < image.Resolution().y; ++y)
                            for (int x = 0; x < image.Resolution().x; ++x) {
                                for (int c = 0; c < 3; ++c)
                                    rgba[offset++] = Half(image.GetChannel({x, y}, c));
                                rgba[offset++] = Half(1.f);
                            }

                        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
                            16, 16, 16, 16, cudaChannelFormatKindFloat);

                        CUDA_CHECK(cudaMallocArray(&texArray, &channelDesc,
                                                   image.Resolution().x,
                                                   image.Resolution().y));

                        int pitch = image.Resolution().x * 4 * sizeof(Half);
                        gpuImageTextureBytes += pitch * image.Resolution().y;

                        CUDA_CHECK(cudaMemcpy2DToArray(texArray,
                                                       /* offset */ 0, 0, rgba.data(),
                                                       pitch, pitch, image.Resolution().y,
                                                       cudaMemcpyHostToDevice));
                        break;
                    }
                    case PixelFormat::Float: {
                        std::vector<float> rgba(4 * image.Resolution().x *
                                                image.Resolution().y);

                        size_t offset = 0;
                        for (int y = 0; y < image.Resolution().y; ++y)
                            for (int x = 0; x < image.Resolution().x; ++x) {
                                for (int c = 0; c < 3; ++c)
                                    rgba[offset++] = image.GetChannel({x, y}, c);
                                rgba[offset++] = 1.f;
                            }

                        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
                            32, 32, 32, 32, cudaChannelFormatKindFloat);

                        CUDA_CHECK(cudaMallocArray(&texArray, &channelDesc,
                                                   image.Resolution().x,
                                                   image.Resolution().y));

                        int pitch = image.Resolution().x * 4 * sizeof(float);
                        gpuImageTextureBytes += pitch * image.Resolution().y;

                        CUDA_CHECK(cudaMemcpy2DToArray(texArray,
                                                       /* offset */ 0, 0, rgba.data(),
                                                       pitch, pitch, image.Resolution().y,
                                                       cudaMemcpyHostToDevice));
                        break;
                    }
                    default:
                        LOG_FATAL("Unexpected PixelFormat");
                    }

                    textureCacheMutex.lock();
                    rgbTextureCache[filename] =
                        RGBTextureCacheItem{texArray, readMode, colorSpace};
                    textureCacheMutex.unlock();
                } else if (image.NChannels() == 1) {
                    texArray = createSingleChannelTextureArray(image);

                    textureCacheMutex.lock();
                    lumTextureCache[filename] =
                        LuminanceTextureCacheItem{texArray, readMode, true};
                    textureCacheMutex.unlock();
                    isSingleChannel = true;
                } else {
                    Warning(loc, "%s: unable to decypher image format", filename);
                    return nullptr;
                }
            }  // profile scope
        }
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texArray;

    std::string wrap = parameters.GetOneString("wrap", "repeat");
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = convertAddressMode(wrap);
    texDesc.addressMode[1] = convertAddressMode(wrap);
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = 1;
    texDesc.maxAnisotropy = 1;
    texDesc.maxMipmapLevelClamp = 99;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.borderColor[0] = texDesc.borderColor[1] = texDesc.borderColor[2] =
        texDesc.borderColor[3] = 0.f;
    texDesc.sRGB = 1;

    cudaTextureObject_t texObj;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    TextureMapping2DHandle mapping =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

    Float scale = parameters.GetOneFloat("scale", 1.f);

    return alloc.new_object<GPUSpectrumImageTexture>(mapping, texObj, scale,
                                                     isSingleChannel, colorSpace);
}

GPUFloatImageTexture *GPUFloatImageTexture::Create(
    const Transform &renderFromTexture, const TextureParameterDictionary &parameters,
    const FileLoc *loc, Allocator alloc) {
    /*
      Float maxAniso = parameters.GetOneFloat("maxanisotropy", 8.f);
      std::string filter = parameters.GetOneString("filter", "bilinear");
      const char *defaultEncoding = HasExtension(filename, "png") ? "sRGB" :
      "linear"; std::string encodingString = parameters.GetOneString("encoding",
      defaultEncoding); const ColorEncoding *encoding =
      ColorEncodingHandle::Get(encodingString);
    */
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));

    cudaArray_t texArray;
    cudaTextureReadMode readMode;

    textureCacheMutex.lock();
    auto iter = lumTextureCache.find(filename);
    if (iter != lumTextureCache.end()) {
        LOG_VERBOSE("Found %s in luminance tex array cache!", filename);
        texArray = iter->second.array;
        readMode = iter->second.readMode;
        textureCacheMutex.unlock();
    } else {
        textureCacheMutex.unlock();

        ImageAndMetadata immeta = Image::Read(filename);
        Image &image = immeta.image;
        ImageChannelDesc alphaDesc = image.GetChannelDesc({"A"});
        bool convertedImage = false;
        if (alphaDesc) {
            image = image.SelectChannels(alphaDesc);
            convertedImage = true;
        } else {
            ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
            if (rgbDesc) {
                // Convert to one channel
                Image avgImage(image.Format(), image.Resolution(), {"Y"},
                               image.Encoding());

                for (int y = 0; y < image.Resolution().y; ++y)
                    for (int x = 0; x < image.Resolution().x; ++x)
                        avgImage.SetChannel({x, y}, 0,
                                            image.GetChannels({x, y}, rgbDesc).Average());

                image = std::move(avgImage);
                convertedImage = true;
            }
        }

        texArray = createSingleChannelTextureArray(image);
        readMode = (image.Format() == PixelFormat::U256) ? cudaReadModeNormalizedFloat
                                                         : cudaReadModeElementType;

        textureCacheMutex.lock();
        lumTextureCache[filename] =
            LuminanceTextureCacheItem{texArray, readMode, !convertedImage};
        textureCacheMutex.unlock();
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texArray;

    std::string wrap = parameters.GetOneString("wrap", "repeat");
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = convertAddressMode(wrap);
    texDesc.addressMode[1] = convertAddressMode(wrap);
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = 1;
    texDesc.maxAnisotropy = 1;
    texDesc.maxMipmapLevelClamp = 99;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.borderColor[0] = texDesc.borderColor[1] = texDesc.borderColor[2] =
        texDesc.borderColor[3] = 0.f;
    texDesc.sRGB = 1;

    cudaTextureObject_t texObj;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    TextureMapping2DHandle mapping =
        TextureMapping2DHandle::Create(parameters, renderFromTexture, loc, alloc);

    Float scale = parameters.GetOneFloat("scale", 1.f);

    return alloc.new_object<GPUFloatImageTexture>(mapping, texObj, scale);
}

#endif  // PBRT_BUILD_GPU_RENDERER

STAT_COUNTER("Scene/Textures", nTextures);

FloatTextureHandle FloatTextureHandle::Create(
    const std::string &name, const Transform &renderFromTexture,
    const TextureParameterDictionary &parameters, const FileLoc *loc, Allocator alloc,
    bool gpu) {
    FloatTextureHandle tex;
    if (name == "constant")
        tex = FloatConstantTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "scale")
        tex = FloatScaledTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "mix")
        tex = FloatMixTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "bilerp")
        tex = FloatBilerpTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "imagemap") {
        if (gpu)
            tex = GPUFloatImageTexture::Create(renderFromTexture, parameters, loc, alloc);
        else
            tex = FloatImageTexture::Create(renderFromTexture, parameters, loc, alloc);
    } else if (name == "checkerboard")
        tex = FloatCheckerboardTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "dots")
        tex = FloatDotsTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "fbm")
        tex = FBmTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "wrinkled")
        tex = WrinkledTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "windy")
        tex = WindyTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "ptex") {
        if (gpu)
            ErrorExit(loc, "ptex texture is not supported on the GPU.");
        else
            tex = FloatPtexTexture::Create(renderFromTexture, parameters, loc, alloc);
    } else
        ErrorExit(loc, "%s: float texture type unknown.", name);

    if (!tex)
        ErrorExit(loc, "%s: unable to create texture.", name);

    ++nTextures;

    // FIXME: reenable this once we handle all the same image texture parameters
    // CO    parameters.ReportUnused();
    return tex;
}

SpectrumTextureHandle SpectrumTextureHandle::Create(
    const std::string &name, const Transform &renderFromTexture,
    const TextureParameterDictionary &parameters, const FileLoc *loc, Allocator alloc,
    bool gpu) {
    SpectrumTextureHandle tex;
    if (name == "constant")
        tex = SpectrumConstantTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "scale")
        tex = SpectrumScaledTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "mix")
        tex = SpectrumMixTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "bilerp")
        tex = SpectrumBilerpTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "imagemap") {
        if (gpu)
            tex = GPUSpectrumImageTexture::Create(renderFromTexture, parameters, loc,
                                                  alloc);
        else
            tex = SpectrumImageTexture::Create(renderFromTexture, parameters, loc, alloc);
    } else if (name == "checkerboard")
        tex = SpectrumCheckerboardTexture::Create(renderFromTexture, parameters, loc,
                                                  alloc);
    else if (name == "dots")
        tex = SpectrumDotsTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "marble")
        tex = MarbleTexture::Create(renderFromTexture, parameters, loc, alloc);
    else if (name == "ptex") {
        if (gpu)
            ErrorExit(loc, "ptex texture is not supported on the GPU.");
        else
            tex = SpectrumPtexTexture::Create(renderFromTexture, parameters, loc, alloc);
    } else
        ErrorExit(loc, "%s: spectrum texture type unknown.", name);

    if (!tex)
        ErrorExit(loc, "%s: unable to create texture.", name);

    ++nTextures;

    // FIXME: reenable this once we handle all the same image texture parameters
    // CO    parameters.ReportUnused();
    return tex;
}

Float UniversalTextureEvaluator::operator()(FloatTextureHandle tex,
                                            TextureEvalContext ctx) {
    return tex.Evaluate(ctx);
}

SampledSpectrum UniversalTextureEvaluator::operator()(SpectrumTextureHandle tex,
                                                      TextureEvalContext ctx,
                                                      SampledWavelengths lambda) {
    return tex.Evaluate(ctx, lambda);
}

}  // namespace pbrt
