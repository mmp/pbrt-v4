// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/mipmap.h>

#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <cmath>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Image maps", imageMapBytes);

///////////////////////////////////////////////////////////////////////////
// MIPMap Helper Declarations

std::string ToString(FilterFunction f) {
    switch (f) {
    case FilterFunction::Point:
        return "Point";
    case FilterFunction::Bilinear:
        return "Bilinear";
    case FilterFunction::Trilinear:
        return "Trilinear";
    case FilterFunction::EWA:
        return "EWA";
    default:
        LOG_FATAL("Unhandled case");
        return "";
    }
}

std::string MIPMapFilterOptions::ToString() const {
    return StringPrintf("[ MIPMapFilterOptions filter: %s maxAnisotropy: %f ]", filter,
                        maxAnisotropy);
}

///////////////////////////////////////////////////////////////////////////

/*
        for (int i = 0; i < WeightLUTSize; ++i) {
            Float alpha = 2;
            Float r2 = Float(i) / Float(WeightLUTSize - 1);
            weightLut[i] = std::exp(-alpha * r2) - std::exp(-alpha);
        }
*/
// MIPMap EWA Lookup Table Definition
static constexpr int MIPFilterLUTSize = 128;
static PBRT_CONST Float MIPFilterLUT[MIPFilterLUTSize] = {
    // MIPMap EWA Lookup Table Values
    0.864664733f,
    0.849040031f,
    0.83365953f,
    0.818519294f,
    0.80361563f,
    0.788944781f,
    0.774503231f,
    0.760287285f,
    0.746293485f,
    0.732518315f,
    0.718958378f,
    0.705610275f,
    0.692470789f,
    0.679536581f,
    0.666804492f,
    0.654271305f,
    0.641933978f,
    0.629789352f,
    0.617834508f,
    0.606066525f,
    0.594482362f,
    0.583079159f,
    0.571854174f,
    0.560804546f,
    0.549927592f,
    0.539220572f,
    0.528680861f,
    0.518305838f,
    0.50809288f,
    0.498039544f,
    0.488143265f,
    0.478401601f,
    0.468812168f,
    0.45937258f,
    0.450080454f,
    0.440933526f,
    0.431929469f,
    0.423066139f,
    0.414341331f,
    0.405752778f,
    0.397298455f,
    0.388976216f,
    0.380784035f,
    0.372719884f,
    0.364781618f,
    0.356967449f,
    0.34927541f,
    0.341703475f,
    0.334249914f,
    0.32691282f,
    0.319690347f,
    0.312580705f,
    0.305582166f,
    0.298692942f,
    0.291911423f,
    0.285235822f,
    0.278664529f,
    0.272195935f,
    0.265828371f,
    0.259560347f,
    0.253390193f,
    0.247316495f,
    0.241337672f,
    0.235452279f,
    0.229658857f,
    0.223955944f,
    0.21834214f,
    0.212816045f,
    0.207376286f,
    0.202021524f,
    0.196750447f,
    0.191561714f,
    0.186454013f,
    0.181426153f,
    0.176476851f,
    0.171604887f,
    0.166809067f,
    0.162088141f,
    0.157441005f,
    0.152866468f,
    0.148363426f,
    0.143930718f,
    0.139567271f,
    0.135272011f,
    0.131043866f,
    0.126881793f,
    0.122784719f,
    0.11875169f,
    0.114781633f,
    0.11087364f,
    0.107026696f,
    0.103239879f,
    0.0995122194f,
    0.0958427936f,
    0.0922307223f,
    0.0886750817f,
    0.0851749927f,
    0.0817295909f,
    0.0783380121f,
    0.0749994367f,
    0.0717130303f,
    0.0684779733f,
    0.0652934611f,
    0.0621587038f,
    0.0590728968f,
    0.0560353249f,
    0.0530452281f,
    0.0501018465f,
    0.0472044498f,
    0.0443523228f,
    0.0415447652f,
    0.0387810767f,
    0.0360605568f,
    0.0333825648f,
    0.0307464004f,
    0.0281514227f,
    0.0255970061f,
    0.0230824798f,
    0.0206072628f,
    0.0181707144f,
    0.0157722086f,
    0.013411209f,
    0.0110870898f,
    0.0087992847f,
    0.0065472275f,
    0.00433036685f,
    0.0021481365f,
    0.f

};

// MIPMap Method Definitions
MIPMap::MIPMap(Image image, const RGBColorSpace *colorSpace, WrapMode wrapMode,
               Allocator alloc, const MIPMapFilterOptions &options)
    : colorSpace(colorSpace), wrapMode(wrapMode), options(options) {
    CHECK(colorSpace != nullptr);
    pyramid = Image::GeneratePyramid(std::move(image), wrapMode, alloc);
    std::for_each(pyramid.begin(), pyramid.end(),
                  [](const Image &im) { imageMapBytes += im.BytesUsed(); });
}

template <>
Float MIPMap::Texel(int level, Point2i st) const {
    CHECK(level >= 0 && level < pyramid.size());
    return pyramid[level].GetChannel(st, 0, wrapMode);
}

template <>
RGB MIPMap::Texel(int level, Point2i st) const {
    CHECK(level >= 0 && level < pyramid.size());
    if (pyramid[level].NChannels() == 3 || pyramid[level].NChannels() == 4) {
        RGB rgb;
        for (int c = 0; c < 3; ++c)
            rgb[c] = pyramid[level].GetChannel(st, c, wrapMode);
        return rgb;
    } else {
        CHECK_EQ(1, pyramid[level].NChannels());
        Float v = pyramid[level].GetChannel(st, 0, wrapMode);
        return RGB(v, v, v);
    }
}

template <typename T>
T MIPMap::Filter(Point2f st, Vector2f dst0, Vector2f dst1) const {
    if (options.filter != FilterFunction::EWA) {
        // Handle non-EWA MIP Map filter
        Float width = 2 * std::max({std::abs(dst0[0]), std::abs(dst0[1]),
                                    std::abs(dst1[0]), std::abs(dst1[1])});
        // Compute MIP Map level for _width_ and handle very wide filter
        int nLevels = Levels();
        Float level = nLevels - 1 + Log2(std::max<Float>(width, 1e-8));
        if (level >= Levels() - 1)
            return Texel<T>(Levels() - 1, {0, 0});
        int iLevel = std::max(0, int(std::floor(level)));

        if (options.filter == FilterFunction::Point) {
            // Return point-sampled value at selected MIP level
            Point2i resolution = LevelResolution(iLevel);
            Point2i sti(std::round(st[0] * resolution[0] - 0.5f),
                        std::round(st[1] * resolution[1] - 0.5f));
            return Texel<T>(iLevel, sti);

        } else if (options.filter == FilterFunction::Bilinear) {
            // Return bilinear-filtered value at selected MIP level
            return Bilerp<T>(iLevel, st);

        } else {
            // Return trilinear-filtered value at selected MIP level
            CHECK(options.filter == FilterFunction::Trilinear);
            if (iLevel == 0)
                return Bilerp<T>(0, st);
            else {
                Float delta = level - iLevel;
                CHECK_LE(delta, 1);
                return Lerp(delta, Bilerp<T>(iLevel, st), Bilerp<T>(iLevel + 1, st));
            }
        }
    }
    // Compute ellipse minor and major axes
    if (LengthSquared(dst0) < LengthSquared(dst1))
        pstd::swap(dst0, dst1);
    Float majorLength = Length(dst0), minorLength = Length(dst1);

    // Clamp ellipse eccentricity if too large
    if (minorLength * options.maxAnisotropy < majorLength && minorLength > 0) {
        Float scale = majorLength / (minorLength * options.maxAnisotropy);
        dst1 *= scale;
        minorLength *= scale;
    }
    if (minorLength == 0)
        return Bilerp<T>(0, st);

    // Choose level of detail for EWA lookup and perform EWA filtering
    Float lod = std::max<Float>(0, Levels() - 1 + Log2(minorLength));
    int ilod = std::floor(lod);
    return ((1 - (lod - ilod)) * EWA<T>(ilod, st, dst0, dst1) +
            (lod - ilod) * EWA<T>(ilod + 1, st, dst0, dst1));
}

template <>
RGB MIPMap::Bilerp(int level, Point2f st) const {
    CHECK(level >= 0 && level < pyramid.size());
    if (pyramid[level].NChannels() == 3 || pyramid[level].NChannels() == 4) {
        RGB rgb;
        for (int c = 0; c < 3; ++c)
            rgb[c] = pyramid[level].BilerpChannel(st, c, wrapMode);
        return rgb;
    } else {
        CHECK_EQ(1, pyramid[level].NChannels());
        Float v = pyramid[level].BilerpChannel(st, 0, wrapMode);
        return RGB(v, v, v);
    }
}

template <typename T>
T MIPMap::EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const {
    if (level >= Levels())
        return Texel<T>(Levels() - 1, {0, 0});
    // Convert EWA coordinates to appropriate scale for level
    Point2i levelRes = LevelResolution(level);
    st[0] = st[0] * levelRes[0] - 0.5f;
    st[1] = st[1] * levelRes[1] - 0.5f;
    dst0[0] *= levelRes[0];
    dst0[1] *= levelRes[1];
    dst1[0] *= levelRes[0];
    dst1[1] *= levelRes[1];

    // Find ellipse coefficients that bound EWA filter region
    Float A = dst0[1] * dst0[1] + dst1[1] * dst1[1] + 1;
    Float B = -2 * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
    Float C = dst0[0] * dst0[0] + dst1[0] * dst1[0] + 1;
    Float invF = 1 / (A * C - B * B * 0.25f);
    A *= invF;
    B *= invF;
    C *= invF;

    // Compute the ellipse's $(s,t)$ bounding box in texture space
    Float det = -B * B + 4 * A * C;
    Float invDet = 1 / det;
    Float uSqrt = SafeSqrt(det * C), vSqrt = SafeSqrt(A * det);
    int s0 = std::ceil(st[0] - 2 * invDet * uSqrt);
    int s1 = std::floor(st[0] + 2 * invDet * uSqrt);
    int t0 = std::ceil(st[1] - 2 * invDet * vSqrt);
    int t1 = std::floor(st[1] + 2 * invDet * vSqrt);

    // Scan over ellipse bound and evaluate quadratic equation to filter image
    T sum{};
    Float sumWts = 0;
    for (int it = t0; it <= t1; ++it) {
        Float tt = it - st[1];
        for (int is = s0; is <= s1; ++is) {
            Float ss = is - st[0];
            // Compute squared radius and filter texel if it is inside the ellipse
            Float r2 = A * ss * ss + B * ss * tt + C * tt * tt;
            if (r2 < 1) {
                int index = std::min<int>(r2 * MIPFilterLUTSize, MIPFilterLUTSize - 1);
                Float weight = MIPFilterLUT[index];
                sum += weight * Texel<T>(level, {is, it});
                sumWts += weight;
            }
        }
    }
    return sum / sumWts;
}

MIPMap *MIPMap::CreateFromFile(const std::string &filename,
                               const MIPMapFilterOptions &options, WrapMode wrapMode,
                               ColorEncoding encoding, Allocator alloc) {
    ImageAndMetadata imageAndMetadata = Image::Read(filename, alloc, encoding);

    Image &image = imageAndMetadata.image;
    if (image.NChannels() != 1) {
        // Get the channels in a canonical order..
        ImageChannelDesc rgbaDesc = image.GetChannelDesc({"R", "G", "B", "A"});
        ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
        if (rgbaDesc) {
            // Is alpha all ones?
            bool allOne = true;
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x)
                    if (image.GetChannels({x, y}, rgbaDesc)[3] != 1)
                        allOne = false;
            if (allOne)
                image = image.SelectChannels(rgbDesc, alloc);
            else
                image = image.SelectChannels(rgbaDesc, alloc);
        } else {
            if (rgbDesc)
                image = image.SelectChannels(rgbDesc, alloc);
            else
                ErrorExit("%s: image doesn't have R, G, and B channels", filename);
        }
    }

    const RGBColorSpace *colorSpace = imageAndMetadata.metadata.GetColorSpace();
    return alloc.new_object<MIPMap>(std::move(image), colorSpace, wrapMode, alloc,
                                    options);
}

template <typename T>
T MIPMap::Texel(int level, Point2i st) const {
    T::unimplemented_function;
}

template <typename T>
T MIPMap::Bilerp(int level, Point2f st) const {
    T::unimplemented_function;
}

template <>
Float MIPMap::Bilerp(int level, Point2f st) const {
    CHECK(level >= 0 && level < pyramid.size());
    switch (pyramid[level].NChannels()) {
    case 1:
        return pyramid[level].BilerpChannel(st, 0, wrapMode);
    case 3:
        return pyramid[level].Bilerp(st, wrapMode).Average();
    case 4:
        // Return alpha
        return pyramid[level].BilerpChannel(st, 3, wrapMode);
    default:
        LOG_FATAL("Unexpected number of image channels: %d", pyramid[level].NChannels());
    }
}

std::string MIPMap::ToString() const {
    return StringPrintf("[ MIPMap pyramid: %s colorSpace: %s wrapMode: %s "
                        "options: %s ]",
                        pyramid, colorSpace->ToString(), wrapMode, options);
}

// Explicit template instantiation..
template Float MIPMap::Filter(Point2f st, Vector2f, Vector2f) const;
template RGB MIPMap::Filter(Point2f st, Vector2f, Vector2f) const;

}  // namespace pbrt
