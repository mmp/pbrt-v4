// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/image.h>

#include <pbrt/util/bluenoise.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/string.h>

#include <lodepng/lodepng.h>
#ifndef PBRT_IS_GPU_CODE
// Work around conflict with "half".
#include <ImfChannelList.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfFrameBuffer.h>
#include <ImfInputFile.h>
#include <ImfIntAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOutputFile.h>
#include <ImfStringVectorAttribute.h>
#endif

#include <cmath>
#include <numeric>

// use lodepng and get 16-bit.
#define STBI_NO_PNG
// too old school
#define STBI_NO_PIC
#define STBI_ASSERT CHECK
#include <stb/stb_image.h>

namespace pbrt {

std::string ToString(PixelFormat format) {
    switch (format) {
    case PixelFormat::U256:
        return "U256";
    case PixelFormat::Half:
        return "Half";
    case PixelFormat::Float:
        return "Float";
    default:
        LOG_FATAL("Unhandled PixelFormat in FormatName()");
        return "";
    }
}

int TexelBytes(PixelFormat format) {
    switch (format) {
    case PixelFormat::U256:
        return 1;
    case PixelFormat::Half:
        return 2;
    case PixelFormat::Float:
        return 4;
    default:
        LOG_FATAL("Unhandled PixelFormat in TexelBytes()");
        return 0;
    }
}

std::string ImageChannelValues::ToString() const {
    return StringPrintf("[ ImageChannelValues %s ]", ((InlinedVector<Float, 4> &)*this));
}

std::string ImageChannelDesc::ToString() const {
    return StringPrintf("[ ImageChannelDesc offset: %s ]", offset);
}

std::string ImageMetadata::ToString() const {
    return StringPrintf("[ ImageMetadata renderTimeSeconds: %s cameraFromWorld: %s "
                        "NDCFromWorld: %s pixelBounds: %s fullResolution: %s "
                        "samplesPerPixel: %s MSE: %s colorSpace: %s ]",
                        renderTimeSeconds, cameraFromWorld, NDCFromWorld, pixelBounds,
                        fullResolution, samplesPerPixel, MSE, colorSpace);
}

const RGBColorSpace *ImageMetadata::GetColorSpace() const {
    if (colorSpace && *colorSpace)
        return *colorSpace;
    return RGBColorSpace::sRGB;
}

template <typename F>
void ForExtent(const Bounds2i &extent, WrapMode2D wrapMode, Image &image, F op) {
    CHECK_LT(extent.pMin.x, extent.pMax.x);
    CHECK_LT(extent.pMin.y, extent.pMax.y);

    int nx = extent.pMax[0] - extent.pMin[0];
    int nc = image.NChannels();
    if (Intersect(extent, Bounds2i({0, 0}, image.Resolution())) == extent) {
        // All in bounds
        for (int y = extent.pMin[1]; y < extent.pMax[1]; ++y) {
            int offset = image.PixelOffset({extent.pMin[0], y});
            for (int x = 0; x < nx; ++x)
                for (int c = 0; c < nc; ++c)
                    op(offset++);
        }
    } else {
        for (int y = extent.pMin[1]; y < extent.pMax[1]; ++y) {
            for (int x = 0; x < nx; ++x) {
                Point2i p(extent.pMin[0] + x, y);
                // FIXME: this will return false on Black wrap mode
                CHECK(RemapPixelCoords(&p, image.Resolution(), wrapMode));
                int offset = image.PixelOffset(p);
                for (int c = 0; c < nc; ++c)
                    op(offset++);
            }
        }
    }
}

// Image Method Definitions
pstd::vector<Image> Image::GenerateMIPMap(Image image, WrapMode2D wrapMode,
                                          Allocator alloc) {
    PixelFormat origFormat = image.format;
    int nChannels = image.NChannels();
    ColorEncodingHandle origEncoding = image.encoding;

    // Set things up so we have a power-of-two sized image stored with
    // floats.
    if (!IsPowerOf2(image.resolution[0]) || !IsPowerOf2(image.resolution[1])) {
        // Resample image to power-of-two resolution
        image = image.FloatResize(
            {RoundUpPow2(image.resolution[0]), RoundUpPow2(image.resolution[1])},
            wrapMode);
    } else if (!Is32Bit(image.format))
        image = image.ConvertToFormat(PixelFormat::Float);
    CHECK(Is32Bit(image.format));

    // Initialize levels of MIPMap from image
    int nLevels = 1 + Log2Int(std::max(image.resolution[0], image.resolution[1]));
    pstd::vector<Image> pyramid(alloc);
    pyramid.reserve(nLevels);

    Point2i levelResolution = image.resolution;
    for (int i = 0; i < nLevels - 1; ++i) {
        // Initialize $i+1$st MIPMap level from $i$th level and also convert
        // i'th level to the internal format
        pyramid.push_back(
            Image(origFormat, levelResolution, image.channelNames, origEncoding, alloc));

        Point2i nextResolution(std::max(1, levelResolution[0] / 2),
                               std::max(1, levelResolution[1] / 2));
        Image nextImage(image.format, nextResolution, image.channelNames, origEncoding);

        // Offsets from the base pixel to the four neighbors that we'll
        // downfilter.
        int srcDeltas[4] = {0, nChannels, nChannels * levelResolution[0],
                            nChannels * levelResolution[0] + nChannels};
        // Clamp offsets once a dimension has a single texel.
        if (levelResolution[0] == 1) {
            srcDeltas[1] = 0;
            srcDeltas[3] -= nChannels;
        }
        if (levelResolution[1] == 1) {
            srcDeltas[2] = 0;
            srcDeltas[3] -= nChannels * levelResolution[0];
        }

        // Work in scanlines for best cache coherence (vs 2d tiles).
        ParallelFor(0, nextResolution[1], [&](int64_t y0, int64_t y1) {
            for (int y = y0; y < y1; ++y) {
                // Downfilter with a box filter for the next MIP level
                int srcOffset = image.PixelOffset({0, 2 * y});
                int nextOffset = nextImage.PixelOffset({0, y});
                for (int x = 0; x < nextResolution[0]; ++x) {
                    for (int c = 0; c < nChannels; ++c) {
                        nextImage.p32[nextOffset] =
                            .25f *
                            (image.p32[srcOffset] + image.p32[srcOffset + srcDeltas[1]] +
                             image.p32[srcOffset + srcDeltas[2]] +
                             image.p32[srcOffset + srcDeltas[3]]);
                        ++srcOffset;
                        ++nextOffset;
                    }
                    srcOffset += nChannels;
                }

                // Copy the current level out to the current pyramid level
                int yStart = 2 * y;
                int yEnd = std::min(2 * y + 2, levelResolution[1]);
                int offset = image.PixelOffset({0, yStart});
                size_t count = (yEnd - yStart) * nChannels * levelResolution[0];
                pyramid[i].CopyRectIn(Bounds2i({0, yStart}, {levelResolution[0], yEnd}),
                                      {image.p32.data() + offset, count});
            }
        });

        image = std::move(nextImage);
        levelResolution = nextResolution;
    }

    // Top level
    CHECK(levelResolution[0] == 1 && levelResolution[1] == 1);
    pyramid.push_back(
        Image(origFormat, levelResolution, image.channelNames, origEncoding, alloc));
    pyramid[nLevels - 1].CopyRectIn({{0, 0}, {1, 1}},
                                    {image.p32.data(), size_t(nChannels)});

    return pyramid;
}

Image::Image(PixelFormat format, Point2i resolution,
             pstd::span<const std::string> channels, ColorEncodingHandle encoding,
             Allocator alloc)
    : format(format),
      resolution(resolution),
      channelNames(channels.begin(), channels.end()),
      encoding(encoding),
      p8(alloc),
      p16(alloc),
      p32(alloc) {
    if (Is8Bit(format)) {
        p8.resize(NChannels() * resolution[0] * resolution[1]);
        CHECK(encoding != nullptr);
    } else if (Is16Bit(format))
        p16.resize(NChannels() * resolution[0] * resolution[1]);
    else if (Is32Bit(format))
        p32.resize(NChannels() * resolution[0] * resolution[1]);
    else
        LOG_FATAL("Unhandled format in Image::Image()");
}

ImageChannelDesc Image::GetChannelDesc(
    pstd::span<const std::string> requestedChannels) const {
    ImageChannelDesc desc;
    desc.offset.resize(requestedChannels.size());
    for (size_t i = 0; i < requestedChannels.size(); ++i) {
        size_t j;
        for (j = 0; j < channelNames.size(); ++j)
            if (requestedChannels[i] == channelNames[j]) {
                desc.offset[i] = j;
                break;
            }
        if (j == channelNames.size())
            return {};
    }

    return desc;
}

ImageChannelValues Image::GetChannels(Point2i p, const ImageChannelDesc &desc,
                                      WrapMode2D wrapMode) const {
    ImageChannelValues cv(desc.offset.size(), Float(0));
    if (!RemapPixelCoords(&p, resolution, wrapMode))
        return cv;

    size_t pixelOffset = PixelOffset(p);
    switch (format) {
    case PixelFormat::U256: {
        for (int i = 0; i < desc.offset.size(); ++i)
            encoding.ToLinear({&p8[pixelOffset + desc.offset[i]], 1}, {&cv[i], 1});
        break;
    }
    case PixelFormat::Half: {
        for (int i = 0; i < desc.offset.size(); ++i)
            cv[i] = Float(p16[pixelOffset + desc.offset[i]]);
        break;
    }
    case PixelFormat::Float: {
        for (int i = 0; i < desc.offset.size(); ++i)
            cv[i] = p32[pixelOffset + desc.offset[i]];
        break;
    }
    default:
        LOG_FATAL("Unhandled PixelFormat");
    }

    return cv;
}

void Image::SetChannels(Point2i p, const ImageChannelDesc &desc,
                        pstd::span<const Float> values) {
    CHECK_LE(values.size(), NChannels());
    for (size_t i = 0; i < values.size(); ++i)
        SetChannel(p, desc.offset[i], values[i]);
}

Image Image::GaussianFilter(const ImageChannelDesc &desc, int halfWidth,
                            Float sigma) const {
    // Compute filter weights
    std::vector<Float> wts(2 * halfWidth + 1, Float(0));
    for (int d = 0; d < 2 * halfWidth + 1; ++d)
        wts[d] = Gaussian(d - halfWidth, 0, sigma);

    // Normalize weights
    Float wtSum = std::accumulate(wts.begin(), wts.end(), Float(0));
    for (Float &w : wts)
        w /= wtSum;

    // Separable blur; first blur in x into blurx, selecting out the
    // desired channels along the way.
    Image blurx(PixelFormat::Float, resolution, ChannelNames(desc));
    int nc = desc.size();
    ParallelFor(0, resolution.y, [&](int64_t y0, int64_t y1) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                ImageChannelValues result(desc.size());
                for (int r = -halfWidth; r <= halfWidth; ++r) {
                    ImageChannelValues cv = GetChannels({x + r, y}, desc);
                    for (int c = 0; c < nc; ++c)
                        result[c] += wts[r + halfWidth] * cv[c];
                }
                blurx.SetChannels({x, y}, result);
            }
        }
    });

    // Now blur in y from blur x to the result; blurx has just the
    // channels we want already.
    Image blury(PixelFormat::Float, resolution, ChannelNames(desc));
    ParallelFor(0, resolution.y, [&](int64_t y0, int64_t y1) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                ImageChannelValues result(desc.size());
                for (int r = -halfWidth; r <= halfWidth; ++r) {
                    ImageChannelValues cv = blurx.GetChannels({x, y + r});
                    for (int c = 0; c < nc; ++c)
                        result[c] += wts[r + halfWidth] * cv[c];
                }
                blury.SetChannels({x, y}, result);
            }
        }
    });
    return blury;
}

std::vector<ResampleWeight> Image::resampleWeights(int oldRes, int newRes) {
    CHECK_GE(newRes, oldRes);
    std::vector<ResampleWeight> wt(newRes);
    Float filterwidth = 2.f;
    for (int i = 0; i < newRes; ++i) {
        // Compute image resampling weights for _i_th texel
        Float center = (i + .5f) * oldRes / newRes;
        wt[i].firstTexel = std::floor((center - filterwidth) + 0.5f);
        for (int j = 0; j < 4; ++j) {
            Float pos = wt[i].firstTexel + j + .5f;
            wt[i].weight[j] = WindowedSinc(pos - center, filterwidth, 2.f);
        }

        // Normalize filter weights for texel resampling
        Float invSumWts =
            1 / (wt[i].weight[0] + wt[i].weight[1] + wt[i].weight[2] + wt[i].weight[3]);
        for (int j = 0; j < 4; ++j)
            wt[i].weight[j] *= invSumWts;
    }
    return wt;
}

Image Image::FloatResize(Point2i newResolution, WrapMode2D wrapMode) const {
    CHECK_GE(newResolution.x, resolution.x);
    CHECK_GE(newResolution.y, resolution.y);

    std::vector<ResampleWeight> xWeights =
        resampleWeights(resolution[0], newResolution[0]);
    std::vector<ResampleWeight> yWeights =
        resampleWeights(resolution[1], newResolution[1]);
    Image resampledImage(PixelFormat::Float, newResolution, channelNames);

    // Note: these aren't freed until the corresponding worker thread
    // exits, but that's probably ok...
    thread_local std::vector<float> inBuf, xBuf, outBuf;

    ParallelFor2D(Bounds2i({0, 0}, newResolution), [&](Bounds2i outExtent) {
        Bounds2i inExtent(
            {xWeights[outExtent[0][0]].firstTexel, yWeights[outExtent[0][1]].firstTexel},
            {xWeights[outExtent[1][0] - 1].firstTexel + 4,
             yWeights[outExtent[1][1] - 1].firstTexel + 4});

        if (inBuf.size() < NChannels() * inExtent.Area())
            inBuf.resize(NChannels() * inExtent.Area());

        // Copy the tile of the input image into inBuf. (The
        // main motivation for this copy is to convert it
        // into floats all at once, rather than repeatedly
        // and pixel-by-pixel during the first resampling
        // step.)
        // FIXME CAST
        ((Image *)this)->CopyRectOut(inExtent, pstd::MakeSpan(inBuf), wrapMode);

        // Zoom in x. We need to do this across all scanlines
        // in inExtent's y dimension so we have the border
        // pixels available for the zoom in y.
        int nxOut = outExtent[1][0] - outExtent[0][0];
        int nyOut = outExtent[1][1] - outExtent[0][1];
        int nxIn = inExtent[1][0] - inExtent[0][0];
        int nyIn = inExtent[1][1] - inExtent[0][1];

        if (xBuf.size() < NChannels() * nyIn * nxOut)
            xBuf.resize(NChannels() * nyIn * nxOut);

        int xBufOffset = 0;
        for (int y = 0; y < nyIn; ++y) {
            for (int x = 0; x < nxOut; ++x) {
                int xOut = x + outExtent[0][0];
                DCHECK(xOut >= 0 && xOut < xWeights.size());
                const ResampleWeight &rsw = xWeights[xOut];

                // w.r.t. inBuf
                int xIn = rsw.firstTexel - inExtent[0][0];
                DCHECK_GE(xIn, 0);
                DCHECK_LT(xIn + 3, nxIn);

                int inOffset = NChannels() * (xIn + y * nxIn);
                DCHECK_GE(inOffset, 0);
                DCHECK_LT(inOffset + 3 * NChannels(), inBuf.size());
                for (int c = 0; c < NChannels(); ++c, ++xBufOffset, ++inOffset) {
                    xBuf[xBufOffset] =
                        (rsw.weight[0] * inBuf[inOffset] +
                         rsw.weight[1] * inBuf[inOffset + NChannels()] +
                         rsw.weight[2] * inBuf[inOffset + 2 * NChannels()] +
                         rsw.weight[3] * inBuf[inOffset + 3 * NChannels()]);
                }
            }
        }

        if (outBuf.size() < NChannels() * nxOut * nyOut)
            outBuf.resize(NChannels() * nxOut * nyOut);

        // Zoom in y from xBuf to outBuf
        for (int x = 0; x < nxOut; ++x) {
            for (int y = 0; y < nyOut; ++y) {
                int yOut = y + outExtent[0][1];
                DCHECK(yOut >= 0 && yOut < yWeights.size());
                const ResampleWeight &rsw = yWeights[yOut];

                DCHECK_GE(rsw.firstTexel - inExtent[0][1], 0);
                int xBufOffset =
                    NChannels() * (x + nxOut * (rsw.firstTexel - inExtent[0][1]));
                DCHECK_GE(xBufOffset, 0);
                int step = NChannels() * nxOut;
                DCHECK_LT(xBufOffset + 3 * step, xBuf.size());

                int outOffset = NChannels() * (x + y * nxOut);
                for (int c = 0; c < NChannels(); ++c, ++outOffset, ++xBufOffset)
                    outBuf[outOffset] =
                        std::max<Float>(0, (rsw.weight[0] * xBuf[xBufOffset] +
                                            rsw.weight[1] * xBuf[xBufOffset + step] +
                                            rsw.weight[2] * xBuf[xBufOffset + 2 * step] +
                                            rsw.weight[3] * xBuf[xBufOffset + 3 * step]));
            }
        }
        // Copy out...
        resampledImage.CopyRectIn(outExtent, outBuf);
    });

    return resampledImage;
}

Image::Image(pstd::vector<uint8_t> p8c, Point2i resolution,
             pstd::span<const std::string> channels, ColorEncodingHandle encoding)
    : format(PixelFormat::U256),
      resolution(resolution),
      channelNames(channels.begin(), channels.end()),
      encoding(encoding),
      p8(std::move(p8c)) {
    CHECK_EQ(p8.size(), NChannels() * resolution[0] * resolution[1]);
}

Image::Image(pstd::vector<Half> p16c, Point2i resolution,
             pstd::span<const std::string> channels)
    : format(PixelFormat::Half),
      resolution(resolution),
      channelNames(channels.begin(), channels.end()),
      p16(std::move(p16c)) {
    CHECK_EQ(p16.size(), NChannels() * resolution[0] * resolution[1]);
    CHECK(Is16Bit(format));
}

Image::Image(pstd::vector<float> p32c, Point2i resolution,
             pstd::span<const std::string> channels)
    : format(PixelFormat::Float),
      resolution(resolution),
      channelNames(channels.begin(), channels.end()),
      p32(std::move(p32c)) {
    CHECK_EQ(p32.size(), NChannels() * resolution[0] * resolution[1]);
    CHECK(Is32Bit(format));
}

Image Image::ConvertToFormat(PixelFormat newFormat, ColorEncodingHandle encoding) const {
    if (newFormat == format)
        return *this;

    Image newImage(newFormat, resolution, channelNames, encoding);
    for (int y = 0; y < resolution.y; ++y)
        for (int x = 0; x < resolution.x; ++x)
            for (int c = 0; c < NChannels(); ++c)
                newImage.SetChannel({x, y}, c, GetChannel({x, y}, c));
    return newImage;
}

ImageChannelValues Image::GetChannels(Point2i p, WrapMode2D wrapMode) const {
    ImageChannelValues cv(NChannels(), Float(0));
    if (!RemapPixelCoords(&p, resolution, wrapMode))
        return cv;

    size_t pixelOffset = PixelOffset(p);
    switch (format) {
    case PixelFormat::U256: {
        encoding.ToLinear({&p8[pixelOffset], size_t(NChannels())},
                          {&cv[0], size_t(NChannels())});
        break;
    }
    case PixelFormat::Half: {
        for (int i = 0; i < NChannels(); ++i)
            cv[i] = Float(p16[pixelOffset + i]);
        break;
    }
    case PixelFormat::Float: {
        for (int i = 0; i < NChannels(); ++i)
            cv[i] = p32[pixelOffset + i];
        break;
    }
    default:
        LOG_FATAL("Unhandled PixelFormat");
    }

    return cv;
}

std::vector<std::string> Image::ChannelNames() const {
    return {channelNames.begin(), channelNames.end()};
}

std::vector<std::string> Image::ChannelNames(const ImageChannelDesc &desc) const {
    std::vector<std::string> names;
    for (int i = 0; i < desc.size(); ++i)
        names.push_back(channelNames[desc.offset[i]]);
    return names;
}

ImageChannelValues Image::L1Error(const ImageChannelDesc &desc, const Image &ref,
                                  Image *errorImage) const {
    std::vector<double> sumError(desc.size(), 0.);

    ImageChannelDesc refDesc = ref.GetChannelDesc(ChannelNames(desc));
    CHECK((bool)refDesc);
    CHECK_EQ(Resolution(), ref.Resolution());

    if (errorImage)
        *errorImage = Image(PixelFormat::Float, Resolution(), ChannelNames());

    for (int y = 0; y < Resolution().y; ++y)
        for (int x = 0; x < Resolution().x; ++x) {
            ImageChannelValues v = GetChannels({x, y}, desc);
            ImageChannelValues vref = ref.GetChannels({x, y}, refDesc);

            for (int c = 0; c < desc.size(); ++c) {
                Float error = v[c] - vref[c];
                if (std::isinf(error))
                    continue;
                sumError[c] += error;
                if (errorImage)
                    errorImage->SetChannel({x, y}, c, error);
            }
        }

    ImageChannelValues error(desc.size());
    for (int c = 0; c < desc.size(); ++c)
        error[c] = sumError[c] / (Resolution().x * Resolution().y);
    return error;
}

ImageChannelValues Image::MSE(const ImageChannelDesc &desc, const Image &ref,
                              Image *mseImage) const {
    std::vector<double> sumSE(desc.size(), 0.);

    ImageChannelDesc refDesc = ref.GetChannelDesc(ChannelNames(desc));
    if (!refDesc)
        ErrorExit("Channels not found in image: %s", ChannelNames(desc));

    CHECK_EQ(Resolution(), ref.Resolution());

    if (mseImage)
        *mseImage = Image(PixelFormat::Float, Resolution(), ChannelNames());

    for (int y = 0; y < Resolution().y; ++y)
        for (int x = 0; x < Resolution().x; ++x) {
            ImageChannelValues v = GetChannels({x, y}, desc);
            ImageChannelValues vref = ref.GetChannels({x, y}, refDesc);

            for (int c = 0; c < desc.size(); ++c) {
                Float se = Sqr(v[c] - vref[c]);
                if (std::isinf(se))
                    continue;
                sumSE[c] += se;
                if (mseImage)
                    mseImage->SetChannel({x, y}, c, se);
            }
        }

    ImageChannelValues mse(desc.size());
    for (int c = 0; c < desc.size(); ++c)
        mse[c] = sumSE[c] / (Resolution().x * Resolution().y);
    return mse;
}

ImageChannelValues Image::MRSE(const ImageChannelDesc &desc, const Image &ref,
                               Image *mrseImage) const {
    std::vector<double> sumRSE(desc.size(), 0.);

    ImageChannelDesc refDesc = ref.GetChannelDesc(ChannelNames(desc));
    CHECK((bool)refDesc);
    CHECK_EQ(Resolution(), ref.Resolution());

    if (mrseImage)
        *mrseImage = Image(PixelFormat::Float, Resolution(), ChannelNames());

    for (int y = 0; y < Resolution().y; ++y)
        for (int x = 0; x < Resolution().x; ++x) {
            ImageChannelValues v = GetChannels({x, y}, desc);
            ImageChannelValues vref = ref.GetChannels({x, y}, refDesc);

            for (int c = 0; c < desc.size(); ++c) {
                Float rse = Sqr(v[c] - vref[c]) / Sqr(vref[c] + 0.01);
                if (std::isinf(rse))
                    continue;
                sumRSE[c] += rse;
                if (mrseImage)
                    mrseImage->SetChannel({x, y}, c, rse);
            }
        }

    ImageChannelValues mrse(desc.size());
    for (int c = 0; c < desc.size(); ++c)
        mrse[c] = sumRSE[c] / (Resolution().x * Resolution().y);
    return mrse;
}

ImageChannelValues Image::Average(const ImageChannelDesc &desc) const {
    std::vector<double> sum(desc.size(), 0.);

    for (int y = 0; y < Resolution().y; ++y)
        for (int x = 0; x < Resolution().x; ++x) {
            ImageChannelValues v = GetChannels({x, y}, desc);
            for (int c = 0; c < desc.size(); ++c)
                sum[c] += v[c];
        }

    ImageChannelValues average(desc.size());
    for (int c = 0; c < desc.size(); ++c)
        average[c] = sum[c] / (Resolution().x * Resolution().y);
    return average;
}

void Image::CopyRectOut(const Bounds2i &extent, pstd::span<float> buf,
                        WrapMode2D wrapMode) {
    CHECK_GE(buf.size(), extent.Area() * NChannels());

    auto bufIter = buf.begin();
    switch (format) {
    case PixelFormat::U256:
        if (Intersect(extent, Bounds2i({0, 0}, resolution)) == extent) {
            // All in bounds
            size_t count = NChannels() * (extent.pMax.x - extent.pMin.x);
            for (int y = extent.pMin.y; y < extent.pMax.y; ++y) {
                // Convert scanlines all at once.
                size_t offset = PixelOffset({extent.pMin.x, y});
#ifdef PBRT_FLOAT_AS_DOUBLE
                for (int i = 0; i < count; ++i) {
                    Float v;
                    encoding.ToLinear({&p8[offset + i], 1}, {&v, 1});
                    *bufIter++ = v;
                }
#else
                encoding.ToLinear({&p8[offset], count}, {&*bufIter, count});
#endif
                bufIter += count;
            }
        } else {
            ForExtent(extent, wrapMode, *this, [&bufIter, this](int offset) {
#ifdef PBRT_FLOAT_AS_DOUBLE
                Float v;
                encoding.ToLinear({&p8[offset], 1}, {&v, 1});
                *bufIter = v;
#else
                encoding.ToLinear({&p8[offset], 1}, {&*bufIter, 1});
#endif
                ++bufIter;
            });
        }
        break;

    case PixelFormat::Half:
        ForExtent(extent, wrapMode, *this,
                  [&bufIter, this](int offset) { *bufIter++ = Float(p16[offset]); });
        break;

    case PixelFormat::Float:
        ForExtent(extent, wrapMode, *this,
                  [&bufIter, this](int offset) { *bufIter++ = Float(p32[offset]); });
        break;

    default:
        LOG_FATAL("Unhandled PixelFormat");
    }
}

void Image::CopyRectIn(const Bounds2i &extent, pstd::span<const float> buf) {
    CHECK_GE(buf.size(), extent.Area() * NChannels());

    auto bufIter = buf.begin();
    switch (format) {
    case PixelFormat::U256:
        if (Intersect(extent, Bounds2i({0, 0}, resolution)) == extent) {
            // All in bounds
            size_t count = NChannels() * (extent.pMax.x - extent.pMin.x);
            for (int y = extent.pMin.y; y < extent.pMax.y; ++y) {
                // Convert scanlines all at once.
                size_t offset = PixelOffset({extent.pMin.x, y});
#ifdef PBRT_FLOAT_AS_DOUBLE
                for (int i = 0; i < count; ++i) {
                    Float v = *bufIter++;
                    encoding.FromLinear({&v, 1}, {&p8[offset + i], 1});
                }
#else
                encoding.FromLinear({&*bufIter, count}, {&p8[offset], count});
                bufIter += count;
#endif
            }
        } else
            ForExtent(extent, WrapMode::Clamp, *this, [&bufIter, this](int offset) {
                Float v = *bufIter++;
                encoding.FromLinear({&v, 1}, {&p8[offset], 1});
            });
        break;

    case PixelFormat::Half:
        ForExtent(extent, WrapMode::Clamp, *this,
                  [&bufIter, this](int offset) { p16[offset] = Half(*bufIter++); });
        break;

    case PixelFormat::Float:
        ForExtent(extent, WrapMode::Clamp, *this,
                  [&bufIter, this](int offset) { p32[offset] = *bufIter++; });
        break;

    default:
        LOG_FATAL("Unhandled PixelFormat");
    }
}

ImageChannelValues Image::LookupNearest(Point2f p, WrapMode2D wrapMode) const {
    ImageChannelValues cv(NChannels(), Float(0));
    for (int c = 0; c < NChannels(); ++c)
        cv[c] = LookupNearestChannel(p, c, wrapMode);
    return cv;
}

ImageChannelValues Image::LookupNearest(Point2f p, const ImageChannelDesc &desc,
                                        WrapMode2D wrapMode) const {
    ImageChannelValues cv(desc.offset.size(), Float(0));
    for (int i = 0; i < desc.offset.size(); ++i)
        cv[i] = LookupNearestChannel(p, desc.offset[i], wrapMode);
    return cv;
}

ImageChannelValues Image::Bilerp(Point2f p, WrapMode2D wrapMode) const {
    ImageChannelValues cv(NChannels(), Float(0));
    for (int c = 0; c < NChannels(); ++c)
        cv[c] = BilerpChannel(p, c, wrapMode);
    return cv;
}

ImageChannelValues Image::Bilerp(Point2f p, const ImageChannelDesc &desc,
                                 WrapMode2D wrapMode) const {
    ImageChannelValues cv(desc.offset.size(), Float(0));
    for (int i = 0; i < desc.offset.size(); ++i)
        cv[i] = BilerpChannel(p, desc.offset[i], wrapMode);
    return cv;
}

void Image::SetChannels(Point2i p, const ImageChannelValues &values) {
    CHECK_LE(values.size(), NChannels());
    int i = 0;
    for (auto iter = values.begin(); iter != values.end(); ++iter, ++i)
        SetChannel(p, i, *iter);
}

void Image::SetChannels(Point2i p, pstd::span<const Float> values) {
    CHECK_LE(values.size(), NChannels());
    for (size_t i = 0; i < values.size(); ++i)
        SetChannel(p, i, values[i]);
}

void Image::FlipY() {
    for (int y = 0; y < resolution.y / 2; ++y) {
        for (int x = 0; x < resolution.x; ++x) {
            size_t o1 = PixelOffset({x, y}), o2 = PixelOffset({x, resolution.y - 1 - y});
            for (int c = 0; c < NChannels(); ++c) {
                if (Is8Bit(format))
                    pstd::swap(p8[o1 + c], p8[o2 + c]);
                else if (Is16Bit(format))
                    pstd::swap(p16[o1 + c], p16[o2 + c]);
                else if (Is32Bit(format))
                    pstd::swap(p32[o1 + c], p32[o2 + c]);
                else
                    LOG_FATAL("unexpected format");
            }
        }
    }
}

Image Image::JointBilateralFilter(const ImageChannelDesc &toFilterDesc, int halfWidth,
                                  const Float xySigma[2],
                                  const ImageChannelDesc &jointDesc,
                                  const ImageChannelValues &jointSigma) const {
    CHECK_EQ(jointDesc.size(), jointSigma.size());
    Image result(PixelFormat::Float, resolution, ChannelNames(toFilterDesc));

    std::vector<Float> fx, fy;
    for (int i = 0; i <= halfWidth; ++i) {
        fx.push_back(Gaussian(i, 0, xySigma[0]));
        fy.push_back(Gaussian(i, 0, xySigma[1]));
    }

    ParallelFor(0, resolution.y, [&](int64_t y0, int64_t y1) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                ImageChannelValues jointPixelChannels = GetChannels({x, y}, jointDesc);
                ImageChannelValues filteredSum(toFilterDesc.size(), Float(0));
                Float weightSum = 0;

                for (int dy = -halfWidth + 1; dy < halfWidth; ++dy) {
                    if (y + dy < 0 || y + dy >= resolution.y)
                        continue;
                    for (int dx = -halfWidth + 1; dx < halfWidth; ++dx) {
                        for (int dx = -halfWidth + 1; dx < halfWidth; ++dx) {
                            if (x + dx < 0 || x + dx >= resolution.x)
                                continue;
                            ImageChannelValues jointOtherChannels =
                                GetChannels({x + dx, y + dy}, jointDesc);
                            Float weight = fx[std::abs(dx)] * fy[std::abs(dy)];
                            for (int c = 0; c < jointDesc.size(); ++c)
                                weight *= Gaussian(jointPixelChannels[c],
                                                   jointOtherChannels[c], jointSigma[c]);
                            weightSum += weight;

                            ImageChannelValues filterChannels =
                                GetChannels({x + dx, y + dy}, toFilterDesc);
                            for (int c = 0; c < filterChannels.size(); ++c)
                                filteredSum[c] += weight * filterChannels[c];
                        }
                    }
                }
                if (weightSum > 0)
                    for (int c = 0; c < filteredSum.size(); ++c)
                        filteredSum[c] /= weightSum;
                result.SetChannels({x, y}, filteredSum);
            }
        }
    });

    return result;
}

Array2D<Float> Image::GetSamplingDistribution(std::function<Float(Point2f)> dxdA,
                                              const Bounds2f &domain, Allocator alloc) {
    Array2D<Float> dist(resolution[0], resolution[1], alloc);
    ParallelFor(0, resolution[1], [&](int64_t y0, int64_t y1) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < resolution[0]; ++x) {
                // This is noticably better than MaxValue: discuss / show
                // example..
                Float value = GetChannels({x, y}).Average();

                // Assume jacobian term is basically constant over the
                // region.
                Point2f p = domain.Lerp(
                    Point2f((x + .5f) / resolution[0], (y + .5f) / resolution[1]));
                dist(x, y) = value * dxdA(p);
            }
        }
    });
    return dist;
}

// ImageIO Local Declarations
static ImageAndMetadata ReadEXR(const std::string &name, Allocator alloc);
static ImageAndMetadata ReadPNG(const std::string &name, Allocator alloc,
                                ColorEncodingHandle encoding);
static ImageAndMetadata ReadPFM(const std::string &filename, Allocator alloc);
static ImageAndMetadata ReadHDR(const std::string &filename, Allocator alloc);

// ImageIO Function Definitions
ImageAndMetadata Image::Read(const std::string &name, Allocator alloc,
                             ColorEncodingHandle encoding) {
    if (HasExtension(name, "exr"))
        return ReadEXR(name, alloc);
    else if (HasExtension(name, "png"))
        return ReadPNG(name, alloc, encoding);
    else if (HasExtension(name, "pfm"))
        return ReadPFM(name, alloc);
    else if (HasExtension(name, "hdr"))
        return ReadHDR(name, alloc);
    else {
        int x, y, n;
        unsigned char *data = stbi_load(name.c_str(), &x, &y, &n, 0);
        if (data) {
            pstd::vector<uint8_t> pixels(data, data + x * y * n, alloc);
            stbi_image_free(data);
            switch (n) {
            case 1:
                return ImageAndMetadata{
                    Image(std::move(pixels), {x, y}, {"Y"}, ColorEncodingHandle::sRGB),
                    ImageMetadata()};
            case 2: {
                Image image(std::move(pixels), {x, y}, {"Y", "A"},
                            ColorEncodingHandle::sRGB);
                return ImageAndMetadata{image.SelectChannels(image.GetChannelDesc({"Y"})),
                                        ImageMetadata()};
            }
            case 3:
                return ImageAndMetadata{Image(std::move(pixels), {x, y}, {"R", "G", "B"},
                                              ColorEncodingHandle::sRGB),
                                        ImageMetadata()};
            case 4: {
                Image image(std::move(pixels), {x, y}, {"R", "G", "B", "A"},
                            ColorEncodingHandle::sRGB);
                return ImageAndMetadata{
                    image.SelectChannels(image.GetChannelDesc({"R", "G", "B"})),
                    ImageMetadata()};
            }
            default:
                ErrorExit("%s: %d channel image unsupported.", name, n);
            }
        } else
            ErrorExit("%s: no support for reading images with this extension", name);
    }
}

bool Image::Write(const std::string &name, const ImageMetadata &metadata) const {
    if (metadata.pixelBounds)
        CHECK_EQ(metadata.pixelBounds->Area(), resolution.x * resolution.y);

    if (HasExtension(name, "exr"))
        return WriteEXR(name, metadata);

    if (NChannels() > 4) {
        Error("%s: unable to write an %d channel image in this format.", name,
              NChannels());
        return false;
    }

    const Image *image = this;
    Image rgbImage;
    if (NChannels() == 4) {
        ImageChannelDesc desc = GetChannelDesc({"R", "G", "B", "A"});
        if (desc) {
            rgbImage = SelectChannels(GetChannelDesc({"R", "G", "B"}));
            image = &rgbImage;
        } else {
            Error("%s: unable to write an 4 channel image that is not RGBA.", name);
            return false;
        }
    }
    if (NChannels() == 3 && *metadata.GetColorSpace() != *RGBColorSpace::sRGB)
        Warning("%s: writing image with non-sRGB color space to a format that "
                "doesn't store color spaces.",
                name);

    if (NChannels() == 3) {
        // Order as RGB
        ImageChannelDesc desc = GetChannelDesc({"R", "G", "B"});
        if (!desc)
            Warning("%s: 3-channels but doesn't have R, G, and B. "
                    "Image may be garbled.",
                    name);
        else {
            rgbImage = SelectChannels(desc);
            image = &rgbImage;
        }
    }

    if (HasExtension(name, "pfm"))
        return image->WritePFM(name, metadata);
    else if (HasExtension(name, "png"))
        return image->WritePNG(name, metadata);
    else {
        Error("%s: no support for writing images with this extension", name);
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////
// OpenEXR

static Imf::FrameBuffer imageToFrameBuffer(const Image &image,
                                           const ImageChannelDesc &desc,
                                           const Imath::Box2i &dataWindow) {
    size_t xStride = image.NChannels() * TexelBytes(image.Format());
    size_t yStride = image.Resolution().x * xStride;
    // Would be nice to use PixelOffset(-dw.min.x, -dw.min.y) but
    // it checks to make sure the coordiantes are >= 0 (which
    // usually makes sense...)
    char *originPtr = (((char *)image.RawPointer({0, 0})) - dataWindow.min.x * xStride -
                       dataWindow.min.y * yStride);

    Imf::FrameBuffer fb;
    std::vector<std::string> channelNames = image.ChannelNames();
    switch (image.Format()) {
    case PixelFormat::Half:
        for (int channelIndex : desc.offset)
            fb.insert(channelNames[channelIndex],
                      Imf::Slice(Imf::HALF, originPtr + channelIndex * sizeof(Half),
                                 xStride, yStride));
        break;
    case PixelFormat::Float:
        for (int channelIndex : desc.offset)
            fb.insert(channelNames[channelIndex],
                      Imf::Slice(Imf::FLOAT, originPtr + channelIndex * sizeof(float),
                                 xStride, yStride));
        break;
    default:
        LOG_FATAL("Unexpected image format");
    }
    return fb;
}

static ImageAndMetadata ReadEXR(const std::string &name, Allocator alloc) {
    try {
        Imf::InputFile file(name.c_str());
        Imath::Box2i dw = file.header().dataWindow();

        ImageMetadata metadata;
        const Imf::FloatAttribute *renderTimeAttrib =
            file.header().findTypedAttribute<Imf::FloatAttribute>("renderTimeSeconds");
        if (renderTimeAttrib != nullptr)
            metadata.renderTimeSeconds = renderTimeAttrib->value();

        const Imf::M44fAttribute *worldToCameraAttrib =
            file.header().findTypedAttribute<Imf::M44fAttribute>("worldToCamera");
        if (worldToCameraAttrib != nullptr) {
            SquareMatrix<4> m;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    // Can't memcpy since Float may be a double...
                    m[i][j] = worldToCameraAttrib->value().getValue()[4 * i + j];
            metadata.cameraFromWorld = m;
        }

        const Imf::M44fAttribute *worldToNDCAttrib =
            file.header().findTypedAttribute<Imf::M44fAttribute>("worldToNDC");
        if (worldToNDCAttrib != nullptr) {
            SquareMatrix<4> m;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] = worldToNDCAttrib->value().getValue()[4 * i + j];
            metadata.NDCFromWorld = m;
        }

        // OpenEXR uses inclusive pixel bounds; adjust to non-inclusive
        // (the convention pbrt uses) in the values returned.
        metadata.pixelBounds = {{dw.min.x, dw.min.y}, {dw.max.x + 1, dw.max.y + 1}};

        Imath::Box2i dispw = file.header().displayWindow();
        metadata.fullResolution =
            Point2i(dispw.max.x - dispw.min.x + 1, dispw.max.y - dispw.min.y + 1);

        const Imf::IntAttribute *sppAttrib =
            file.header().findTypedAttribute<Imf::IntAttribute>("samplesPerPixel");
        if (sppAttrib != nullptr)
            metadata.samplesPerPixel = sppAttrib->value();

        const Imf::FloatAttribute *mseAttrib =
            file.header().findTypedAttribute<Imf::FloatAttribute>("MSE");
        if (mseAttrib != nullptr)
            metadata.MSE = mseAttrib->value();

        // Find any string vector attributes
        for (auto iter = file.header().begin(); iter != file.header().end(); ++iter) {
            if (strcmp(iter.attribute().typeName(), "stringvector") == 0) {
                Imf::StringVectorAttribute &sv =
                    (Imf::StringVectorAttribute &)iter.attribute();
                metadata.stringVectors[iter.name()] = sv.value();
            }
        }

        // Figure out the color space
        const RGBColorSpace *colorSpace = RGBColorSpace::sRGB;  // default
        const Imf::ChromaticitiesAttribute *chromaticitiesAttrib =
            file.header().findTypedAttribute<Imf::ChromaticitiesAttribute>(
                "chromaticities");
        if (chromaticitiesAttrib != nullptr) {
            Imf::Chromaticities c = chromaticitiesAttrib->value();
            const RGBColorSpace *cs = RGBColorSpace::Lookup(
                Point2f(c.red.x, c.red.y), Point2f(c.green.x, c.green.y),
                Point2f(c.blue.x, c.blue.y), Point2f(c.white.x, c.white.y));
            if (!cs) {
                Warning("Couldn't find supported color space that matches "
                        "chromaticities: "
                        "r (%f, %f) g (%f, %f) b (%f, %f), w (%f, %f). Using sRGB.",
                        c.red.x, c.red.y, c.green.x, c.green.y, c.blue.x, c.blue.y,
                        c.white.x, c.white.y);
                metadata.colorSpace = RGBColorSpace::sRGB;
            } else
                metadata.colorSpace = cs;
        }

        int width = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;

        std::vector<std::string> channelNames;
        int nChannels = 0;
        Imf::PixelType pixelType;
        const Imf::ChannelList &channels = file.header().channels();
        for (auto iter = channels.begin(); iter != channels.end(); ++iter) {
            if (nChannels++ == 0)
                pixelType = iter.channel().type;
            else {
                // TODO: someday handle mixed types but seems like a
                // bother...
                if (pixelType != iter.channel().type)
                    LOG_FATAL("ReadEXR() doesn't currently support images with "
                              "multiple channel types.");
            }
            channelNames.push_back(iter.name());
        }

        CHECK(pixelType == Imf::HALF || pixelType == Imf::FLOAT);
        Image image(pixelType == Imf::HALF ? PixelFormat::Half : PixelFormat::Float,
                    {width, height}, channelNames, nullptr, alloc);
        file.setFrameBuffer(imageToFrameBuffer(image, image.AllChannelsDesc(), dw));
        file.readPixels(dw.min.y, dw.max.y);

        LOG_VERBOSE("Read EXR image %s (%d x %d)", name, width, height);
        return ImageAndMetadata{std::move(image), metadata};
    } catch (const std::exception &e) {
        ErrorExit("Unable to read image file \"%s\": %s", name, e.what());
    }

    return {};
}

bool Image::WriteEXR(const std::string &name, const ImageMetadata &metadata) const {
    if (Is8Bit(format))
        return ConvertToFormat(PixelFormat::Half).WriteEXR(name, metadata);
    CHECK(Is16Bit(format) || Is32Bit(format));

    try {
        Imath::Box2i displayWindow, dataWindow;
        if (metadata.fullResolution)
            // Agan, -1 offsets to handle inclusive indexing in OpenEXR...
            displayWindow = {Imath::V2i(0, 0),
                             Imath::V2i(metadata.fullResolution->x - 1,
                                        metadata.fullResolution->y - 1)};
        else
            displayWindow = {Imath::V2i(0, 0),
                             Imath::V2i(resolution.x - 1, resolution.y - 1)};

        if (metadata.pixelBounds)
            dataWindow = {
                Imath::V2i(metadata.pixelBounds->pMin.x, metadata.pixelBounds->pMin.y),
                Imath::V2i(metadata.pixelBounds->pMax.x - 1,
                           metadata.pixelBounds->pMax.y - 1)};
        else
            dataWindow = {Imath::V2i(0, 0),
                          Imath::V2i(resolution.x - 1, resolution.y - 1)};

        Imf::FrameBuffer fb = imageToFrameBuffer(*this, AllChannelsDesc(), dataWindow);

        Imf::Header header(displayWindow, dataWindow);
        for (auto iter = fb.begin(); iter != fb.end(); ++iter)
            header.channels().insert(iter.name(), iter.slice().type);

        if (metadata.renderTimeSeconds)
            header.insert("renderTimeSeconds",
                          Imf::FloatAttribute(*metadata.renderTimeSeconds));
        if (metadata.cameraFromWorld) {
            float m[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] = (*metadata.cameraFromWorld)[i][j];
            header.insert("worldToCamera", Imf::M44fAttribute(m));
        }
        if (metadata.NDCFromWorld) {
            float m[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] = (*metadata.NDCFromWorld)[i][j];
            header.insert("worldToNDC", Imf::M44fAttribute(m));
        }
        if (metadata.samplesPerPixel)
            header.insert("samplesPerPixel",
                          Imf::IntAttribute(*metadata.samplesPerPixel));
        if (metadata.MSE)
            header.insert("MSE", Imf::FloatAttribute(*metadata.MSE));
        for (const auto &iter : metadata.stringVectors)
            header.insert(iter.first, Imf::StringVectorAttribute(iter.second));

        // The OpenEXR spec says that the default is sRGB if no
        // chromaticities are provided.  It should be innocuous to write
        // the sRGB primaries anyway, but for completely indecipherable
        // reasons, OSX's Preview.app decides to gamma correct the pixels
        // in EXR files if it finds primaries.  So, we don't write them in
        // that case in the interests of nicer looking images on the
        // screen.
        if (*metadata.GetColorSpace() != *RGBColorSpace::sRGB) {
            const RGBColorSpace &cs = *metadata.GetColorSpace();
            Imf::Chromaticities chromaticities(
                Imath::V2f(cs.r.x, cs.r.y), Imath::V2f(cs.g.x, cs.g.y),
                Imath::V2f(cs.b.x, cs.b.y), Imath::V2f(cs.w.x, cs.w.y));
            header.insert("chromaticities", Imf::ChromaticitiesAttribute(chromaticities));
        }

        Imf::OutputFile file(name.c_str(), header);
        file.setFrameBuffer(fb);
        file.writePixels(resolution.y);
    } catch (const std::exception &exc) {
        Error("%s: error writing EXR: %s", name.c_str(), exc.what());
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////
// PNG Function Definitions

static ImageAndMetadata ReadPNG(const std::string &name, Allocator alloc,
                                ColorEncodingHandle encoding) {
    std::string contents = ReadFileContents(name);

    if (encoding == nullptr)
        encoding = ColorEncodingHandle::sRGB;

    unsigned width, height;
    LodePNGState state;
    lodepng_state_init(&state);
    unsigned int error = lodepng_inspect(
        &width, &height, &state, (const unsigned char *)contents.data(), contents.size());
    if (error != 0)
        ErrorExit("%s: %s", name, lodepng_error_text(error));

    Image image(alloc);
    switch (state.info_png.color.colortype) {
    case LCT_GREY:
    case LCT_GREY_ALPHA: {
        std::vector<unsigned char> buf;
        int bpp = state.info_png.color.bitdepth == 16 ? 16 : 8;
        error =
            lodepng::decode(buf, width, height, (const unsigned char *)contents.data(),
                            contents.size(), LCT_GREY, bpp);
        if (error != 0)
            ErrorExit("%s: %s", name, lodepng_error_text(error));

        if (state.info_png.color.bitdepth == 16) {
            image = Image(PixelFormat::Half, Point2i(width, height), {"Y"});
            auto bufIter = buf.begin();
            for (unsigned int y = 0; y < height; ++y)
                for (unsigned int x = 0; x < width; ++x, bufIter += 2) {
                    // Convert from little endian.
                    Float v = (((int)bufIter[0] << 8) + (int)bufIter[1]) / 65535.f;
                    v = encoding.ToFloatLinear(v);
                    image.SetChannel(Point2i(x, y), 0, v);
                }
            CHECK(bufIter == buf.end());
        } else {
            image = Image(PixelFormat::U256, Point2i(width, height), {"Y"}, encoding);
            std::copy(buf.begin(), buf.end(), (uint8_t *)image.RawPointer({0, 0}));
        }
        return ImageAndMetadata{image, ImageMetadata()};
    }
    default: {
        std::vector<unsigned char> buf;
        int bpp = state.info_png.color.bitdepth == 16 ? 16 : 8;
        bool hasAlpha = (state.info_png.color.colortype == LCT_RGBA);
        // Force RGB if it's palletted or whatever.
        error =
            lodepng::decode(buf, width, height, (const unsigned char *)contents.data(),
                            contents.size(), hasAlpha ? LCT_RGBA : LCT_RGB, bpp);
        if (error != 0)
            ErrorExit("%s: %s", name, lodepng_error_text(error));

        ImageMetadata metadata;
        metadata.colorSpace = RGBColorSpace::sRGB;
        if (state.info_png.color.bitdepth == 16) {
            if (hasAlpha) {
                image = Image(PixelFormat::Half, Point2i(width, height),
                              {"R", "G", "B", "A"});
                auto bufIter = buf.begin();
                for (unsigned int y = 0; y < height; ++y)
                    for (unsigned int x = 0; x < width; ++x, bufIter += 8) {
                        CHECK(bufIter < buf.end());
                        // Convert from little endian.
                        Float rgba[4] = {
                            (((int)bufIter[0] << 8) + (int)bufIter[1]) / 65535.f,
                            (((int)bufIter[2] << 8) + (int)bufIter[3]) / 65535.f,
                            (((int)bufIter[4] << 8) + (int)bufIter[5]) / 65535.f,
                            (((int)bufIter[6] << 8) + (int)bufIter[7]) / 65535.f};
                        for (int c = 0; c < 4; ++c) {
                            rgba[c] = encoding.ToFloatLinear(rgba[c]);
                            image.SetChannel(Point2i(x, y), c, rgba[c]);
                        }
                    }
                CHECK(bufIter == buf.end());
            } else {
                image = Image(PixelFormat::Half, Point2i(width, height), {"R", "G", "B"});
                auto bufIter = buf.begin();
                for (unsigned int y = 0; y < height; ++y)
                    for (unsigned int x = 0; x < width; ++x, bufIter += 6) {
                        CHECK(bufIter < buf.end());
                        // Convert from little endian.
                        Float rgb[3] = {
                            (((int)bufIter[0] << 8) + (int)bufIter[1]) / 65535.f,
                            (((int)bufIter[2] << 8) + (int)bufIter[3]) / 65535.f,
                            (((int)bufIter[4] << 8) + (int)bufIter[5]) / 65535.f};
                        for (int c = 0; c < 3; ++c) {
                            rgb[c] = encoding.ToFloatLinear(rgb[c]);
                            image.SetChannel(Point2i(x, y), c, rgb[c]);
                        }
                    }
                CHECK(bufIter == buf.end());
            }
        } else if (hasAlpha) {
            image = Image(PixelFormat::U256, Point2i(width, height), {"R", "G", "B", "A"},
                          encoding);
            std::copy(buf.begin(), buf.end(), (uint8_t *)image.RawPointer({0, 0}));
        } else {
            image = Image(PixelFormat::U256, Point2i(width, height), {"R", "G", "B"},
                          encoding);
            std::copy(buf.begin(), buf.end(), (uint8_t *)image.RawPointer({0, 0}));
        }
        return ImageAndMetadata{image, metadata};
    }
    }
}

Image Image::SelectChannels(const ImageChannelDesc &desc, Allocator alloc) const {
    std::vector<std::string> descChannelNames;
    // TODO: descChannelNames = ChannelNames(desc)
    for (size_t i = 0; i < desc.offset.size(); ++i)
        descChannelNames.push_back(channelNames[desc.offset[i]]);

    Image image(format, resolution, descChannelNames, encoding, alloc);
    for (int y = 0; y < resolution.y; ++y)
        for (int x = 0; x < resolution.x; ++x)
            image.SetChannels({x, y}, GetChannels({x, y}, desc));
    return image;
}

Image Image::Crop(const Bounds2i &bounds, Allocator alloc) const {
    CHECK_GT(bounds.Area(), 0);
    CHECK(bounds.pMin.x >= 0 && bounds.pMin.y >= 0);
    Image image(format, Point2i(bounds.pMax - bounds.pMin), channelNames, encoding,
                alloc);
    for (Point2i p : bounds)
        for (int c = 0; c < NChannels(); ++c)
            image.SetChannel(Point2i(p - bounds.pMin), c, GetChannel(p, c));
    return image;
}

std::string Image::ToString() const {
    return StringPrintf("[ Image format: %s resolution: %s channelNames: %s "
                        "encoding: %s ]",
                        format, resolution, channelNames,
                        encoding ? encoding.ToString().c_str() : "(nullptr)");
}

bool Image::WritePNG(const std::string &name, const ImageMetadata &metadata) const {
    unsigned int error = 0;
    int nOutOfGamut = 0;

    if (format == PixelFormat::U256) {
        if (NChannels() == 1)
            error = lodepng_encode_file(name.c_str(), p8.data(), resolution.x,
                                        resolution.y, LCT_GREY, 8 /* bitdepth */);
        else if (NChannels() == 3)
            // TODO: it would be nice to store the color encoding used in the
            // PNG metadata...
            error = lodepng_encode24_file(name.c_str(), p8.data(), resolution.x,
                                          resolution.y);
        else
            LOG_FATAL("Unhandled channel count in WritePNG(): %d", NChannels());
    } else if (NChannels() == 3) {
        // It may not actually be RGB, but that's what PNG's going to
        // assume..
        std::unique_ptr<uint8_t[]> rgb8 =
            std::make_unique<uint8_t[]>(3 * resolution.x * resolution.y);
        for (int y = 0; y < resolution.y; ++y)
            for (int x = 0; x < resolution.x; ++x)
                for (int c = 0; c < 3; ++c) {
                    Float dither = -.5f + BlueNoise(c, x, y);
                    Float v = GetChannel({x, y}, c);
                    if (v < 0 || v > 1)
                        ++nOutOfGamut;
                    rgb8[3 * (y * resolution.x + x) + c] = LinearToSRGB8(v, dither);
                }

        error =
            lodepng_encode24_file(name.c_str(), rgb8.get(), resolution.x, resolution.y);
    } else if (NChannels() == 1) {
        std::unique_ptr<uint8_t[]> y8 =
            std::make_unique<uint8_t[]>(resolution.x * resolution.y);
        for (int y = 0; y < resolution.y; ++y)
            for (int x = 0; x < resolution.x; ++x) {
                Float dither = -.5f + BlueNoise(0, x, y);
                Float v = GetChannel({x, y}, 0);
                if (v < 0 || v > 1)
                    ++nOutOfGamut;
                y8[y * resolution.x + x] = LinearToSRGB8(v, dither);
            }

        error = lodepng_encode_file(name.c_str(), y8.get(), resolution.x, resolution.y,
                                    LCT_GREY, 8 /* bitdepth */);
    }

    if (nOutOfGamut > 0)
        Warning("%s: %d out of gamut pixel channels clamped to [0,1].", name,
                nOutOfGamut);

    if (error != 0) {
        Error("Error writing PNG \"%s\": %s", name, lodepng_error_text(error));
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////
// PFM Function Definitions

/*
 * PFM reader/writer code courtesy Jiawen "Kevin" Chen
 * (http://people.csail.mit.edu/jiawen/)
 */

static constexpr bool hostLittleEndian =
#if defined(__BYTE_ORDER__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    true
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    false
#else
#error "__BYTE_ORDER__ defined but has unexpected value"
#endif
#else
#if defined(__LITTLE_ENDIAN__) || defined(__i386__) || defined(__x86_64__) || \
    defined(_WIN32) || defined(WIN32)
    true
#elif defined(__BIG_ENDIAN__)
    false
#elif defined(__sparc) || defined(__sparc__)
    false
#else
#error "Can't detect machine endian-ness at compile-time."
#endif
#endif
    ;

#define BUFFER_SIZE 80

static inline int isWhitespace(char c) {
    return static_cast<int>(c == ' ' || c == '\n' || c == '\t');
}

// Reads a "word" from the fp and puts it into buffer and adds a null
// terminator.  i.e. it keeps reading until whitespace is reached.  Returns
// the number of characters read *not* including the whitespace, and
// returns -1 on an error.
static int readWord(FILE *fp, char *buffer, int bufferLength) {
    int n;
    int c;

    if (bufferLength < 1)
        return -1;

    n = 0;
    c = fgetc(fp);
    while (c != EOF && (isWhitespace(c) == 0) && n < bufferLength) {
        buffer[n] = c;
        ++n;
        c = fgetc(fp);
    }

    if (n < bufferLength) {
        buffer[n] = '\0';
        return n;
    }

    return -1;
}

static ImageAndMetadata ReadPFM(const std::string &filename, Allocator alloc) {
    pstd::vector<float> rgb32(alloc);
    char buffer[BUFFER_SIZE];
    unsigned int nFloats;
    int nChannels, width, height;
    float scale;
    bool fileLittleEndian;
    ImageMetadata metadata;

    FILE *fp = fopen(filename.c_str(), "rb");
    if (fp == nullptr)
        ErrorExit("%s: unable to open PFM file", filename);

    // read either "Pf" or "PF"
    if (readWord(fp, buffer, BUFFER_SIZE) == -1)
        ErrorExit("%s: unable to read PFM file", filename);

    if (strcmp(buffer, "Pf") == 0)
        nChannels = 1;
    else if (strcmp(buffer, "PF") == 0)
        nChannels = 3;
    else
        ErrorExit("%s: unable to decode PFM file type \"%c%c\"", filename, buffer[0],
                  buffer[1]);

    // read the rest of the header
    // read width
    if (readWord(fp, buffer, BUFFER_SIZE) == -1)
        goto fail;
    if (!Atoi(buffer, &width))
        ErrorExit("%s: unable to decode width \"%s\"", filename, buffer);

    // read height
    if (readWord(fp, buffer, BUFFER_SIZE) == -1)
        goto fail;
    if (!Atoi(buffer, &height))
        ErrorExit("%s: unable to decode height \"%s\"", filename, buffer);

    // read scale
    if (readWord(fp, buffer, BUFFER_SIZE) == -1)
        goto fail;
    if (!Atof(buffer, &scale))
        ErrorExit("%s: unable to decode scale \"%s\"", filename, buffer);

    // read the data
    nFloats = nChannels * width * height;
    rgb32.resize(nFloats);
    for (int y = height - 1; y >= 0; --y)
        if (fread(&rgb32[nChannels * y * width], sizeof(float), nChannels * width, fp) !=
            nChannels * width)
            goto fail;

    // apply endian conversian and scale if appropriate
    fileLittleEndian = (scale < 0.f);
    if (hostLittleEndian ^ fileLittleEndian) {
        uint8_t bytes[4];
        for (unsigned int i = 0; i < nFloats; ++i) {
            memcpy(bytes, &rgb32[i], 4);
            pstd::swap(bytes[0], bytes[3]);
            pstd::swap(bytes[1], bytes[2]);
            memcpy(&rgb32[i], bytes, 4);
        }
    }
    if (std::abs(scale) != 1.f)
        for (unsigned int i = 0; i < nFloats; ++i)
            rgb32[i] *= std::abs(scale);

    fclose(fp);
    LOG_VERBOSE("Read PFM image %s (%d x %d)", filename, width, height);
    metadata.colorSpace = RGBColorSpace::sRGB;
    if (nChannels == 1)
        return ImageAndMetadata{Image(std::move(rgb32), {width, height}, {"Y"}),
                                metadata};
    else
        return ImageAndMetadata{Image(std::move(rgb32), {width, height}, {"R", "G", "B"}),
                                metadata};

fail:
    if (fp != nullptr)
        fclose(fp);
    ErrorExit("%s: premature end of file in PFM file", filename);
}

static ImageAndMetadata ReadHDR(const std::string &filename, Allocator alloc) {
    int x, y, n;
    float *data = stbi_loadf(filename.c_str(), &x, &y, &n, 0);
    if (!data)
        ErrorExit("%s: %s", filename, stbi_failure_reason());

    pstd::vector<float> pixels(data, data + x * y * n, alloc);
    stbi_image_free(data);

    switch (n) {
    case 1:
        return ImageAndMetadata{Image(std::move(pixels), {x, y}, {"Y"}), ImageMetadata()};
    case 2: {
        Image image(std::move(pixels), {x, y}, {"Y", "A"});
        return ImageAndMetadata{image.SelectChannels(image.GetChannelDesc({"Y"})),
                                ImageMetadata()};
    }
    case 3:
        return ImageAndMetadata{Image(std::move(pixels), {x, y}, {"R", "G", "B"}),
                                ImageMetadata()};
    case 4: {
        Image image(std::move(pixels), {x, y}, {"R", "G", "B", "A"});
        return ImageAndMetadata{
            image.SelectChannels(image.GetChannelDesc({"R", "G", "B"})), ImageMetadata()};
    }
    default:
        ErrorExit("%s: %d channel image unsupported.", filename, n);
    }
}

bool Image::WritePFM(const std::string &filename, const ImageMetadata &metadata) const {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr) {
        Error("Unable to open output PFM file \"%s\"", filename);
        return false;
    }

    std::unique_ptr<float[]> scanline = std::make_unique<float[]>(3 * resolution.x);
    float scale;

    // only write 3 channel PFMs here...
    if (fprintf(fp, "PF\n") < 0)
        goto fail;

    // write the width and height, which must be positive
    if (fprintf(fp, "%d %d\n", resolution.x, resolution.y) < 0)
        goto fail;

    // write the scale, which encodes endianness
    scale = hostLittleEndian ? -1.f : 1.f;
    if (fprintf(fp, "%f\n", scale) < 0)
        goto fail;

    // write the data from bottom left to upper right as specified by
    // http://netpbm.sourceforge.net/doc/pfm.html
    // The raster is a sequence of pixels, packed one after another, with no
    // delimiters of any kind. They are grouped by row, with the pixels in each
    // row ordered left to right and the rows ordered bottom to top.
    for (int y = resolution.y - 1; y >= 0; y--) {
        for (int x = 0; x < resolution.x; ++x) {
            if (NChannels() == 1) {
                Float v = GetChannel({x, y}, 0);
                scanline[3 * x] = scanline[3 * x + 1] = scanline[3 * x + 2] = v;
            } else {
                CHECK_EQ(3, NChannels());
                for (int c = 0; c < 3; ++c)
                    // Assign element-wise in case Float is typedefed as
                    // 'double'.
                    scanline[3 * x + c] = GetChannel({x, y}, c);
            }
        }
        if (fwrite(&scanline[0], sizeof(float), 3 * resolution.x, fp) <
            (size_t)(3 * resolution.x))
            goto fail;
    }

    fclose(fp);
    return true;

fail:
    Error("Error writing PFM file \"%s\"", filename);
    fclose(fp);
    return false;
}

}  // namespace pbrt
