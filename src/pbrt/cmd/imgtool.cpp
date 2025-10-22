// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/filters.h>
#include <pbrt/options.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#ifndef __HIP_PLATFORM_AMD__
#include <pbrt/gpu/optix/denoiser.h>
#include <pbrt/gpu/optix/scaler.h>
#endif // __HIP_PLATFORM_AMD__
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/util/args.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/string.h>
#include <pbrt/util/vecmath.h>

extern "C" {
#include <skymodel/ArHosekSkyModel.h>
}

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include <flip.h>

using namespace pbrt;

struct CommandUsage {
    std::string usage;
    std::string description;
    std::string options;
};

static std::map<std::string, CommandUsage> commandUsage = {
    {"assemble",
     {"assemble [options] <filenames...>",
      "Assemble multiple images representing cropped regions of a larger\n"
      "    image into a single composite image.",
      std::string(R"(
    --outfile <name>   Output image filename.
)")}},
    {"average",
     {"average [options] <filename base>",
      "Generate an image that is the result of averaging multiple images.\n"
      "    The constituent images are specified with a path that gives their\n"
      "    prefix.",
      std::string(R"(
    --outfile <name>   Output image filename.
)")}},
    {"cat",
     {"cat [options] <filename>",
      "Print the pixel values of the specified image to standard output.", std::string(R"(
    --csv              Output pixel values in CSV format.
    --list             Output pixel values in a brace-delimited list
                       (Mathematica-compatible).
    --sort             Sort output by pixel luminance.
)")}},
    {"bloom",
     {"bloom [options] <filename>",
      "Apply a bloom effect to the specified image where pixels that are\n"
      "    brighter than a threshold are blurred and added to nearby pixels.",
      std::string(R"(
    --iterations <n>   Number of filtering iterations used to generate the bloom
                       image. Default: 5
    --level <n>        Minimum RGB value for a pixel for it to contribute to bloom.
                       Default: Infinity (i.e., no bloom is applied)
    --outfile <name>   Output image filename.
    --scale <s>        Amount by which the bloom image is scaled before being
                       added to the original image. Default: 0.3
    --width <w>        Width of Gaussian used to generate bloom images.
                       Default: 15
)")}},
    {"convert",
     {"convert [options] <filename>",
      "Convert an image, possibly going from one format to another.\n"
      "    A variety of image processing operations can be performed as well.",
      std::string(R"(
    --aces-filmic      Apply the ACES filmic s-curve to map values to [0,1].
    --bw               Convert to black and white (average channels)
    --channels <names> Process the provided comma-delineated set of channels.
                       Default: R,G,B.
    --clamp <value>    Rescale pixel components if necessary so that they do not
                       exceed <value>. Default: no clamping.
    --crop <x0,x1,y0,y1> Crop image to the given dimensions. Default: no crop.
    --colorspace <n>   Convert image to given colorspace.
                       (Options: "ACES2065-1", "Rec2020", "DCI-P3", "sRGB")
    --despike <v>      For any pixels with a luminance value greater than <v>,
                       replace the pixel with the median of the 3x3 neighboring
                       pixels. Default: infinity (i.e., disabled).
    --flipy            Flip the image along the y axis
    --fp16             Convert image to fp16 pixel format.
    --gamma <v>        Apply a gamma curve with exponent v. (Default: 1 (none)).
    --maxluminance <n> Luminance value mapped to white by tonemapping.
                       Default: 1
    --outfile <name>   Output image filename.
    --preservecolors   By default, out-of-gammut colors have each component
                       clamped to [0,1] when written to non-HDR formats. With
                       this option enabled, such colors are scaled by their
                       maximum component, which preserves the relative ratio
                       between RGB components.
    --repeatpix <n>    Repeat each pixel value n times in both directions
    --scale <scale>    Scale pixel values by given amount
    --tonemap          Apply tonemapping to the image (Reinhard et al.'s
                       photographic tone mapping operator)
)")}},
    {"diff",
     {"diff [options] <filename>",
      "Compute the difference between an image and a reference image\n"
      "    according to a variety of metrics and optionally output a\n"
      "    difference image.",
      std::string(R"(
    --channels <names> Comma-separated list of channels to include in comparison.
                       Default: R,G,B.
    --crop <x0,x1,y0,y1> Crop images before performing diff.
    --difftol <v>      Acceptable image difference percentage before differences
                       are reported. Default: 0
    --metric <name>    Error metric to use. (Options: "MAE", "MSE", "MRSE", "FLIP")
    --outfile <name>   Filename to use for saving an image that encodes the
                       absolute value of per-pixel differences.
    --reference <name> Filename for reference image
)")}},
#ifdef PBRT_BUILD_GPU_RENDERER
    {"denoise-optix",
     {"denoise-optix [options] <filename>",
      "Denoises the image using NVIDIA's OptiX denoiser which is\n"
      "    based on a deep neural network. The provided image should\n"
      "    be a multi-channel EXR as generated by pbrt's \"gbuffer\" film.",
      std::string(R"( options:
    --outfile <name>   Filename to use for the denoised image.
)")}},
    {"scale-optix",
     {"scale-optix [options] <filename>",
      "2X Scales the image using NVIDIA's OptiX denoiser which is\n"
      "    based on a deep neural network. The provided image should\n"
      "    be a multi-channel EXR as generated by pbrt's \"gbuffer\" film.",
      std::string(R"( options:
    --outfile <name>   Filename to use for the scaled image.
)")}},
#endif  // PBRT_BUILD_GPU_RENDERER
    {"error",
     {"error [options] <filename prefix>",
      "Compute the average error of a set of images with respect to a\n"
      "   reference image. All image files starting with <filename prefix>\n"
      "    are used to compute error.",
      std::string(R"(
    --crop <x0,x1,y0,y1> Crop images before performing diff.
    --errorfile <name>   Output average error image.
    --metric <name>      Error metric to use. (Options: "MAE", MSE", "MRSE")
    --reference <name>   Reference image filename.
)")}},
    {"falsecolor",
     {"falsecolor [options] <filename>",
      "Generate a false color image encoding the magnitude of the\n"
      "    average of the channels in each pixel in the provided image.",
      std::string(R"(
    --maxvalue <v>     Value to map to the last value in the color ramp.
                       (Default: maximum pixel value in the image.)
    --outfile <name>   Filename for output image.
    --plusminus        Visualize green > 0, red < 0.
    --ramp             Generate an image of the color ramp; <filename> is ignored
                       if specified.
)")}},
    {"info",
     {"info <filename>",
      "Prints out image information including resolution, colorspace and pixel stats",
      std::string(R"(
    (No options)
)")}},
    {"makeequiarea",
     {"makeequiarea [options] <filename>",
      "Convert a equirectangular environment map (as used in pbrt-v3)\n"
      "    to an equi-area parameterization (as used in pbrt-v4).",
      std::string(R"(
    --outfile <name>   Filename of lat-long (equirect) environment map image.
    --resolution <n>   Resolution of output image. Default: calculated
                       from resolution of provided image.
)")}},
    {"makeemitters",
     {"makeemitters [options] <filename>",
      "Generate the description of a small quadrilateral emitter for\n"
      "    every pixel in the image.",
      std::string(R"(
    --downsample <n>   Downsample the image by a factor of n in both dimensions
                       (using simple box filtering). Default: 1.
)")}},
    {"makesky",
     {"makesky [options] <filename>",
      "Generate an environment map based on the Hosek-Wilkie sky model.", std::string(R"(
    --albedo <a>       Albedo of ground-plane (range 0-1). Default: 0.5
    --elevation <e>    Elevation of the sun in degrees (range 0-90). Default: 10
    --outfile <name>   Filename to store environment map in.
    --turbidity <t>    Atmospheric turbidity (range 1.7-10). Default: 3
    --resolution <r>   Resolution of generated environment map. Default: 2048
)")}},
    {"splitn",
     {"splitn [options] <filenames>",
      "Generate a composite image from the specified images by taking\n"
      "    a strip of pixels from each image, in the order given. An additional\n"
      "    image of cropped regions of each of the images may be optionally\n"
      "    generated as well.",
      std::string(R"(
    --crop <x,y>       Upper-left coordinate of one of the crops. May be specified
                       multiple times.
    --cropsize <n>     Pixel extent of crops in x and y. Default: 96
    --outfile <name>   Filename to store final image in. Crop images are stored in
                       the file "crop-<name>".
)")}},
    {"scalenormalmap",
     {"scalenormalmap [options] <filename>",
      "Scale the provided normal map by applying the given factor for x and y\n"
      "    and output the resulting normal map.",
      std::string(R"(
    --scale <s>        Scale factor. Default: 1
    --outfile <name>   Filename to store final image in.
)")}},
    {"whitebalance",
     {"whitebalance [options] <filename>",
      "Apply white balancing to the specified image.", std::string(R"(
    --illuminant <n>   Apply white balance for the given standard illuminant
                       (e.g. D65, D50, A, F1, F2, ...)
    --outfile <name>   Filename to store result image.
    --primaries <x,y>  Apply white balance for the primaries (x,y).
    --temperature <T>  Apply white balance for a color temperature T.
)")}},

};

static void usage(const char *cmd, const char *msg = nullptr, ...) {
    if (msg != nullptr) {
        va_list args;
        va_start(args, msg);
        fprintf(stderr, "imgtool %s: ", cmd);
        vfprintf(stderr, msg, args);
        fprintf(stderr, "\n\n");
    }

    auto iter = commandUsage.find(cmd);
    CHECK(iter != commandUsage.end());
    fprintf(stderr, "usage: imgtool %s\n\n", iter->second.usage.c_str());
    if (!iter->second.options.empty())
        fprintf(stderr, "options:%s\n", iter->second.options.c_str());

    exit(1);
}

void help() {
    fprintf(stderr, "usage: imgtool <command> [options]\n\n");
    fprintf(stderr, "where <command> is:\n\n");
    for (const auto &cmd : commandUsage)
        fprintf(stderr, "%s: %s\n\n", cmd.first.c_str(), cmd.second.description.c_str());
    fprintf(stderr, "\n");
    fprintf(stderr, "\"imgtool help <command>\" provides detailed information "
                    "about <command>.\n");
}

int help(std::vector<std::string> args) {
    if (args.empty()) {
        help();
        return 0;
    }
    for (std::string cmd : args) {
        auto iter = commandUsage.find(cmd);
        if (iter == commandUsage.end()) {
            fprintf(stderr, "imgtool help: command \"%s\" not known.\n", cmd.c_str());
            help();
            return 1;
        } else {
            fprintf(stderr, "usage: imgtool %s\n\n", iter->second.usage.c_str());
            fprintf(stderr, "options:%s\n", iter->second.options.c_str());
        }
    }
    return 0;
}

static bool checkImageCompatibility(std::string fn1, const Image &im1,
                                    const RGBColorSpace *cs1, std::string fn2,
                                    const Image &im2, const RGBColorSpace *cs2) {
    if (im1.Resolution() != im2.Resolution()) {
        Error("%s: image resolution (%d, %d) doesn't match \"%s\" (%d, %d).", fn1.c_str(),
              im1.Resolution().x, im1.Resolution().y, fn2.c_str(), im2.Resolution().x,
              im2.Resolution().y);
        return false;
    }
    if (im1.NChannels() != im2.NChannels()) {
        Error("%s: image channel count %d doesn't match \"%s\", %d.", fn1.c_str(),
              im1.NChannels(), fn2.c_str(), im2.NChannels());
        return false;
    }
    if (im1.ChannelNames() != im2.ChannelNames()) {
        auto print = [](const std::vector<std::string> &n) {
            std::string s = n[0];
            for (size_t i = 1; i < n.size(); ++i) {
                s += ", ";
                s += n[i];
            }
            return s;
        };
        Error("%s: warning: image channel names \"%s\" don't match \"%s\" with \"%s\".",
              fn1.c_str(), print(im1.ChannelNames()).c_str(), fn2.c_str(),
              print(im2.ChannelNames()).c_str());
        return false;
    }

    if (*cs1 != *cs2) {
        Error("%s: color space does not match \"%s\".", fn1, fn2);
        return false;
    }

    return true;
}

int makesky(std::vector<std::string> args) {
    std::string outfile;
    Float albedo = 0.5;
    Float turbidity = 3.;
    Float elevation = 10;
    int resolution = 2048;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("makesky", "%s", err.c_str());
            exit(1);
        };
        if (ParseArg(&iter, args.end(), "outfile", &outfile, onError) ||
            ParseArg(&iter, args.end(), "albedo", &albedo, onError) ||
            ParseArg(&iter, args.end(), "turbidity", &turbidity, onError) ||
            ParseArg(&iter, args.end(), "elevation", &elevation, onError) ||
            ParseArg(&iter, args.end(), "resolution", &resolution, onError)) {
            // success
        } else
            onError(StringPrintf("argument %s invalid", *iter));
    }

    if (outfile.empty())
        usage("makesky", "--outfile must be specified");
    if (albedo < 0. || albedo > 1.)
        usage("makesky", "--albedo must be between 0 and 1");
    if (turbidity < 1.7 || turbidity > 10.)
        usage("makesky", "--turbidity must be between 1.7 and 10.");
    if (elevation < 0. || elevation > 90.)
        usage("makesky", "--elevation must be between 0. and 90.");
    elevation = Radians(elevation);
    if (resolution < 1)
        usage("makesky", "--resolution must be >= 1");

    // Vector pointing at the sun. Note that elevation is measured from the
    // horizon--not the zenith, as it is elsewhere in pbrt.
    Vector3f sunDir(0., std::cos(elevation), std::sin(elevation));

    Image img(PixelFormat::Float, {resolution, resolution}, {"R", "G", "B"});

    // They assert wavelengths are in this range...
    int nLambda = 1 + (720 - 320) / 32;
    std::vector<Float> lambda(nLambda, Float(0));
    for (int i = 0; i < nLambda; ++i)
        lambda[i] = Lerp(i / Float(nLambda - 1), 320, 720);

    // Assume a uniform spectral albedo
    ArHosekSkyModelState *skymodel_state =
        arhosekskymodelstate_alloc_init(elevation, turbidity, albedo);

    const RGBColorSpace *colorSpace = RGBColorSpace::ACES2065_1;

    ParallelFor(0, resolution, [&](int64_t start, int64_t end) {
        std::vector<Float> skyv(lambda.size());
        for (int64_t iy = start; iy < end; ++iy) {
            Float y = (iy + 0.5f) / resolution;
            for (int ix = 0; ix < resolution; ++ix) {
                Float x = (ix + 0.5f) / resolution;
                Vector3f v = EqualAreaSquareToSphere({x, y});
                if (v.z <= 0)
                    // downward hemisphere
                    continue;

                Float theta = SphericalTheta(v);

                // Compute the angle between the pixel's direction and the sun
                // direction.
                Float gamma = SafeACos(Dot(v, sunDir));
                DCHECK(gamma >= 0 && gamma <= Pi);

                for (int i = 0; i < lambda.size(); ++i)
                    skyv[i] = arhosekskymodel_solar_radiance(skymodel_state, theta, gamma,
                                                             lambda[i]);

                PiecewiseLinearSpectrum spec(pstd::MakeConstSpan(lambda),
                                             pstd::MakeConstSpan(skyv));
                XYZ xyz = SpectrumToXYZ(&spec);
                RGB rgb = colorSpace->ToRGB(xyz);

                for (int c = 0; c < 3; ++c)
                    img.SetChannel({ix, int(iy)}, c, rgb[c]);
            }
        }
    });

    ImageMetadata metadata;
    metadata.colorSpace = colorSpace;
    std::map<std::string, std::vector<std::string>> stringVectors;
    stringVectors["makesky.albedo"] = {std::to_string(albedo)};
    stringVectors["makesky.elevation"] = {std::to_string(elevation)};
    stringVectors["makesky.turbidity"] = {std::to_string(turbidity)};
    metadata.stringVectors = stringVectors;
    CHECK(img.Write(outfile, metadata));

    return 0;
}

int assemble(std::vector<std::string> args) {
    if (args.empty())
        usage("assemble", "no filenames provided to \"assemble\"?");
    std::string outfile;
    std::vector<std::string> infiles;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("assemble", "%s", err.c_str());
        };
        if (ParseArg(&iter, args.end(), "outfile", &outfile, onError))
            ;  // success
        else if ((*iter)[0] == '-')
            usage("assemble", "%s: unknown command flag", iter->c_str());
        else {
            infiles.push_back(*iter);
        }
    }

    if (outfile.empty())
        usage("assemble", "--outfile not provided for \"assemble\"");

    Image fullImage;
    std::vector<bool> seenPixel;
    int seenMultiple = 0;
    Bounds2i fullBounds;
    const RGBColorSpace *colorSpace = nullptr;
    for (const std::string &file : infiles) {
        if (!HasExtension(file, "exr"))
            usage("assemble", "only EXR images include the image bounding boxes that "
                              "\"assemble\" needs.");

        ImageAndMetadata im = Image::Read(file);
        Image &image = im.image;
        ImageMetadata &metadata = im.metadata;

        if (!metadata.fullResolution) {
            Error("%s: doesn't have full resolution in image metadata. Skipping.", file);
            continue;
        }
        if (!metadata.pixelBounds) {
            Error("%s: doesn't have pixel bounds in image metadata. Skipping.", file);
            continue;
        }

        if (fullImage.Resolution() == Point2i(0, 0)) {
            // First image read.
            fullImage = Image(image.Format(), *metadata.fullResolution,
                              image.ChannelNames(), image.Encoding());
            colorSpace = metadata.GetColorSpace();
            seenPixel.resize(fullImage.Resolution().x * fullImage.Resolution().y);
            fullBounds = Bounds2i({0, 0}, fullImage.Resolution());
        } else {
            // Make sure that this image's info is compatible with the
            // first image's.
            if (*metadata.fullResolution != fullImage.Resolution()) {
                Warning("%s: full resolution (%d, %d) in EXR file doesn't match the full "
                        "resolution of first EXR file (%d, %d). "
                        "Ignoring this file.",
                        file, metadata.fullResolution->x, metadata.fullResolution->y,
                        fullImage.Resolution().x, fullImage.Resolution().y);
                continue;
            }
            if (Union(*metadata.pixelBounds, fullBounds) != fullBounds) {
                Warning("%s: pixel bounds (%d, %d) - (%d, %d) in EXR file isn't inside "
                        "the full image (0, 0) - (%d, %d). "
                        "Ignoring this file.",
                        file, metadata.pixelBounds->pMin.x, metadata.pixelBounds->pMin.y,
                        metadata.pixelBounds->pMax.x, metadata.pixelBounds->pMax.y,
                        fullBounds.pMax.x, fullBounds.pMax.y);
                continue;
            }
            if (fullImage.NChannels() != image.NChannels()) {
                Warning("%s: %d channel image; expecting %d channels.", file,
                        image.NChannels(), fullImage.NChannels());
                continue;
            }
            const RGBColorSpace *cs = metadata.GetColorSpace();
            if (*cs != *colorSpace) {
                Warning(
                    "%s: color space (%s) doesn't match first image's color space (%s).",
                    file, cs->ToString(), colorSpace->ToString());
                continue;
            }
        }

        // Copy pixels.
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Point2i fullp{x + metadata.pixelBounds->pMin.x,
                              y + metadata.pixelBounds->pMin.y};
                CHECK(InsideExclusive(fullp, fullBounds));
                size_t fullOffset = fullp.x + fullp.y * fullImage.Resolution().x;
                if (seenPixel[fullOffset])
                    ++seenMultiple;
                seenPixel[fullOffset] = true;
                for (int c = 0; c < fullImage.NChannels(); ++c)
                    fullImage.SetChannel(fullp, c, image.GetChannel({x, y}, c));
            }
    }

    int unseenPixels = 0;
    for (int y = 0; y < fullImage.Resolution().y; ++y)
        for (int x = 0; x < fullImage.Resolution().x; ++x)
            if (!seenPixel[y * fullImage.Resolution().x + x])
                ++unseenPixels;

    if (seenMultiple > 0)
        Warning("%s: %d pixels present in multiple images.", outfile, seenMultiple);
    if (unseenPixels > 0)
        Warning("%s: %d pixels not present in any images.", outfile, unseenPixels);

    ImageMetadata outMetadata;
    outMetadata.colorSpace = colorSpace;

    fullImage.Write(outfile, outMetadata);

    return 0;
}

int splitn(std::vector<std::string> args) {
    if (args.empty())
        usage("splitn", "no filenames provided to \"splitn\"?");
    std::string outfile;
    std::vector<std::string> infiles;
    std::vector<Bounds2i> crops;
    int cropSize = 96;

    std::string crop;
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) { usage("splitn", "%s", err.c_str()); };
        if (ParseArg(&iter, args.end(), "outfile", &outfile, onError) ||
            ParseArg(&iter, args.end(), "cropsize", &cropSize, onError))
            ;  // success
        else if (ParseArg(&iter, args.end(), "crop", &crop, onError)) {
            std::vector<int> c = SplitStringToInts(crop, ',');
            if (c.size() != 2)
                usage("splitn", "2 values not provided to --crop");
            crops.push_back(Bounds2i({c[0], c[1]}, {c[0] + cropSize, c[1] + cropSize}));
        } else if ((*iter)[0] == '-')
            usage("splitn", "%s: unknown command flag", iter->c_str());
        else {
            infiles.push_back(*iter);
        }
    }

    if (outfile.empty())
        usage("splitn", "--outfile not provided for \"splitn\"");

    std::vector<Image> images;
    const RGBColorSpace *colorSpace = nullptr;
    for (const std::string &file : infiles) {
        ImageAndMetadata im = Image::Read(file);
        Image &image = im.image;

        if (!colorSpace)
            colorSpace = im.metadata.GetColorSpace();
        else if (!checkImageCompatibility(infiles[0], images.front(), colorSpace, file,
                                          image, im.metadata.GetColorSpace()))
            return false;

        images.push_back(image);
    }

    Image result(images.front().Format(), images.front().Resolution(),
                 images.front().ChannelNames(), images.front().Encoding());
    float m = 15.f;
    int pad = 6;
    for (int y = 0; y < result.Resolution().y; ++y) {
        int x = 0;
        for (int im = 0; im < images.size(); ++im) {
            if (im == images.size() - 1)
                for (; x < result.Resolution().x; ++x)
                    for (int c = 0; c < result.NChannels(); ++c)
                        result.SetChannel({x, y}, c, images[im].GetChannel({x, y}, c));
            else {
                int x1 = (float(im + 1) / float(images.size()) * result.Resolution().x +
                          (2 * (float(y) / float(result.Resolution().y)) - 1) *
                              result.Resolution().x / -m);

                for (; x < x1 - pad / 2; ++x)
                    for (int c = 0; c < result.NChannels(); ++c)
                        result.SetChannel({x, y}, c, images[im].GetChannel({x, y}, c));
                for (; x < x1 + pad / 2; ++x)
                    for (int c = 0; c < result.NChannels(); ++c)
                        result.SetChannel({x, y}, c, 0);
            }
        }
    }

    // Crops
    RGB edges[] = {RGB(0.8, .15, .15), RGB(.15, 0.8, .15), RGB(.15, .15, 0.8)};
    int borderWidth = 5;
    int xCropRes =
        (cropSize + 2 * borderWidth) * images.size() + pad * (images.size() - 1);
    int yCropRes = (cropSize + 2 * borderWidth) * crops.size() + pad * (crops.size() - 1);
    if (xCropRes < 1)
        xCropRes = 1;
    if (yCropRes < 1)
        yCropRes = 1;
    Image cropsImage(PixelFormat::Float, {xCropRes, yCropRes}, {"R", "G", "B"});
    for (int y = 0; y < yCropRes; ++y)
        for (int x = 0; x < xCropRes; ++x)
            for (int c = 0; c < 3; ++c)
                cropsImage.SetChannel({x, y}, c, 1.f);

    auto drawBox = [](Bounds2i b, int borderWidth, RGB rgb, Image &result) {
        ImageChannelDesc desc = result.GetChannelDesc({"R", "G", "B"});
        CHECK(bool(desc));

        for (int y = b.pMin.y - borderWidth; y < b.pMin.y; ++y)
            for (int x = b.pMin.x - borderWidth; x < b.pMax.x + borderWidth; ++x)
                for (int ch = 0; ch < 3; ++ch)
                    result.SetChannels({x, y}, desc, {rgb.r, rgb.g, rgb.b});
        for (int y = b.pMax.y; y < b.pMax.y + borderWidth; ++y)
            for (int x = b.pMin.x - borderWidth; x < b.pMax.x + borderWidth; ++x)
                for (int ch = 0; ch < 3; ++ch)
                    result.SetChannels({x, y}, desc, {rgb.r, rgb.g, rgb.b});
        for (int x = b.pMin.x - borderWidth; x < b.pMin.x; ++x)
            for (int y = b.pMin.y; y < b.pMax.y; ++y)
                for (int ch = 0; ch < 3; ++ch)
                    result.SetChannels({x, y}, desc, {rgb.r, rgb.g, rgb.b});
        for (int x = b.pMax.x; x < b.pMax.x + borderWidth; ++x)
            for (int y = b.pMin.y; y < b.pMax.y; ++y)
                for (int ch = 0; ch < 3; ++ch)
                    result.SetChannels({x, y}, desc, {rgb.r, rgb.g, rgb.b});
    };

    CHECK_LE(crops.size(), sizeof(edges) / sizeof(edges[0]));
    ImageChannelDesc cropDesc = cropsImage.GetChannelDesc({"R", "G", "B"});
    CHECK(bool(cropDesc));

    for (int c = 0; c < crops.size(); ++c) {
        Bounds2i crop = crops[c];
        drawBox(crop, borderWidth, edges[c], result);

        // extract crops from images
        for (int i = 0; i < images.size(); ++i) {
            // output start point
            int y0 = c * (cropSize + 2 * borderWidth + pad);
            int x0 = i * (cropSize + 2 * borderWidth + pad);
            drawBox(Bounds2i({x0 + borderWidth, y0 + borderWidth},
                             {x0 + borderWidth + cropSize, y0 + borderWidth + cropSize}),
                    borderWidth, edges[c], cropsImage);

            ImageChannelDesc imageDesc = images[i].GetChannelDesc({"R", "G", "B"});
            CHECK(bool(imageDesc));
            for (Point2i p : crop) {
                Point2i pOut =
                    Point2i(x0 + borderWidth, y0 + borderWidth) + (p - crop.pMin);
                cropsImage.SetChannels(pOut, cropDesc,
                                       images[i].GetChannels(p, imageDesc));
            }
        }
    }

    // TODO: we could try to be more clever about carrying through more
    // metadata, when applicable.
    ImageMetadata outMetadata;
    outMetadata.colorSpace = colorSpace;

    result.Write(outfile, outMetadata);
    if (crops.size())
        cropsImage.Write(std::string("crops-") + outfile, outMetadata);

    return 0;
}

int scalenormalmap(std::vector<std::string> args) {
    std::string filename, outfile;
    Float scale = 1;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("error", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "scale", &scale, onError) ||
            ParseArg(&iter, args.end(), "outfile", &outfile, onError)) {
            // success
        } else if (filename.empty() && (*iter)[0] != '-') {
            filename = *iter;
        } else
            usage("error", "%s: unknown argument", iter->c_str());
    }

    if (filename.empty())
        usage("error", "Must provide base filename.");

    ImageAndMetadata im = Image::Read(filename);
    const Image &image = im.image;
    ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!rgbDesc) {
        Error("%s: doesn't have R, G, B channels.", filename);
        return 1;
    }

    Image scaledImage = image;
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x) {
            ImageChannelValues rgb = image.GetChannels({x, y}, rgbDesc);
            for (int c = 0; c < 3; ++c)
                rgb[c] = 2 * rgb[c] - 1;

            rgb[0] *= scale;
            rgb[1] *= scale;
            rgb[2] = SafeSqrt(1 - Sqr(rgb[0]) - Sqr(rgb[1]));

            for (int c = 0; c < 3; ++c)
                rgb[c] = (rgb[c] + 1) / 2;

            scaledImage.SetChannels({x, y}, rgbDesc, rgb);
        }

    if (!scaledImage.Write(outfile, im.metadata))
        return 1;

    return 0;
}

int cat(std::vector<std::string> args) {
    if (args.empty())
        usage("cat", "no filenames provided to \"cat\"?");
    bool sort = false;
    bool csv = false;
    bool list = false;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if (*iter == "--sort" || *iter == "-sort") {
            sort = !sort;
            continue;
        } else if (*iter == "--csv" || *iter == "-csv") {
            csv = !csv;
            continue;
        } else if (*iter == "--list" || *iter == "-list") {
            list = !list;
            continue;
        }

        if (sort && csv) {
            fprintf(stderr, "imgtool: --sort and --csv don't make sense to use "
                            "together.\n");
            return 1;
        }
        if (sort && list) {
            fprintf(stderr, "imgtool: --sort and --list don't make sense to "
                            "use together.\n");
            return 1;
        }

        ImageAndMetadata im = Image::Read(*iter);
        ImageMetadata &metadata = im.metadata;
        Image &image = im.image;

        Bounds2i pixelBounds =
            metadata.pixelBounds.value_or(Bounds2i({0, 0}, image.Resolution()));
        if (sort) {
            std::vector<std::pair<Point2i, ImageChannelValues>> sorted;
            sorted.reserve(pixelBounds.Area());
            for (Point2i p : pixelBounds) {
                ImageChannelValues v = image.GetChannels(
                    {p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y});
                sorted.push_back(std::make_pair(p, v));
            }

            std::sort(sorted.begin(), sorted.end(),
                      [](const std::pair<Point2i, ImageChannelValues> &a,
                         const std::pair<Point2i, ImageChannelValues> &b) {
                          return a.second.Average() < b.second.Average();
                      });
            for (const auto &v : sorted) {
                const ImageChannelValues &values = v.second;
                if (!csv)
                    printf("(%d, %d): ", v.first.x, v.first.y);
                for (size_t i = 0; i < values.size(); ++i)
                    Printf("%f%c", values[i], (i == values.size() - 1) ? '\n' : ',');
            }
        } else {
            if (list) {
                CHECK_EQ(image.NChannels(), 1);
                for (int y = pixelBounds.pMin.y; y < pixelBounds.pMax.y; ++y) {
                    for (int x = pixelBounds.pMin.x; x < pixelBounds.pMax.x; ++x)
                        printf("%f ",
                               image.GetChannel(
                                   {x - pixelBounds.pMin.x, y - pixelBounds.pMin.y}, 0));
                    printf("\n");
                }
            } else {
                for (Point2i p : pixelBounds) {
                    ImageChannelValues values = image.GetChannels(
                        {p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y});
                    if (!csv)
                        printf("(%d, %d): ", p.x, p.y);
                    for (size_t i = 0; i < values.size(); ++i)
                        Printf("%f%c", values[i], (i == values.size() - 1) ? '\n' : ',');
                }
            }
        }
    }
    return 0;
}

int average(std::vector<std::string> args) {
    std::string avgFile, filenameBase;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("average", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "outfile", &avgFile, onError)) {
            // success
        } else if (filenameBase.empty() && (*iter)[0] != '-') {
            filenameBase = *iter;
        } else
            usage("average", "%s: unknown argument", iter->c_str());
    }

    if (filenameBase.empty())
        usage("average", "must provide base filename.");
    if (avgFile.empty())
        usage("average", "must provide --outfile.");

    std::vector<std::string> filenames = MatchingFilenames(filenameBase);
    if (filenames.empty()) {
        Error("%s: no matching filenames!", filenameBase);
        return 1;
    }

    // Compute average image
    ThreadLocal<Image> avgImages;
    ThreadLocal<const RGBColorSpace *> colorSpaces;
    std::atomic<bool> failed{false};

    ParallelFor(0, filenames.size(), [&](size_t i) {
        ImageAndMetadata imRead = Image::Read(filenames[i]);
        Image &im = imRead.image;

        Image &avg = avgImages.Get();
        if (avg.Resolution() == Point2i(0, 0)) {
            avg = Image(PixelFormat::Float, im.Resolution(), im.ChannelNames());
            colorSpaces.Get() = imRead.metadata.GetColorSpace();
        } else if (!checkImageCompatibility(filenames[i], im,
                                            imRead.metadata.GetColorSpace(), filenames[0],
                                            avg, colorSpaces.Get())) {
            failed = true;
            return;
        }

        for (int y = 0; y < avg.Resolution().y; ++y)
            for (int x = 0; x < avg.Resolution().x; ++x)
                for (int c = 0; c < avg.NChannels(); ++c) {
                    Float v = im.GetChannel({x, y}, c) / filenames.size();
                    if (std::isnan(v))
                        LOG_FATAL("NAN Pixel at %s in %s", Point2f(x, y), filenames[i]);
                    if (std::isinf(v))
                        v = 0;
                    avg.SetChannel({x, y}, c, avg.GetChannel({x, y}, c) + v);
                }
    });

    if (failed)
        return 1;

    // Average per-thread average images
    Image avgImage;
    avgImages.ForAll([&](const Image &im) {
        CHECK_NE(im.Resolution(), Point2i(0, 0));
        if (avgImage.Resolution() == Point2i(0, 0)) {
            // First valid one
            avgImage = im;
        } else {
            for (int y = 0; y < avgImage.Resolution().y; ++y)
                for (int x = 0; x < avgImage.Resolution().x; ++x)
                    for (int c = 0; c < avgImage.NChannels(); ++c) {
                        Float v = im.GetChannel({x, y}, c);
                        if (!std::isinf(v))
                            avgImage.SetChannel({x, y}, c,
                                                avgImage.GetChannel({x, y}, c) + v);
                    }
        }
    });

    ImageMetadata avgMetadata;
    colorSpaces.ForAll([&](const RGBColorSpace *cs) {
        // Will redundantly set it, but whatever.
        avgMetadata.colorSpace = cs;
    });

    if (!avgImage.Write(avgFile, avgMetadata))
        return 1;

    return 0;
}

int error(std::vector<std::string> args) {
    std::string referenceFile, errorFile, metric = "MSE";
    std::string filenameBase;
    std::array<int, 4> cropWindow = {-1, 0, -1, 0};

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("error", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "reference", &referenceFile, onError) ||
            ParseArg(&iter, args.end(), "errorfile", &errorFile, onError) ||
            ParseArg(&iter, args.end(), "metric", &metric, onError) ||
            ParseArg(&iter, args.end(), "crop", pstd::MakeSpan(cropWindow), onError)) {
            // success
        } else if (filenameBase.empty() && (*iter)[0] != '-') {
            filenameBase = *iter;
        } else
            usage("error", "%s: unknown argument", iter->c_str());
    }

    if (filenameBase.empty())
        usage("error", "Must provide base filename.");
    if (metric != "MSE" && metric != "MRSE" && metric != "MAE")
        usage("error", "%s: --metric must be \"MAE\", \"MSE\", or \"MRSE\".",
              metric.c_str());

    std::vector<std::string> filenames = MatchingFilenames(filenameBase);
    if (filenames.empty()) {
        Error("%s: no matching filenames!", filenameBase);
        return 1;
    }

    if (referenceFile.empty())
        usage("error", "must provide --reference file.");
    ImageAndMetadata ref = Image::Read(referenceFile);
    Image &referenceImage = ref.image;

    // If last 2 are negative, they're taken as deltas
    if (cropWindow[1] < 0)
        cropWindow[1] = cropWindow[0] - cropWindow[1];
    if (cropWindow[3] < 0)
        cropWindow[3] = cropWindow[2] - cropWindow[3];
    auto crop = [&cropWindow](Image &image) {
        if (cropWindow[0] >= 0 && cropWindow[2] >= 0)
            image = image.Crop(
                Bounds2i({cropWindow[0], cropWindow[2]}, {cropWindow[1], cropWindow[3]}));
    };

    crop(referenceImage);

    // Compute error and error image
    using MultiChannelVarianceEstimator = std::vector<VarianceEstimator<double>>;
    ThreadLocal<Array2D<MultiChannelVarianceEstimator>> pixelVariances([&]() {
        Array2D<MultiChannelVarianceEstimator> amcve(referenceImage.Resolution().x,
                                                     referenceImage.Resolution().y);
        for (auto &mcve : amcve)
            mcve.resize(referenceImage.NChannels());
        return amcve;
    });

    ThreadLocal<double> sumErrors;
    std::vector<int> spp(filenames.size());
    ParallelFor(0, filenames.size(), [&](size_t i) {
        ImageAndMetadata imRead = Image::Read(filenames[i]);
        Image &im = imRead.image;
        crop(im);

        CHECK(imRead.metadata.samplesPerPixel.has_value());
        spp[i] = *imRead.metadata.samplesPerPixel;

        Image diffImage;
        ImageChannelValues error(referenceImage.NChannels());
        if (metric == "MAE")
            error = im.MAE(im.AllChannelsDesc(), referenceImage, &diffImage);
        else if (metric == "MSE")
            error = im.MSE(im.AllChannelsDesc(), referenceImage, &diffImage);
        else
            error = im.MRSE(im.AllChannelsDesc(), referenceImage, &diffImage);
        sumErrors.Get() += error.Average();

        for (int y = 0; y < im.Resolution().y; ++y)
            for (int x = 0; x < im.Resolution().x; ++x) {
                MultiChannelVarianceEstimator &pixelVariance = pixelVariances.Get()(x, y);
                for (int c = 0; c < im.NChannels(); ++c)
                    if (metric == "MRSE")
                        pixelVariance[c].Add(
                            im.GetChannel({x, y}, c) /
                            (0.01f + referenceImage.GetChannel({x, y}, c)));
                    else
                        pixelVariance[c].Add(im.GetChannel({x, y}, c));
            }
    });

    for (int i = 1; i < filenames.size(); ++i) {
        if (spp[i] != spp[0]) {
            Error("%s: spp %d mismatch. %s has %d.", filenames[i], spp[i], filenames[0],
                  spp[0]);
            return 1;
        }
    }

    double sumError = 0.;
    sumErrors.ForAll([&sumError](double err) { sumError += err; });

    Array2D<MultiChannelVarianceEstimator> pixelVariance(referenceImage.Resolution().x,
                                                         referenceImage.Resolution().y);
    for (auto &mcve : pixelVariance)
        mcve.resize(referenceImage.NChannels());

    pixelVariances.ForAll([&](const auto &pixVar) {
        for (int y = 0; y < referenceImage.Resolution().y; ++y)
            for (int x = 0; x < referenceImage.Resolution().x; ++x)
                for (int c = 0; c < referenceImage.NChannels(); ++c)
                    pixelVariance(x, y)[c].Merge(pixVar(x, y)[c]);
    });

    Image errorImage(PixelFormat::Float,
                     {referenceImage.Resolution().x, referenceImage.Resolution().y},
                     {metric});
    for (int y = 0; y < referenceImage.Resolution().y; ++y)
        for (int x = 0; x < referenceImage.Resolution().x; ++x) {
            Float varSum = 0;
            for (int c = 0; c < referenceImage.NChannels(); ++c)
                varSum += pixelVariance(x, y)[c].Variance();
            errorImage.SetChannel({x, y}, 0, varSum / referenceImage.NChannels());
        }

    // MSE is the average over all of the pixels
    double error = sumError / (filenames.size() - 1);
    printf("%s estimate = %.9g\n", metric.c_str(), error);

    if (!errorFile.empty() && !errorImage.Write(errorFile)) {
        return 1;
    }

    return 0;
}

int diff(std::vector<std::string> args) {
    std::string outFile, imageFile, referenceFile, metric = "MSE";
    std::string channels = "R,G,B";
    std::array<int, 4> cropWindow = {-1, 0, -1, 0};

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("diff", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "outfile", &outFile, onError) ||
            ParseArg(&iter, args.end(), "reference", &referenceFile, onError) ||
            ParseArg(&iter, args.end(), "metric", &metric, onError) ||
            ParseArg(&iter, args.end(), "channels", &channels, onError) ||
            ParseArg(&iter, args.end(), "crop", pstd::MakeSpan(cropWindow), onError)) {
            // success
        } else if ((*iter)[0] == '-') {
            usage("diff", "%s: unknown command flag", iter->c_str());
        } else if (!imageFile.empty()) {
            usage("diff", "%s: excess argument", iter->c_str());
        } else {
            imageFile = *iter;
        }
    }

    if (imageFile.empty())
        usage("diff", "must specify image to compute difference with.");

    if (referenceFile.empty())
        usage("diff", "must specify --reference image");

    if (metric != "MAE" && metric != "MSE" && metric != "MRSE" && metric != "FLIP")
        usage("diff", "%s: --metric must be \"MAE\", \"MSE\", \"MRSE\", or \"FLIP\".",
              metric.c_str());

    ImageAndMetadata refRead = Image::Read(referenceFile);
    Image &refImage = refRead.image;
    const ImageMetadata &refMetadata = refRead.metadata;

    // If last 2 are negative, they're taken as deltas
    if (cropWindow[1] < 0)
        cropWindow[1] = cropWindow[0] - cropWindow[1];
    if (cropWindow[3] < 0)
        cropWindow[3] = cropWindow[2] - cropWindow[3];
    if (cropWindow[0] >= 0 && cropWindow[2] >= 0)
        refImage = refImage.Crop(
            Bounds2i({cropWindow[0], cropWindow[2]}, {cropWindow[1], cropWindow[3]}));

    ImageAndMetadata im = Image::Read(imageFile);
    Image &image = im.image;

    // Select channels
    std::vector<std::string> splitChannels = SplitString(channels, ',');
    ImageChannelDesc refDesc = refImage.GetChannelDesc(splitChannels);
    ImageChannelDesc imgDesc = image.GetChannelDesc(splitChannels);
    if (refDesc.size() != splitChannels.size()) {
        Error("%s: image does not have \"%s\" channels.", referenceFile, channels);
        return 1;
    }
    refImage = refImage.SelectChannels(refDesc);

    if (imgDesc.size() != splitChannels.size()) {
        Error("%s: image does not have \"%s\" channels.", imageFile, channels);
        return 1;
    }
    image = image.SelectChannels(imgDesc);

    // Crop before comparing resolutions.
    if (cropWindow[0] >= 0 && cropWindow[2] >= 0)
        image = image.Crop(
            Bounds2i({cropWindow[0], cropWindow[2]}, {cropWindow[1], cropWindow[3]}));

    if (!checkImageCompatibility(imageFile, image, im.metadata.GetColorSpace(),
                                 referenceFile, refImage, refMetadata.GetColorSpace()))
        return 1;

    // Clamp Infs
    int nClamped = 0, nRefClamped = 0;
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x)
            for (int c = 0; c < image.NChannels(); ++c) {
                if (std::isinf(image.GetChannel({x, y}, c))) {
                    ++nClamped;
                    image.SetChannel({x, y}, c, 0);
                }
                if (std::isinf(refImage.GetChannel({x, y}, c))) {
                    ++nRefClamped;
                    refImage.SetChannel({x, y}, c, 0);
                }
            }
    if (nClamped > 0)
        Warning("%s: clamped %d infinite pixel values.", imageFile, nClamped);
    if (nRefClamped > 0)
        Warning("%s: clamped %d infinite pixel values.", referenceFile, nRefClamped);

    // Image averages. Compute before FLIP potentially goes and clamps things...
    Float refAverage = refImage.Average(refImage.AllChannelsDesc()).Average();
    Float imageAverage = image.Average(image.AllChannelsDesc()).Average();

    Image errorImage;
    ImageChannelValues error(refImage.NChannels());
    if (metric == "MAE")
        error = image.MAE(image.AllChannelsDesc(), refImage, &errorImage);
    else if (metric == "MSE")
        error = image.MSE(image.AllChannelsDesc(), refImage, &errorImage);
    else if (metric == "MRSE")
        error = image.MRSE(image.AllChannelsDesc(), refImage, &errorImage);
    else {
        // FLIP
        if (refImage.NChannels() != 3) {
            Error("%s: only 3 channel images are currently supported for FLIP.",
                  referenceFile);
            return 1;
        }

        errorImage = Image(PixelFormat::Float, image.Resolution(), {"Error"});
        FLIPOptions opt; /* use defaults for now */
        image = image.ConvertToFormat(PixelFormat::Float);
        refImage = refImage.ConvertToFormat(PixelFormat::Float);

        // Clamp to [0,1]...
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                for (int c = 0; c < image.NChannels(); ++c) {
                    image.SetChannel({x, y}, c, Clamp(image.GetChannel({x, y}, c), 0, 1));
                    refImage.SetChannel({x, y}, c,
                                        Clamp(refImage.GetChannel({x, y}, c), 0, 1));
                }

        ComputeFLIPError((float *)image.RawPointer({0, 0}),
                         (float *)refImage.RawPointer({0, 0}),
                         (float *)errorImage.RawPointer({0, 0}), image.Resolution().x,
                         image.Resolution().y, opt);

        for (int c = 0; c < image.NChannels(); ++c)
            error[c] = 0;
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                for (int c = 0; c < image.NChannels(); ++c)
                    error[c] += errorImage.GetChannel({x, y}, 0);
        for (int c = 0; c < image.NChannels(); ++c)
            error[c] /= image.Resolution().x * image.Resolution().y;
    }

    if (error.MaxValue() == 0)
        // Same same.
        return 0;

    float delta = 100.f * (imageAverage - refAverage) / refAverage;
    std::string deltaString = StringPrintf("%f%% delta", delta);
    if (std::abs(delta) > 0.1)
        deltaString = Red(deltaString);
    else if (std::abs(delta) > 0.001)
        deltaString = Yellow(deltaString);
    Printf("Images differ:\n\t%s %s\n\tavg = %f / %f (%s), %s = %f\n", imageFile,
           referenceFile, imageAverage, refAverage, deltaString, metric, error.Average());

    if (!outFile.empty()) {
        if (!errorImage.Write(outFile, im.metadata))
            return 1;
    }

    return 1;
}

static void printImageStats(const char *name, const Image &image,
                            const ImageMetadata &metadata) {
    printf("%s:\n\tresolution (%d, %d)\n", name, image.Resolution().x,
           image.Resolution().y);
    Printf("\tpixel format: %s\n", image.Format());

    printf("\tcolor space : ");
    if (metadata.colorSpace && *metadata.colorSpace != nullptr) {
        if (**metadata.colorSpace == *RGBColorSpace::sRGB)
            printf("sRGB");
        else if (**metadata.colorSpace == *RGBColorSpace::DCI_P3)
            printf("DCI-P3");
        else if (**metadata.colorSpace == *RGBColorSpace::Rec2020)
            printf("rec2020");
        else if (**metadata.colorSpace == *RGBColorSpace::ACES2065_1)
            printf("ACES");
        else {
            const RGBColorSpace &cs = **metadata.colorSpace;
            printf(" r: %f %f g: %f %f b: %f %f w: %f %f", cs.r.x, cs.r.y, cs.g.x, cs.g.y,
                   cs.b.x, cs.b.y, cs.w.x, cs.w.y);
        }
        printf("\n");
    } else
        printf(" (unspecified)\n");

    if (metadata.fullResolution)
        printf("\tfull resolution (%d, %d)\n", metadata.fullResolution->x,
               metadata.fullResolution->y);
    if (metadata.pixelBounds)
        printf("\tpixel bounds (%d, %d) - (%d, %d)\n", metadata.pixelBounds->pMin.x,
               metadata.pixelBounds->pMin.y, metadata.pixelBounds->pMax.x,
               metadata.pixelBounds->pMax.y);
    if (metadata.renderTimeSeconds) {
        float s = *metadata.renderTimeSeconds;
        int h = int(s) / 3600;
        s -= h * 3600;
        int m = int(s) / 60;
        s -= m * 60;

        printf("\trender time: %dh %dm %.02fs (total %.02fs)\n", h, m, s,
               *metadata.renderTimeSeconds);
    }
    if (metadata.cameraFromWorld)
        printf("\tcamera from world: %s\n", metadata.cameraFromWorld->ToString().c_str());
    if (metadata.NDCFromWorld)
        printf("\tNDC from world: %s\n", metadata.NDCFromWorld->ToString().c_str());
    if (metadata.samplesPerPixel)
        printf("\tsamples per pixel: %d\n", *metadata.samplesPerPixel);

    if (metadata.MSE)
        printf("\tMSE vs. reference image: %g\n", *metadata.MSE);

    for (const auto &iter : metadata.strings)
        printf("\t\"%s\": \"%s\"\n", iter.first.c_str(), iter.second.c_str());

    for (const auto &iter : metadata.stringVectors) {
        printf("\t\"%s\": [ ", iter.first.c_str());
        for (const std::string &str : iter.second)
            printf("\"%s\" ", str.c_str());
        printf("]\n");
    }

    printf("\tChannels:\n");

    std::vector<std::string> channelNames = image.ChannelNames();
    for (const auto &channel : channelNames) {
        Float min = Infinity, max = -Infinity;
        double sum = 0.;
        int nNaN = 0, nInf = 0, nValid = 0;
        ImageChannelDesc desc = image.GetChannelDesc({channel});

        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Float v = image.GetChannels({x, y}, desc);

                if (std::isnan(v))
                    ++nNaN;
                else if (std::isinf(v))
                    ++nInf;
                else {
                    min = std::min(min, v);
                    max = std::max(max, v);
                    sum += v;
                    ++nValid;
                }
            }

        printf("\t    %20s: min %12g max %12g avg %12g (%d infinite, %d "
               "not-a-number)\n",
               channel.c_str(), min, max, sum / nValid, nInf, nNaN);
    }
}

// via tev's FalseColor.cpp, which developed by Thomas Müller
// <thomas94@gmx.net> and is published under the BSD 3-Clause License
// within the LICENSE file.

// "viridis" colormap data generated with scripts/sample-colormap.py
static const std::vector<RGB> falseColorValues = {
    RGB(0.267004f, 0.004874f, 0.329415f), RGB(0.26851f, 0.009605f, 0.335427f),
    RGB(0.269944f, 0.014625f, 0.341379f), RGB(0.271305f, 0.019942f, 0.347269f),
    RGB(0.272594f, 0.025563f, 0.353093f), RGB(0.273809f, 0.031497f, 0.358853f),
    RGB(0.274952f, 0.037752f, 0.364543f), RGB(0.276022f, 0.044167f, 0.370164f),
    RGB(0.277018f, 0.050344f, 0.375715f), RGB(0.277941f, 0.056324f, 0.381191f),
    RGB(0.278791f, 0.062145f, 0.386592f), RGB(0.279566f, 0.067836f, 0.391917f),
    RGB(0.280267f, 0.073417f, 0.397163f), RGB(0.280894f, 0.078907f, 0.402329f),
    RGB(0.281446f, 0.08432f, 0.407414f),  RGB(0.281924f, 0.089666f, 0.412415f),
    RGB(0.282327f, 0.094955f, 0.417331f), RGB(0.282656f, 0.100196f, 0.42216f),
    RGB(0.28291f, 0.105393f, 0.426902f),  RGB(0.283091f, 0.110553f, 0.431554f),
    RGB(0.283197f, 0.11568f, 0.436115f),  RGB(0.283229f, 0.120777f, 0.440584f),
    RGB(0.283187f, 0.125848f, 0.44496f),  RGB(0.283072f, 0.130895f, 0.449241f),
    RGB(0.282884f, 0.13592f, 0.453427f),  RGB(0.282623f, 0.140926f, 0.457517f),
    RGB(0.28229f, 0.145912f, 0.46151f),   RGB(0.281887f, 0.150881f, 0.465405f),
    RGB(0.281412f, 0.155834f, 0.469201f), RGB(0.280868f, 0.160771f, 0.472899f),
    RGB(0.280255f, 0.165693f, 0.476498f), RGB(0.279574f, 0.170599f, 0.479997f),
    RGB(0.278826f, 0.17549f, 0.483397f),  RGB(0.278012f, 0.180367f, 0.486697f),
    RGB(0.277134f, 0.185228f, 0.489898f), RGB(0.276194f, 0.190074f, 0.493001f),
    RGB(0.275191f, 0.194905f, 0.496005f), RGB(0.274128f, 0.199721f, 0.498911f),
    RGB(0.273006f, 0.20452f, 0.501721f),  RGB(0.271828f, 0.209303f, 0.504434f),
    RGB(0.270595f, 0.214069f, 0.507052f), RGB(0.269308f, 0.218818f, 0.509577f),
    RGB(0.267968f, 0.223549f, 0.512008f), RGB(0.26658f, 0.228262f, 0.514349f),
    RGB(0.265145f, 0.232956f, 0.516599f), RGB(0.263663f, 0.237631f, 0.518762f),
    RGB(0.262138f, 0.242286f, 0.520837f), RGB(0.260571f, 0.246922f, 0.522828f),
    RGB(0.258965f, 0.251537f, 0.524736f), RGB(0.257322f, 0.25613f, 0.526563f),
    RGB(0.255645f, 0.260703f, 0.528312f), RGB(0.253935f, 0.265254f, 0.529983f),
    RGB(0.252194f, 0.269783f, 0.531579f), RGB(0.250425f, 0.27429f, 0.533103f),
    RGB(0.248629f, 0.278775f, 0.534556f), RGB(0.246811f, 0.283237f, 0.535941f),
    RGB(0.244972f, 0.287675f, 0.53726f),  RGB(0.243113f, 0.292092f, 0.538516f),
    RGB(0.241237f, 0.296485f, 0.539709f), RGB(0.239346f, 0.300855f, 0.540844f),
    RGB(0.237441f, 0.305202f, 0.541921f), RGB(0.235526f, 0.309527f, 0.542944f),
    RGB(0.233603f, 0.313828f, 0.543914f), RGB(0.231674f, 0.318106f, 0.544834f),
    RGB(0.229739f, 0.322361f, 0.545706f), RGB(0.227802f, 0.326594f, 0.546532f),
    RGB(0.225863f, 0.330805f, 0.547314f), RGB(0.223925f, 0.334994f, 0.548053f),
    RGB(0.221989f, 0.339161f, 0.548752f), RGB(0.220057f, 0.343307f, 0.549413f),
    RGB(0.21813f, 0.347432f, 0.550038f),  RGB(0.21621f, 0.351535f, 0.550627f),
    RGB(0.214298f, 0.355619f, 0.551184f), RGB(0.212395f, 0.359683f, 0.55171f),
    RGB(0.210503f, 0.363727f, 0.552206f), RGB(0.208623f, 0.367752f, 0.552675f),
    RGB(0.206756f, 0.371758f, 0.553117f), RGB(0.204903f, 0.375746f, 0.553533f),
    RGB(0.203063f, 0.379716f, 0.553925f), RGB(0.201239f, 0.38367f, 0.554294f),
    RGB(0.19943f, 0.387607f, 0.554642f),  RGB(0.197636f, 0.391528f, 0.554969f),
    RGB(0.19586f, 0.395433f, 0.555276f),  RGB(0.1941f, 0.399323f, 0.555565f),
    RGB(0.192357f, 0.403199f, 0.555836f), RGB(0.190631f, 0.407061f, 0.556089f),
    RGB(0.188923f, 0.41091f, 0.556326f),  RGB(0.187231f, 0.414746f, 0.556547f),
    RGB(0.185556f, 0.41857f, 0.556753f),  RGB(0.183898f, 0.422383f, 0.556944f),
    RGB(0.182256f, 0.426184f, 0.55712f),  RGB(0.180629f, 0.429975f, 0.557282f),
    RGB(0.179019f, 0.433756f, 0.55743f),  RGB(0.177423f, 0.437527f, 0.557565f),
    RGB(0.175841f, 0.44129f, 0.557685f),  RGB(0.174274f, 0.445044f, 0.557792f),
    RGB(0.172719f, 0.448791f, 0.557885f), RGB(0.171176f, 0.45253f, 0.557965f),
    RGB(0.169646f, 0.456262f, 0.55803f),  RGB(0.168126f, 0.459988f, 0.558082f),
    RGB(0.166617f, 0.463708f, 0.558119f), RGB(0.165117f, 0.467423f, 0.558141f),
    RGB(0.163625f, 0.471133f, 0.558148f), RGB(0.162142f, 0.474838f, 0.55814f),
    RGB(0.160665f, 0.47854f, 0.558115f),  RGB(0.159194f, 0.482237f, 0.558073f),
    RGB(0.157729f, 0.485932f, 0.558013f), RGB(0.15627f, 0.489624f, 0.557936f),
    RGB(0.154815f, 0.493313f, 0.55784f),  RGB(0.153364f, 0.497f, 0.557724f),
    RGB(0.151918f, 0.500685f, 0.557587f), RGB(0.150476f, 0.504369f, 0.55743f),
    RGB(0.149039f, 0.508051f, 0.55725f),  RGB(0.147607f, 0.511733f, 0.557049f),
    RGB(0.14618f, 0.515413f, 0.556823f),  RGB(0.144759f, 0.519093f, 0.556572f),
    RGB(0.143343f, 0.522773f, 0.556295f), RGB(0.141935f, 0.526453f, 0.555991f),
    RGB(0.140536f, 0.530132f, 0.555659f), RGB(0.139147f, 0.533812f, 0.555298f),
    RGB(0.13777f, 0.537492f, 0.554906f),  RGB(0.136408f, 0.541173f, 0.554483f),
    RGB(0.135066f, 0.544853f, 0.554029f), RGB(0.133743f, 0.548535f, 0.553541f),
    RGB(0.132444f, 0.552216f, 0.553018f), RGB(0.131172f, 0.555899f, 0.552459f),
    RGB(0.129933f, 0.559582f, 0.551864f), RGB(0.128729f, 0.563265f, 0.551229f),
    RGB(0.127568f, 0.566949f, 0.550556f), RGB(0.126453f, 0.570633f, 0.549841f),
    RGB(0.125394f, 0.574318f, 0.549086f), RGB(0.124395f, 0.578002f, 0.548287f),
    RGB(0.123463f, 0.581687f, 0.547445f), RGB(0.122606f, 0.585371f, 0.546557f),
    RGB(0.121831f, 0.589055f, 0.545623f), RGB(0.121148f, 0.592739f, 0.544641f),
    RGB(0.120565f, 0.596422f, 0.543611f), RGB(0.120092f, 0.600104f, 0.54253f),
    RGB(0.119738f, 0.603785f, 0.5414f),   RGB(0.119512f, 0.607464f, 0.540218f),
    RGB(0.119423f, 0.611141f, 0.538982f), RGB(0.119483f, 0.614817f, 0.537692f),
    RGB(0.119699f, 0.61849f, 0.536347f),  RGB(0.120081f, 0.622161f, 0.534946f),
    RGB(0.120638f, 0.625828f, 0.533488f), RGB(0.12138f, 0.629492f, 0.531973f),
    RGB(0.122312f, 0.633153f, 0.530398f), RGB(0.123444f, 0.636809f, 0.528763f),
    RGB(0.12478f, 0.640461f, 0.527068f),  RGB(0.126326f, 0.644107f, 0.525311f),
    RGB(0.128087f, 0.647749f, 0.523491f), RGB(0.130067f, 0.651384f, 0.521608f),
    RGB(0.132268f, 0.655014f, 0.519661f), RGB(0.134692f, 0.658636f, 0.517649f),
    RGB(0.137339f, 0.662252f, 0.515571f), RGB(0.14021f, 0.665859f, 0.513427f),
    RGB(0.143303f, 0.669459f, 0.511215f), RGB(0.146616f, 0.67305f, 0.508936f),
    RGB(0.150148f, 0.676631f, 0.506589f), RGB(0.153894f, 0.680203f, 0.504172f),
    RGB(0.157851f, 0.683765f, 0.501686f), RGB(0.162016f, 0.687316f, 0.499129f),
    RGB(0.166383f, 0.690856f, 0.496502f), RGB(0.170948f, 0.694384f, 0.493803f),
    RGB(0.175707f, 0.6979f, 0.491033f),   RGB(0.180653f, 0.701402f, 0.488189f),
    RGB(0.185783f, 0.704891f, 0.485273f), RGB(0.19109f, 0.708366f, 0.482284f),
    RGB(0.196571f, 0.711827f, 0.479221f), RGB(0.202219f, 0.715272f, 0.476084f),
    RGB(0.20803f, 0.718701f, 0.472873f),  RGB(0.214f, 0.722114f, 0.469588f),
    RGB(0.220124f, 0.725509f, 0.466226f), RGB(0.226397f, 0.728888f, 0.462789f),
    RGB(0.232815f, 0.732247f, 0.459277f), RGB(0.239374f, 0.735588f, 0.455688f),
    RGB(0.24607f, 0.73891f, 0.452024f),   RGB(0.252899f, 0.742211f, 0.448284f),
    RGB(0.259857f, 0.745492f, 0.444467f), RGB(0.266941f, 0.748751f, 0.440573f),
    RGB(0.274149f, 0.751988f, 0.436601f), RGB(0.281477f, 0.755203f, 0.432552f),
    RGB(0.288921f, 0.758394f, 0.428426f), RGB(0.296479f, 0.761561f, 0.424223f),
    RGB(0.304148f, 0.764704f, 0.419943f), RGB(0.311925f, 0.767822f, 0.415586f),
    RGB(0.319809f, 0.770914f, 0.411152f), RGB(0.327796f, 0.77398f, 0.40664f),
    RGB(0.335885f, 0.777018f, 0.402049f), RGB(0.344074f, 0.780029f, 0.397381f),
    RGB(0.35236f, 0.783011f, 0.392636f),  RGB(0.360741f, 0.785964f, 0.387814f),
    RGB(0.369214f, 0.788888f, 0.382914f), RGB(0.377779f, 0.791781f, 0.377939f),
    RGB(0.386433f, 0.794644f, 0.372886f), RGB(0.395174f, 0.797475f, 0.367757f),
    RGB(0.404001f, 0.800275f, 0.362552f), RGB(0.412913f, 0.803041f, 0.357269f),
    RGB(0.421908f, 0.805774f, 0.35191f),  RGB(0.430983f, 0.808473f, 0.346476f),
    RGB(0.440137f, 0.811138f, 0.340967f), RGB(0.449368f, 0.813768f, 0.335384f),
    RGB(0.458674f, 0.816363f, 0.329727f), RGB(0.468053f, 0.818921f, 0.323998f),
    RGB(0.477504f, 0.821444f, 0.318195f), RGB(0.487026f, 0.823929f, 0.312321f),
    RGB(0.496615f, 0.826376f, 0.306377f), RGB(0.506271f, 0.828786f, 0.300362f),
    RGB(0.515992f, 0.831158f, 0.294279f), RGB(0.525776f, 0.833491f, 0.288127f),
    RGB(0.535621f, 0.835785f, 0.281908f), RGB(0.545524f, 0.838039f, 0.275626f),
    RGB(0.555484f, 0.840254f, 0.269281f), RGB(0.565498f, 0.84243f, 0.262877f),
    RGB(0.575563f, 0.844566f, 0.256415f), RGB(0.585678f, 0.846661f, 0.249897f),
    RGB(0.595839f, 0.848717f, 0.243329f), RGB(0.606045f, 0.850733f, 0.236712f),
    RGB(0.616293f, 0.852709f, 0.230052f), RGB(0.626579f, 0.854645f, 0.223353f),
    RGB(0.636902f, 0.856542f, 0.21662f),  RGB(0.647257f, 0.8584f, 0.209861f),
    RGB(0.657642f, 0.860219f, 0.203082f), RGB(0.668054f, 0.861999f, 0.196293f),
    RGB(0.678489f, 0.863742f, 0.189503f), RGB(0.688944f, 0.865448f, 0.182725f),
    RGB(0.699415f, 0.867117f, 0.175971f), RGB(0.709898f, 0.868751f, 0.169257f),
    RGB(0.720391f, 0.87035f, 0.162603f),  RGB(0.730889f, 0.871916f, 0.156029f),
    RGB(0.741388f, 0.873449f, 0.149561f), RGB(0.751884f, 0.874951f, 0.143228f),
    RGB(0.762373f, 0.876424f, 0.137064f), RGB(0.772852f, 0.877868f, 0.131109f),
    RGB(0.783315f, 0.879285f, 0.125405f), RGB(0.79376f, 0.880678f, 0.120005f),
    RGB(0.804182f, 0.882046f, 0.114965f), RGB(0.814576f, 0.883393f, 0.110347f),
    RGB(0.82494f, 0.88472f, 0.106217f),   RGB(0.83527f, 0.886029f, 0.102646f),
    RGB(0.845561f, 0.887322f, 0.099702f), RGB(0.85581f, 0.888601f, 0.097452f),
    RGB(0.866013f, 0.889868f, 0.095953f), RGB(0.876168f, 0.891125f, 0.09525f),
    RGB(0.886271f, 0.892374f, 0.095374f), RGB(0.89632f, 0.893616f, 0.096335f),
    RGB(0.906311f, 0.894855f, 0.098125f), RGB(0.916242f, 0.896091f, 0.100717f),
    RGB(0.926106f, 0.89733f, 0.104071f),  RGB(0.935904f, 0.89857f, 0.108131f),
    RGB(0.945636f, 0.899815f, 0.112838f), RGB(0.9553f, 0.901065f, 0.118128f),
    RGB(0.964894f, 0.902323f, 0.123941f), RGB(0.974417f, 0.90359f, 0.130215f),
    RGB(0.983868f, 0.904867f, 0.136897f), RGB(0.993248f, 0.906157f, 0.143936f),
};

int falsecolor(std::vector<std::string> args) {
    std::string outFile, inFile;
    bool plusMinus = false, ramp = false;
    Float maxValue = -Infinity;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("falsecolor", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "outfile", &outFile, onError) ||
            ParseArg(&iter, args.end(), "plusminus", &plusMinus, onError) ||
            ParseArg(&iter, args.end(), "ramp", &ramp, onError) ||
            ParseArg(&iter, args.end(), "maxValue", &maxValue, onError)) {
            // success
        } else if (inFile.empty() && (*iter)[0] != '-') {
            inFile = *iter;
        } else {
            usage("falsecolor", "%s: unknown command flag", iter->c_str());
        }
    }

    if (!ramp && inFile.empty())
        usage("falsecolor", "expecting input image filename.");
    if (outFile.empty())
        usage("falsecolor", "expecting --outfile filename.");

    Image image;
    if (ramp) {
        int xres = 10, yres = 300;
        image = Image(PixelFormat::Float, {xres, yres}, {"Y"});
        for (int y = 0; y < yres; ++y)
            for (int x = 0; x < xres; ++x)
                image.SetChannel({x, y}, 0, Float(yres - 1 - y) / Float(yres - 1));
    } else
        image = Image::Read(inFile).image;

    if (maxValue == -Infinity)
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                maxValue =
                    std::max(maxValue, std::abs(image.GetChannels({x, y}).Average()));

    Image outImage(PixelFormat::Half, image.Resolution(), {"R", "G", "B"});
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x) {
            Float relativeValue = image.GetChannels({x, y}).Average() / maxValue;
            RGB rgb;
            if (plusMinus) {
                if (relativeValue > 0)
                    rgb = RGB(0, relativeValue, 0);
                else
                    rgb = RGB(std::abs(relativeValue), 0, 0);
            } else {
                relativeValue = Clamp(relativeValue, 0, 1);
                int index = relativeValue * falseColorValues.size();
                index = std::min<int>(index, falseColorValues.size() - 1);
                rgb = falseColorValues[index];
            }

            outImage.SetChannels({x, y}, {SRGBToLinear(rgb[0]), SRGBToLinear(rgb[1]),
                                          SRGBToLinear(rgb[2])});
        }

    if (!outImage.Write(outFile))
        return 1;

    return 0;
}

int info(std::vector<std::string> args) {
    int err = 0;
    for (std::string fn : args) {
        ImageAndMetadata im = Image::Read(fn);
        printImageStats(fn.c_str(), im.image, im.metadata);
    }
    return err;
}

int bloom(std::vector<std::string> args) {
    std::string inFile, outFile;
    Float level = Infinity;
    int width = 15;
    Float scale = .3;
    int iterations = 5;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("bloom", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "outfile", &outFile, onError) ||
            ParseArg(&iter, args.end(), "level", &level, onError) ||
            ParseArg(&iter, args.end(), "width", &width, onError) ||
            ParseArg(&iter, args.end(), "iterations", &iterations, onError) ||
            ParseArg(&iter, args.end(), "scale", &scale, onError)) {
            // success
        } else if (inFile.empty() && (*iter)[0] != '-') {
            inFile = *iter;
        } else {
            onError(StringPrintf("argument %s invalid", *iter));
        }
    }

    if (outFile.empty())
        usage("bloom", "--outfile must be specified");
    if (inFile.empty())
        usage("bloom", "input filename must be specified");

    ImageAndMetadata imRead = Image::Read(inFile);
    Image &image = imRead.image;

    std::vector<Image> blurred;

    // First, threshold the source image
    int nSurvivors = 0;
    Point2i res = image.Resolution();
    int nc = image.NChannels();
    Image thresholdedImage(PixelFormat::Float, image.Resolution(), image.ChannelNames());
    for (int y = 0; y < res.y; ++y) {
        for (int x = 0; x < res.x; ++x) {
            bool overThreshold = false;
            for (int c = 0; c < nc; ++c)
                if (image.GetChannel({x, y}, c) > level)
                    overThreshold = true;
            if (overThreshold) {
                ++nSurvivors;
                for (int c = 0; c < nc; ++c)
                    thresholdedImage.SetChannel({x, y}, c, image.GetChannel({x, y}, c));
            } else
                for (int c = 0; c < nc; ++c)
                    thresholdedImage.SetChannel({x, y}, c, 0.f);
        }
    }
    if (nSurvivors == 0) {
        Warning("No pixels were above bloom threshold %f", level);
        return 1;
    }
    blurred.push_back(std::move(thresholdedImage));

    if ((width % 2) == 0) {
        ++width;
        Warning("Bloom width must be an odd value. Rounding up to %d.", width);
    }
    int radius = width / 2;

    // Blur thresholded image.
    Float sigma = radius / 2.;  // TODO: make a parameter

    for (int iter = 0; iter < iterations; ++iter) {
        Image blur =
            blurred.back().GaussianFilter(image.AllChannelsDesc(), radius, sigma);
        blurred.push_back(blur);
    }

    // Finally, add all of the blurred images, scaled, to the original.
    for (int y = 0; y < res.y; ++y) {
        for (int x = 0; x < res.x; ++x) {
            for (int c = 0; c < nc; ++c) {
                Float blurredSum = 0.f;
                // Skip the thresholded image, since it's already
                // present in the original; just add pixels from the
                // blurred ones.
                for (size_t j = 1; j < blurred.size(); ++j)
                    blurredSum += blurred[j].GetChannel({x, y}, c);
                image.SetChannel(
                    {x, y}, c,
                    image.GetChannel({x, y}, c) + (scale / iterations) * blurredSum);
            }
        }
    }

    image.Write(outFile, imRead.metadata);

    return 0;
}

int convert(std::vector<std::string> args) {
    bool acesFilmic = false;
    float scale = 1.f, gamma = 1.f;
    int repeat = 1;
    bool flipy = false;
    bool tonemap = false;
    Float maxY = 1.;
    Float despikeLimit = Infinity;
    bool preserveColors = false;
    bool bw = false;
    bool fp16 = false;
    std::string inFile, outFile;
    std::string colorspace;
    std::string channelNames;
    std::array<int, 4> cropWindow = {-1, 0, -1, 0};
    Float clamp = Infinity;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("convert", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "acesfilmic", &acesFilmic, onError) ||
            ParseArg(&iter, args.end(), "bw", &bw, onError) ||
            ParseArg(&iter, args.end(), "channels", &channelNames, onError) ||
            ParseArg(&iter, args.end(), "clamp", &clamp, onError) ||
            ParseArg(&iter, args.end(), "colorspace", &colorspace, onError) ||
            ParseArg(&iter, args.end(), "crop", pstd::MakeSpan(cropWindow), onError) ||
            ParseArg(&iter, args.end(), "despike", &despikeLimit, onError) ||
            ParseArg(&iter, args.end(), "fp16", &fp16, onError) ||
            ParseArg(&iter, args.end(), "flipy", &flipy, onError) ||
            ParseArg(&iter, args.end(), "gamma", &gamma, onError) ||
            ParseArg(&iter, args.end(), "maxluminance", &maxY, onError) ||
            ParseArg(&iter, args.end(), "outfile", &outFile, onError) ||
            ParseArg(&iter, args.end(), "preservecolors", &preserveColors, onError) ||
            ParseArg(&iter, args.end(), "repeatpix", &repeat, onError) ||
            ParseArg(&iter, args.end(), "scale", &scale, onError) ||
            ParseArg(&iter, args.end(), "tonemap", &tonemap, onError)) {
            // success
        } else if ((*iter)[0] != '-' && inFile.empty()) {
            inFile = *iter;
        } else
            usage("convert", "%s: unknown command flag", iter->c_str());
    }

    if (maxY <= 0)
        usage("convert", "--maxluminance value must be greater than zero");
    if (repeat <= 0)
        usage("convert", "--repeatpix value must be greater than zero");
    if (scale == 0)
        usage("convert", "--scale value must be non-zero");
    if (outFile.empty())
        usage("convert", "--outfile filename must be specified");
    if (inFile.empty())
        usage("convert", "input filename not specified");

    ImageAndMetadata imRead = Image::Read(inFile);
    Image image = std::move(imRead.image);
    ImageMetadata metadata = std::move(imRead.metadata);

    ImageMetadata outMetadata;
    // These are the only entries it makes sense to copy along (so far).
    outMetadata.colorSpace = metadata.colorSpace;
    outMetadata.cameraFromWorld = metadata.cameraFromWorld;
    outMetadata.NDCFromWorld = metadata.NDCFromWorld;
    outMetadata.pixelBounds = metadata.pixelBounds;
    outMetadata.fullResolution = metadata.fullResolution;

    if (channelNames.empty()) {
        // If the input image has AOVs and the target image is a regular
        // format, then just grab R,G,B...
        bool hasAOVs = false;
        for (const std::string &name : image.ChannelNames())
            if (name != "R" && name != "G" && name != "B" && name != "A") {
                hasAOVs = true;
                break;
            }

        if (hasAOVs && !HasExtension(outFile, "exr")) {
            Warning("%s: image has non-RGBA channels but converting to an image format "
                    "that can't store them. Converting RGB only.",
                    inFile);
            channelNames = "R,G,B";
        }
    }

    if (!channelNames.empty()) {
        std::vector<std::string> splitChannelNames = SplitString(channelNames, ',');
        ImageChannelDesc desc = image.GetChannelDesc(splitChannelNames);
        if (!desc) {
            Error("%s: image doesn't have channels \"%s\".", inFile, channelNames);
            return 1;
        }
        image = image.SelectChannels(desc);
    }

    Point2i res = image.Resolution();
    int nc = image.NChannels();

    // Crop
    // If last 2 are negative, they're taken as deltas
    if (cropWindow[1] < 0)
        cropWindow[1] = cropWindow[0] - cropWindow[1];
    if (cropWindow[3] < 0)
        cropWindow[3] = cropWindow[2] - cropWindow[3];
    if (cropWindow[0] >= 0 && cropWindow[2] >= 0) {
        Bounds2i cropBounds({cropWindow[0], cropWindow[2]},
                            {cropWindow[1], cropWindow[3]});

        Point2i fullRes =
            metadata.fullResolution ? *metadata.fullResolution : image.Resolution();

        if (metadata.pixelBounds && !Inside(cropBounds, *metadata.pixelBounds)) {
            Error("%s: crop window bounds (%d,%d)-(%d,%d) are not inside "
                  "image's pixel bounds (%d,%d)-(%d,%d).",
                  inFile, cropBounds.pMin.x, cropBounds.pMin.y, cropBounds.pMax.x,
                  cropBounds.pMax.y, metadata.pixelBounds->pMin.x,
                  metadata.pixelBounds->pMin.y, metadata.pixelBounds->pMax.x,
                  metadata.pixelBounds->pMax.y);
            return 1;
        } else if (!Inside(cropBounds, Bounds2i(Point2i(0, 0), fullRes))) {
            Error("%s: crop window bounds (%d,%d)-(%d,%d) are not inside "
                  "image's resolution (%d,%d).",
                  inFile, cropBounds.pMin.x, cropBounds.pMin.y, cropBounds.pMax.x,
                  cropBounds.pMax.y, fullRes.x, fullRes.y);
            return 1;
        }

        outMetadata.fullResolution = fullRes;
        outMetadata.pixelBounds = cropBounds;

        // Adjust crop bounds for call to Crop()
        if (metadata.pixelBounds) {
            cropBounds.pMin = Point2i(cropBounds.pMin - metadata.pixelBounds->pMin);
            cropBounds.pMax = Point2i(cropBounds.pMax - metadata.pixelBounds->pMin);
        }
        image = image.Crop(cropBounds);

        res = image.Resolution();
    }

    // Count how many of the following operations modify pixel values
    // rather than copying existing ones.  (Takes heavy advantage of
    // implicit bool->int conversions.)
    int nModifyingOps = (!colorspace.empty() + (scale != 1) + (gamma != 1) + tonemap +
                         preserveColors + acesFilmic);
    // Convert to a 32-bit format for maximum accuracy if we're applying
    // multiple operations to the pixel values.
    if (nModifyingOps > 1 && !Is32Bit(image.Format()))
        image = image.ConvertToFormat(PixelFormat::Float);

    if (clamp < Infinity) {
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x) {
                Float maxValue = 0;
                for (int c = 0; c < nc; ++c)
                    maxValue = std::max(maxValue, image.GetChannel({x, y}, c));
                if (maxValue > clamp) {
                    Float scale = clamp / maxValue;
                    for (int c = 0; c < nc; ++c)
                        image.SetChannel({x, y}, c, scale * image.GetChannel({x, y}, c));
                }
            }
    }

    if (!colorspace.empty()) {
        const RGBColorSpace *dest = RGBColorSpace::GetNamed(colorspace);
        if (!dest) {
            Error("%s: color space unknown.", colorspace);
            return 1;
        }
        ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
        if (!rgbDesc) {
            Error("%s: doesn't have R, G, B channels.", inFile);
            return 1;
        }

        const RGBColorSpace *srcColorSpace = metadata.GetColorSpace();
        SquareMatrix<3> m = ConvertRGBColorSpace(*srcColorSpace, *dest);
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x) {
                ImageChannelValues channels = image.GetChannels({x, y}, rgbDesc);
                RGB rgb = Mul<RGB>(m, channels);
                image.SetChannels({x, y}, rgbDesc, {rgb.r, rgb.g, rgb.b});
            }
        outMetadata.colorSpace = dest;
    }

    if (bw) {
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x) {
                Float sum = 0;
                for (int c = 0; c < nc; ++c)
                    sum += image.GetChannel({x, y}, c);
                sum /= nc;
                for (int c = 0; c < nc; ++c)
                    image.SetChannel({x, y}, c, sum);
            }
    }

    if (despikeLimit < Infinity) {
        Image filteredImg = image;
        int despikeCount = 0;
        std::vector<ImageChannelValues> neighbors;
        for (int i = 0; i < 9; ++i)
            neighbors.push_back(ImageChannelValues(image.NChannels()));

        for (int y = 0; y < res.y; ++y) {
            for (int x = 0; x < res.x; ++x) {
                if (image.GetChannels({x, y}).Average() < despikeLimit)
                    continue;

                // Copy all of the valid neighbor pixels into neighbors[].
                ++despikeCount;
                int validNeighbors = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    if (y + dy < 0 || y + dy >= res.y)
                        continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (x + dx < 0 || x + dx > res.x)
                            continue;
                        neighbors[validNeighbors++] = image.GetChannels({x + dx, y + dy});
                    }
                }

                // Find the median of the neighbors, sorted by average value.
                int mid = validNeighbors / 2;
                std::nth_element(
                    &neighbors[0], &neighbors[mid], &neighbors[validNeighbors],
                    [](const ImageChannelValues &a, const ImageChannelValues &b) -> bool {
                        return a.Average() < b.Average();
                    });
                filteredImg.SetChannels({x, y}, neighbors[mid]);
            }
        }
        pstd::swap(image, filteredImg);
        fprintf(stderr, "%s: despiked %d pixels\n", inFile.c_str(), despikeCount);
    }

    if (scale != 1) {
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x)
                for (int c = 0; c < nc; ++c)
                    image.SetChannel({x, y}, c, scale * image.GetChannel({x, y}, c));
    }

    if (gamma != 1) {
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x)
                for (int c = 0; c < nc; ++c)
                    image.SetChannel(
                        {x, y}, c,
                        std::pow(std::max<Float>(0, image.GetChannel({x, y}, c)), gamma));
    }

    if (tonemap) {
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x) {
                Float lum = image.GetChannels({x, y}).Average();
                // Reinhard et al. photographic tone mapping operator.
                Float scale = (1 + lum / (maxY * maxY)) / (1 + lum);
                for (int c = 0; c < nc; ++c)
                    image.SetChannel({x, y}, c, scale * image.GetChannel({x, y}, c));
            }
    }

    if (preserveColors) {
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x) {
                Float m = image.GetChannel({x, y}, 0);
                for (int c = 1; c < nc; ++c)
                    m = std::max(m, image.GetChannel({x, y}, c));
                if (m > 1) {
                    for (int c = 0; c < nc; ++c)
                        image.SetChannel({x, y}, c, image.GetChannel({x, y}, c) / m);
                }
            }
    }

    if (acesFilmic) {
        // Approximation via
        // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
        auto ACESFilm = [](Float x) -> Float {
            if (x <= 0)
                return 0;
            Float a = 2.51f;
            Float b = 0.03f;
            Float c = 2.43f;
            Float d = 0.59f;
            Float e = 0.14f;
            return Clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1);
        };

        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x) {
                for (int c = 0; c < nc; ++c) {
                    Float v = image.GetChannel({x, y}, c);
                    v = ACESFilm(v);
                    image.SetChannel({x, y}, c, v);
                }
            }
    }

    if (repeat > 1) {
        Image scaledImage(image.Format(), Point2i(res.x * repeat, res.y * repeat),
                          image.ChannelNames(), image.Encoding());
        for (int y = 0; y < repeat * res.y; ++y) {
            int yy = y / repeat;
            for (int x = 0; x < repeat * res.x; ++x) {
                int xx = x / repeat;
                for (int c = 0; c < nc; ++c)
                    scaledImage.SetChannel({x, y}, c, image.GetChannel({xx, yy}, c));
            }
        }
        image = std::move(scaledImage);
        res = image.Resolution();

        if (outMetadata.fullResolution)
            *outMetadata.fullResolution *= repeat;
        if (outMetadata.pixelBounds) {
            outMetadata.pixelBounds->pMin *= repeat;
            outMetadata.pixelBounds->pMax *= repeat;
        }
    }

    if (flipy) {
        image.FlipY();
        if (outMetadata.pixelBounds && outMetadata.fullResolution) {
            // Flip the pixel bounds in the metadata as well
            int by[2] = {
                outMetadata.fullResolution->y - 1 - outMetadata.pixelBounds->pMax.y,
                outMetadata.fullResolution->y - 1 - outMetadata.pixelBounds->pMin.y};
            outMetadata.pixelBounds->pMin.y = by[0];
            outMetadata.pixelBounds->pMax.y = by[1];
        }
    }

    if (fp16)
        image = image.ConvertToFormat(PixelFormat::Half);

    if (!image.Write(outFile, outMetadata))
        return 1;

    return 0;
}

int whitebalance(std::vector<std::string> args) {
    std::string inFile, outFile;
    Float temperature = 0;
    std::array<Float, 2> xy = {Float(0), Float(0)};
    std::string illuminant;

    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        auto onError = [](const std::string &err) {
            usage("whitebalance", "%s", err.c_str());
            exit(1);
        };

        if (ParseArg(&iter, args.end(), "outfile", &outFile, onError) ||
            ParseArg(&iter, args.end(), "primaries", pstd::MakeSpan(xy), onError) ||
            ParseArg(&iter, args.end(), "illuminant", &illuminant, onError) ||
            ParseArg(&iter, args.end(), "temperature", &temperature, onError)) {
            // success
        } else if (inFile.empty() && (*iter)[0] != '-') {
            inFile = *iter;
        } else {
            onError(StringPrintf("argument %s invalid", *iter));
        }
    }

    if (outFile.empty())
        usage("whitebalance", "--outfile must be specified");
    if (inFile.empty())
        usage("whitebalance", "input filename must be specified");

    if ((!illuminant.empty() + (temperature > 0) + (xy[0] != 0)) > 1)
        usage("whitebalance",
              "can only provide one of --illuminant, --primaries, --temperature");
    if (illuminant.empty() && temperature == 0 && xy[0] == 0)
        usage("whitebalance",
              "must provide one of --illuminant, --primaries, or --temperature");

    ImageAndMetadata imRead = Image::Read(inFile);
    Image &image = imRead.image;

    ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!rgbDesc) {
        Error("%s: doesn't have R, G, B channels.", inFile);
        return 1;
    }

    const RGBColorSpace *colorSpace = imRead.metadata.GetColorSpace();
    Point2f srcWhite, targetWhite = colorSpace->w;
    if (!illuminant.empty()) {
        std::string name = "stdillum-" + illuminant;
        Spectrum illum = GetNamedSpectrum(name);
        if (!illum) {
            Error("%s: illuminant unknown.", name);
            return 1;
        }
        srcWhite = SpectrumToXYZ(illum).xy();
    } else if (temperature > 0) {
        // Sensor::Create uses Spectra::D() rather than BlackbodySpectrum
        // here---make consistent?
        DenselySampledSpectrum illum = Spectra::D(temperature, Allocator());
        srcWhite = SpectrumToXYZ(&illum).xy();
    } else
        srcWhite = Point2f(xy[0], xy[1]);

    SquareMatrix<3> ccMatrix = colorSpace->RGBFromXYZ *
                               WhiteBalance(srcWhite, targetWhite) *
                               colorSpace->XYZFromRGB;

    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x) {
            ImageChannelValues channels = image.GetChannels({x, y}, rgbDesc);
            RGB rgb = Mul<RGB>(ccMatrix, channels);
            image.SetChannels({x, y}, rgbDesc, {rgb.r, rgb.g, rgb.b});
        }

    image.Write(outFile, imRead.metadata);

    return 0;
}

int makeemitters(std::vector<std::string> args) {
    std::string filename;
    int downsampleRate = 1;

    auto onError = [](const std::string &err) {
        usage("makeemitters", "%s", err.c_str());
        exit(1);
    };
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if (ParseArg(&iter, args.end(), "downsample", &downsampleRate, onError)) {
            // success
        } else if ((*iter)[0] == '-')
            usage("makeemitters", "%s: unknown command flag", iter->c_str());
        else if (filename.empty()) {
            filename = *iter;
        } else
            usage("makeemitters", "multiple input filenames provided.");
    }

    if (filename.empty())
        usage("makeemitters", "missing image filename");

    ImageAndMetadata im = Image::Read(filename);
    const Image &image = im.image;

    ImageChannelDesc rgbDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!rgbDesc) {
        Error("%s: didn't find R, G, and B channels", filename);
        return 1;
    }

    Point2i res = image.Resolution();
    float aspect = float(res.x) / float(res.y);
    printf("AttributeBegin\n");
    printf("Material \"matte\" \"rgb Kd\" [0 0 0]\n");
    for (int y = 0; y < image.Resolution().y; y += downsampleRate)
        for (int x = 0; x < image.Resolution().x; x += downsampleRate) {
            ImageChannelValues pSum(rgbDesc.size());
            for (int dy = 0; dy < downsampleRate; ++dy)
                for (int dx = 0; dx < downsampleRate; ++dx) {
                    Point2i pp(x + dx, y + dy);
                    if (pp.x >= res.x && pp.y >= res.y)
                        continue;

                    ImageChannelValues p = image.GetChannels(pp, rgbDesc);
                    for (int c = 0; c < p.size(); ++c)
                        pSum[c] += p[c];
                }
            for (int c = 0; c < pSum.size(); ++c)
                pSum[c] /= downsampleRate * downsampleRate;

            printf("AreaLightSource \"diffuse\" \"rgb L\" [ %f %f %f ]\n", pSum[0],
                   pSum[1], pSum[2]);

            float x0 = aspect * (1 - float(x) / image.Resolution().x) - aspect / 2;
            float x1 = aspect * (1 - float(std::min(x + downsampleRate, res.x)) /
                                         image.Resolution().x) -
                       aspect / 2;
            float y0 = 1 - float(y) / image.Resolution().y;
            float y1 =
                1 - float(std::min(y + downsampleRate, res.y)) / image.Resolution().y;
            printf("Shape \"bilinear\" \"point3 P\" [ %f %f 0 %f %f 0 %f %f 0 "
                   "%f %f 0 ]\n",
                   x0, y0, x1, y0, x0, y1, x1, y1);
        }
    printf("AttributeEnd\n");

    return 0;
}

int makeequiarea(std::vector<std::string> args) {
    std::string inFilename, outFilename;
    int resolution = 0;

    auto onError = [](const std::string &err) {
        usage("makeequiarea", "%s", err.c_str());
        exit(1);
    };
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if (ParseArg(&iter, args.end(), "resolution", &resolution, onError) ||
            ParseArg(&iter, args.end(), "outfile", &outFilename, onError)) {
            // success
        } else if ((*iter)[0] == '-')
            usage("makeequiarea", "%s: unknown command flag", iter->c_str());
        else if (inFilename.empty()) {
            inFilename = *iter;
        } else
            usage("makeequiarea", "multiple input filenames provided.");
    }
    if (inFilename.empty())
        usage("makeequiarea", "input image filename must be provided.");
    if (outFilename.empty())
        usage("makeequiarea", "output image filename must be provided.");

    ImageAndMetadata latlong = Image::Read(inFilename);
    const Image &latlongImage = latlong.image;

    if (2 * latlongImage.Resolution().y != latlongImage.Resolution().x)
        Warning("%s: esolution (%d, %d) doesn't have a 2:1 aspect ratio. "
                "It is doubtful that this is a lat-long environment map.",
                inFilename, latlongImage.Resolution().x, latlongImage.Resolution().y);

    if (resolution == 0)
        // resolution = 1.25f * latlongImage.Resolution().x * std::sqrt(2.f) /
        // 4;
        resolution = latlongImage.Resolution().x;

    // TODO: should we check that lat-long has RGB here?
    // Should we only copy over RGB?
    Image equiRectImage(latlongImage.Format(), {resolution, resolution},
                        latlongImage.ChannelNames(), latlongImage.Encoding());

    GaussianFilter filter(Vector2f(1.5, 1.5), 2);
    // MitchellFilter filter(Vector2f(2.f, 2.f), 1.f/3.f, 1.f/3.f);
    int sqrtSamples = 6;
    WrapMode2D latlongWrap(WrapMode::Repeat, WrapMode::Clamp);

    ParallelFor(0, resolution, [&](int64_t v0, int64_t v1) {
        RNG rng(v0);
        for (int v = v0; v < v1; ++v)
            for (int u = 0; u < resolution; ++u) {
                Float sumWeight = 0;
                ImageChannelValues sumSamples(latlongImage.NChannels(), 0.f);

                for (int dv = 0; dv < sqrtSamples; ++dv)
                    for (int du = 0; du < sqrtSamples; ++du) {
                        // Stratified samples
                        Point2f s2((du + rng.Uniform<Float>()) / sqrtSamples,
                                   (dv + rng.Uniform<Float>()) / sqrtSamples);
                        FilterSample fs = filter.Sample(s2);
                        // Map to a point in the equirect map for the current
                        // pixel.
                        Point2f pSquare((u + 0.5f + fs.p.x) / resolution,
                                        (v + 0.5f + fs.p.y) / resolution);
                        pSquare = WrapEqualAreaSquare(pSquare);

                        // Get corresponding pixel values from the lat-long map.
                        Vector3f dir = EqualAreaSquareToSphere(pSquare);
                        Float theta = SphericalTheta(dir), phi = SphericalPhi(dir);
                        Point2f p(phi / (2 * Pi), theta / Pi);
                        ImageChannelValues values = latlongImage.Bilerp(p, latlongWrap);

                        // Accumulate
                        for (size_t i = 0; i < values.size(); ++i)
                            sumSamples[i] += fs.weight * values[i];
                        sumWeight += fs.weight;
                    }
                for (size_t i = 0; i < sumSamples.size(); ++i)
                    sumSamples[i] = std::max<Float>(0, sumSamples[i] / sumWeight);
                equiRectImage.SetChannels({u, v}, sumSamples);
            }
    });

    ImageMetadata equiRectMetadata;
    equiRectMetadata.cameraFromWorld = latlong.metadata.cameraFromWorld;
    equiRectMetadata.colorSpace = latlong.metadata.colorSpace;
    equiRectMetadata.strings = latlong.metadata.strings;
    equiRectMetadata.stringVectors = latlong.metadata.stringVectors;
    equiRectImage.Write(outFilename, equiRectMetadata);

    return 0;
}

#ifdef PBRT_BUILD_GPU_RENDERER
int denoise_optix(std::vector<std::string> args) {
    std::string inFilename, outFilename;

    auto onError = [](const std::string &err) {
        usage("denoise-optix", "%s", err.c_str());
        exit(1);
    };
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if (ParseArg(&iter, args.end(), "outfile", &outFilename, onError)) {
            // success
        } else if ((*iter)[0] == '-')
            usage("denoise-optix", "%s: unknown command flag", iter->c_str());
        else if (inFilename.empty()) {
            inFilename = *iter;
        } else
            usage("denoise-optix", "multiple input filenames provided.");
    }
    if (inFilename.empty())
        usage("denoise-optix", "input image filename must be provided.");
    if (outFilename.empty())
        usage("denoise-optix", "output image filename must be provided.");

    ImageAndMetadata im = Image::Read(inFilename);
    Image &image = im.image;

    CUDA_CHECK(cudaFree(nullptr));

    int nLayers = 3;
    bool oldNormalNaming = false;
    ImageChannelDesc desc[3] = {
        image.GetChannelDesc({"R", "G", "B"}),
        image.GetChannelDesc({"Albedo.R", "Albedo.G", "Albedo.B"}),
        image.GetChannelDesc({"Ns.X", "Ns.Y", "Ns.Z"})};
    if (!desc[0]) {
        Error("%s: image doesn't have R, G, B channels.", inFilename);
        return 1;
    }
    if (!desc[1]) {
        Warning("%s: image doesn't have Albedo.{R,G,B} channels. "
                "Denoising quality may suffer.",
                inFilename);
        nLayers = 1;
    }
    if (!desc[2]) {
        // Try the old naming scheme
        desc[2] = image.GetChannelDesc({"Nsx", "Nsy", "Nsz"});
        if (desc[2])
            oldNormalNaming = true;
        else {
            Warning("%s: image doesn't have Ns.X, Ns.Y, Ns.Z channels. "
                    "Denoising quality may suffer.",
                    inFilename);
            nLayers = 1;
        }
    }

    Denoiser denoiser((Vector2i)image.Resolution(), nLayers == 3);

    size_t imageBytes = 3 * image.Resolution().x * image.Resolution().y * sizeof(float);

    auto copyChannelsToGPU = [&](std::array<std::string, 3> ch, bool flipZ = false) {
        void *bufGPU;
        CUDA_CHECK(cudaMalloc(&bufGPU, imageBytes));
        std::vector<float> hostStaging(imageBytes / sizeof(float));

        ImageChannelDesc desc = image.GetChannelDesc(ch);
        CHECK(desc);
        int offset = 0;
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                ImageChannelValues v = image.GetChannels({x, y}, desc);
                if (flipZ)
                    v[2] *= -1;  // flip normal's z--right handed...
                for (int c = 0; c < 3; ++c)
                    hostStaging[offset++] = v[c];
            }
        CUDA_CHECK(
            cudaMemcpy(bufGPU, hostStaging.data(), imageBytes, cudaMemcpyHostToDevice));
        return bufGPU;
    };
    RGB *rgbGPU = (RGB *)copyChannelsToGPU({"R", "G", "B"});

    RGB *albedoGPU = nullptr;
    Normal3f *normalGPU = nullptr;
    if (nLayers == 3) {
        albedoGPU = (RGB *)copyChannelsToGPU({"Albedo.R", "Albedo.G", "Albedo.B"});
        if (oldNormalNaming)
            normalGPU = (Normal3f *)copyChannelsToGPU({"Nsx", "Nsy", "Nsz"}, true);
        else
            normalGPU = (Normal3f *)copyChannelsToGPU({"Ns.X", "Ns.Y", "Ns.Z"}, true);
    }

    RGB *rgbResultGPU;
    CUDA_CHECK(cudaMalloc(&rgbResultGPU, imageBytes));

    denoiser.Denoise(rgbGPU, normalGPU, albedoGPU, rgbResultGPU);

    CUDA_CHECK(cudaDeviceSynchronize());

    Image result(PixelFormat::Float, image.Resolution(), {"R", "G", "B"});
    CUDA_CHECK(cudaMemcpy(result.RawPointer({0, 0}), (const void *)rgbResultGPU,
                          imageBytes, cudaMemcpyDeviceToHost));

    ImageMetadata outMetadata;
    outMetadata.cameraFromWorld = im.metadata.cameraFromWorld;
    outMetadata.NDCFromWorld = im.metadata.NDCFromWorld;
    outMetadata.pixelBounds = im.metadata.pixelBounds;
    outMetadata.fullResolution = im.metadata.fullResolution;
    outMetadata.colorSpace = im.metadata.colorSpace;

    CHECK(result.Write(outFilename, outMetadata));

    return 0;
}

int scale_optix(std::vector<std::string> args) {
    std::string inFilename, outFilename;

    auto onError = [](const std::string &err) {
        usage("scale-optix", "%s", err.c_str());
        exit(1);
    };
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if (ParseArg(&iter, args.end(), "outfile", &outFilename, onError)) {
            // success
        } else if ((*iter)[0] == '-')
            usage("scale-optix", "%s: unknown command flag", iter->c_str());
        else if (inFilename.empty()) {
            inFilename = *iter;
        } else
            usage("scale-optix", "multiple input filenames provided.");
    }
    if (inFilename.empty())
        usage("scale-optix", "input image filename must be provided.");
    if (outFilename.empty())
        usage("scale-optix", "output image filename must be provided.");

    ImageAndMetadata im = Image::Read(inFilename);
    Image &image = im.image;

    CUDA_CHECK(cudaFree(nullptr));

    int nLayers = 3;
    bool oldNormalNaming = false;
    ImageChannelDesc desc[3] = {
        image.GetChannelDesc({"R", "G", "B"}),
        image.GetChannelDesc({"Albedo.R", "Albedo.G", "Albedo.B"}),
        image.GetChannelDesc({"Ns.X", "Ns.Y", "Ns.Z"})};
    if (!desc[0]) {
        Error("%s: image doesn't have R, G, B channels.", inFilename);
        return 1;
    }
    if (!desc[1]) {
        Warning("%s: image doesn't have Albedo.{R,G,B} channels. "
                "Denoising quality may suffer.",
                inFilename);
        nLayers = 1;
    }
    if (!desc[2]) {
        // Try the old naming scheme
        desc[2] = image.GetChannelDesc({"Nsx", "Nsy", "Nsz"});
        if (desc[2])
            oldNormalNaming = true;
        else {
            Warning("%s: image doesn't have Ns.X, Ns.Y, Ns.Z channels. "
                    "Denoising quality may suffer.",
                    inFilename);
            nLayers = 1;
        }
    }

    Scaler scaler((Vector2i)image.Resolution(), nLayers == 3);

    size_t imageBytes = 3 * image.Resolution().x * image.Resolution().y * sizeof(float);

    auto copyChannelsToGPU = [&](std::array<std::string, 3> ch, bool flipZ = false) {
        void *bufGPU;
        CUDA_CHECK(cudaMalloc(&bufGPU, imageBytes));
        std::vector<float> hostStaging(imageBytes / sizeof(float));

        ImageChannelDesc desc = image.GetChannelDesc(ch);
        CHECK(desc);
        int offset = 0;
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                ImageChannelValues v = image.GetChannels({x, y}, desc);
                if (flipZ)
                    v[2] *= -1;  // flip normal's z--right handed...
                for (int c = 0; c < 3; ++c)
                    hostStaging[offset++] = v[c];
            }
        CUDA_CHECK(
            cudaMemcpy(bufGPU, hostStaging.data(), imageBytes, cudaMemcpyHostToDevice));
        return bufGPU;
    };
    RGB *rgbGPU = (RGB *)copyChannelsToGPU({"R", "G", "B"});

    RGB *albedoGPU = nullptr;
    Normal3f *normalGPU = nullptr;
    if (nLayers == 3) {
        albedoGPU = (RGB *)copyChannelsToGPU({"Albedo.R", "Albedo.G", "Albedo.B"});
        if (oldNormalNaming)
            normalGPU = (Normal3f *)copyChannelsToGPU({"Nsx", "Nsy", "Nsz"}, true);
        else
            normalGPU = (Normal3f *)copyChannelsToGPU({"Ns.X", "Ns.Y", "Ns.Z"}, true);
    }

    RGB *rgbResultGPU;
    Point2i destRes(image.Resolution().x * 2, image.Resolution().y * 2);
    size_t destBytes = 3 * destRes.x * destRes.y * sizeof(float);
    CUDA_CHECK(cudaMalloc(&rgbResultGPU, destBytes));

    scaler.Scale(rgbGPU, normalGPU, albedoGPU, rgbResultGPU);

    CUDA_CHECK(cudaDeviceSynchronize());

    Image result(PixelFormat::Float, destRes, {"R", "G", "B"});
    CUDA_CHECK(cudaMemcpy(result.RawPointer({0, 0}), (const void *)rgbResultGPU,
                          destBytes, cudaMemcpyDeviceToHost));

    ImageMetadata outMetadata;
    outMetadata.cameraFromWorld = im.metadata.cameraFromWorld;
    outMetadata.NDCFromWorld = im.metadata.NDCFromWorld;
    outMetadata.pixelBounds = Bounds2i(
        {im.metadata.pixelBounds->pMin.x * 2, im.metadata.pixelBounds->pMin.y * 2},
        {im.metadata.pixelBounds->pMax.x * 2, im.metadata.pixelBounds->pMax.y * 2});
    outMetadata.fullResolution = destRes;
    outMetadata.colorSpace = im.metadata.colorSpace;

    CHECK(result.Write(outFilename, outMetadata));

    return 0;
}
#endif  // PBRT_BUILD_GPU_RENDERER

int main(int argc, char *argv[]) {
    PBRTOptions opt;
    InitPBRT(opt);

    if (argc < 2) {
        help();
        return 0;
    }

    std::vector<std::string> args = GetCommandLineArguments(argv);
    std::string cmd = args[0];
    args.erase(args.begin());

    if (cmd == "average")
        return average(args);
    else if (cmd == "assemble")
        return assemble(args);
    else if (cmd == "bloom")
        return bloom(args);
    else if (cmd == "cat")
        return cat(args);
    else if (cmd == "convert")
        return convert(args);
    else if (cmd == "diff")
        return diff(args);
#ifdef PBRT_BUILD_GPU_RENDERER
    else if (cmd == "denoise-optix")
        return denoise_optix(args);
    else if (cmd == "scale-optix")
        return scale_optix(args);
#endif  // PBRT_BUILD_GPU_RENDERER
    else if (cmd == "error")
        return error(args);
    else if (cmd == "falsecolor")
        return falsecolor(args);
    else if (cmd == "help" || cmd == "-help" || cmd == "--help" || cmd == "-h")
        help(args);
    else if (cmd == "info")
        return info(args);
    else if (cmd == "makeequiarea")
        return makeequiarea(args);
    else if (cmd == "makeemitters")
        return makeemitters(args);
    else if (cmd == "makesky")
        return makesky(args);
    else if (cmd == "whitebalance")
        return whitebalance(args);
    else if (cmd == "scalenormalmap")
        return scalenormalmap(args);
    else if (cmd == "splitn")
        return splitn(args);
    else if (cmd == "noisybit") {
        // hack for brute force computation of ideal filter weights.

        argv += 2;
        std::string filename;
        std::array<int, 2> pixel = {0, 0};
        int width = 10;
        Float sigma = 1;
        int nInstances = 100;

        for (auto iter = args.begin(); iter != args.end(); ++iter) {
            auto onError = [](const std::string &err) {
                usage("%s", err.c_str());
                exit(1);
            };
            if (ParseArg(&iter, args.end(), "pixel", pstd::MakeSpan(pixel), onError) ||
                ParseArg(&iter, args.end(), "width", &width, onError) ||
                ParseArg(&iter, args.end(), "sigma", &sigma, onError) ||
                ParseArg(&iter, args.end(), "n", &nInstances, onError))
                ;  // yaay
            else if (filename.empty()) {
                filename = *iter;
            } else
                onError(StringPrintf("unexpected argument \"%s\"", iter->c_str()));
        }
        CHECK(!filename.empty());

        ImageAndMetadata imRead = Image::Read(filename);
        ImageChannelDesc rgbDesc = imRead.image.GetChannelDesc({"R", "G", "B"});
        CHECK((bool)rgbDesc);
        Image image = imRead.image.SelectChannels(rgbDesc);

        if (pixel[0] - width < 0 || pixel[0] + width >= image.Resolution().x ||
            pixel[0] - width < 0 || pixel[0] + width >= image.Resolution().y) {
            Error(
                "%s: pixel (%d, %d) with width %d doesn't work with resolution (%d, %d).",
                filename, pixel[0], pixel[0], width, image.Resolution().x,
                image.Resolution().y);
            return 1;
        }

        int nPixels = Sqr(2 * width + 1);
        CHECK_GE(3 * nInstances, nPixels);  // want to be overconstrained

        RNG rng;
        //
        FILE *f = fopen("m.csv", "w");
        for (int i = 0; i < nInstances; ++i)
            for (int c = 0; c < 3; ++c) {
                for (int dx = -width; dx <= width; ++dx)
                    for (int dy = -width; dy <= width; ++dy) {
                        // Float noise = .1 * std::exp(-rng.Uniform<Float>() *
                        // 3); if (rng.Uniform<Float>() < .5) noise = -noise;
                        Float noise = SampleNormal(rng.Uniform<Float>(), 0., .1);
                        // TODO: use sigma, make this controllable, etc.
                        fprintf(f, "%c%f ", (dx > -width || dy > -width) ? ',' : ' ',
                                image.GetChannel({pixel[0] + dx, pixel[1] + dy}, c)
                                    // *  (.95 + .05 * rng.Uniform<Float>())
                                    + noise);
                    }
                fprintf(f, "\n");
            }
        fclose(f);

        // what it should equal
        f = fopen("b.csv", "w");
        for (int i = 0; i < nInstances; ++i)
            for (int c = 0; c < 3; ++c)
                fprintf(f, "%f\n", image.GetChannel({pixel[0], pixel[1]}, c));
        fclose(f);

        /*
          LeastSquares[Import["m.csv"], Import["b.csv"]]
          ArrayPlot[ArrayReshape[ %, {21, 21}], ColorFunction -> Function[a,
          GrayLevel[4 a]]]
        */
    } else {
        fprintf(stderr, "imgtool: unknown command \"%s\".\n", cmd.c_str());
        help();
        CleanupPBRT();
        return 1;
    }

    CleanupPBRT();

    return 0;
}
