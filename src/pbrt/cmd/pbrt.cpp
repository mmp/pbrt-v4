// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/cpu/render.h>
#include <pbrt/options.h>
#include <pbrt/parsedscene.h>
#include <pbrt/parser.h>
#include <pbrt/util/args.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/string.h>
#include <pbrt/wavefront/wavefront.h>

using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pbrt: %s\n\n", msg.c_str());

    fprintf(stderr,
            R"(usage: pbrt [<options>] <filename.pbrt...>

Rendering options:
  --cropwindow <x0,x1,y0,y1>   Specify an image crop window w.r.t. [0,1]^2
  --debugstart <values>        Inform the Integrator where to start rendering for
                               faster debugging. (<values> are Integrator-specific
                               and come from error message text.)
  --disable-pixel-jitter       Always sample pixels at their centers.
  --disable-wavelength-jitter  Always sample the same %d wavelengths of light.
  --display-server <addr:port> Connect to display server at given address and port
                               to display the image as it's being rendered.
  --force-diffuse              Convert all materials to be diffuse.)"
#ifdef PBRT_BUILD_GPU_RENDERER
            R"(
  --gpu                        Use the GPU for rendering. (Default: disabled)
  --gpu-device <index>         Use specified GPU for rendering.)"
#endif
            R"(
  --help                       Print this help text.
  --mse-reference-image        Filename for reference image to use for MSE computation.
  --mse-reference-out          File to write MSE error vs spp results.
  --nthreads <num>             Use specified number of threads for rendering.
  --outfile <filename>         Write the final image to the given filename.
  --pixel <x,y>                Render just the specified pixel.
  --pixelbounds <x0,x1,y0,y1>  Specify an image crop window w.r.t. pixel coordinates.
  --pixelmaterial <x,y>        Print information about the material visible in the
                               center of the pixel's extent.
  --pixelstats                 Record per-pixel statistics and write additional images
                               with their values.
  --quick                      Automatically reduce a number of quality settings
                               to render more quickly.
  --quiet                      Suppress all text output other than error messages.
  --render-coord-sys <name>    Coordinate system to use for the scene when rendering,
                               where name is "camera", "cameraworld", or "world".
  --seed <n>                   Set random number generator seed. Default: 0.
  --stats                      Print various statistics after rendering completes.
  --spp <n>                    Override number of pixel samples specified in scene
                               description file.
  --wavefront                  Use wavefront volumetric path integrator.
  --write-partial-images       Periodically write the current image to disk, rather
                               than waiting for the end of rendering. Default: disabled.

Logging options:
  --log-file <filename>        Filename to write logging messages to. Default: none;
                               messages are printed to standard error. Implies
                               --log-level verbose if specified.
  --log-level <level>          Log messages at or above this level, where <level>
                               is "verbose", "error", or "fatal". Default: "error".

Reformatting options:
  --format                     Print a reformatted version of the input file(s) to
                               standard output. Does not render an image.
  --toply                      Print a reformatted version of the input file(s) to
                               standard output and convert all triangle meshes to
                               PLY files. Does not render an image.
  --upgrade                    Upgrade a pbrt-v3 file to pbrt-v4's format.
)",
            NSpectrumSamples);
    exit(msg.empty() ? 0 : 1);
}

// main program
int main(int argc, char *argv[]) {
    PBRTOptions options;
    std::vector<std::string> filenames;
    std::string logLevel = "error";
    std::string renderCoordSys = "cameraworld";
    bool format = false, toPly = false;

    // Process command-line arguments
    ++argv;
    while (*argv != nullptr) {
        if ((*argv)[0] != '-') {
            filenames.push_back(*argv);
            ++argv;
            continue;
        }

        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        std::string cropWindow, pixelBounds, pixel, pixelMaterial;
        if (ParseArg(&argv, "cropwindow", &cropWindow, onError)) {
            std::vector<Float> c = SplitStringToFloats(cropWindow, ',');
            if (c.size() != 4) {
                usage("Didn't find four values after --cropwindow");
                return 1;
            }
            options.cropWindow = Bounds2f(Point2f(c[0], c[2]), Point2f(c[1], c[3]));
        } else if (ParseArg(&argv, "pixel", &pixel, onError)) {
            std::vector<int> p = SplitStringToInts(pixel, ',');
            if (p.size() != 2) {
                usage("Didn't find two values after --pixel");
                return 1;
            }
            options.pixelBounds =
                Bounds2i(Point2i(p[0], p[1]), Point2i(p[0] + 1, p[1] + 1));
        } else if (ParseArg(&argv, "pixelbounds", &pixelBounds, onError)) {
            std::vector<int> p = SplitStringToInts(pixelBounds, ',');
            if (p.size() != 4) {
                usage("Didn't find four integer values after --pixelbounds");
                return 1;
            }
            options.pixelBounds = Bounds2i(Point2i(p[0], p[2]), Point2i(p[1], p[3]));
        } else if (ParseArg(&argv, "pixelmaterial", &pixelMaterial, onError)) {
            std::vector<int> p = SplitStringToInts(pixelMaterial, ',');
            if (p.size() != 2) {
                usage("Didn't find two values after --pixelmaterial");
                return 1;
            }
            options.pixelMaterial = Point2i(p[0], p[1]);
        } else if (
#ifdef PBRT_BUILD_GPU_RENDERER
            ParseArg(&argv, "gpu", &options.useGPU, onError) ||
            ParseArg(&argv, "gpu-device", &options.gpuDevice, onError) ||
#endif
            ParseArg(&argv, "wavefront", &options.wavefront, onError) ||
            ParseArg(&argv, "debugstart", &options.debugStart, onError) ||
            ParseArg(&argv, "disable-pixel-jitter", &options.disablePixelJitter,
                     onError) ||
            ParseArg(&argv, "disable-wavelength-jitter", &options.disableWavelengthJitter,
                     onError) ||
            ParseArg(&argv, "display-server", &options.displayServer, onError) ||
            ParseArg(&argv, "force-diffuse", &options.forceDiffuse, onError) ||
            ParseArg(&argv, "format", &format, onError) ||
            ParseArg(&argv, "log-level", &logLevel, onError) ||
            ParseArg(&argv, "log-file", &options.logFile, onError) ||
            ParseArg(&argv, "mse-reference-image", &options.mseReferenceImage, onError) ||
            ParseArg(&argv, "mse-reference-out", &options.mseReferenceOutput, onError) ||
            ParseArg(&argv, "nthreads", &options.nThreads, onError) ||
            ParseArg(&argv, "outfile", &options.imageFile, onError) ||
            ParseArg(&argv, "pixelstats", &options.recordPixelStatistics, onError) ||
            ParseArg(&argv, "quick", &options.quickRender, onError) ||
            ParseArg(&argv, "quiet", &options.quiet, onError) ||
            ParseArg(&argv, "render-coord-sys", &renderCoordSys, onError) ||
            ParseArg(&argv, "seed", &options.seed, onError) ||
            ParseArg(&argv, "spp", &options.pixelSamples, onError) ||
            ParseArg(&argv, "stats", &options.printStatistics, onError) ||
            ParseArg(&argv, "toply", &toPly, onError) ||
            ParseArg(&argv, "write-partial-images", &options.writePartialImages,
                     onError) ||
            ParseArg(&argv, "upgrade", &options.upgrade, onError)) {
            // success
        } else if ((strcmp(*argv, "--help") == 0) || (strcmp(*argv, "-help") == 0) ||
                   (strcmp(*argv, "-h") == 0)) {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *argv));
            return 1;
        }
    }

    // Print welcome banner
    if (!options.quiet && !format && !toPly && !options.upgrade) {
        printf("pbrt version 4 (built %s at %s)\n", __DATE__, __TIME__);
#ifdef PBRT_DEBUG_BUILD
        LOG_VERBOSE("Running debug build");
        printf("*** DEBUG BUILD ***\n");
#endif
        printf("Copyright (c)1998-2021 Matt Pharr, Wenzel Jakob, and Greg Humphreys.\n");
        printf("The source code to pbrt (but *not* the book contents) is covered "
               "by the Apache 2.0 License.\n");
        printf("See the file LICENSE.txt for the conditions of the license.\n");
        fflush(stdout);
    }

    // Check validity of provided arguments
    if (renderCoordSys == "camera")
        options.renderingSpace = RenderingCoordinateSystem::Camera;
    else if (renderCoordSys == "cameraworld")
        options.renderingSpace = RenderingCoordinateSystem::CameraWorld;
    else if (renderCoordSys == "world")
        options.renderingSpace = RenderingCoordinateSystem::World;
    else
        ErrorExit("%s: unknown rendering coordinate system.", renderCoordSys);

    if (!options.mseReferenceImage.empty() && options.mseReferenceOutput.empty())
        ErrorExit("Must provide MSE reference output filename via "
                  "--mse-reference-out");
    if (!options.mseReferenceOutput.empty() && options.mseReferenceImage.empty())
        ErrorExit("Must provide MSE reference image via --mse-reference-image");

    if (options.pixelMaterial && options.useGPU) {
        Warning("Disabling --use-gpu since --pixelmaterial was specified.");
        options.useGPU = false;
    }

    if (options.useGPU && options.wavefront)
        Warning("Both --gpu and --wavefront were specified; --gpu takes precidence.");

    if (options.pixelMaterial && options.wavefront) {
        Warning("Disabling --wavefront since --pixelmaterial was specified.");
        options.wavefront = false;
    }

    options.logLevel = LogLevelFromString(logLevel);

    // Initialize pbrt
    InitPBRT(options);

    if (format || toPly || options.upgrade) {
        FormattingScene formattingScene(toPly, options.upgrade);
        ParseFiles(&formattingScene, filenames);
    } else {
        // Parse provided scene description files
        ParsedScene scene;
        ParseFiles(&scene, filenames);

        // Render the scene
        if (options.useGPU || options.wavefront)
            RenderWavefront(scene);
        else
            CPURender(scene);

        LOG_VERBOSE("Memory used after post-render cleanup: %s", GetCurrentRSS());
        // Clean up after rendering the scene
        CleanupPBRT();
    }
    return 0;
}
