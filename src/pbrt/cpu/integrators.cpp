// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/integrators.h>

#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/shapes.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/display.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/string.h>

#include <algorithm>

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// RandomWalkIntegrator Method Definitions
std::unique_ptr<RandomWalkIntegrator> RandomWalkIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    return std::make_unique<RandomWalkIntegrator>(maxDepth, camera, sampler, aggregate,
                                                  lights);
}

std::string RandomWalkIntegrator::ToString() const {
    return StringPrintf("[ RandomWalkIntegrator maxDepth: %d ]", maxDepth);
}

// Integrator Method Definitions
Integrator::~Integrator() {}

// ImageTileIntegrator Method Definitions
void ImageTileIntegrator::Render() {
    // Handle debugStart, if set
    if (!Options->debugStart.empty()) {
        std::vector<int> c = SplitStringToInts(Options->debugStart, ',');
        if (c.empty())
            ErrorExit("Didn't find integer values after --debugstart: %s",
                      Options->debugStart);
        if (c.size() != 3)
            ErrorExit("Didn't find three integer values after --debugstart: %s",
                      Options->debugStart);

        Point2i pPixel(c[0], c[1]);
        int sampleIndex = c[2];

        ScratchBuffer scratchBuffer(65536);
        Sampler tileSampler = samplerPrototype.Clone(Allocator());
        tileSampler.StartPixelSample(pPixel, sampleIndex);

        EvaluatePixelSample(pPixel, sampleIndex, tileSampler, scratchBuffer);

        return;
    }

    thread_local Point2i threadPixel;
    thread_local int threadSampleIndex;
    CheckCallbackScope _([&]() {
        return StringPrintf("Rendering failed at pixel (%d, %d) sample %d. Debug with "
                            "\"--debugstart %d,%d,%d\"\n",
                            threadPixel.x, threadPixel.y, threadSampleIndex,
                            threadPixel.x, threadPixel.y, threadSampleIndex);
    });

    // Declare common variables for rendering image in tiles
    ThreadLocal<ScratchBuffer> scratchBuffers([]() { return ScratchBuffer(); });

    ThreadLocal<Sampler> samplers([this]() { return samplerPrototype.Clone(); });

    Bounds2i pixelBounds = camera.GetFilm().PixelBounds();
    int spp = samplerPrototype.SamplesPerPixel();
    ProgressReporter progress(int64_t(spp) * pixelBounds.Area(), "Rendering",
                              Options->quiet);

    int waveStart = 0, waveEnd = 1, nextWaveSize = 1;

    if (Options->recordPixelStatistics)
        StatsEnablePixelStats(pixelBounds,
                              RemoveExtension(camera.GetFilm().GetFilename()));
    // Handle MSE reference image, if provided
    pstd::optional<Image> referenceImage;
    FILE *mseOutFile = nullptr;
    if (!Options->mseReferenceImage.empty()) {
        auto mse = Image::Read(Options->mseReferenceImage);
        referenceImage = mse.image;

        Bounds2i msePixelBounds =
            mse.metadata.pixelBounds
                ? *mse.metadata.pixelBounds
                : Bounds2i(Point2i(0, 0), referenceImage->Resolution());
        if (!Inside(pixelBounds, msePixelBounds))
            ErrorExit("Output image pixel bounds %s aren't inside the MSE "
                      "image's pixel bounds %s.",
                      pixelBounds, msePixelBounds);

        // Transform the pixelBounds of the image we're rendering to the
        // coordinate system with msePixelBounds.pMin at the origin, which
        // in turn gives us the section of the MSE image to crop. (This is
        // complicated by the fact that Image doesn't support pixel
        // bounds...)
        Bounds2i cropBounds(Point2i(pixelBounds.pMin - msePixelBounds.pMin),
                            Point2i(pixelBounds.pMax - msePixelBounds.pMin));
        *referenceImage = referenceImage->Crop(cropBounds);
        CHECK_EQ(referenceImage->Resolution(), Point2i(pixelBounds.Diagonal()));

        mseOutFile = FOpenWrite(Options->mseReferenceOutput);
        if (!mseOutFile)
            ErrorExit("%s: %s", Options->mseReferenceOutput, ErrorString());
    }

    // Connect to display server if needed
    if (!Options->displayServer.empty()) {
        Film film = camera.GetFilm();
        DisplayDynamic(film.GetFilename(), Point2i(pixelBounds.Diagonal()),
                       {"R", "G", "B"},
                       [&](Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                           int index = 0;
                           for (Point2i p : b) {
                               RGB rgb = film.GetPixelRGB(pixelBounds.pMin + p,
                                                          2.f / (waveStart + waveEnd));
                               for (int c = 0; c < 3; ++c)
                                   displayValue[c][index] = rgb[c];
                               ++index;
                           }
                       });
    }

    // Render image in waves
    while (waveStart < spp) {
        // Render current wave's image tiles in parallel
        ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
            // Render image tile given by _tileBounds_
            ScratchBuffer &scratchBuffer = scratchBuffers.Get();
            Sampler &sampler = samplers.Get();
            PBRT_DBG("Starting image tile (%d,%d)-(%d,%d) waveStart %d, waveEnd %d\n",
                     tileBounds.pMin.x, tileBounds.pMin.y, tileBounds.pMax.x,
                     tileBounds.pMax.y, waveStart, waveEnd);
            for (Point2i pPixel : tileBounds) {
                StatsReportPixelStart(pPixel);
                threadPixel = pPixel;
                // Render samples in pixel _pPixel_
                for (int sampleIndex = waveStart; sampleIndex < waveEnd; ++sampleIndex) {
                    threadSampleIndex = sampleIndex;
                    sampler.StartPixelSample(pPixel, sampleIndex);
                    EvaluatePixelSample(pPixel, sampleIndex, sampler, scratchBuffer);
                    scratchBuffer.Reset();
                }

                StatsReportPixelEnd(pPixel);
            }
            PBRT_DBG("Finished image tile (%d,%d)-(%d,%d)\n", tileBounds.pMin.x,
                     tileBounds.pMin.y, tileBounds.pMax.x, tileBounds.pMax.y);
            progress.Update((waveEnd - waveStart) * tileBounds.Area());
        });

        // Update start and end wave
        waveStart = waveEnd;
        waveEnd = std::min(spp, waveEnd + nextWaveSize);
        if (!referenceImage)
            nextWaveSize = std::min(2 * nextWaveSize, 64);
        if (waveStart == spp)
            progress.Done();

        // Optionally write current image to disk
        if (waveStart == spp || Options->writePartialImages || referenceImage) {
            LOG_VERBOSE("Writing image with spp = %d", waveStart);
            ImageMetadata metadata;
            metadata.renderTimeSeconds = progress.ElapsedSeconds();
            metadata.samplesPerPixel = waveStart;
            if (referenceImage) {
                ImageMetadata filmMetadata;
                Image filmImage =
                    camera.GetFilm().GetImage(&filmMetadata, 1.f / waveStart);
                ImageChannelValues mse =
                    filmImage.MSE(filmImage.AllChannelsDesc(), *referenceImage);
                fprintf(mseOutFile, "%d, %.9g\n", waveStart, mse.Average());
                metadata.MSE = mse.Average();
                fflush(mseOutFile);
            }
            if (waveStart == spp || Options->writePartialImages) {
                camera.InitMetadata(&metadata);
                camera.GetFilm().WriteImage(metadata, 1.0f / waveStart);
            }
        }
    }

    if (mseOutFile)
        fclose(mseOutFile);
    DisconnectFromDisplayServer();
    LOG_VERBOSE("Rendering finished");
}

// RayIntegrator Method Definitions
void RayIntegrator::EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                                        ScratchBuffer &scratchBuffer) {
    // Sample wavelengths for the ray
    Float lu = sampler.Get1D();
    if (Options->disableWavelengthJitter)
        lu = 0.5;
    SampledWavelengths lambda = camera.GetFilm().SampleWavelengths(lu);

    // Initialize _CameraSample_ for current sample
    Filter filter = camera.GetFilm().GetFilter();
    CameraSample cameraSample = GetCameraSample(sampler, pPixel, filter);

    // Generate camera ray for current sample
    pstd::optional<CameraRayDifferential> cameraRay =
        camera.GenerateRayDifferential(cameraSample, lambda);

    // Trace _cameraRay_ if valid
    SampledSpectrum L(0.);
    VisibleSurface visibleSurface;
    if (cameraRay) {
        // Double check that the ray's direction is normalized.
        DCHECK_GT(Length(cameraRay->ray.d), .999f);
        DCHECK_LT(Length(cameraRay->ray.d), 1.001f);
        // Scale camera ray differentials based on image sampling rate
        Float rayDiffScale =
            std::max<Float>(.125f, 1 / std::sqrt((Float)sampler.SamplesPerPixel()));
        if (!Options->disablePixelJitter)
            cameraRay->ray.ScaleDifferentials(rayDiffScale);

        ++nCameraRays;
        // Evaluate radiance along camera ray
        bool initializeVisibleSurface = camera.GetFilm().UsesVisibleSurface();
        L = cameraRay->weight * Li(cameraRay->ray, lambda, sampler, scratchBuffer,
                                   initializeVisibleSurface ? &visibleSurface : nullptr);

        // Issue warning if unexpected radiance value is returned
        if (L.HasNaNs()) {
            LOG_ERROR("Not-a-number radiance value returned for pixel (%d, "
                      "%d), sample %d. Setting to black.",
                      pPixel.x, pPixel.y, sampleIndex);
            L = SampledSpectrum(0.f);
        } else if (IsInf(L.y(lambda))) {
            LOG_ERROR("Infinite radiance value returned for pixel (%d, %d), "
                      "sample %d. Setting to black.",
                      pPixel.x, pPixel.y, sampleIndex);
            L = SampledSpectrum(0.f);
        }

        if (cameraRay)
            PBRT_DBG(
                "%s\n",
                StringPrintf("Camera sample: %s -> ray %s -> L = %s, visibleSurface %s",
                             cameraSample, cameraRay->ray, L,
                             (visibleSurface ? visibleSurface.ToString() : "(none)"))
                    .c_str());
        else
            PBRT_DBG("%s\n",
                     StringPrintf("Camera sample: %s -> no ray generated", cameraSample)
                         .c_str());
    }

    // Add camera ray's contribution to image
    camera.GetFilm().AddSample(pPixel, L, lambda, &visibleSurface,
                               cameraSample.filterWeight);
}

// Integrator Utility Functions
STAT_COUNTER("Intersections/Regular ray intersection tests", nIntersectionTests);
STAT_COUNTER("Intersections/Shadow ray intersection tests", nShadowTests);

// Integrator Method Definitions
pstd::optional<ShapeIntersection> Integrator::Intersect(const Ray &ray,
                                                        Float tMax) const {
    ++nIntersectionTests;
    DCHECK_NE(ray.d, Vector3f(0, 0, 0));
    if (aggregate)
        return aggregate.Intersect(ray, tMax);
    else
        return {};
}

bool Integrator::IntersectP(const Ray &ray, Float tMax) const {
    ++nShadowTests;
    DCHECK_NE(ray.d, Vector3f(0, 0, 0));
    if (aggregate)
        return aggregate.IntersectP(ray, tMax);
    else
        return false;
}

SampledSpectrum Integrator::Tr(const Interaction &p0, const Interaction &p1,
                               const SampledWavelengths &lambda) const {
    RNG rng(Hash(p0.p()), Hash(p1.p()));

    // :-(
    Ray ray =
        p0.IsSurfaceInteraction() ? p0.AsSurface().SpawnRayTo(p1) : p0.SpawnRayTo(p1);
    SampledSpectrum Tr(1.f), inv_w(1.f);
    if (LengthSquared(ray.d) == 0)
        return Tr;

    while (true) {
        pstd::optional<ShapeIntersection> si = Intersect(ray, 1 - ShadowEpsilon);
        // Handle opaque surface along ray's path
        if (si && si->intr.material)
            return SampledSpectrum(0.0f);

        // Update transmittance for current ray segment
        if (ray.medium) {
            Point3f pExit = ray(si ? si->tHit : (1 - ShadowEpsilon));
            ray.d = pExit - ray.o;

            SampledSpectrum T_maj =
                SampleT_maj(ray, 1.f, rng.Uniform<Float>(), rng, lambda,
                            [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                                SampledSpectrum T_maj) {
                                SampledSpectrum sigma_n =
                                    ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);

                                // ratio-tracking: only evaluate null scattering
                                Float pr = T_maj[0] * sigma_maj[0];
                                Tr *= T_maj * sigma_n / pr;
                                inv_w *= T_maj * sigma_maj / pr;

                                if (!Tr || !inv_w)
                                    return false;

                                return true;
                            });
            Tr *= T_maj / T_maj[0];
            inv_w *= T_maj / T_maj[0];
        }

        // Generate next ray segment or return final transmittance
        if (!si)
            break;
        ray = si->intr.SpawnRayTo(p1);
    }
    PBRT_DBG("%s\n", StringPrintf("Tr from %s to %s = %s", p0.pi, p1.pi, Tr).c_str());
    return Tr / inv_w.Average();
}

std::string Integrator::ToString() const {
    std::string s = StringPrintf("[ Integrator aggregate: %s lights[%d]: [ ", aggregate,
                                 lights.size());
    for (const auto &l : lights)
        s += StringPrintf("%s, ", l.ToString());
    s += StringPrintf("] infiniteLights[%d]: [ ", infiniteLights.size());
    for (const auto &l : infiniteLights)
        s += StringPrintf("%s, ", l.ToString());
    return s + " ]";
}

// SimplePathIntegrator Method Definitions
SimplePathIntegrator::SimplePathIntegrator(int maxDepth, bool sampleLights,
                                           bool sampleBSDF, Camera camera,
                                           Sampler sampler, Primitive aggregate,
                                           std::vector<Light> lights)
    : RayIntegrator(camera, sampler, aggregate, lights),
      maxDepth(maxDepth),
      sampleLights(sampleLights),
      sampleBSDF(sampleBSDF),
      lightSampler(lights, Allocator()) {}

SampledSpectrum SimplePathIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                         Sampler sampler, ScratchBuffer &scratchBuffer,
                                         VisibleSurface *) const {
    // Estimate radiance along ray using simple path tracing
    SampledSpectrum L(0.f), beta(1.f);
    bool specularBounce = true;
    int depth = 0;
    while (beta) {
        // Find next _SimplePathIntegrator_ vertex and accumulate contribution
        // Intersect _ray_ with scene
        pstd::optional<ShapeIntersection> si = Intersect(ray);

        // Account for infinite lights if ray has no intersection
        if (!si) {
            if (!sampleLights || specularBounce)
                for (const auto &light : infiniteLights)
                    L += beta * light.Le(ray, lambda);
            break;
        }

        // Account for emissive surface if light was not sampled
        SurfaceInteraction &isect = si->intr;
        if (!sampleLights || specularBounce)
            L += beta * isect.Le(-ray.d, lambda);

        // End path if maximum depth reached
        if (depth++ == maxDepth)
            break;

        // Get BSDF and skip over medium boundaries
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        // Sample direct illumination if _sampleLights_ is true
        Vector3f wo = -ray.d;
        if (sampleLights) {
            pstd::optional<SampledLight> sampledLight =
                lightSampler.Sample(sampler.Get1D());
            if (sampledLight) {
                // Sample point on _sampledLight_ to estimate direct illumination
                Point2f uLight = sampler.Get2D();
                pstd::optional<LightLiSample> ls =
                    sampledLight->light.SampleLi(isect, uLight, lambda);
                if (ls && ls->L && ls->pdf > 0) {
                    // Evaluate BSDF for light and possibly add scattered radiance
                    Vector3f wi = ls->wi;
                    SampledSpectrum f = bsdf.f(wo, wi) * AbsDot(wi, isect.shading.n);
                    if (f && Unoccluded(isect, ls->pLight))
                        L += beta * f * ls->L / (sampledLight->p * ls->pdf);
                }
            }
        }

        // Sample outgoing direction at intersection to continue path
        if (sampleBSDF) {
            // Sample BSDF for new path direction
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
            if (!bs)
                break;
            beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
            specularBounce = bs->IsSpecular();
            ray = isect.SpawnRay(bs->wi);

        } else {
            // Uniformly sample sphere or hemisphere to get new path direction
            Float pdf;
            Vector3f wi;
            BxDFFlags flags = bsdf.Flags();
            if (IsReflective(flags) && IsTransmissive(flags)) {
                wi = SampleUniformSphere(sampler.Get2D());
                pdf = UniformSpherePDF();
            } else {
                wi = SampleUniformHemisphere(sampler.Get2D());
                pdf = UniformHemispherePDF();
                if (IsReflective(flags) && Dot(wo, isect.n) * Dot(wi, isect.n) < 0)
                    wi = -wi;
                else if (IsTransmissive(flags) && Dot(wo, isect.n) * Dot(wi, isect.n) > 0)
                    wi = -wi;
            }
            beta *= bsdf.f(wo, wi) * AbsDot(wi, isect.shading.n) / pdf;
            specularBounce = false;
            ray = isect.SpawnRay(wi);
        }

        CHECK_GE(beta.y(lambda), 0.f);
        DCHECK(!IsInf(beta.y(lambda)));
    }
    return L;
}

std::string SimplePathIntegrator::ToString() const {
    return StringPrintf("[ SimplePathIntegrator maxDepth: %d sampleLights: %s "
                        "sampleBSDF: %s ]",
                        maxDepth, sampleLights, sampleBSDF);
}

std::unique_ptr<SimplePathIntegrator> SimplePathIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    bool sampleLights = parameters.GetOneBool("samplelights", true);
    bool sampleBSDF = parameters.GetOneBool("samplebsdf", true);
    return std::make_unique<SimplePathIntegrator>(maxDepth, sampleLights, sampleBSDF,
                                                  camera, sampler, aggregate, lights);
}

// LightPathIntegrator Method Definitions
LightPathIntegrator::LightPathIntegrator(int maxDepth, Camera camera, Sampler sampler,
                                         Primitive aggregate, std::vector<Light> lights)
    : ImageTileIntegrator(camera, sampler, aggregate, lights),
      maxDepth(maxDepth),
      lightSampler(lights, Allocator()) {}

void LightPathIntegrator::EvaluatePixelSample(Point2i pPixel, int sampleIndex,
                                              Sampler sampler,
                                              ScratchBuffer &scratchBuffer) {
    // Sample wavelengths for the ray
    Float lu = sampler.Get1D();
    if (Options->disableWavelengthJitter)
        lu = 0.5;
    SampledWavelengths lambda = camera.GetFilm().SampleWavelengths(lu);

    // Sample light to start light path
    Float ul = sampler.Get1D();
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(ul);
    if (!sampledLight)
        return;
    Light light = sampledLight->light;
    Float lightPDF = sampledLight->p;

    // Sample point on light source for light path
    Float time = camera.SampleTime(sampler.Get1D());
    Point2f ul0 = sampler.Get2D();
    Point2f ul1 = sampler.Get2D();
    pstd::optional<LightLeSample> les = light.SampleLe(ul0, ul1, lambda, time);
    if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L)
        return;

    // Add contribution of directly visible light source
    if (les->intr) {
        pstd::optional<CameraWiSample> cs =
            camera.SampleWi(*les->intr, sampler.Get2D(), lambda);
        if (cs && cs->pdf != 0) {
            if (Float pdf = light.PDF_Li(cs->pLens, -cs->wi); pdf > 0) {
                // Add light's emitted radiance if nonzero and light is visible
                SampledSpectrum Le =
                    light.L(les->intr->p(), les->intr->n, les->intr->uv, cs->wi, lambda);
                if (Le && Unoccluded(cs->pRef, cs->pLens)) {
                    // Compute visible light's path contribution and add to film
                    SampledSpectrum L = Le * DistanceSquared(cs->pRef.p(), cs->pLens.p()) * cs->Wi /
                                        (lightPDF * pdf * cs->pdf);
                    camera.GetFilm().AddSplat(cs->pRaster, L, lambda);
                }
            }
        }
    }

    // Follow light path and accumulate contributions to image
    int depth = 0;
    // Initialize light path ray and weighted path throughput _beta_
    RayDifferential ray(les->ray);
    SampledSpectrum beta =
        les->L * les->AbsCosTheta(ray.d) / (lightPDF * les->pdfPos * les->pdfDir);

    while (true) {
        // Intersect light path ray with scene
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (!si)
            break;
        SurfaceInteraction &isect = si->intr;

        // Get BSDF and skip over medium boundaries
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        // End path if maximum depth reached
        if (depth++ == maxDepth)
            break;

        // Splat contribution into film if intersection point is visible to camera
        Point2f u = sampler.Get2D();
        pstd::optional<CameraWiSample> cs = camera.SampleWi(isect, u, lambda);
        if (cs && cs->pdf != 0) {
            SampledSpectrum L = beta *
                                bsdf.f(isect.wo, cs->wi, TransportMode::Importance) *
                                AbsDot(cs->wi, isect.shading.n) * cs->Wi / cs->pdf;
            if (L && Unoccluded(cs->pRef, cs->pLens))
                camera.GetFilm().AddSplat(cs->pRaster, L, lambda);
        }

        // Sample BSDF and update light path state
        Float uc = sampler.Get1D();
        pstd::optional<BSDFSample> bs =
            bsdf.Sample_f(isect.wo, uc, sampler.Get2D(), TransportMode::Importance);
        if (!bs)
            break;
        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);
    }
}

std::string LightPathIntegrator::ToString() const {
    return StringPrintf("[ LightPathIntegrator maxDepth: %d lightSampler: %s ]", maxDepth,
                        lightSampler);
}

std::unique_ptr<LightPathIntegrator> LightPathIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    return std::make_unique<LightPathIntegrator>(maxDepth, camera, sampler, aggregate,
                                                 lights);
}

STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_PERCENT("Integrator/Regularized BSDFs", regularizedBSDFs, totalBSDFs);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

// PathIntegrator Method Definitions
PathIntegrator::PathIntegrator(int maxDepth, Camera camera, Sampler sampler,
                               Primitive aggregate, std::vector<Light> lights,
                               const std::string &lightSampleStrategy, bool regularize)
    : RayIntegrator(camera, sampler, aggregate, lights),
      maxDepth(maxDepth),
      lightSampler(LightSampler::Create(lightSampleStrategy, lights, Allocator())),
      regularize(regularize) {}

SampledSpectrum PathIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                   Sampler sampler, ScratchBuffer &scratchBuffer,
                                   VisibleSurface *visibleSurf) const {
    // Declare local variables for _PathIntegrator::Li()_
    SampledSpectrum L(0.f), beta(1.f);
    int depth = 0;

    Float bsdfPDF, etaScale = 1;
    bool specularBounce = false, anyNonSpecularBounces = false;
    LightSampleContext prevIntrCtx;

    // Sample path from camera and accumulate radiance estimate
    while (true) {
        // Trace ray and find closest path vertex and its BSDF
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        // Add emitted light at intersection point or from the environment
        if (!si) {
            // Incorporate emission from infinite lights for escaped ray
            for (const auto &light : infiniteLights) {
                SampledSpectrum Le = light.Le(ray, lambda);
                if (depth == 0 || specularBounce)
                    L += beta * Le;
                else {
                    // Compute MIS weight for infinite light
                    Float lightPDF = lightSampler.PMF(prevIntrCtx, light) *
                                     light.PDF_Li(prevIntrCtx, ray.d, true);
                    Float w_b = PowerHeuristic(1, bsdfPDF, 1, lightPDF);

                    L += beta * w_b * Le;
                }
            }

            break;
        }
        // Incorporate emission from surface hit by ray
        SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
        if (Le) {
            if (depth == 0 || specularBounce)
                L += beta * Le;
            else {
                // Compute MIS weight for area light
                Light areaLight(si->intr.areaLight);
                Float lightPDF = lightSampler.PMF(prevIntrCtx, areaLight) *
                                 areaLight.PDF_Li(prevIntrCtx, ray.d, true);
                Float w_l = PowerHeuristic(1, bsdfPDF, 1, lightPDF);

                L += beta * w_l * Le;
            }
        }

        SurfaceInteraction &isect = si->intr;
        // Get BSDF and skip over medium boundaries
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        // Initialize _visibleSurf_ at first intersection
        if (depth == 0 && visibleSurf) {
            // Estimate BSDF's albedo
            // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
            constexpr int nRhoSamples = 16;
            const Float ucRho[nRhoSamples] = {
                0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                0.9674518,  0.2995429,  0.5083201, 0.047338516};
            const Point2f uRho[nRhoSamples] = {
                Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

            SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

            *visibleSurf = VisibleSurface(isect, albedo, lambda);
        }

        // Possibly regularize the BSDF
        if (regularize && anyNonSpecularBounces) {
            ++regularizedBSDFs;
            bsdf.Regularize();
        }

        ++totalBSDFs;

        // End path if maximum depth reached
        if (depth++ == maxDepth)
            break;

        // Sample direct illumination from the light sources
        if (IsNonSpecular(bsdf.Flags())) {
            ++totalPaths;
            SampledSpectrum Ld = SampleLd(isect, &bsdf, lambda, sampler);
            if (!Ld)
                ++zeroRadiancePaths;
            L += beta * Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
        if (!bs)
            break;
        // Update path state variables after surface scattering
        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        bsdfPDF = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi) : bs->pdf;
        DCHECK(!IsInf(beta.y(lambda)));
        specularBounce = bs->IsSpecular();
        anyNonSpecularBounces |= !bs->IsSpecular();
        if (bs->IsTransmission())
            etaScale *= Sqr(bs->eta);
        prevIntrCtx = si->intr;

        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);

        // Possibly terminate the path with Russian roulette
        SampledSpectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
            Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q)
                break;
            beta /= 1 - q;
            DCHECK(!IsInf(beta.y(lambda)));
        }
    }
    pathLength << depth;
    return L;
}

SampledSpectrum PathIntegrator::SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                                         SampledWavelengths &lambda,
                                         Sampler sampler) const {
    // Initialize _LightSampleContext_ for light sampling
    LightSampleContext ctx(intr);
    // Try to nudge the light sampling position to correct side of the surface
    BxDFFlags flags = bsdf->Flags();
    if (IsReflective(flags) && !IsTransmissive(flags))
        ctx.pi = intr.OffsetRayOrigin(intr.wo);
    else if (IsTransmissive(flags) && !IsReflective(flags))
        ctx.pi = intr.OffsetRayOrigin(-intr.wo);

    // Choose a light source for the direct lighting calculation
    Float u = sampler.Get1D();
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(ctx, u);
    Point2f uLight = sampler.Get2D();
    if (!sampledLight)
        return {};

    // Sample a point on the light source for direct lighting
    Light light = sampledLight->light;
    DCHECK(light && sampledLight->p > 0);
    pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
    if (!ls || !ls->L || ls->pdf == 0)
        return {};

    // Evaluate BSDF for light sample and check light visibility
    Vector3f wo = intr.wo, wi = ls->wi;
    SampledSpectrum f = bsdf->f(wo, wi) * AbsDot(wi, intr.shading.n);
    if (!f || !Unoccluded(intr, ls->pLight))
        return {};

    // Return light's contribution to reflected radiance
    Float p_l = sampledLight->p * ls->pdf;
    if (IsDeltaLight(light.Type()))
        return ls->L * f / p_l;
    else {
        Float p_b = bsdf->PDF(wo, wi);
        Float w_l = PowerHeuristic(1, p_l, 1, p_b);
        return w_l * ls->L * f / p_l;
    }
}

std::string PathIntegrator::ToString() const {
    return StringPrintf("[ PathIntegrator maxDepth: %d lightSampler: %s regularize: %s ]",
                        maxDepth, lightSampler, regularize);
}

std::unique_ptr<PathIntegrator> PathIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    std::string lightStrategy = parameters.GetOneString("lightsampler", "bvh");
    bool regularize = parameters.GetOneBool("regularize", false);
    return std::make_unique<PathIntegrator>(maxDepth, camera, sampler, aggregate, lights,
                                            lightStrategy, regularize);
}

// SimpleVolPathIntegrator Method Definitions
SimpleVolPathIntegrator::SimpleVolPathIntegrator(int maxDepth, Camera camera,
                                                 Sampler sampler, Primitive aggregate,
                                                 std::vector<Light> lights)
    : RayIntegrator(camera, sampler, aggregate, lights), maxDepth(maxDepth) {
    for (Light light : lights) {
        if (IsDeltaLight(light.Type()))
            ErrorExit("SimpleVolPathIntegrator only supports area and infinite light "
                      "sources");
    }
}

SampledSpectrum SimpleVolPathIntegrator::Li(RayDifferential ray,
                                            SampledWavelengths &lambda, Sampler sampler,
                                            ScratchBuffer &buf, VisibleSurface *) const {
    // Declare local variables for delta tracking integration
    SampledSpectrum L(0.f);
    Float beta = 1.f;
    int depth = 0;

    // Terminate secondary wavelengths before starting random walk
    lambda.TerminateSecondary();

    while (true) {
        // Estimate radiance for ray path using delta tracking
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        bool scattered = false, terminated = false;
        if (ray.medium) {
            // Initialize _RNG_ for sampling the majorant transmittance
            uint64_t hash0 = Hash(sampler.Get1D());
            uint64_t hash1 = Hash(sampler.Get1D());
            RNG rng(hash0, hash1);

            // Sample medium using delta tracking
            Float tMax = si ? si->tHit : Infinity;
            Float u = sampler.Get1D();
            Float uMode = sampler.Get1D();
            SampleT_maj(ray, tMax, u, rng, lambda,
                        [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                            SampledSpectrum T_maj) {
                            // Compute medium event probabilities for interaction
                            Float pAbsorb = mp.sigma_a[0] / sigma_maj[0];
                            Float pScatter = mp.sigma_s[0] / sigma_maj[0];
                            Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);

                            // Randomly sample medium scattering event for delta tracking
                            int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, uMode);
                            if (mode == 0) {
                                // Handle absorption event for medium sample
                                L += beta * mp.Le;
                                terminated = true;
                                return false;

                            } else if (mode == 1) {
                                // Handle regular scattering event for medium sample
                                // Stop path sampling if maximum depth has been reached
                                if (depth++ >= maxDepth) {
                                    terminated = true;
                                    return false;
                                }

                                // Sample phase function for medium scattering event
                                Point2f u{rng.Uniform<Float>(), rng.Uniform<Float>()};
                                pstd::optional<PhaseFunctionSample> ps =
                                    mp.phase.Sample_p(-ray.d, u);
                                if (!ps) {
                                    terminated = true;
                                    return false;
                                }

                                // Update state for recursive evaluation of $L_\roman{i}$
                                beta *= ps->p / ps->pdf;
                                ray.o = p;
                                ray.d = ps->wi;
                                scattered = true;
                                return false;

                            } else {
                                // Handle null scattering event for medium sample
                                uMode = rng.Uniform<Float>();
                                return true;
                            }
                        });
        }
        // Handle terminated and unscattered rays after medium sampling
        if (terminated)
            return L;
        if (scattered)
            continue;
        // Add emission to surviving ray
        if (si)
            L += beta * si->intr.Le(-ray.d, lambda);
        else {
            for (const auto &light : infiniteLights)
                L += beta * light.Le(ray, lambda);
            return L;
        }

        // Handle surface intersection along ray path
        BSDF bsdf = si->intr.GetBSDF(ray, lambda, camera, buf, sampler);
        if (!bsdf)
            si->intr.SkipIntersection(&ray, si->tHit);
        else {
            // Report error if BSDF returns a valid sample
            Float uc = sampler.Get1D();
            Point2f u = sampler.Get2D();
            if (bsdf.Sample_f(-ray.d, uc, u))
                ErrorExit("SimpleVolPathIntegrator doesn't support surface scattering.");
            else
                break;
        }
    }
    return L;
}

std::string SimpleVolPathIntegrator::ToString() const {
    return StringPrintf("[ SimpleVolPathIntegrator maxDepth: %d ] ", maxDepth);
}

std::unique_ptr<SimpleVolPathIntegrator> SimpleVolPathIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    return std::make_unique<SimpleVolPathIntegrator>(maxDepth, camera, sampler, aggregate,
                                                     lights);
}

STAT_COUNTER("Integrator/Volume interactions", volumeInteractions);
STAT_COUNTER("Integrator/Surface interactions", surfaceInteractions);

// VolPathIntegrator Method Definitions
SampledSpectrum VolPathIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                      Sampler sampler, ScratchBuffer &scratchBuffer,
                                      VisibleSurface *visibleSurf) const {
    // Declare state variables for volumetric path sampling
    SampledSpectrum L(0.f), beta(1.f), r_u(1.f), r_l(1.f);
    bool specularBounce = false, anyNonSpecularBounces = false;
    int depth = 0;
    Float etaScale = 1;

    LightSampleContext prevIntrContext;

    while (true) {
        // Sample segment of volumetric scattering path
        PBRT_DBG("%s\n", StringPrintf("Path tracer depth %d, current L = %s, beta = %s\n",
                                      depth, L, beta)
                             .c_str());
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (ray.medium) {
            // Sample the participating medium
            bool scattered = false, terminated = false;
            Float tMax = si ? si->tHit : Infinity;
            // Initialize _RNG_ for sampling the majorant transmittance
            uint64_t hash0 = Hash(sampler.Get1D());
            uint64_t hash1 = Hash(sampler.Get1D());
            RNG rng(hash0, hash1);

            SampledSpectrum T_maj = SampleT_maj(
                ray, tMax, sampler.Get1D(), rng, lambda,
                [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                    SampledSpectrum T_maj) {
                    // Handle medium scattering event for ray
                    if (!beta) {
                        terminated = true;
                        return false;
                    }
                    ++volumeInteractions;
                    // Add emission from medium scattering event
                    if (depth < maxDepth && mp.Le) {
                        // Compute $\beta'$ at new path vertex
                        Float pdf = sigma_maj[0] * T_maj[0];
                        SampledSpectrum betap = beta * T_maj / pdf;

                        // Compute rescaled path probability for absorption at path vertex
                        SampledSpectrum r_e = r_u * sigma_maj * T_maj / pdf;

                        // Update _L_ for medium emission
                        if (r_e)
                            L += betap * mp.sigma_a * mp.Le / r_e.Average();
                    }

                    // Compute medium event probabilities for interaction
                    Float pAbsorb = mp.sigma_a[0] / sigma_maj[0];
                    Float pScatter = mp.sigma_s[0] / sigma_maj[0];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);

                    CHECK_GE(1 - pAbsorb - pScatter, -1e-6);
                    // Sample medium scattering event type and update path
                    Float um = rng.Uniform<Float>();
                    int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, um);
                    if (mode == 0) {
                        // Handle absorption along ray path
                        terminated = true;
                        return false;

                    } else if (mode == 1) {
                        // Handle scattering along ray path
                        // Stop path sampling if maximum depth has been reached
                        if (depth++ >= maxDepth) {
                            terminated = true;
                            return false;
                        }

                        // Update _beta_ and _r_u_ for real scattering event
                        Float pdf = T_maj[0] * mp.sigma_s[0];
                        beta *= T_maj * mp.sigma_s / pdf;
                        r_u *= T_maj * mp.sigma_s / pdf;

                        if (beta && r_u) {
                            // Sample direct lighting at volume scattering event
                            MediumInteraction intr(p, -ray.d, ray.time, ray.medium,
                                                   mp.phase);
                            L += SampleLd(intr, nullptr, lambda, sampler, beta, r_u);

                            // Sample new direction at real scattering event
                            Point2f u = sampler.Get2D();
                            pstd::optional<PhaseFunctionSample> ps =
                                intr.phase.Sample_p(-ray.d, u);
                            if (!ps || ps->pdf == 0)
                                terminated = true;
                            else {
                                // Update ray path state for indirect volume scattering
                                beta *= ps->p / ps->pdf;
                                r_l = r_u / ps->pdf;
                                prevIntrContext = LightSampleContext(intr);
                                scattered = true;
                                ray.o = p;
                                ray.d = ps->wi;
                                specularBounce = false;
                                anyNonSpecularBounces = true;
                            }
                        }
                        return false;

                    } else {
                        // Handle null scattering along ray path
                        SampledSpectrum sigma_n =
                            ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);
                        Float pdf = T_maj[0] * sigma_n[0];
                        beta *= T_maj * sigma_n / pdf;
                        if (pdf == 0)
                            beta = SampledSpectrum(0.f);
                        r_u *= T_maj * sigma_n / pdf;
                        r_l *= T_maj * sigma_maj / pdf;
                        return beta && r_u;
                    }
                });
            // Handle terminated, scattered, and unscattered medium rays
            if (terminated || !beta || !r_u)
                return L;
            if (scattered)
                continue;

            beta *= T_maj / T_maj[0];
            r_u *= T_maj / T_maj[0];
            r_l *= T_maj / T_maj[0];
        }
        // Handle surviving unscattered rays
        // Add emitted light at volume path vertex or from the environment
        if (!si) {
            // Accumulate contributions from infinite light sources
            for (const auto &light : infiniteLights) {
                if (SampledSpectrum Le = light.Le(ray, lambda); Le) {
                    if (depth == 0 || specularBounce)
                        L += beta * Le / r_u.Average();
                    else {
                        // Add infinite light contribution using both PDFs with MIS
                        Float lightPDF = lightSampler.PMF(prevIntrContext, light) *
                                         light.PDF_Li(prevIntrContext, ray.d, true);
                        r_l *= lightPDF;
                        L += beta * Le / (r_u + r_l).Average();
                    }
                }
            }

            break;
        }
        SurfaceInteraction &isect = si->intr;
        if (SampledSpectrum Le = isect.Le(-ray.d, lambda); Le) {
            // Add contribution of emission from intersected surface
            if (depth == 0 || specularBounce)
                L += beta * Le / r_u.Average();
            else {
                // Add surface light contribution using both PDFs with MIS
                Light areaLight(isect.areaLight);
                Float lightPDF = lightSampler.PMF(prevIntrContext, areaLight) *
                                 areaLight.PDF_Li(prevIntrContext, ray.d, true);
                r_l *= lightPDF;
                L += beta * Le / (r_u + r_l).Average();
            }
        }

        // Get BSDF and skip over medium boundaries
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        // Initialize _visibleSurf_ at first intersection
        if (depth == 0 && visibleSurf) {
            // Estimate BSDF's albedo
            // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
            constexpr int nRhoSamples = 16;
            const Float ucRho[nRhoSamples] = {
                0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                0.9674518,  0.2995429,  0.5083201, 0.047338516};
            const Point2f uRho[nRhoSamples] = {
                Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

            SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

            *visibleSurf = VisibleSurface(isect, albedo, lambda);
        }

        // Terminate path if maximum depth reached
        if (depth++ >= maxDepth)
            return L;

        ++surfaceInteractions;
        // Possibly regularize the BSDF
        if (regularize && anyNonSpecularBounces) {
            ++regularizedBSDFs;
            bsdf.Regularize();
        }

        // Sample illumination from lights to find attenuated path contribution
        if (IsNonSpecular(bsdf.Flags())) {
            L += SampleLd(isect, &bsdf, lambda, sampler, beta, r_u);
            DCHECK(IsInf(L.y(lambda)) == false);
        }
        prevIntrContext = LightSampleContext(isect);

        // Sample BSDF to get new volumetric path direction
        Vector3f wo = isect.wo;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
        if (!bs)
            break;
        // Update _beta_ and rescaled path probabilities for BSDF scattering
        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        if (bs->pdfIsProportional)
            r_l = r_u / bsdf.PDF(wo, bs->wi);
        else
            r_l = r_u / bs->pdf;

        PBRT_DBG("%s\n", StringPrintf("Sampled BSDF, f = %s, pdf = %f -> beta = %s",
                                      bs->f, bs->pdf, beta)
                             .c_str());
        DCHECK(IsInf(beta.y(lambda)) == false);
        // Update volumetric integrator path state after surface scattering
        specularBounce = bs->IsSpecular();
        anyNonSpecularBounces |= !bs->IsSpecular();
        if (bs->IsTransmission())
            etaScale *= Sqr(bs->eta);
        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);

        // Account for attenuated subsurface scattering, if applicable
        BSSRDF bssrdf = isect.GetBSSRDF(ray, lambda, camera, scratchBuffer);
        if (bssrdf && bs->IsTransmission()) {
            // Sample BSSRDF probe segment to find exit point
            Float uc = sampler.Get1D();
            Point2f up = sampler.Get2D();
            pstd::optional<BSSRDFProbeSegment> probeSeg = bssrdf.SampleSp(uc, up);
            if (!probeSeg)
                break;

            // Sample random intersection along BSSRDF probe segment
            uint64_t seed = MixBits(FloatToBits(sampler.Get1D()));
            WeightedReservoirSampler<SubsurfaceInteraction> interactionSampler(seed);
            // Intersect BSSRDF sampling ray against the scene geometry
            Interaction base(probeSeg->p0, ray.time, Medium());
            while (true) {
                Ray r = base.SpawnRayTo(probeSeg->p1);
                if (r.d == Vector3f(0, 0, 0))
                    break;
                pstd::optional<ShapeIntersection> si = Intersect(r, 1);
                if (!si)
                    break;
                base = si->intr;
                if (si->intr.material == isect.material)
                    interactionSampler.Add(SubsurfaceInteraction(si->intr), 1.f);
            }

            if (!interactionSampler.HasSample())
                break;

            // Convert probe intersection to _BSSRDFSample_
            SubsurfaceInteraction ssi = interactionSampler.GetSample();
            BSSRDFSample bssrdfSample =
                bssrdf.ProbeIntersectionToSample(ssi, scratchBuffer);
            if (!bssrdfSample.Sp || !bssrdfSample.pdf)
                break;

            // Update path state for subsurface scattering
            Float pdf = interactionSampler.SampleProbability() * bssrdfSample.pdf[0];
            beta *= bssrdfSample.Sp / pdf;
            r_u *= bssrdfSample.pdf / bssrdfSample.pdf[0];
            SurfaceInteraction pi = ssi;
            pi.wo = bssrdfSample.wo;
            prevIntrContext = LightSampleContext(pi);
            // Possibly regularize subsurface BSDF
            BSDF &Sw = bssrdfSample.Sw;
            anyNonSpecularBounces = true;
            if (regularize) {
                ++regularizedBSDFs;
                Sw.Regularize();
            } else
                ++totalBSDFs;

            // Account for attenuated direct illumination subsurface scattering
            L += SampleLd(pi, &Sw, lambda, sampler, beta, r_u);

            // Sample ray for indirect subsurface scattering
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = Sw.Sample_f(pi.wo, u, sampler.Get2D());
            if (!bs)
                break;
            beta *= bs->f * AbsDot(bs->wi, pi.shading.n) / bs->pdf;
            r_l = r_u / bs->pdf;
            // Don't increment depth this time...
            DCHECK(!IsInf(beta.y(lambda)));
            specularBounce = bs->IsSpecular();
            ray = RayDifferential(pi.SpawnRay(bs->wi));
        }

        // Possibly terminate volumetric path with Russian roulette
        if (!beta)
            break;
        SampledSpectrum rrBeta = beta * etaScale / r_u.Average();
        Float uRR = sampler.Get1D();
        PBRT_DBG("%s\n",
                 StringPrintf("etaScale %f -> rrBeta %s", etaScale, rrBeta).c_str());
        if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
            Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
            if (uRR < q)
                break;
            beta /= 1 - q;
        }
    }
    return L;
}

SampledSpectrum VolPathIntegrator::SampleLd(const Interaction &intr, const BSDF *bsdf,
                                            SampledWavelengths &lambda, Sampler sampler,
                                            SampledSpectrum beta,
                                            SampledSpectrum r_p) const {
    // Estimate light-sampled direct illumination at _intr_
    // Initialize _LightSampleContext_ for volumetric light sampling
    LightSampleContext ctx;
    if (bsdf) {
        ctx = LightSampleContext(intr.AsSurface());
        // Try to nudge the light sampling position to correct side of the surface
        BxDFFlags flags = bsdf->Flags();
        if (IsReflective(flags) && !IsTransmissive(flags))
            ctx.pi = intr.OffsetRayOrigin(intr.wo);
        else if (IsTransmissive(flags) && !IsReflective(flags))
            ctx.pi = intr.OffsetRayOrigin(-intr.wo);

    } else
        ctx = LightSampleContext(intr);

    // Sample a light source using _lightSampler_
    Float u = sampler.Get1D();
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(ctx, u);
    Point2f uLight = sampler.Get2D();
    if (!sampledLight)
        return SampledSpectrum(0.f);
    Light light = sampledLight->light;
    DCHECK(light && sampledLight->p != 0);

    // Sample a point on the light source
    pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
    if (!ls || !ls->L || ls->pdf == 0)
        return SampledSpectrum(0.f);
    Float lightPDF = sampledLight->p * ls->pdf;

    // Evaluate BSDF or phase function for light sample direction
    Float scatterPDF;
    SampledSpectrum f_hat;
    Vector3f wo = intr.wo, wi = ls->wi;
    if (bsdf) {
        // Update _f_hat_ and _scatterPDF_ accounting for the BSDF
        f_hat = bsdf->f(wo, wi) * AbsDot(wi, intr.AsSurface().shading.n);
        scatterPDF = bsdf->PDF(wo, wi);

    } else {
        // Update _f_hat_ and _scatterPDF_ accounting for the phase function
        CHECK(intr.IsMediumInteraction());
        PhaseFunction phase = intr.AsMedium().phase;
        f_hat = SampledSpectrum(phase.p(wo, wi));
        scatterPDF = phase.PDF(wo, wi);
    }
    if (!f_hat)
        return SampledSpectrum(0.f);

    // Declare path state variables for ray to light source
    Ray lightRay = intr.SpawnRayTo(ls->pLight);
    SampledSpectrum T_ray(1.f), r_l(1.f), r_u(1.f);
    RNG rng(Hash(lightRay.o), Hash(lightRay.d));

    while (lightRay.d != Vector3f(0, 0, 0)) {
        // Trace ray through media to estimate transmittance
        pstd::optional<ShapeIntersection> si = Intersect(lightRay, 1 - ShadowEpsilon);
        // Handle opaque surface along ray's path
        if (si && si->intr.material)
            return SampledSpectrum(0.f);

        // Update transmittance for current ray segment
        if (lightRay.medium) {
            Float tMax = si ? si->tHit : (1 - ShadowEpsilon);
            Float u = rng.Uniform<Float>();
            SampledSpectrum T_maj =
                SampleT_maj(lightRay, tMax, u, rng, lambda,
                            [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                                SampledSpectrum T_maj) {
                                // Update ray transmittance estimate at sampled point
                                // Update _T_ray_ and PDFs using ratio-tracking estimator
                                SampledSpectrum sigma_n =
                                    ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);
                                Float pdf = T_maj[0] * sigma_maj[0];
                                T_ray *= T_maj * sigma_n / pdf;
                                r_l *= T_maj * sigma_maj / pdf;
                                r_u *= T_maj * sigma_n / pdf;

                                // Possibly terminate transmittance computation using
                                // Russian roulette
                                SampledSpectrum Tr = T_ray / (r_l + r_u).Average();
                                if (Tr.MaxComponentValue() < 0.05f) {
                                    Float q = 0.75f;
                                    if (rng.Uniform<Float>() < q)
                                        T_ray = SampledSpectrum(0.);
                                    else
                                        T_ray /= 1 - q;
                                }

                                if (!T_ray)
                                    return false;
                                return true;
                            });
            // Update transmittance estimate for final segment
            T_ray *= T_maj / T_maj[0];
            r_l *= T_maj / T_maj[0];
            r_u *= T_maj / T_maj[0];
        }

        // Generate next ray segment or return final transmittance
        if (!T_ray)
            return SampledSpectrum(0.f);
        if (!si)
            break;
        lightRay = si->intr.SpawnRayTo(ls->pLight);
    }
    // Return path contribution function estimate for direct lighting
    r_l *= r_p * lightPDF;
    r_u *= r_p * scatterPDF;
    if (IsDeltaLight(light.Type()))
        return beta * f_hat * T_ray * ls->L / r_l.Average();
    else
        return beta * f_hat * T_ray * ls->L / (r_l + r_u).Average();
}

std::string VolPathIntegrator::ToString() const {
    return StringPrintf(
        "[ VolPathIntegrator maxDepth: %d lightSampler: %s regularize: %s ]", maxDepth,
        lightSampler, regularize);
}

std::unique_ptr<VolPathIntegrator> VolPathIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    std::string lightStrategy = parameters.GetOneString("lightsampler", "bvh");
    bool regularize = parameters.GetOneBool("regularize", false);
    return std::make_unique<VolPathIntegrator>(maxDepth, camera, sampler, aggregate,
                                               lights, lightStrategy, regularize);
}

// AOIntegrator Method Definitions
AOIntegrator::AOIntegrator(bool cosSample, Float maxDist, Camera camera, Sampler sampler,
                           Primitive aggregate, std::vector<Light> lights,
                           Spectrum illuminant)
    : RayIntegrator(camera, sampler, aggregate, lights),
      cosSample(cosSample),
      maxDist(maxDist),
      illuminant(illuminant),
      illumScale(1.f / SpectrumToPhotometric(illuminant)) {}

SampledSpectrum AOIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                 Sampler sampler, ScratchBuffer &scratchBuffer,
                                 VisibleSurface *visibleSurface) const {
    SampledSpectrum L(0.f);

    // Intersect _ray_ with scene and store intersection in _isect_
    pstd::optional<ShapeIntersection> si;
retry:
    si = Intersect(ray);
    if (si) {
        SurfaceInteraction &isect = si->intr;
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            goto retry;
        }

        // Compute coordinate frame based on true geometry, not shading
        // geometry.
        Normal3f n = FaceForward(isect.n, -ray.d);

        Vector3f wi;
        Float pdf;
        Point2f u = sampler.Get2D();
        if (cosSample) {
            wi = SampleCosineHemisphere(u);
            pdf = CosineHemispherePDF(std::abs(wi.z));
        } else {
            wi = SampleUniformHemisphere(u);
            pdf = UniformHemispherePDF();
        }
        if (pdf == 0)
            return SampledSpectrum(0.);

        Frame f = Frame::FromZ(n);
        wi = f.FromLocal(wi);

        // Divide by pi so that fully visible is one.
        Ray r = isect.SpawnRay(wi);
        if (!IntersectP(r, maxDist)) {
            return illumScale * illuminant.Sample(lambda) *
                   SampledSpectrum(Dot(wi, n) / (Pi * pdf));
        }
    }
    return SampledSpectrum(0.);
}

std::string AOIntegrator::ToString() const {
    return StringPrintf("[ AOIntegrator cosSample: %s maxDist: %f illuminant: %s ]",
                        cosSample, maxDist, illuminant);
}

std::unique_ptr<AOIntegrator> AOIntegrator::Create(const ParameterDictionary &parameters,
                                                   Spectrum illuminant, Camera camera,
                                                   Sampler sampler, Primitive aggregate,
                                                   std::vector<Light> lights,
                                                   const FileLoc *loc) {
    bool cosSample = parameters.GetOneBool("cossample", true);
    Float maxDist = parameters.GetOneFloat("maxdistance", Infinity);
    return std::make_unique<AOIntegrator>(cosSample, maxDist, camera, sampler, aggregate,
                                          lights, illuminant);
}

// BDPT Utility Function Declarations
int RandomWalk(const Integrator &integrator, SampledWavelengths &lambda,
               RayDifferential ray, Sampler sampler, Camera camera,
               ScratchBuffer &scratchBuffer, SampledSpectrum beta, Float pdf,
               int maxDepth, TransportMode mode, Vertex *path, bool regularize);

SampledSpectrum ConnectBDPT(const Integrator &integrator, SampledWavelengths &lambda,
                            Vertex *lightVertices, Vertex *cameraVertices, int s, int t,
                            LightSampler lightSampler, Camera camera, Sampler sampler,
                            pstd::optional<Point2f> *pRaster,
                            Float *misWeightPtr = nullptr);

Float InfiniteLightDensity(const std::vector<Light> &infiniteLights,
                           LightSampler lightSampler, Vector3f w);

// VertexType Definition
enum class VertexType { Camera, Light, Surface, Medium };

// ScopedAssignment Definition
template <typename Type>
class ScopedAssignment {
  public:
    // ScopedAssignment Public Methods
    ScopedAssignment(Type *target = nullptr, Type value = Type()) : target(target) {
        if (target) {
            backup = *target;
            *target = value;
        }
    }
    ~ScopedAssignment() {
        if (target)
            *target = backup;
    }
    ScopedAssignment(const ScopedAssignment &) = delete;
    ScopedAssignment &operator=(const ScopedAssignment &) = delete;

    ScopedAssignment &operator=(ScopedAssignment &&other) {
        target = other.target;
        backup = other.backup;
        other.target = nullptr;
        return *this;
    }

  private:
    Type *target, backup;
};

// EndpointInteraction Definition
struct EndpointInteraction : Interaction {
    union {
        Camera camera;
        Light light;
    };
    // EndpointInteraction Public Methods
    EndpointInteraction() : Interaction(), light(nullptr) {}
    EndpointInteraction(const Interaction &it, Camera camera)
        : Interaction(it), camera(camera) {}
    EndpointInteraction(Camera camera, const Ray &ray)
        : Interaction(ray.o, ray.time, ray.medium), camera(camera) {}
    EndpointInteraction(const EndpointInteraction &ei)
        : Interaction(ei), camera(ei.camera) {
        static_assert(sizeof(Light) == sizeof(Camera),
                      "Expect both union members have same size");
    }

    EndpointInteraction(Light light, const Interaction &intr)
        : Interaction(intr), light(light) {}
    EndpointInteraction(Light light, const Ray &r)
        : Interaction(r.o, r.time, r.medium), light(light) {}
    EndpointInteraction(const Ray &ray)
        : Interaction(ray(1), Normal3f(-ray.d), ray.time, ray.medium), light(nullptr) {}
};

// BDPT Vertex Definition
struct Vertex {
    // Vertex Public Members
    VertexType type;
    SampledSpectrum beta;
    union {
        EndpointInteraction ei;
        MediumInteraction mi;
        SurfaceInteraction si;
    };
    BSDF bsdf;
    bool delta = false;
    Float pdfFwd = 0, pdfRev = 0;

    // Vertex Public Methods
    // Need to define these two to make compilers happy with the non-POD
    // objects in the anonymous union above.
    Vertex(const Vertex &v) { memcpy(this, &v, sizeof(Vertex)); }
    Vertex &operator=(const Vertex &v) {
        memcpy(this, &v, sizeof(Vertex));
        return *this;
    }

    Vertex() : ei() {}

    Vertex(VertexType type, const EndpointInteraction &ei, const SampledSpectrum &beta)
        : type(type), beta(beta), ei(ei) {}

    Vertex(const SurfaceInteraction &si, const BSDF &bsdf, const SampledSpectrum &beta)
        : type(VertexType::Surface), beta(beta), si(si), bsdf(bsdf) {}

    static inline Vertex CreateCamera(Camera camera, const Ray &ray,
                                      const SampledSpectrum &beta);
    static inline Vertex CreateCamera(Camera camera, const Interaction &it,
                                      const SampledSpectrum &beta);

    static inline Vertex CreateLight(Light light, const Ray &ray,
                                     const SampledSpectrum &Le, Float pdf);
    static inline Vertex CreateLight(Light light, const Interaction &intr,
                                     const SampledSpectrum &Le, Float pdf);
    static inline Vertex CreateLight(const EndpointInteraction &ei,
                                     const SampledSpectrum &beta, Float pdf);

    static inline Vertex CreateMedium(const MediumInteraction &mi,
                                      const SampledSpectrum &beta, Float pdf,
                                      const Vertex &prev);
    static inline Vertex CreateSurface(const SurfaceInteraction &si, const BSDF &bsdf,
                                       const SampledSpectrum &beta, Float pdf,
                                       const Vertex &prev);

    Vertex(const MediumInteraction &mi, const SampledSpectrum &beta)
        : type(VertexType::Medium), beta(beta), mi(mi) {}

    const Interaction &GetInteraction() const {
        switch (type) {
        case VertexType::Medium:
            return mi;
        case VertexType::Surface:
            return si;
        default:
            return ei;
        }
    }

    Point3f p() const { return GetInteraction().p(); }

    Float time() const { return GetInteraction().time; }
    const Normal3f &ng() const { return GetInteraction().n; }
    const Normal3f &ns() const {
        if (type == VertexType::Surface)
            return si.shading.n;
        else
            return GetInteraction().n;
    }

    bool IsOnSurface() const { return ng() != Normal3f(); }

    SampledSpectrum f(const Vertex &next, TransportMode mode) const {
        Vector3f wi = next.p() - p();
        if (LengthSquared(wi) == 0)
            return {};
        wi = Normalize(wi);
        switch (type) {
        case VertexType::Surface:
            return bsdf.f(si.wo, wi, mode);
        case VertexType::Medium:
            return SampledSpectrum(mi.phase.p(mi.wo, wi));
        default:
            LOG_FATAL("Vertex::f(): Unimplemented");
            return SampledSpectrum(0.f);
        }
    }

    bool IsConnectible() const {
        switch (type) {
        case VertexType::Medium:
            return true;
        case VertexType::Light:
            return ei.light.Type() != LightType::DeltaDirection;
        case VertexType::Camera:
            return true;
        case VertexType::Surface:
            return IsNonSpecular(bsdf.Flags());
        }
        LOG_FATAL("Unhandled vertex type in IsConnectible()");
    }

    bool IsLight() const {
        return type == VertexType::Light || (type == VertexType::Surface && si.areaLight);
    }

    bool IsDeltaLight() const {
        return type == VertexType::Light && ei.light &&
               pbrt::IsDeltaLight(ei.light.Type());
    }

    bool IsInfiniteLight() const {
        return type == VertexType::Light &&
               (!ei.light || ei.light.Type() == LightType::Infinite ||
                ei.light.Type() == LightType::DeltaDirection);
    }

    SampledSpectrum Le(const std::vector<Light> &infiniteLights, const Vertex &v,
                       const SampledWavelengths &lambda) const {
        if (!IsLight())
            return SampledSpectrum(0.f);
        Vector3f w = v.p() - p();
        if (LengthSquared(w) == 0)
            return SampledSpectrum(0.);
        w = Normalize(w);
        if (IsInfiniteLight()) {
            // Return emitted radiance for infinite light sources
            SampledSpectrum Le(0.f);
            for (const auto &light : infiniteLights)
                Le += light.Le(Ray(p(), -w), lambda);
            return Le;

        } else if (si.areaLight)
            return si.areaLight.L(si.p(), si.n, si.uv, w, lambda);
        else
            return SampledSpectrum(0.f);
    }

    std::string ToString() const {
        std::string s = std::string("[ Vertex type: ");
        switch (type) {
        case VertexType::Camera:
            s += "camera";
            break;
        case VertexType::Light:
            s += "light";
            break;
        case VertexType::Surface:
            s += "surface";
            break;
        case VertexType::Medium:
            s += "medium";
            break;
        }
        s += StringPrintf(" connectible: %s p: %s ng: %s pdfFwd: %f pdfRev: %f beta: %s",
                          IsConnectible(), p(), ng(), pdfFwd, pdfRev, beta);
        switch (type) {
        case VertexType::Camera:
            // TODO
            break;
        case VertexType::Light:
            // TODO
            break;
        case VertexType::Surface:
            s += std::string("\n  bsdf: ") + bsdf.ToString();
            break;
        case VertexType::Medium:
            s += std::string("\n  phase: ") + mi.phase.ToString();
            break;
        }
        s += std::string(" ]");
        return s;
    }

    Float ConvertDensity(Float pdf, const Vertex &next) const {
        // Return solid angle density if _next_ is an infinite area light
        if (next.IsInfiniteLight())
            return pdf;

        Vector3f w = next.p() - p();
        if (LengthSquared(w) == 0)
            return 0;
        Float invDist2 = 1 / LengthSquared(w);
        if (next.IsOnSurface())
            pdf *= AbsDot(next.ng(), w * std::sqrt(invDist2));
        return pdf * invDist2;
    }

    Float PDF(const Integrator &integrator, const Vertex *prev,
              const Vertex &next) const {
        if (type == VertexType::Light)
            return PDFLight(integrator, next);
        // Compute directions to preceding and next vertex
        Vector3f wn = next.p() - p();
        if (LengthSquared(wn) == 0)
            return 0;
        wn = Normalize(wn);
        Vector3f wp;
        if (prev) {
            wp = prev->p() - p();
            if (LengthSquared(wp) == 0)
                return 0;
            wp = Normalize(wp);
        } else
            CHECK(type == VertexType::Camera);

        // Compute directional density depending on the vertex type
        Float pdf = 0, unused;
        if (type == VertexType::Camera)
            ei.camera.PDF_We(ei.SpawnRay(wn), &unused, &pdf);
        else if (type == VertexType::Surface)
            pdf = bsdf.PDF(wp, wn);
        else if (type == VertexType::Medium)
            pdf = mi.phase.p(wp, wn);
        else
            LOG_FATAL("Vertex::PDF(): Unimplemented");

        // Return probability per unit area at vertex _next_
        return ConvertDensity(pdf, next);
    }

    Float PDFLight(const Integrator &integrator, const Vertex &v) const {
        Vector3f w = v.p() - p();
        Float invDist2 = 1 / LengthSquared(w);
        w *= std::sqrt(invDist2);
        // Compute sampling density _pdf_ for light type
        Float pdf;
        if (IsInfiniteLight()) {
            // Compute planar sampling density for infinite light sources
            Bounds3f sceneBounds = integrator.aggregate.Bounds();
            Point3f sceneCenter;
            Float sceneRadius;
            sceneBounds.BoundingSphere(&sceneCenter, &sceneRadius);
            pdf = 1 / (Pi * Sqr(sceneRadius));

        } else if (IsOnSurface()) {
            // Compute sampling density at emissive surface
            if (type == VertexType::Light)
                CHECK(ei.light.Is<DiffuseAreaLight>());  // since that's all we've
                                                         // got currently...
            Light light = (type == VertexType::Light) ? ei.light : si.areaLight;
            Float pdfPos, pdfDir;
            light.PDF_Le(ei, w, &pdfPos, &pdfDir);
            pdf = pdfDir * invDist2;

        } else {
            // Compute sampling density for noninfinite light sources
            CHECK(type == VertexType::Light);
            CHECK(ei.light);
            Float pdfPos, pdfDir;
            ei.light.PDF_Le(Ray(p(), w, time()), &pdfPos, &pdfDir);
            pdf = pdfDir * invDist2;
        }

        if (v.IsOnSurface())
            pdf *= AbsDot(v.ng(), w);
        return pdf;
    }

    Float PDFLightOrigin(const std::vector<Light> &infiniteLights, const Vertex &v,
                         LightSampler lightSampler) {
        Vector3f w = v.p() - p();
        if (LengthSquared(w) == 0)
            return 0.;
        w = Normalize(w);
        if (IsInfiniteLight()) {
            // Return sampling density for infinite light sources
            return InfiniteLightDensity(infiniteLights, lightSampler, w);

        } else {
            // Return sampling density for noninfinite light source
            Light light = (type == VertexType::Light) ? ei.light : si.areaLight;
            Float pdfPos, pdfDir, pdfChoice = lightSampler.PMF(light);
            if (IsOnSurface())
                light.PDF_Le(ei, w, &pdfPos, &pdfDir);
            else
                light.PDF_Le(Ray(p(), w, time()), &pdfPos, &pdfDir);
            return pdfPos * pdfChoice;
        }
    }
};

// BDPT Vertex Inline Method Definitions
inline Vertex Vertex::CreateCamera(Camera camera, const Ray &ray,
                                   const SampledSpectrum &beta) {
    return Vertex(VertexType::Camera, EndpointInteraction(camera, ray), beta);
}

inline Vertex Vertex::CreateCamera(Camera camera, const Interaction &it,
                                   const SampledSpectrum &beta) {
    return Vertex(VertexType::Camera, EndpointInteraction(it, camera), beta);
}

inline Vertex Vertex::CreateLight(Light light, const Ray &ray, const SampledSpectrum &Le,
                                  Float pdf) {
    Vertex v(VertexType::Light, EndpointInteraction(light, ray), Le);
    v.pdfFwd = pdf;
    return v;
}

inline Vertex Vertex::CreateLight(Light light, const Interaction &intr,
                                  const SampledSpectrum &Le, Float pdf) {
    Vertex v(VertexType::Light, EndpointInteraction(light, intr), Le);
    v.pdfFwd = pdf;
    return v;
}

inline Vertex Vertex::CreateSurface(const SurfaceInteraction &si, const BSDF &bsdf,
                                    const SampledSpectrum &beta, Float pdf,
                                    const Vertex &prev) {
    Vertex v(si, bsdf, beta);
    v.pdfFwd = prev.ConvertDensity(pdf, v);
    return v;
}

inline Vertex Vertex::CreateMedium(const MediumInteraction &mi,
                                   const SampledSpectrum &beta, Float pdf,
                                   const Vertex &prev) {
    Vertex v(mi, beta);
    v.pdfFwd = prev.ConvertDensity(pdf, v);
    return v;
}

inline Vertex Vertex::CreateLight(const EndpointInteraction &ei,
                                  const SampledSpectrum &beta, Float pdf) {
    Vertex v(VertexType::Light, ei, beta);
    v.pdfFwd = pdf;
    return v;
}

// BDPT Utility Functions
inline int BufferIndex(int s, int t) {
    int above = s + t - 2;
    return s + above * (5 + above) / 2;
}

int GenerateCameraSubpath(const Integrator &integrator, const RayDifferential &ray,
                          SampledWavelengths &lambda, Sampler sampler,
                          ScratchBuffer &scratchBuffer, int maxDepth, Camera camera,
                          Vertex *path, bool regularize) {
    if (maxDepth == 0)
        return 0;
    SampledSpectrum beta(1.f);
    // Generate first vertex on camera subpath and start random walk
    Float pdfPos, pdfDir;
    path[0] = Vertex::CreateCamera(camera, ray, beta);
    camera.PDF_We(ray, &pdfPos, &pdfDir);
    return RandomWalk(integrator, lambda, ray, sampler, camera, scratchBuffer, beta,
                      pdfDir, maxDepth - 1, TransportMode::Radiance, path + 1,
                      regularize) +
           1;
}

int GenerateLightSubpath(const Integrator &integrator, SampledWavelengths &lambda,
                         Sampler sampler, Camera camera, ScratchBuffer &scratchBuffer,
                         int maxDepth, Float time, LightSampler lightSampler,
                         Vertex *path, bool regularize) {
    // Generate light subpath and initialize _path_ vertices
    if (maxDepth == 0)
        return 0;
    // Sample initial ray for light subpath
    // Sample light for BDPT light subpath
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(sampler.Get1D());
    if (!sampledLight)
        return 0;
    Light light = sampledLight->light;
    Float lightSamplePDF = sampledLight->p;

    Point2f ul0 = sampler.Get2D();
    Point2f ul1 = sampler.Get2D();
    pstd::optional<LightLeSample> les = light.SampleLe(ul0, ul1, lambda, time);
    if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L)
        return 0;
    RayDifferential ray(les->ray);

    // Generate first vertex of light subpath
    Float lightPDF = lightSamplePDF * les->pdfPos;
    path[0] = les->intr ? Vertex::CreateLight(light, *les->intr, les->L, lightPDF)
                        : Vertex::CreateLight(light, ray, les->L, lightPDF);

    // Follow light subpath random walk
    SampledSpectrum beta = les->L * les->AbsCosTheta(ray.d) / (lightPDF * les->pdfDir);
    PBRT_DBG("%s\n",
             StringPrintf(
                 "Starting light subpath. Ray: %s, Le %s, beta %s, pdfPos %f, pdfDir %f",
                 ray, les->L, beta, les->pdfPos, les->pdfDir)
                 .c_str());
    int nVertices = RandomWalk(integrator, lambda, ray, sampler, camera, scratchBuffer,
                               beta, les->pdfDir, maxDepth - 1, TransportMode::Importance,
                               path + 1, regularize);
    // Correct subpath sampling densities for infinite area lights
    if (path[0].IsInfiniteLight()) {
        // Set spatial density of _path[1]_ for infinite area light
        if (nVertices > 0) {
            path[1].pdfFwd = les->pdfPos;
            if (path[1].IsOnSurface())
                path[1].pdfFwd *= AbsDot(ray.d, path[1].ng());
        }

        // Set spatial density of _path[0]_ for infinite area light
        path[0].pdfFwd =
            InfiniteLightDensity(integrator.infiniteLights, lightSampler, ray.d);
    }

    return nVertices + 1;
}

int RandomWalk(const Integrator &integrator, SampledWavelengths &lambda,
               RayDifferential ray, Sampler sampler, Camera camera,
               ScratchBuffer &scratchBuffer, SampledSpectrum beta, Float pdf,
               int maxDepth, TransportMode mode, Vertex *path, bool regularize) {
    if (maxDepth == 0)
        return 0;
    // Follow random walk to initialize BDPT path vertices
    int bounces = 0;
    bool anyNonSpecularBounces = false;
    Float pdfFwd = pdf;
    while (true) {
        // Attempt to create the next subpath vertex in _path_
        PBRT_DBG("%s\n", StringPrintf("Random walk. Bounces %d, beta %s, pdfFwd %f",
                                      bounces, beta, pdfFwd)
                             .c_str());
        if (!beta)
            break;
        bool scattered = false, terminated = false;
        // Trace a ray and sample the medium, if any
        Vertex &vertex = path[bounces], &prev = path[bounces - 1];
        pstd::optional<ShapeIntersection> si = integrator.Intersect(ray);
        if (ray.medium) {
            // Sample participating medium for _RandomWalk()_ ray
            Float tMax = si ? si->tHit : Infinity;
            RNG rng(Hash(ray.o, tMax), Hash(ray.d));
            Float u = sampler.Get1D();
            SampledSpectrum T_maj = SampleT_maj(
                ray, tMax, u, rng, lambda,
                [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                    SampledSpectrum T_maj) {
                    // Compute medium event probabilities for interaction
                    Float pAbsorb = mp.sigma_a[0] / sigma_maj[0];
                    Float pScatter = mp.sigma_s[0] / sigma_maj[0];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);

                    // Randomly sample medium event for _RandomRalk()_ ray
                    Float um = sampler.Get1D();
                    int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, um);
                    if (mode == 0) {
                        // Handle absorption for _RandomWalk()_ ray
                        terminated = true;
                        return false;

                    } else if (mode == 1) {
                        // Handle scattering for _RandomWalk()_ ray
                        beta *= T_maj * mp.sigma_s / (T_maj[0] * mp.sigma_s[0]);
                        // Record medium interaction in _path_ and compute forward density
                        MediumInteraction intr(p, -ray.d, ray.time, ray.medium, mp.phase);
                        vertex = Vertex::CreateMedium(intr, beta, pdfFwd, prev);
                        if (++bounces >= maxDepth) {
                            terminated = true;
                            return false;
                        }

                        // Sample direction and compute reverse density at preceding
                        // vertex
                        pstd::optional<PhaseFunctionSample> ps =
                            intr.phase.Sample_p(-ray.d, sampler.Get2D());
                        if (!ps || ps->pdf == 0) {
                            terminated = true;
                            return false;
                        }
                        // Update path state and previous path vertex after medium
                        // scattering
                        pdfFwd = ps->pdf;
                        beta *= ps->p / ps->pdf;
                        ray = intr.SpawnRay(ps->wi);
                        anyNonSpecularBounces = true;
                        prev.pdfRev = vertex.ConvertDensity(ps->pdf, prev);

                        scattered = true;
                        return false;

                    } else {
                        // Handle null scattering for _RandomWalk()_ ray
                        SampledSpectrum sigma_n =
                            ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);
                        Float pdf = T_maj[0] * sigma_n[0];
                        if (pdf == 0)
                            beta = SampledSpectrum(0.f);
                        else
                            beta *= T_maj * sigma_n / pdf;
                        return bool(beta);
                    }
                });
            // Update _beta_ for medium transmittance
            if (!scattered)
                beta *= T_maj / T_maj[0];
        }

        if (terminated)
            return bounces;
        if (scattered)
            continue;
        // Handle escaped rays after no medium scattering event
        if (!si) {
            // Capture escaped rays when tracing from the camera
            if (mode == TransportMode::Radiance) {
                vertex = Vertex::CreateLight(EndpointInteraction(ray), beta, pdfFwd);
                ++bounces;
            }

            break;
        }

        // Handle surface interaction for path generation
        SurfaceInteraction &isect = si->intr;
        // Get BSDF and skip over medium boundaries
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        // Possibly regularize the BSDF
        if (regularize && anyNonSpecularBounces) {
            ++regularizedBSDFs;
            bsdf.Regularize();
        }

        ++totalBSDFs;
        // Initialize _vertex_ with surface intersection information
        vertex = Vertex::CreateSurface(isect, bsdf, beta, pdfFwd, prev);

        if (++bounces >= maxDepth)
            break;
        // Sample BSDF at current vertex
        Vector3f wo = isect.wo;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D(), mode);
        if (!bs)
            break;
        pdfFwd = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi, mode) : bs->pdf;
        anyNonSpecularBounces |= !bs->IsSpecular();
        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);

        // Compute path probabilities at surface vertex
        // TODO: confirm. I believe that !mode is right. Interestingly,
        // it makes no difference in the test suite either way.
        Float pdfRev = bsdf.PDF(bs->wi, wo, !mode);
        if (bs->IsSpecular()) {
            vertex.delta = true;
            pdfRev = pdfFwd = 0;
        }
        PBRT_DBG("%s\n",
                 StringPrintf("Random walk beta after shading normal correction %s", beta)
                     .c_str());
        prev.pdfRev = vertex.ConvertDensity(pdfRev, prev);
    }
    return bounces;
}

SampledSpectrum G(const Integrator &integrator, Sampler sampler, const Vertex &v0,
                  const Vertex &v1, const SampledWavelengths &lambda) {
    Vector3f d = v0.p() - v1.p();
    Float g = 1 / LengthSquared(d);
    d *= std::sqrt(g);
    if (v0.IsOnSurface())
        g *= AbsDot(v0.ns(), d);
    if (v1.IsOnSurface())
        g *= AbsDot(v1.ns(), d);
    return g * integrator.Tr(v0.GetInteraction(), v1.GetInteraction(), lambda);
}

Float MISWeight(const Integrator &integrator, Vertex *lightVertices,
                Vertex *cameraVertices, Vertex &sampled, int s, int t,
                LightSampler lightSampler) {
    if (s + t == 2)
        return 1;
    Float sumRi = 0;
    // Define helper function _remap0_ that deals with Dirac delta functions
    auto remap0 = [](float f) -> Float { return f != 0 ? f : 1; };

    // Temporarily update vertex properties for current strategy
    // Look up connection vertices and their predecessors
    Vertex *qs = s > 0 ? &lightVertices[s - 1] : nullptr,
           *pt = t > 0 ? &cameraVertices[t - 1] : nullptr,
           *qsMinus = s > 1 ? &lightVertices[s - 2] : nullptr,
           *ptMinus = t > 1 ? &cameraVertices[t - 2] : nullptr;

    // Update sampled vertex for $s=1$ or $t=1$ strategy
    ScopedAssignment<Vertex> a1;
    if (s == 1)
        a1 = {qs, sampled};
    else if (t == 1)
        a1 = {pt, sampled};

    // Mark connection vertices as non-degenerate
    ScopedAssignment<bool> a2, a3;
    if (pt)
        a2 = {&pt->delta, false};
    if (qs)
        a3 = {&qs->delta, false};

    // Update reverse density of vertex $\pt{}_{t-1}$
    ScopedAssignment<Float> a4;
    if (pt)
        a4 = {&pt->pdfRev, s > 0 ? qs->PDF(integrator, qsMinus, *pt)
                                 : pt->PDFLightOrigin(integrator.infiniteLights, *ptMinus,
                                                      lightSampler)};

    // Update reverse density of vertex $\pt{}_{t-2}$
    ScopedAssignment<Float> a5;
    if (ptMinus)
        a5 = {&ptMinus->pdfRev, s > 0 ? pt->PDF(integrator, qs, *ptMinus)
                                      : pt->PDFLight(integrator, *ptMinus)};

    // Update reverse density of vertices $\pq{}_{s-1}$ and $\pq{}_{s-2}$
    ScopedAssignment<Float> a6;
    if (qs)
        a6 = {&qs->pdfRev, pt->PDF(integrator, ptMinus, *qs)};
    ScopedAssignment<Float> a7;
    if (qsMinus)
        a7 = {&qsMinus->pdfRev, qs->PDF(integrator, pt, *qsMinus)};

    // Consider hypothetical connection strategies along the camera subpath
    Float ri = 1;
    for (int i = t - 1; i > 0; --i) {
        ri *= remap0(cameraVertices[i].pdfRev) / remap0(cameraVertices[i].pdfFwd);
        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta)
            sumRi += ri;
    }

    // Consider hypothetical connection strategies along the light subpath
    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        bool deltaLightvertex =
            i > 0 ? lightVertices[i - 1].delta : lightVertices[0].IsDeltaLight();
        if (!lightVertices[i].delta && !deltaLightvertex)
            sumRi += ri;
    }

    return 1 / (1 + sumRi);
}

Float InfiniteLightDensity(const std::vector<Light> &infiniteLights,
                           LightSampler lightSampler, Vector3f w) {
    Float pdf = 0;
    for (const auto &light : infiniteLights)
        pdf += light.PDF_Li(Interaction(), -w) * lightSampler.PMF(light);
    return pdf;
}

// BDPT Method Definitions
void BDPTIntegrator::Render() {
    // Allocate buffers for debug visualization
    if (visualizeStrategies || visualizeWeights) {
        const int bufferCount = (1 + maxDepth) * (6 + maxDepth) / 2;
        weightFilms.resize(bufferCount);
        for (int depth = 0; depth <= maxDepth; ++depth) {
            for (int s = 0; s <= depth + 2; ++s) {
                int t = depth + 2 - s;
                if (t == 0 || (s == 1 && t == 1))
                    continue;

                std::string filename =
                    StringPrintf("bdpt_d%02i_s%02i_t%02i.exr", depth, s, t);

                FilmBaseParameters p(
                    camera.GetFilm().FullResolution(),
                    Bounds2i(Point2i(0, 0), camera.GetFilm().FullResolution()),
                    new BoxFilter,  // FIXME: leaks
                    camera.GetFilm().Diagonal() * 1000, PixelSensor::CreateDefault(),
                    filename);
                weightFilms[BufferIndex(s, t)] = new RGBFilm(p, RGBColorSpace::sRGB);
            }
        }
    }

    RayIntegrator::Render();

    // Write buffers for debug visualization
    if (visualizeStrategies || visualizeWeights) {
        const Float invSampleCount = 1.0f / samplerPrototype.SamplesPerPixel();
        for (size_t i = 0; i < weightFilms.size(); ++i) {
            ImageMetadata metadata;
            if (weightFilms[i])
                weightFilms[i].WriteImage(metadata, invSampleCount);
        }
        weightFilms.clear();
    }
}

SampledSpectrum BDPTIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                   Sampler sampler, ScratchBuffer &scratchBuffer,
                                   VisibleSurface *) const {
    // Trace the camera and light subpaths
    Vertex *cameraVertices = scratchBuffer.Alloc<Vertex[]>(maxDepth + 2);
    int nCamera = GenerateCameraSubpath(*this, ray, lambda, sampler, scratchBuffer,
                                        maxDepth + 2, camera, cameraVertices, regularize);
    Vertex *lightVertices = scratchBuffer.Alloc<Vertex[]>(maxDepth + 1);
    int nLight = GenerateLightSubpath(*this, lambda, sampler, camera, scratchBuffer,
                                      maxDepth + 1, cameraVertices[0].time(),
                                      lightSampler, lightVertices, regularize);

    SampledSpectrum L(0.f);
    // Execute all BDPT connection strategies
    for (int t = 1; t <= nCamera; ++t) {
        for (int s = 0; s <= nLight; ++s) {
            int depth = t + s - 2;
            if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth)
                continue;
            // Execute the $(s, t)$ connection strategy and update _L_
            pstd::optional<Point2f> pFilmNew;
            Float misWeight = 0.f;
            SampledSpectrum Lpath =
                ConnectBDPT(*this, lambda, lightVertices, cameraVertices, s, t,
                            lightSampler, camera, sampler, &pFilmNew, &misWeight);
            PBRT_DBG("%s\n",
                     StringPrintf("Connect bdpt s: %d, t: %d, Lpath: %s, misWeight: %f\n",
                                  s, t, Lpath, misWeight)
                         .c_str());
            if (pFilmNew && (visualizeStrategies || visualizeWeights)) {
                SampledSpectrum value;
                if (visualizeStrategies)
                    value = misWeight == 0 ? SampledSpectrum(0.) : Lpath / misWeight;
                if (visualizeWeights)
                    value = Lpath;
                CHECK(pFilmNew.has_value());
                weightFilms[BufferIndex(s, t)].AddSplat(*pFilmNew, value, lambda);
            }
            if (t != 1)
                L += Lpath;
            else if (Lpath) {
                CHECK(pFilmNew.has_value());
                camera.GetFilm().AddSplat(*pFilmNew, Lpath, lambda);
            }
        }
    }

    return L;
}

SampledSpectrum ConnectBDPT(const Integrator &integrator, SampledWavelengths &lambda,
                            Vertex *lightVertices, Vertex *cameraVertices, int s, int t,
                            LightSampler lightSampler, Camera camera, Sampler sampler,
                            pstd::optional<Point2f> *pRaster, Float *misWeightPtr) {
    SampledSpectrum L(0.f);
    // Ignore invalid connections related to infinite area lights
    if (t > 1 && s != 0 && cameraVertices[t - 1].type == VertexType::Light)
        return SampledSpectrum(0.f);

    // Perform connection and write contribution to _L_
    Vertex sampled;
    if (s == 0) {
        // Interpret the camera subpath as a complete path
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsLight())
            L = pt.Le(integrator.infiniteLights, cameraVertices[t - 2], lambda) * pt.beta;
        DCHECK(!L.HasNaNs());

    } else if (t == 1) {
        // Sample a point on the camera and connect it to the light subpath
        const Vertex &qs = lightVertices[s - 1];
        if (qs.IsConnectible()) {
            pstd::optional<CameraWiSample> cs =
                camera.SampleWi(qs.GetInteraction(), sampler.Get2D(), lambda);
            if (cs) {
                *pRaster = cs->pRaster;
                // Initialize dynamically sampled vertex and _L_ for $t=1$ case
                sampled = Vertex::CreateCamera(camera, cs->pLens, cs->Wi / cs->pdf);
                L = qs.beta * qs.f(sampled, TransportMode::Importance) * sampled.beta;
                if (qs.IsOnSurface())
                    L *= AbsDot(cs->wi, qs.ns());
                DCHECK(!L.HasNaNs());
                if (L)
                    L *= integrator.Tr(cs->pRef, cs->pLens, lambda);
            }
        }

    } else if (s == 1) {
        // Sample a point on a light and connect it to the camera subpath
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsConnectible()) {
            pstd::optional<SampledLight> sampledLight =
                lightSampler.Sample(sampler.Get1D());

            if (sampledLight) {
                Light light = sampledLight->light;
                Float lightPDF = sampledLight->p;

                LightSampleContext ctx;
                if (pt.IsOnSurface()) {
                    const SurfaceInteraction &si = pt.GetInteraction().AsSurface();
                    ctx = LightSampleContext(si);
                    // Try to nudge the light sampling position to correct side of the
                    // surface
                    BxDFFlags flags = pt.bsdf.Flags();
                    if (IsReflective(flags) && !IsTransmissive(flags))
                        ctx.pi = si.OffsetRayOrigin(si.wo);
                    else if (IsTransmissive(flags) && !IsReflective(flags))
                        ctx.pi = si.OffsetRayOrigin(-si.wo);
                } else
                    ctx = LightSampleContext(pt.GetInteraction());
                pstd::optional<LightLiSample> lightWeight =
                    light.SampleLi(ctx, sampler.Get2D(), lambda);
                if (lightWeight && lightWeight->L && lightWeight->pdf > 0) {
                    EndpointInteraction ei(light, lightWeight->pLight);
                    sampled = Vertex::CreateLight(
                        ei, lightWeight->L / (lightWeight->pdf * lightPDF), 0);
                    sampled.pdfFwd = sampled.PDFLightOrigin(integrator.infiniteLights, pt,
                                                            lightSampler);
                    L = pt.beta * pt.f(sampled, TransportMode::Radiance) * sampled.beta;
                    if (pt.IsOnSurface())
                        L *= AbsDot(lightWeight->wi, pt.ns());
                    // Only check visibility if the path would carry radiance.
                    if (L)
                        L *= integrator.Tr(pt.GetInteraction(), lightWeight->pLight,
                                           lambda);
                }
            }
        }

    } else {
        // Handle all other bidirectional connection cases
        const Vertex &qs = lightVertices[s - 1], &pt = cameraVertices[t - 1];
        if (qs.IsConnectible() && pt.IsConnectible()) {
            L = qs.beta * qs.f(pt, TransportMode::Importance) *
                pt.f(qs, TransportMode::Radiance) * pt.beta;
            PBRT_DBG("%s\n",
                     StringPrintf(
                         "General connect s: %d, t: %d, qs: %s, pt: %s, qs.f(pt): %s, "
                         "pt.f(qs): %s, G: %s, dist^2: %f",
                         s, t, qs, pt, qs.f(pt, TransportMode::Importance),
                         pt.f(qs, TransportMode::Radiance),
                         G(integrator, sampler, qs, pt, lambda),
                         DistanceSquared(qs.p(), pt.p()))
                         .c_str());
            if (L)
                L *= G(integrator, sampler, qs, pt, lambda);
        }
    }

    ++totalPaths;
    if (!L)
        ++zeroRadiancePaths;
    pathLength << s + t - 2;
    // Compute MIS weight for connection strategy
    Float misWeight = L ? MISWeight(integrator, lightVertices, cameraVertices, sampled, s,
                                    t, lightSampler)
                        : 0.f;
    PBRT_DBG("MIS weight for (s,t) = (%d, %d) connection: %f\n", s, t, misWeight);
    DCHECK(!IsNaN(misWeight));
    L *= misWeight;
    if (misWeightPtr)
        *misWeightPtr = misWeight;

    return L;
}

std::string BDPTIntegrator::ToString() const {
    return StringPrintf("[ BDPTIntegrator maxDepth: %d visualizeStrategies: %s "
                        "visualizeWeights: %s regularize: %s lightSampler: %s ]",
                        maxDepth, visualizeStrategies, visualizeWeights, regularize,
                        lightSampler);
}

std::unique_ptr<BDPTIntegrator> BDPTIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    bool visualizeStrategies = parameters.GetOneBool("visualizestrategies", false);
    bool visualizeWeights = parameters.GetOneBool("visualizeweights", false);

    if ((visualizeStrategies || visualizeWeights) && maxDepth > 5) {
        Warning(loc, "visualizestrategies/visualizeweights was enabled, limiting "
                     "maxdepth to 5");
        maxDepth = 5;
    }

    bool regularize = parameters.GetOneBool("regularize", false);
    return std::make_unique<BDPTIntegrator>(camera, sampler, aggregate, lights, maxDepth,
                                            visualizeStrategies, visualizeWeights,
                                            regularize);
}

STAT_PERCENT("Integrator/Acceptance rate", acceptedMutations, totalMutations);

// MLTIntegrator Method Definitions
SampledSpectrum MLTIntegrator::L(ScratchBuffer &scratchBuffer, MLTSampler &sampler,
                                 int depth, Point2f *pRaster,
                                 SampledWavelengths *lambda) {
    if (lights.empty())
        return SampledSpectrum(0.f);
    sampler.StartStream(cameraStreamIndex);
    // Determine the number of available strategies and pick a specific one
    int s, t, nStrategies;
    if (depth == 0) {
        nStrategies = 1;
        s = 0;
        t = 2;
    } else {
        nStrategies = depth + 2;
        s = std::min<int>(sampler.Get1D() * nStrategies, nStrategies - 1);
        t = nStrategies - s;
    }

    // Sample wavelengths for MLT path
    if (Options->disableWavelengthJitter)
        *lambda = camera.GetFilm().SampleWavelengths(0.5);
    else
        *lambda = camera.GetFilm().SampleWavelengths(sampler.Get1D());

    // Generate a camera subpath with exactly _t_ vertices
    Vertex *cameraVertices = scratchBuffer.Alloc<Vertex[]>(t);
    // Compute camera sample for MLT camera path
    Bounds2f sampleBounds = camera.GetFilm().SampleBounds();
    *pRaster = sampleBounds.Lerp(sampler.GetPixel2D());
    CameraSample cameraSample;
    cameraSample.pFilm = *pRaster;
    cameraSample.time = sampler.Get1D();
    cameraSample.pLens = sampler.Get2D();

    // Generate camera ray for MLT camera path
    pstd::optional<CameraRayDifferential> crd =
        camera.GenerateRayDifferential(cameraSample, *lambda);
    if (!crd || !crd->weight)
        return SampledSpectrum(0.f);
    Float rayDiffScale =
        std::max<Float>(.125, 1 / std::sqrt((Float)sampler.SamplesPerPixel()));
    crd->ray.ScaleDifferentials(rayDiffScale);

    if (GenerateCameraSubpath(*this, crd->ray, *lambda, &sampler, scratchBuffer, t,
                              camera, cameraVertices, regularize) != t)
        return SampledSpectrum(0.f);

    // Generate a light subpath with exactly _s_ vertices
    sampler.StartStream(lightStreamIndex);
    Vertex *lightVertices = scratchBuffer.Alloc<Vertex[]>(s);
    if (GenerateLightSubpath(*this, *lambda, &sampler, camera, scratchBuffer, s,
                             cameraVertices[0].time(), lightSampler, lightVertices,
                             regularize) != s)
        return SampledSpectrum(0.f);

    // Execute connection strategy and return the radiance estimate
    sampler.StartStream(connectionStreamIndex);
    pstd::optional<Point2f> pRasterNew;
    SampledSpectrum L = ConnectBDPT(*this, *lambda, lightVertices, cameraVertices, s, t,
                                    lightSampler, camera, &sampler, &pRasterNew) *
                        nStrategies;
    if (pRasterNew)
        *pRaster = *pRasterNew;
    return L;
}

void MLTIntegrator::Render() {
    // Handle statistics and debugstart for MLTIntegrator
    if (Options->recordPixelStatistics)
        StatsEnablePixelStats(camera.GetFilm().PixelBounds(),
                              RemoveExtension(camera.GetFilm().GetFilename()));

    if (!Options->debugStart.empty()) {
        std::vector<std::string> c = SplitString(Options->debugStart, ',');
        if (c.empty())
            ErrorExit("Didn't find comma-separated values after --debugstart: %s",
                      Options->debugStart);

        int depth;
        if (!Atoi(c[0], &depth))
            ErrorExit("Unable to decode first --debugstart value: %s", c[0]);

        pstd::span<const std::string> span = pstd::MakeSpan(c);
        span.remove_prefix(1);
        DebugMLTSampler sampler = DebugMLTSampler::Create(span, nSampleStreams);

        Point2f pRaster;
        SampledWavelengths lambda;
        ScratchBuffer scratchBuffer(65536);
        (void)L(scratchBuffer, sampler, depth, &pRaster, &lambda);
        return;
    }

    thread_local MLTSampler *threadSampler = nullptr;
    thread_local int threadDepth;
    CheckCallbackScope _([&]() -> std::string {
        return StringPrintf("Rendering failed. Debug with --debugstart %d,%s\"\n",
                            threadDepth, threadSampler->DumpState());
    });

    // Generate bootstrap samples and compute normalization constant $b$
    Timer timer;
    int nBootstrapSamples = nBootstrap * (maxDepth + 1);
    std::vector<Float> bootstrapWeights(nBootstrapSamples, 0);
    // Allocate scratch buffers for MLT samples
    ThreadLocal<ScratchBuffer> threadScratchBuffers([]() { return ScratchBuffer(); });

    // Generate bootstrap samples in parallel
    ProgressReporter progress(nBootstrap, "Generating bootstrap paths", Options->quiet);
    ParallelFor(0, nBootstrap, [&](int64_t start, int64_t end) {
        ScratchBuffer &buf = threadScratchBuffers.Get();
        for (int64_t i = start; i < end; ++i) {
            // Generate _i_th bootstrap sample
            for (int depth = 0; depth <= maxDepth; ++depth) {
                int rngIndex = i * (maxDepth + 1) + depth;
                MLTSampler sampler(mutationsPerPixel, rngIndex, sigma,
                                   largeStepProbability, nSampleStreams);
                threadSampler = &sampler;
                threadDepth = depth;
                // Evaluate path radiance using _sampler_ and update _bootstrapWeights_
                Point2f pRaster;
                SampledWavelengths lambda;
                SampledSpectrum L_i = L(buf, sampler, depth, &pRaster, &lambda);
                bootstrapWeights[rngIndex] = c(L_i, lambda);

                buf.Reset();
            }
        }
        progress.Update(end - start);
    });
    progress.Done();

    if (std::accumulate(bootstrapWeights.begin(), bootstrapWeights.end(), 0.) == 0.)
        ErrorExit("No light carrying paths found during bootstrap sampling! "
                  "Are you trying to render a black image?");
    AliasTable bootstrapTable(bootstrapWeights);
    Float b = Float(maxDepth + 1) / bootstrapWeights.size() *
              std::accumulate(bootstrapWeights.begin(), bootstrapWeights.end(), 0.);

    // Set up connection to display server, if enabled
    std::atomic<int> finishedChains(0);
    if (!Options->displayServer.empty()) {
        DisplayDynamic(
            camera.GetFilm().GetFilename(),
            Point2i(camera.GetFilm().PixelBounds().Diagonal()), {"R", "G", "B"},
            [&](Bounds2i bounds, pstd::span<pstd::span<Float>> displayValue) {
                Film film = camera.GetFilm();
                Bounds2i pixelBounds = film.PixelBounds();
                int index = 0;
                for (Point2i p : bounds) {
                    Float finishedPixelMutations =
                        Float(finishedChains.load(std::memory_order_relaxed)) /
                        Float(nChains) * mutationsPerPixel;
                    Float scale = b / std::max<Float>(1, finishedPixelMutations);
                    RGB rgb = film.GetPixelRGB(pixelBounds.pMin + p, scale);
                    for (int c = 0; c < 3; ++c)
                        displayValue[c][index] = rgb[c];
                    ++index;
                }
            });
    }

    // Follow _nChains_ Markov chains to render image
    Film film = camera.GetFilm();
    int64_t nTotalMutations =
        (int64_t)film.SampleBounds().Area() * (int64_t)mutationsPerPixel;
    ProgressReporter progressRender(nChains, "Rendering", Options->quiet);
    // Run _nChains_ Markov chains in parallel
    ParallelFor(0, nChains, [&](int i) {
        ScratchBuffer &scratchBuffer = threadScratchBuffers.Get();
        // Compute number of mutations to apply in current Markov chain
        int64_t nChainMutations =
            std::min((i + 1) * nTotalMutations / nChains, nTotalMutations) -
            i * nTotalMutations / nChains;

        // Select initial state from the set of bootstrap samples
        RNG rng(i);
        int bootstrapIndex = bootstrapTable.Sample(rng.Uniform<Float>());
        int depth = bootstrapIndex % (maxDepth + 1);
        threadDepth = depth;

        // Initialize local variables for selected state
        MLTSampler sampler(mutationsPerPixel, bootstrapIndex, sigma, largeStepProbability,
                           nSampleStreams);
        threadSampler = &sampler;
        Point2f pCurrent;
        SampledWavelengths lambdaCurrent;
        SampledSpectrum LCurrent =
            L(scratchBuffer, sampler, depth, &pCurrent, &lambdaCurrent);

        // Run the Markov chain for _nChainMutations_ steps
        for (int64_t j = 0; j < nChainMutations; ++j) {
            StatsReportPixelStart(Point2i(pCurrent));
            sampler.StartIteration();
            // Generate proposed sample and compute its radiance
            Point2f pProposed;
            SampledWavelengths lambdaProposed;
            SampledSpectrum LProposed =
                L(scratchBuffer, sampler, depth, &pProposed, &lambdaProposed);

            // Compute acceptance probability for proposed sample
            Float cProposed = c(LProposed, lambdaProposed);
            Float cCurrent = c(LCurrent, lambdaCurrent);
            Float accept = std::min<Float>(1, cProposed / cCurrent);

            // Splat both current and proposed samples to _film_
            if (accept > 0)
                film.AddSplat(pProposed, LProposed * accept / cProposed, lambdaProposed);
            film.AddSplat(pCurrent, LCurrent * (1 - accept) / cCurrent, lambdaCurrent);

            // Accept or reject the proposal
            if (rng.Uniform<Float>() < accept) {
                StatsReportPixelEnd(Point2i(pCurrent));
                StatsReportPixelStart(Point2i(pProposed));
                pCurrent = pProposed;
                LCurrent = LProposed;
                lambdaCurrent = lambdaProposed;
                sampler.Accept();
                ++acceptedMutations;
            } else
                sampler.Reject();

            ++totalMutations;
            scratchBuffer.Reset();
            StatsReportPixelEnd(Point2i(pCurrent));
        }

        ++finishedChains;
        progressRender.Update(1);
    });

    progressRender.Done();

    // Store final image computed with MLT
    ImageMetadata metadata;
    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    camera.InitMetadata(&metadata);
    camera.GetFilm().WriteImage(metadata, b / mutationsPerPixel);
    DisconnectFromDisplayServer();
}

std::string MLTIntegrator::ToString() const {
    return StringPrintf("[ MLTIntegrator camera: %s maxDepth: %d nBootstrap: %d "
                        "nChains: %d mutationsPerPixel: %d sigma: %f "
                        "largeStepProbability: %f lightSampler: %s regularize: %s ]",
                        camera, maxDepth, nBootstrap, nChains, mutationsPerPixel, sigma,
                        largeStepProbability, lightSampler, regularize);
}

std::unique_ptr<MLTIntegrator> MLTIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Primitive aggregate,
    std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    int nBootstrap = parameters.GetOneInt("bootstrapsamples", 100000);
    int64_t nChains = parameters.GetOneInt("chains", 1000);
    int mutationsPerPixel = parameters.GetOneInt("mutationsperpixel", 100);
    if (Options->pixelSamples)
        mutationsPerPixel = *Options->pixelSamples;
    Float largeStepProbability = parameters.GetOneFloat("largestepprobability", 0.3f);
    Float sigma = parameters.GetOneFloat("sigma", .01f);
    if (Options->quickRender) {
        mutationsPerPixel = std::max(1, mutationsPerPixel / 16);
        nBootstrap = std::max(1, nBootstrap / 16);
    }
    bool regularize = parameters.GetOneBool("regularize", false);
    return std::make_unique<MLTIntegrator>(camera, aggregate, lights, maxDepth,
                                           nBootstrap, nChains, mutationsPerPixel, sigma,
                                           largeStepProbability, regularize);
}

STAT_RATIO("Stochastic Progressive Photon Mapping/Visible points checked per photon "
           "intersection",
           visiblePointsChecked, totalPhotonSurfaceInteractions);
STAT_COUNTER("Stochastic Progressive Photon Mapping/Photon paths followed", photonPaths);
STAT_INT_DISTRIBUTION(
    "Stochastic Progressive Photon Mapping/Grid cells per visible point",
    gridCellsPerVisiblePoint);
STAT_MEMORY_COUNTER("Memory/SPPM Pixels", pixelMemoryBytes);
STAT_MEMORY_COUNTER("Memory/SPPM BSDF and Grid Memory", sppmMemoryArenaBytes);

// SPPMPixel Definition
struct SPPMPixel {
    // SPPMPixel Public Members
    Float radius = 0;
    RGB Ld;
    struct VisiblePoint {
        // VisiblePoint Public Methods
        VisiblePoint() = default;
        VisiblePoint(const Point3f &p, const Vector3f &wo, const BSDF &bsdf,
                     const SampledSpectrum &beta, bool secondaryLambdaTerminated)
            : p(p),
              wo(wo),
              bsdf(bsdf),
              beta(beta),
              secondaryLambdaTerminated(secondaryLambdaTerminated) {}

        // VisiblePoint Public Members
        Point3f p;
        Vector3f wo;
        BSDF bsdf;
        SampledSpectrum beta;
        bool secondaryLambdaTerminated;

    } vp;
    AtomicFloat Phi_i[3];
    std::atomic<int> m{0};
    RGB tau;
    Float n = 0;
};

// SPPMPixelListNode Definition
struct SPPMPixelListNode {
    SPPMPixel *pixel;
    SPPMPixelListNode *next;
};

// SPPM Utility Functions
static bool ToGrid(Point3f p, const Bounds3f &bounds, const int gridRes[3], Point3i *pi) {
    bool inBounds = true;
    Vector3f pg = bounds.Offset(p);
    for (int i = 0; i < 3; ++i) {
        (*pi)[i] = (int)(gridRes[i] * pg[i]);
        inBounds &= (*pi)[i] >= 0 && (*pi)[i] < gridRes[i];
        (*pi)[i] = Clamp((*pi)[i], 0, gridRes[i] - 1);
    }
    return inBounds;
}

// SPPM Method Definitions
void SPPMIntegrator::Render() {
    // Initialize local variables for _SPPMIntegrator::Render()_
    if (Options->recordPixelStatistics)
        StatsEnablePixelStats(camera.GetFilm().PixelBounds(),
                              RemoveExtension(camera.GetFilm().GetFilename()));
    // Define variables for commonly used values in SPPM rendering
    int nIterations = samplerPrototype.SamplesPerPixel();
    ProgressReporter progress(2 * nIterations, "Rendering", Options->quiet);
    const Float invSqrtSPP = 1.f / std::sqrt(nIterations);
    Film film = camera.GetFilm();
    Bounds2i pixelBounds = film.PixelBounds();
    int nPixels = pixelBounds.Area();

    // Initialize _pixels_ array for SPPM
    CHECK(!pixelBounds.IsEmpty());
    Array2D<SPPMPixel> pixels(pixelBounds);
    for (SPPMPixel &p : pixels)
        p.radius = initialSearchRadius;
    pixelMemoryBytes += pixels.size() * sizeof(SPPMPixel);

    // Create light samplers for SPPM rendering
    BVHLightSampler lightSampler(lights, Allocator());
    PowerLightSampler shootLightSampler(lights, Allocator());

    // Allocate per-thread _ScratchBuffer_s for SPPM rendering
    ThreadLocal<ScratchBuffer> threadScratchBuffers(
        []() { return ScratchBuffer(1024 * 1024); });

    // Allocate samplers for SPPM rendering
    ThreadLocal<Sampler> threadSamplers(
        [this]() { return samplerPrototype.Clone(Allocator()); });
    pstd::vector<DigitPermutation> *digitPermutations(
        ComputeRadicalInversePermutations(digitPermutationsSeed));

    for (int iter = 0; iter < nIterations; ++iter) {
        // Connect to display server for SPPM if requested
        if (iter == 0 && !Options->displayServer.empty()) {
            DisplayDynamic(
                film.GetFilename(), Point2i(pixelBounds.Diagonal()), {"R", "G", "B"},
                [&](Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                    int index = 0;
                    uint64_t np = (uint64_t)(iter + 1) * (uint64_t)photonsPerIteration;
                    for (Point2i pPixel : b) {
                        const SPPMPixel &pixel = pixels[pPixel];
                        RGB rgb = pixel.Ld / (iter + 1) +
                                  pixel.tau / (np * Pi * Sqr(pixel.radius));
                        for (int c = 0; c < 3; ++c)
                            displayValue[c][index] = rgb[c];
                        ++index;
                    }
                });
        }

        // Generate SPPM visible points
        // Sample wavelengths for SPPM pass
        Float uLambda =
            Options->disableWavelengthJitter ? Float(0.5) : RadicalInverse(1, iter);
        const SampledWavelengths passLambda = film.SampleWavelengths(uLambda);

        ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
            // Follow camera paths for _tileBounds_ in image for SPPM
            ScratchBuffer &scratchBuffer = threadScratchBuffers.Get();
            Sampler sampler = threadSamplers.Get();
            for (Point2i pPixel : tileBounds) {
                sampler.StartPixelSample(pPixel, iter);
                // Generate camera ray for pixel for SPPM
                SampledWavelengths lambda = passLambda;
                CameraSample cs = GetCameraSample(sampler, pPixel, film.GetFilter());
                pstd::optional<CameraRayDifferential> crd =
                    camera.GenerateRayDifferential(cs, lambda);
                if (!crd || !crd->weight)
                    continue;
                SampledSpectrum beta = crd->weight;
                RayDifferential &ray = crd->ray;
                if (!Options->disablePixelJitter)
                    ray.ScaleDifferentials(invSqrtSPP);

                // Follow camera ray path until a visible point is created
                SPPMPixel &pixel = pixels[pPixel];
                Float etaScale = 1, bsdfPDF;
                bool specularBounce = true, haveSetVisiblePoint = false;
                LightSampleContext prevIntrCtx;
                int depth = 0;
                while (true) {
                    ++totalPhotonSurfaceInteractions;
                    pstd::optional<ShapeIntersection> si = Intersect(ray);
                    // Accumulate light contributions for ray with no intersection
                    if (!si) {
                        SampledSpectrum L(0.f);
                        // Incorporate emission from infinite lights for escaped ray
                        for (const auto &light : infiniteLights) {
                            SampledSpectrum Le = light.Le(ray, lambda);
                            if (depth == 0 || specularBounce)
                                L += beta * Le;
                            else {
                                // Compute MIS weight for infinite light
                                Float lightPDF = lightSampler.PMF(prevIntrCtx, light) *
                                                 light.PDF_Li(prevIntrCtx, ray.d, true);
                                Float w_b = PowerHeuristic(1, bsdfPDF, 1, lightPDF);

                                L += beta * w_b * Le;
                            }
                        }

                        pixel.Ld += film.ToOutputRGB(L, lambda);
                        break;
                    }

                    // Process SPPM camera ray intersection
                    // Compute BSDF at SPPM camera ray intersection
                    SurfaceInteraction &isect = si->intr;
                    BSDF bsdf =
                        isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
                    if (!bsdf) {
                        isect.SkipIntersection(&ray, si->tHit);
                        continue;
                    }

                    ++totalBSDFs;
                    // Add emission from directly visible emissive surfaces to _pixel.Ld_
                    Vector3f wo = -ray.d;
                    SampledSpectrum L(0.f);
                    // Incorporate emission from surface hit by ray
                    SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
                    if (Le) {
                        if (depth == 0 || specularBounce)
                            L += beta * Le;
                        else {
                            // Compute MIS weight for area light
                            Light areaLight(si->intr.areaLight);
                            Float lightPDF = lightSampler.PMF(prevIntrCtx, areaLight) *
                                             areaLight.PDF_Li(prevIntrCtx, ray.d, true);
                            Float w_l = PowerHeuristic(1, bsdfPDF, 1, lightPDF);

                            L += beta * w_l * Le;
                        }
                    }

                    pixel.Ld += film.ToOutputRGB(L, lambda);

                    // Terminate path at maximum depth or if visible point has been set
                    if (depth++ == maxDepth || haveSetVisiblePoint)
                        break;

                    // Accumulate direct illumination at SPPM camera ray intersection
                    SampledSpectrum Ld =
                        SampleLd(isect, bsdf, lambda, sampler, &lightSampler);
                    if (Ld)
                        pixel.Ld += film.ToOutputRGB(beta * Ld, lambda);

                    // Possibly create visible point and end camera path
                    BxDFFlags flags = bsdf.Flags();
                    if (IsDiffuse(flags) || (IsGlossy(flags) && depth == maxDepth)) {
                        pixel.vp = {isect.p(), wo, bsdf, beta,
                                    lambda.SecondaryTerminated()};
                        haveSetVisiblePoint = true;
                    }

                    // Spawn ray from SPPM camera path vertex
                    Float u = sampler.Get1D();
                    pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
                    if (!bs)
                        break;
                    specularBounce = bs->IsSpecular();
                    if (bs->IsTransmission())
                        etaScale *= Sqr(bs->eta);

                    beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
                    bsdfPDF = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi) : bs->pdf;

                    SampledSpectrum rrBeta = beta * etaScale;
                    if (rrBeta.MaxComponentValue() < 1) {
                        Float q = std::max<Float>(.05f, 1 - rrBeta.MaxComponentValue());
                        if (sampler.Get1D() < q)
                            break;
                        beta /= 1 - q;
                    }
                    ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);
                    prevIntrCtx = LightSampleContext(isect);
                }
            }
        });
        progress.Update();
        // Create grid of all SPPM visible points
        // Allocate grid for SPPM visible points
        int hashSize = NextPrime(nPixels);
        std::vector<std::atomic<SPPMPixelListNode *>> grid(hashSize);

        // Compute grid bounds for SPPM visible points
        Bounds3f gridBounds;
        Float maxRadius = 0;
        for (const SPPMPixel &pixel : pixels) {
            if (!pixel.vp.beta)
                continue;
            Bounds3f vpBound = Expand(Bounds3f(pixel.vp.p), pixel.radius);
            gridBounds = Union(gridBounds, vpBound);
            maxRadius = std::max(maxRadius, pixel.radius);
        }

        // Compute resolution of SPPM grid in each dimension
        int gridRes[3];
        Vector3f diag = gridBounds.Diagonal();
        Float maxDiag = MaxComponentValue(diag);
        int baseGridRes = int(maxDiag / maxRadius);
        for (int i = 0; i < 3; ++i)
            gridRes[i] = std::max<int>(baseGridRes * diag[i] / maxDiag, 1);

        // Add visible points to SPPM grid
        ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
            ScratchBuffer &scratchBuffer = threadScratchBuffers.Get();
            for (Point2i pPixel : tileBounds) {
                SPPMPixel &pixel = pixels[pPixel];
                if (pixel.vp.beta) {
                    // Add pixel's visible point to applicable grid cells
                    // Find grid cell bounds for pixel's visible point, _pMin_ and _pMax_
                    Float r = pixel.radius;
                    Point3i pMin, pMax;
                    ToGrid(pixel.vp.p - Vector3f(r, r, r), gridBounds, gridRes, &pMin);
                    ToGrid(pixel.vp.p + Vector3f(r, r, r), gridBounds, gridRes, &pMax);

                    for (int z = pMin.z; z <= pMax.z; ++z)
                        for (int y = pMin.y; y <= pMax.y; ++y)
                            for (int x = pMin.x; x <= pMax.x; ++x) {
                                // Add visible point to grid cell $(x, y, z)$
                                int h = Hash(Point3i(x, y, z)) % hashSize;
                                CHECK_GE(h, 0);
                                SPPMPixelListNode *node =
                                    scratchBuffer.Alloc<SPPMPixelListNode>();
                                node->pixel = &pixel;

                                // Atomically add _node_ to the start of _grid[h]_'s
                                // linked list
                                node->next = grid[h];
                                while (!grid[h].compare_exchange_weak(node->next, node))
                                    ;
                            }
                    gridCellsPerVisiblePoint << (1 + pMax.x - pMin.x) *
                                                    (1 + pMax.y - pMin.y) *
                                                    (1 + pMax.z - pMin.z);
                }
            }
        });

        // Trace photons and accumulate contributions
        // Create per-thread scratch buffers for photon shooting
        ThreadLocal<ScratchBuffer> photonShootScratchBuffers(
            []() { return ScratchBuffer(); });

        ParallelFor(0, photonsPerIteration, [&](int64_t start, int64_t end) {
            // Follow photon paths for photon index range _start_ - _end_
            ScratchBuffer &scratchBuffer = photonShootScratchBuffers.Get();
            Sampler sampler = threadSamplers.Get();
            for (int64_t photonIndex = start; photonIndex < end; ++photonIndex) {
                // Follow photon path for _photonIndex_
                // Define sampling lambda functions for photon shooting
                uint64_t haltonIndex =
                    (uint64_t)iter * (uint64_t)photonsPerIteration + photonIndex;
                int haltonDim = 0;
                auto Sample1D = [&]() {
                    Float u = ScrambledRadicalInverse(haltonDim, haltonIndex,
                                                      (*digitPermutations)[haltonDim]);
                    ++haltonDim;
                    return u;
                };

                auto Sample2D = [&]() {
                    Point2f u{
                        ScrambledRadicalInverse(haltonDim, haltonIndex,
                                                (*digitPermutations)[haltonDim]),
                        ScrambledRadicalInverse(haltonDim + 1, haltonIndex,
                                                (*digitPermutations)[haltonDim + 1])};
                    haltonDim += 2;
                    return u;
                };

                // Choose light to shoot photon from
                Float ul = Sample1D();
                pstd::optional<SampledLight> sampledLight = shootLightSampler.Sample(ul);
                if (!sampledLight)
                    continue;
                Light light = sampledLight->light;
                Float lightPDF = sampledLight->p;

                // Compute sample values for photon ray leaving light source
                Point2f uLight0 = Sample2D();
                Point2f uLight1 = Sample2D();
                Float uLightTime = camera.SampleTime(Sample1D());

                // Generate _photonRay_ from light source and initialize _beta_
                SampledWavelengths lambda = passLambda;
                pstd::optional<LightLeSample> les =
                    light.SampleLe(uLight0, uLight1, lambda, uLightTime);
                if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L)
                    continue;
                RayDifferential photonRay = RayDifferential(les->ray);
                SampledSpectrum beta = (les->AbsCosTheta(photonRay.d) * les->L) /
                                       (lightPDF * les->pdfPos * les->pdfDir);
                if (!beta)
                    continue;

                // Follow photon path through scene and record intersections
                SurfaceInteraction isect;
                for (int depth = 0; depth < maxDepth; ++depth) {
                    // Intersect photon ray with scene and return if ray escapes
                    pstd::optional<ShapeIntersection> si = Intersect(photonRay);
                    if (!si)
                        break;
                    SurfaceInteraction &isect = si->intr;

                    ++totalPhotonSurfaceInteractions;
                    if (depth > 0) {
                        // Add photon contribution to nearby visible points
                        Point3i photonGridIndex;
                        if (ToGrid(isect.p(), gridBounds, gridRes, &photonGridIndex)) {
                            int h = Hash(photonGridIndex) % hashSize;
                            CHECK_GE(h, 0);
                            // Add photon contribution to visible points in _grid[h]_
                            for (SPPMPixelListNode *node =
                                     grid[h].load(std::memory_order_relaxed);
                                 node; node = node->next) {
                                ++visiblePointsChecked;
                                SPPMPixel &pixel = *node->pixel;
                                if (DistanceSquared(pixel.vp.p, isect.p()) >
                                    Sqr(pixel.radius))
                                    continue;
                                // Update _pixel_ $\Phi$ and $m$ for nearby photon
                                Vector3f wi = -photonRay.d;
                                SampledSpectrum Phi =
                                    beta * pixel.vp.bsdf.f(pixel.vp.wo, wi);
                                // Update _Phi_i_ for photon contribution
                                SampledWavelengths photonLambda = lambda;
                                if (pixel.vp.secondaryLambdaTerminated)
                                    photonLambda.TerminateSecondary();
                                RGB Phi_i =
                                    film.ToOutputRGB(pixel.vp.beta * Phi, photonLambda);
                                for (int i = 0; i < 3; ++i)
                                    pixel.Phi_i[i].Add(Phi_i[i]);

                                ++pixel.m;
                            }
                        }
                    }
                    // Sample new photon ray direction
                    // Compute BSDF at photon intersection point
                    BSDF photonBSDF =
                        isect.GetBSDF(photonRay, lambda, camera, scratchBuffer, sampler);
                    if (!photonBSDF) {
                        isect.SkipIntersection(&photonRay, si->tHit);
                        --depth;
                        continue;
                    }

                    // Sample BSDF _fr_ and direction _wi_ for reflected photon
                    Vector3f wo = -photonRay.d;
                    pstd::optional<BSDFSample> bs = photonBSDF.Sample_f(
                        wo, Sample1D(), Sample2D(), TransportMode::Importance);
                    if (!bs)
                        break;
                    SampledSpectrum bnew =
                        beta * bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;

                    // Possibly terminate photon path with Russian roulette
                    Float betaRatio = bnew.MaxComponentValue() / beta.MaxComponentValue();
                    Float q = std::max<Float>(0, 1 - betaRatio);
                    if (Sample1D() < q)
                        break;
                    beta = bnew / (1 - q);

                    photonRay = RayDifferential(isect.SpawnRay(bs->wi));
                }

                scratchBuffer.Reset();
            }
        });
        // Reset _threadScratchBuffers_ after tracing photons
        threadScratchBuffers.ForAll([](ScratchBuffer &buffer) { buffer.Reset(); });

        progress.Update();
        photonPaths += photonsPerIteration;

        // Update pixel values from this pass's photons
        ParallelFor2D(pixelBounds, [&](Point2i pPixel) {
            SPPMPixel &p = pixels[pPixel];
            if (int m = p.m.load(std::memory_order_relaxed); m > 0) {
                // Compute new photon count and search radius given photons
                Float gamma = (Float)2 / (Float)3;
                Float nNew = p.n + gamma * m;
                Float rNew = p.radius * std::sqrt(nNew / (p.n + m));

                // Update $\tau$ for pixel
                RGB Phi_i(p.Phi_i[0], p.Phi_i[1], p.Phi_i[2]);
                p.tau = (p.tau + Phi_i) * Sqr(rNew) / Sqr(p.radius);

                // Set remaining pixel values for next photon pass
                p.n = nNew;
                p.radius = rNew;
                p.m = 0;
                for (int i = 0; i < 3; ++i)
                    p.Phi_i[i] = (Float)0;
            }
            // Reset _VisiblePoint_ in pixel
            p.vp.beta = SampledSpectrum(0.);
            p.vp.bsdf = BSDF();
        });

        // Periodically write SPPM image to disk
        if (iter + 1 == nIterations || (iter + 1 <= 64 && IsPowerOf2(iter + 1)) ||
            ((iter + 1) % 64 == 0)) {
            uint64_t np = (uint64_t)(iter + 1) * (uint64_t)photonsPerIteration;
            Image rgbImage(PixelFormat::Float, Point2i(pixelBounds.Diagonal()),
                           {"R", "G", "B"});

            ParallelFor2D(pixelBounds, [&](Point2i pPixel) {
                // Compute radiance _L_ for SPPM pixel _pPixel_
                const SPPMPixel &pixel = pixels[pPixel];
                RGB L = pixel.Ld / (iter + 1) + pixel.tau / (np * Pi * Sqr(pixel.radius));

                Point2i pImage = Point2i(pPixel - pixelBounds.pMin);
                rgbImage.SetChannels(pImage, {L.r, L.g, L.b});
            });

            ImageMetadata metadata;
            metadata.renderTimeSeconds = progress.ElapsedSeconds();
            metadata.samplesPerPixel = iter + 1;
            metadata.pixelBounds = pixelBounds;
            metadata.fullResolution = camera.GetFilm().FullResolution();
            metadata.colorSpace = colorSpace;
            camera.InitMetadata(&metadata);
            rgbImage.Write(camera.GetFilm().GetFilename(), metadata);

            // Write SPPM radius image, if requested
            if (getenv("SPPM_RADIUS")) {
                Image rimg(PixelFormat::Float, Point2i(pixelBounds.Diagonal()),
                           {"Radius"});
                Float minrad = 1e30f, maxrad = 0;
                for (const SPPMPixel &p : pixels) {
                    minrad = std::min(minrad, p.radius);
                    maxrad = std::max(maxrad, p.radius);
                }
                fprintf(stderr, "iterations: %d (%.2f s) radius range: %f - %f\n",
                        iter + 1, progress.ElapsedSeconds(), minrad, maxrad);
                for (Point2i pPixel : pixelBounds) {
                    const SPPMPixel &p = pixels[pPixel];
                    Float v = 1.f - (p.radius - minrad) / (maxrad - minrad);
                    Point2i pImage = Point2i(pPixel - pixelBounds.pMin);
                    rimg.SetChannel(pImage, 0, v);
                }
                ImageMetadata metadata;
                metadata.pixelBounds = pixelBounds;
                metadata.fullResolution = camera.GetFilm().FullResolution();
                rimg.Write("sppm_radius.png", metadata);
            }
        }
    }
#if 0
    // FIXME
    sppmMemoryArenaBytes += std::accumulate(perThreadArenas.begin(), perThreadArenas.end(),
                                            size_t(0), [&](size_t v, const MemoryArena &arena) {
                                                           return v + arena.BytesAllocated();
                                                       });
#endif
    progress.Done();
    DisconnectFromDisplayServer();
}

SampledSpectrum SPPMIntegrator::SampleLd(const SurfaceInteraction &intr, const BSDF &b,
                                         SampledWavelengths &lambda, Sampler sampler,
                                         LightSampler lightSampler) const {
    const BSDF *bsdf = &b;
    // Initialize _LightSampleContext_ for light sampling
    LightSampleContext ctx(intr);
    // Try to nudge the light sampling position to correct side of the surface
    BxDFFlags flags = bsdf->Flags();
    if (IsReflective(flags) && !IsTransmissive(flags))
        ctx.pi = intr.OffsetRayOrigin(intr.wo);
    else if (IsTransmissive(flags) && !IsReflective(flags))
        ctx.pi = intr.OffsetRayOrigin(-intr.wo);

    // Choose a light source for the direct lighting calculation
    Float u = sampler.Get1D();
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(ctx, u);
    Point2f uLight = sampler.Get2D();
    if (!sampledLight)
        return {};

    // Sample a point on the light source for direct lighting
    Light light = sampledLight->light;
    DCHECK(light && sampledLight->p > 0);
    pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
    if (!ls || !ls->L || ls->pdf == 0)
        return {};

    // Evaluate BSDF for light sample and check light visibility
    Vector3f wo = intr.wo, wi = ls->wi;
    SampledSpectrum f = bsdf->f(wo, wi) * AbsDot(wi, intr.shading.n);
    if (!f || !Unoccluded(intr, ls->pLight))
        return {};

    // Return light's contribution to reflected radiance
    Float p_l = sampledLight->p * ls->pdf;
    if (IsDeltaLight(light.Type()))
        return ls->L * f / p_l;
    else {
        Float p_b = bsdf->PDF(wo, wi);
        Float w_l = PowerHeuristic(1, p_l, 1, p_b);
        return w_l * ls->L * f / p_l;
    }
}

std::string SPPMIntegrator::ToString() const {
    return StringPrintf("[ SPPMIntegrator camera: %s initialSearchRadius: %f "
                        "maxDepth: %d photonsPerIteration: %d "
                        "colorSpace: %s digitPermutations: (elided) ]",
                        camera, initialSearchRadius, maxDepth, photonsPerIteration,
                        *colorSpace);
}

std::unique_ptr<SPPMIntegrator> SPPMIntegrator::Create(
    const ParameterDictionary &parameters, const RGBColorSpace *colorSpace, Camera camera,
    Sampler sampler, Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    int photonsPerIter = parameters.GetOneInt("photonsperiteration", -1);
    Float radius = parameters.GetOneFloat("radius", 1.f);
    int seed = parameters.GetOneInt("seed", 6502);
    return std::make_unique<SPPMIntegrator>(camera, sampler, aggregate, lights,
                                            photonsPerIter, maxDepth, radius, seed,
                                            colorSpace);
}

// FunctionIntegrator Method Definitions
FunctionIntegrator::FunctionIntegrator(std::function<double(Point2f)> func,
                                       const std::string &outputFilename, Camera camera,
                                       Sampler sampler, bool skipBad,
                                       std::string imageFilename)
    : Integrator(nullptr, {}),
      func(func),
      outputFilename(outputFilename),
      camera(camera),
      baseSampler(sampler),
      skipBad(skipBad),
      imageFilename(imageFilename) {}

namespace funcs {

static double step(Point2f p) {
    return (p.x < 0.5) ? 2 : 0;
}
static double diagonal(Point2f p) {
    return (p.x + p.y < 1) ? 2 : 0;
}
static double disk(Point2f p) {
    return Distance(p, Point2f(0.5, 0.5)) < 0.5 ? (1 / (Pi * Sqr(0.5))) : 0;
}
static double checkerboard(Point2f p) {
    int freq = 10;
    Point2i pi(p * freq);
    return ((pi.x & 1) ^ (pi.y & 1)) ? 2 : 0;
}
static double rotatedCheckerboard(Point2f p) {
    double angle = Radians(45);
    double nrm = 1.00006866455078125;
    static double sa = std::sin(angle), ca = std::cos(angle);
    return (double)checkerboard(
               {Float(10 + p.x * ca - p.y * sa), Float(10 + p.x * sa + p.y * ca)}) /
           nrm;
}
static double gaussian(Point2f p) {
    auto Gaussian = [](double x, double mu = 0, double sigma = 1) {
        return 1 / std::sqrt(2 * Pi * sigma * sigma) *
               std::exp(-Sqr(x - mu) / (2 * sigma * sigma));
    };
    auto GaussianIntegral = [](double x0, double x1, double mu = 0, double sigma = 1) {
        double sigmaRoot2 = sigma * double(1.414213562373095);
        return 0.5f *
               (std::erf((mu - x0) / sigmaRoot2) - std::erf((mu - x1) / sigmaRoot2));
    };

    double mu = 0.5, sigma = 0.25;
    static double nrm = Sqr(GaussianIntegral(0, 1, mu, sigma));
    return Gaussian(p.x, mu, sigma) * Gaussian(p.y, mu, sigma) / nrm;
}

}  // namespace funcs

std::unique_ptr<FunctionIntegrator> FunctionIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    const FileLoc *loc) {
    std::string funcName = parameters.GetOneString("function", "step");
    std::string defaultOut = Options->imageFile.empty()
                                 ? StringPrintf("%s-mse.txt", funcName)
                                 : Options->imageFile;
    std::string outputFilename = parameters.GetOneString("filename", defaultOut);

    std::function<Float(Point2f)> func;
    if (funcName == "step")
        func = funcs::step;
    else if (funcName == "diagonal")
        func = funcs::diagonal;
    else if (funcName == "disk")
        func = funcs::disk;
    else if (funcName == "checkerboard")
        func = funcs::checkerboard;
    else if (funcName == "rotatedcheckerboard")
        func = funcs::rotatedCheckerboard;
    else if (funcName == "gaussian")
        func = funcs::gaussian;

    if (!func)
        ErrorExit(loc, "%s: function for FunctionIntegrator unknown", funcName);

    if (sampler.Is<SobolSampler>())
        ErrorExit(loc, "\"sobol\" sampler should be replaced with \"paddedsobol\" for "
                       "the \"function\" integrator.");

    bool skipBad = parameters.GetOneBool("skipbad", true);
    std::string imageFilename = parameters.GetOneString("imagefilename", "");

    return std::make_unique<FunctionIntegrator>(func, outputFilename, camera, sampler,
                                                skipBad, imageFilename);
}

void FunctionIntegrator::Render() {
    std::string result;
    Bounds2i pixelBounds = camera.GetFilm().PixelBounds();
    int nPixels = pixelBounds.Area();
    Array2D<double> sumv(pixelBounds);
    int nSamples = baseSampler.SamplesPerPixel();

    if (!imageFilename.empty()) {
        RNG rng;
        Vector2i res = pixelBounds.pMax - pixelBounds.pMin;
        Image image(PixelFormat::Float, {res.x, res.y}, {"Y"});
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x) {
                int nSamples = 256;
                Float sum = 0;
                for (int i = 0; i < nSamples; ++i)
                    sum += func(Point2f((x + rng.Uniform<Float>()) / res.x,
                                        (y + rng.Uniform<Float>()) / res.y));
                image.SetChannel({x, y}, 0, sum / nSamples);
            }
        image.Write(imageFilename);
    }

    ProgressReporter prog(nSamples, "Sampling", Options->quiet);

    bool isHalton = baseSampler.Is<HaltonSampler>();
    bool isStratified = baseSampler.Is<StratifiedSampler>();
    bool isSobol = baseSampler.Is<PaddedSobolSampler>();
    std::vector<DigitPermutation> digitPermutations[2];
    std::vector<uint64_t> owenHash[2];
    RNG rng;
    if (isHalton) {
        for (int d = 0; d < 2; ++d) {
            digitPermutations[d].resize(nPixels);
            for (int i = 0; i < nPixels; ++i)
                digitPermutations[d][i] =
                    DigitPermutation(d == 0 ? 2 : 3, rng.Uniform<uint32_t>(), {});
        }
    }
    if (isHalton || isSobol) {
        for (int d = 0; d < 2; ++d) {
            owenHash[d].resize(nPixels);
            for (int i = 0; i < nPixels; ++i)
                owenHash[d][i] = rng.Uniform<uint64_t>();
        }
    }

    int nTakenSamples = 0;
    ThreadLocal<Sampler> threadSamplers([this]() { return baseSampler.Clone({}); });
    for (int sampleIndex = 0; sampleIndex < nSamples; ++sampleIndex) {
        bool reportResult = true;
        if (skipBad) {
            int nSamples = sampleIndex + 1;
            if (isStratified && Sqr(int(std::sqrt(nSamples))) != nSamples) {
                prog.Update();
                continue;
            } else if (isSobol && !IsPowerOf2(nSamples))
                reportResult = false;
            else if (isHalton) {
                int n2 = 0, n3 = 0;
                while (true) {
                    if ((nSamples % 2) == 0) {
                        nSamples /= 2;
                        ++n2;
                    } else if ((nSamples % 3) == 0) {
                        nSamples /= 3;
                        ++n3;
                    } else
                        break;
                }
                if (nSamples != 1 || n2 != n3)
                    reportResult = false;
            }
        }

        ++nTakenSamples;

        if (isStratified) {
            int spp = sampleIndex + 1;
            int factor = int(std::sqrt(spp));

            while ((spp % factor) != 0)
                --factor;

            ParallelFor2D(pixelBounds, [&](Point2i pPixel) {
                int pixelIndex = (pPixel.x - pixelBounds.pMin.x) +
                                 (pPixel.y - pixelBounds.pMin.y) *
                                     (pixelBounds.pMax.x - pixelBounds.pMin.x);
                RNG rng(pixelIndex, MixBits(sampleIndex + 1));

                Float v = 0;
                int nx = factor, ny = spp / factor;
                for (int x = 0; x < nx; ++x) {
                    for (int y = 0; y < ny; ++y) {
                        Point2f u{(x + rng.Uniform<Float>()) / nx,
                                  (y + rng.Uniform<Float>()) / ny};
                        v += func(u);
                    }
                }
                // Need nTakenSamples factor to cancel out division in sumSE computation.
                sumv[pPixel] = v * nTakenSamples / (nx * ny);
            });
        } else {
            ParallelFor2D(pixelBounds, [&](Bounds2i bounds) {
                for (Point2i pPixel : bounds) {
                    Point2f u;
                    if (isHalton) {
                        int pixelIndex = (pPixel.x - pixelBounds.pMin.x) +
                                         (pPixel.y - pixelBounds.pMin.y) *
                                             (pixelBounds.pMax.x - pixelBounds.pMin.x);
                        DCHECK_GE(pixelIndex, 0);
                        DCHECK_LT(pixelIndex, nPixels);

                        switch (
                            baseSampler.Cast<HaltonSampler>()->GetRandomizeStrategy()) {
                        case RandomizeStrategy::None:
                            u = Point2f(RadicalInverse(0, sampleIndex),
                                        RadicalInverse(1, sampleIndex));
                            break;
                        case RandomizeStrategy::PermuteDigits:
                            u = Point2f(
                                ScrambledRadicalInverse(0, sampleIndex,
                                                        digitPermutations[0][pixelIndex]),
                                ScrambledRadicalInverse(
                                    1, sampleIndex, digitPermutations[1][pixelIndex]));
                            break;
                        case RandomizeStrategy::Owen:
                            u = Point2f(OwenScrambledRadicalInverse(
                                            0, sampleIndex, owenHash[0][pixelIndex]),
                                        OwenScrambledRadicalInverse(
                                            1, sampleIndex, owenHash[1][pixelIndex]));
                            break;
                        default:
                            LOG_FATAL("Unhandled randomization strategy");
                        }
                    } else if (isSobol) {
                        int pixelIndex = (pPixel.x - pixelBounds.pMin.x) +
                                         (pPixel.y - pixelBounds.pMin.y) *
                                             (pixelBounds.pMax.x - pixelBounds.pMin.x);
                        DCHECK_GE(pixelIndex, 0);
                        DCHECK_LT(pixelIndex, nPixels);

                        switch (baseSampler.Cast<PaddedSobolSampler>()
                                    ->GetRandomizeStrategy()) {
                        case RandomizeStrategy::None:
                            u = Point2f(SobolSample(sampleIndex, 0, NoRandomizer()),
                                        SobolSample(sampleIndex, 1, NoRandomizer()));
                            break;
                        case RandomizeStrategy::PermuteDigits:
                            u = Point2f(
                                SobolSample(
                                    sampleIndex, 0,
                                    BinaryPermuteScrambler(owenHash[0][pixelIndex])),
                                SobolSample(
                                    sampleIndex, 1,
                                    BinaryPermuteScrambler(owenHash[1][pixelIndex])));
                            break;
                        case RandomizeStrategy::FastOwen:
                            u = Point2f(
                                SobolSample(sampleIndex, 0,
                                            FastOwenScrambler(owenHash[0][pixelIndex])),
                                SobolSample(sampleIndex, 1,
                                            FastOwenScrambler(owenHash[1][pixelIndex])));
                            break;
                        case RandomizeStrategy::Owen:
                            u = Point2f(
                                SobolSample(sampleIndex, 0,
                                            OwenScrambler(owenHash[0][pixelIndex])),
                                SobolSample(sampleIndex, 1,
                                            OwenScrambler(owenHash[1][pixelIndex])));
                            break;
                        default:
                            LOG_FATAL("Unhandled randomization strategy");
                        }

                    } else {
                        Sampler sampler = threadSamplers.Get();
                        sampler.StartPixelSample(pPixel, sampleIndex, 0);
                        u = sampler.GetPixel2D();
                    }
                    sumv[pPixel] += func(u);
                }
            });
        }

        // Compute average MSE/variance
        if (reportResult) {
            double sumSE = 0;
            for (double v : sumv)
                sumSE += Sqr(v / nTakenSamples - 1);
            Float mse = sumSE / nPixels;

            result += StringPrintf("%d %f\n", sampleIndex + 1, mse);
        }
        prog.Update();
    }

    // Make sure that it's basically one...
    double sum = std::accumulate(sumv.begin(), sumv.end(), 0.);
    double avg = sum / (double(sumv.size()) * double(nTakenSamples));
    if (avg < 0.999 || avg > 1.001)
        Warning("Average estimate is %f, which is suspiciously far from 1.", avg);

    prog.Done();

    WriteFileContents(outputFilename, result);
}

std::string FunctionIntegrator::ToString() const {
    return StringPrintf(
        "[ FunctionIntegrator outputFilename: %s camera: %s baseSampler: %s ]",
        outputFilename, camera, baseSampler);
}

std::unique_ptr<Integrator> Integrator::Create(
    const std::string &name, const ParameterDictionary &parameters, Camera camera,
    Sampler sampler, Primitive aggregate, std::vector<Light> lights,
    const RGBColorSpace *colorSpace, const FileLoc *loc) {
    std::unique_ptr<Integrator> integrator;
    if (name == "path")
        integrator =
            PathIntegrator::Create(parameters, camera, sampler, aggregate, lights, loc);
    else if (name == "function")
        integrator = FunctionIntegrator::Create(parameters, camera, sampler, loc);
    else if (name == "simplepath")
        integrator = SimplePathIntegrator::Create(parameters, camera, sampler, aggregate,
                                                  lights, loc);
    else if (name == "lightpath")
        integrator = LightPathIntegrator::Create(parameters, camera, sampler, aggregate,
                                                 lights, loc);
    else if (name == "simplevolpath")
        integrator = SimpleVolPathIntegrator::Create(parameters, camera, sampler,
                                                     aggregate, lights, loc);
    else if (name == "volpath")
        integrator = VolPathIntegrator::Create(parameters, camera, sampler, aggregate,
                                               lights, loc);
    else if (name == "bdpt")
        integrator =
            BDPTIntegrator::Create(parameters, camera, sampler, aggregate, lights, loc);
    else if (name == "mlt")
        integrator = MLTIntegrator::Create(parameters, camera, aggregate, lights, loc);
    else if (name == "ambientocclusion")
        integrator = AOIntegrator::Create(parameters, &colorSpace->illuminant, camera,
                                          sampler, aggregate, lights, loc);
    else if (name == "randomwalk")
        integrator = RandomWalkIntegrator::Create(parameters, camera, sampler, aggregate,
                                                  lights, loc);
    else if (name == "sppm")
        integrator = SPPMIntegrator::Create(parameters, colorSpace, camera, sampler,
                                            aggregate, lights, loc);
    else
        ErrorExit(loc, "%s: integrator type unknown.", name);

    if (!integrator)
        ErrorExit(loc, "%s: unable to create integrator.", name);

    parameters.ReportUnused();
    return integrator;
}

}  // namespace pbrt
