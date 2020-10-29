// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_INTEGRATORS_H
#define PBRT_CPU_INTEGRATORS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/sampler.h>
#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

// Integrator Definition
class Integrator {
  public:
    // Integrator Public Methods
    virtual ~Integrator();

    static std::unique_ptr<Integrator> Create(const std::string &name,
                                              const ParameterDictionary &parameters,
                                              CameraHandle camera, SamplerHandle sampler,
                                              PrimitiveHandle aggregate,
                                              std::vector<LightHandle> lights,
                                              const RGBColorSpace *colorSpace,
                                              const FileLoc *loc);

    virtual std::string ToString() const = 0;

    const Bounds3f &SceneBounds() const { return sceneBounds; }

    virtual void Render() = 0;

    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const;
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    bool Unoccluded(const Interaction &p0, const Interaction &p1) const {
        return !IntersectP(p0.SpawnRayTo(p1), 1 - ShadowEpsilon);
    }

    SampledSpectrum Tr(const Interaction &p0, const Interaction &p1,
                       const SampledWavelengths &lambda, RNG &rng) const;

    // Integrator Public Members
    std::vector<LightHandle> lights;
    PrimitiveHandle aggregate;
    std::vector<LightHandle> infiniteLights;

  protected:
    // Integrator Private Methods
    Integrator(PrimitiveHandle aggregate, std::vector<LightHandle> lights)
        : lights(lights), aggregate(aggregate) {
        // Integrator Constructor Implementation
        if (aggregate)
            sceneBounds = aggregate.Bounds();

        for (auto &light : lights) {
            light.Preprocess(sceneBounds);
            if (light.Type() == LightType::Infinite)
                infiniteLights.push_back(light);
        }
    }

    // Integrator Private Members
    Bounds3f sceneBounds;
};

// ImageTileIntegrator Definition
class ImageTileIntegrator : public Integrator {
  public:
    // ImageTileIntegrator Public Methods
    ImageTileIntegrator(CameraHandle camera, SamplerHandle sampler,
                        PrimitiveHandle aggregate, std::vector<LightHandle> lights)
        : Integrator(aggregate, lights), camera(camera), samplerPrototype(sampler) {}

    void Render();

    virtual void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                                     SamplerHandle sampler,
                                     ScratchBuffer &scratchBuffer) = 0;

  protected:
    // ImageTileIntegrator Protected Members
    CameraHandle camera;
    SamplerHandle samplerPrototype;
};

// RayIntegrator Definition
class RayIntegrator : public ImageTileIntegrator {
  public:
    // RayIntegrator Public Methods
    RayIntegrator(CameraHandle camera, SamplerHandle sampler, PrimitiveHandle aggregate,
                  std::vector<LightHandle> lights)
        : ImageTileIntegrator(camera, sampler, aggregate, lights) {}

    void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                             SamplerHandle sampler, ScratchBuffer &scratchBuffer) final;

    virtual SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                               SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                               VisibleSurface *visibleSurface) const = 0;
};

// RandomWalkIntegrator Definition
class RandomWalkIntegrator : public RayIntegrator {
  public:
    // RandomWalkIntegrator Public Methods
    RandomWalkIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                         PrimitiveHandle aggregate, std::vector<LightHandle> lights)
        : RayIntegrator(camera, sampler, aggregate, lights), maxDepth(maxDepth) {}
    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface = nullptr) const;

    static std::unique_ptr<RandomWalkIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // RandomWalkIntegrator Private Methods
    SampledSpectrum LiRandomWalk(RayDifferential ray, SampledWavelengths &lambda,
                                 SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                                 int depth) const;

    // RandomWalkIntegrator Private Members
    int maxDepth;
};

// SimplePathIntegrator Definition
class SimplePathIntegrator : public RayIntegrator {
  public:
    // SimplePathIntegrator Public Methods
    SimplePathIntegrator(int maxDepth, bool sampleLights, bool sampleBSDF,
                         CameraHandle camera, SamplerHandle sampler,
                         PrimitiveHandle aggregate, std::vector<LightHandle> lights);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<SimplePathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SimplePathIntegrator Private Members
    int maxDepth;
    bool sampleLights, sampleBSDF;
    UniformLightSampler lightSampler;
};

// PathIntegrator Definition
class PathIntegrator : public RayIntegrator {
  public:
    // PathIntegrator Public Methods
    PathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                   PrimitiveHandle aggregate, std::vector<LightHandle> lights,
                   const std::string &lightSampleStrategy = "bvh",
                   bool regularize = false);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<PathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // PathIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, SamplerHandle sampler) const;

    // PathIntegrator Private Members
    int maxDepth;
    LightSamplerHandle lightSampler;
    bool regularize;
};

// SimpleVolPathIntegrator Definition
class SimpleVolPathIntegrator : public RayIntegrator {
  public:
    // SimpleVolPathIntegrator Public Methods
    SimpleVolPathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                            PrimitiveHandle aggregate, std::vector<LightHandle> lights);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<SimpleVolPathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SimpleVolPathIntegrator Private Members
    int maxDepth;
};

// VolPathIntegrator Definition
class VolPathIntegrator : public RayIntegrator {
  public:
    // VolPathIntegrator Public Methods
    VolPathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                      PrimitiveHandle aggregate, std::vector<LightHandle> lights,
                      const std::string &lightSampleStrategy = "bvh",
                      bool regularize = false)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          lightSampler(
              LightSamplerHandle::Create(lightSampleStrategy, lights, Allocator())),
          regularize(regularize) {}

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<VolPathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // VolPathIntegrator Private Methods
    SampledSpectrum SampleLd(const Interaction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, SamplerHandle sampler,
                             const SampledSpectrum &beta,
                             const SampledSpectrum &pathPDF) const;

    static void rescale(SampledSpectrum &beta, SampledSpectrum &pdfLight,
                        SampledSpectrum &pdfUni) {
        if (beta.MaxComponentValue() > 0x1p24f ||
            pdfLight.MaxComponentValue() > 0x1p24f ||
            pdfUni.MaxComponentValue() > 0x1p24f) {
            beta *= 1.f / 0x1p24f;
            pdfLight *= 1.f / 0x1p24f;
            pdfUni *= 1.f / 0x1p24f;
        } else if (beta.MaxComponentValue() < 0x1p-24f ||
                   pdfLight.MaxComponentValue() < 0x1p-24f ||
                   pdfUni.MaxComponentValue() < 0x1p-24f) {
            beta *= 0x1p24f;
            pdfLight *= 0x1p24f;
            pdfUni *= 0x1p24f;
        }
    }

    // VolPathIntegrator Private Members
    int maxDepth;
    LightSamplerHandle lightSampler;
    bool regularize;
};

// AOIntegrator Definition
class AOIntegrator : public RayIntegrator {
  public:
    // AOIntegrator Public Methods
    AOIntegrator(bool cosSample, Float maxDist, CameraHandle camera,
                 SamplerHandle sampler, PrimitiveHandle aggregate,
                 std::vector<LightHandle> lights, SpectrumHandle illuminant);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<AOIntegrator> Create(
        const ParameterDictionary &parameters, SpectrumHandle illuminant,
        CameraHandle camera, SamplerHandle sampler, PrimitiveHandle aggregate,
        std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    bool cosSample;
    Float maxDist;
    SpectrumHandle illuminant;
    Float illumScale;
};

// LightPathIntegrator Definition
class LightPathIntegrator : public ImageTileIntegrator {
  public:
    // LightPathIntegrator Public Methods
    LightPathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                        PrimitiveHandle aggregate, std::vector<LightHandle> lights);

    void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                             SamplerHandle sampler, ScratchBuffer &scratchBuffer);

    static std::unique_ptr<LightPathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // LightPathIntegrator Private Data
    int maxDepth;
    std::unique_ptr<PowerLightSampler> lightSampler;
};

// BDPTIntegrator Definition
struct Vertex;
class BDPTIntegrator : public RayIntegrator {
  public:
    // BDPTIntegrator Public Methods
    BDPTIntegrator(CameraHandle camera, SamplerHandle sampler, PrimitiveHandle aggregate,
                   std::vector<LightHandle> lights, int maxDepth,
                   bool visualizeStrategies, bool visualizeWeights,
                   const std::string &lightSampleStrategy = "power",
                   bool regularize = false)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          visualizeStrategies(visualizeStrategies),
          visualizeWeights(visualizeWeights),
          lightSampleStrategy(lightSampleStrategy),
          lightSampler(new PowerLightSampler(lights, Allocator())),
          regularize(regularize) {}

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<BDPTIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

    void Render();

  private:
    // BDPTIntegrator Private Members
    int maxDepth;
    bool visualizeStrategies;
    bool visualizeWeights;
    std::string lightSampleStrategy;
    bool regularize;
    LightSamplerHandle lightSampler;
    mutable std::vector<FilmHandle> weightFilms;
};

// MLTIntegrator Definition
class MLTSampler;

class MLTIntegrator : public Integrator {
  public:
    // MLTIntegrator Public Methods
    MLTIntegrator(CameraHandle camera, PrimitiveHandle aggregate,
                  std::vector<LightHandle> lights, int maxDepth, int nBootstrap,
                  int nChains, int mutationsPerPixel, Float sigma,
                  Float largeStepProbability, bool regularize)
        : Integrator(aggregate, lights),
          lightSampler(new PowerLightSampler(lights, Allocator())),
          camera(camera),
          maxDepth(maxDepth),
          nBootstrap(nBootstrap),
          nChains(nChains),
          mutationsPerPixel(mutationsPerPixel),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          regularize(regularize) {}

    void Render();

    static std::unique_ptr<MLTIntegrator> Create(const ParameterDictionary &parameters,
                                                 CameraHandle camera,
                                                 PrimitiveHandle aggregate,
                                                 std::vector<LightHandle> lights,
                                                 const FileLoc *loc);

    std::string ToString() const;

  private:
    // MLTIntegrator Constants
    static constexpr int cameraStreamIndex = 0;
    static constexpr int lightStreamIndex = 1;
    static constexpr int connectionStreamIndex = 2;
    static constexpr int nSampleStreams = 3;

    // MLTIntegrator Private Methods
    SampledSpectrum L(ScratchBuffer &scratchBuffer, MLTSampler &sampler, int k,
                      Point2f *pRaster, SampledWavelengths *lambda);

    // MLTIntegrator Private Members
    LightSamplerHandle lightSampler;
    bool regularize;
    CameraHandle camera;
    int maxDepth;
    int nBootstrap;
    int mutationsPerPixel;
    Float sigma, largeStepProbability;
    int nChains;
};

// SPPMIntegrator Definition
class SPPMIntegrator : public Integrator {
  public:
    // SPPMIntegrator Public Methods
    SPPMIntegrator(CameraHandle camera, PrimitiveHandle aggregate,
                   std::vector<LightHandle> lights, int nIterations,
                   int photonsPerIteration, int maxDepth, Float initialSearchRadius,
                   bool regularize, int seed, const RGBColorSpace *colorSpace)
        : Integrator(aggregate, lights),
          camera(camera),
          initialSearchRadius(initialSearchRadius),
          nIterations(nIterations),
          maxDepth(maxDepth),
          photonsPerIteration(photonsPerIteration > 0
                                  ? photonsPerIteration
                                  : camera.GetFilm().PixelBounds().Area()),
          regularize(regularize),
          colorSpace(colorSpace),
          digitPermutationsSeed(seed) {}

    static std::unique_ptr<SPPMIntegrator> Create(const ParameterDictionary &parameters,
                                                  const RGBColorSpace *colorSpace,
                                                  CameraHandle camera,
                                                  PrimitiveHandle aggregate,
                                                  std::vector<LightHandle> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

    void Render();

  private:
    // SPPMIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, SamplerHandle sampler,
                             LightSamplerHandle lightSampler) const;

    // SPPMIntegrator Private Members
    CameraHandle camera;
    Float initialSearchRadius;
    int digitPermutationsSeed;
    int nIterations;
    bool regularize;
    int maxDepth;
    int photonsPerIteration;
    const RGBColorSpace *colorSpace;
};

}  // namespace pbrt

#endif  // PBRT_CPU_INTEGRATORS_H
