// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/cameras.h>
#include <pbrt/cpu/accelerators.h>
#include <pbrt/cpu/integrators.h>
#include <pbrt/filters.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/pbrt.h>
#include <pbrt/samplers.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/image.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <memory>

using namespace pbrt;

static std::string inTestDir(const std::string &path) {
    return path;
}

struct TestScene {
    PrimitiveHandle aggregate;
    std::vector<LightHandle> lights;
    std::string description;
    float expected;
};

struct TestIntegrator {
    Integrator *integrator;
    const FilmHandle film;
    std::string description;
    TestScene scene;
};

void PrintTo(const TestIntegrator &tr, ::std::ostream *os) {
    *os << tr.description;
}

void CheckSceneAverage(const std::string &filename, float expected) {
    pstd::optional<ImageAndMetadata> im = Image::Read(filename);
    ASSERT_TRUE((bool)im);
    ASSERT_EQ(im->image.NChannels(), 3);

    float delta = .025;
    float sum = 0;

    Image &image = im->image;
    for (int t = 0; t < image.Resolution()[1]; ++t)
        for (int s = 0; s < image.Resolution()[0]; ++s)
            for (int c = 0; c < 3; ++c)
                sum += image.GetChannel(Point2i(s, t), c);
    int nPixels = image.Resolution().x * image.Resolution().y * 3;
    EXPECT_NEAR(expected, sum / nPixels, delta);
}

std::vector<TestScene> GetScenes() {
    std::vector<TestScene> scenes;

    Allocator alloc;
    static Transform identity;
    {
        // Unit sphere, Kd = 0.5, point light I = 3.1415 at center
        // -> With GI, should have radiance of 1.
        ShapeHandle sphere = new Sphere(&identity, &identity,
                                        true /* reverse orientation */, 1, -1, 1, 360);

        static ConstantSpectrum cs(0.5);
        SpectrumTextureHandle Kd = alloc.new_object<SpectrumConstantTexture>(&cs);
        FloatTextureHandle sigma = alloc.new_object<FloatConstantTexture>(0.);
        // FIXME: here and below, Materials leak...
        MaterialHandle material = new DiffuseMaterial(Kd, sigma, nullptr);

        MediumInterface mediumInterface;
        std::vector<PrimitiveHandle> prims;
        prims.push_back(PrimitiveHandle(
            new GeometricPrimitive(sphere, material, nullptr, mediumInterface)));
        PrimitiveHandle bvh(new BVHAccel(std::move(prims)));

        // We have to do this little dance here to make sure the spectrum is
        // properly normalized (this is usually all handled inside *Light::Create())
        ConstantSpectrum I(1);
        Float scale = Pi / SpectrumToPhotometric(&I);
        std::vector<LightHandle> lights;
        lights.push_back(new PointLight(identity, MediumInterface(), &I, scale, Allocator()));

        scenes.push_back({bvh, lights, "Sphere, 1 light, Kd = 0.5", 1.0});
    }

    {
        // Unit sphere, Kd = 0.5, 4 point lights I = 3.1415/4 at center
        // -> With GI, should have radiance of 1.
        ShapeHandle sphere = new Sphere(&identity, &identity,
                                        true /* reverse orientation */, 1, -1, 1, 360);

        static ConstantSpectrum cs(0.5);
        SpectrumTextureHandle Kd = alloc.new_object<SpectrumConstantTexture>(&cs);
        FloatTextureHandle sigma = alloc.new_object<FloatConstantTexture>(0.);
        const MaterialHandle material = new DiffuseMaterial(Kd, sigma, nullptr);

        MediumInterface mediumInterface;
        std::vector<PrimitiveHandle> prims;
        prims.push_back(PrimitiveHandle(
            new GeometricPrimitive(sphere, material, nullptr, mediumInterface)));
        PrimitiveHandle bvh(new BVHAccel(std::move(prims)));

        // We have to do this little dance here to make sure the spectrum is
        // properly normalized (this is usually all handled inside *Light::Create())
        ConstantSpectrum I(1);
        Float scale = Pi / (4 * SpectrumToPhotometric(&I));
        std::vector<LightHandle> lights;
        lights.push_back(new PointLight(identity, MediumInterface(), &I, scale, Allocator()));
        lights.push_back(new PointLight(identity, MediumInterface(), &I, scale, Allocator()));
        lights.push_back(new PointLight(identity, MediumInterface(), &I, scale, Allocator()));
        lights.push_back(new PointLight(identity, MediumInterface(), &I, scale, Allocator()));

        scenes.push_back({bvh, lights, "Sphere, 1 light, Kd = 0.5", 1.0});
    }

    {
        // Unit sphere, Kd = 0.5, Le = 0.5
        // -> With GI, should have radiance of 1.
        ShapeHandle sphere = new Sphere(&identity, &identity,
                                        true /* reverse orientation */, 1, -1, 1, 360);

        static ConstantSpectrum cs(0.5);
        SpectrumTextureHandle Kd = alloc.new_object<SpectrumConstantTexture>(&cs);
        FloatTextureHandle sigma = alloc.new_object<FloatConstantTexture>(0.);
        const MaterialHandle material = new DiffuseMaterial(Kd, sigma, nullptr);

        // We have to do this little dance here to make sure the spectrum is
        // properly normalized (this is usually all handled inside *Light::Create())
        ConstantSpectrum Le(1);
        Float scale = 0.5 / SpectrumToPhotometric(&Le);
        LightHandle areaLight =
            new DiffuseAreaLight(identity, MediumInterface(), &Le, scale, sphere, Image(),
                                 nullptr, false, Allocator());

        std::vector<LightHandle> lights;
        lights.push_back(areaLight);

        MediumInterface mediumInterface;
        std::vector<PrimitiveHandle> prims;
        prims.push_back(PrimitiveHandle(
            new GeometricPrimitive(sphere, material, lights.back(), mediumInterface)));
        PrimitiveHandle bvh(new BVHAccel(std::move(prims)));

        scenes.push_back({bvh, lights, "Sphere, Kd = 0.5, Le = 0.5", 1.0});
    }

#if 0
    {
        // Unit sphere, Kd = 0.25, Kr = .5, point light I = 7.4 at center
        // -> With GI, should have radiance of ~1.
        ShapeHandle sphere = new Sphere(
            &identity, &identity, true /* reverse orientation */, 1, -1, 1, 360);

        static ConstantSpectrum cs5(0.5), cs25(0.25);
        SpectrumTextureHandle Kd =
            alloc.new_object<SpectrumConstantTexture>(&cs25);
        SpectrumTextureHandle Kr =
            alloc.new_object<SpectrumConstantTexture>(&cs5);
        SpectrumTextureHandle black =
            alloc.new_object<SpectrumConstantTexture>(Spectra::Zero());
        SpectrumTextureHandle white =
            alloc.new_object<SpectrumConstantTexture>(Spectra::One());
        FloatTextureHandle zero =
            alloc.new_object<FloatConstantTexture>(0.);
        FloatTextureHandle one =
            alloc.new_object<FloatConstantTexture>(1.);
        const MaterialHandle material = new UberMaterial(
            Kd, black, Kr, black, zero, zero, one, nullptr, false, nullptr);

        MediumInterface mediumInterface;
        std::vector<PrimitiveHandle> prims;
        prims.push_back(PrimitiveHandle(new GeometricPrimitive(
            sphere, material, nullptr, mediumInterface)));
        PrimitiveHandle bvh(new BVHAccel(std::move(prims)));

        static ConstantSpectrum I(3. * Pi);
        std::vector<LightHandle> lights;
        lights.push_back(std::make_unique<PointLight>(identity,
                                                      nullptr, &I, Allocator()));

        scenes.push_back({bvh, lights, "Sphere, 1 light, Kd = 0.25 Kr = 0.5", 1.0});
    }
#endif

#if 0
  {
    // Unit sphere, Kd = 0.25, Kr = .5, Le .587
    // -> With GI, should have radiance of ~1.
    ShapeHandle sphere = new Sphere(
        &identity, &identity, true /* reverse orientation */, 1, -1, 1, 360);

    static ConstantSpectrum cs5(0.5), cs25(0.25);
    SpectrumTextureHandle Kd =
        alloc.new_object<SpectrumConstantTexture>(&cs25);
    SpectrumTextureHandle Kr =
        alloc.new_object<SpectrumConstantTexture>(&cs5);
    SpectrumTextureHandle black =
        alloc.new_object<SpectrumConstantTexture>(Spectra::Zero());
    SpectrumTextureHandle white =
        alloc.new_object<SpectrumConstantTexture>(Spectra::One());
    FloatTextureHandle zero =
        alloc.new_object<FloatConstantTexture>(0.);
    FloatTextureHandle one =
        alloc.new_object<FloatConstantTexture>(1.);
    std::shared_ptr<Material> material = std::make_shared<UberMaterial>(
        Kd, black, Kr, black, zero, zero, zero, white, one, nullptr, false, nullptr);

    static ConstantSpectrum Le(0.587);
    std::shared_ptr<AreaLight> areaLight = std::make_shared<DiffuseAreaLight>(
        identity, nullptr, &Le, 8, sphere, true, false,
        std::make_shared<ParameterDictionary>(std::initializer_list<const NamedValues *>{}, nullptr));

    MediumInterface mediumInterface;
    std::vector<std::shared_ptr<Primitive>> prims;
    prims.push_back(PrimitiveHandle(new GeometricPrimitive(
        sphere, material, areaLight, mediumInterface)));
    PrimitiveHandle bvh(new BVHAccel(std::move(prims)));

    std::vector<std::shared_ptr<Light>> lights;
    lights.push_back(std::move(areaLight));

    scenes.push_back({bvh, lights, "Sphere, Kd = 0.25 Kr = 0.5, Le = 0.587", 1.0});
  }
#endif

    return scenes;
}

std::vector<std::pair<SamplerHandle, std::string>> GetSamplers(
    const Point2i &resolution) {
    std::vector<std::pair<SamplerHandle, std::string>> samplers;

    samplers.push_back(std::make_pair(new HaltonSampler(256, resolution), "Halton 256"));
    samplers.push_back(std::make_pair(new PaddedSobolSampler(256, RandomizeStrategy::Xor),
                                      "Padded Sobol 256"));
    samplers.push_back(
        std::make_pair(new SobolSampler(256, resolution, RandomizeStrategy::None),
                       "Sobol 256 Not Randomized"));
    samplers.push_back(std::make_pair(
        new SobolSampler(256, resolution, RandomizeStrategy::CranleyPatterson),
        "Sobol 256 Cranley Patterson Randomization"));
    samplers.push_back(
        std::make_pair(new SobolSampler(256, resolution, RandomizeStrategy::Xor),
                       "Sobol 256 XOR Scramble"));
    samplers.push_back(
        std::make_pair(new SobolSampler(256, resolution, RandomizeStrategy::Owen),
                       "Sobol 256 Owen Scramble"));
    samplers.push_back(std::make_pair(new RandomSampler(256), "Random 256"));
    samplers.push_back(
        std::make_pair(new StratifiedSampler(16, 16, true), "Stratified 16x16"));
    samplers.push_back(std::make_pair(new PMJ02BNSampler(256), "PMJ02bn 256"));

    return samplers;
}

std::vector<TestIntegrator> GetIntegrators() {
    std::vector<TestIntegrator> integrators;

    Point2i resolution(10, 10);
    static Transform id;
    AnimatedTransform identity(id, 0, id, 1);
    for (const auto &scene : GetScenes()) {
        // Path tracing integrators
        for (auto &sampler : GetSamplers(resolution)) {
            FilterHandle filter = new BoxFilter(Vector2f(0.5, 0.5));
            FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution),
                                  filter, 1., PixelSensor::CreateDefault(), inTestDir("test.exr"));
            RGBFilm *film = new RGBFilm(fp, RGBColorSpace::sRGB);
            CameraBaseParameters cbp(CameraTransform(identity), film, nullptr, {}, nullptr);
            PerspectiveCamera *camera = new PerspectiveCamera(cbp, 45,
                Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 10.);

            const FilmHandle filmp = camera->GetFilm();
            Integrator *integrator = new PathIntegrator(8, camera, sampler.first,
                                                        scene.aggregate, scene.lights);
            integrators.push_back({integrator, filmp,
                                   "Path, depth 8, Perspective, " + sampler.second +
                                       ", " + scene.description,
                                   scene});
        }

        for (auto &sampler : GetSamplers(resolution)) {
            FilterHandle filter = new BoxFilter(Vector2f(0.5, 0.5));
            FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution),
                                  filter, 1., PixelSensor::CreateDefault(), inTestDir("test.exr"));
            RGBFilm *film = new RGBFilm(fp, RGBColorSpace::sRGB);
            CameraBaseParameters cbp(CameraTransform(identity), film, nullptr, {}, nullptr);
            OrthographicCamera *camera = new OrthographicCamera(cbp,
                Bounds2f(Point2f(-.1, -.1), Point2f(.1, .1)), 0., 10.);
            const FilmHandle filmp = camera->GetFilm();

            Integrator *integrator = new PathIntegrator(8, camera, sampler.first,
                                                        scene.aggregate, scene.lights);
            integrators.push_back(
                {integrator, filmp,
                 "Path, depth 8, Ortho, " + sampler.second + ", " + scene.description,
                 scene});
        }

        // Volume path tracing integrators
        for (auto &sampler : GetSamplers(resolution)) {
            FilterHandle filter = new BoxFilter(Vector2f(0.5, 0.5));
            FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution),
                                  filter, 1., PixelSensor::CreateDefault(), inTestDir("test.exr"));
            RGBFilm *film = new RGBFilm(fp, RGBColorSpace::sRGB);
            CameraBaseParameters cbp(CameraTransform(identity), film, nullptr, {}, nullptr);
            PerspectiveCamera *camera = new PerspectiveCamera(cbp, 45,
                Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 10.);
            const FilmHandle filmp = camera->GetFilm();

            Integrator *integrator = new VolPathIntegrator(8, camera, sampler.first,
                                                           scene.aggregate, scene.lights);
            integrators.push_back({integrator, filmp,
                                   "VolPath, depth 8, Perspective, " + sampler.second +
                                       ", " + scene.description,
                                   scene});
        }
        for (auto &sampler : GetSamplers(resolution)) {
            FilterHandle filter = new BoxFilter(Vector2f(0.5, 0.5));
            FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution),
                                  filter, 1., PixelSensor::CreateDefault(), inTestDir("test.exr"));
            RGBFilm *film = new RGBFilm(fp, RGBColorSpace::sRGB);
            CameraBaseParameters cbp(CameraTransform(identity), film, nullptr, {}, nullptr);
            OrthographicCamera *camera = new OrthographicCamera(cbp,
                Bounds2f(Point2f(-.1, -.1), Point2f(.1, .1)), 0., 10.);
            const FilmHandle filmp = camera->GetFilm();

            Integrator *integrator = new VolPathIntegrator(8, camera, sampler.first,
                                                           scene.aggregate, scene.lights);
            integrators.push_back(
                {integrator, filmp,
                 "VolPath, depth 8, Ortho, " + sampler.second + ", " + scene.description,
                 scene});
        }

        // Simple path (perspective only, still sample light and BSDFs). Yolo
        for (auto &sampler : GetSamplers(resolution)) {
            FilterHandle filter = new BoxFilter(Vector2f(0.5, 0.5));
            FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution),
                                  filter, 1., PixelSensor::CreateDefault(), inTestDir("test.exr"));
            RGBFilm *film = new RGBFilm(fp, RGBColorSpace::sRGB);
            CameraBaseParameters cbp(CameraTransform(identity), film, nullptr, {}, nullptr);
            PerspectiveCamera *camera = new PerspectiveCamera(cbp, 45,
                Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 10.);

            const FilmHandle filmp = camera->GetFilm();
            Integrator *integrator = new SimplePathIntegrator(
                8, true, true, camera, sampler.first, scene.aggregate, scene.lights);
            integrators.push_back({integrator, filmp,
                                   "SimplePath, depth 8, Perspective, " + sampler.second +
                                       ", " + scene.description,
                                   scene});
        }

        // BDPT
        for (auto &sampler : GetSamplers(resolution)) {
            FilterHandle filter = new BoxFilter(Vector2f(0.5, 0.5));
            FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution),
                                  filter, 1., PixelSensor::CreateDefault(), inTestDir("test.exr"));
            RGBFilm *film = new RGBFilm(fp, RGBColorSpace::sRGB);
            CameraBaseParameters cbp(CameraTransform(identity), film, nullptr, {}, nullptr);
            PerspectiveCamera *camera = new PerspectiveCamera(cbp, 45,
                Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 10.);
            const FilmHandle filmp = camera->GetFilm();

            Integrator *integrator =
                new BDPTIntegrator(camera, sampler.first, scene.aggregate, scene.lights,
                                   6, false, false, "power", false);
            integrators.push_back({integrator, filmp,
                                   "BDPT, depth 8, Perspective, " + sampler.second +
                                       ", " + scene.description,
                                   scene});
        }

        // MLT
        {
            FilterHandle filter = new BoxFilter(Vector2f(0.5, 0.5));
            FilmBaseParameters fp(resolution, Bounds2i(Point2i(0, 0), resolution),
                                  filter, 1., PixelSensor::CreateDefault(), inTestDir("test.exr"));
            RGBFilm *film = new RGBFilm(fp, RGBColorSpace::sRGB);
            CameraBaseParameters cbp(CameraTransform(identity), film, nullptr, {}, nullptr);
            PerspectiveCamera *camera = new PerspectiveCamera(cbp, 45,
                Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 10.);
            const FilmHandle filmp = camera->GetFilm();

            Integrator *integrator =
                new MLTIntegrator(camera, scene.aggregate, scene.lights, 8 /* depth */,
                                  100000 /* n bootstrap */, 1000 /* nchains */,
                                  1024 /* mutations per pixel */, 0.01 /* sigma */,
                                  0.3 /* large step prob */, false /* regularize */);
            integrators.push_back({integrator, filmp,
                                   "MLT, depth 8, Perspective, " + scene.description,
                                   scene});
        }
    }

    return integrators;
}

struct RenderTest : public testing::TestWithParam<TestIntegrator> {};

TEST_P(RenderTest, RadianceMatches) {
    const TestIntegrator &tr = GetParam();
    tr.integrator->Render();
    CheckSceneAverage(inTestDir("test.exr"), tr.scene.expected);
    // The SpatialLightSampler class keeps a per-thread cache that
    // must be cleared out between test runs. In turn, this means that we
    // must delete the Integrator here in order to make sure that its
    // destructor runs. (This is ugly and should be fixed in a better way.)
    delete tr.integrator;

    EXPECT_EQ(0, remove(inTestDir("test.exr").c_str()));
}

INSTANTIATE_TEST_CASE_P(AnalyticTestScenes, RenderTest,
                        testing::ValuesIn(GetIntegrators()));
