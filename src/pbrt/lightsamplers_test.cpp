// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/shapes.h>
#include <pbrt/util/math.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace pbrt;

TEST(BVHLightSampling, OneSpot) {
    Transform id;
    std::vector<LightHandle> lights;
    ConstantSpectrum one(1.f);
    lights.push_back(new SpotLight(id, MediumInterface(), &one, 1.f /* scale */,
                                   45.f /* total width */,
                                   44.f /* falloff start */, Allocator()));
    BVHLightSampler distrib(lights, Allocator());

    RNG rng;
    for (int i = 0; i < 100; ++i) {
        // Random point in [-5, 5]
        Point3f p(Lerp(rng.Uniform<Float>(), -5, 5), Lerp(rng.Uniform<Float>(), -5, 5),
                  Lerp(rng.Uniform<Float>(), -5, 5));

        Interaction in(p, 0., (MediumHandle) nullptr);
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        SampledWavelengths lambda = SampledWavelengths::SampleUniform(0.5);
        LightLiSample ls = lights[0].SampleLi(in, u, lambda);

        pstd::optional<SampledLight> sampledLight =
            distrib.Sample(in, rng.Uniform<Float>());

        if (!sampledLight) {
            EXPECT_FALSE((bool)ls);
            continue;
        } else
            EXPECT_TRUE((bool)ls);

        EXPECT_EQ(1, sampledLight->pdf);
        EXPECT_TRUE(sampledLight->light == lights[0]);
        EXPECT_TRUE((bool)ls.L) << ls.L << " @ " << p;
    }
}

// For a random collection of point lights, make sure that they're all sampled
// with an appropriate ratio of frequency to pdf.
TEST(BVHLightSampling, Point) {
    RNG rng;
    std::vector<LightHandle> lights;
    std::unordered_map<LightHandle, int, LightHandleHash> lightToIndex;
    ConstantSpectrum one(1.f);
    for (int i = 0; i < 33; ++i) {
        // Random point in [-5, 5]
        Vector3f p(Lerp(rng.Uniform<Float>(), -5, 5), Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5));
        lights.push_back(
            new PointLight(Translate(p), MediumInterface(), &one, 1.f, Allocator()));
        lightToIndex[lights.back()] = i;
    }
    BVHLightSampler distrib(lights, Allocator());

    for (int i = 0; i < 10; ++i) {
        // Don't get too close to the light bbox
        auto r = [&rng]() {
            return rng.Uniform<Float>() < .5 ? Lerp(rng.Uniform<Float>(), -15, -7)
                                             : Lerp(rng.Uniform<Float>(), 7, 16);
        };
        Point3f p(r(), r(), r());

        std::vector<Float> sumWt(lights.size(), 0.f);
        const int nSamples = 10000;
        for (Float u : Stratified1D(nSamples)) {
            Interaction intr(p, 0, (MediumHandle) nullptr);
            pstd::optional<SampledLight> sampledLight = distrib.Sample(intr, u);
            // Can assume this because it's all point lights
            ASSERT_TRUE((bool)sampledLight);

            EXPECT_GT(sampledLight->pdf, 0);
            sumWt[lightToIndex[sampledLight->light]] +=
                1 / (sampledLight->pdf * nSamples);

            EXPECT_FLOAT_EQ(sampledLight->pdf, distrib.PDF(intr, sampledLight->light));
        }

        for (int i = 0; i < lights.size(); ++i) {
            EXPECT_GE(sumWt[i], .98);
            EXPECT_LT(sumWt[i], 1.02);
        }
    }
}

// Similar to BVHLightSampling.Point, but vary light power
TEST(BVHLightSampling, PointVaryPower) {
    RNG rng(53251);
    std::vector<LightHandle> lights;
    std::vector<Float> lightPower;
    std::vector<std::unique_ptr<ConstantSpectrum>> lightSpectra;
    Float sumPower = 0;
    std::unordered_map<LightHandle, int, LightHandleHash> lightToIndex;
    for (int i = 0; i < 82; ++i) {
        // Random point in [-5, 5]
        Vector3f p(Lerp(rng.Uniform<Float>(), -5, 5), Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5));
        lightPower.push_back(rng.Uniform<Float>());
        lightSpectra.push_back(std::make_unique<ConstantSpectrum>(lightPower.back()));
        sumPower += lightPower.back();
        lights.push_back(new PointLight(Translate(p), MediumInterface(),
                                        lightSpectra.back().get(), 1.f, Allocator()));
        lightToIndex[lights.back()] = i;
    }
    BVHLightSampler distrib(lights, Allocator());

    for (int i = 0; i < 10; ++i) {
        // Don't get too close to the light bbox
        auto r = [&rng]() {
            return rng.Uniform<Float>() < .5 ? Lerp(rng.Uniform<Float>(), -15, -7)
                                             : Lerp(rng.Uniform<Float>(), 7, 16);
        };
        Point3f p(r(), r(), r());

        std::vector<Float> sumWt(lights.size(), 0.f);
        const int nSamples = 100000;
        for (Float u : Stratified1D(nSamples)) {
            Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
            pstd::optional<SampledLight> sampledLight = distrib.Sample(intr, u);
            // Again because it's all point lights...
            ASSERT_TRUE((bool)sampledLight);

            LightHandle light = sampledLight->light;
            Float pdf = sampledLight->pdf;
            EXPECT_GT(pdf, 0);
            sumWt[lightToIndex[light]] += 1 / (pdf * nSamples);

            EXPECT_LT(std::abs(distrib.PDF(intr, light) - pdf) / pdf, 1e-4);
        }

        for (int i = 0; i < lights.size(); ++i) {
            EXPECT_GE(sumWt[i], .95);
            EXPECT_LT(sumWt[i], 1.05);
        }
    }

    // Now, for very far away points (so d^2 is about the same for all
    // lights), make sure that sampling frequencies for each light are
    // basically proportional to their power
    for (int i = 0; i < 10; ++i) {
        // Don't get too close to the light bbox
        auto r = [&rng]() {
            return rng.Uniform<Float>() < .5 ? Lerp(rng.Uniform<Float>(), -15, -7)
                                             : Lerp(rng.Uniform<Float>(), 7, 16);
        };
        Point3f p(10000 * r(), 10000 * r(), 10000 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));

        std::vector<int> counts(lights.size(), 0);
        const int nSamples = 100000;
        for (Float u : Stratified1D(nSamples)) {
            pstd::optional<SampledLight> sampledLight = distrib.Sample(intr, u);
            ASSERT_TRUE((bool)sampledLight);
            LightHandle light = sampledLight->light;
            Float pdf = sampledLight->pdf;
            EXPECT_GT(pdf, 0);
            ++counts[lightToIndex[light]];

            EXPECT_FLOAT_EQ(pdf, distrib.PDF(intr, light));
        }

        for (int i = 0; i < lights.size(); ++i) {
            Float expected = nSamples * lightPower[i] / sumPower;
            EXPECT_GE(counts[i], .97 * expected);
            EXPECT_LT(counts[i], 1.03 * expected);
        }
    }
}

TEST(BVHLightSampling, OneTri) {
    RNG rng(5251);
    Transform id;
    std::vector<int> indices{0, 1, 2};
    // Light is illuminating points with z > 0
    std::vector<Point3f> p{Point3f(-1, -1, 0), Point3f(1, -1, 0), Point3f(0, 1, 0)};
    TriangleMesh mesh(id, false /* rev orientation */, indices, p, {}, {}, {}, {});
    auto tris = Triangle::CreateTriangles(&mesh, Allocator());

    ASSERT_EQ(1, tris.size());
    std::vector<LightHandle> lights;
    ConstantSpectrum one(1.f);
    lights.push_back(new DiffuseAreaLight(id, MediumInterface(), &one, 1.f, tris[0],
                                          Image(), nullptr, false /* two sided */,
                                          Allocator()));

    BVHLightSampler distrib(lights, Allocator());

    for (int i = 0; i < 10; ++i) {
        // Random point in [-5, 5]
        Point3f p(Lerp(rng.Uniform<Float>(), -5, 5), Lerp(rng.Uniform<Float>(), -5, 5),
                  Lerp(rng.Uniform<Float>(), -5, 5));

        Interaction in(p, 0., (MediumHandle) nullptr);
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        SampledWavelengths lambda = SampledWavelengths::SampleUniform(0.5);
        LightLiSample ls = lights[0].SampleLi(in, u, lambda);

        pstd::optional<SampledLight> sampledLight =
            distrib.Sample(in, rng.Uniform<Float>());

        // Note that the converse (sampledLight ->
        // ls) isn't always true since the light importance
        // metric is conservative.
        if (!sampledLight) {
            EXPECT_FALSE((bool)ls);
        }
    }
}

static std::tuple<std::vector<LightHandle>, std::vector<ShapeHandle>> randomLights(
    int n, Allocator alloc) {
    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> allTris;
    RNG rng(6502);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    Transform id;
    for (int i = 0; i < n; ++i) {
        // Triangle
        {
            std::vector<int> indices{0, 1, 2};
            std::vector<Point3f> p{Point3f(r(), r(), r()), Point3f(r(), r(), r()),
                                   Point3f(r(), r(), r())};
            // leaks...
            TriangleMesh *mesh = new TriangleMesh(id, false /* rev orientation */,
                                                  indices, p, {}, {}, {}, {});
            auto tris = Triangle::CreateTriangles(mesh, Allocator());
            CHECK_EQ(1, tris.size());  // EXPECT doesn't work since this is in a
                                       // function :-p
            static Transform id;
            lights.push_back(alloc.new_object<DiffuseAreaLight>(
                id, MediumInterface(), alloc.new_object<ConstantSpectrum>(r()), 1.f,
                tris[0], Image(), nullptr, false /* two sided */, Allocator()));
            allTris.push_back(tris[0]);
        }

        // Random point light
        {
            Vector3f p(Lerp(rng.Uniform<Float>(), -5, 5),
                       Lerp(rng.Uniform<Float>(), -5, 5),
                       Lerp(rng.Uniform<Float>(), -5, 5));
            lights.push_back(new PointLight(Translate(p), MediumInterface(),
                                            alloc.new_object<ConstantSpectrum>(r()),
                                            1.f, Allocator()));
        }
    }

    return {std::move(lights), std::move(allTris)};
}

TEST(BVHLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    std::tie(lights, tris) = randomLights(20, Allocator());

    BVHLightSampler distrib(lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Float u = rng.Uniform<Float>();
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        pstd::optional<SampledLight> sampledLight = distrib.Sample(intr, u);
        if (sampledLight)
            // It's actually legit to sometimes get no lights; as the bounds
            // tighten up as we get deeper in the tree, it may turn out that
            // the path we followed didn't have any lights after all.
            EXPECT_FLOAT_EQ(sampledLight->pdf, distrib.PDF(intr, sampledLight->light));
    }
}

TEST(ExhaustiveLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    std::tie(lights, tris) = randomLights(20, Allocator());

    ExhaustiveLightSampler distrib(lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        pstd::optional<SampledLight> sampledLight =
            distrib.Sample(intr, rng.Uniform<Float>());
        ASSERT_TRUE((bool)sampledLight) << i << " - " << p;
        EXPECT_FLOAT_EQ(sampledLight->pdf, distrib.PDF(intr, sampledLight->light));
    }
}

TEST(UniformLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    std::tie(lights, tris) = randomLights(20, Allocator());

    UniformLightSampler distrib(lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        pstd::optional<SampledLight> sampledLight =
            distrib.Sample(intr, rng.Uniform<Float>());
        ASSERT_TRUE((bool)sampledLight) << i << " - " << p;
        EXPECT_FLOAT_EQ(sampledLight->pdf, distrib.PDF(intr, sampledLight->light));
    }
}

TEST(PowerLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    std::tie(lights, tris) = randomLights(20, Allocator());

    PowerLightSampler distrib(lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        pstd::optional<SampledLight> sampledLight =
            distrib.Sample(intr, rng.Uniform<Float>());
        ASSERT_TRUE((bool)sampledLight) << i << " - " << p;
        EXPECT_FLOAT_EQ(sampledLight->pdf, distrib.PDF(intr, sampledLight->light));
    }
}
