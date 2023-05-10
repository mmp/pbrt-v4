// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/media.h>

#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/textures.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/stats.h>

#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>

#include <algorithm>
#include <cmath>

namespace pbrt {

std::string MediumInterface::ToString() const {
    return StringPrintf("[ MediumInterface inside: %s outside: %s ]",
                        inside ? inside.ToString().c_str() : "(nullptr)",
                        outside ? outside.ToString().c_str() : "(nullptr)");
}

std::string PhaseFunction::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

std::string RayMajorantSegment::ToString() const {
    return StringPrintf("[ RayMajorantSegment tMin: %f tMax: %f sigma_maj: %s ]", tMin,
                        tMax, sigma_maj);
}

std::string RayMajorantIterator::ToString() const {
    auto tostr = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(tostr);
}

std::string HomogeneousMajorantIterator::ToString() const {
    return StringPrintf("[ HomogeneousMajorantIterator seg: %s called: %s ]", seg,
                        called);
}

std::string DDAMajorantIterator::ToString() const {
    return StringPrintf("[ DDAMajorantIterator tMin: %f tMax: %f sigma_t: %s "
                        "nextCrossingT: [ %f %f %f ] deltaT: [ %f %f %f ] "
                        "step: [ %d %d %d ] voxelLimit: [ %d %d %d ] voxel: [ %d %d %d ] "
                        "grid: %p ]",
                        tMin, tMax, sigma_t, nextCrossingT[0], nextCrossingT[1],
                        nextCrossingT[2], deltaT[0], deltaT[1], deltaT[2], step[0],
                        step[1], step[2], voxelLimit[0], voxelLimit[1], voxelLimit[2],
                        voxel[0], voxel[1], voxel[2], grid);
}

// HenyeyGreenstein Method Definitions
std::string HGPhaseFunction::ToString() const {
    return StringPrintf("[ HGPhaseFunction g: %f ]", g);
}

struct MeasuredSS {
    const char *name;
    RGB sigma_prime_s, sigma_a;  // mm^-1
};

bool GetMediumScatteringProperties(const std::string &name, Spectrum *sigma_a,
                                   Spectrum *sigma_s, Allocator alloc) {
    static MeasuredSS SubsurfaceParameterTable[] = {
        // From "A Practical Model for Subsurface Light Transport"
        // Jensen, Marschner, Levoy, Hanrahan
        // Proc SIGGRAPH 2001
        // clang-format off
        {"Apple", RGB(2.29, 2.39, 1.97), RGB(0.0030, 0.0034, 0.046)},
        {"Chicken1", RGB(0.15, 0.21, 0.38), RGB(0.015, 0.077, 0.19)},
        {"Chicken2", RGB(0.19, 0.25, 0.32), RGB(0.018, 0.088, 0.20)},
        {"Cream", RGB(7.38, 5.47, 3.15), RGB(0.0002, 0.0028, 0.0163)},
        {"Ketchup", RGB(0.18, 0.07, 0.03), RGB(0.061, 0.97, 1.45)},
        {"Marble", RGB(2.19, 2.62, 3.00), RGB(0.0021, 0.0041, 0.0071)},
        {"Potato", RGB(0.68, 0.70, 0.55), RGB(0.0024, 0.0090, 0.12)},
        {"Skimmilk", RGB(0.70, 1.22, 1.90), RGB(0.0014, 0.0025, 0.0142)},
        {"Skin1", RGB(0.74, 0.88, 1.01), RGB(0.032, 0.17, 0.48)},
        {"Skin2", RGB(1.09, 1.59, 1.79), RGB(0.013, 0.070, 0.145)},
        {"Spectralon", RGB(11.6, 20.4, 14.9), RGB(0.00, 0.00, 0.00)},
        {"Wholemilk", RGB(2.55, 3.21, 3.77), RGB(0.0011, 0.0024, 0.014)},

        // From "Acquiring Scattering Properties of Participating Media by
        // Dilution",
        // Narasimhan, Gupta, Donner, Ramamoorthi, Nayar, Jensen
        // Proc SIGGRAPH 2006
        {"Lowfat Milk", RGB(0.89187, 1.5136, 2.532), RGB(0.002875, 0.00575, 0.0115)},
        {"Reduced Milk", RGB(2.4858, 3.1669, 4.5214), RGB(0.0025556, 0.0051111, 0.012778)},
        {"Regular Milk", RGB(4.5513, 5.8294, 7.136), RGB(0.0015333, 0.0046, 0.019933)},
        {"Espresso", RGB(0.72378, 0.84557, 1.0247), RGB(4.7984, 6.5751, 8.8493)},
        {"Mint Mocha Coffee", RGB(0.31602, 0.38538, 0.48131), RGB(3.772, 5.8228, 7.82)},
        {"Lowfat Soy Milk", RGB(0.30576, 0.34233, 0.61664), RGB(0.0014375, 0.0071875, 0.035937)},
        {"Regular Soy Milk", RGB(0.59223, 0.73866, 1.4693), RGB(0.0019167, 0.0095833, 0.065167)},
        {"Lowfat Chocolate Milk", RGB(0.64925, 0.83916, 1.1057), RGB(0.0115, 0.0368, 0.1564)},
        {"Regular Chocolate Milk", RGB(1.4585, 2.1289, 2.9527), RGB(0.010063, 0.043125, 0.14375)},
        {"Coke", RGB(8.9053e-05, 8.372e-05, 0), RGB(0.10014, 0.16503, 0.2468)},
        {"Pepsi", RGB(6.1697e-05, 4.2564e-05, 0), RGB(0.091641, 0.14158, 0.20729)},
        {"Sprite", RGB(6.0306e-06, 6.4139e-06, 6.5504e-06), RGB(0.001886, 0.0018308, 0.0020025)},
        {"Gatorade", RGB(0.0024574, 0.003007, 0.0037325), RGB(0.024794, 0.019289, 0.008878)},
        {"Chardonnay", RGB(1.7982e-05, 1.3758e-05, 1.2023e-05), RGB(0.010782, 0.011855, 0.023997)},
        {"White Zinfandel", RGB(1.7501e-05, 1.9069e-05, 1.288e-05), RGB(0.012072, 0.016184, 0.019843)},
        {"Merlot", RGB(2.1129e-05, 0, 0), RGB(0.11632, 0.25191, 0.29434)},
        {"Budweiser Beer", RGB(2.4356e-05, 2.4079e-05, 1.0564e-05), RGB(0.011492, 0.024911, 0.057786)},
        {"Coors Light Beer", RGB(5.0922e-05, 4.301e-05, 0), RGB(0.006164, 0.013984, 0.034983)},
        {"Clorox", RGB(0.0024035, 0.0031373, 0.003991), RGB(0.0033542, 0.014892, 0.026297)},
        {"Apple Juice", RGB(0.00013612, 0.00015836, 0.000227), RGB(0.012957, 0.023741, 0.052184)},
        {"Cranberry Juice", RGB(0.00010402, 0.00011646, 7.8139e-05), RGB(0.039437, 0.094223, 0.12426)},
        {"Grape Juice", RGB(5.382e-05, 0, 0), RGB(0.10404, 0.23958, 0.29325)},
        {"Ruby Grapefruit Juice", RGB(0.011002, 0.010927, 0.011036), RGB(0.085867, 0.18314, 0.25262)},
        {"White Grapefruit Juice", RGB(0.22826, 0.23998, 0.32748), RGB(0.0138, 0.018831, 0.056781)},
        {"Shampoo", RGB(0.0007176, 0.0008303, 0.0009016), RGB(0.014107, 0.045693, 0.061717)},
        {"Strawberry Shampoo", RGB(0.00015671, 0.00015947, 1.518e-05), RGB(0.01449, 0.05796, 0.075823)},
        {"Head & Shoulders Shampoo", RGB(0.023805, 0.028804, 0.034306), RGB(0.084621, 0.15688, 0.20365)},
        {"Lemon Tea Powder", RGB(0.040224, 0.045264, 0.051081), RGB(2.4288, 4.5757, 7.2127)},
        {"Orange Powder", RGB(0.00015617, 0.00017482, 0.0001762), RGB(0.001449, 0.003441, 0.007863)},
        {"Pink Lemonade Powder", RGB(0.00012103, 0.00013073, 0.00012528), RGB(0.001165, 0.002366, 0.003195)},
        {"Cappuccino Powder", RGB(1.8436, 2.5851, 2.1662), RGB(35.844, 49.547, 61.084)},
        {"Salt Powder", RGB(0.027333, 0.032451, 0.031979), RGB(0.28415, 0.3257, 0.34148)},
        {"Sugar Powder", RGB(0.00022272, 0.00025513, 0.000271), RGB(0.012638, 0.031051, 0.050124)},
        {"Suisse Mocha Powder", RGB(2.7979, 3.5452, 4.3365), RGB(17.502, 27.004, 35.433)},
        {"Pacific Ocean Surface Water", RGB(0.0001764, 0.00032095, 0.00019617), RGB(0.031845, 0.031324, 0.030147)}
        // clang-format on
    };

    for (MeasuredSS &mss : SubsurfaceParameterTable) {
        if (name == mss.name) {
            *sigma_a =
                alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB, mss.sigma_a);
            *sigma_s = alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB,
                                                              mss.sigma_prime_s);
            return true;
        }
    }
    return false;
}

bool Medium::IsEmissive() const {
    auto is = [&](auto ptr) { return ptr->IsEmissive(); };
    return DispatchCPU(is);
}

std::string Medium::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

// HomogeneousMedium Method Definitions
HomogeneousMedium *HomogeneousMedium::Create(const ParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc) {
    Spectrum sig_a = nullptr, sig_s = nullptr;
    std::string preset = parameters.GetOneString("preset", "");
    if (!preset.empty()) {
        if (!GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc))
            Warning(loc, "Material preset \"%s\" not found.", preset);
    }
    if (!sig_a) {
        sig_a =
            parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::Unbounded, alloc);
        if (!sig_a)
            sig_a = alloc.new_object<ConstantSpectrum>(1.f);
    }
    if (!sig_s) {
        sig_s =
            parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::Unbounded, alloc);
        if (!sig_s)
            sig_s = alloc.new_object<ConstantSpectrum>(1.f);
    }

    Spectrum Le =
        parameters.GetOneSpectrum("Le", nullptr, SpectrumType::Illuminant, alloc);
    Float LeScale = parameters.GetOneFloat("Lescale", 1.f);
    if (!Le || Le.MaxValue() == 0)
        Le = alloc.new_object<ConstantSpectrum>(0.f);
    else
        LeScale /= SpectrumToPhotometric(Le);

    Float sigmaScale = parameters.GetOneFloat("scale", 1.f);
    Float g = parameters.GetOneFloat("g", 0.0f);

    return alloc.new_object<HomogeneousMedium>(sig_a, sig_s, sigmaScale, Le, LeScale, g,
                                               alloc);
}

std::string HomogeneousMedium::ToString() const {
    return StringPrintf(
        "[ Homogeneous medium sigma_a_spec: %s sigma_s_spec: %s Le_spec: %s phase: %s ]",
        sigma_a_spec, sigma_s_spec, Le_spec, phase);
}

STAT_MEMORY_COUNTER("Memory/Volume grids", volumeGridBytes);

// GridMedium Method Definitions
GridMedium::GridMedium(const Bounds3f &bounds, const Transform &renderFromMedium,
                       Spectrum sigma_a, Spectrum sigma_s, Float sigmaScale, Float g,
                       SampledGrid<Float> d,
                       pstd::optional<SampledGrid<Float>> temperature,
                       Float temperatureScale, Float temperatureOffset,
                       Spectrum Le, SampledGrid<Float> LeGrid, Allocator alloc)
    : bounds(bounds),
      renderFromMedium(renderFromMedium),
      sigma_a_spec(sigma_a, alloc),
      sigma_s_spec(sigma_s, alloc),
      densityGrid(std::move(d)),
      phase(g),
      temperatureGrid(std::move(temperature)),
      temperatureScale(temperatureScale),
      temperatureOffset(temperatureOffset),
      Le_spec(Le, alloc),
      LeScale(std::move(LeGrid)),
      majorantGrid(bounds, {16, 16, 16}, alloc) {
    sigma_a_spec.Scale(sigmaScale);
    sigma_s_spec.Scale(sigmaScale);

    volumeGridBytes += LeScale.BytesAllocated();
    volumeGridBytes += densityGrid.BytesAllocated();
    if (temperatureGrid)
        volumeGridBytes += temperatureGrid->BytesAllocated();

    isEmissive = temperatureGrid ? true : (Le_spec.MaxValue() > 0);

    // Initialize _majorantGrid_ for _GridMedium_
    for (int z = 0; z < majorantGrid.res.z; ++z)
        for (int y = 0; y < majorantGrid.res.y; ++y)
            for (int x = 0; x < majorantGrid.res.x; ++x) {
                Bounds3f bounds = majorantGrid.VoxelBounds(x, y, z);
                majorantGrid.Set(x, y, z, densityGrid.MaxValue(bounds));
            }
}

GridMedium *GridMedium::Create(const ParameterDictionary &parameters,
                               const Transform &renderFromMedium, const FileLoc *loc,
                               Allocator alloc) {
    std::vector<Float> density = parameters.GetFloatArray("density");
    std::vector<Float> temperature = parameters.GetFloatArray("temperature");

    size_t nDensity;
    if (density.empty())
        ErrorExit(loc, "No \"density\" value provided for grid medium.");
    nDensity = density.size();

    if (!temperature.empty())
        if (nDensity != temperature.size())
            ErrorExit(loc,
                      "Different number of samples (%d vs %d) provided for "
                      "\"density\" and \"temperature\".",
                      nDensity, temperature.size());

    int nx = parameters.GetOneInt("nx", 1);
    int ny = parameters.GetOneInt("ny", 1);
    int nz = parameters.GetOneInt("nz", 1);
    if (nDensity != nx * ny * nz)
        ErrorExit(loc, "Grid medium has %d density values; expected nx*ny*nz = %d",
                  nDensity, nx * ny * nz);

    // Create Density Grid
    SampledGrid<Float> densityGrid = SampledGrid<Float>(density, nx, ny, nz, alloc);

    pstd::optional<SampledGrid<Float>> temperatureGrid;
    if (temperature.size())
        temperatureGrid = SampledGrid<Float>(temperature, nx, ny, nz, alloc);

    Spectrum Le =
        parameters.GetOneSpectrum("Le", nullptr, SpectrumType::Illuminant, alloc);

    if (Le && !temperature.empty())
        ErrorExit(loc, "Both \"Le\" and \"temperature\" values were provided.");

    Float LeNorm = 1;
    if (!Le || Le.MaxValue() == 0)
        Le = alloc.new_object<ConstantSpectrum>(0.f);
    else
        LeNorm = 1 / SpectrumToPhotometric(Le);

    SampledGrid<Float> LeGrid(alloc);
    std::vector<Float> LeScale = parameters.GetFloatArray("Lescale");

    if (LeScale.empty())
        LeGrid = SampledGrid<Float>({LeNorm}, 1, 1, 1, alloc);
    else {
        if (LeScale.size() != nx * ny * nz)
            ErrorExit("Expected %d x %d %d = %d values for \"Lescale\" but were "
                      "given %d.",
                      nx, ny, nz, nx * ny * nz, LeScale.size());
        for (int i = 0; i < nx * ny * nz; ++i)
            LeScale[i] *= LeNorm;
        LeGrid = SampledGrid<Float>(LeScale, nx, ny, nz, alloc);
    }

    Point3f p0 = parameters.GetOnePoint3f("p0", Point3f(0.f, 0.f, 0.f));
    Point3f p1 = parameters.GetOnePoint3f("p1", Point3f(1.f, 1.f, 1.f));

    Float g = parameters.GetOneFloat("g", 0.);
    Spectrum sigma_a =
        parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::Unbounded, alloc);
    if (!sigma_a)
        sigma_a = alloc.new_object<ConstantSpectrum>(1.f);
    Spectrum sigma_s =
        parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::Unbounded, alloc);
    if (!sigma_s)
        sigma_s = alloc.new_object<ConstantSpectrum>(1.f);
    Float sigmaScale = parameters.GetOneFloat("scale", 1.f);

    Float temperatureOffset = parameters.GetOneFloat("temperatureoffset",
                                                     parameters.GetOneFloat("temperaturecutoff", 0.f));
    Float temperatureScale = parameters.GetOneFloat("temperaturescale", 1.f);

    return alloc.new_object<GridMedium>(
        Bounds3f(p0, p1), renderFromMedium, sigma_a, sigma_s, sigmaScale, g,
        std::move(densityGrid), std::move(temperatureGrid), temperatureScale,
        temperatureOffset, Le, std::move(LeGrid), alloc);
}

std::string GridMedium::ToString() const {
    return StringPrintf("[ GridMedium bounds: %s renderFromMedium: %s phase: %s "
                        "LeScale: %f (grids elided) ]",
                        bounds, renderFromMedium, phase, LeScale);
}

// RGBGridMedium Method Definitions
RGBGridMedium::RGBGridMedium(const Bounds3f &bounds, const Transform &renderFromMedium,
                             Float g,
                             pstd::optional<SampledGrid<RGBUnboundedSpectrum>> rgbA,
                             pstd::optional<SampledGrid<RGBUnboundedSpectrum>> rgbS,
                             Float sigmaScale,
                             pstd::optional<SampledGrid<RGBIlluminantSpectrum>> rgbLe,
                             Float LeScale, Allocator alloc)
    : bounds(bounds),
      renderFromMedium(renderFromMedium),
      phase(g),
      sigma_aGrid(std::move(rgbA)),
      sigma_sGrid(std::move(rgbS)),
      sigmaScale(sigmaScale),
      majorantGrid(bounds, {16, 16, 16}, alloc),
      LeGrid(std::move(rgbLe)),
      LeScale(LeScale) {
    if (LeGrid)
        CHECK(sigma_aGrid);
    if (sigma_aGrid)
        volumeGridBytes += sigma_aGrid->BytesAllocated();
    if (sigma_sGrid)
        volumeGridBytes += sigma_sGrid->BytesAllocated();
    if (LeGrid)
        volumeGridBytes += LeGrid->BytesAllocated();

    // Initialize _majorantGrid_ for _RGBGridMedium_
    for (int z = 0; z < majorantGrid.res.z; ++z)
        for (int y = 0; y < majorantGrid.res.y; ++y)
            for (int x = 0; x < majorantGrid.res.x; ++x) {
                Bounds3f bounds = majorantGrid.VoxelBounds(x, y, z);
                // Initialize _majorantGrid_ voxel for RGB $\sigmaa$ and $\sigmas$
                auto max = [] PBRT_CPU_GPU(RGBUnboundedSpectrum s) {
                    return s.MaxValue();
                };
                Float maxSigma_t =
                    (sigma_aGrid ? sigma_aGrid->MaxValue(bounds, max) : 1) +
                    (sigma_sGrid ? sigma_sGrid->MaxValue(bounds, max) : 1);
                majorantGrid.Set(x, y, z, sigmaScale * maxSigma_t);
            }
}

RGBGridMedium *RGBGridMedium::Create(const ParameterDictionary &parameters,
                                     const Transform &renderFromMedium,
                                     const FileLoc *loc, Allocator alloc) {
    std::vector<RGB> sigma_a = parameters.GetRGBArray("sigma_a");
    std::vector<RGB> sigma_s = parameters.GetRGBArray("sigma_s");
    std::vector<RGB> Le = parameters.GetRGBArray("Le");

    if (sigma_a.empty() && sigma_s.empty())
        ErrorExit(loc,
                  "RGB grid requires \"sigma_a\" and/or \"sigma_s\" parameter values.");

    size_t nDensity;
    if (!sigma_a.empty()) {
        nDensity = sigma_a.size();
        if (!sigma_s.empty() && nDensity != sigma_s.size())
            ErrorExit(loc,
                      "Different number of samples (%d vs %d) provided for \"sigma_a\" "
                      "and \"sigma_s\".",
                      nDensity, sigma_s.size());
    } else
        nDensity = sigma_s.size();

    if (!Le.empty() && sigma_a.empty())
        ErrorExit(loc, "RGB grid requires \"sigma_a\" if \"Le\" value provided.");

    if (!Le.empty() && nDensity != Le.size())
        ErrorExit("Expected %d values for \"Le\" parameter but were given %d.", nDensity,
                  Le.size());

    int nx = parameters.GetOneInt("nx", 1);
    int ny = parameters.GetOneInt("ny", 1);
    int nz = parameters.GetOneInt("nz", 1);
    if (nDensity != nx * ny * nz)
        ErrorExit(loc, "RGB grid medium has %d density values; expected nx*ny*nz = %d",
                  nDensity, nx * ny * nz);

    pstd::optional<SampledGrid<RGBUnboundedSpectrum>> sigma_aGrid, sigma_sGrid;
    pstd::optional<SampledGrid<RGBIlluminantSpectrum>> LeGrid;

    if (!sigma_a.empty()) {
        const RGBColorSpace *colorSpace = parameters.ColorSpace();
        std::vector<RGBUnboundedSpectrum> rgbSpectrumDensity;
        for (RGB rgb : sigma_a)
            rgbSpectrumDensity.push_back(RGBUnboundedSpectrum(*colorSpace, rgb));
        sigma_aGrid =
            SampledGrid<RGBUnboundedSpectrum>(rgbSpectrumDensity, nx, ny, nz, alloc);
    }
    if (!sigma_s.empty()) {
        const RGBColorSpace *colorSpace = parameters.ColorSpace();
        std::vector<RGBUnboundedSpectrum> rgbSpectrumDensity;
        for (RGB rgb : sigma_s)
            rgbSpectrumDensity.push_back(RGBUnboundedSpectrum(*colorSpace, rgb));
        sigma_sGrid =
            SampledGrid<RGBUnboundedSpectrum>(rgbSpectrumDensity, nx, ny, nz, alloc);
    }
    if (!Le.empty()) {
        const RGBColorSpace *colorSpace = parameters.ColorSpace();
        std::vector<RGBIlluminantSpectrum> rgbSpectrumDensity;
        for (RGB rgb : Le)
            rgbSpectrumDensity.push_back(RGBIlluminantSpectrum(*colorSpace, rgb));
        LeGrid =
            SampledGrid<RGBIlluminantSpectrum>(rgbSpectrumDensity, nx, ny, nz, alloc);
    }

    Point3f p0 = parameters.GetOnePoint3f("p0", Point3f(0.f, 0.f, 0.f));
    Point3f p1 = parameters.GetOnePoint3f("p1", Point3f(1.f, 1.f, 1.f));
    Float LeScale = parameters.GetOneFloat("Lescale", 1.f);
    Float g = parameters.GetOneFloat("g", 0.f);
    Float sigmaScale = parameters.GetOneFloat("scale", 1.f);

    return alloc.new_object<RGBGridMedium>(Bounds3f(p0, p1), renderFromMedium, g,
                                           std::move(sigma_aGrid), std::move(sigma_sGrid),
                                           sigmaScale, std::move(LeGrid), LeScale, alloc);
}

std::string RGBGridMedium::ToString() const {
    return StringPrintf("[ RGBGridMedium bounds: %s renderFromMedium: %s phase: %s "
                        "sigmaScale: %f LeScale: %f (grids elided) ]",
                        bounds, renderFromMedium, phase, sigmaScale, LeScale);
}

// CloudMedium Method Definitions
CloudMedium *CloudMedium::Create(const ParameterDictionary &parameters,
                                 const Transform &renderFromMedium, const FileLoc *loc,
                                 Allocator alloc) {
    Float density = parameters.GetOneFloat("density", 1);
    Float g = parameters.GetOneFloat("g", 0.);
    Float wispiness = parameters.GetOneFloat("wispiness", 1);
    Float frequency = parameters.GetOneFloat("frequency", 5);
    Spectrum sigma_a =
        parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::Unbounded, alloc);
    if (!sigma_a)
        sigma_a = alloc.new_object<ConstantSpectrum>(1.f);
    Spectrum sigma_s =
        parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::Unbounded, alloc);
    if (!sigma_s)
        sigma_s = alloc.new_object<ConstantSpectrum>(1.f);

    Point3f p0 = parameters.GetOnePoint3f("p0", Point3f(0.f, 0.f, 0.f));
    Point3f p1 = parameters.GetOnePoint3f("p1", Point3f(1.f, 1.f, 1.f));

    return alloc.new_object<CloudMedium>(Bounds3f(p0, p1), renderFromMedium, sigma_a,
                                         sigma_s, g, density, wispiness, frequency,
                                         alloc);
}

// NanoVDBMedium Method Definitions
template <typename Buffer>
static nanovdb::GridHandle<Buffer> readGrid(const std::string &filename,
                                            const std::string &gridName,
                                            const FileLoc *loc, Allocator alloc) {
    NanoVDBBuffer buf(alloc);
    nanovdb::GridHandle<Buffer> grid;
    try {
        grid =
            nanovdb::io::readGrid<Buffer>(filename, gridName, 0 /* not verbose */, buf);
    } catch (const std::exception &e) {
        ErrorExit("nanovdb: %s: %s", filename, e.what());
    }

    if (grid) {
        if (!grid.gridMetaData()->isFogVolume() && !grid.gridMetaData()->isUnknown())
            ErrorExit(loc, "%s: \"%s\" isn't a FogVolume grid?", filename, gridName);

        LOG_VERBOSE("%s: found %d \"%s\" voxels", filename,
                    grid.gridMetaData()->activeVoxelCount(), gridName);
    }

    return grid;
}

NanoVDBMedium::NanoVDBMedium(const Transform &renderFromMedium, Spectrum sigma_a,
                             Spectrum sigma_s, Float sigmaScale, Float g,
                             nanovdb::GridHandle<NanoVDBBuffer> dg,
                             nanovdb::GridHandle<NanoVDBBuffer> tg, Float LeScale,
                             Float temperatureOffset, Float temperatureScale,
                             Allocator alloc)
    : renderFromMedium(renderFromMedium),
      sigma_a_spec(sigma_a, alloc),
      sigma_s_spec(sigma_s, alloc),
      phase(g),
      majorantGrid(Bounds3f(), {64, 64, 64}, alloc),
      densityGrid(std::move(dg)),
      temperatureGrid(std::move(tg)),
      LeScale(LeScale),
      temperatureOffset(temperatureOffset),
      temperatureScale(temperatureScale) {
    densityFloatGrid = densityGrid.grid<float>();

    sigma_a_spec.Scale(sigmaScale);
    sigma_s_spec.Scale(sigmaScale);

    nanovdb::BBox<nanovdb::Vec3R> bbox = densityFloatGrid->worldBBox();
    bounds = Bounds3f(Point3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                      Point3f(bbox.max()[0], bbox.max()[1], bbox.max()[2]));

    if (temperatureGrid) {
        temperatureFloatGrid = temperatureGrid.grid<float>();
        float minTemperature, maxTemperature;
        temperatureFloatGrid->tree().extrema(minTemperature, maxTemperature);
        LOG_VERBOSE("Max temperature: %f", maxTemperature);

        nanovdb::BBox<nanovdb::Vec3R> bbox = temperatureFloatGrid->worldBBox();
        bounds =
            Union(bounds, Bounds3f(Point3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                                   Point3f(bbox.max()[0], bbox.max()[1], bbox.max()[2])));
    }

    majorantGrid.bounds = bounds;

    // Initialize majorantGrid
#if 0
    // For debugging: single, medium-wide majorant...
    majorantGridRes = Point3i(1, 1, 1);
    Float minDensity, maxDensity;
    densityFloatGrid->tree().extrema(minDensity, maxDensity);
    majorantGrid.Set(0, 0, 0, maxDensity);
#else
    LOG_VERBOSE("Starting nanovdb grid GetMaxDensityGrid()");

    int gridSize = majorantGrid.res.x * majorantGrid.res.y * majorantGrid.res.z;
    ParallelFor(0, gridSize, [&](size_t index) {
        // Indices into majorantGrid
        int x = index % majorantGrid.res.x;
        int y = (index / majorantGrid.res.x) % majorantGrid.res.y;
        int z = index / (majorantGrid.res.x * majorantGrid.res.y);
        CHECK_EQ(index, x + majorantGrid.res.x * (y + majorantGrid.res.y * z));

        // World (aka medium) space bounds of this max grid cell
        Bounds3f wb(bounds.Lerp(Point3f(Float(x) / majorantGrid.res.x,
                                        Float(y) / majorantGrid.res.y,
                                        Float(z) / majorantGrid.res.z)),
                    bounds.Lerp(Point3f(Float(x + 1) / majorantGrid.res.x,
                                        Float(y + 1) / majorantGrid.res.y,
                                        Float(z + 1) / majorantGrid.res.z)));

        // Compute corresponding NanoVDB index-space bounds in floating-point.
        nanovdb::Vec3R i0 = densityFloatGrid->worldToIndexF(
            nanovdb::Vec3R(wb.pMin.x, wb.pMin.y, wb.pMin.z));
        nanovdb::Vec3R i1 = densityFloatGrid->worldToIndexF(
            nanovdb::Vec3R(wb.pMax.x, wb.pMax.y, wb.pMax.z));

        // Now find integer index-space bounds, accounting for both
        // filtering and the overall index bounding box.
        auto bbox = densityFloatGrid->indexBBox();
        Float delta = 1.f;  // Filter slop
        int nx0 = std::max(int(i0[0] - delta), bbox.min()[0]);
        int nx1 = std::min(int(i1[0] + delta), bbox.max()[0]);
        int ny0 = std::max(int(i0[1] - delta), bbox.min()[1]);
        int ny1 = std::min(int(i1[1] + delta), bbox.max()[1]);
        int nz0 = std::max(int(i0[2] - delta), bbox.min()[2]);
        int nz1 = std::min(int(i1[2] + delta), bbox.max()[2]);

        // FIXME: While the following is properly conservative, it can lead
        // to voxels with majorants that are much higher than any actual
        // volume density value in their extent. The issue comes up when
        // a) the density at a sample outside the is much higher than in the
        // voxel's interior and b) that sample only has a minimal
        // contribution in practice due to trilinear interpolation.  We
        // compute a majorant as if it might fully contribute, even though
        // it can't.  Fixing this would require careful handling of the
        // boundary samples.  The impact of these majorants is not
        // insignificant; they cause a roughly 10% slowdown in practice
        // due to excess null scattering in such voxels.
        float maxValue = 0;
        auto accessor = densityFloatGrid->getAccessor();
        // Apparently nanovdb integer bounding boxes are inclusive on
        // the upper end...
        for (int nz = nz0; nz <= nz1; ++nz)
            for (int ny = ny0; ny <= ny1; ++ny)
                for (int nx = nx0; nx <= nx1; ++nx)
                    maxValue = std::max(maxValue, accessor.getValue({nx, ny, nz}));

        // Only write into maxGrid once when we're done to minimize
        // cache thrashing..
        majorantGrid.Set(x, y, z, maxValue);
    });

    LOG_VERBOSE("Finished nanovdb grid GetMaxDensityGrid()");
#endif
}

std::string NanoVDBMedium::ToString() const {
    return StringPrintf("[ NanoVDBMedium bounds: %s LeScale: %f "
                        "temperatureOffset: %f temperatureScale: %f (grids elided) ]",
                        bounds, LeScale, temperatureOffset, temperatureScale);
}

NanoVDBMedium *NanoVDBMedium::Create(const ParameterDictionary &parameters,
                                     const Transform &renderFromMedium,
                                     const FileLoc *loc, Allocator alloc) {
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    if (filename.empty())
        ErrorExit(loc, "Must supply \"filename\" to \"nanovdb\" medium.");

    nanovdb::GridHandle<NanoVDBBuffer> densityGrid;
    std::string gridname = parameters.GetOneString("gridname", "density");
    densityGrid = readGrid<NanoVDBBuffer>(filename, gridname, loc, alloc);
    if (!densityGrid)
        ErrorExit(loc, "%s: didn't find \"density\" grid.", filename);

    nanovdb::GridHandle<NanoVDBBuffer> temperatureGrid;
    std::string temperaturename =
        parameters.GetOneString("temperaturename", "temperature");
    temperatureGrid = readGrid<NanoVDBBuffer>(filename, temperaturename, loc, alloc);

    Float LeScale = parameters.GetOneFloat("Lescale", 1.f);
    Float temperatureOffset = parameters.GetOneFloat("temperatureoffset",
                                                     parameters.GetOneFloat("temperaturecutoff", 0.f));
    Float temperatureScale = parameters.GetOneFloat("temperaturescale", 1.f);

    Float g = parameters.GetOneFloat("g", 0.);
    Spectrum sigma_a =
        parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::Unbounded, alloc);
    if (!sigma_a)
        sigma_a = alloc.new_object<ConstantSpectrum>(1.f);
    Spectrum sigma_s =
        parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::Unbounded, alloc);
    if (!sigma_s)
        sigma_s = alloc.new_object<ConstantSpectrum>(1.f);
    Float sigmaScale = parameters.GetOneFloat("scale", 1.f);

    return alloc.new_object<NanoVDBMedium>(
        renderFromMedium, sigma_a, sigma_s, sigmaScale, g, std::move(densityGrid),
        std::move(temperatureGrid), LeScale, temperatureOffset, temperatureScale, alloc);
}

Medium Medium::Create(const std::string &name, const ParameterDictionary &parameters,
                      const Transform &renderFromMedium, const FileLoc *loc,
                      Allocator alloc) {
    Medium m = nullptr;
    if (name == "homogeneous")
        m = HomogeneousMedium::Create(parameters, loc, alloc);
    else if (name == "uniformgrid") {
        m = GridMedium::Create(parameters, renderFromMedium, loc, alloc);
    } else if (name == "rgbgrid") {
        m = RGBGridMedium::Create(parameters, renderFromMedium, loc, alloc);
    } else if (name == "cloud") {
        m = CloudMedium::Create(parameters, renderFromMedium, loc, alloc);
    } else if (name == "nanovdb") {
        m = NanoVDBMedium::Create(parameters, renderFromMedium, loc, alloc);
    } else
        ErrorExit(loc, "%s: medium unknown.", name);

    if (!m)
        ErrorExit(loc, "%s: unable to create medium.", name);

    parameters.ReportUnused();
    return m;
}

}  // namespace pbrt
