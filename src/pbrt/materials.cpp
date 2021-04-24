// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/materials.h>

#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/media.h>
#include <pbrt/paramdict.h>
#include <pbrt/textures.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>

#include <cmath>
#include <numeric>
#include <string>

namespace pbrt {

// DielectricMaterial Method Definitions
std::string DielectricMaterial::ToString() const {
    return StringPrintf("[ DielectricMaterial displacement: %s uRoughness: %s "
                        "vRoughness: %s eta: %s tint: %s remapRoughness: %s ]",
                        displacement, uRoughness, vRoughness, eta, tint, remapRoughness);
}

DielectricMaterial *DielectricMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    Spectrum eta;
    if (!parameters.GetFloatArray("eta").empty())
        eta = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("eta")[0]);
    else
        eta = parameters.GetOneSpectrum("eta", nullptr, SpectrumType::Unbounded, alloc);
    if (!eta)
        eta = alloc.new_object<ConstantSpectrum>(1.5f);

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    SpectrumTexture tint =
        parameters.GetSpectrumTextureOrNull("tint", SpectrumType::Albedo, alloc);
    return alloc.new_object<DielectricMaterial>(uRoughness, vRoughness, eta, displacement,
                                                normalMap, tint, remapRoughness);
}

// ThinDielectricMaterial Method Definitions
std::string ThinDielectricMaterial::ToString() const {
    return StringPrintf("[ ThinDielectricMaterial displacement: %s eta: %s ]",
                        displacement, eta);
}

ThinDielectricMaterial *ThinDielectricMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    Spectrum eta;
    if (!parameters.GetFloatArray("eta").empty())
        eta = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("eta")[0]);
    else
        eta = parameters.GetOneSpectrum("eta", nullptr, SpectrumType::Unbounded, alloc);
    if (!eta)
        eta = alloc.new_object<ConstantSpectrum>(1.5f);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);

    return alloc.new_object<ThinDielectricMaterial>(eta, displacement, normalMap);
}

// MixMaterial Method Definitions
std::string MixMaterial::ToString() const {
    return StringPrintf("[ MixMaterial materials: [ %s %s ] amount: %s ]", materials[0],
                        materials[1], amount);
}

MixMaterial *MixMaterial::Create(Material materials[2],
                                 const TextureParameterDictionary &parameters,
                                 const FileLoc *loc, Allocator alloc) {
    FloatTexture amount = parameters.GetFloatTexture("amount", 0.5f, alloc);

    if (Options->useGPU) {
        // Check for this stuff here, where we can include the FileLoc in
        // the error message. Note that both of these limitations could be
        // relaxed if they were problematic; the issue is that we currently
        // resolve MixMaterials in the closest hit shader...
        if (!BasicTextureEvaluator().CanEvaluate({amount}, {}))
            ErrorExit(loc, "The GPU renderer currently only supports basic textures "
                           "for its \"amount\" parameter.");
    }

    return alloc.new_object<MixMaterial>(materials, amount);
}

// HairMaterial Method Definitions
std::string HairMaterial::ToString() const {
    return StringPrintf("[ HairMaterial sigma_a: %s color: %s eumelanin: %s "
                        "pheomelanin: %s eta: %s beta_m: %s beta_n: %s alpha: %s ]",
                        sigma_a, color, eumelanin, pheomelanin, eta, beta_m, beta_n,
                        alpha);
}

HairMaterial *HairMaterial::Create(const TextureParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc) {
    SpectrumTexture sigma_a =
        parameters.GetSpectrumTextureOrNull("sigma_a", SpectrumType::Unbounded, alloc);
    SpectrumTexture color =
        parameters.GetSpectrumTextureOrNull("color", SpectrumType::Albedo, alloc);
    FloatTexture eumelanin = parameters.GetFloatTextureOrNull("eumelanin", alloc);
    FloatTexture pheomelanin = parameters.GetFloatTextureOrNull("pheomelanin", alloc);
    if (sigma_a) {
        if (color)
            Warning(loc, R"(Ignoring "color" parameter since "sigma_a" was provided.)");
        if (eumelanin)
            Warning(loc, "Ignoring \"eumelanin\" parameter since \"sigma_a\" was "
                         "provided.");
        if (pheomelanin)
            Warning(loc, "Ignoring \"pheomelanin\" parameter since \"sigma_a\" was "
                         "provided.");
    } else if (color) {
        if (sigma_a)
            Warning(loc, R"(Ignoring "sigma_a" parameter since "color" was provided.)");
        if (eumelanin)
            Warning(loc, "Ignoring \"eumelanin\" parameter since \"color\" was "
                         "provided.");
        if (pheomelanin)
            Warning(loc, "Ignoring \"pheomelanin\" parameter since \"color\" was "
                         "provided.");
    } else if (eumelanin || pheomelanin) {
        if (sigma_a)
            Warning(loc, "Ignoring \"sigma_a\" parameter since "
                         "\"eumelanin\"/\"pheomelanin\" was provided.");
        if (color)
            Warning(loc, "Ignoring \"color\" parameter since "
                         "\"eumelanin\"/\"pheomelanin\" was provided.");
    } else {
        // Default: brown-ish hair.
        sigma_a = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<RGBUnboundedSpectrum>(
                HairBxDF::SigmaAFromConcentration(1.3, 0.)));
    }

    FloatTexture eta = parameters.GetFloatTexture("eta", 1.55f, alloc);
    FloatTexture beta_m = parameters.GetFloatTexture("beta_m", 0.3f, alloc);
    FloatTexture beta_n = parameters.GetFloatTexture("beta_n", 0.3f, alloc);
    FloatTexture alpha = parameters.GetFloatTexture("alpha", 2.f, alloc);

    return alloc.new_object<HairMaterial>(sigma_a, color, eumelanin, pheomelanin, eta,
                                          beta_m, beta_n, alpha);
}

// DiffuseMaterial Method Definitions
std::string DiffuseMaterial::ToString() const {
    return StringPrintf("[ DiffuseMaterial displacement: %s reflectance: %s sigma: %s ]",
                        displacement, reflectance, sigma);
}

DiffuseMaterial *DiffuseMaterial::Create(const TextureParameterDictionary &parameters,
                                         Image *normalMap, const FileLoc *loc,
                                         Allocator alloc) {
    SpectrumTexture reflectance = parameters.GetSpectrumTexture(
        "reflectance", nullptr, SpectrumType::Albedo, alloc);
    if (!reflectance)
        reflectance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.5f));
    FloatTexture sigma = parameters.GetFloatTexture("sigma", 0.f, alloc);
    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);

    return alloc.new_object<DiffuseMaterial>(reflectance, sigma, displacement, normalMap);
}

// ConductorMaterial Method Definitions
std::string ConductorMaterial::ToString() const {
    return StringPrintf("[ ConductorMaterial displacement: %s eta: %s k: %s reflectance: "
                        "%s uRoughness: %s "
                        "vRoughness: %s remapRoughness: %s]",
                        displacement, eta, k, reflectance, uRoughness, vRoughness,
                        remapRoughness);
}

ConductorMaterial *ConductorMaterial::Create(const TextureParameterDictionary &parameters,
                                             Image *normalMap, const FileLoc *loc,
                                             Allocator alloc) {
    SpectrumTexture eta =
        parameters.GetSpectrumTextureOrNull("eta", SpectrumType::Unbounded, alloc);
    SpectrumTexture k =
        parameters.GetSpectrumTextureOrNull("k", SpectrumType::Unbounded, alloc);
    SpectrumTexture reflectance =
        parameters.GetSpectrumTextureOrNull("reflectance", SpectrumType::Albedo, alloc);

    if (reflectance && (eta || k))
        ErrorExit(loc, "For the conductor material, both \"reflectance\" "
                       "and \"eta\" and \"k\" can't be provided.");
    if (!reflectance) {
        if (!eta)
            eta = alloc.new_object<SpectrumConstantTexture>(
                GetNamedSpectrum("metal-Cu-eta"));
        if (!k)
            k = alloc.new_object<SpectrumConstantTexture>(GetNamedSpectrum("metal-Cu-k"));
    }

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<ConductorMaterial>(eta, k, reflectance, uRoughness,
                                               vRoughness, displacement, normalMap,
                                               remapRoughness);
}

// CoatedDiffuseMaterial Method Definitions
std::string CoatedDiffuseMaterial::ToString() const {
    return StringPrintf(
        "[ CoatedDiffuseMaterial displacement: %s reflectance: %s uRoughness: %s "
        "vRoughness: %s thickness: %s eta: %s remapRoughness: %s ]",
        displacement, reflectance, uRoughness, vRoughness, thickness, eta,
        remapRoughness);
}

CoatedDiffuseMaterial *CoatedDiffuseMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture reflectance = parameters.GetSpectrumTexture(
        "reflectance", nullptr, SpectrumType::Albedo, alloc);
    if (!reflectance)
        reflectance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.5f));

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture thickness = parameters.GetFloatTexture("thickness", .01, alloc);

    Spectrum eta;
    if (!parameters.GetFloatArray("eta").empty())
        eta = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("eta")[0]);
    else
        eta = parameters.GetOneSpectrum("eta", nullptr, SpectrumType::Unbounded, alloc);
    if (!eta)
        eta = alloc.new_object<ConstantSpectrum>(1.5f);

    int maxDepth = parameters.GetOneInt("maxdepth", 10);
    int nSamples = parameters.GetOneInt("nsamples", 1);

    FloatTexture g = parameters.GetFloatTexture("g", 0.f, alloc);
    SpectrumTexture albedo =
        parameters.GetSpectrumTexture("albedo", nullptr, SpectrumType::Albedo, alloc);
    if (!albedo)
        albedo = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.f));

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<CoatedDiffuseMaterial>(
        reflectance, uRoughness, vRoughness, thickness, albedo, g, eta, displacement,
        normalMap, remapRoughness, maxDepth, nSamples);
}

std::string CoatedConductorMaterial::ToString() const {
    return StringPrintf("[ CoatedConductorMaterial displacement: %f interfaceURoughness: "
                        "%f interfaceVRoughness: %f thickness: %f "
                        "interfaceEta: %f g: %s albedo: %s conductorURoughness: %s "
                        "conductorVRoughness: %s "
                        "conductorEta: %s k: %s conductorReflectance: %s remapRoughness: "
                        "%s maxDepth: %d nSamples: %d ]",
                        displacement, interfaceURoughness, interfaceVRoughness, thickness,
                        interfaceEta, g, albedo, conductorURoughness, conductorVRoughness,
                        conductorEta, k, reflectance, remapRoughness, maxDepth, nSamples);
}

CoatedConductorMaterial *CoatedConductorMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    // interface
    FloatTexture interfaceURoughness =
        parameters.GetFloatTextureOrNull("interface.uroughness", alloc);
    FloatTexture interfaceVRoughness =
        parameters.GetFloatTextureOrNull("interface.vroughness", alloc);
    if (!interfaceURoughness)
        interfaceURoughness =
            parameters.GetFloatTexture("interface.roughness", 0.f, alloc);
    if (!interfaceVRoughness)
        interfaceVRoughness =
            parameters.GetFloatTexture("interface.roughness", 0.f, alloc);

    FloatTexture thickness = parameters.GetFloatTexture("thickness", .01, alloc);

    Spectrum interfaceEta;
    if (!parameters.GetFloatArray("interface.eta").empty())
        interfaceEta = alloc.new_object<ConstantSpectrum>(
            parameters.GetFloatArray("interface.eta")[0]);
    else
        interfaceEta = parameters.GetOneSpectrum("interface.eta", nullptr,
                                                 SpectrumType::Unbounded, alloc);
    if (!interfaceEta)
        interfaceEta = alloc.new_object<ConstantSpectrum>(1.5f);

    // conductor
    FloatTexture conductorURoughness =
        parameters.GetFloatTextureOrNull("conductor.uroughness", alloc);
    FloatTexture conductorVRoughness =
        parameters.GetFloatTextureOrNull("conductor.vroughness", alloc);
    if (!conductorURoughness)
        conductorURoughness =
            parameters.GetFloatTexture("conductor.roughness", 0.f, alloc);
    if (!conductorVRoughness)
        conductorVRoughness =
            parameters.GetFloatTexture("conductor.roughness", 0.f, alloc);
    SpectrumTexture conductorEta = parameters.GetSpectrumTextureOrNull(
        "conductor.eta", SpectrumType::Unbounded, alloc);
    SpectrumTexture k = parameters.GetSpectrumTextureOrNull(
        "conductor.k", SpectrumType::Unbounded, alloc);
    SpectrumTexture reflectance =
        parameters.GetSpectrumTextureOrNull("reflectance", SpectrumType::Albedo, alloc);

    if (reflectance && (conductorEta || k))
        ErrorExit(loc, "For the coated conductor material, both \"reflectance\" "
                       "and \"eta\" and \"k\" can't be provided.");
    if (!reflectance) {
        if (!conductorEta)
            conductorEta = alloc.new_object<SpectrumConstantTexture>(
                GetNamedSpectrum("metal-Cu-eta"));
        if (!k)
            k = alloc.new_object<SpectrumConstantTexture>(GetNamedSpectrum("metal-Cu-k"));
    }

    int maxDepth = parameters.GetOneInt("maxdepth", 10);
    int nSamples = parameters.GetOneInt("nsamples", 1);

    FloatTexture g = parameters.GetFloatTexture("g", 0.f, alloc);
    SpectrumTexture albedo =
        parameters.GetSpectrumTexture("albedo", nullptr, SpectrumType::Albedo, alloc);
    if (!albedo)
        albedo = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.f));

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<CoatedConductorMaterial>(
        interfaceURoughness, interfaceVRoughness, thickness, interfaceEta, g, albedo,
        conductorURoughness, conductorVRoughness, conductorEta, k, reflectance,
        displacement, normalMap, remapRoughness, maxDepth, nSamples);
}

// SubsurfaceMaterial Method Definitions
std::string SubsurfaceMaterial::ToString() const {
    return StringPrintf("[ SubsurfaceMaterial displacment: %s scale: %f "
                        "sigma_a: %s sigma_s: %s "
                        "reflectance: %s mfp: %s uRoughness: %s vRoughness: %s "
                        "eta: %f remapRoughness: %s ]",
                        displacement, scale, sigma_a, sigma_s, reflectance, mfp,
                        uRoughness, vRoughness, eta, remapRoughness);
}

SubsurfaceMaterial *SubsurfaceMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture sigma_a, sigma_s, reflectance, mfp;

    Float g = parameters.GetOneFloat("g", 0.0f);

    // 4, mutually-exclusive, ways to specify the subsurface properties...
    std::string name = parameters.GetOneString("name", "");
    if (!name.empty()) {
        // 1. By name
        Spectrum sig_a, sig_s;
        if (!GetMediumScatteringProperties(name, &sig_a, &sig_s, alloc))
            ErrorExit(loc, "%s: named medium not found.", name);
        if (g != 0)
            Warning(loc, "Non-zero \"g\" ignored with named scattering coefficients.");
        g = 0; /* Enforce g=0 (the database specifies reduced scattering
                  coefficients) */
        sigma_a = alloc.new_object<SpectrumConstantTexture>(sig_a);
        sigma_s = alloc.new_object<SpectrumConstantTexture>(sig_s);
    } else {
        // 2. sigma_a and sigma_s directly specified
        sigma_a = parameters.GetSpectrumTextureOrNull("sigma_a", SpectrumType::Unbounded,
                                                      alloc);
        sigma_s = parameters.GetSpectrumTextureOrNull("sigma_s", SpectrumType::Unbounded,
                                                      alloc);
        if (sigma_a && !sigma_s)
            ErrorExit(loc, "Provided \"sigma_a\" parameter without \"sigma_s\".");
        if (sigma_s && !sigma_a)
            ErrorExit(loc, "Provided \"sigma_s\" parameter without \"sigma_a\".");

        if (!sigma_a && !sigma_s) {
            // 3. RGB/Spectrum, reflectance
            reflectance = parameters.GetSpectrumTextureOrNull(
                "reflectance", SpectrumType::Albedo, alloc);
            if (reflectance) {
                Spectrum one = alloc.new_object<ConstantSpectrum>(1.);
                mfp = parameters.GetSpectrumTexture("mfp", one, SpectrumType::Unbounded,
                                                    alloc);
            } else {
                // 4. nothing specified -- use defaults
                RGBUnboundedSpectrum *defaultSigma_a =
                    alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB,
                                                           RGB(.0011f, .0024f, .014f));
                RGBUnboundedSpectrum *defaultSigma_s =
                    alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB,
                                                           RGB(2.55f, 3.21f, 3.77f));
                sigma_a = alloc.new_object<SpectrumConstantTexture>(defaultSigma_a);
                sigma_s = alloc.new_object<SpectrumConstantTexture>(defaultSigma_s);
            }
        }
    }

    Float scale = parameters.GetOneFloat("scale", 1.f);
    Float eta = parameters.GetOneFloat("eta", 1.33f);

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<SubsurfaceMaterial>(
        scale, sigma_a, sigma_s, reflectance, mfp, g, eta, uRoughness, vRoughness,
        displacement, normalMap, remapRoughness, alloc);
}

// DiffuseTransmissionMaterial Method Definitions
std::string DiffuseTransmissionMaterial::ToString() const {
    return StringPrintf("[ DiffuseTransmissionMaterial displacment: %s reflectance: %s "
                        "transmittance: %s sigma: %s ]",
                        displacement, reflectance, transmittance, sigma);
}

DiffuseTransmissionMaterial *DiffuseTransmissionMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture reflectance = parameters.GetSpectrumTexture(
        "reflectance", nullptr, SpectrumType::Albedo, alloc);
    if (!reflectance)
        reflectance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.25f));

    SpectrumTexture transmittance = parameters.GetSpectrumTexture(
        "transmittance", nullptr, SpectrumType::Albedo, alloc);
    if (!transmittance)
        transmittance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.25f));

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);
    FloatTexture sigma = parameters.GetFloatTexture("sigma", 0.f, alloc);
    Float scale = parameters.GetOneFloat("scale", 1.f);

    return alloc.new_object<DiffuseTransmissionMaterial>(
        reflectance, transmittance, sigma, displacement, normalMap, scale);
}

MeasuredMaterial::MeasuredMaterial(const std::string &filename, FloatTexture displacement,
                                   Image *normalMap, Allocator alloc)
    : displacement(displacement), normalMap(normalMap) {
    brdf = MeasuredBxDF::BRDFDataFromFile(filename, alloc);
}

std::string MeasuredMaterial::ToString() const {
    return StringPrintf("[ MeasuredMaterial displacement: %s normalMap: %p ]",
                        displacement, normalMap);
}

MeasuredMaterial *MeasuredMaterial::Create(const TextureParameterDictionary &parameters,
                                           Image *normalMap, const FileLoc *loc,
                                           Allocator alloc) {
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    if (filename.empty()) {
        Error("Filename must be provided for MeasuredMaterial");
        return nullptr;
    }
    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);

    return alloc.new_object<MeasuredMaterial>(filename, displacement, normalMap, alloc);
}

std::string Material::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto toStr = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(toStr);
}

STAT_COUNTER("Scene/Materials", nMaterialsCreated);

Material Material::Create(const std::string &name,
                          const TextureParameterDictionary &parameters, Image *normalMap,
                          /*const */ std::map<std::string, Material> &namedMaterials,
                          const FileLoc *loc, Allocator alloc) {
    Material material;
    if (name.empty() || name == "none") {
        Warning(loc, "Material \"%s\" is deprecated; use \"interface\" instead.",
                name.c_str());
        return nullptr;
    } else if (name == "interface")
        return nullptr;
    else if (name == "diffuse")
        material = DiffuseMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "coateddiffuse")
        material = CoatedDiffuseMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "coatedconductor")
        material = CoatedConductorMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "diffusetransmission")
        material = DiffuseTransmissionMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "dielectric")
        material = DielectricMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "thindielectric")
        material = ThinDielectricMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "hair")
        material = HairMaterial::Create(parameters, loc, alloc);
    else if (name == "conductor")
        material = ConductorMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "measured")
        material = MeasuredMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "subsurface")
        material = SubsurfaceMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "mix") {
        std::vector<std::string> materialNames = parameters.GetStringArray("materials");
        if (materialNames.size() != 2)
            ErrorExit(
                "Must provide two values for \"string materials\" for mix material.");

        Material materials[2];
        for (int i = 0; i < 2; ++i) {
            auto iter = namedMaterials.find(materialNames[i]);
            if (iter == namedMaterials.end())
                ErrorExit("%s: named material not found.", materialNames[i]);
            materials[i] = iter->second;
        }
        material = MixMaterial::Create(materials, parameters, loc, alloc);
    } else
        ErrorExit(loc, "%s: material type unknown.", name);

    if (!material)
        ErrorExit(loc, "%s: unable to create material.", name);

    parameters.ReportUnused();
    ++nMaterialsCreated;
    return material;
}

}  // namespace pbrt
