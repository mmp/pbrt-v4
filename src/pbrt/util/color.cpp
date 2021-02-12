// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/color.h>

#if defined(PBRT_BUILD_GPU_RENDERER)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <pbrt/options.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/string.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <vector>

namespace pbrt {

std::string RGB::ToString() const {
    return StringPrintf("[ %f %f %f ]", r, g, b);
}

std::string RGBSigmoidPolynomial::ToString() const {
    return StringPrintf("[ RGBSigmoidPolynomial c0: %f c1: %f c2: %f ]", c0, c1, c2);
}

// RGBToSpectrumTable Method Definitions
RGBSigmoidPolynomial RGBToSpectrumTable::operator()(const RGB &rgb) const {
    CHECK(rgb[0] >= 0.f && rgb[1] >= 0.f && rgb[2] >= 0.f && rgb[0] <= 1.f &&
          rgb[1] <= 1.f && rgb[2] <= 1.f);

    // Find largest RGB component and handle uniform _rgb_
    if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
        Float v = rgb[0], inv = (v - .5f) / std::sqrt(v * (1.f - v));
        return {Float(0), Float(0), inv};
    }
    int i = 0;
    for (int j = 1; j < 3; ++j)
        if (rgb[j] >= rgb[i])
            i = j;

    // Compute floating-point offsets into polynomial coefficient table
    float z = rgb[i], sc = (res - 1) / z, x = rgb[(i + 1) % 3] * sc,
          y = rgb[(i + 2) % 3] * sc;

    // Compute integer indices and offsets for coefficient interpolation
    constexpr int nCoeffs = 3;
    int xi = std::min((int)x, res - 2), yi = std::min((int)y, res - 2),
        zi = FindInterval(res, [&](int i) { return scale[i] < z; }),
        offset = (((i * res + zi) * res + yi) * res + xi) * nCoeffs, dx = nCoeffs,
        dy = nCoeffs * res, dz = nCoeffs * res * res;
    Float x1 = x - xi, x0 = 1 - x1, y1 = y - yi, y0 = 1 - y1,
          z1 = (z - scale[zi]) / (scale[zi + 1] - scale[zi]), z0 = 1 - z1;

    // Bilinearly interpolate sigmoid polynomial coefficients _c_
    pstd::array<Float, nCoeffs> c;
    for (int j = 0; j < nCoeffs; ++j) {
        c[j] = ((data[offset] * x0 + data[offset + dx] * x1) * y0 +
                (data[offset + dy] * x0 + data[offset + dy + dx] * x1) * y1) *
                   z0 +
               ((data[offset + dz] * x0 + data[offset + dz + dx] * x1) * y0 +
                (data[offset + dz + dy] * x0 + data[offset + dz + dy + dx] * x1) * y1) *
                   z1;
        offset++;
    }

    return RGBSigmoidPolynomial(c[0], c[1], c[2]);
}

extern const int sRGBToSpectrumTable_Res;
extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[2359296];

const RGBToSpectrumTable *RGBToSpectrumTable::sRGB;

extern const int DCI_P3ToSpectrumTable_Res;
extern const float DCI_P3ToSpectrumTable_Scale[64];
extern const float DCI_P3ToSpectrumTable_Data[2359296];

const RGBToSpectrumTable *RGBToSpectrumTable::DCI_P3;

extern const int REC2020ToSpectrumTable_Res;
extern const float REC2020ToSpectrumTable_Scale[64];
extern const float REC2020ToSpectrumTable_Data[2359296];

const RGBToSpectrumTable *RGBToSpectrumTable::Rec2020;

extern const int ACES2065_1ToSpectrumTable_Res;
extern const float ACES2065_1ToSpectrumTable_Scale[64];
extern const float ACES2065_1ToSpectrumTable_Data[2359296];

const RGBToSpectrumTable *RGBToSpectrumTable::ACES2065_1;

void RGBToSpectrumTable::Init(Allocator alloc) {
#if defined(PBRT_BUILD_GPU_RENDERER)
    if (Options->useGPU) {
        extern const int sRGBToSpectrumTable_Res;
        extern const float sRGBToSpectrumTable_Scale[64];
        extern const float sRGBToSpectrumTable_Data[2359296];

        extern const int DCI_P3ToSpectrumTable_Res;
        extern const float DCI_P3ToSpectrumTable_Scale[64];
        extern const float DCI_P3ToSpectrumTable_Data[2359296];

        extern const int REC2020ToSpectrumTable_Res;
        extern const float REC2020ToSpectrumTable_Scale[64];
        extern const float REC2020ToSpectrumTable_Data[2359296];

        extern const int ACES2065_1ToSpectrumTable_Res;
        extern const float ACES2065_1ToSpectrumTable_Scale[64];
        extern const float ACES2065_1ToSpectrumTable_Data[2359296];

        // sRGB
        float *sRGBToSpectrumTableScalePtr =
            (float *)alloc.allocate_bytes(sizeof(sRGBToSpectrumTable_Scale));
        memcpy(sRGBToSpectrumTableScalePtr, sRGBToSpectrumTable_Scale,
               sizeof(sRGBToSpectrumTable_Scale));
        float *sRGBToSpectrumTableDataPtr =
            (float *)alloc.allocate_bytes(sizeof(sRGBToSpectrumTable_Data));
        memcpy(sRGBToSpectrumTableDataPtr, sRGBToSpectrumTable_Data,
               sizeof(sRGBToSpectrumTable_Data));

        sRGB = alloc.new_object<RGBToSpectrumTable>(sRGBToSpectrumTable_Res,
                                                    sRGBToSpectrumTableScalePtr,
                                                    sRGBToSpectrumTableDataPtr);

        // DCI_P3
        float *DCI_P3ToSpectrumTableScalePtr =
            (float *)alloc.allocate_bytes(sizeof(DCI_P3ToSpectrumTable_Scale));
        memcpy(DCI_P3ToSpectrumTableScalePtr, DCI_P3ToSpectrumTable_Scale,
               sizeof(DCI_P3ToSpectrumTable_Scale));
        float *DCI_P3ToSpectrumTableDataPtr =
            (float *)alloc.allocate_bytes(sizeof(DCI_P3ToSpectrumTable_Data));
        memcpy(DCI_P3ToSpectrumTableDataPtr, DCI_P3ToSpectrumTable_Data,
               sizeof(DCI_P3ToSpectrumTable_Data));

        DCI_P3 = alloc.new_object<RGBToSpectrumTable>(DCI_P3ToSpectrumTable_Res,
                                                      DCI_P3ToSpectrumTableScalePtr,
                                                      DCI_P3ToSpectrumTableDataPtr);

        // Rec2020
        float *REC2020ToSpectrumTableScalePtr =
            (float *)alloc.allocate_bytes(sizeof(REC2020ToSpectrumTable_Scale));
        memcpy(REC2020ToSpectrumTableScalePtr, REC2020ToSpectrumTable_Scale,
               sizeof(REC2020ToSpectrumTable_Scale));
        float *REC2020ToSpectrumTableDataPtr =
            (float *)alloc.allocate_bytes(sizeof(REC2020ToSpectrumTable_Data));
        memcpy(REC2020ToSpectrumTableDataPtr, REC2020ToSpectrumTable_Data,
               sizeof(REC2020ToSpectrumTable_Data));

        Rec2020 = alloc.new_object<RGBToSpectrumTable>(REC2020ToSpectrumTable_Res,
                                                       REC2020ToSpectrumTableScalePtr,
                                                       REC2020ToSpectrumTableDataPtr);

        // ACES2065_1
        float *ACES2065_1ToSpectrumTableScalePtr =
            (float *)alloc.allocate_bytes(sizeof(ACES2065_1ToSpectrumTable_Scale));
        memcpy(ACES2065_1ToSpectrumTableScalePtr, ACES2065_1ToSpectrumTable_Scale,
               sizeof(ACES2065_1ToSpectrumTable_Scale));
        float *ACES2065_1ToSpectrumTableDataPtr =
            (float *)alloc.allocate_bytes(sizeof(ACES2065_1ToSpectrumTable_Data));
        memcpy(ACES2065_1ToSpectrumTableDataPtr, ACES2065_1ToSpectrumTable_Data,
               sizeof(ACES2065_1ToSpectrumTable_Data));

        ACES2065_1 = alloc.new_object<RGBToSpectrumTable>(
            ACES2065_1ToSpectrumTable_Res, ACES2065_1ToSpectrumTableScalePtr,
            ACES2065_1ToSpectrumTableDataPtr);
        return;
    }
#endif
    sRGB = alloc.new_object<RGBToSpectrumTable>(
        sRGBToSpectrumTable_Res, sRGBToSpectrumTable_Scale, sRGBToSpectrumTable_Data);
    DCI_P3 = alloc.new_object<RGBToSpectrumTable>(DCI_P3ToSpectrumTable_Res,
                                                  DCI_P3ToSpectrumTable_Scale,
                                                  DCI_P3ToSpectrumTable_Data);
    Rec2020 = alloc.new_object<RGBToSpectrumTable>(REC2020ToSpectrumTable_Res,
                                                   REC2020ToSpectrumTable_Scale,
                                                   REC2020ToSpectrumTable_Data);
    ACES2065_1 = alloc.new_object<RGBToSpectrumTable>(ACES2065_1ToSpectrumTable_Res,
                                                      ACES2065_1ToSpectrumTable_Scale,
                                                      ACES2065_1ToSpectrumTable_Data);
}

std::string RGBToSpectrumTable::ToString() const {
    std::string id;
    if (this == RGBToSpectrumTable::sRGB)
        id = "(sRGB) ";
    else if (this == RGBToSpectrumTable::DCI_P3)
        id = "(DCI_P3) ";
    else if (this == RGBToSpectrumTable::Rec2020)
        id = "(Rec2020) ";
    else if (this == RGBToSpectrumTable::ACES2065_1)
        id = "(ACES2065_1) ";
    CHECK(!id.empty());

    return StringPrintf("[ RGBToSpectrumTable res: %d %s]", res, id);
}

std::string XYZ::ToString() const {
    return StringPrintf("[ %f %f %f ]", X, Y, Z);
}

// ColorEncoding Method Definitions
void sRGBColorEncoding::FromLinear(pstd::span<const Float> vin,
                                   pstd::span<uint8_t> vout) const {
    DCHECK_EQ(vin.size(), vout.size());
    for (size_t i = 0; i < vin.size(); ++i)
        vout[i] = LinearToSRGB8(vin[i]);
}

void sRGBColorEncoding::ToLinear(pstd::span<const uint8_t> vin,
                                 pstd::span<Float> vout) const {
    DCHECK_EQ(vin.size(), vout.size());
    for (size_t i = 0; i < vin.size(); ++i)
        vout[i] = SRGB8ToLinear(vin[i]);
}

Float sRGBColorEncoding::ToFloatLinear(Float v) const {
    return SRGBToLinear(v);
}

void ColorEncoding::Init(Allocator alloc) {
    Linear = alloc.new_object<LinearColorEncoding>();
    sRGB = alloc.new_object<sRGBColorEncoding>();
}

std::string ColorEncoding::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

ColorEncoding ColorEncoding::Linear;
ColorEncoding ColorEncoding::sRGB;

const ColorEncoding ColorEncoding::Get(const std::string &name, Allocator alloc) {
    if (name == "linear")
        return Linear;
    else if (name == "sRGB")
        return sRGB;
    else {
        static std::map<float, ColorEncoding> cache;

        std::vector<std::string> params = SplitStringsFromWhitespace(name);
        if (params.size() != 2 || params[0] != "gamma")
            ErrorExit("%s: expected \"gamma <value>\" for color encoding", name);
        Float gamma = atof(params[1].c_str());
        if (gamma == 0)
            ErrorExit("%s: unable to parse gamma value", params[1]);

        auto iter = cache.find(gamma);
        if (iter != cache.end())
            return iter->second;

        ColorEncoding enc = alloc.new_object<GammaColorEncoding>(gamma);
        cache[gamma] = enc;
        LOG_VERBOSE("Added ColorEncoding %s for gamma %f -> %s", name, gamma, enc);
        return enc;
    }
}

GammaColorEncoding::GammaColorEncoding(Float gamma) : gamma(gamma) {
    for (int i = 0; i < 256; ++i) {
        Float v = Float(i) / 255.f;
        applyLUT[i] = std::pow(v, gamma);
    }
    for (int i = 0; i < int(inverseLUT.size()); ++i) {
        Float v = Float(i) / Float(inverseLUT.size() - 1);
        inverseLUT[i] = Clamp(255.f * std::pow(v, 1.f / gamma) + .5f, 0, 255);
    }
}

void GammaColorEncoding::ToLinear(pstd::span<const uint8_t> vin,
                                  pstd::span<Float> vout) const {
    DCHECK_EQ(vin.size(), vout.size());
    for (size_t i = 0; i < vin.size(); ++i)
        vout[i] = applyLUT[vin[i]];
}

Float GammaColorEncoding::ToFloatLinear(Float v) const {
    return std::pow(v, gamma);
}

void GammaColorEncoding::FromLinear(pstd::span<const Float> vin,
                                    pstd::span<uint8_t> vout) const {
    DCHECK_EQ(vin.size(), vout.size());
    for (size_t i = 0; i < vin.size(); ++i)
        vout[i] =
            inverseLUT[Clamp(vin[i] * (inverseLUT.size() - 1), 0, inverseLUT.size() - 1)];
}

std::string GammaColorEncoding::ToString() const {
    return StringPrintf("[ GammaColorEncoding gamma: %f ]", gamma);
}

PBRT_CONST Float SRGBToLinearLUT[256] = {
    0.0000000000, 0.0003035270, 0.0006070540, 0.0009105810, 0.0012141080, 0.0015176350,
    0.0018211619, 0.0021246888, 0.0024282159, 0.0027317430, 0.0030352699, 0.0033465356,
    0.0036765069, 0.0040247170, 0.0043914421, 0.0047769533, 0.0051815170, 0.0056053917,
    0.0060488326, 0.0065120910, 0.0069954102, 0.0074990317, 0.0080231922, 0.0085681248,
    0.0091340570, 0.0097212177, 0.0103298230, 0.0109600937, 0.0116122449, 0.0122864870,
    0.0129830306, 0.0137020806, 0.0144438436, 0.0152085144, 0.0159962922, 0.0168073755,
    0.0176419523, 0.0185002182, 0.0193823613, 0.0202885624, 0.0212190095, 0.0221738834,
    0.0231533647, 0.0241576303, 0.0251868572, 0.0262412224, 0.0273208916, 0.0284260381,
    0.0295568332, 0.0307134409, 0.0318960287, 0.0331047624, 0.0343398079, 0.0356013142,
    0.0368894450, 0.0382043645, 0.0395462364, 0.0409151986, 0.0423114114, 0.0437350273,
    0.0451862030, 0.0466650836, 0.0481718220, 0.0497065634, 0.0512694679, 0.0528606549,
    0.0544802807, 0.0561284944, 0.0578054339, 0.0595112406, 0.0612460710, 0.0630100295,
    0.0648032799, 0.0666259527, 0.0684781820, 0.0703601092, 0.0722718611, 0.0742135793,
    0.0761853904, 0.0781874284, 0.0802198276, 0.0822827145, 0.0843762159, 0.0865004659,
    0.0886556059, 0.0908417329, 0.0930589810, 0.0953074843, 0.0975873619, 0.0998987406,
    0.1022417471, 0.1046164930, 0.1070231125, 0.1094617173, 0.1119324341, 0.1144353822,
    0.1169706732, 0.1195384338, 0.1221387982, 0.1247718409, 0.1274376959, 0.1301364899,
    0.1328683347, 0.1356333494, 0.1384316236, 0.1412633061, 0.1441284865, 0.1470272839,
    0.1499598026, 0.1529261619, 0.1559264660, 0.1589608639, 0.1620294005, 0.1651322246,
    0.1682693958, 0.1714410931, 0.1746473908, 0.1778884083, 0.1811642349, 0.1844749898,
    0.1878207624, 0.1912016720, 0.1946178079, 0.1980693042, 0.2015562356, 0.2050787061,
    0.2086368501, 0.2122307271, 0.2158605307, 0.2195262313, 0.2232279778, 0.2269658893,
    0.2307400703, 0.2345506549, 0.2383976579, 0.2422811985, 0.2462013960, 0.2501583695,
    0.2541521788, 0.2581829131, 0.2622507215, 0.2663556635, 0.2704978585, 0.2746773660,
    0.2788943350, 0.2831487954, 0.2874408960, 0.2917706966, 0.2961383164, 0.3005438447,
    0.3049873710, 0.3094689548, 0.3139887452, 0.3185468316, 0.3231432438, 0.3277781308,
    0.3324515820, 0.3371636569, 0.3419144452, 0.3467040956, 0.3515326977, 0.3564002514,
    0.3613068759, 0.3662526906, 0.3712377846, 0.3762622178, 0.3813261092, 0.3864295185,
    0.3915725648, 0.3967553079, 0.4019778669, 0.4072403014, 0.4125427008, 0.4178851545,
    0.4232677519, 0.4286905527, 0.4341537058, 0.4396572411, 0.4452012479, 0.4507858455,
    0.4564110637, 0.4620770514, 0.4677838385, 0.4735315442, 0.4793202281, 0.4851499796,
    0.4910208881, 0.4969330430, 0.5028865933, 0.5088814497, 0.5149177909, 0.5209956765,
    0.5271152258, 0.5332764983, 0.5394796133, 0.5457245708, 0.5520114899, 0.5583404899,
    0.5647116303, 0.5711249113, 0.5775805116, 0.5840784907, 0.5906189084, 0.5972018838,
    0.6038274169, 0.6104956269, 0.6172066331, 0.6239604354, 0.6307572126, 0.6375969648,
    0.6444797516, 0.6514056921, 0.6583748460, 0.6653873324, 0.6724432111, 0.6795425415,
    0.6866854429, 0.6938719153, 0.7011020184, 0.7083759308, 0.7156936526, 0.7230552435,
    0.7304608822, 0.7379105687, 0.7454043627, 0.7529423237, 0.7605246305, 0.7681512833,
    0.7758223414, 0.7835379243, 0.7912980318, 0.7991028428, 0.8069523573, 0.8148466945,
    0.8227858543, 0.8307699561, 0.8387991190, 0.8468732834, 0.8549926877, 0.8631572723,
    0.8713672161, 0.8796223402, 0.8879231811, 0.8962693810, 0.9046613574, 0.9130986929,
    0.9215820432, 0.9301108718, 0.9386858940, 0.9473065734, 0.9559735060, 0.9646862745,
    0.9734454751, 0.9822505713, 0.9911022186, 1.0000000000};

}  // namespace pbrt
