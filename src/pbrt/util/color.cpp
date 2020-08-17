// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/color.h>

#if defined(PBRT_BUILD_GPU_RENDERER)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

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

    // Find largest RGB component and handle black _rgb_
    int i = 0;
    for (int j = 1; j < 3; ++j)
        if (rgb[j] >= rgb[i])
            i = j;
    if (rgb[i] == 0)
        return {Float(0), Float(0), -std::numeric_limits<Float>::infinity()};

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

    sRGB = alloc.new_object<RGBToSpectrumTable>(
        sRGBToSpectrumTable_Res, sRGBToSpectrumTableScalePtr, sRGBToSpectrumTableDataPtr);

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

    ACES2065_1 = alloc.new_object<RGBToSpectrumTable>(ACES2065_1ToSpectrumTable_Res,
                                                      ACES2065_1ToSpectrumTableScalePtr,
                                                      ACES2065_1ToSpectrumTableDataPtr);
#else
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
#endif
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
std::string ColorEncodingHandle::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

const ColorEncodingHandle ColorEncodingHandle::Linear = new LinearColorEncoding;
const ColorEncodingHandle ColorEncodingHandle::sRGB = new sRGBColorEncoding;

const ColorEncodingHandle ColorEncodingHandle::Get(const std::string &name) {
    if (name == "linear")
        return Linear;
    else if (name == "sRGB")
        return sRGB;
    else {
        static std::map<float, ColorEncodingHandle> cache;

        std::vector<std::string> params = SplitStringsFromWhitespace(name);
        if (params.size() != 2 || params[0] != "gamma")
            ErrorExit("%s: expected \"gamma <value>\" for color encoding", name);
        Float gamma = atof(params[1].c_str());
        if (gamma == 0)
            ErrorExit("%s: unable to parse gamma value", params[1]);

        auto iter = cache.find(gamma);
        if (iter != cache.end())
            return iter->second;

        ColorEncodingHandle enc = new GammaColorEncoding(gamma);
        cache[gamma] = enc;
        LOG_VERBOSE("Added ColorEncoding %s for gamma %f -> %s", name, gamma, enc);
        return enc;
    }
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

void sRGBColorEncoding::FromLinear(pstd::span<const Float> vin,
                                   pstd::span<uint8_t> vout) const {
    DCHECK_EQ(vin.size(), vout.size());
    for (size_t i = 0; i < vin.size(); ++i)
        vout[i] = LinearToSRGB8(vin[i]);
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

PBRT_CONST PiecewiseLinearSegment LinearToSRGBPiecewise[] = {
    {0.000077473, 12.8513}, {0.0156956, 8.98367}, {0.0336267, 6.59881},
    {0.0473576, 5.40496},   {0.0588506, 4.66076}, {0.0688938, 4.14229},
    {0.0779005, 3.75559},   {0.0861196, 3.45349}, {0.0937148, 3.20944},
    {0.1008, 3.00718},      {0.10746, 2.83617},   {0.113757, 2.68923},
    {0.119741, 2.56128},    {0.12545, 2.44861},   {0.130916, 2.34846},
    {0.136166, 2.2587},     {0.141221, 2.17768},  {0.1461, 2.1041},
    {0.150818, 2.0369},     {0.15539, 1.97522},   {0.159826, 1.91836},
    {0.164138, 1.86574},    {0.168334, 1.81686},  {0.172422, 1.7713},
    {0.17641, 1.72872},     {0.180304, 1.68881},  {0.184109, 1.6513},
    {0.187832, 1.61598},    {0.191476, 1.58263},  {0.195047, 1.55108},
    {0.198547, 1.52119},    {0.201981, 1.49281},  {0.205352, 1.46582},
    {0.208663, 1.44012},    {0.211917, 1.41561},  {0.215116, 1.39219},
    {0.218263, 1.3698},     {0.22136, 1.34836},   {0.224409, 1.32781},
    {0.227412, 1.30808},    {0.230371, 1.28913},  {0.233288, 1.27091},
    {0.236164, 1.25337},    {0.239001, 1.23647},  {0.2418, 1.22018},
    {0.244562, 1.20446},    {0.247289, 1.18927},  {0.249983, 1.17459},
    {0.252643, 1.1604},     {0.255272, 1.14666},  {0.257869, 1.13335},
    {0.260437, 1.12046},    {0.262976, 1.10795},  {0.265487, 1.09582},
    {0.267971, 1.08404},    {0.270428, 1.0726},   {0.272859, 1.06148},
    {0.275266, 1.05067},    {0.277648, 1.04015},  {0.280006, 1.02991},
    {0.282342, 1.01995},    {0.284655, 1.01024},  {0.286946, 1.00077},
    {0.289216, 0.991545},   {0.291465, 0.982545}, {0.293694, 0.973764},
    {0.295903, 0.965192},   {0.298093, 0.956822}, {0.300264, 0.948647},
    {0.302416, 0.940658},   {0.304551, 0.932849}, {0.306668, 0.925214},
    {0.308768, 0.917746},   {0.31085, 0.91044},   {0.312917, 0.903289},
    {0.314967, 0.896288},   {0.317002, 0.889433}, {0.319021, 0.882719},
    {0.321025, 0.87614},    {0.323014, 0.869693}, {0.324988, 0.863372},
    {0.326949, 0.857175},   {0.328895, 0.851098}, {0.330827, 0.845135},
    {0.332747, 0.839285},   {0.334652, 0.833544}, {0.336545, 0.827907},
    {0.338426, 0.822374},   {0.340293, 0.816939}, {0.342149, 0.811601},
    {0.343992, 0.806356},   {0.345824, 0.801202}, {0.347644, 0.796137},
    {0.349452, 0.791158},   {0.351249, 0.786262}, {0.353035, 0.781448},
    {0.354811, 0.776713},   {0.356575, 0.772055}, {0.358329, 0.767472},
    {0.360073, 0.762962},   {0.361806, 0.758524}, {0.36353, 0.754154},
    {0.365243, 0.749853},   {0.366947, 0.745617}, {0.368641, 0.741446},
    {0.370326, 0.737337},   {0.372002, 0.73329},  {0.373668, 0.729302},
    {0.375325, 0.725373},   {0.376974, 0.721501}, {0.378614, 0.717684},
    {0.380245, 0.713922},   {0.381867, 0.710212}, {0.383481, 0.706555},
    {0.385087, 0.702948},   {0.386685, 0.69939},  {0.388275, 0.695882},
    {0.389856, 0.69242},    {0.39143, 0.689005},  {0.392996, 0.685635},
    {0.394555, 0.68231},    {0.396106, 0.679028}, {0.39765, 0.675788},
    {0.399186, 0.67259},    {0.400715, 0.669433}, {0.402237, 0.666316},
    {0.403751, 0.663238},   {0.405259, 0.660198}, {0.40676, 0.657195},
    {0.408254, 0.65423},    {0.409742, 0.6513},   {0.411223, 0.648406},
    {0.412697, 0.645546},   {0.414165, 0.642721}, {0.415626, 0.639928},
    {0.417081, 0.637169},   {0.41853, 0.634441},  {0.419972, 0.631745},
    {0.421409, 0.62908},    {0.422839, 0.626445}, {0.424264, 0.62384},
    {0.425682, 0.621264},   {0.427095, 0.618717}, {0.428502, 0.616198},
    {0.429903, 0.613706},   {0.431299, 0.611242}, {0.432689, 0.608804},
    {0.434073, 0.606393},   {0.435452, 0.604007}, {0.436826, 0.601647},
    {0.438194, 0.599311},   {0.439557, 0.597},    {0.440915, 0.594713},
    {0.442268, 0.59245},    {0.443615, 0.590209}, {0.444957, 0.587992},
    {0.446295, 0.585797},   {0.447627, 0.583624}, {0.448955, 0.581473},
    {0.450277, 0.579344},   {0.451595, 0.577235}, {0.452908, 0.575147},
    {0.454216, 0.57308},    {0.455519, 0.571032}, {0.456818, 0.569004},
    {0.458113, 0.566996},   {0.459402, 0.565007}, {0.460688, 0.563036},
    {0.461968, 0.561085},   {0.463245, 0.559151}, {0.464516, 0.557235},
    {0.465784, 0.555338},   {0.467047, 0.553457}, {0.468306, 0.551594},
    {0.469561, 0.549747},   {0.470812, 0.547918}, {0.472058, 0.546105},
    {0.473301, 0.544308},   {0.474539, 0.542526}, {0.475773, 0.540761},
    {0.477003, 0.539011},   {0.478229, 0.537277}, {0.479452, 0.535557},
    {0.48067, 0.533853},    {0.481885, 0.532163}, {0.483095, 0.530487},
    {0.484302, 0.528826},   {0.485505, 0.527179}, {0.486705, 0.525545},
    {0.487901, 0.523925},   {0.489093, 0.522319}, {0.490281, 0.520726},
    {0.491466, 0.519146},   {0.492647, 0.51758},  {0.493824, 0.516025},
    {0.494998, 0.514484},   {0.496169, 0.512955}, {0.497336, 0.511438},
    {0.4985, 0.509933},     {0.49966, 0.508441},  {0.500817, 0.50696},
    {0.501971, 0.50549},    {0.503121, 0.504033}, {0.504268, 0.502586},
    {0.505411, 0.501151},   {0.506551, 0.499727}, {0.507689, 0.498313},
    {0.508822, 0.496911},   {0.509953, 0.495519}, {0.511081, 0.494138},
    {0.512205, 0.492767},   {0.513326, 0.491407}, {0.514445, 0.490056},
    {0.51556, 0.488716},    {0.516672, 0.487385}, {0.517781, 0.486065},
    {0.518887, 0.484754},   {0.51999, 0.483452},  {0.52109, 0.48216},
    {0.522187, 0.480877},   {0.523282, 0.479604}, {0.524373, 0.47834},
    {0.525462, 0.477084},   {0.526547, 0.475838}, {0.52763, 0.4746},
    {0.52871, 0.473371},    {0.529787, 0.472151}, {0.530862, 0.470939},
    {0.531934, 0.469735},   {0.533003, 0.46854},  {0.534069, 0.467353},
    {0.535133, 0.466175},   {0.536193, 0.465004}, {0.537252, 0.463841},
    {0.538307, 0.462686},   {0.53936, 0.461539},  {0.540411, 0.460399},
    {0.541459, 0.459267},   {0.542504, 0.458143}, {0.543546, 0.457026},
    {0.544587, 0.455917},   {0.545624, 0.454814}, {0.546659, 0.453719},
    {0.547692, 0.452631},   {0.548722, 0.45155},  {0.54975, 0.450476},
    {0.550775, 0.449409},   {0.551798, 0.448349}, {0.552818, 0.447296},
    {0.553836, 0.446249},   {0.554852, 0.445209}, {0.555865, 0.444175},
    {0.556876, 0.443148},   {0.557885, 0.442128}, {0.558891, 0.441113},
    {0.559895, 0.440105}};

}  // namespace pbrt
