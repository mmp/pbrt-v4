// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/paramdict.h>

#include <pbrt/options.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>

#include <algorithm>
#include <utility>

namespace pbrt {

template <>
struct ParameterTypeTraits<ParameterType::Boolean> {
    static constexpr char typeName[] = "bool";
    static constexpr int nPerItem = 1;
    using ReturnType = uint8_t;
    static bool Convert(const uint8_t *v, const FileLoc *loc) { return *v; }
    static const auto &GetValues(const ParsedParameter &param) { return param.bools; }
};

constexpr char ParameterTypeTraits<ParameterType::Boolean>::typeName[];

template <>
struct ParameterTypeTraits<ParameterType::Float> {
    static constexpr char typeName[] = "float";
    static constexpr int nPerItem = 1;
    using ReturnType = Float;
    static Float Convert(const Float *v, const FileLoc *loc) { return *v; }
    static const auto &GetValues(const ParsedParameter &param) { return param.floats; }
};

constexpr char ParameterTypeTraits<ParameterType::Float>::typeName[];

template <>
struct ParameterTypeTraits<ParameterType::Integer> {
    static constexpr char typeName[] = "integer";
    static constexpr int nPerItem = 1;
    using ReturnType = int;
    static int Convert(const int *i, const FileLoc *loc) { return *i; }
    static const auto &GetValues(const ParsedParameter &param) { return param.ints; }
};

constexpr char ParameterTypeTraits<ParameterType::Integer>::typeName[];

template <>
struct ParameterTypeTraits<ParameterType::Point2f> {
    static constexpr char typeName[] = "point2";
    static constexpr int nPerItem = 2;
    using ReturnType = Point2f;
    static Point2f Convert(const Float *v, const FileLoc *loc) {
        return Point2f(v[0], v[1]);
    }
    static const auto &GetValues(const ParsedParameter &param) { return param.floats; }
};

constexpr char ParameterTypeTraits<ParameterType::Point2f>::typeName[];

template <>
struct ParameterTypeTraits<ParameterType::Vector2f> {
    static constexpr char typeName[] = "vector2";
    static constexpr int nPerItem = 2;
    using ReturnType = Vector2f;
    static Vector2f Convert(const Float *v, const FileLoc *loc) {
        return Vector2f(v[0], v[1]);
    }
    static const auto &GetValues(const ParsedParameter &param) { return param.floats; }
};

constexpr char ParameterTypeTraits<ParameterType::Vector2f>::typeName[];

// Point3f ParameterTypeTraits Definition
template <>
struct ParameterTypeTraits<ParameterType::Point3f> {
    // ParameterType::Point3f Type Traits
    using ReturnType = Point3f;

    static constexpr char typeName[] = "point3";

    static const auto &GetValues(const ParsedParameter &param) { return param.floats; }

    static constexpr int nPerItem = 3;

    static Point3f Convert(const Float *f, const FileLoc *loc) {
        return Point3f(f[0], f[1], f[2]);
    }
};

constexpr char ParameterTypeTraits<ParameterType::Point3f>::typeName[];

template <>
struct ParameterTypeTraits<ParameterType::Vector3f> {
    static constexpr char typeName[] = "vector3";
    static constexpr int nPerItem = 3;
    using ReturnType = Vector3f;
    static Vector3f Convert(const Float *v, const FileLoc *loc) {
        return Vector3f(v[0], v[1], v[2]);
    }
    static const auto &GetValues(const ParsedParameter &param) { return param.floats; }
};

constexpr char ParameterTypeTraits<ParameterType::Vector3f>::typeName[];

template <>
struct ParameterTypeTraits<ParameterType::Normal3f> {
    static constexpr char typeName[] = "normal";
    static constexpr int nPerItem = 3;
    using ReturnType = Normal3f;
    static Normal3f Convert(const Float *v, const FileLoc *loc) {
        return Normal3f(v[0], v[1], v[2]);
    }
    static const auto &GetValues(const ParsedParameter &param) { return param.floats; }
};

constexpr char ParameterTypeTraits<ParameterType::Normal3f>::typeName[];

template <>
struct ParameterTypeTraits<ParameterType::String> {
    static constexpr char typeName[] = "string";
    static constexpr int nPerItem = 1;
    using ReturnType = std::string;
    static std::string Convert(const std::string *s, const FileLoc *loc) { return *s; }
    static const auto &GetValues(const ParsedParameter &param) { return param.strings; }
};

constexpr char ParameterTypeTraits<ParameterType::String>::typeName[];

///////////////////////////////////////////////////////////////////////////
// ParameterDictionary

ParameterDictionary::ParameterDictionary(ParsedParameterVector p,
                                         const RGBColorSpace *colorSpace)
    : params(std::move(p)), colorSpace(colorSpace) {
    nOwnedParams = params.size();
    std::reverse(params.begin(), params.end());
    CHECK(colorSpace);
    checkParameterTypes();
}

ParameterDictionary::ParameterDictionary(ParsedParameterVector p0,
                                         const ParsedParameterVector &params1,
                                         const RGBColorSpace *colorSpace)
    : params(std::move(p0)), colorSpace(colorSpace) {
    nOwnedParams = params.size();
    std::reverse(params.begin(), params.end());
    CHECK(colorSpace);
    params.insert(params.end(), params1.rbegin(), params1.rend());
    checkParameterTypes();
}

void ParameterDictionary::checkParameterTypes() {
    for (const ParsedParameter *p : params)
        if (p->type != ParameterTypeTraits<ParameterType::Boolean>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::Float>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::Integer>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::Point2f>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::Vector2f>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::Point3f>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::Vector3f>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::Normal3f>::typeName &&
            p->type != ParameterTypeTraits<ParameterType::String>::typeName &&
            p->type != "texture" && p->type != "rgb" && p->type != "spectrum" &&
            p->type != "blackbody")
            ErrorExit(&p->loc, "%s: unknown parameter type", p->type);
}

// ParameterDictionary Method Definitions
Point3f ParameterDictionary::GetOnePoint3f(const std::string &name,
                                           const Point3f &def) const {
    return lookupSingle<ParameterType::Point3f>(name, def);
}

template <ParameterType PT>
typename ParameterTypeTraits<PT>::ReturnType ParameterDictionary::lookupSingle(
    const std::string &name,
    typename ParameterTypeTraits<PT>::ReturnType defaultValue) const {
    // Search _params_ for parameter _name_
    using traits = ParameterTypeTraits<PT>;
    for (const ParsedParameter *p : params) {
        if (p->name != name || p->type != traits::typeName)
            continue;
        // Extract parameter values from _p_
        const auto &values = traits::GetValues(*p);

        // Issue error if incorrect number of parameter values were provided
        if (values.empty())
            ErrorExit(&p->loc, "No values provided for parameter \"%s\".", name);
        if (values.size() > traits::nPerItem)
            ErrorExit(&p->loc, "More than one value provided for parameter \"%s\".",
                      name);

        // Return parameter values as _ReturnType_
        p->lookedUp = true;
        return traits::Convert(values.data(), &p->loc);
    }

    return defaultValue;
}

void ParameterDictionary::FreeParameters() {
    for (int i = 0; i < nOwnedParams; ++i)
        delete params[i];
    params.clear();
}

Float ParameterDictionary::GetOneFloat(const std::string &name, Float def) const {
    return lookupSingle<ParameterType::Float>(name, def);
}

int ParameterDictionary::GetOneInt(const std::string &name, int def) const {
    return lookupSingle<ParameterType::Integer>(name, def);
}

bool ParameterDictionary::GetOneBool(const std::string &name, bool def) const {
    return lookupSingle<ParameterType::Boolean>(name, def);
}

Point2f ParameterDictionary::GetOnePoint2f(const std::string &name,
                                           const Point2f &def) const {
    return lookupSingle<ParameterType::Point2f>(name, def);
}

Vector2f ParameterDictionary::GetOneVector2f(const std::string &name,
                                             const Vector2f &def) const {
    return lookupSingle<ParameterType::Vector2f>(name, def);
}

Vector3f ParameterDictionary::GetOneVector3f(const std::string &name,
                                             const Vector3f &def) const {
    return lookupSingle<ParameterType::Vector3f>(name, def);
}

Normal3f ParameterDictionary::GetOneNormal3f(const std::string &name,
                                             const Normal3f &def) const {
    return lookupSingle<ParameterType::Normal3f>(name, def);
}

Spectrum ParameterDictionary::GetOneSpectrum(const std::string &name,
                                             Spectrum defaultValue,
                                             SpectrumType spectrumType,
                                             Allocator alloc) const {
    for (const ParsedParameter *p : params) {
        if (p->name != name)
            continue;

        std::vector<Spectrum> s = extractSpectrumArray(*p, spectrumType, alloc);
        if (!s.empty()) {
            if (s.size() > 1)
                ErrorExit(&p->loc, "More than one value provided for parameter \"%s\".",
                          name);
            return s[0];
        }
    }

    return defaultValue;
}

std::string ParameterDictionary::GetOneString(const std::string &name,
                                              const std::string &def) const {
    return lookupSingle<ParameterType::String>(name, def);
}

template <typename ReturnType, typename ValuesType, typename C>
static std::vector<ReturnType> returnArray(const ValuesType &values,
                                           const ParsedParameter &param, int nPerItem,
                                           C convert) {
    if (values.empty())
        ErrorExit(&param.loc, "No values provided for \"%s\".", param.name);
    if (values.size() % nPerItem)
        ErrorExit(&param.loc, "Number of values provided for \"%s\" not a multiple of %d",
                  param.name, nPerItem);

    param.lookedUp = true;
    size_t n = values.size() / nPerItem;
    std::vector<ReturnType> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = convert(&values[nPerItem * i], &param.loc);
    return v;
}

template <typename ReturnType, typename G, typename C>
std::vector<ReturnType> ParameterDictionary::lookupArray(const std::string &name,
                                                         ParameterType type,
                                                         const char *typeName,
                                                         int nPerItem, G getValues,
                                                         C convert) const {
    for (const ParsedParameter *p : params)
        if (p->name == name && p->type == typeName)
            return returnArray<ReturnType>(getValues(*p), *p, nPerItem, convert);

    return {};
}

template <ParameterType PT>
std::vector<typename ParameterTypeTraits<PT>::ReturnType>
ParameterDictionary::lookupArray(const std::string &name) const {
    using traits = ParameterTypeTraits<PT>;
    return lookupArray<typename traits::ReturnType>(
        name, PT, traits::typeName, traits::nPerItem, traits::GetValues, traits::Convert);
}

std::vector<Float> ParameterDictionary::GetFloatArray(const std::string &name) const {
    return lookupArray<ParameterType::Float>(name);
}

std::vector<int> ParameterDictionary::GetIntArray(const std::string &name) const {
    return lookupArray<ParameterType::Integer>(name);
}

std::vector<uint8_t> ParameterDictionary::GetBoolArray(const std::string &name) const {
    return lookupArray<ParameterType::Boolean>(name);
}

std::vector<Point2f> ParameterDictionary::GetPoint2fArray(const std::string &name) const {
    return lookupArray<ParameterType::Point2f>(name);
}

std::vector<Vector2f> ParameterDictionary::GetVector2fArray(
    const std::string &name) const {
    return lookupArray<ParameterType::Vector2f>(name);
}

std::vector<Point3f> ParameterDictionary::GetPoint3fArray(const std::string &name) const {
    return lookupArray<ParameterType::Point3f>(name);
}

std::vector<Vector3f> ParameterDictionary::GetVector3fArray(
    const std::string &name) const {
    return lookupArray<ParameterType::Vector3f>(name);
}

std::vector<Normal3f> ParameterDictionary::GetNormal3fArray(
    const std::string &name) const {
    return lookupArray<ParameterType::Normal3f>(name);
}

static std::map<std::string, Spectrum> cachedSpectra;

// TODO: move this functionality (but not the caching?) to a Spectrum method.
static Spectrum readSpectrumFromFile(const std::string &filename, Allocator alloc) {
    std::string fn = ResolveFilename(filename);
    if (cachedSpectra.find(fn) != cachedSpectra.end())
        return cachedSpectra[fn];

    pstd::optional<Spectrum> pls = PiecewiseLinearSpectrum::Read(fn, alloc);
    if (!pls)
        return nullptr;

    cachedSpectra[fn] = *pls;
    return *pls;
}

std::vector<Spectrum> ParameterDictionary::extractSpectrumArray(
    const ParsedParameter &param, SpectrumType spectrumType, Allocator alloc) const {
    if (param.type == "rgb" || (Options->upgrade && param.type == "color"))
        return returnArray<Spectrum>(
            param.floats, param, 3,
            [this, spectrumType, &alloc, &param](const Float *v,
                                                 const FileLoc *loc) -> Spectrum {
                RGB rgb(v[0], v[1], v[2]);
                const RGBColorSpace &cs =
                    param.colorSpace ? *param.colorSpace : *colorSpace;
                if (rgb.r < 0 || rgb.g < 0 || rgb.b < 0)
                    ErrorExit(loc, "RGB parameter \"%s\" has negative component.",
                              param.name);
                if (spectrumType == SpectrumType::Albedo) {
                    if (rgb.r > 1 || rgb.g > 1 || rgb.b > 1)
                        ErrorExit(loc, "RGB parameter \"%s\" has > 1 component.",
                                  param.name);
                    return alloc.new_object<RGBAlbedoSpectrum>(cs, rgb);
                } else if (spectrumType == SpectrumType::Unbounded) {
                    return alloc.new_object<RGBUnboundedSpectrum>(cs, rgb);
                } else {
                    CHECK(spectrumType == SpectrumType::Illuminant);
                    return alloc.new_object<RGBIlluminantSpectrum>(cs, rgb);
                }
            });
    else if (param.type == "blackbody")
        return returnArray<Spectrum>(
            param.floats, param, 1,
            [this, &alloc](const Float *v, const FileLoc *loc) -> Spectrum {
                return alloc.new_object<BlackbodySpectrum>(v[0]);
            });
    else if (param.type == "spectrum" && !param.floats.empty()) {
        if (param.floats.size() % 2 != 0)
            ErrorExit(&param.loc, "Found odd number of values for \"%s\"", param.name);

        int nSamples = param.floats.size() / 2;
        return returnArray<Spectrum>(
            param.floats, param, param.floats.size(),
            [this, nSamples, &alloc, param](const Float *v,
                                            const FileLoc *Loc) -> Spectrum {
                std::vector<Float> lambda(nSamples), value(nSamples);
                for (int i = 0; i < nSamples; ++i) {
                    if (i > 0 && v[2 * i] <= lambda[i - 1])
                        ErrorExit(&param.loc,
                                  "Spectrum description invalid: at %d'th entry, "
                                  "wavelengths aren't increasing: %f >= %f.",
                                  i - 1, lambda[i - 1], v[2 * i]);
                    lambda[i] = v[2 * i];
                    value[i] = v[2 * i + 1];
                }
                return alloc.new_object<PiecewiseLinearSpectrum>(lambda, value, alloc);
            });
    } else if (param.type == "spectrum" && !param.strings.empty())
        return returnArray<Spectrum>(
            param.strings, param, 1,
            [param, &alloc](const std::string *s, const FileLoc *loc) -> Spectrum {
                Spectrum spd = GetNamedSpectrum(*s);
                if (spd)
                    return spd;

                spd = readSpectrumFromFile(*s, alloc);
                if (!spd)
                    ErrorExit(&param.loc, "%s: unable to read valid spectrum file", *s);
                return spd;
            });

    return {};
}

std::vector<Spectrum> ParameterDictionary::GetSpectrumArray(const std::string &name,
                                                            SpectrumType spectrumType,
                                                            Allocator alloc) const {
    for (const ParsedParameter *p : params) {
        if (p->name != name)
            continue;

        std::vector<Spectrum> s = extractSpectrumArray(*p, spectrumType, alloc);
        if (!s.empty())
            return s;
    }
    return {};
}

std::vector<std::string> ParameterDictionary::GetStringArray(
    const std::string &name) const {
    return lookupArray<ParameterType::String>(name);
}

std::string ParameterDictionary::GetTexture(const std::string &name) const {
    for (const ParsedParameter *p : params) {
        if (p->name != name || p->type != "texture")
            continue;

        if (p->strings.empty())
            ErrorExit(&p->loc, "No string values provided for parameter \"%s\".", name);
        if (p->strings.size() > 1)
            ErrorExit(&p->loc, "More than one value provided for parameter \"%s\".",
                      name);
        p->lookedUp = true;
        return p->strings[0];
    }

    return "";
}

std::vector<RGB> ParameterDictionary::GetRGBArray(const std::string &name) const {
    for (const ParsedParameter *p : params) {
        if (p->name == name && p->type == "rgb") {
            if (p->floats.size() % 3)
                ErrorExit(&p->loc, "Number of values given for \"rgb\" parameter %d "
                                   "\"name\" isn't a multiple of 3.");

            std::vector<RGB> rgb(p->floats.size() / 3);
            for (int i = 0; i < p->floats.size() / 3; ++i)
                rgb[i] =
                    RGB(p->floats[3 * i], p->floats[3 * i + 1], p->floats[3 * i + 2]);

            p->lookedUp = true;
            return rgb;
        }
    }
    return {};
}

pstd::optional<RGB> ParameterDictionary::GetOneRGB(const std::string &name) const {
    for (const ParsedParameter *p : params) {
        if (p->name == name && p->type == "rgb") {
            if (p->floats.size() < 3)
                ErrorExit(&p->loc, "Insufficient values for \"rgb\" parameter \"%s\".",
                          p->name);
            return RGB(p->floats[0], p->floats[1], p->floats[2]);
        }
    }
    return {};
}

Float ParameterDictionary::UpgradeBlackbody(const std::string &name) {
    Float scale = 1;
    for (ParsedParameter *p : params) {
        if (p->name == name && p->type == "blackbody") {
            if (p->floats.size() != 2)
                ErrorExit(&p->loc,
                          "Expected two values for legacy \"blackbody\" parameter.");
            scale *= p->floats[1];
            p->floats.pop_back();
        }
    }
    return scale;
}

void ParameterDictionary::remove(const std::string &name, const char *typeName) {
    for (auto iter = params.begin(); iter != params.end(); ++iter)
        if ((*iter)->name == name && (*iter)->type == typeName) {
            params.erase(iter);
            return;
        }
}

void ParameterDictionary::RemoveFloat(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Float>::typeName);
}

void ParameterDictionary::RemoveInt(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Integer>::typeName);
}

void ParameterDictionary::RemoveBool(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Boolean>::typeName);
}

void ParameterDictionary::RemovePoint2f(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Point2f>::typeName);
}

void ParameterDictionary::RemoveVector2f(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Vector2f>::typeName);
}

void ParameterDictionary::RemovePoint3f(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Point3f>::typeName);
}

void ParameterDictionary::RemoveVector3f(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Vector3f>::typeName);
}

void ParameterDictionary::RemoveNormal3f(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::Normal3f>::typeName);
}

void ParameterDictionary::RemoveString(const std::string &name) {
    remove(name, ParameterTypeTraits<ParameterType::String>::typeName);
}

void ParameterDictionary::RemoveTexture(const std::string &name) {
    remove(name, "texture");
}

void ParameterDictionary::RemoveSpectrum(const std::string &name) {
    remove(name, "spectrum");
    remove(name, "rgb");
    remove(name, "blackbody");
}

void ParameterDictionary::RenameParameter(const std::string &before,
                                          const std::string &after) {
    for (ParsedParameter *p : params)
        if (p->name == before)
            p->name = after;
}

void ParameterDictionary::RenameUsedTextures(
    const std::map<std::string, std::string> &m) {
    for (ParsedParameter *p : params) {
        if (p->type != "texture")
            continue;

        CHECK_EQ(1, p->strings.size());
        auto iter = m.find(p->strings[0]);
        if (iter != m.end())
            p->strings[0] = iter->second;
    }
}

void ParameterDictionary::ReportUnused() const {
    // type / name
    InlinedVector<std::pair<const std::string *, const std::string *>, 16> seen;

    for (const ParsedParameter *p : params) {
        if (p->mayBeUnused)
            continue;

        bool haveSeen =
            std::find_if(seen.begin(), seen.end(),
                         [&p](std::pair<const std::string *, const std::string *> p2) {
                             return *p2.first == p->type && *p2.second == p->name;
                         }) != seen.end();
        if (p->lookedUp) {
            // A parameter may be used when creating an initial Material, say,
            // but then an override from a Shape may shadow it such that its
            // name is already in the seen array.
            if (!haveSeen)
                seen.push_back(std::make_pair(&p->type, &p->name));
        } else if (haveSeen) {
            // It's shadowed by another parameter; that's fine.
        } else
            ErrorExit(&p->loc, "\"%s\": unused parameter.", p->name);
    }
}

std::string ParameterDictionary::ToParameterDefinition(const ParsedParameter *p,
                                                       int indentCount) {
    std::string s = StringPrintf("\"%s %s\" [ ", p->type, p->name);
    int continuationIndent = indentCount + 10 + p->type.size() + p->name.size();
    int column = indentCount + 4 + s.size();
    auto printOne = [&](const std::string &val) {
        if (column > 80) {  //   && i % 3 == 0) {
            s += "\n";
            s.append(continuationIndent, ' ');
            column = continuationIndent;
        }
        column += val.size();
        s += val;
    };

    for (Float v : p->floats)
        printOne(StringPrintf("%f ", v));
    for (int i : p->ints)
        printOne(StringPrintf("%i ", i));
    for (const auto &str : p->strings)
        printOne('"' + str + "\" ");
    for (bool b : p->bools)
        printOne(b ? "true " : "false ");
    s += "]";

    return s;
}

std::string ParameterDictionary::ToParameterList(int indentCount) const {
    std::string s;
    for (const ParsedParameter *p : params) {
        s.append(indentCount + 4, ' ');
        s += ToParameterDefinition(p, indentCount);
        s += "\n";
    }

    return s;
}

std::string ParameterDictionary::ToParameterDefinition(const std::string &name) const {
    for (const ParsedParameter *p : params)
        if (p->name == name)
            return ToParameterDefinition(p, 0);
    return "";
}

std::string ParameterDictionary::ToString() const {
    std::string s = "[ ParameterDictionary params: ";
    for (const ParsedParameter *p : params) {
        s += "[ " + p->ToString() + "] ";
    }
    s += StringPrintf("colorSpace: %s ",
                      colorSpace ? colorSpace->ToString().c_str() : "<nullptr>");
    s += "]";
    return s;
}

const FileLoc *ParameterDictionary::loc(const std::string &name) const {
    for (const ParsedParameter *p : params)
        if (p->name == name)
            return &p->loc;
    return nullptr;
}

// TextureParameterDictionary Method Definitions
TextureParameterDictionary::TextureParameterDictionary(const ParameterDictionary *dict,
                                                       const NamedTextures *textures)
    : dict(dict), textures(textures) {}

Float TextureParameterDictionary::GetOneFloat(const std::string &name, Float def) const {
    return dict->GetOneFloat(name, def);
}

int TextureParameterDictionary::GetOneInt(const std::string &name, int def) const {
    return dict->GetOneInt(name, def);
}

bool TextureParameterDictionary::GetOneBool(const std::string &name, bool def) const {
    return dict->GetOneBool(name, def);
}

Point2f TextureParameterDictionary::GetOnePoint2f(const std::string &name,
                                                  const Point2f &def) const {
    return dict->GetOnePoint2f(name, def);
}

Vector2f TextureParameterDictionary::GetOneVector2f(const std::string &name,
                                                    const Vector2f &def) const {
    return dict->GetOneVector2f(name, def);
}

Point3f TextureParameterDictionary::GetOnePoint3f(const std::string &name,
                                                  const Point3f &def) const {
    return dict->GetOnePoint3f(name, def);
}

Vector3f TextureParameterDictionary::GetOneVector3f(const std::string &name,
                                                    const Vector3f &def) const {
    return dict->GetOneVector3f(name, def);
}

Normal3f TextureParameterDictionary::GetOneNormal3f(const std::string &name,
                                                    const Normal3f &def) const {
    return dict->GetOneNormal3f(name, def);
}

Spectrum TextureParameterDictionary::GetOneSpectrum(const std::string &name, Spectrum def,
                                                    SpectrumType spectrumType,
                                                    Allocator alloc) const {
    return dict->GetOneSpectrum(name, def, spectrumType, alloc);
}

std::string TextureParameterDictionary::GetOneString(const std::string &name,
                                                     const std::string &def) const {
    return dict->GetOneString(name, def);
}

std::vector<Float> TextureParameterDictionary::GetFloatArray(
    const std::string &name) const {
    return dict->GetFloatArray(name);
}

std::vector<int> TextureParameterDictionary::GetIntArray(const std::string &name) const {
    return dict->GetIntArray(name);
}

std::vector<uint8_t> TextureParameterDictionary::GetBoolArray(
    const std::string &name) const {
    return dict->GetBoolArray(name);
}

std::vector<Point2f> TextureParameterDictionary::GetPoint2fArray(
    const std::string &name) const {
    return dict->GetPoint2fArray(name);
}

std::vector<Vector2f> TextureParameterDictionary::GetVector2fArray(
    const std::string &name) const {
    return dict->GetVector2fArray(name);
}

std::vector<Point3f> TextureParameterDictionary::GetPoint3fArray(
    const std::string &name) const {
    return dict->GetPoint3fArray(name);
}

std::vector<Vector3f> TextureParameterDictionary::GetVector3fArray(
    const std::string &name) const {
    return dict->GetVector3fArray(name);
}

std::vector<Normal3f> TextureParameterDictionary::GetNormal3fArray(
    const std::string &name) const {
    return dict->GetNormal3fArray(name);
}

std::vector<Spectrum> TextureParameterDictionary::GetSpectrumArray(
    const std::string &name, SpectrumType spectrumType, Allocator alloc) const {
    return dict->GetSpectrumArray(name, spectrumType, alloc);
}

std::vector<std::string> TextureParameterDictionary::GetStringArray(
    const std::string &name) const {
    return dict->GetStringArray(name);
}

SpectrumTexture TextureParameterDictionary::GetSpectrumTexture(std::string name,
                                                               Spectrum defaultValue,
                                                               SpectrumType spectrumType,
                                                               Allocator alloc) const {
    SpectrumTexture tex = GetSpectrumTextureOrNull(name, spectrumType, alloc);
    if (tex)
        return tex;
    else if (defaultValue)
        return alloc.new_object<SpectrumConstantTexture>(defaultValue);
    else
        return nullptr;
}

SpectrumTexture TextureParameterDictionary::GetSpectrumTextureOrNull(
    std::string name, SpectrumType spectrumType, Allocator alloc) const {
    const auto &spectrumTextures = (spectrumType == SpectrumType::Unbounded)
                                       ? textures->unboundedSpectrumTextures
                                       : ((spectrumType == SpectrumType::Albedo)
                                              ? textures->albedoSpectrumTextures
                                              : textures->illuminantSpectrumTextures);

    for (const ParsedParameter *p : dict->params) {
        if (p->name != name)
            continue;

        if (p->type == "texture") {
            if (p->strings.empty())
                ErrorExit(&p->loc, "No texture name provided for parameter \"%s\".",
                          name);
            if (p->strings.size() != 1)
                ErrorExit(&p->loc,
                          "More than one texture name provided for parameter \"%s\".",
                          name);

            p->lookedUp = true;
            auto iter = spectrumTextures.find(p->strings[0]);
            if (iter != spectrumTextures.end())
                return iter->second;

            ErrorExit(&p->loc,
                      R"(Couldn't find spectrum texture named "%s" for parameter "%s")",
                      p->strings[0], p->name);
        } else if (p->type == "rgb") {
            if (p->floats.size() != 3)
                ErrorExit(&p->loc,
                          "Didn't find three values for \"rgb\" parameter \"%s\".",
                          p->name);
            p->lookedUp = true;

            RGB rgb(p->floats[0], p->floats[1], p->floats[2]);
            if (rgb.r < 0 || rgb.g < 0 || rgb.b < 0)
                ErrorExit(&p->loc, "Negative value provided for RGB parameter \"%s\".",
                          p->name);
            Spectrum s;
            if (spectrumType == SpectrumType::Illuminant)
                s = alloc.new_object<RGBIlluminantSpectrum>(*dict->ColorSpace(), rgb);
            else if (spectrumType == SpectrumType::Unbounded)
                s = alloc.new_object<RGBUnboundedSpectrum>(*dict->ColorSpace(), rgb);
            else {
                CHECK(spectrumType == SpectrumType::Albedo);
                if (rgb.r > 1 || rgb.g > 1 || rgb.b > 1)
                    ErrorExit(&p->loc,
                              "RGB parameter \"%s\" used as an albedo has > 1 component.",
                              p->name);
                s = alloc.new_object<RGBAlbedoSpectrum>(*dict->ColorSpace(), rgb);
            }
            return alloc.new_object<SpectrumConstantTexture>(s);
        } else if (p->type == "spectrum" || p->type == "blackbody") {
            Spectrum s = GetOneSpectrum(name, nullptr, spectrumType, alloc);
            CHECK(s);
            return alloc.new_object<SpectrumConstantTexture>(s);
        }
    }

    return nullptr;
}

FloatTexture TextureParameterDictionary::GetFloatTexture(const std::string &name,
                                                         Float defaultValue,
                                                         Allocator alloc) const {
    FloatTexture tex = GetFloatTextureOrNull(name, alloc);
    return tex ? tex : alloc.new_object<FloatConstantTexture>(defaultValue);
}

FloatTexture TextureParameterDictionary::GetFloatTextureOrNull(const std::string &name,
                                                               Allocator alloc) const {
    for (const ParsedParameter *p : dict->params) {
        if (p->name != name)
            continue;

        if (p->type == "texture") {
            if (p->strings.empty())
                ErrorExit(&p->loc, "No texture name provided for parameter \"%s\".",
                          name);
            if (p->strings.size() != 1)
                ErrorExit(&p->loc,
                          "More than one texture name provided for parameter \"%s\".",
                          name);

            p->lookedUp = true;
            auto iter = textures->floatTextures.find(p->strings[0]);
            if (iter != textures->floatTextures.end())
                return iter->second;

            ErrorExit(&p->loc,
                      R"(Couldn't find float texture named "%s" for parameter "%s")",
                      p->strings[0], p->name);
        } else if (p->type == "float") {
            Float v = GetOneFloat(name, 0.f);  // we know this will be found
            return alloc.new_object<FloatConstantTexture>(v);
        }
    }

    return nullptr;
}

void TextureParameterDictionary::ReportUnused() const {
    dict->ReportUnused();
}
}  // namespace pbrt
