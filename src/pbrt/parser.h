// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_PARSER_H
#define PBRT_PARSER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/error.h>
#include <pbrt/util/pstd.h>

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace pbrt {

// ParsedParameter Definition
class ParsedParameter {
  public:
    // ParsedParameter Public Methods
    ParsedParameter(FileLoc loc) : loc(loc) {}

    void AddFloat(Float v);
    void AddInt(int i);
    void AddString(std::string_view str);
    void AddBool(bool v);

    std::string ToString() const;

    // ParsedParameter Public Members
    std::string type, name;
    FileLoc loc;
    pstd::vector<Float> floats;
    pstd::vector<int> ints;
    pstd::vector<std::string> strings;
    pstd::vector<uint8_t> bools;
    mutable bool lookedUp = false;
    mutable const RGBColorSpace *colorSpace = nullptr;
    bool mayBeUnused = false;
};

// ParsedParameterVector Definition
using ParsedParameterVector = InlinedVector<ParsedParameter *, 8>;

// SceneRepresentation Definition
class SceneRepresentation {
  public:
    // SceneRepresentation Interface
    virtual void Scale(Float sx, Float sy, Float sz, FileLoc loc) = 0;

    virtual void Shape(const std::string &name, ParsedParameterVector params,
                       FileLoc loc) = 0;

    virtual ~SceneRepresentation();

    virtual void Option(const std::string &name, const std::string &value,
                        FileLoc loc) = 0;

    virtual void Identity(FileLoc loc) = 0;
    virtual void Translate(Float dx, Float dy, Float dz, FileLoc loc) = 0;
    virtual void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc) = 0;
    virtual void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                        Float ux, Float uy, Float uz, FileLoc loc) = 0;
    virtual void ConcatTransform(Float transform[16], FileLoc loc) = 0;
    virtual void Transform(Float transform[16], FileLoc loc) = 0;
    virtual void CoordinateSystem(const std::string &, FileLoc loc) = 0;
    virtual void CoordSysTransform(const std::string &, FileLoc loc) = 0;
    virtual void ActiveTransformAll(FileLoc loc) = 0;
    virtual void ActiveTransformEndTime(FileLoc loc) = 0;
    virtual void ActiveTransformStartTime(FileLoc loc) = 0;
    virtual void TransformTimes(Float start, Float end, FileLoc loc) = 0;

    virtual void ColorSpace(const std::string &n, FileLoc loc) = 0;
    virtual void PixelFilter(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) = 0;
    virtual void Film(const std::string &type, ParsedParameterVector params,
                      FileLoc loc) = 0;
    virtual void Accelerator(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) = 0;
    virtual void Integrator(const std::string &name, ParsedParameterVector params,
                            FileLoc loc) = 0;
    virtual void Camera(const std::string &, ParsedParameterVector params,
                        FileLoc loc) = 0;
    virtual void MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                                 FileLoc loc) = 0;
    virtual void MediumInterface(const std::string &insideName,
                                 const std::string &outsideName, FileLoc loc) = 0;
    virtual void Sampler(const std::string &name, ParsedParameterVector params,
                         FileLoc loc) = 0;

    virtual void WorldBegin(FileLoc loc) = 0;
    virtual void AttributeBegin(FileLoc loc) = 0;
    virtual void AttributeEnd(FileLoc loc) = 0;
    virtual void Attribute(const std::string &target, ParsedParameterVector params,
                           FileLoc loc) = 0;
    virtual void Texture(const std::string &name, const std::string &type,
                         const std::string &texname, ParsedParameterVector params,
                         FileLoc loc) = 0;
    virtual void Material(const std::string &name, ParsedParameterVector params,
                          FileLoc loc) = 0;
    virtual void MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                                   FileLoc loc) = 0;
    virtual void NamedMaterial(const std::string &name, FileLoc loc) = 0;
    virtual void LightSource(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) = 0;
    virtual void AreaLightSource(const std::string &name, ParsedParameterVector params,
                                 FileLoc loc) = 0;
    virtual void ReverseOrientation(FileLoc loc) = 0;
    virtual void ObjectBegin(const std::string &name, FileLoc loc) = 0;
    virtual void ObjectEnd(FileLoc loc) = 0;
    virtual void ObjectInstance(const std::string &name, FileLoc loc) = 0;

    virtual void EndOfFiles() = 0;

  protected:
    // SceneRepresentation Protected Methods
    template <typename... Args>
    void ErrorExitDeferred(const char *fmt, Args &&...args) const {
        errorExit = true;
        Error(fmt, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void ErrorExitDeferred(const FileLoc *loc, const char *fmt, Args &&...args) const {
        errorExit = true;
        Error(loc, fmt, std::forward<Args>(args)...);
    }

    mutable bool errorExit = false;
};

// Scene Parsing Declarations
void ParseFiles(SceneRepresentation *scene, pstd::span<const std::string> filenames);
void ParseString(SceneRepresentation *scene, std::string str);

// Token Definition
struct Token {
    Token() = default;
    Token(std::string_view token, FileLoc loc) : token(token), loc(loc) {}
    std::string ToString() const;
    std::string_view token;
    FileLoc loc;
};

// Tokenizer Definition
class Tokenizer {
  public:
    // Tokenizer Public Methods
    Tokenizer(std::string str,
              std::function<void(const char *, const FileLoc *)> errorCallback);
#if defined(PBRT_HAVE_MMAP) || defined(PBRT_IS_WINDOWS)
    Tokenizer(void *ptr, size_t len, std::string filename,
              std::function<void(const char *, const FileLoc *)> errorCallback);
#endif
    ~Tokenizer();

    static std::unique_ptr<Tokenizer> CreateFromFile(
        const std::string &filename,
        std::function<void(const char *, const FileLoc *)> errorCallback);
    static std::unique_ptr<Tokenizer> CreateFromString(
        std::string str,
        std::function<void(const char *, const FileLoc *)> errorCallback);

    pstd::optional<Token> Next();

    // Just for parse().
    // TODO? Have a method to set this?
    FileLoc loc;

  private:
    // Tokenizer Private Methods
    int getChar() {
        if (pos == end)
            return EOF;
        int ch = *pos++;
        if (ch == '\n') {
            ++loc.line;
            loc.column = 0;
        } else
            ++loc.column;
        return ch;
    }
    void ungetChar() {
        --pos;
        if (*pos == '\n')
            // Don't worry about the column; we'll be going to the start of
            // the next line again shortly...
            --loc.line;
    }

    // Tokenizer Private Members
    // This function is called if there is an error during lexing.
    std::function<void(const char *, const FileLoc *)> errorCallback;

#if defined(PBRT_HAVE_MMAP) || defined(PBRT_IS_WINDOWS)
    // Scene files on disk are mapped into memory for lexing.  We need to
    // hold on to the starting pointer and total length so they can be
    // unmapped in the destructor.
    void *unmapPtr = nullptr;
    size_t unmapLength = 0;
#endif

    // If the input is stdin, then we copy everything until EOF into this
    // string and then start lexing.  This is a little wasteful (versus
    // tokenizing directly from stdin), but makes the implementation
    // simpler.
    std::string contents;

    // Pointers to the current position in the file and one past the end of
    // the file.
    const char *pos, *end;

    // If there are escaped characters in the string, we can't just return
    // a std::string_view into the mapped file. In that case, we handle the
    // escaped characters and return a std::string_view to sEscaped.  (And
    // thence, std::string_views from previous calls to Next() must be invalid
    // after a subsequent call, since we may reuse sEscaped.)
    std::string sEscaped;
};

}  // namespace pbrt

#endif  // PBRT_PARSER_H
