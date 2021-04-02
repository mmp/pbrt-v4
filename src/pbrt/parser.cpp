// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/parser.h>

#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/parsedscene.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <double-conversion/double-conversion.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#ifdef PBRT_HAVE_MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(PBRT_IS_WINDOWS)
#include <windows.h>  // Windows file mapping API
#endif
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pbrt {

///////////////////////////////////////////////////////////////////////////
// ParsedParameter

void ParsedParameter::AddFloat(Float v) {
    CHECK(ints.empty() && strings.empty() && bools.empty());
    floats.push_back(v);
}

void ParsedParameter::AddInt(int i) {
    CHECK(floats.empty() && strings.empty() && bools.empty());
    ints.push_back(i);
}

void ParsedParameter::AddString(std::string_view str) {
    CHECK(floats.empty() && ints.empty() && bools.empty());
    strings.push_back({str.begin(), str.end()});
}

void ParsedParameter::AddBool(bool v) {
    CHECK(floats.empty() && ints.empty() && strings.empty());
    bools.push_back(v);
}

std::string ParsedParameter::ToString() const {
    std::string str;
    str += std::string("\"") + type + " " + name + std::string("\" [ ");
    if (!floats.empty())
        for (Float d : floats)
            str += StringPrintf("%f ", d);
    if (!ints.empty())
        for (int i : ints)
            str += StringPrintf("%d ", i);
    else if (!strings.empty())
        for (const auto &s : strings)
            str += '\"' + s + "\" ";
    else if (!bools.empty())
        for (bool b : bools)
            str += b ? "true " : "false ";
    str += "] ";

    return str;
}

SceneRepresentation::~SceneRepresentation() {}

static std::string toString(std::string_view s) {
    return std::string(s.data(), s.size());
}

std::string Token::ToString() const {
    return StringPrintf("[ Token token: %s loc: %s ]", toString(token), loc);
}

STAT_MEMORY_COUNTER("Memory/Tokenizer buffers", tokenizerMemory);

// Tokenizer Implementation
static char decodeEscaped(int ch, const FileLoc &loc) {
    switch (ch) {
    case EOF:
        ErrorExit(&loc, "premature EOF after character escape '\\'");
    case 'b':
        return '\b';
    case 'f':
        return '\f';
    case 'n':
        return '\n';
    case 'r':
        return '\r';
    case 't':
        return '\t';
    case '\\':
        return '\\';
    case '\'':
        return '\'';
    case '\"':
        return '\"';
    default:
        ErrorExit(&loc, "unexpected escaped character \"%c\"", ch);
    }
    return 0;  // NOTREACHED
}

static double_conversion::StringToDoubleConverter floatParser(
    double_conversion::StringToDoubleConverter::ALLOW_HEX, 0. /* empty string value */,
    0. /* junk string value */, nullptr /* infinity symbol */, nullptr /* NaN symbol */);

std::unique_ptr<Tokenizer> Tokenizer::CreateFromFile(
    const std::string &filename,
    std::function<void(const char *, const FileLoc *)> errorCallback) {
    if (filename == "-") {
        // Handle stdin by slurping everything into a string.
        std::string str;
        int ch;
        while ((ch = getchar()) != EOF)
            str.push_back((char)ch);
        return std::make_unique<Tokenizer>(std::move(str), std::move(errorCallback));
    }

#ifdef PBRT_HAVE_MMAP
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        errorCallback(StringPrintf("%s: %s", filename, ErrorString()).c_str(), nullptr);
        return nullptr;
    }

    struct stat stat;
    if (fstat(fd, &stat) != 0) {
        errorCallback(StringPrintf("%s: %s", filename, ErrorString()).c_str(), nullptr);
        return nullptr;
    }

    size_t len = stat.st_size;
    void *ptr = mmap(nullptr, len, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
    if (close(fd) != 0) {
        errorCallback(StringPrintf("%s: %s", filename, ErrorString()).c_str(), nullptr);
        return nullptr;
    }

    return std::make_unique<Tokenizer>(ptr, len, filename, std::move(errorCallback));
#elif defined(PBRT_IS_WINDOWS)
    auto errorReportLambda = [&errorCallback, &filename]() -> std::unique_ptr<Tokenizer> {
        LPSTR messageBuffer = nullptr;
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                           FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL, ::GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                       (LPSTR)&messageBuffer, 0, NULL);

        errorCallback(StringPrintf("%s: %s", filename, messageBuffer).c_str(), nullptr);

        LocalFree(messageBuffer);
        return nullptr;
    };

    HANDLE fileHandle = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, 0,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (!fileHandle) {
        return errorReportLambda();
    }

    size_t len = GetFileSize(fileHandle, 0);

    HANDLE mapping = CreateFileMapping(fileHandle, 0, PAGE_READONLY, 0, 0, 0);
    CloseHandle(fileHandle);
    if (mapping == 0) {
        return errorReportLambda();
    }

    LPVOID ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    if (ptr == nullptr) {
        return errorReportLambda();
    }

    std::string str(static_cast<const char *>(ptr), len);

    return std::make_unique<Tokenizer>(ptr, len, filename, std::move(errorCallback));
#else
    FILE *f = fopen(filename.c_str(), "r");
    if (!f) {
        errorCallback(StringPrintf("%s: %s", filename, ErrorString()).c_str(), nullptr);
        return nullptr;
    }

    std::string str;
    int ch;
    while ((ch = fgetc(f)) != EOF)
        str.push_back(char(ch));
    fclose(f);
    return std::make_unique_ptr<Tokenizer>(std::move(str), std::move(errorCallback));
#endif
}

std::unique_ptr<Tokenizer> Tokenizer::CreateFromString(
    std::string str, std::function<void(const char *, const FileLoc *)> errorCallback) {
    return std::make_unique<Tokenizer>(std::move(str), std::move(errorCallback));
}

Tokenizer::Tokenizer(std::string str,
                     std::function<void(const char *, const FileLoc *)> errorCallback)
    : loc("<stdin>"), errorCallback(std::move(errorCallback)), contents(std::move(str)) {
    pos = contents.data();
    end = pos + contents.size();
    tokenizerMemory += contents.size();
}

#if defined(PBRT_HAVE_MMAP) || defined(PBRT_IS_WINDOWS)
Tokenizer::Tokenizer(void *ptr, size_t len, std::string filename,
                     std::function<void(const char *, const FileLoc *)> errorCallback)
    : errorCallback(std::move(errorCallback)), unmapPtr(ptr), unmapLength(len) {
    // This is disgusting and leaks memory, but it ensures that the
    // filename in FileLocs returned by the Tokenizer remain valid even
    // after it has been destroyed.
    loc = FileLoc(*new std::string(filename));
    pos = (const char *)ptr;
    end = pos + len;
}
#endif

Tokenizer::~Tokenizer() {
#ifdef PBRT_HAVE_MMAP
    if ((unmapPtr != nullptr) && unmapLength > 0)
        if (munmap(unmapPtr, unmapLength) != 0)
            errorCallback(StringPrintf("munmap: %s", ErrorString()).c_str(), nullptr);
#elif defined(PBRT_IS_WINDOWS)
    if (unmapPtr && UnmapViewOfFile(unmapPtr) == 0)
        errorCallback(StringPrintf("UnmapViewOfFile: %s", ErrorString()).c_str(),
                      nullptr);
#endif
}

pstd::optional<Token> Tokenizer::Next() {
    while (true) {
        const char *tokenStart = pos;
        FileLoc startLoc = loc;

        int ch = getChar();
        if (ch == EOF)
            return {};
        else if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') {
            // nothing
        } else if (ch == '"') {
            // scan to closing quote
            bool haveEscaped = false;
            while ((ch = getChar()) != '"') {
                if (ch == EOF) {
                    errorCallback("premature EOF", &startLoc);
                    return {};
                } else if (ch == '\n') {
                    errorCallback("unterminated string", &startLoc);
                    return {};
                } else if (ch == '\\') {
                    haveEscaped = true;
                    // Grab the next character
                    if ((ch = getChar()) == EOF) {
                        errorCallback("premature EOF", &startLoc);
                        return {};
                    }
                }
            }

            if (!haveEscaped)
                return Token({tokenStart, size_t(pos - tokenStart)}, startLoc);
            else {
                sEscaped.clear();
                for (const char *p = tokenStart; p < pos; ++p) {
                    if (*p != '\\')
                        sEscaped.push_back(*p);
                    else {
                        ++p;
                        CHECK_LT(p, pos);
                        sEscaped.push_back(decodeEscaped(*p, startLoc));
                    }
                }
                return Token({sEscaped.data(), sEscaped.size()}, startLoc);
            }
        } else if (ch == '[' || ch == ']') {
            return Token({tokenStart, size_t(1)}, startLoc);
        } else if (ch == '#') {
            // comment: scan to EOL (or EOF)
            while ((ch = getChar()) != EOF) {
                if (ch == '\n' || ch == '\r') {
                    ungetChar();
                    break;
                }
            }

            return Token({tokenStart, size_t(pos - tokenStart)}, startLoc);
        } else {
            // Regular statement or numeric token; scan until we hit a
            // space, opening quote, or bracket.
            while ((ch = getChar()) != EOF) {
                if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r' || ch == '"' ||
                    ch == '[' || ch == ']') {
                    ungetChar();
                    break;
                }
            }
            return Token({tokenStart, size_t(pos - tokenStart)}, startLoc);
        }
    }
}

static int parseInt(const Token &t) {
    bool negate = t.token[0] == '-';

    int index = 0;
    if (t.token[0] == '+' || t.token[0] == '-')
        ++index;

    int64_t value = 0;
    while (index < t.token.size()) {
        if (!(t.token[index] >= '0' && t.token[index] <= '9'))
            ErrorExit(&t.loc, "\"%c\": expected a number", t.token[index]);
        value = 10 * value + (t.token[index] - '0');
        ++index;

        if (value > std::numeric_limits<int>::max())
            ErrorExit(&t.loc,
                      "Numeric value too large to represent as a 32-bit integer.");
        else if (value < std::numeric_limits<int>::lowest())
            Warning(&t.loc, "Numeric value %d too low to represent as a 32-bit integer.");
    }

    return negate ? -value : value;
}

static double parseFloat(const Token &t) {
    // Fast path for a single digit
    if (t.token.size() == 1) {
        if (!(t.token[0] >= '0' && t.token[0] <= '9'))
            ErrorExit(&t.loc, "\"%c\": expected a number", t.token[0]);
        return t.token[0] - '0';
    }

    // Copy to a buffer so we can NUL-terminate it, as strto[idf]() expect.
    char buf[64];
    char *bufp = buf;
    std::unique_ptr<char[]> allocBuf;
    CHECK_RARE(1e-5, t.token.size() + 1 >= sizeof(buf));
    if (t.token.size() + 1 >= sizeof(buf)) {
        // This should be very unusual, but is necessary in case we get a
        // goofball number with lots of leading zeros, for example.
        allocBuf = std::make_unique<char[]>(t.token.size() + 1);
        bufp = allocBuf.get();
    }

    std::copy(t.token.begin(), t.token.end(), bufp);
    bufp[t.token.size()] = '\0';

    // Can we just use strtol?
    auto isInteger = [](std::string_view str) {
        for (char ch : str)
            if (!(ch >= '0' && ch <= '9'))
                return false;
        return true;
    };

    int length = 0;
    double val;
    if (isInteger(t.token)) {
        char *endptr;
        val = double(strtol(bufp, &endptr, 10));
        length = endptr - bufp;
    } else if (sizeof(Float) == sizeof(float))
        val = floatParser.StringToFloat(bufp, t.token.size(), &length);
    else
        val = floatParser.StringToDouble(bufp, t.token.size(), &length);

    if (length == 0)
        ErrorExit(&t.loc, "%s: expected a number", toString(t.token));

    return val;
}

inline bool isQuotedString(std::string_view str) {
    return str.size() >= 2 && str[0] == '"' && str.back() == '"';
}

static std::string_view dequoteString(const Token &t) {
    if (!isQuotedString(t.token))
        ErrorExit(&t.loc, "\"%s\": expected quoted string", toString(t.token));

    std::string_view str = t.token;
    str.remove_prefix(1);
    str.remove_suffix(1);
    return str;
}

constexpr int TokenOptional = 0;
constexpr int TokenRequired = 1;

template <typename Next, typename Unget>
static ParsedParameterVector parseParameters(
    Next nextToken, Unget ungetToken, bool formatting,
    const std::function<void(const Token &token, const char *)> &errorCallback) {
    ParsedParameterVector parameterVector;

    while (true) {
        pstd::optional<Token> t = nextToken(TokenOptional);
        if (!t.has_value())
            return parameterVector;

        if (!isQuotedString(t->token)) {
            ungetToken(*t);
            return parameterVector;
        }

        ParsedParameter *param = new ParsedParameter(t->loc);

        std::string_view decl = dequoteString(*t);

        auto skipSpace = [&decl](std::string_view::const_iterator iter) {
            while (iter != decl.end() && (*iter == ' ' || *iter == '\t'))
                ++iter;
            return iter;
        };
        // Skip to the next whitespace character (or the end of the string).
        auto skipToSpace = [&decl](std::string_view::const_iterator iter) {
            while (iter != decl.end() && *iter != ' ' && *iter != '\t')
                ++iter;
            return iter;
        };

        auto typeBegin = skipSpace(decl.begin());
        if (typeBegin == decl.end())
            ErrorExit(&t->loc, "Parameter \"%s\" doesn't have a type declaration?!",
                      std::string(decl.begin(), decl.end()));

        // Find end of type declaration
        auto typeEnd = skipToSpace(typeBegin);
        param->type.assign(typeBegin, typeEnd);

        if (formatting) {  // close enough: upgrade...
            if (param->type == "point")
                param->type = "point3";
            if (param->type == "vector")
                param->type = "vector3";
            if (param->type == "color")
                param->type = "rgb";
        }

        auto nameBegin = skipSpace(typeEnd);
        if (nameBegin == decl.end())
            ErrorExit(&t->loc, "Unable to find parameter name from \"%s\"",
                      std::string(decl.begin(), decl.end()));

        auto nameEnd = skipToSpace(nameBegin);
        param->name.assign(nameBegin, nameEnd);

        enum ValType { Unknown, String, Bool, Float, Int } valType = Unknown;

        if (param->type == "integer")
            valType = Int;

        auto addVal = [&](const Token &t) {
            if (isQuotedString(t.token)) {
                switch (valType) {
                case Unknown:
                    valType = String;
                    break;
                case String:
                    break;
                case Float:
                    errorCallback(t, "expected floating-point value");
                case Int:
                    errorCallback(t, "expected integer value");
                case Bool:
                    errorCallback(t, "expected Boolean value");
                }

                param->AddString(dequoteString(t));
            } else if (t.token[0] == 't' && t.token == "true") {
                switch (valType) {
                case Unknown:
                    valType = Bool;
                    break;
                case String:
                    errorCallback(t, "expected string value");
                case Float:
                    errorCallback(t, "expected floating-point value");
                case Int:
                    errorCallback(t, "expected integer value");
                case Bool:
                    break;
                }

                param->AddBool(true);
            } else if (t.token[0] == 'f' && t.token == "false") {
                switch (valType) {
                case Unknown:
                    valType = Bool;
                    break;
                case String:
                    errorCallback(t, "expected string value");
                case Float:
                    errorCallback(t, "expected floating-point value");
                case Int:
                    errorCallback(t, "expected integer value");
                case Bool:
                    break;
                }

                param->AddBool(false);
            } else {
                switch (valType) {
                case Unknown:
                    valType = Float;
                    break;
                case String:
                    errorCallback(t, "expected string value");
                case Float:
                    break;
                case Int:
                    break;
                case Bool:
                    errorCallback(t, "expected Boolean value");
                }

                if (valType == Int)
                    param->AddInt(parseInt(t));
                else
                    param->AddFloat(parseFloat(t));
            }
        };

        Token val = *nextToken(TokenRequired);

        if (val.token == "[") {
            while (true) {
                val = *nextToken(TokenRequired);
                if (val.token == "]")
                    break;
                addVal(val);
            }
        } else {
            addVal(val);
        }

        if (formatting && param->type == "bool") {
            for (const auto &b : param->strings) {
                if (b == "true")
                    param->bools.push_back(true);
                else if (b == "false")
                    param->bools.push_back(false);
                else
                    Error(&param->loc,
                          "%s: neither \"true\" nor \"false\" in bool "
                          "parameter list.",
                          b);
            }
            param->strings.clear();
        }

        parameterVector.push_back(param);
    }

    return parameterVector;
}

void parse(SceneRepresentation *scene, std::unique_ptr<Tokenizer> t) {
    FormattingScene *formattingScene = dynamic_cast<FormattingScene *>(scene);
    bool formatting = formattingScene != nullptr;

    static std::atomic<bool> warnedTransformBeginEndDeprecated{false};

    std::vector<std::pair<std::thread, ParsedScene *>> imports;

    LOG_VERBOSE("Started parsing %s",
                std::string(t->loc.filename.begin(), t->loc.filename.end()));
    std::vector<std::unique_ptr<Tokenizer>> fileStack;
    fileStack.push_back(std::move(t));

    pstd::optional<Token> ungetToken;

    auto parseError = [&](const char *msg, const FileLoc *loc) {
        ErrorExit(loc, "%s", msg);
    };

    // nextToken is a little helper function that handles the file stack,
    // returning the next token from the current file until reaching EOF,
    // at which point it switches to the next file (if any).
    std::function<pstd::optional<Token>(int)> nextToken;
    nextToken = [&](int flags) -> pstd::optional<Token> {
        if (ungetToken.has_value())
            return std::exchange(ungetToken, {});

        if (fileStack.empty()) {
            if ((flags & TokenRequired) != 0) {
                ErrorExit("premature end of file");
            }
            return {};
        }

        pstd::optional<Token> tok = fileStack.back()->Next();

        if (!tok) {
            // We've reached EOF in the current file. Anything more to parse?
            LOG_VERBOSE("Finished parsing %s",
                        std::string(fileStack.back()->loc.filename.begin(),
                                    fileStack.back()->loc.filename.end()));
            fileStack.pop_back();
            return nextToken(flags);
        } else if (tok->token[0] == '#') {
            // Swallow comments, unless --format or --toply was given, in
            // which case they're printed to stdout.
            if (formatting)
                printf("%s%s\n", dynamic_cast<FormattingScene *>(scene)->indent().c_str(),
                       toString(tok->token).c_str());
            return nextToken(flags);
        } else
            // Regular token; success.
            return tok;
    };

    auto unget = [&](Token t) {
        CHECK(!ungetToken.has_value());
        ungetToken = t;
    };

    // Helper function for pbrt API entrypoints that take a single string
    // parameter and a ParameterVector (e.g. pbrtShape()).
    // using BasicEntrypoint = void (ParsedScene::*)(const std::string &,
    // ParsedParameterVector, FileLoc);
    auto basicParamListEntrypoint =
        [&](void (SceneRepresentation::*apiFunc)(const std::string &,
                                                 ParsedParameterVector, FileLoc),
            FileLoc loc) {
            Token t = *nextToken(TokenRequired);
            std::string_view dequoted = dequoteString(t);
            std::string n = toString(dequoted);
            ParsedParameterVector parameterVector = parseParameters(
                nextToken, unget, formatting, [&](const Token &t, const char *msg) {
                    std::string token = toString(t.token);
                    std::string str = StringPrintf("%s: %s", token, msg);
                    parseError(str.c_str(), &t.loc);
                });
            (scene->*apiFunc)(n, std::move(parameterVector), loc);
        };

    auto syntaxError = [&](const Token &t) {
        ErrorExit(&t.loc, "Unknown directive: %s", toString(t.token));
    };

    pstd::optional<Token> tok;
    CheckCallbackScope _([&tok]() -> std::string {
        if (!tok.has_value())
            return "";
        std::string filename(tok->loc.filename.begin(), tok->loc.filename.end());
        return StringPrintf("Current parser location %s:%d:%d", filename, tok->loc.line,
                            tok->loc.column);
    });

    while (true) {
        tok = nextToken(TokenOptional);
        if (!tok.has_value())
            break;

        switch (tok->token[0]) {
        case 'A':
            if (tok->token == "AttributeBegin")
                scene->AttributeBegin(tok->loc);
            else if (tok->token == "AttributeEnd")
                scene->AttributeEnd(tok->loc);
            else if (tok->token == "Attribute")
                basicParamListEntrypoint(&SceneRepresentation::Attribute, tok->loc);
            else if (tok->token == "ActiveTransform") {
                Token a = *nextToken(TokenRequired);
                if (a.token == "All")
                    scene->ActiveTransformAll(tok->loc);
                else if (a.token == "EndTime")
                    scene->ActiveTransformEndTime(tok->loc);
                else if (a.token == "StartTime")
                    scene->ActiveTransformStartTime(tok->loc);
                else
                    syntaxError(*tok);
            } else if (tok->token == "AreaLightSource")
                basicParamListEntrypoint(&SceneRepresentation::AreaLightSource, tok->loc);
            else if (tok->token == "Accelerator")
                basicParamListEntrypoint(&SceneRepresentation::Accelerator, tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'C':
            if (tok->token == "ConcatTransform") {
                if (nextToken(TokenRequired)->token != "[")
                    syntaxError(*tok);
                Float m[16];
                for (int i = 0; i < 16; ++i)
                    m[i] = parseFloat(*nextToken(TokenRequired));
                if (nextToken(TokenRequired)->token != "]")
                    syntaxError(*tok);
                scene->ConcatTransform(m, tok->loc);
            } else if (tok->token == "CoordinateSystem") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                scene->CoordinateSystem(toString(n), tok->loc);
            } else if (tok->token == "CoordSysTransform") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                scene->CoordSysTransform(toString(n), tok->loc);
            } else if (tok->token == "ColorSpace") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                scene->ColorSpace(toString(n), tok->loc);
            } else if (tok->token == "Camera")
                basicParamListEntrypoint(&SceneRepresentation::Camera, tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'F':
            if (tok->token == "Film")
                basicParamListEntrypoint(&SceneRepresentation::Film, tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'I':
            if (tok->token == "Integrator")
                basicParamListEntrypoint(&SceneRepresentation::Integrator, tok->loc);
            else if (tok->token == "Include") {
                Token filenameToken = *nextToken(TokenRequired);
                std::string filename = toString(dequoteString(filenameToken));
                if (formatting)
                    Printf("%sInclude \"%s\"\n",
                           dynamic_cast<FormattingScene *>(scene)->indent(), filename);
                else {
                    filename = ResolveFilename(filename);
                    std::unique_ptr<Tokenizer> tinc =
                        Tokenizer::CreateFromFile(filename, parseError);
                    if (tinc) {
                        LOG_VERBOSE("Started parsing %s",
                                    std::string(tinc->loc.filename.begin(),
                                                tinc->loc.filename.end()));
                        fileStack.push_back(std::move(tinc));
                    }
                }
            } else if (tok->token == "Import") {
                Token filenameToken = *nextToken(TokenRequired);
                std::string filename = toString(dequoteString(filenameToken));
                if (formatting)
                    Printf("%sImport \"%s\"\n",
                           dynamic_cast<FormattingScene *>(scene)->indent(), filename);
                else {
                    ParsedScene *parsedScene = dynamic_cast<ParsedScene *>(scene);
                    CHECK(parsedScene != nullptr);

                    if (parsedScene->currentBlock != ParsedScene::BlockState::WorldBlock)
                        ErrorExit(&tok->loc, "Import statement only allowed inside world "
                                             "definition block.");

                    filename = ResolveFilename(filename);
                    std::unique_ptr<Tokenizer> timport =
                        Tokenizer::CreateFromFile(filename, parseError);
                    if (timport) {
                        ParsedScene *importScene = parsedScene->CopyForImport();

                        std::thread importThread(
                            [](ParsedScene *scene, std::unique_ptr<Tokenizer> timport) {
                                parse(scene, std::move(timport));
                            },
                            importScene, std::move(timport));
                        imports.push_back(
                            std::make_pair(std::move(importThread), importScene));
                    }
                }
            } else if (tok->token == "Identity")
                scene->Identity(tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'L':
            if (tok->token == "LightSource")
                basicParamListEntrypoint(&SceneRepresentation::LightSource, tok->loc);
            else if (tok->token == "LookAt") {
                Float v[9];
                for (int i = 0; i < 9; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                scene->LookAt(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                              tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'M':
            if (tok->token == "MakeNamedMaterial")
                basicParamListEntrypoint(&SceneRepresentation::MakeNamedMaterial,
                                         tok->loc);
            else if (tok->token == "MakeNamedMedium")
                basicParamListEntrypoint(&SceneRepresentation::MakeNamedMedium, tok->loc);
            else if (tok->token == "Material")
                basicParamListEntrypoint(&SceneRepresentation::Material, tok->loc);
            else if (tok->token == "MediumInterface") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                std::string names[2];
                names[0] = toString(n);

                // Check for optional second parameter
                pstd::optional<Token> second = nextToken(TokenOptional);
                if (second.has_value()) {
                    if (isQuotedString(second->token))
                        names[1] = toString(dequoteString(*second));
                    else {
                        unget(*second);
                        names[1] = names[0];
                    }
                } else
                    names[1] = names[0];

                scene->MediumInterface(names[0], names[1], tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'N':
            if (tok->token == "NamedMaterial") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                scene->NamedMaterial(toString(n), tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'O':
            if (tok->token == "ObjectBegin") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                scene->ObjectBegin(toString(n), tok->loc);
            } else if (tok->token == "ObjectEnd")
                scene->ObjectEnd(tok->loc);
            else if (tok->token == "ObjectInstance") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                scene->ObjectInstance(toString(n), tok->loc);
            } else if (tok->token == "Option") {
                std::string name = toString(dequoteString(*nextToken(TokenRequired)));
                std::string value = toString(nextToken(TokenRequired)->token);
                scene->Option(name, value, tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'P':
            if (tok->token == "PixelFilter")
                basicParamListEntrypoint(&SceneRepresentation::PixelFilter, tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'R':
            if (tok->token == "ReverseOrientation")
                scene->ReverseOrientation(tok->loc);
            else if (tok->token == "Rotate") {
                Float v[4];
                for (int i = 0; i < 4; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                scene->Rotate(v[0], v[1], v[2], v[3], tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'S':
            if (tok->token == "Shape")
                basicParamListEntrypoint(&SceneRepresentation::Shape, tok->loc);
            else if (tok->token == "Sampler")
                basicParamListEntrypoint(&SceneRepresentation::Sampler, tok->loc);
            else if (tok->token == "Scale") {
                Float v[3];
                for (int i = 0; i < 3; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                scene->Scale(v[0], v[1], v[2], tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'T':
            if (tok->token == "TransformBegin") {
                if (formattingScene)
                    formattingScene->TransformBegin(tok->loc);
                else {
                    if (!warnedTransformBeginEndDeprecated) {
                        Warning(&tok->loc, "TransformBegin/End are deprecated and should "
                                           "be replaced with AttributeBegin/End");
                        warnedTransformBeginEndDeprecated = true;
                    }
                    scene->AttributeBegin(tok->loc);
                }
            } else if (tok->token == "TransformEnd") {
                if (formattingScene)
                    formattingScene->TransformEnd(tok->loc);
                else
                    scene->AttributeEnd(tok->loc);
            } else if (tok->token == "Transform") {
                if (nextToken(TokenRequired)->token != "[")
                    syntaxError(*tok);
                Float m[16];
                for (int i = 0; i < 16; ++i)
                    m[i] = parseFloat(*nextToken(TokenRequired));
                if (nextToken(TokenRequired)->token != "]")
                    syntaxError(*tok);
                scene->Transform(m, tok->loc);
            } else if (tok->token == "Translate") {
                Float v[3];
                for (int i = 0; i < 3; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                scene->Translate(v[0], v[1], v[2], tok->loc);
            } else if (tok->token == "TransformTimes") {
                Float v[2];
                for (int i = 0; i < 2; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                scene->TransformTimes(v[0], v[1], tok->loc);
            } else if (tok->token == "Texture") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                std::string name = toString(n);
                n = dequoteString(*nextToken(TokenRequired));
                std::string type = toString(n);

                Token t = *nextToken(TokenRequired);
                std::string_view dequoted = dequoteString(t);
                std::string texName = toString(dequoted);
                ParsedParameterVector params = parseParameters(
                    nextToken, unget, formatting, [&](const Token &t, const char *msg) {
                        std::string token = toString(t.token);
                        std::string str = StringPrintf("%s: %s", token, msg);
                        parseError(str.c_str(), &t.loc);
                    });

                scene->Texture(name, type, texName, std::move(params), tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'W':
            if (tok->token == "WorldBegin")
                scene->WorldBegin(tok->loc);
            else if (tok->token == "WorldEnd" && formatting)
                ;  // just swallow it
            else
                syntaxError(*tok);
            break;

        default:
            syntaxError(*tok);
        }
    }

    for (auto &import : imports) {
        import.first.join();

        ParsedScene *parsedScene = dynamic_cast<ParsedScene *>(scene);
        CHECK(parsedScene != nullptr);
        parsedScene->MergeImported(import.second);
        // HACK: let import.second leak so that its TransformCache isn't deallocated...
    }
}

void ParseFiles(SceneRepresentation *scene, pstd::span<const std::string> filenames) {
    auto tokError = [](const char *msg, const FileLoc *loc) {
        ErrorExit(loc, "%s", msg);
    };

    // Process scene description
    if (filenames.empty()) {
        // Parse scene from standard input
        std::unique_ptr<Tokenizer> t = Tokenizer::CreateFromFile("-", tokError);
        if (t)
            parse(scene, std::move(t));
    } else {
        // Parse scene from input files
        for (const std::string &fn : filenames) {
            if (fn != "-")
                SetSearchDirectory(fn);

            std::unique_ptr<Tokenizer> t = Tokenizer::CreateFromFile(fn, tokError);
            if (t)
                parse(scene, std::move(t));
        }
    }

    scene->EndOfFiles();
}

void ParseString(SceneRepresentation *scene, std::string str) {
    auto tokError = [](const char *msg, const FileLoc *loc) {
        ErrorExit(loc, "%s", msg);
    };
    std::unique_ptr<Tokenizer> t = Tokenizer::CreateFromString(std::move(str), tokError);
    if (!t)
        return;
    parse(scene, std::move(t));

    scene->EndOfFiles();
}

}  // namespace pbrt
