// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/parser.h>

#include <pbrt/options.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/util/args.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/string.h>

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
    else if (!ints.empty())
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

ParserTarget::~ParserTarget() {}

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
        return std::make_unique<Tokenizer>(std::move(str), "<stdin>",
                                           std::move(errorCallback));
    }

    if (HasExtension(filename, ".gz")) {
        std::string str = ReadDecompressedFileContents(filename);
        return std::make_unique<Tokenizer>(std::move(str), filename,
                                           std::move(errorCallback));
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
    if (len < 16 * 1024 * 1024) {
        close(fd);

        std::string str = ReadFileContents(filename);
        return std::make_unique<Tokenizer>(std::move(str), filename,
                                           std::move(errorCallback));
    }

    void *ptr = mmap(nullptr, len, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, 0);
    if (ptr == MAP_FAILED)
        errorCallback(StringPrintf("%s: %s", filename, ErrorString()).c_str(), nullptr);

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

    HANDLE fileHandle =
        CreateFileW(WStringFromUTF8(filename).c_str(), GENERIC_READ, FILE_SHARE_READ, 0,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (fileHandle == INVALID_HANDLE_VALUE)
        return errorReportLambda();

    size_t len = GetFileSize(fileHandle, 0);

    HANDLE mapping = CreateFileMapping(fileHandle, 0, PAGE_READONLY, 0, 0, 0);
    CloseHandle(fileHandle);
    if (mapping == 0)
        return errorReportLambda();

    LPVOID ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    if (!ptr)
        return errorReportLambda();

    return std::make_unique<Tokenizer>(ptr, len, filename, std::move(errorCallback));
#else
    std::string str = ReadFileContents(filename);
    return std::make_unique<Tokenizer>(std::move(str), filename,
                                       std::move(errorCallback));
#endif
}

std::unique_ptr<Tokenizer> Tokenizer::CreateFromString(
    std::string str, std::function<void(const char *, const FileLoc *)> errorCallback) {
    return std::make_unique<Tokenizer>(std::move(str), "<stdin>",
                                       std::move(errorCallback));
}

Tokenizer::Tokenizer(std::string str, std::string filename,
                     std::function<void(const char *, const FileLoc *)> errorCallback)
    : errorCallback(std::move(errorCallback)), contents(std::move(str)) {
    loc = FileLoc(*new std::string(filename));
    pos = contents.data();
    end = pos + contents.size();
    tokenizerMemory += contents.size();
    CheckUTF(contents.data(), contents.size());
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
    CheckUTF(ptr, len);
}
#endif

Tokenizer::~Tokenizer() {
#ifdef PBRT_HAVE_MMAP
    if (unmapPtr && unmapLength > 0)
        if (munmap(unmapPtr, unmapLength) != 0)
            errorCallback(StringPrintf("munmap: %s", ErrorString()).c_str(), nullptr);
#elif defined(PBRT_IS_WINDOWS)
    if (unmapPtr && UnmapViewOfFile(unmapPtr) == 0)
        errorCallback(StringPrintf("UnmapViewOfFile: %s", ErrorString()).c_str(),
                      nullptr);
#endif
}

void Tokenizer::CheckUTF(const void *ptr, int len) const {
    const unsigned char *c = (const unsigned char *)ptr;
    // https://en.wikipedia.org/wiki/Byte_order_mark
    if (len >= 2 && ((c[0] == 0xfe && c[1] == 0xff) || (c[0] == 0xff && c[1] == 0xfe)))
        errorCallback("File is encoded with UTF-16, which is not currently "
                      "supported by pbrt (https://github.com/mmp/pbrt-v4/issues/136).",
                      &loc);
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

void parse(ParserTarget *target, std::unique_ptr<Tokenizer> t) {
    FormattingParserTarget *formattingTarget =
        dynamic_cast<FormattingParserTarget *>(target);
    bool formatting = formattingTarget;

    static std::atomic<bool> warnedTransformBeginEndDeprecated{false};

    std::vector<std::pair<AsyncJob<int> *, BasicSceneBuilder *>> imports;

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
                printf("%s%s\n",
                       dynamic_cast<FormattingParserTarget *>(target)->indent().c_str(),
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
    auto basicParamListEntrypoint =
        [&](void (ParserTarget::*apiFunc)(const std::string &, ParsedParameterVector,
                                          FileLoc),
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
            (target->*apiFunc)(n, std::move(parameterVector), loc);
        };

    auto syntaxError = [&](const Token &t) {
        ErrorExit(&t.loc, "Unknown directive: %s", toString(t.token));
    };

    pstd::optional<Token> tok;

    while (true) {
        tok = nextToken(TokenOptional);
        if (!tok.has_value())
            break;

        switch (tok->token[0]) {
        case 'A':
            if (tok->token == "AttributeBegin")
                target->AttributeBegin(tok->loc);
            else if (tok->token == "AttributeEnd")
                target->AttributeEnd(tok->loc);
            else if (tok->token == "Attribute")
                basicParamListEntrypoint(&ParserTarget::Attribute, tok->loc);
            else if (tok->token == "ActiveTransform") {
                Token a = *nextToken(TokenRequired);
                if (a.token == "All")
                    target->ActiveTransformAll(tok->loc);
                else if (a.token == "EndTime")
                    target->ActiveTransformEndTime(tok->loc);
                else if (a.token == "StartTime")
                    target->ActiveTransformStartTime(tok->loc);
                else
                    syntaxError(*tok);
            } else if (tok->token == "AreaLightSource")
                basicParamListEntrypoint(&ParserTarget::AreaLightSource, tok->loc);
            else if (tok->token == "Accelerator")
                basicParamListEntrypoint(&ParserTarget::Accelerator, tok->loc);
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
                target->ConcatTransform(m, tok->loc);
            } else if (tok->token == "CoordinateSystem") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                target->CoordinateSystem(toString(n), tok->loc);
            } else if (tok->token == "CoordSysTransform") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                target->CoordSysTransform(toString(n), tok->loc);
            } else if (tok->token == "ColorSpace") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                target->ColorSpace(toString(n), tok->loc);
            } else if (tok->token == "Camera")
                basicParamListEntrypoint(&ParserTarget::Camera, tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'F':
            if (tok->token == "Film")
                basicParamListEntrypoint(&ParserTarget::Film, tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'I':
            if (tok->token == "Integrator")
                basicParamListEntrypoint(&ParserTarget::Integrator, tok->loc);
            else if (tok->token == "Include") {
                Token filenameToken = *nextToken(TokenRequired);
                std::string filename = toString(dequoteString(filenameToken));
                if (formatting)
                    Printf("%sInclude \"%s\"\n",
                           dynamic_cast<FormattingParserTarget *>(target)->indent(),
                           filename);
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
                           dynamic_cast<FormattingParserTarget *>(target)->indent(),
                           filename);
                else {
                    BasicSceneBuilder *builder =
                        dynamic_cast<BasicSceneBuilder *>(target);
                    CHECK(builder);

                    if (builder->currentBlock !=
                        BasicSceneBuilder::BlockState::WorldBlock)
                        ErrorExit(&tok->loc, "Import statement only allowed inside world "
                                             "definition block.");

                    filename = ResolveFilename(filename);
                    BasicSceneBuilder *importBuilder = builder->CopyForImport();

                    if (RunningThreads() == 1) {
                        std::unique_ptr<Tokenizer> timport =
                            Tokenizer::CreateFromFile(filename, parseError);
                        if (timport)
                            parse(importBuilder, std::move(timport));
                        builder->MergeImported(importBuilder);
                    } else {
                        auto job = [=](std::string filename) {
                            Timer timer;
                            std::unique_ptr<Tokenizer> timport =
                                Tokenizer::CreateFromFile(filename, parseError);
                            if (timport)
                                parse(importBuilder, std::move(timport));
                            LOG_VERBOSE("Elapsed time to parse \"%s\": %.2fs", filename,
                                        timer.ElapsedSeconds());
                            return 0;
                        };
                        AsyncJob<int> *jobFinished = RunAsync(job, filename);
                        imports.push_back(std::make_pair(jobFinished, importBuilder));
                    }
                }
            } else if (tok->token == "Identity")
                target->Identity(tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'L':
            if (tok->token == "LightSource")
                basicParamListEntrypoint(&ParserTarget::LightSource, tok->loc);
            else if (tok->token == "LookAt") {
                Float v[9];
                for (int i = 0; i < 9; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                target->LookAt(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                               tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'M':
            if (tok->token == "MakeNamedMaterial")
                basicParamListEntrypoint(&ParserTarget::MakeNamedMaterial, tok->loc);
            else if (tok->token == "MakeNamedMedium")
                basicParamListEntrypoint(&ParserTarget::MakeNamedMedium, tok->loc);
            else if (tok->token == "Material")
                basicParamListEntrypoint(&ParserTarget::Material, tok->loc);
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

                target->MediumInterface(names[0], names[1], tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'N':
            if (tok->token == "NamedMaterial") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                target->NamedMaterial(toString(n), tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'O':
            if (tok->token == "ObjectBegin") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                target->ObjectBegin(toString(n), tok->loc);
            } else if (tok->token == "ObjectEnd")
                target->ObjectEnd(tok->loc);
            else if (tok->token == "ObjectInstance") {
                std::string_view n = dequoteString(*nextToken(TokenRequired));
                target->ObjectInstance(toString(n), tok->loc);
            } else if (tok->token == "Option") {
                std::string name = toString(dequoteString(*nextToken(TokenRequired)));
                std::string value = toString(nextToken(TokenRequired)->token);
                target->Option(name, value, tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'P':
            if (tok->token == "PixelFilter")
                basicParamListEntrypoint(&ParserTarget::PixelFilter, tok->loc);
            else
                syntaxError(*tok);
            break;

        case 'R':
            if (tok->token == "ReverseOrientation")
                target->ReverseOrientation(tok->loc);
            else if (tok->token == "Rotate") {
                Float v[4];
                for (int i = 0; i < 4; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                target->Rotate(v[0], v[1], v[2], v[3], tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'S':
            if (tok->token == "Shape")
                basicParamListEntrypoint(&ParserTarget::Shape, tok->loc);
            else if (tok->token == "Sampler")
                basicParamListEntrypoint(&ParserTarget::Sampler, tok->loc);
            else if (tok->token == "Scale") {
                Float v[3];
                for (int i = 0; i < 3; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                target->Scale(v[0], v[1], v[2], tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'T':
            if (tok->token == "TransformBegin") {
                if (formattingTarget)
                    formattingTarget->TransformBegin(tok->loc);
                else {
                    if (!warnedTransformBeginEndDeprecated) {
                        Warning(&tok->loc, "TransformBegin/End are deprecated and should "
                                           "be replaced with AttributeBegin/End");
                        warnedTransformBeginEndDeprecated = true;
                    }
                    target->AttributeBegin(tok->loc);
                }
            } else if (tok->token == "TransformEnd") {
                if (formattingTarget)
                    formattingTarget->TransformEnd(tok->loc);
                else
                    target->AttributeEnd(tok->loc);
            } else if (tok->token == "Transform") {
                if (nextToken(TokenRequired)->token != "[")
                    syntaxError(*tok);
                Float m[16];
                for (int i = 0; i < 16; ++i)
                    m[i] = parseFloat(*nextToken(TokenRequired));
                if (nextToken(TokenRequired)->token != "]")
                    syntaxError(*tok);
                target->Transform(m, tok->loc);
            } else if (tok->token == "Translate") {
                Float v[3];
                for (int i = 0; i < 3; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                target->Translate(v[0], v[1], v[2], tok->loc);
            } else if (tok->token == "TransformTimes") {
                Float v[2];
                for (int i = 0; i < 2; ++i)
                    v[i] = parseFloat(*nextToken(TokenRequired));
                target->TransformTimes(v[0], v[1], tok->loc);
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

                target->Texture(name, type, texName, std::move(params), tok->loc);
            } else
                syntaxError(*tok);
            break;

        case 'W':
            if (tok->token == "WorldBegin")
                target->WorldBegin(tok->loc);
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
        import.first->Wait();

        BasicSceneBuilder *builder = dynamic_cast<BasicSceneBuilder *>(target);
        CHECK(builder);
        builder->MergeImported(import.second);
        // HACK: let import.second leak so that its TransformCache isn't deallocated...
    }
}

void ParseFiles(ParserTarget *target, pstd::span<const std::string> filenames) {
    auto tokError = [](const char *msg, const FileLoc *loc) {
        ErrorExit(loc, "%s", msg);
    };

    // Process scene description
    if (filenames.empty()) {
        // Parse scene from standard input
        std::unique_ptr<Tokenizer> t = Tokenizer::CreateFromFile("-", tokError);
        if (t)
            parse(target, std::move(t));
    } else {
        // Parse scene from input files
        for (const std::string &fn : filenames) {
            if (fn != "-")
                SetSearchDirectory(fn);

            std::unique_ptr<Tokenizer> t = Tokenizer::CreateFromFile(fn, tokError);
            if (t)
                parse(target, std::move(t));
        }
    }

    target->EndOfFiles();
}

void ParseString(ParserTarget *target, std::string str) {
    auto tokError = [](const char *msg, const FileLoc *loc) {
        ErrorExit(loc, "%s", msg);
    };
    std::unique_ptr<Tokenizer> t = Tokenizer::CreateFromString(std::move(str), tokError);
    if (!t)
        return;
    parse(target, std::move(t));

    target->EndOfFiles();
}

// FormattingParserTarget Method Definitions
FormattingParserTarget::~FormattingParserTarget() {
    if (errorExit)
        ErrorExit("Fatal errors during scene updating.");
}

void FormattingParserTarget::Option(const std::string &name, const std::string &value,
                                    FileLoc loc) {
    std::string nName = normalizeArg(name);
    Printf("%sOption \"%s\" %s\n", indent(), name, value);
}

void FormattingParserTarget::Identity(FileLoc loc) {
    Printf("%sIdentity\n", indent());
}

void FormattingParserTarget::Translate(Float dx, Float dy, Float dz, FileLoc loc) {
    Printf("%sTranslate %f %f %f\n", indent(), dx, dy, dz);
}

void FormattingParserTarget::Rotate(Float angle, Float ax, Float ay, Float az,
                                    FileLoc loc) {
    Printf("%sRotate %f %f %f %f\n", indent(), angle, ax, ay, az);
}

void FormattingParserTarget::Scale(Float sx, Float sy, Float sz, FileLoc loc) {
    Printf("%sScale %f %f %f\n", indent(), sx, sy, sz);
}

void FormattingParserTarget::LookAt(Float ex, Float ey, Float ez, Float lx, Float ly,
                                    Float lz, Float ux, Float uy, Float uz, FileLoc loc) {
    Printf("%sLookAt %f %f %f\n%s    %f %f %f\n%s    %f %f %f\n", indent(), ex, ey, ez,
           indent(), lx, ly, lz, indent(), ux, uy, uz);
}

void FormattingParserTarget::ConcatTransform(Float transform[16], FileLoc loc) {
    Printf("%sConcatTransform [ ", indent());
    for (int i = 0; i < 16; ++i)
        Printf("%f ", transform[i]);
    Printf(" ]\n");
}

void FormattingParserTarget::Transform(Float transform[16], FileLoc loc) {
    Printf("%sTransform [ ", indent());
    for (int i = 0; i < 16; ++i)
        Printf("%f ", transform[i]);
    Printf(" ]\n");
}

void FormattingParserTarget::CoordinateSystem(const std::string &name, FileLoc loc) {
    Printf("%sCoordinateSystem \"%s\"\n", indent(), name);
}

void FormattingParserTarget::CoordSysTransform(const std::string &name, FileLoc loc) {
    Printf("%sCoordSysTransform \"%s\"\n", indent(), name);
}

void FormattingParserTarget::ActiveTransformAll(FileLoc loc) {
    Printf("%sActiveTransform All\n", indent());
}

void FormattingParserTarget::ActiveTransformEndTime(FileLoc loc) {
    Printf("%sActiveTransform EndTime\n", indent());
}

void FormattingParserTarget::ActiveTransformStartTime(FileLoc loc) {
    Printf("%sActiveTransform StartTime\n", indent());
}

void FormattingParserTarget::TransformTimes(Float start, Float end, FileLoc loc) {
    Printf("%sTransformTimes %f %f\n", indent(), start, end);
}

void FormattingParserTarget::ColorSpace(const std::string &n, FileLoc loc) {
    Printf("%sColorSpace \"%s\"\n", indent(), n);
}

void FormattingParserTarget::PixelFilter(const std::string &name,
                                         ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    std::string extra;
    if (upgrade) {
        std::vector<Float> xr = dict.GetFloatArray("xwidth");
        if (xr.size() == 1) {
            dict.RemoveFloat("xwidth");
            extra += StringPrintf("%s\"float xradius\" [ %f ]\n", indent(1), xr[0]);
        }
        std::vector<Float> yr = dict.GetFloatArray("ywidth");
        if (yr.size() == 1) {
            dict.RemoveFloat("ywidth");
            extra += StringPrintf("%s\"float yradius\" [ %f ]\n", indent(1), yr[0]);
        }

        if (name == "gaussian") {
            std::vector<Float> alpha = dict.GetFloatArray("alpha");
            if (alpha.size() == 1) {
                dict.RemoveFloat("alpha");
                extra += StringPrintf("%s\"float sigma\" [ %f ]\n", indent(1),
                                      1 / std::sqrt(2 * alpha[0]));
            }
        }
    }

    Printf("%sPixelFilter \"%s\"\n", indent(), name);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::Film(const std::string &type, ParsedParameterVector params,
                                  FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    std::string extra;
    if (upgrade) {
        std::vector<Float> m = dict.GetFloatArray("maxsampleluminance");
        if (!m.empty()) {
            dict.RemoveFloat("maxsampleluminance");
            extra +=
                StringPrintf("%s\"float maxcomponentvalue\" [ %f ]\n", indent(1), m[0]);
        }
        std::vector<Float> s = dict.GetFloatArray("scale");
        if (!s.empty()) {
            dict.RemoveFloat("scale");
            extra += StringPrintf("%s\"float iso\" [ %f ]\n", indent(1), 100 * s[0]);
        }
    }

    if (upgrade && type == "image")
        Printf("%sFilm \"rgb\"\n", indent());
    else
        Printf("%sFilm \"%s\"\n", indent(), type);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::Sampler(const std::string &name,
                                     ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    if (upgrade) {
        if (name == "lowdiscrepancy" || name == "02sequence")
            Printf("%sSampler \"paddedsobol\"\n", indent());
        else if (name == "maxmindist")
            Printf("%sSampler \"pmj02bn\"\n", indent());
        else if (name == "random")
            Printf("%sSampler \"independent\"\n", indent());
        else
            Printf("%sSampler \"%s\"\n", indent(), name);
    } else
        Printf("%sSampler \"%s\"\n", indent(), name);
    std::cout << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::Accelerator(const std::string &name,
                                         ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    Printf("%sAccelerator \"%s\"\n%s", indent(), name,
           dict.ToParameterList(catIndentCount));
}

void FormattingParserTarget::Integrator(const std::string &name,
                                        ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    std::string extra;
    if (upgrade) {
        dict.RemoveFloat("rrthreshold");

        if (name == "sppm") {
            dict.RemoveInt("imagewritefrequency");

            std::vector<int> iterations = dict.GetIntArray("numiterations");
            if (!iterations.empty()) {
                dict.RemoveInt("numiterations");
                Warning(
                    &loc,
                    "The SPPM integrator no longer takes a \"numiterations\" parameter. "
                    "This value is now set via the Sampler's number of pixel samples.");
            }
        }
        std::string lss = dict.GetOneString("lightsamplestrategy", "");
        if (lss == "spatial") {
            dict.RemoveString("lightsamplestrategy");
            extra += indent(1) + "\"string lightsamplestrategy\" \"bvh\"\n";
        }
    }

    if (upgrade && name == "directlighting") {
        Printf("%sIntegrator \"path\"\n", indent());
        extra += indent(1) + "\"integer maxdepth\" [ 1 ]\n";
    } else
        Printf("%sIntegrator \"%s\"\n", indent(), name);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::Camera(const std::string &name, ParsedParameterVector params,
                                    FileLoc loc) {
    ParameterDictionary dict(std::move(params), RGBColorSpace::sRGB);

    if (upgrade && name == "environment")
        Printf("%sCamera \"spherical\" \"string mapping\" \"equirectangular\"\n",
               indent());
    else
        Printf("%sCamera \"%s\"\n", indent(), name);
    if (upgrade && name == "realistic")
        dict.RemoveBool("simpleweighting");

    std::cout << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::MakeNamedMedium(const std::string &name,
                                             ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    if (upgrade && name == "heterogeneous")
        Printf("%sMakeNamedMedium \"%s\"\n%s\n", indent(), "uniformgrid",
               dict.ToParameterList(catIndentCount));
    else
        Printf("%sMakeNamedMedium \"%s\"\n%s\n", indent(), name,
               dict.ToParameterList(catIndentCount));
}

void FormattingParserTarget::MediumInterface(const std::string &insideName,
                                             const std::string &outsideName,
                                             FileLoc loc) {
    Printf("%sMediumInterface \"%s\" \"%s\"\n", indent(), insideName, outsideName);
}

void FormattingParserTarget::WorldBegin(FileLoc loc) {
    Printf("\n\nWorldBegin\n\n");
}

void FormattingParserTarget::AttributeBegin(FileLoc loc) {
    Printf("\n%sAttributeBegin\n", indent());
    catIndentCount += 4;
}

void FormattingParserTarget::AttributeEnd(FileLoc loc) {
    catIndentCount -= 4;
    Printf("%sAttributeEnd\n", indent());
}

void FormattingParserTarget::Attribute(const std::string &target,
                                       ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    Printf("%sAttribute \"%s\" ", indent(), target);
    if (params.size() == 1)
        // just one; put it on the same line
        std::cout << dict.ToParameterList(0) << '\n';
    else
        std::cout << '\n' << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::TransformBegin(FileLoc loc) {
    Warning(&loc, "Rewriting \"TransformBegin\" to \"AttributeBegin\".");
    Printf("%sAttributeBegin\n", indent());
    catIndentCount += 4;
}

void FormattingParserTarget::TransformEnd(FileLoc loc) {
    catIndentCount -= 4;
    Warning(&loc, "Rewriting \"TransformEnd\" to \"AttributeEnd\".");
    Printf("%sAttributeEnd\n", indent());
}

void FormattingParserTarget::Texture(const std::string &name, const std::string &type,
                                     const std::string &texname,
                                     ParsedParameterVector params, FileLoc loc) {
    if (upgrade) {
        if (definedTextures.find(name) != definedTextures.end()) {
            static int count = 0;
            Warning(&loc, "%s: renaming multiply-defined texture", name);
            definedTextures[name] = StringPrintf("%s-renamed-%d", name, count++);
        } else
            definedTextures[name] = name;
    }

    if (upgrade && texname == "scale") {
        // This is easier to do in the raw ParsedParameterVector...
        if (type == "float") {
            for (ParsedParameter *p : params) {
                if (p->name == "tex1")
                    p->name = "tex";
                if (p->name == "tex2")
                    p->name = "scale";
            }
        } else {
            // more subtle: rename one of them as float, but need one of them
            // to be an RGB and spectrally constant...
            bool foundRGB = false, foundTexture = false;
            for (ParsedParameter *p : params) {
                if (p->name != "tex1" && p->name != "tex2")
                    continue;

                if (p->type == "rgb") {
                    if (foundRGB) {
                        ErrorExitDeferred(
                            &p->loc,
                            "Two \"rgb\" textures found for \"scale\" "
                            "texture \"%s\". Please manually edit the file to "
                            "upgrade.",
                            name);
                        return;
                    }
                    if (p->floats.size() != 3) {
                        ErrorExitDeferred(
                            &p->loc, "Didn't find 3 values for \"rgb\" \"%s\".", p->name);
                        return;
                    }
                    if (p->floats[0] != p->floats[1] || p->floats[1] != p->floats[2]) {
                        ErrorExitDeferred(&p->loc,
                                          "Non-constant \"rgb\" value found for "
                                          "\"scale\" texture parameter \"%s\". Please "
                                          "manually "
                                          "edit the file to upgrade.",
                                          p->name);
                        return;
                    }

                    foundRGB = true;
                    p->type = "float";
                    p->name = "scale";
                    p->floats.resize(1);
                } else {
                    if (foundTexture) {
                        ErrorExitDeferred(
                            &p->loc,
                            "Two textures found for \"scale\" "
                            "texture \"%s\". Please manually edit the file to "
                            "upgrade.",
                            name);
                        return;
                    }
                    p->name = "tex";
                    foundTexture = true;
                }
            }
        }
    }

    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    if (upgrade)
        dict.RenameUsedTextures(definedTextures);

    std::string extra;
    if (upgrade) {
        if (texname == "imagemap") {
            std::vector<uint8_t> tri = dict.GetBoolArray("trilinear");
            if (tri.size() == 1) {
                dict.RemoveBool("trilinear");
                extra += indent(1) + "\"string filter\" ";
                extra += tri[0] != 0u ? "\"trilinear\"\n" : "\"bilinear\"\n";
            }
        }

        if (texname == "imagemap" || texname == "ptex") {
            Float gamma = dict.GetOneFloat("gamma", 0);
            if (gamma != 0) {
                dict.RemoveFloat("gamma");
                extra +=
                    indent(1) + StringPrintf("\"string encoding \" \"gamma %f\" ", gamma);
            } else {
                std::vector<uint8_t> gamma = dict.GetBoolArray("gamma");
                if (gamma.size() == 1) {
                    dict.RemoveBool("gamma");
                    extra += indent(1) + "\"string encoding\" ";
                    extra += gamma[0] != 0u ? "\"sRGB\"\n" : "\"linear\"\n";
                }
            }
        }
    }

    if (upgrade) {
        if (type == "color")
            Printf("%sTexture \"%s\" \"spectrum\" \"%s\"\n", indent(),
                   definedTextures[name], texname);
        else
            Printf("%sTexture \"%s\" \"%s\" \"%s\"\n", indent(), definedTextures[name],
                   type, texname);
    } else
        Printf("%sTexture \"%s\" \"%s\" \"%s\"\n", indent(), name, type, texname);

    std::cout << extra << dict.ToParameterList(catIndentCount);
}

std::string FormattingParserTarget::upgradeMaterialIndex(const std::string &name,
                                                         ParameterDictionary *dict,
                                                         FileLoc loc) const {
    if (name != "glass" && name != "uber")
        return "";

    std::string tex = dict->GetTexture("index");
    if (!tex.empty()) {
        if (!dict->GetTexture("eta").empty()) {
            ErrorExitDeferred(
                &loc, R"(Material "%s" has both "index" and "eta" parameters.)", name);
            return "";
        }

        dict->RemoveTexture("index");
        return indent(1) + StringPrintf("\"texture eta\" \"%s\"\n", tex);
    } else {
        auto index = dict->GetFloatArray("index");
        if (index.empty())
            return "";
        if (index.size() != 1) {
            ErrorExitDeferred(&loc, "Multiple values provided for \"index\" parameter.");
            return "";
        }
        if (!dict->GetFloatArray("eta").empty()) {
            ErrorExitDeferred(
                &loc, R"(Material "%s" has both "index" and "eta" parameters.)", name);
            return "";
        }

        Float value = index[0];
        dict->RemoveFloat("index");
        return indent(1) + StringPrintf("\"float eta\" [ %f ]\n", value);
    }
}

std::string FormattingParserTarget::upgradeMaterial(std::string *name,
                                                    ParameterDictionary *dict,
                                                    FileLoc loc) const {
    std::string extra = upgradeMaterialIndex(*name, dict, loc);

    dict->RenameParameter("bumpmap", "displacement");

    auto removeParamSilentIfConstant = [&](const char *paramName, Float value) {
        pstd::optional<RGB> rgb = dict->GetOneRGB(paramName);
        bool matches = (rgb && rgb->r == value && rgb->g == value && rgb->b == value);

        if (!matches &&
            !dict->GetSpectrumArray(paramName, SpectrumType::Unbounded, {}).empty())
            Warning(&loc,
                    "Parameter is being removed when converting "
                    "to \"%s\" material: %s",
                    *name, dict->ToParameterDefinition(paramName));
        dict->RemoveSpectrum(paramName);
        dict->RemoveTexture(paramName);
        return matches;
    };

    if (*name == "uber") {
        *name = "coateddiffuse";
        if (removeParamSilentIfConstant("Ks", 0)) {
            *name = "diffuse";
            dict->RemoveFloat("eta");
            dict->RemoveFloat("roughness");
        }
        removeParamSilentIfConstant("Kr", 0);
        removeParamSilentIfConstant("Kt", 0);
        dict->RenameParameter("Kd", "reflectance");

        if (!dict->GetTexture("opacity").empty()) {
            ErrorExitDeferred(&loc, "Non-opaque \"opacity\" texture in \"uber\" "
                                    "material not supported "
                                    "in pbrt-v4. Please edit the file manually.");
            return "";
        }

        if (dict->GetSpectrumArray("opacity", SpectrumType::Unbounded, {}).empty())
            return "";

        pstd::optional<RGB> opacity = dict->GetOneRGB("opacity");
        if (opacity && opacity->r == 1 && opacity->g == 1 && opacity->b == 1) {
            dict->RemoveSpectrum("opacity");
            return "";
        }

        ErrorExitDeferred(&loc, "A non-opaque \"opacity\" in the \"uber\" "
                                "material is not supported "
                                "in pbrt-v4. Please edit the file manually.");
    } else if (*name == "mix") {
        // Convert the amount to a scalar
        pstd::optional<RGB> rgb = dict->GetOneRGB("amount");
        if (rgb) {
            if (rgb->r == rgb->g && rgb->g == rgb->b)
                extra += indent(1) + StringPrintf("\"float amount\" [ %f ]\n", rgb->r);
            else {
                Float avg = (rgb->r + rgb->g + rgb->b) / 3;
                Warning(&loc, "Changing RGB \"amount\" (%f, %f, %f) to scalar average %f",
                        rgb->r, rgb->g, rgb->b, avg);
                extra += indent(1) + StringPrintf("\"float amount\" [ %f ]\n", avg);
            }
        } else if (dict->GetSpectrumArray("amount", SpectrumType::Unbounded, {}).size() >
                   0)
            ErrorExitDeferred(
                &loc, "Unable to update non-RGB spectrum \"amount\" to a scalar: %s",
                dict->ToParameterDefinition("amount"));

        dict->RemoveSpectrum("amount");

        // And rename...
        std::string m1 = dict->GetOneString("namedmaterial1", "");
        if (m1.empty())
            ErrorExitDeferred(
                &loc, "Didn't find \"namedmaterial1\" parameter for \"mix\" material.");
        dict->RemoveString("namedmaterial1");

        std::string m2 = dict->GetOneString("namedmaterial2", "");
        if (m2.empty())
            ErrorExitDeferred(
                &loc, "Didn't find \"namedmaterial2\" parameter for \"mix\" material.");
        dict->RemoveString("namedmaterial2");

        // Note: swapped order vs pbrt-v3!
        extra +=
            indent(1) + StringPrintf("\"string materials\" [ \"%s\" \"%s\" ]\n", m2, m1);
    } else if (*name == "substrate") {
        *name = "coateddiffuse";
        removeParamSilentIfConstant("Ks", 1);
        dict->RenameParameter("Kd", "reflectance");
    } else if (*name == "glass") {
        *name = "dielectric";
        removeParamSilentIfConstant("Kr", 1);
        removeParamSilentIfConstant("Kt", 1);
    } else if (*name == "plastic") {
        *name = "coateddiffuse";
        if (removeParamSilentIfConstant("Ks", 0)) {
            *name = "diffuse";
            dict->RemoveFloat("roughness");
            dict->RemoveFloat("eta");
        }
        dict->RenameParameter("Kd", "reflectance");
    } else if (*name == "fourier")
        Warning(&loc, "\"fourier\" material is no longer supported. (But there "
                      "is \"measured\"!)");
    else if (*name == "kdsubsurface") {
        *name = "subsurface";
        dict->RenameParameter("Kd", "reflectance");
    } else if (*name == "matte") {
        *name = "diffuse";
        dict->RenameParameter("Kd", "reflectance");
        dict->RemoveFloat("sigma");
        dict->RemoveTexture("sigma");
    } else if (*name == "metal") {
        *name = "conductor";
        removeParamSilentIfConstant("Kr", 1);
    } else if (*name == "translucent") {
        *name = "diffusetransmission";

        dict->RenameParameter("Kd", "transmittance");

        removeParamSilentIfConstant("reflect", 0);
        removeParamSilentIfConstant("transmit", 1);

        removeParamSilentIfConstant("Ks", 0);
        dict->RemoveFloat("roughness");
    } else if (*name == "mirror") {
        *name = "conductor";
        extra += indent(1) + "\"float roughness\" [ 0 ]\n";
        extra += indent(1) + "\"spectrum eta\" [ \"metal-Ag-eta\" ]\n";
        extra += indent(1) + "\"spectrum k\" [ \"metal-Ag-k\" ]\n";

        removeParamSilentIfConstant("Kr", 0);
    } else if (*name == "disney") {
        *name = "diffuse";
        dict->RenameParameter("color", "reflectance");
    } else if (*name == "hair") {
        dict->RenameParameter("color", "reflectance");
    } else if (name->empty() || *name == "none")
        *name = "interface";

    return extra;
}

void FormattingParserTarget::Material(const std::string &name,
                                      ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    if (upgrade)
        dict.RenameUsedTextures(definedTextures);

    std::string extra;
    std::string newName = name;
    if (upgrade)
        extra = upgradeMaterial(&newName, &dict, loc);

#if 0
    // Hack for landscape upgrade
    if (upgrade && name == "mix") {
        ParameterDictionary dict(params, RGBColorSpace::sRGB);
        const ParameterDictionary &d1 = namedMaterialDictionaries[dict.GetOneString("namedmaterial1", "")];
        const ParameterDictionary &d2 = namedMaterialDictionaries[dict.GetOneString("namedmaterial2", "")];

        if (!d1.GetTexture("reflectance").empty() &&
            !d2.GetTexture("transmittance").empty()) {
            Printf("%sMaterial \"diffusetransmission\"\n", indent());
            Printf("%s\"texture reflectance\" \"%s\"\n", indent(1), d1.GetTexture("reflectance"));
            Printf("%s\"texture transmittance\" \"%s\"\n", indent(1), d2.GetTexture("transmittance"));

            if (!d1.GetTexture("displacement").empty())
                Printf("%s\"texture displacement\" \"%s\"\n", indent(1), d1.GetTexture("displacement"));
            else if (!d2.GetTexture("displacement").empty())
                Printf("%s\"texture displacement\" \"%s\"\n", indent(1), d2.GetTexture("displacement"));

            Printf("%s\"float scale\" 0.5\n", indent(1));

            return;
        }
    }
#endif

    Printf("%sMaterial \"%s\"\n", indent(), newName);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::MakeNamedMaterial(const std::string &name,
                                               ParsedParameterVector params,
                                               FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);

    if (upgrade) {
        dict.RenameUsedTextures(definedTextures);

        if (definedNamedMaterials.find(name) != definedNamedMaterials.end()) {
            static int count = 0;
            Warning(&loc, "%s: renaming multiply-defined named material", name);
            definedNamedMaterials[name] = StringPrintf("%s-renamed-%d", name, count++);
        } else
            definedNamedMaterials[name] = name;
        Printf("%sMakeNamedMaterial \"%s\"\n", indent(), definedNamedMaterials[name]);
    } else
        Printf("%sMakeNamedMaterial \"%s\"\n", indent(), name);

    std::string extra;
    if (upgrade) {
        std::string matName = dict.GetOneString("type", "");
        extra = upgradeMaterial(&matName, &dict, loc);
        dict.RemoveString("type");
        extra = indent(1) + StringPrintf("\"string type\" [ \"%s\" ]\n", matName) + extra;
    }
    std::cout << extra << dict.ToParameterList(catIndentCount);

    if (upgrade)
        namedMaterialDictionaries[definedNamedMaterials[name]] = std::move(dict);
}

void FormattingParserTarget::NamedMaterial(const std::string &name, FileLoc loc) {
    Printf("%sNamedMaterial \"%s\"\n", indent(), name);
}

static bool upgradeRGBToScale(ParameterDictionary *dict, const char *name,
                              Float *totalScale) {
    std::vector<Spectrum> s = dict->GetSpectrumArray(name, SpectrumType::Unbounded, {});
    if (s.empty())
        return true;

    pstd::optional<RGB> rgb = dict->GetOneRGB(name);
    if (!rgb || rgb->r != rgb->g || rgb->g != rgb->b)
        return false;

    *totalScale *= rgb->r;
    dict->RemoveSpectrum(name);
    return true;
}

static std::string upgradeMapname(const FormattingParserTarget &scene,
                                  ParameterDictionary *dict) {
    std::string n = dict->GetOneString("mapname", "");
    if (n.empty())
        return "";

    dict->RemoveString("mapname");
    return scene.indent(1) + StringPrintf("\"string filename\" \"%s\"\n", n);
}

void FormattingParserTarget::LightSource(const std::string &name,
                                         ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);

    Printf("%sLightSource \"%s\"\n", indent(), name);

    std::string extra;
    if (upgrade) {
        Float totalScale = 1;
        if (!upgradeRGBToScale(&dict, "scale", &totalScale)) {
            ErrorExitDeferred(dict.loc("scale"),
                              "In pbrt-v4, \"scale\" is a \"float\" parameter "
                              "to light sources. "
                              "Please modify your scene file manually.");
            return;
        }
        dict.RemoveInt("samples");

        if (dict.GetOneString("mapname", "").empty() == false) {
            if (name == "infinite" && !upgradeRGBToScale(&dict, "L", &totalScale)) {
                ErrorExitDeferred(
                    dict.loc("L"),
                    "Non-constant \"L\" is no longer supported with "
                    "\"mapname\" for "
                    "the \"infinite\" light source. Please upgrade your scene "
                    "file manually.");
                return;
            } else if (name == "projection" &&
                       !upgradeRGBToScale(&dict, "I", &totalScale)) {
                ErrorExitDeferred(
                    dict.loc("I"),
                    "\"I\" is no longer supported with \"mapname\" for "
                    "the \"projection\" light source. Please upgrade your scene "
                    "file manually.");
                return;
            }
        }

        totalScale *= dict.UpgradeBlackbody("I");
        totalScale *= dict.UpgradeBlackbody("L");

        // Do this after we've handled infinite "L" with a map, since
        // it removes the "mapname" parameter from the dictionary.
        extra += upgradeMapname(*this, &dict);

        if (totalScale != 1) {
            totalScale *= dict.GetOneFloat("scale", 1.f);
            dict.RemoveFloat("scale");
            Printf("%s\"float scale\" [%f]\n", indent(1), totalScale);
        }
    }

    std::cout << extra << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::AreaLightSource(const std::string &name,
                                             ParsedParameterVector params, FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);
    std::string extra;
    Float totalScale = 1;
    if (upgrade) {
        if (!upgradeRGBToScale(&dict, "scale", &totalScale)) {
            ErrorExitDeferred(dict.loc("scale"),
                              "In pbrt-v4, \"scale\" is a \"float\" parameter "
                              "to light sources. "
                              "Please modify your scene file manually.");
            return;
        }

        totalScale *= dict.UpgradeBlackbody("L");

        if (name == "area")
            Printf("%sAreaLightSource \"diffuse\"\n", indent());
        else
            Printf("%sAreaLightSource \"%s\"\n", indent(), name);
        dict.RemoveInt("nsamples");
    } else
        Printf("%sAreaLightSource \"%s\"\n", indent(), name);

    if (totalScale != 1)
        Printf("%s\"float scale\" [%f]\n", indent(1), totalScale);
    std::cout << extra << dict.ToParameterList(catIndentCount);
}

static std::string upgradeTriMeshUVs(const FormattingParserTarget &scene,
                                     ParameterDictionary *dict) {
    std::vector<Point2f> uv = dict->GetPoint2fArray("st");
    if (!uv.empty())
        dict->RemovePoint2f("st");
    else {
        auto upgradeFloatArray = [&](const char *name) {
            std::vector<Float> fuv = dict->GetFloatArray(name);
            if (fuv.empty())
                return;

            std::vector<Point2f> tempUVs;
            tempUVs.reserve(fuv.size() / 2);
            for (size_t i = 0; i < fuv.size() / 2; ++i)
                tempUVs.push_back(Point2f(fuv[2 * i], fuv[2 * i + 1]));
            dict->RemoveFloat(name);
            uv = tempUVs;
        };
        upgradeFloatArray("uv");
        upgradeFloatArray("st");
    }

    if (uv.empty())
        return "";

    std::string s = scene.indent(1) + "\"point2 uv\" [ ";
    for (size_t i = 0; i < uv.size(); ++i) {
        s += StringPrintf("%f %f ", uv[i][0], uv[i][1]);
        if ((i + 1) % 4 == 0) {
            s += "\n";
            s += scene.indent(2);
        }
    }
    s += "]\n";
    return s;
}

void FormattingParserTarget::Shape(const std::string &name, ParsedParameterVector params,
                                   FileLoc loc) {
    ParameterDictionary dict(params, RGBColorSpace::sRGB);

    if (toPly && name == "trianglemesh") {
        std::vector<int> vi = dict.GetIntArray("indices");

        if (vi.size() < 500) {
            // It's a small mesh; don't bother with a PLY file after all.
            Printf("%sShape \"%s\"\n", indent(), name);
            std::cout << dict.ToParameterList(catIndentCount);
        } else {
            static int count = 1;
            const char *plyPrefix = getenv("PLY_PREFIX") ? getenv("PLY_PREFIX") : "mesh";
            std::string fn = StringPrintf("%s_%05d.ply", plyPrefix, count++);

            class Transform identity;
            const TriangleMesh *mesh =
                Triangle::CreateMesh(&identity, false, dict, &loc, Allocator());
            if (!mesh->WritePLY(fn))
                ErrorExit(&loc, "%s: unable to write PLY file.", fn);

            dict.RemoveInt("indices");
            dict.RemovePoint3f("P");
            dict.RemovePoint2f("uv");
            dict.RemoveNormal3f("N");
            dict.RemoveVector3f("S");
            dict.RemoveInt("faceIndices");

            Printf("%sShape \"plymesh\" \"string filename\" \"%s\"\n", indent(), fn);
            std::cout << dict.ToParameterList(catIndentCount);
        }
        return;
    }

    Printf("%sShape \"%s\"\n", indent(), name);

    if (upgrade) {
        if (name == "trianglemesh") {
            // Remove indices if they're [0 1 2] and we have a single triangle
            auto indices = dict.GetIntArray("indices");
            if (indices.size() == 3 && dict.GetPoint3fArray("P").size() == 3 &&
                indices[0] == 0 && indices[1] == 1 && indices[2] == 2)
                dict.RemoveInt("indices");
        }

        if (name == "bilinearmesh") {
            // Remove indices if they're [0 1 2 3] and we have a single blp
            auto indices = dict.GetIntArray("indices");
            if (indices.size() == 4 && dict.GetPoint3fArray("P").size() == 4 &&
                indices[0] == 0 && indices[1] == 1 && indices[2] == 2 && indices[3] == 3)
                dict.RemoveInt("indices");
        }

        if (name == "loopsubdiv") {
            auto levels = dict.GetIntArray("nlevels");
            if (!levels.empty()) {
                Printf("%s\"integer levels\" [ %d ]\n", indent(1), levels[0]);
                dict.RemoveInt("nlevels");
            }
        }
        if (name == "trianglemesh" || name == "plymesh") {
            dict.RemoveBool("discarddegenerateUVs");
            dict.RemoveTexture("shadowalpha");
        }

        if (name == "trianglemesh") {
            std::string extra = upgradeTriMeshUVs(*this, &dict);
            std::cout << extra;
        }

        dict.RenameParameter("Kd", "reflectance");
    }

    std::cout << dict.ToParameterList(catIndentCount);
}

void FormattingParserTarget::ReverseOrientation(FileLoc loc) {
    Printf("%sReverseOrientation\n", indent());
}

void FormattingParserTarget::ObjectBegin(const std::string &name, FileLoc loc) {
    if (upgrade) {
        if (definedObjectInstances.find(name) != definedObjectInstances.end()) {
            static int count = 0;
            Warning(&loc, "%s: renaming multiply-defined object instance", name);
            definedObjectInstances[name] = StringPrintf("%s-renamed-%d", name, count++);
        } else
            definedObjectInstances[name] = name;
        Printf("%sObjectBegin \"%s\"\n", indent(), definedObjectInstances[name]);
    } else
        Printf("%sObjectBegin \"%s\"\n", indent(), name);
}

void FormattingParserTarget::ObjectEnd(FileLoc loc) {
    Printf("%sObjectEnd\n", indent());
}

void FormattingParserTarget::ObjectInstance(const std::string &name, FileLoc loc) {
    if (upgrade) {
        if (definedObjectInstances.find(name) == definedObjectInstances.end())
            // this is legit if we're upgrading multiple files separately...
            Printf("%sObjectInstance \"%s\"\n", indent(), name);
        else
            // use the most recent renaming of it
            Printf("%sObjectInstance \"%s\"\n", indent(), definedObjectInstances[name]);
    } else
        Printf("%sObjectInstance \"%s\"\n", indent(), name);
}

void FormattingParserTarget::EndOfFiles() {}

}  // namespace pbrt
