// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <gtest/gtest.h>

#include <pbrt/parser.h>
#include <pbrt/pbrt.h>
#include <pbrt/util/pstd.h>

#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

using namespace pbrt;

static std::string inTestDir(const std::string &path) {
    return path;
}

static std::vector<std::string> extract(Tokenizer *t) {
    std::vector<std::string> tokens;
    while (true) {
        pstd::optional<Token> tok = t->Next();
        if (!tok)
            return tokens;
        tokens.push_back(std::string(tok->token.begin(), tok->token.end()));
    }
}

static void checkTokens(Tokenizer *t, std::initializer_list<std::string> expected) {
    std::vector<std::string> tokens = extract(t);
    auto iter = expected.begin();
    for (const std::string &s : tokens) {
        EXPECT_TRUE(iter != expected.end());
        EXPECT_EQ(*iter, s);
        ++iter;
    }
    EXPECT_TRUE(iter == expected.end());
}

TEST(Parser, TokenizerBasics) {
    std::vector<std::string> errors;
    auto err = [&](const char *err, const FileLoc *) { errors.push_back(err); };

    {
        auto t =
            Tokenizer::CreateFromString("Shape \"sphere\" \"float radius\" [1]", err);
        ASSERT_TRUE(t.get() != nullptr);
        checkTokens(t.get(), {"Shape", "\"sphere\"", "\"float radius\"", "[", "1", "]"});
    }

    {
        auto t =
            Tokenizer::CreateFromString("Shape \"sphere\"\n\"float radius\" [1]", err);
        ASSERT_TRUE(t.get() != nullptr);
        checkTokens(t.get(), {"Shape", "\"sphere\"", "\"float radius\"", "[", "1", "]"});
    }

    {
        auto t = Tokenizer::CreateFromString(R"(
Shape"sphere" # foo bar [
"float radius\"" 1)",
                                             err);
        ASSERT_TRUE(t.get() != nullptr);
        checkTokens(t.get(),
                    {"Shape", "\"sphere\"", "# foo bar [", R"("float radius"")", "1"});
    }
}

TEST(Parser, TokenizerErrors) {
    {
        bool gotError = false;
        auto err = [&](const char *err, const FileLoc *) {
            gotError = !strcmp(err, "premature EOF");
        };
        auto t = Tokenizer::CreateFromString(
            "Shape\"sphere\"\t\t # foo bar\n\"float radius", err);
        ASSERT_TRUE(t.get() != nullptr);
        extract(t.get());
        EXPECT_TRUE(gotError);
    }

    {
        bool gotError = false;
        auto err = [&](const char *err, const FileLoc *) {
            gotError = !strcmp(err, "premature EOF");
        };
        auto t = Tokenizer::CreateFromString(
            "Shape\"sphere\"\t\t # foo bar\n\"float radius", err);
        ASSERT_TRUE(t.get() != nullptr);
        extract(t.get());
        EXPECT_TRUE(gotError);
    }

    {
        bool gotError = false;
        auto err = [&](const char *err, const FileLoc *) {
            gotError = !strcmp(err, "premature EOF");
        };
        auto t = Tokenizer::CreateFromString(
            "Shape\"sphere\"\t\t # foo bar\n\"float radius\\", err);
        ASSERT_TRUE(t.get() != nullptr);
        extract(t.get());
        EXPECT_TRUE(gotError);
    }

    {
        bool gotError = false;
        auto err = [&](const char *err, const FileLoc *) {
            gotError = !strcmp(err, "unterminated string");
        };
        auto t = Tokenizer::CreateFromString(
            "Shape\"sphere\"\t\t # foo bar\n\"float radius\n\" 5", err);
        ASSERT_TRUE(t.get() != nullptr);
        extract(t.get());
        EXPECT_TRUE(gotError);
    }
}

TEST(Parser, TokenizeFile) {
    std::string filename = inTestDir("test.tok");
    std::ofstream out(filename);
    out << R"(
WorldBegin # hello
Integrator "deep" "float density" [ 2 2.66612 -5e-51]
)";
    out.close();
    ASSERT_TRUE(out.good());

    auto err = [](const char *err, const FileLoc *) {
        EXPECT_TRUE(false) << "Unexpected error: " << err;
    };
    // Windows won't let us remove the file on disk if we hold on to a mapping
    // view. So enclose the tokenizer in a scope so that it releases any file
    // mapping view before the remove.
    {
        auto t = Tokenizer::CreateFromFile(filename, err);
        ASSERT_TRUE(t.get() != nullptr);
        checkTokens(t.get(), {"WorldBegin", "# hello", "Integrator", "\"deep\"",
                              "\"float density\"", "[", "2", "2.66612", "-5e-51", "]"});
    }

    EXPECT_EQ(0, remove(filename.c_str()));
}
