// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifdef PBRT_IS_WINDOWS
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif  // PBRT_IS_WINDOWS

#include <pbrt/util/string.h>

#include <pbrt/util/check.h>

#include <ctype.h>
#include <codecvt>
#include <locale>
#include <string>

namespace pbrt {

bool Atoi(std::string_view str, int *ptr) {
    try {
        *ptr = std::stoi(std::string(str.begin(), str.end()));
    } catch (...) {
        return false;
    }
    return true;
}

bool Atof(std::string_view str, float *ptr) {
    try {
        *ptr = std::stof(std::string(str.begin(), str.end()));
    } catch (...) {
        return false;
    }
    return true;
}

bool Atof(std::string_view str, double *ptr) {
    try {
        *ptr = std::stod(std::string(str.begin(), str.end()));
    } catch (...) {
        return false;
    }
    return true;
}

std::vector<std::string> SplitStringsFromWhitespace(std::string_view str) {
    std::vector<std::string> ret;

    std::string_view::iterator start = str.begin();
    do {
        // skip leading ws
        while (start != str.end() && isspace(*start))
            ++start;

        // |start| is at the start of the current word
        auto end = start;
        while (end != str.end() && !isspace(*end))
            ++end;

        ret.push_back(std::string(start, end));
        start = end;
    } while (start != str.end());

    return ret;
}

std::vector<std::string> SplitString(std::string_view str, char ch) {
    std::vector<std::string> strings;

    if (str.empty())
        return strings;

    std::string_view::iterator begin = str.begin();
    while (true) {
        std::string_view::iterator end = begin;
        while (end != str.end() && *end != ch)
            ++end;

        strings.push_back(std::string(begin, end));

        if (end == str.end())
            break;

        begin = end + 1;
    }

    return strings;
}

std::vector<int> SplitStringToInts(std::string_view str, char ch) {
    std::vector<std::string> strs = SplitString(str, ch);
    std::vector<int> ints(strs.size());

    for (size_t i = 0; i < strs.size(); ++i)
        if (!Atoi(strs[i], &ints[i]))
            return {};
    return ints;
}

std::vector<Float> SplitStringToFloats(std::string_view str, char ch) {
    std::vector<std::string> strs = SplitString(str, ch);
    std::vector<Float> floats(strs.size());

    for (size_t i = 0; i < strs.size(); ++i)
        if (!Atof(strs[i], &floats[i]))
            return {};
    return floats;
}

std::vector<double> SplitStringToDoubles(std::string_view str, char ch) {
    std::vector<std::string> strs = SplitString(str, ch);
    std::vector<double> doubles(strs.size());

    for (size_t i = 0; i < strs.size(); ++i)
        if (!Atof(strs[i], &doubles[i]))
            return {};
    return doubles;
}

#ifdef PBRT_IS_WINDOWS
std::wstring WStringFromU16String(std::u16string str) {
    std::wstring ws;
    ws.reserve(str.size());
    for (char16_t c : str)
        ws.push_back(c);
    return ws;
}

std::wstring WStringFromUTF8(std::string str) {
    return WStringFromU16String(UTF16FromUTF8(str));
}

std::u16string U16StringFromWString(std::wstring str) {
    std::u16string su16;
    su16.reserve(str.size());
    for (wchar_t c : str)
        su16.push_back(c);
    return su16;
}

std::string UTF8FromWString(std::wstring str) {
    return UTF8FromUTF16(U16StringFromWString(str));
}

#endif  // PBRT_IS_WINDOWS

// https://stackoverflow.com/a/52703954
std::string UTF8FromUTF16(std::u16string str) {
    std::wstring_convert<
        std::codecvt_utf8_utf16<char16_t, 0x10ffff, std::codecvt_mode::little_endian>,
        char16_t>
        cnv;
    std::string utf8 = cnv.to_bytes(str);
    CHECK_GE(cnv.converted(), str.size());
    return utf8;
}

std::u16string UTF16FromUTF8(std::string str) {
    std::wstring_convert<
        std::codecvt_utf8_utf16<char16_t, 0x10ffff, std::codecvt_mode::little_endian>,
        char16_t>
        cnv;
    std::u16string utf16 = cnv.from_bytes(str);
    CHECK_GE(cnv.converted(), str.size());
    return utf16;
}

}  // namespace pbrt
