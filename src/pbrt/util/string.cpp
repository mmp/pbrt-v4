// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/string.h>

#include <ctype.h>
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

bool Atod(std::string_view str, double *ptr) {
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
        if (!Atod(strs[i], &doubles[i]))
            return {};
    return doubles;
}

}  // namespace pbrt
