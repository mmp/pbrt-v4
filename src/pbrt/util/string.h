// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_STRING_H
#define PBRT_UTIL_STRING_H

#include <pbrt/pbrt.h>

#include <string>
#include <string_view>
#include <vector>

namespace pbrt {

bool Atoi(std::string_view str, int *);
bool Atof(std::string_view str, float *);
bool Atof(std::string_view str, double *);

std::vector<std::string> SplitStringsFromWhitespace(std::string_view str);

std::vector<std::string> SplitString(std::string_view str, char ch);
std::vector<int> SplitStringToInts(std::string_view str, char ch);
std::vector<Float> SplitStringToFloats(std::string_view str, char ch);
std::vector<double> SplitStringToDoubles(std::string_view str, char ch);

// String Utility Function Declarations
std::string UTF8FromUTF16(std::u16string str);
std::u16string UTF16FromUTF8(std::string str);

#ifdef PBRT_IS_WINDOWS
std::wstring WStringFromUTF8(std::string str);
std::string UTF8FromWString(std::wstring str);
#endif  // PBRT_IS_WINDOWS

}  // namespace pbrt

#endif  // PBRT_UTIL_STRING_H
