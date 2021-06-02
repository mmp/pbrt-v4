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

#ifdef PBRT_IS_WINDOWS
std::wstring UTF8ToWString(std::string str);
std::wstring U16StringToWString(std::u16string str);
std::u16string WStringToU16String(std::wstring str);
#endif // PBRT_IS_WINDOWS

std::string UTF16ToUTF8(std::u16string str);
std::u16string UTF8ToUTF16(std::string str);

}  // namespace pbrt

#endif  // PBRT_UTIL_STRING_H
