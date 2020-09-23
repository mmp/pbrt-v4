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
bool Atod(std::string_view str, double *);

std::vector<std::string> SplitStringsFromWhitespace(std::string_view str);

std::vector<std::string> SplitString(std::string_view str, char ch);
std::vector<int> SplitStringToInts(std::string_view str, char ch);
std::vector<Float> SplitStringToFloats(std::string_view str, char ch);
std::vector<double> SplitStringToDoubles(std::string_view str, char ch);

}  // namespace pbrt

#endif  // PBRT_UTIL_STRING_H
