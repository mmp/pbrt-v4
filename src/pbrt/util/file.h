// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_FILE_H
#define PBRT_UTIL_FILE_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

#include <string>
#include <vector>

namespace pbrt {

// File and Filename Function Declarations
std::string ReadFileContents(const std::string &filename);
bool WriteFile(const std::string &filename, const std::string &contents);

std::vector<float> ReadFloatFile(const std::string &filename);

bool FileExists(const std::string &filename);
std::string ResolveFilename(const std::string &filename);
void SetSearchDirectory(const std::string &filename);

bool HasExtension(const std::string &filename, const std::string &ext);
std::string RemoveExtension(const std::string &filename);

std::vector<std::string> MatchingFilenames(const std::string &base);

}  // namespace pbrt

#endif  // PBRT_UTIL_FILE_H
