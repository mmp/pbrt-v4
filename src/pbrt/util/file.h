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
std::string ReadFileContents(std::string filename);
bool WriteFileContents(std::string filename, const std::string &contents);

std::vector<Float> ReadFloatFile(std::string filename);

bool FileExists(std::string filename);
std::string ResolveFilename(std::string filename);
void SetSearchDirectory(std::string filename);

bool HasExtension(std::string filename, std::string ext);
std::string RemoveExtension(std::string filename);

std::vector<std::string> MatchingFilenames(std::string filename);

FILE *FOpenRead(std::string filename);
FILE *FOpenWrite(std::string filename);

}  // namespace pbrt

#endif  // PBRT_UTIL_FILE_H
