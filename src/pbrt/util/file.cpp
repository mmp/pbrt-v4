// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/file.h>

#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/string.h>

#include <filesystem/path.h>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#ifndef PBRT_IS_WINDOWS
#include <dirent.h>
#include <sys/dir.h>
#include <sys/types.h>
#endif

namespace pbrt {

static filesystem::path searchDirectory;

void SetSearchDirectory(std::string filename) {
    filesystem::path path(filename);
    if (!path.is_directory())
        path = path.parent_path();
    searchDirectory = path;
}

static bool IsAbsolutePath(std::string filename) {
    if (filename.empty())
        return false;
    return filesystem::path(filename).is_absolute();
}

bool HasExtension(std::string filename, std::string e) {
    std::string ext = e;
    if (!ext.empty() && ext[0] == '.')
        ext.erase(0, 1);

    std::string filenameExtension = filesystem::path(filename).extension();
    if (ext.size() > filenameExtension.size())
        return false;
    return std::equal(ext.rbegin(), ext.rend(), filenameExtension.rbegin(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

std::string RemoveExtension(std::string filename) {
    std::string ext = filesystem::path(filename).extension();
    if (ext.empty())
        return filename;
    std::string f = filename;
    f.erase(f.end() - ext.size() - 1, f.end());
    return f;
}

std::string ResolveFilename(std::string filename) {
    if (searchDirectory.empty() || filename.empty())
        return filename;
    else if (IsAbsolutePath(filename))
        return filename;
    else
        return (searchDirectory / filesystem::path(filename)).make_absolute().str();
}

std::vector<std::string> MatchingFilenames(std::string filenameBase) {
    std::vector<std::string> filenames;

    filesystem::path basePath(filenameBase);
    std::string dirStr = basePath.parent_path().str();
    if (dirStr.empty())
        dirStr = ".";
#ifdef PBRT_IS_WINDOWS
    LOG_FATAL("Need Windows implementation of MatchingFilenames()");
#else
    DIR *dir = opendir(dirStr.c_str());
    if (!dir)
        ErrorExit("%s: unable to open directory\n", basePath.parent_path().str().c_str());

    struct dirent *ent;
    size_t n = basePath.filename().size();
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_type == DT_REG &&
            strncmp(basePath.filename().c_str(), ent->d_name, n) == 0)
            filenames.push_back(
                (basePath.parent_path() / filesystem::path(ent->d_name)).str());
    }
    closedir(dir);
#endif

    return filenames;
}

bool FileExists(std::string filename) {
#ifdef PBRT_IS_WINDOWS
    std::ifstream ifs(WStringFromUTF8(filename).c_str());
#else
    std::ifstream ifs(filename);
#endif
    return (bool)ifs;
}

std::string ReadFileContents(std::string filename) {
#ifdef PBRT_IS_WINDOWS
    std::ifstream ifs(WStringFromUTF8(filename).c_str(), std::ios::binary);
#else
    std::ifstream ifs(filename, std::ios::binary);
#endif
    if (!ifs)
        ErrorExit("%s: %s", filename, ErrorString());
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       (std::istreambuf_iterator<char>()));
}

FILE *FOpenRead(std::string filename) {
#ifdef PBRT_IS_WINDOWS
    return _wfopen(WStringFromUTF8(filename).c_str(), L"rb");
#else
    return fopen(filename.c_str(), "rb");
#endif
}

FILE *FOpenWrite(std::string filename) {
#ifdef PBRT_IS_WINDOWS
    return _wfopen(WStringFromUTF8(filename).c_str(), L"wb");
#else
    return fopen(filename.c_str(), "wb");
#endif
}

std::vector<Float> ReadFloatFile(std::string filename) {
    FILE *f = FOpenRead(filename);
    if (f == nullptr) {
        Error("%s: unable to open file", filename);
        return {};
    }

    int c;
    bool inNumber = false;
    char curNumber[32];
    int curNumberPos = 0;
    int lineNumber = 1;
    std::vector<Float> values;
    while ((c = getc(f)) != EOF) {
        if (c == '\n')
            ++lineNumber;
        if (inNumber) {
            if (curNumberPos >= (int)sizeof(curNumber))
                LOG_FATAL("Overflowed buffer for parsing number in file: %s at "
                          "line %d",
                          filename, lineNumber);
            // Note: this is not very robust, and would accept something
            // like 0.0.0.0eeee-+--2 as a valid number.
            if ((isdigit(c) != 0) || c == '.' || c == 'e' || c == 'E' || c == '-' ||
                c == '+') {
                CHECK_LT(curNumberPos, sizeof(curNumber));
                curNumber[curNumberPos++] = c;
            } else {
                curNumber[curNumberPos++] = '\0';
                Float v;
                if (!Atof(curNumber, &v))
                    ErrorExit("%s: unable to parse float value \"%s\"", filename,
                              curNumber);
                values.push_back(v);
                inNumber = false;
                curNumberPos = 0;
            }
        } else {
            if ((isdigit(c) != 0) || c == '.' || c == '-' || c == '+') {
                inNumber = true;
                curNumber[curNumberPos++] = c;
            } else if (c == '#') {
                while ((c = getc(f)) != '\n' && c != EOF)
                    ;
                ++lineNumber;
            } else if (isspace(c) == 0) {
                Error("%s: unexpected character \"%c\" found at line %d.", filename, c,
                      lineNumber);
                return {};
            }
        }
    }
    fclose(f);
    return values;
}

bool WriteFileContents(std::string filename, const std::string &contents) {
#ifdef PBRT_IS_WINDOWS
    std::ofstream out(WStringFromUTF8(filename).c_str(), std::ios::binary);
#else
    std::ofstream out(filename, std::ios::binary);
#endif
    out << contents;
    out.close();
    if (!out.good()) {
        Error("%s: %s", filename, ErrorString());
        return false;
    }
    return true;
}

}  // namespace pbrt
