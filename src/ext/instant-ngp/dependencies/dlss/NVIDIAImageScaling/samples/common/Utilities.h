// The MIT License(MIT)
//
// Copyright(c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <vector>

inline std::vector<std::filesystem::path> getFiles(const std::string& path, const std::string& ext = "")
{
    namespace fs = std::filesystem;
    std::vector<fs::path> ret;
    for (const auto& e : fs::directory_iterator(path)) {
        if (ext == "" || e.path().extension() == ext)
            ret.push_back(e.path());
    }
    return ret;
}


class FPS {
public:
    FPS(double maxTime = 100000) : m_maxTime(maxTime), m_totalTime(0), m_averageTime(0), m_numFrames(0) {
        m_t0 = std::chrono::high_resolution_clock::now();
    }
    void update() {
        m_t1 = std::chrono::high_resolution_clock::now();
        m_totalTime += std::chrono::duration_cast<std::chrono::microseconds>(m_t1 - m_t0).count();
        m_t0 = m_t1;
        m_numFrames++;
        // update only when time is up
        if (m_totalTime > m_maxTime)
        {
            m_averageTime = m_totalTime / m_numFrames;
            m_totalTime = 0.0;
            m_numFrames = 0;
        }
    }
    void setMaxTime(double maxTime) {
        m_maxTime = maxTime;
    }
    double averageTime_us() {
        return m_averageTime;
    }
    double averageTime_ms() {
        return m_averageTime / 1E3;
    }
    double fps() {
        return 1 / m_averageTime * 1E6;
    }
private:
    double m_maxTime;
    double m_totalTime;
    double m_averageTime;
    size_t m_numFrames;
    std::chrono::high_resolution_clock::time_point m_t0, m_t1;
};

class ElapsedTimer {
public:
    ElapsedTimer(uint32_t maxIterations = 500) : m_maxIterations(maxIterations), m_totalTime(0), m_averageTime(0) {
        m_t0 = std::chrono::high_resolution_clock::now();
    }
    void start() {
        m_t0 = std::chrono::high_resolution_clock::now();
    }
    void end() {
        m_t1 = std::chrono::high_resolution_clock::now();
        m_totalTime += std::chrono::duration_cast<std::chrono::microseconds>(m_t1 - m_t0).count();
        m_numIterations++;
        if (m_numIterations > m_maxIterations)
        {
            m_averageTime = m_totalTime / m_maxIterations;
            m_totalTime = 0.0;
            m_numIterations = 0;
        }
    }
    void setMaxTime(uint32_t maxTime) {
        m_maxIterations = maxTime;
    }
    double averageTime_us() {
        return m_averageTime;
    }
    double averageTime_ms() {
        return m_averageTime / 1E3;
    }
private:
    uint32_t m_maxIterations;
    double m_totalTime;
    double m_averageTime;
    size_t m_numIterations;
    std::chrono::high_resolution_clock::time_point m_t0, m_t1;
};



class ArgParser {
public:
    ArgParser(int argc, char* argv[]) {
        programName = argv[0];
        addOption("-h", "Print this help");
        for (size_t i = 1; i < argc; ++i)
            arguments.push_back(argv[i]);
    }
    void addOption(const std::string& opt, const std::string& description) {
        options[opt] = description;
    }
    void printHelp() {
        std::cout << "Usage: " << programName << " " << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (auto& e : options) {
            std::cout << e.first << " " << e.second << std::endl;;
        }
    }
    bool parse(bool requiredArgs = false) {
        if (arguments.size() > 0 && arguments[0] == "-h") {
            printHelp();
            return false;
        }
        if (arguments.size() < 1) {
            return !requiredArgs;
        }
        bool lastOpt = false;
        size_t i = 0;
        while (i < arguments.size()) {
            if (options.find(arguments[i]) != options.end() && i < arguments.size() - 1) {
                argMap[arguments[i]] = arguments[i + 1];
                i += 2;
            }
            else {
                std::cout << "Argument not found : " << arguments[i] << std::endl;
                printHelp();
                return false;
            }
        }
        return true;
    }
    template<typename T>
    T get(const std::string& opt, T defaultVal = T()) {
        T val;
        auto sval = argMap.find(opt);
        if (sval == argMap.end())
            return defaultVal;
        std::stringstream ss;
        ss << sval->second;
        ss >> val;
        return val;
    }
private:
    std::map<std::string, std::string> options;
    std::vector<std::string> arguments;
    std::unordered_map<std::string, std::string> argMap;
    std::string programName;
};

#ifndef NIS_VK_SAMPLE
inline std::wstring widen(const std::string& str)
{
    int size = MultiByteToWideChar(CP_ACP, 0, str.c_str(), int(str.size()) + 1, 0, 0);
    std::vector<wchar_t> temp(size);
    MultiByteToWideChar(CP_ACP, 0, str.c_str(), int(str.size()) + 1, &temp[0], int(temp.size()));
    return std::wstring(&temp[0]);
}
#endif

template <typename T>
inline std::string toStr(T value)
{
    return std::to_string(value);
}

template <>
inline std::string toStr<bool>(bool value)
{
    return value ? "1" : "0";
}

template <>
inline std::string toStr<std::string>(std::string value)
{
    return value;
}

template <>
inline std::string toStr<const char*>(const char* value)
{
    return value;
}

inline uint32_t Align(uint32_t x, uint32_t alignment) {
    return (x + alignment - 1) / alignment * alignment;
}

