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
#define NOMINMAX

#include <iostream>
#include <string>
#include <vector>

namespace img
{
    enum class Fmt : uint8_t
    {
        R8G8B8A8 = 0,
        R32G32B32A32 = 1,
        R16G16B16A16 = 2
    };

    uint32_t bytesPerPixel(Fmt fmt);

    void load(const std::string& fileName, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& outRowPitch, Fmt outFormat, uint32_t outRowPitchAlignment = 1);
    void loadPNG(const std::string& fileName, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& outRowPitch, Fmt outFormat, uint32_t outRowPitchAlignment = 1);
    void loadEXR(const std::string& fileName, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& outRowPitch, Fmt outFormat, uint32_t outRowPitchAlignment = 1);

    void save(const std::string& fileName, uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t rowPitch, Fmt format);
    void savePNG(const std::string& fileName, uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t rowPitch, Fmt format);
    void saveEXR(const std::string& fileName, uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t rowPitch, Fmt format);
}