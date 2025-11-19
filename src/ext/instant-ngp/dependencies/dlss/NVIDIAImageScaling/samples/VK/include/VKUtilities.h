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

#include <fstream>
#include <vector>

// With NIS_DXC, bindings are specified in HLSL shader.
// Otherwise, we adjust bindings at build-time using arguments to dxc.exe
#ifdef NIS_DXC
static const uint32_t CB_BINDING = 0;
static const uint32_t IN_TEX_BINDING = 2;
static const uint32_t OUT_TEX_BINDING = 3;
static const uint32_t COEF_SCALAR_BINDING = 4;
static const uint32_t COEF_USM_BINDING = 5;
#else
static const uint32_t CB_BINDING = 0;
static const uint32_t IN_TEX_BINDING = 3;
static const uint32_t OUT_TEX_BINDING = 2;
static const uint32_t COEF_SCALAR_BINDING = 4;
static const uint32_t COEF_USM_BINDING = 5;
#endif
static const VkDescriptorType CB_DESC_TYPE = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
static const VkDescriptorType IN_TEX_DESC_TYPE = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
static const VkDescriptorType OUT_TEX_DESC_TYPE = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

#define VK_COMMON_DESC_LAYOUT(immutableSampler) \
    { CB_BINDING, CB_DESC_TYPE, 1, VK_SHADER_STAGE_COMPUTE_BIT}, \
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, (immutableSampler) }, \
    { OUT_TEX_BINDING, OUT_TEX_DESC_TYPE, 1, VK_SHADER_STAGE_COMPUTE_BIT }, \
    { IN_TEX_BINDING, IN_TEX_DESC_TYPE, 1, VK_SHADER_STAGE_COMPUTE_BIT }

// TODO: combine with DXUtilities.h:IncludeHeader::Open()
inline std::vector<char> readBytes(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        fprintf(stderr, "Failed to open: %s\n", filename.c_str());
        assert(0);
    }
    const auto sizeBytes = file.tellg();
    std::vector<char> buffer(sizeBytes);
    file.seekg(0);
    file.read(buffer.data(), sizeBytes);
    file.close();
    return buffer;
}
