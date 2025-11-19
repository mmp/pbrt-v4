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

#include <iostream>

#include "DeviceResources.h"
#include "Utilities.h"
#include "NIS_Config.h"

class NVSharpen {
public:
    NVSharpen(DeviceResources& deviceResources, const std::vector<std::string>& shaderPaths, bool glsl);
    void update(float sharpness, uint32_t inputWidth, uint32_t inputHeight);
    void dispatch(VkImageView inputSrv, VkImageView outputUav);
    void cleanUp();
private:
    DeviceResources&                    m_deviceResources;
    NISConfig                           m_config;

    VkShaderModule                      m_shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout               m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout                    m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet                     m_descriptorSet = VK_NULL_HANDLE;
    VkBuffer                            m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory                      m_constantBufferDeviceMemory = VK_NULL_HANDLE;
    VkPipeline                          m_pipeline = VK_NULL_HANDLE;

    uint8_t*                            m_constantMemory = nullptr;
    VkDeviceSize                        m_constantBufferStride = 0;

    uint32_t                            m_outputWidth;
    uint32_t                            m_outputHeight;
    uint32_t                            m_blockWidth;
    uint32_t                            m_blockHeight;
};