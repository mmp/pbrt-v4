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
#include <sstream>
#include <iomanip>

#include "DeviceResources.h"
#include "NVScaler.h"
#include "NVSharpen.h"
#include "UIRenderer.h"
#include "Image.h"


class AppRenderer
{
public:
    AppRenderer(DeviceResources& deviceResources, UIData& ui, const std::vector<std::string>& shaderPaths, bool glsl);
    bool update();
    void render();
    void present();
    void cleanUp();
    uint32_t width() { return m_outputWidth; }
    uint32_t height() { return m_outputHeight; }
private:
    void blitInputToTemp();
    void blitCopyToRenderTarget();

    UIData&                             m_ui;
    DeviceResources&                    m_deviceResources;
    NVSharpen							m_NVSharpen;
    NVScaler							m_NVScaler;
    uint32_t							m_inputWidth = 0;
    uint32_t							m_inputHeight = 0;
    uint32_t							m_outputWidth = 0;
    uint32_t							m_outputHeight = 0;
    VkImage				                m_input = VK_NULL_HANDLE;
    VkDeviceMemory                      m_inputDeviceMemory;
    VkImageView	                        m_inputSRV;

    std::vector<uint8_t>                m_image;
    uint32_t                            m_rowPitch;
    std::filesystem::path				m_currentFilePath;

    VkImage				                m_temp = VK_NULL_HANDLE;
    VkDeviceMemory                      m_tempDeviceMemory;
    VkImageView	                        m_tempSRV;

    uint32_t                            m_currentOutputWidth;
    uint32_t                            m_currentOutputHeight;
    float							    m_currentScale = 100;
    float   							m_currentSharpness = 0;
    bool                                m_updateWindowSize = false;
    bool                                m_updateSharpness = false;
    bool                                m_uploadCoefficients = true;
    bool                                m_init = false;
};