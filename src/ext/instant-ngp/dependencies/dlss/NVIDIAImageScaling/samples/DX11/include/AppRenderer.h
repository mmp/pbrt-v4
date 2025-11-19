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
#include <d3d11.h>
#include <wrl.h>

#include "DeviceResources.h"
#include "NVScaler.h"
#include "NVSharpen.h"
#include "UIRenderer.h"
#include "BilinearUpscale.h"
#include "Image.h"

using namespace Microsoft::WRL;

class AppRenderer
{
public:
    AppRenderer(DeviceResources& deviceResources, UIData& ui, const std::vector<std::string>& shaderPaths);
    bool update();
    void render();
    uint32_t width() { return m_outputWidth; }
    uint32_t height() { return m_outputHeight; }
    void saveOutput(const std::string& filename);
private:
    UIData& m_ui;
    DeviceResources& m_deviceResources;
    NVSharpen							m_NVSharpen;
    NVScaler							m_NVScaler;
    BilinearUpscale						m_upscale;
    uint32_t							m_inputWidth = 0;
    uint32_t							m_inputHeight = 0;
    uint32_t							m_outputWidth = 0;
    uint32_t							m_outputHeight = 0;
    ComPtr<ID3D11Texture2D>				m_input;
    ComPtr<ID3D11Texture2D>				m_output;
    ComPtr<ID3D11ShaderResourceView>	m_inputSRV;
    ComPtr<ID3D11UnorderedAccessView>	m_outputUAV;

    std::filesystem::path				m_currentFilePath;
    float							    m_currentScale = 100.f;
    float    							m_currentSharpness = 0.f;

    ComPtr<ID3D11Query>                 m_timeStampDis;
    ComPtr<ID3D11Query>                 m_timeStampStart;
    ComPtr<ID3D11Query>                 m_timeStampEnd;
};