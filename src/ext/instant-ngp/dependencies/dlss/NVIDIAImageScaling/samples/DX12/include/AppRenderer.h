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
#include "d3dx12.h"

using namespace Microsoft::WRL;

class AppRenderer
{
public:
    AppRenderer(DeviceResources& deviceResources, UIData& ui, const std::vector<std::string>& shaderPaths);
    bool updateSize();
    void update();
    void render();
    uint32_t width() { return m_outputWidth; }
    uint32_t height() { return m_outputHeight; }
    bool isInitialized() { return m_init; }
    void saveOutput(const std::string& filename);
protected:
    void scheduleCopyOutput();
    void saveOutputToFile();
private:
    UIData& m_ui;
    DeviceResources& m_deviceResources;
    NVScaler							m_NVScaler;
    NVSharpen							m_NVSharpen;
    BilinearUpscale						m_upscale;

    uint32_t							m_inputWidth = 0;
    uint32_t							m_inputHeight = 0;
    uint32_t							m_outputWidth = 0;
    uint32_t							m_outputHeight = 0;

    ComPtr<ID3D12Resource>				m_input;
    ComPtr<ID3D12Resource>				m_output;
    ComPtr<ID3D12Resource>              m_inputUpload;

    DescriptorHeap                      m_samplerDescriptorHeap;
    DescriptorHeap                      m_RVDescriptorHeap;

    std::vector<uint8_t>                m_image;
    uint32_t                            m_rowPitch;
    std::filesystem::path				m_currentFilePath;

    uint32_t                            m_currentOutputWidth;
    uint32_t                            m_currentOutputHeight;
    float							    m_currentScale = 100;
    float   							m_currentSharpness = 0;
    bool                                m_updateWindowSize = false;
    bool                                m_updateSharpness = false;
    bool                                m_uploadCoefficients = true;
    bool                                m_init = false;

    bool                                m_saveOutput = false;
    bool                                m_saveReadBack = false;
    std::string                         m_saveFileName = "";
    ComPtr<ID3D12Resource>              m_outputReadBack;
    uint32_t                            m_saveWidth = 0;
    uint32_t                            m_saveHeight = 0;
    uint32_t                            m_saveRowPitch = 0;

    enum DescriptorHeapCount : uint32_t
    {
        cCB = 3,
        cUAV = 1,
        cSRV = 3,
    };
    enum DescriptorHeapIndex : uint32_t
    {
        iCB = 0,
        iUAV = iCB + cCB,
        iSRV = iUAV + cUAV,
        iHeapEnd = iSRV + cSRV
    };
};