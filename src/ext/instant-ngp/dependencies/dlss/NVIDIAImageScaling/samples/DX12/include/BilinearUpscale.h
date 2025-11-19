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

#include <dxgi1_4.h>
#include <d3d12.h>
#include "d3dx12.h"
extern "C" {
#include <dxcapi.h>
}

#include "DeviceResources.h"

__declspec(align(256))
struct BilinearUpscaleConfig
{
    uint32_t kInputViewportOriginX;
    uint32_t kInputViewportOriginY;
    uint32_t kInputViewportWidth;
    uint32_t kInputViewportHeight;
    uint32_t kOutputViewportOriginX;
    uint32_t kOutputViewportOriginY;
    uint32_t kOutputViewportWidth;
    uint32_t kOutputViewportHeight;
    float    kScaleX;
    float    kScaleY;
    float    kDstNormX;
    float    kDstNormY;
    float    kSrcNormX;
    float    kSrcNormY;
};

void BilinearUpdateConfig(BilinearUpscaleConfig& config,
    uint32_t inputViewportOriginX, uint32_t inputViewportOriginY,
    uint32_t inputViewportWidth, uint32_t inputViewportHeight,
    uint32_t inputTextureWidth, uint32_t inputTextureHeight,
    uint32_t outputViewportOriginX, uint32_t outputViewportOriginY,
    uint32_t outputViewportWidth, uint32_t outputViewportHeight,
    uint32_t outputTextureWidth, uint32_t outputTextureHeight);

class BilinearUpscale {
public:
    BilinearUpscale(DeviceResources& deviceResources, const std::vector<std::string>& shaderPaths);
    void update(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight);
    ID3D12PipelineState* getComputePSO() { return m_computePSO.Get(); }
    ID3D12Resource* getConstantBuffer() { return m_constatBuffer.Get(); }
    ID3D12RootSignature* getRootSignature() { return m_computeRootSignature.Get(); }
    std::vector<UINT> getDispatchDim() {
        return { UINT(std::ceil(m_outputWidth / float(m_BlockWidth))), UINT(std::ceil(m_outputHeight / float(m_BlockHeight))), 1 };
    }
private:
    DeviceResources& m_deviceResources;
    BilinearUpscaleConfig               m_config;
    ComPtr<ID3D12RootSignature>         m_computeRootSignature;
    ComPtr<ID3D12PipelineState>         m_computePSO;

    ComPtr<ID3D12Resource>              m_constatBuffer;
    ComPtr<ID3D12Resource>              m_stagingBuffer;

    uint32_t                            m_outputWidth;
    uint32_t                            m_outputHeight;
    uint32_t                            m_BlockWidth = 16;
    uint32_t                            m_BlockHeight = 16;
};
