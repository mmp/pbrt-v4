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

#include "NVSharpen.h"

#include <dxgi1_4.h>
#include <d3d11_3.h>
#include <d3dcompiler.h>
#include <iostream>

#include "DXUtilities.h"
#include "DeviceResources.h"
#include "Utilities.h"


NVSharpen::NVSharpen(DeviceResources& deviceResources, const std::vector<std::string>& shaderPaths)
    : m_deviceResources(deviceResources)
    , m_outputWidth(1)
    , m_outputHeight(1)
{
    NISOptimizer opt(false, NISGPUArchitecture::NVIDIA_Generic);
    m_blockWidth = opt.GetOptimalBlockWidth();
    m_blockHeight = opt.GetOptimalBlockHeight();
    uint32_t threadGroupSize = opt.GetOptimalThreadGroupSize();

    DX::Defines defines;
    defines.add("NIS_SCALER", false);
    defines.add("NIS_HDR_MODE", uint32_t(NISHDRMode::None));
    defines.add("NIS_BLOCK_WIDTH", m_blockWidth);
    defines.add("NIS_BLOCK_HEIGHT", m_blockHeight);
    defines.add("NIS_THREAD_GROUP_SIZE", threadGroupSize);

    std::string shaderName = "NIS_Main.hlsl";
    std::string shaderFolder;
    for (auto& e : shaderPaths)
    {
        if (std::filesystem::exists(e + "/" + shaderName))
        {
            shaderFolder = e;
            break;
        }
    }
    if (shaderFolder.empty())
        throw std::runtime_error("Shader file not found" + shaderName);
    std::wstring wShaderFilename = widen(shaderFolder + "/" + "NIS_Main.hlsl");
    DX::IncludeHeader includeHeader({ shaderFolder });
    DX::CompileComputeShader(m_deviceResources.device(),
        wShaderFilename.c_str(),
        "main",
        &m_computeShader,
        defines.get(),
        &includeHeader);

    const int rowPitch = kFilterSize * 4;
    const int imageSize = rowPitch * kPhaseCount;
    m_deviceResources.createLinearClampSampler(&m_LinearClampSampler);
    m_deviceResources.createConstBuffer(&m_config, sizeof(NISConfig), &m_csBuffer);
}

void NVSharpen::update(float sharpness, uint32_t inputWidth, uint32_t inputHeight)
{
    NVSharpenUpdateConfig(m_config, sharpness,
        0, 0, inputWidth, inputHeight, inputWidth, inputHeight,
        0, 0, NISHDRMode::None);
    m_deviceResources.updateConstBuffer(&m_config, sizeof(NISConfig), m_csBuffer.Get());
    m_outputWidth = inputWidth;
    m_outputHeight = inputHeight;
}

void NVSharpen::dispatch(ID3D11ShaderResourceView* const* input, ID3D11UnorderedAccessView* const* output)
{
    auto context = m_deviceResources.context();
    context->CSSetShaderResources(0, 1, input);
    context->CSSetUnorderedAccessViews(0, 1, output, nullptr);
    context->CSSetSamplers(0, 1, m_LinearClampSampler.GetAddressOf());
    context->CSSetConstantBuffers(0, 1, m_csBuffer.GetAddressOf());
    context->CSSetShader(m_computeShader.Get(), nullptr, 0);
    context->Dispatch(UINT(std::ceil(m_outputWidth / float(m_blockWidth))), UINT(std::ceil(m_outputHeight / float(m_blockHeight))), 1);
}