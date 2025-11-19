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

#include "BilinearUpscale.h"
#include "DXUtilities.h"
#include "Utilities.h"

#include <iostream>
#include <d3dcompiler.h>

void BilinearUpdateConfig(BilinearUpscaleConfig& config,
    uint32_t inputViewportOriginX, uint32_t inputViewportOriginY,
    uint32_t inputViewportWidth, uint32_t inputViewportHeight,
    uint32_t inputTextureWidth, uint32_t inputTextureHeight,
    uint32_t outputViewportOriginX, uint32_t outputViewportOriginY,
    uint32_t outputViewportWidth, uint32_t outputViewportHeight,
    uint32_t outputTextureWidth, uint32_t outputTextureHeight)
{
    config.kInputViewportHeight = inputViewportHeight;
    config.kInputViewportWidth = inputViewportWidth;
    config.kOutputViewportHeight = outputViewportHeight;
    config.kOutputViewportWidth = outputViewportWidth;

    config.kInputViewportOriginX = inputViewportOriginX;
    config.kInputViewportOriginY = inputViewportOriginY;
    config.kOutputViewportOriginX = outputViewportOriginX;
    config.kOutputViewportOriginY = outputViewportOriginY;

    config.kScaleX = inputTextureWidth / float(outputTextureWidth);
    config.kScaleY = inputTextureWidth / float(outputTextureWidth);
    config.kDstNormX = 1.f / outputTextureWidth;
    config.kDstNormY = 1.f / outputTextureHeight;
    config.kSrcNormX = 1.f / inputTextureWidth;
    config.kSrcNormY = 1.f / inputTextureHeight;
}

BilinearUpscale::BilinearUpscale(DeviceResources& deviceResources)
        : m_deviceResources(deviceResources)
        , m_outputWidth(0)
        , m_outputHeight(0)
{
    std::string skBlockWidth = std::to_string(kBlockWidth);
    std::string skBlockHeight = std::to_string(kBlockHeight);
    const D3D_SHADER_MACRO shaderMacros[] = { {"kBlockWidth", skBlockWidth.c_str() },
                                              {"kBlockHeight", skBlockHeight.c_str() },
                                              { nullptr, nullptr } };

    static const char* upscaleShader =
    R"(
        Texture2D                 in_texture   : register(t0); // image srv
        RWTexture2D<unorm float4> out_texture  : register(u1); // working uav
        cbuffer cb : register(b0) {
            uint     kInputViewportOriginX;
            uint     kInputViewportOriginY;
            uint     kInputViewportWidth;
            uint     kInputViewportHeight;
            uint     kOutputViewportOriginX;
            uint     kOutputViewportOriginY;
            uint     kOutputViewportWidth;
            uint     kOutputViewportHeight;
            float    kScaleX;
            float    kScaleY;
            float    kDstNormX;
            float    kDstNormY;
            float    kSrcNormX;
            float    kSrcNormY;
        }
        SamplerState  samplerLinearClamp : register(s0);
        [numthreads(kBlockWidth, kBlockHeight, 1)]
        void Bilinear(uint3 id : SV_DispatchThreadID) {
            float dX = (id.x + 0.5f) * kScaleX;
            float dY = (id.y + 0.5f) * kScaleY;
            if (id.x < kOutputViewportWidth && id.y < kOutputViewportHeight && dX < kInputViewportWidth && dY < kInputViewportHeight) {
              float uvX = (dX + kInputViewportOriginX) * kSrcNormX;
              float uvY = (dY + kInputViewportOriginY) * kSrcNormY;
              uint dstX = id.x + kOutputViewportOriginX;
              uint dstY = id.y + kOutputViewportOriginY;
              out_texture[uint2(dstX, dstY)] = in_texture.SampleLevel(samplerLinearClamp, float2(uvX, uvY), 0);
            }
        }
    )";

    DX::CompileComputeShader(m_deviceResources.device(), upscaleShader, strlen(upscaleShader), "Bilinear", &m_computeShader, shaderMacros);

    BilinearUpdateConfig(m_config, 0, 0, 100, 100, 100, 100, 0, 0, 100, 100, 100, 100);

    m_deviceResources.createLinearClampSampler(&m_LinearClampSampler);

    D3D11_BUFFER_DESC bDesc;
    bDesc.ByteWidth = sizeof(BilinearUpscaleConfig);
    bDesc.Usage = D3D11_USAGE_DYNAMIC;
    bDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bDesc.MiscFlags = 0;
    bDesc.StructureByteStride = 0;

    D3D11_SUBRESOURCE_DATA srData;
    srData.pSysMem = &m_config;
    srData.SysMemPitch = sizeof(BilinearUpscaleConfig);
    srData.SysMemSlicePitch = 1;
    DX::ThrowIfFailed(m_deviceResources.device()->CreateBuffer(&bDesc, &srData, &m_csBuffer));
}

void BilinearUpscale::update(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight)
{
    BilinearUpdateConfig(m_config, 0, 0, inputWidth, inputHeight, inputWidth, inputHeight, 0, 0, outputWidth, outputHeight, outputWidth, outputHeight);
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    m_deviceResources.context()->Map(m_csBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    BilinearUpscaleConfig* dataPtr = (BilinearUpscaleConfig*)mappedResource.pData;
    memcpy(dataPtr, &m_config, sizeof(BilinearUpscaleConfig));
    m_deviceResources.context()->Unmap(m_csBuffer.Get(), 0);
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;
}

void BilinearUpscale::dispatch(ID3D11ShaderResourceView* const* input, ID3D11UnorderedAccessView* const* output)
{
    auto context = m_deviceResources.context();
    context->CSSetShader(m_computeShader.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, input);
    context->CSSetUnorderedAccessViews(1, 1, output, nullptr);
    context->CSSetSamplers(0, 1, m_LinearClampSampler.GetAddressOf());
    context->CSSetConstantBuffers(0, 1, m_csBuffer.GetAddressOf());
    context->Dispatch(UINT(std::ceil(m_outputWidth / float(kBlockWidth))), UINT(std::ceil(m_outputHeight / float(kBlockHeight))), 1);
}