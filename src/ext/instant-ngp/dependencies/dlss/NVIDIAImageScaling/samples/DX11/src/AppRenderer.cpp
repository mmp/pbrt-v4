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

#include "AppRenderer.h"

using namespace Microsoft::WRL;

AppRenderer::AppRenderer(DeviceResources& deviceResources, UIData& ui, const std::vector<std::string>& shaderPaths)
    : m_ui(ui)
    , m_deviceResources(deviceResources)
    , m_NVSharpen(deviceResources, shaderPaths)
    , m_NVScaler(deviceResources, shaderPaths)
    , m_upscale(deviceResources)
{}

bool AppRenderer::update()
{
    bool updateWindowSize = m_currentFilePath != m_ui.FilePath || m_currentScale != m_ui.Scale;
    bool updateSharpeness = m_ui.Sharpness != m_currentSharpness;
    if (updateWindowSize)
    {
        if (m_currentFilePath != m_ui.FilePath)
        {
            std::vector<uint8_t> image;
            uint32_t rowPitch;
            img::load(m_ui.FilePath.string(), image, m_inputWidth, m_inputHeight, rowPitch, img::Fmt::R8G8B8A8);
            m_deviceResources.createTexture2D(m_inputWidth, m_inputHeight, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_USAGE_DEFAULT, image.data(), rowPitch, rowPitch * m_inputHeight, &m_input);
            m_deviceResources.createSRV(m_input.Get(), DXGI_FORMAT_R8G8B8A8_UNORM, &m_inputSRV);
            m_currentFilePath = m_ui.FilePath;
        }
        if (m_ui.Scale == 100)
        {
            m_outputWidth = m_inputWidth;
            m_outputHeight = m_inputHeight;
        }
        else
        {
            m_outputWidth = uint32_t(std::ceil(m_inputWidth * 100.f / m_ui.Scale));
            m_outputHeight = uint32_t(std::ceil(m_inputHeight * 100.f / m_ui.Scale));
        }
        m_deviceResources.createTexture2D(m_outputWidth, m_outputHeight, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_USAGE_DEFAULT, nullptr , 0, 0, &m_output);
        m_deviceResources.createUAV(m_output.Get(), DXGI_FORMAT_R8G8B8A8_UNORM, &m_outputUAV);
        m_currentScale = m_ui.Scale;
        m_ui.InputWidth = m_inputWidth;
        m_ui.InputHeight = m_inputHeight;
        m_ui.OutputWidth = m_outputWidth;
        m_ui.OutputHeight = m_outputHeight;
    }
    if (updateSharpeness)
    {
        m_currentSharpness = m_ui.Sharpness;
    }
    if (updateSharpeness || updateWindowSize)
    {
        m_upscale.update(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
        m_NVScaler.update(m_currentSharpness / 100.f, m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
        m_NVSharpen.update(m_currentSharpness / 100.f, m_inputWidth, m_inputHeight);
    }
    return updateWindowSize;
}

void AppRenderer::saveOutput(const std::string& filename)
{
    D3D11_TEXTURE2D_DESC desc;
    m_output->GetDesc(&desc);
    img::Fmt format = img::Fmt::R8G8B8A8;

    switch (desc.Format)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        format = img::Fmt::R8G8B8A8;
        break;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        format = img::Fmt::R32G32B32A32;
        break;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        format = img::Fmt::R16G16B16A16;
        break;
    }
    std::vector<uint8_t> data;
    uint32_t width, height, rowPitch = 0;
    constexpr uint32_t channels = 4;
    m_deviceResources.getTextureData(m_output.Get(), data, width, height, rowPitch);
    img::save(filename, data.data(), width, height, channels, rowPitch, format);
}

void AppRenderer::render()
{
    auto context = m_deviceResources.context();
    D3D11_QUERY_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_QUERY_DESC));
    desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
    m_deviceResources.device()->CreateQuery(&desc, &m_timeStampDis);
    desc.Query = D3D11_QUERY_TIMESTAMP;
    m_deviceResources.device()->CreateQuery(&desc, &m_timeStampStart);
    m_deviceResources.device()->CreateQuery(&desc, &m_timeStampEnd);
    context->Begin(m_timeStampDis.Get());
    context->End(m_timeStampStart.Get());
    if (!m_ui.EnableNVScaler)
    {
        if (m_ui.Scale == 100)
        {
            context->CopyResource(m_output.Get(), m_input.Get());
        }
        else
        {
            m_upscale.dispatch(m_inputSRV.GetAddressOf(), m_outputUAV.GetAddressOf());
        }
    }
    else {
        if (m_ui.Scale == 100)
        {
            m_NVSharpen.dispatch(m_inputSRV.GetAddressOf(), m_outputUAV.GetAddressOf());
        }
        else
        {
            m_NVScaler.dispatch(m_inputSRV.GetAddressOf(), m_outputUAV.GetAddressOf());
        }
    }

    context->End(m_timeStampEnd.Get());
    context->End(m_timeStampDis.Get());
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disData;
    while (context->GetData(m_timeStampDis.Get(), &disData, sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT), 0) != S_OK);
    UINT64 startime;
    UINT64 endtime;
    while (context->GetData(m_timeStampStart.Get(), &startime, sizeof(UINT64), 0) != S_OK);
    while (context->GetData(m_timeStampEnd.Get(), &endtime, sizeof(UINT64), 0) != S_OK);
    if (!disData.Disjoint)
    {
        m_ui.FilterTime = (endtime - startime) / double(disData.Frequency) * 1E6;
    }
    D3D11_BOX sourceRegion;
    sourceRegion.left = 0;
    sourceRegion.right = m_deviceResources.width();
    sourceRegion.top = 0;
    sourceRegion.bottom = m_deviceResources.height();
    sourceRegion.front = 0;
    sourceRegion.back = 1;
    context->CopySubresourceRegion(m_deviceResources.renderTarget(), 0, 0, 0, 0, m_output.Get(), 0, &sourceRegion);
    context->OMSetRenderTargets(1, m_deviceResources.targetViewAddress(), nullptr);
}