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
    , m_NVScaler(deviceResources, shaderPaths)
    , m_NVSharpen(deviceResources, shaderPaths)
    , m_upscale(deviceResources, shaderPaths)
{

    D3D12_SAMPLER_DESC samplerDesc;
    ZeroMemory(&samplerDesc, sizeof(samplerDesc));
    samplerDesc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplerDesc.MipLODBias = 0.0f;
    samplerDesc.MaxAnisotropy = 1;
    samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;

    m_samplerDescriptorHeap.Create(m_deviceResources.device(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, L"samplerDescriptorHeap");
    m_deviceResources.device()->CreateSampler(&samplerDesc, m_samplerDescriptorHeap.getCPUDescriptorHandle(0));

    m_RVDescriptorHeap.Create(m_deviceResources.device(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, DescriptorHeapIndex::iHeapEnd, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, L"RVDescriptorHeap");

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    // Bilinear
    cbvDesc.BufferLocation = m_upscale.getConstantBuffer()->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = sizeof(BilinearUpscaleConfig);
    m_deviceResources.device()->CreateConstantBufferView(&cbvDesc, m_RVDescriptorHeap.getCPUDescriptorHandle(iCB));

    // NVScaler
    cbvDesc.BufferLocation = m_NVScaler.getConstantBuffer()->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = sizeof(NISConfig);
    m_deviceResources.device()->CreateConstantBufferView(&cbvDesc, m_RVDescriptorHeap.getCPUDescriptorHandle(iCB + 1));
    m_deviceResources.device()->CreateShaderResourceView(m_NVScaler.getCoefScaler(), nullptr, m_RVDescriptorHeap.getCPUDescriptorHandle(iSRV + 1));
    m_deviceResources.device()->CreateShaderResourceView(m_NVScaler.getCoefUSM(), nullptr, m_RVDescriptorHeap.getCPUDescriptorHandle(iSRV + 2));

    // NVSharpen
    cbvDesc.BufferLocation = m_NVSharpen.getConstantBuffer()->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = sizeof(NISConfig);
    m_deviceResources.device()->CreateConstantBufferView(&cbvDesc, m_RVDescriptorHeap.getCPUDescriptorHandle(iCB + 2));
}

bool AppRenderer::updateSize()
{
    bool updateWindowSize = m_currentFilePath != m_ui.FilePath || m_currentScale != m_ui.Scale;
    bool updateSharpness = m_ui.Sharpness != m_currentSharpness;
    if (updateWindowSize)
    {
        if (m_currentFilePath != m_ui.FilePath)
        {
            img::load(m_ui.FilePath.string(), m_image, m_inputWidth, m_inputHeight, m_rowPitch, img::Fmt::R16G16B16A16, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
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

        m_currentScale = m_ui.Scale;
        m_ui.InputWidth = m_inputWidth;
        m_ui.InputHeight = m_inputHeight;
        m_ui.OutputWidth = m_outputWidth;
        m_ui.OutputHeight = m_outputHeight;
        m_updateWindowSize = true;
    }
    if (updateSharpness) {
        m_currentSharpness = m_ui.Sharpness;
        m_updateSharpness = true;
    }
    m_init = true;
    return updateWindowSize;
}

void AppRenderer::update()
{
    if (m_updateWindowSize) {
        m_deviceResources.CreateBuffer(uint32_t(m_image.size()), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, &m_inputUpload);
        m_deviceResources.CreateTexture2D(m_inputWidth, m_inputHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON, &m_input);
        m_deviceResources.UploadTextureData(m_image.data(), uint32_t(m_image.size()), m_rowPitch , m_input.Get(), m_inputUpload.Get());

        m_deviceResources.device()->CreateShaderResourceView(m_input.Get(), nullptr, m_RVDescriptorHeap.getCPUDescriptorHandle(iSRV));

        m_deviceResources.CreateTexture2D(m_outputWidth, m_outputHeight, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_STATE_COMMON, &m_output);
        m_deviceResources.device()->CreateShaderResourceView(m_output.Get(), nullptr, m_RVDescriptorHeap.getCPUDescriptorHandle(iUAV));
    }
    if (m_updateSharpness || m_updateWindowSize) {
        m_upscale.update(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
        m_NVScaler.update(m_currentSharpness / 100.f, m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
        m_NVSharpen.update(m_currentSharpness / 100.f, m_inputWidth, m_inputHeight);
        m_updateWindowSize = false;
        m_updateSharpness = false;
    }
    if (m_uploadCoefficients) {
        m_NVScaler.uploadCoefficients();
        m_uploadCoefficients = false;
    }
    if (m_saveOutput && m_saveReadBack) {
        saveOutputToFile();
        m_saveOutput = false;
        m_saveReadBack = false;
    }
}

void AppRenderer::render()
{
    auto computeCommandList = m_deviceResources.computeCommandList();
    std::vector<ID3D12DescriptorHeap*> pHeaps{ m_RVDescriptorHeap.getDescriptorHeap(), m_samplerDescriptorHeap.getDescriptorHeap() };
    computeCommandList->SetDescriptorHeaps(uint32_t(pHeaps.size()), pHeaps.data());
    std::vector<uint32_t> dispatch;
    if (m_ui.EnableNVScaler) {
        if (m_currentScale != 100) {
            dispatch = m_NVScaler.getDispatchDim();
            computeCommandList->SetComputeRootSignature(m_NVScaler.getRootSignature());
            computeCommandList->SetComputeRootDescriptorTable(0, m_RVDescriptorHeap.getGPUDescriptorHandle(iCB + 1));  // ConstantBuffer config
            computeCommandList->SetComputeRootDescriptorTable(4, m_RVDescriptorHeap.getGPUDescriptorHandle(iSRV + 1)); // coef_scaler
            computeCommandList->SetComputeRootDescriptorTable(5, m_RVDescriptorHeap.getGPUDescriptorHandle(iSRV + 2)); // coef_USM
            computeCommandList->SetPipelineState(m_NVScaler.getComputePSO());
        }
        else
        {
            dispatch = m_NVSharpen.getDispatchDim();
            computeCommandList->SetComputeRootSignature(m_NVSharpen.getRootSignature());
            computeCommandList->SetComputeRootDescriptorTable(0, m_RVDescriptorHeap.getGPUDescriptorHandle(iCB + 2)); // ConstantBuffer config
            computeCommandList->SetPipelineState(m_NVSharpen.getComputePSO());
        }
    }
    else {
        dispatch = m_upscale.getDispatchDim();
        computeCommandList->SetComputeRootSignature(m_upscale.getRootSignature());
        computeCommandList->SetComputeRootDescriptorTable(0, m_RVDescriptorHeap.getGPUDescriptorHandle(iCB));    // ConstantBuffer config
        computeCommandList->SetPipelineState(m_upscale.getComputePSO());

    }
    computeCommandList->SetComputeRootDescriptorTable(1, m_samplerDescriptorHeap.getGPUDescriptorHandle(0));   // sampler
    computeCommandList->SetComputeRootDescriptorTable(2, m_RVDescriptorHeap.getGPUDescriptorHandle(iSRV));     // input
    computeCommandList->SetComputeRootDescriptorTable(3, m_RVDescriptorHeap.getGPUDescriptorHandle(iUAV));     // output
    m_deviceResources.StartComputeTimer();
    computeCommandList->Dispatch(dispatch[0], dispatch[1], dispatch[2]);
    m_deviceResources.StopComputeTimer();
    m_deviceResources.ResolveComputeTimerQuery();
    if (m_saveOutput) {
        scheduleCopyOutput();
        m_saveReadBack = true;
    }
    computeCommandList->Close();
    m_deviceResources.CopyToRenderTarget(m_output.Get());
    m_deviceResources.m_output = m_output.Get();
}

void AppRenderer::saveOutput(const std::string& filename)
{
    m_saveOutput = true;
    m_saveFileName = filename;
}

void AppRenderer::saveOutputToFile()
{
    D3D12_RESOURCE_DESC desc = m_output->GetDesc();
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
    constexpr uint32_t channels = 4;
    uint32_t imageSize = m_saveRowPitch * m_saveHeight;
    data.resize(imageSize);
    uint8_t* mappedData = nullptr;
    m_outputReadBack->Map(0, nullptr, reinterpret_cast<void**>(&mappedData));
    memcpy(data.data(), mappedData, imageSize);
    m_outputReadBack->Unmap(0, nullptr);
    img::save(m_saveFileName, data.data(), m_saveWidth, m_saveHeight, channels, m_saveRowPitch, format);
}

void AppRenderer::scheduleCopyOutput()
{
    static std::unordered_map<DXGI_FORMAT, uint32_t> Bpp{ {DXGI_FORMAT_R32G32B32A32_FLOAT, 16}, { DXGI_FORMAT_R16G16B16A16_FLOAT, 8}, {DXGI_FORMAT_R8G8B8A8_UNORM, 4} };
    D3D12_RESOURCE_DESC desc = m_output->GetDesc();
    if (Bpp.find(desc.Format) == Bpp.end())
        return;
    m_saveWidth = uint32_t(desc.Width);
    m_saveHeight = uint32_t(desc.Height);
    m_saveRowPitch = Align(m_saveWidth * Bpp[desc.Format], D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    m_deviceResources.CreateBuffer(m_saveRowPitch * desc.Height, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &m_outputReadBack);
    CD3DX12_TEXTURE_COPY_LOCATION src(m_output.Get(), 0);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    footprint.Footprint.Width = m_saveWidth;
    footprint.Footprint.Height = m_saveHeight;
    footprint.Footprint.Depth = 1;
    footprint.Footprint.RowPitch = m_saveRowPitch;
    footprint.Footprint.Format = desc.Format;
    CD3DX12_TEXTURE_COPY_LOCATION dst(m_outputReadBack.Get(), footprint);
    m_deviceResources.computeCommandList()->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}