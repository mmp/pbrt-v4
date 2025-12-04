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

BilinearUpscale::BilinearUpscale(DeviceResources& deviceResources, const std::vector<std::string>& shaderPaths)
    : m_deviceResources(deviceResources)
    , m_outputWidth(0)
    , m_outputHeight(0)
{
    std::string shaderName = "bilinearUpscale.hlsl";
    std::string shaderPath;
    for (auto& e : shaderPaths)
    {
        if (std::filesystem::exists(e + "/" + shaderName))
        {
            shaderPath = e + "/" + shaderName;
            break;
        }
    }
    if (shaderPath.empty())
        throw std::runtime_error("Shader file not found" + shaderName);

    ComPtr<IDxcLibrary> library;
    DX::ThrowIfFailed(DxcCreateInstance(CLSID_DxcLibrary, __uuidof(IDxcLibrary), &library));
    ComPtr<IDxcCompiler> compiler;
    DX::ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler), &compiler));
    std::wstring wShaderFilename = widen(shaderPath);

    uint32_t codePage = CP_UTF8;
    ComPtr<IDxcBlobEncoding> sourceBlob;
    DX::ThrowIfFailed(library->CreateBlobFromFile(wShaderFilename.c_str(), &codePage, &sourceBlob));

    constexpr uint32_t nDefines = 2;
    std::wstring wBlockWidth = widen(toStr(m_BlockWidth));
    std::wstring wBlockHeight = widen(toStr(m_BlockHeight));
    DxcDefine defines[nDefines] = {
        {L"BLOCK_WIDTH", wBlockWidth.c_str()},
        {L"BLOCK_HEIGHT", wBlockHeight.c_str()},
    };

    ComPtr<IDxcOperationResult> result;
    HRESULT hr = compiler->Compile(sourceBlob.Get(), wShaderFilename.c_str(), L"main", L"cs_6_2", nullptr, 0, defines, nDefines, nullptr, &result);
    if (SUCCEEDED(hr))
    {
        result->GetStatus(&hr);
    }
    if (FAILED(hr))
    {
        if (result)
        {
            ComPtr<IDxcBlobEncoding> errorsBlob;
            hr = result->GetErrorBuffer(&errorsBlob);
            if (SUCCEEDED(hr) && errorsBlob)
            {
                wprintf(L"Compilation failed with errors:\n%hs\n", (const char*)errorsBlob->GetBufferPointer());
            }
        }
        DX::ThrowIfFailed(hr);
    }
    ComPtr<IDxcBlob> computeShaderBlob;
    result->GetResult(&computeShaderBlob);

    m_deviceResources.CreateBuffer(sizeof(BilinearUpscaleConfig), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, &m_stagingBuffer);
    m_deviceResources.CreateBuffer(sizeof(BilinearUpscaleConfig), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &m_constatBuffer);

    // Define root table layout
    constexpr uint32_t nParams = 4;
    CD3DX12_DESCRIPTOR_RANGE descriptorRange[nParams] = {};
    descriptorRange[0] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
    descriptorRange[1] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);
    descriptorRange[2] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    descriptorRange[3] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    CD3DX12_ROOT_PARAMETER m_rootParams[nParams] = {};
    m_rootParams[0].InitAsDescriptorTable(1, &descriptorRange[0]);
    m_rootParams[1].InitAsDescriptorTable(1, &descriptorRange[1]);
    m_rootParams[2].InitAsDescriptorTable(1, &descriptorRange[2]);
    m_rootParams[3].InitAsDescriptorTable(1, &descriptorRange[3]);

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = nParams;
    rootSignatureDesc.pParameters = m_rootParams;
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.pStaticSamplers = nullptr;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serializedSignature;
    DX::ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serializedSignature, nullptr));

    // Create the root signature
    DX::ThrowIfFailed(
        m_deviceResources.device()->CreateRootSignature(
            0,
            serializedSignature->GetBufferPointer(),
            serializedSignature->GetBufferSize(),
            __uuidof(ID3D12RootSignature),
            &m_computeRootSignature));
    m_computeRootSignature->SetName(L"BilinearUpscale");

    D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
    descComputePSO.pRootSignature = m_computeRootSignature.Get();
    descComputePSO.CS.pShaderBytecode = computeShaderBlob->GetBufferPointer();
    descComputePSO.CS.BytecodeLength = computeShaderBlob->GetBufferSize();

    DX::ThrowIfFailed(
        m_deviceResources.device()->CreateComputePipelineState(&descComputePSO, __uuidof(ID3D12PipelineState), &m_computePSO));
    m_computePSO->SetName(L"BilinearUpscale Compute PSO");
}

void BilinearUpscale::update(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight)
{
    BilinearUpdateConfig(m_config, 0, 0, inputWidth, inputHeight, inputWidth, inputHeight, 0, 0, outputWidth, outputHeight, outputWidth, outputHeight);
    m_deviceResources.UploadBufferData(&m_config, sizeof(BilinearUpscaleConfig), m_constatBuffer.Get(), m_stagingBuffer.Get());
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;
}