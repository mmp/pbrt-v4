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

#include "DeviceResources.h"
#include "DXUtilities.h"


void DeviceResources::create(HWND hWnd, uint32_t adapterIdx)
{
    DXGI_SWAP_CHAIN_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.BufferCount = 2;
    desc.BufferDesc.Width = 320;
    desc.BufferDesc.Height = 200;
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferDesc.RefreshRate.Numerator = 0;
    desc.BufferDesc.RefreshRate.Denominator = 0;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.OutputWindow = hWnd;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    desc.Windowed = TRUE;

    ComPtr<IDXGIFactory> pFactory;
    CreateDXGIFactory(__uuidof(IDXGIFactory) ,(void**)&pFactory);
    ComPtr<IDXGIAdapter> pAdapter;
    if(pFactory->EnumAdapters(adapterIdx, &pAdapter) != DXGI_ERROR_NOT_FOUND)
    {
        DXGI_ADAPTER_DESC desc;
        pAdapter->GetDesc(&desc);
        char description[256]{};
        snprintf(description, sizeof(description), "%ls", desc.Description);
        m_adapter.Description = description;
        m_adapter.DeviceId = desc.DeviceId;
        m_adapter.VendorId = desc.VendorId;
        m_adapter.DedicatedSystemMemory = desc.DedicatedSystemMemory;
        m_adapter.DedicatedVideoMemory = desc.DedicatedVideoMemory;
        m_adapter.SharedSystemMemory = desc.SharedSystemMemory;
    }
    else
    {
        throw std::runtime_error("Adapter not found");
    }

    uint32_t createDeviceFlags = 0;

#ifdef DX11_ENABLE_DEBUG_LAYER
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };

    HRESULT hr = D3D11CreateDeviceAndSwapChain(pAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
            createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &desc,
            &m_swapChain, &m_d3dDevice, &featureLevel, &m_d3dContext);
    DX::ThrowIfFailed(hr);

    initRenderTarget();
    m_initialized = true;
}

void DeviceResources::initRenderTarget()
{
    DX::ThrowIfFailed(m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), &m_d3dRenderTarget));
    DX::ThrowIfFailed(m_d3dDevice->CreateRenderTargetView(m_d3dRenderTarget.Get(), NULL, &m_d3dRenderTargetView));
}

void DeviceResources::resizeRenderTarget(uint32_t Width, uint32_t Height, DXGI_FORMAT format)
{
    m_width = Width;
    m_height = Height;
    m_d3dRenderTargetUAV = nullptr;
    m_d3dRenderTargetView = nullptr;
    m_d3dRenderTarget = nullptr;
    DX::ThrowIfFailed(m_swapChain->ResizeBuffers(0, Width, Height, format, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH));
    initRenderTarget();
}

void DeviceResources::clearRenderTargetView(const float color[4])
{
    m_d3dContext->ClearRenderTargetView(m_d3dRenderTargetView.Get(), color);
}

void DeviceResources::createUAV(ID3D11Resource* pResource, DXGI_FORMAT format, ID3D11UnorderedAccessView** ppUAView)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
    ZeroMemory(&uavDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
    uavDesc.Format = format;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    DX::ThrowIfFailed(m_d3dDevice->CreateUnorderedAccessView(pResource, &uavDesc, ppUAView));
}

void DeviceResources::createSRV(ID3D11Resource* pResource, DXGI_FORMAT format, ID3D11ShaderResourceView** ppSRView)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
    srvDesc.Format = format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = 1;
    DX::ThrowIfFailed(m_d3dDevice->CreateShaderResourceView(pResource, &srvDesc, ppSRView));
}

void DeviceResources::createLinearClampSampler(ID3D11SamplerState** ppSampleState)
{
    D3D11_SAMPLER_DESC samplerDesc;
    ZeroMemory(&samplerDesc, sizeof(D3D11_SAMPLER_DESC));
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.MipLODBias = 0.0f;
    samplerDesc.MaxAnisotropy = 1;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    DX::ThrowIfFailed(m_d3dDevice->CreateSamplerState(&samplerDesc, ppSampleState));
}

void DeviceResources::createTexture2D(int w, int h, DXGI_FORMAT format, D3D11_USAGE heapType, const void* data, uint32_t rowPitch, uint32_t imageSize, ID3D11Texture2D** ppTexture2D)
{
    D3D11_TEXTURE2D_DESC desc;
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;

    desc.MiscFlags = 0;
    desc.Usage = heapType;
    if (heapType == D3D11_USAGE_STAGING)
    {
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        desc.BindFlags = 0;
    }
    else
    {
        desc.CPUAccessFlags = 0;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
    }

    D3D11_SUBRESOURCE_DATA* pInitialData = nullptr;
    D3D11_SUBRESOURCE_DATA initData;
    if (data)
    {
        initData.pSysMem = data;
        initData.SysMemPitch = rowPitch;
        initData.SysMemSlicePitch = imageSize;
        pInitialData = &initData;
    }

    DX::ThrowIfFailed(m_d3dDevice->CreateTexture2D(&desc, pInitialData, ppTexture2D));
}

void DeviceResources::getTextureData(ID3D11Texture2D* texture, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& rowPitch)
{
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.BindFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    ComPtr<ID3D11Texture2D> stage;
    m_d3dDevice->CreateTexture2D(&desc, nullptr, &stage);
    m_d3dContext->CopyResource(stage.Get(), texture);
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    DX::ThrowIfFailed(m_d3dContext->Map(stage.Get(), 0, D3D11_MAP_READ, 0, &mappedResource));
    uint8_t* mappData = (uint8_t*)mappedResource.pData;
    width = desc.Width;
    height = desc.Height;
    rowPitch = mappedResource.RowPitch;
    data.resize(mappedResource.DepthPitch);
    memcpy(data.data(), mappData, mappedResource.DepthPitch);
    m_d3dContext->Unmap(stage.Get(), 0);
}

void DeviceResources::updateConstBuffer(void* data, uint32_t size, ID3D11Buffer* ppBuffer)
{
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    m_d3dContext->Map(ppBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    uint8_t* mappData = (uint8_t*)mappedResource.pData;
    memcpy(mappData, data, size);
    m_d3dContext->Unmap(ppBuffer, 0);
}

void DeviceResources::createConstBuffer(void* initialData, uint32_t size, ID3D11Buffer** ppBuffer)
{
    D3D11_BUFFER_DESC bDesc;
    bDesc.ByteWidth = size;
    bDesc.Usage = D3D11_USAGE_DYNAMIC;
    bDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bDesc.MiscFlags = 0;
    bDesc.StructureByteStride = 0;

    D3D11_SUBRESOURCE_DATA srData;
    srData.pSysMem = initialData;
    DX::ThrowIfFailed(m_d3dDevice->CreateBuffer(&bDesc, &srData, ppBuffer));
}