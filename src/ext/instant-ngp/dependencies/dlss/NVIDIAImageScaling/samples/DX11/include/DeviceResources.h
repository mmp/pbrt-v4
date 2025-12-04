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
#include <d3d11.h>
#include <wrl.h>
#include <iostream>
#include <vector>

using namespace Microsoft::WRL;

struct Adapter
{
    std::string Description;
    uint32_t VendorId;
    uint32_t DeviceId;
    uint32_t SubSysId;
    uint32_t Revision;
    size_t DedicatedVideoMemory;
    size_t DedicatedSystemMemory;
    size_t SharedSystemMemory;
};

class DeviceResources
{
public:
    void create(HWND hWnd, uint32_t adapterIdx = 0);
    void initRenderTarget();
    void resizeRenderTarget(uint32_t Width, uint32_t Height, DXGI_FORMAT format);
    void clearRenderTargetView(const float color[4]);
    void present(uint32_t SyncInterval, uint32_t Flags) { m_swapChain->Present(SyncInterval, Flags); }
    ID3D11Device* device() { return m_d3dDevice.Get(); }
    ID3D11DeviceContext* context() { return m_d3dContext.Get(); }
    bool isInitialized() { return m_initialized; }

    IDXGISwapChain* swapChain() { return m_swapChain.Get();  }
    ID3D11Texture2D* renderTarget() { return m_d3dRenderTarget.Get(); }
    void DeviceResources::setRenderTarget() {
        m_d3dContext->OMSetRenderTargets(1, m_d3dRenderTargetView.GetAddressOf(), NULL);
    }
    ID3D11RenderTargetView* targetView() { return m_d3dRenderTargetView.Get(); }
    ID3D11RenderTargetView* const* targetViewAddress() { return m_d3dRenderTargetView.GetAddressOf(); }
    ID3D11UnorderedAccessView* const* targetUAVAddress() { return m_d3dRenderTargetUAV.GetAddressOf(); }
    void createTexture2D(int w, int h, DXGI_FORMAT format, D3D11_USAGE heapType, const void* data, uint32_t rowPitch, uint32_t imageSize, ID3D11Texture2D** ppTexture2D);
    void createUAV(ID3D11Resource* pResource, DXGI_FORMAT format, ID3D11UnorderedAccessView** ppUAView);
    void createSRV(ID3D11Resource* pResource, DXGI_FORMAT format, ID3D11ShaderResourceView** ppSRView);
    void createLinearClampSampler(ID3D11SamplerState** ppSampleState);
    void createConstBuffer(void* initialData, uint32_t size, ID3D11Buffer** ppBuffer);
    void updateConstBuffer(void* data, uint32_t size, ID3D11Buffer* ppBuffer);
    void getTextureData(ID3D11Texture2D* texture, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& rowPitch);

    uint32_t width() { return m_width; }
    uint32_t height() { return m_height; }
    Adapter getAdapter() { return m_adapter; }
private:
    ComPtr<ID3D11Device>               m_d3dDevice;
    ComPtr<ID3D11DeviceContext>        m_d3dContext;
    ComPtr<IDXGISwapChain>             m_swapChain;
    ComPtr<ID3D11Texture2D>            m_d3dRenderTarget;
    ComPtr<ID3D11RenderTargetView>	   m_d3dRenderTargetView;
    ComPtr<ID3D11UnorderedAccessView>  m_d3dRenderTargetUAV;
    uint32_t                           m_width;
    uint32_t                           m_height;
    bool                               m_initialized = false;
    Adapter                            m_adapter;
};