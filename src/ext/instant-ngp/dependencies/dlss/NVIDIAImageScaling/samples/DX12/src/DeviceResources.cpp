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
#include "d3dx12.h"
#include "Utilities.h"
#include "DXUtilities.h"


void GPUTimer::Initialize(DeviceResources* deviceResources) {
    uint64_t GpuFrequency;
    deviceResources->commandQueue()->GetTimestampFrequency(&GpuFrequency);
    m_gpuTickDelta = 1.0 / static_cast<double>(GpuFrequency);
    deviceResources->CreateBuffer(sizeof(uint64_t) * 2, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &m_readBackBuffer);
    m_readBackBuffer->SetName(L"GpuTimeStamp Buffer");
    D3D12_QUERY_HEAP_DESC QueryHeapDesc;
    QueryHeapDesc.Count = 2;
    QueryHeapDesc.NodeMask = 1;
    QueryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    deviceResources->device()->CreateQueryHeap(&QueryHeapDesc, __uuidof(ID3D12QueryHeap), &m_queryHeap);
    m_queryHeap->SetName(L"GpuTimeStamp QueryHeap");
}

void GPUTimer::ReadBack() {
    uint64_t* mappedBuffer = nullptr;
    D3D12_RANGE range{ 0, sizeof(uint64_t) * 2 };
    m_readBackBuffer->Map(0, &range, reinterpret_cast<void**>(&mappedBuffer));
    m_timeStart = mappedBuffer[0];
    m_timeEnd = mappedBuffer[1];
    m_readBackBuffer->Unmap(0, nullptr);
    if (m_timeEnd < m_timeStart)
    {
        m_timeStart = 0;
        m_timeEnd = 0;
    }
}

void DeviceResources::create(HWND hWnd, uint32_t adapterIdx)
{

    ComPtr<IDXGIFactory> pFactory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
    ComPtr<IDXGIAdapter> pAdapter;
    if (pFactory->EnumAdapters(adapterIdx, &pAdapter) != DXGI_ERROR_NOT_FOUND)
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


#ifdef DX12_ENABLE_DEBUG_LAYER
    ComPtr<ID3D12Debug> pdx12Debug = nullptr;
    if (D3D12GetDebugInterface(__uuidof(ID3D12Debug), &pdx12Debug) == S_OK)
        pdx12Debug->EnableDebugLayer();
#endif

    // Create device
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_0;
    HRESULT hr = D3D12CreateDevice(pAdapter.Get(), featureLevel, __uuidof(ID3D12Device), &m_device);

    // [DEBUG] Setup debug interface to break on any warnings/errors
#ifdef DX12_ENABLE_DEBUG_LAYER
    ComPtr<ID3D12InfoQueue> pInfoQueue;
    m_device->QueryInterface(__uuidof(ID3D12InfoQueue), &pInfoQueue);
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, true);
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_INFO, true);
    pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_MESSAGE, true);
#endif
    {
        // Command Queue
        D3D12_COMMAND_QUEUE_DESC desc = {};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.NodeMask = 1;
        DX::ThrowIfFailed(m_device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), &m_commandQueue));
    }

    {
        // Setup swap chain
        DXGI_SWAP_CHAIN_DESC1 desc;
        ZeroMemory(&desc, sizeof(DXGI_SWAP_CHAIN_DESC1));
        desc.BufferCount = NUM_BACK_BUFFERS;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        desc.Scaling = DXGI_SCALING_NONE;
        desc.Stereo = false;
        ComPtr<IDXGIFactory4> dxgiFactory;
        ComPtr<IDXGISwapChain1> swapChain1;
        DX::ThrowIfFailed(CreateDXGIFactory1(__uuidof(IDXGIFactory4), &dxgiFactory));
        DX::ThrowIfFailed(dxgiFactory->CreateSwapChainForHwnd(m_commandQueue.Get(), hWnd, &desc, nullptr, nullptr, &swapChain1));
        DX::ThrowIfFailed(swapChain1->QueryInterface(__uuidof(IDXGISwapChain3), &m_swapChain));
        m_swapChain->SetMaximumFrameLatency(NUM_BACK_BUFFERS);
        m_swapChainWaitableObject = m_swapChain->GetFrameLatencyWaitableObject();
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
    }

    m_RTVDescHeap.Create(m_device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NUM_BACK_BUFFERS, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    m_SRVDescHeap.Create(m_device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

    for (UINT i = 0; i < NUM_BACK_BUFFERS; i++)
    {
        DX::ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), &m_frameContext[i].m_allocator));
        DX::ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), &m_frameContext[i].m_computeAllocator));
    }

    DX::ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_frameContext[0].m_allocator.Get(), nullptr, __uuidof(ID3D12GraphicsCommandList), &m_commandList));
    DX::ThrowIfFailed(m_commandList->Close());

    DX::ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_frameContext[0].m_computeAllocator.Get(), nullptr, __uuidof(ID3D12CommandList), &m_computeCommandList));
    m_computeCommandList->Close();

    CreateRenderTarget();
    m_initialized = true;

    m_timer.Initialize(this);
}

void DeviceResources::CreateRenderTarget()
{
    for (uint32_t i = 0; i < NUM_BACK_BUFFERS; i++)
    {
        m_swapChain->GetBuffer(i, __uuidof(ID3D12Resource), &m_RTResource[i]);
        m_RTResource[i]->SetName(widen("RenderTarget_" + toStr(i)).c_str());
        m_device->CreateRenderTargetView(m_RTResource[i].Get(), nullptr, m_RTVDescHeap.getCPUDescriptorHandle(i));
        DX::ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), &m_frameContext[i].m_fence));
    }
}

void DeviceResources::ReleaseRenderTarget()
{
    for (uint32_t i = 0; i < NUM_BACK_BUFFERS; i++)
    {
        m_RTResource[i].Reset();
        m_frameContext[i].m_fenceValue = 0;
    }
}

void DeviceResources::resizeRenderTarget(uint32_t width, uint32_t height)
{
    WaitForGPU();
    ReleaseRenderTarget();
    DXGI_MODE_DESC desc;
    desc.Width = width;
    desc.Height = height;
    desc.RefreshRate = DXGI_RATIONAL{ 60, 1 };
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    desc.Scaling = DXGI_MODE_SCALING_STRETCHED;
    DX::ThrowIfFailed(m_swapChain->ResizeTarget(&desc));
    DX::ThrowIfFailed(m_swapChain->ResizeBuffers(NUM_BACK_BUFFERS, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT));
    CreateRenderTarget();
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
    m_windowResized = true;
}

void DeviceResources::WaitForGPU()
{
    for (int i = 0; i < NUM_BACK_BUFFERS; i++)
    {
        FrameContext* ctx = &m_frameContext[i];
        uint64_t fenceValueForSignal = ctx->m_fenceValue + 1;
        m_commandQueue->Signal(ctx->m_fence.Get(), fenceValueForSignal);
        if (ctx->m_fence.Get()->GetCompletedValue() < fenceValueForSignal)
        {
            ctx->m_fence.Get()->SetEventOnCompletion(fenceValueForSignal, m_fenceEvent.Get());
            WaitForSingleObject(m_fenceEvent.Get(), INFINITE);
        }
    }
}

void DeviceResources::PopulateCommandList()
{
    FrameContext* ctx = &m_frameContext[m_frameIndex];
    DX::ThrowIfFailed(ctx->m_allocator->Reset());
    m_commandList->Reset(ctx->m_allocator.Get(), nullptr);
    DX::ThrowIfFailed(ctx->m_computeAllocator->Reset());
    m_computeCommandList->Reset(ctx->m_computeAllocator.Get(), nullptr);

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_RTResource[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    const float clear_color_with_alpha[4] = { 0.45f, 0.55f, 0.6f, 1.f };
    m_commandList->ClearRenderTargetView(m_RTVDescHeap.getCPUDescriptorHandle(m_frameIndex), clear_color_with_alpha, 0, nullptr);
    m_commandList->OMSetRenderTargets(1, &m_RTVDescHeap.getCPUDescriptorHandle(m_frameIndex), false, nullptr);
    std::vector<ID3D12DescriptorHeap*> pHeaps{ m_SRVDescHeap.getDescriptorHeap() };
    m_commandList->SetDescriptorHeaps(uint32_t(pHeaps.size()), pHeaps.data());
}

void DeviceResources::MoveToNextFrame()
{
    FrameContext* ctx = &m_frameContext[m_frameIndex];
    DX::ThrowIfFailed(m_commandQueue->Signal(ctx->m_fence.Get(), ctx->m_fenceValue));
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
    if (ctx->m_fence->GetCompletedValue() < ctx->m_fenceValue)
    {
        DX::ThrowIfFailed(ctx->m_fence->SetEventOnCompletion(ctx->m_fenceValue, m_fenceEvent.Get()));
        WaitForSingleObjectEx(m_fenceEvent.Get(), INFINITE, false);
    }
    ctx->m_fenceValue++;
}

void DeviceResources::Present(uint32_t SyncInterval, uint32_t Flags)
{
    FrameContext* ctx = &m_frameContext[m_frameIndex];
    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_RTResource[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));
    ID3D12CommandList* commandList[] = { m_computeCommandList.Get(), m_commandList.Get() };
    m_commandList->Close();
    m_commandQueue->ExecuteCommandLists(2, commandList);
    if (m_windowResized)
    {
        m_windowResized = false;
        WaitForGPU();
    }
    else
    {
        m_swapChain->Present(SyncInterval, Flags);
        m_timer.ReadBack();
    }
    MoveToNextFrame();
}

void DeviceResources::CreateTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format, D3D12_RESOURCE_STATES resourceState, ID3D12Resource** pResource)
{
    auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    DX::ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc, resourceState, nullptr, __uuidof(ID3D12Resource), (void**)pResource));
}

void DeviceResources::CreateTexture1D(uint32_t width, DXGI_FORMAT format, D3D12_RESOURCE_STATES resourceState, ID3D12Resource** pResource)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Tex1D(format, width, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    DX::ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc, resourceState, nullptr, __uuidof(ID3D12Resource), (void**)pResource));
}

void DeviceResources::CreateBuffer(uint32_t size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES resourceState, ID3D12Resource** pResource)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(size);
    auto heapProperties = CD3DX12_HEAP_PROPERTIES(heapType);
    DX::ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc, resourceState, nullptr, __uuidof(ID3D12Resource), (void**)pResource));
}

void DeviceResources::UploadBufferData(void* data, uint32_t size, ID3D12Resource* pResource, ID3D12Resource* pStagingResource)
{
    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
    uint8_t* mappedData = nullptr;
    pStagingResource->Map(0, nullptr, reinterpret_cast<void**>(&mappedData));
    memcpy(mappedData, data, size);
    pStagingResource->Unmap(0, nullptr);
    m_commandList->CopyBufferRegion(pResource, 0, pStagingResource, 0, size);
    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));
}

void DeviceResources::UploadTextureData(void* data, uint32_t size, uint32_t rowPitch, ID3D12Resource* pResource, ID3D12Resource* pStagingResource)
{
    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
    uint8_t* mappedData = nullptr;
    pStagingResource->Map(0, nullptr, reinterpret_cast<void**>(&mappedData));
    memcpy(mappedData, data, size);
    pStagingResource->Unmap(0, nullptr);
    D3D12_RESOURCE_DESC desc = pResource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    footprint.Footprint.Width = uint32_t(desc.Width);
    footprint.Footprint.Height = uint32_t(desc.Height);
    footprint.Footprint.Depth = 1;
    footprint.Footprint.RowPitch = Align(rowPitch, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    footprint.Footprint.Format = desc.Format;
    CD3DX12_TEXTURE_COPY_LOCATION src(pStagingResource, footprint);
    CD3DX12_TEXTURE_COPY_LOCATION dst(pResource, 0);
    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));
}

ID3D12Resource* DeviceResources::getRenderTarget()
{
    uint32_t backBufferIdx = m_swapChain->GetCurrentBackBufferIndex();
    return m_RTResource[backBufferIdx].Get();
}

void DeviceResources::CopyToRenderTarget(ID3D12Resource* pSrc)
{
    auto pDst = getRenderTarget();
    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pDst, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST));
    D3D12_RESOURCE_DESC desc = pDst->GetDesc();
    CD3DX12_TEXTURE_COPY_LOCATION src(pSrc, 0);
    CD3DX12_TEXTURE_COPY_LOCATION dst(pDst, 0);
    D3D12_BOX box{ 0, 0, 0, uint32_t(desc.Width), uint32_t(desc.Height), 1 };
    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pDst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET));
}