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

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>
#include <iostream>
#include <vector>
#include "DXUtilities.h"

using namespace Microsoft::WRL;

constexpr uint32_t NUM_BACK_BUFFERS = 4;

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

class DescriptorHeap
{
public:
    DescriptorHeap() {}
    void Create(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE Type, uint32_t numDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags, LPCWSTR debugHeapName = L"") {
        m_HeapDesc.Type = Type;
        m_HeapDesc.NumDescriptors = numDescriptors;
        m_HeapDesc.Flags = flags;
        m_HeapDesc.NodeMask = 1;

        DX::ThrowIfFailed(device->CreateDescriptorHeap(&m_HeapDesc, __uuidof(ID3D12DescriptorHeap), &m_Heap));
        m_Heap->SetName(debugHeapName);
        m_DescriptorSize = device->GetDescriptorHandleIncrementSize(m_HeapDesc.Type);
        D3D12_CPU_DESCRIPTOR_HANDLE ret = m_Heap->GetCPUDescriptorHandleForHeapStart();
    }
    ID3D12DescriptorHeap* getDescriptorHeap() { return m_Heap.Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE getCPUDescriptorHandle(uint32_t idx) {
        D3D12_CPU_DESCRIPTOR_HANDLE ret = m_Heap->GetCPUDescriptorHandleForHeapStart();
        ret.ptr += m_DescriptorSize * idx;
        return ret;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE getGPUDescriptorHandle(uint32_t idx) {
        D3D12_GPU_DESCRIPTOR_HANDLE ret = m_Heap->GetGPUDescriptorHandleForHeapStart();
        ret.ptr += m_DescriptorSize * idx;
        return ret;
    }
private:
    ComPtr<ID3D12DescriptorHeap> m_Heap;
    D3D12_DESCRIPTOR_HEAP_DESC m_HeapDesc = {};
    uint64_t m_DescriptorSize = 0;
};

class DeviceResources;

class GPUTimer
{
public:
    void Initialize(DeviceResources* deviceResources);
    void StartTimer(ID3D12GraphicsCommandList* commandList) {
        commandList->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
    void StopTimer(ID3D12GraphicsCommandList* commandList) {
        commandList->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    }
    void ResolveQuery(ID3D12GraphicsCommandList* commandList) {
        commandList->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, m_readBackBuffer.Get(), 0);
    }
    void ReadBack();
    double GetTime_us() {
        return static_cast<double>(m_gpuTickDelta * (m_timeEnd - m_timeStart)) * 1E6;
    }
private:
    ComPtr<ID3D12Resource> m_readBackBuffer;
    ComPtr<ID3D12QueryHeap> m_queryHeap;
    uint64_t m_fenceValue = 0;
    uint64_t m_timeStart = 0;
    uint64_t m_timeEnd = 0;
    double m_gpuTickDelta = 0.0;
};

class DeviceResources
{
public:
    void create(HWND hWnd, uint32_t adapterIdx);
    bool isInitialized() { return m_initialized; }
    ID3D12Device* device() { return m_device.Get(); }
    ID3D12DescriptorHeap* SRVDescHeap() { return m_SRVDescHeap.getDescriptorHeap(); }
    ID3D12CommandQueue* commandQueue() { return m_commandQueue.Get(); }
    ID3D12GraphicsCommandList* commandList() { return m_commandList.Get(); }

    void WaitForGPU();
    void MoveToNextFrame();
    void PopulateCommandList();
    void Present(uint32_t SyncInterval, uint32_t Flags);
    void resizeRenderTarget(uint32_t width, uint32_t height);

    void CreateTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format, D3D12_RESOURCE_STATES resourceState, ID3D12Resource** pResource);
    void CreateTexture1D(uint32_t width, DXGI_FORMAT format, D3D12_RESOURCE_STATES resourceState, ID3D12Resource** pResource);
    void CreateBuffer(uint32_t size, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES resourceState, ID3D12Resource** pResource);
    void UploadTextureData(void* data, uint32_t size, uint32_t rowPitch, ID3D12Resource* pResource, ID3D12Resource* pStagingResource);
    void UploadBufferData(void* data, uint32_t size, ID3D12Resource* pResource, ID3D12Resource* pStagingResource);
    void CopyToRenderTarget(ID3D12Resource* pSrc);

    ID3D12Resource* getRenderTarget();
    Adapter adapter() { return m_adapter; }
    ID3D12GraphicsCommandList* computeCommandList() { return m_computeCommandList.Get(); }
    void StartComputeTimer() { m_timer.StartTimer(m_computeCommandList.Get()); }
    void StopComputeTimer() { m_timer.StopTimer(m_computeCommandList.Get()); }
    void ResolveComputeTimerQuery() { m_timer.ResolveQuery(m_computeCommandList.Get()); }
    double GetTime_us() { return m_timer.GetTime_us(); }
    ID3D12Resource* m_output;
protected:
    void CreateRenderTarget();
    void ReleaseRenderTarget();
private:
    struct FrameContext
    {
        ComPtr<ID3D12CommandAllocator> m_allocator;
        ComPtr<ID3D12CommandAllocator> m_computeAllocator;
        ComPtr<ID3D12Fence>            m_fence;
        uint64_t                       m_fenceValue = 0;
    };

    ComPtr<ID3D12Device>               m_device;
    ComPtr<ID3D12CommandQueue>         m_commandQueue;
    ComPtr<ID3D12GraphicsCommandList>  m_commandList;
    ComPtr<ID3D12GraphicsCommandList>  m_computeCommandList;
    ComPtr<ID3D12PipelineState>        m_pipelineState;

    FrameContext                       m_frameContext[NUM_BACK_BUFFERS] = {};

    uint32_t                           m_frameIndex = 0;

    Wrappers::Event                    m_fenceEvent;

    ComPtr<IDXGISwapChain3>            m_swapChain;
    HANDLE                             m_swapChainWaitableObject = nullptr;

    ComPtr<ID3D12Resource>             m_RTResource[NUM_BACK_BUFFERS] = {};
    DescriptorHeap                     m_RTVDescHeap;
    DescriptorHeap                     m_SRVDescHeap;

    GPUTimer                           m_timer;

    bool                               m_initialized = false;
    bool                               m_windowResized = false;
    Adapter                            m_adapter;
};