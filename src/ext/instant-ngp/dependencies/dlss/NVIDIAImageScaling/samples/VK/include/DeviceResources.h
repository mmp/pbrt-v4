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

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif // _WIN32

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui_impl_vulkan.h>

#include <stdio.h>
#include <cstdint>
#include <cassert>
#include <vector>

#define IMGUI_VK_HELPERS

#define VK_OK(exp) \
    { \
        const auto _result = (exp); \
        if (_result != VK_SUCCESS) { \
            fprintf(stderr, "%s(%d): %d=%s\n", __FUNCTION__, __LINE__, _result, #exp); \
            assert(0); \
        } \
    }


#define VK_DIE(reason) \
    fprintf(stderr, "%s(%d): %s\n", __FUNCTION__, __LINE__, (reason)); \
    assert(0);

struct PerFrameResources
{
    VkCommandPool       commandPool = VK_NULL_HANDLE;
    VkCommandBuffer     commandBuffer = VK_NULL_HANDLE;
    VkFence             fence = VK_NULL_HANDLE;
    VkImage             backbuffer = VK_NULL_HANDLE;
    VkImageView         backbufferView = VK_NULL_HANDLE;
    VkFramebuffer       framebuffer = VK_NULL_HANDLE;
};

struct FrameSync
{
    VkSemaphore         imageAcquired = VK_NULL_HANDLE;
    VkSemaphore         renderComplete = VK_NULL_HANDLE;
};

class DeviceResources
{
public:
    static const VkFormat SwapchainFormat;
    static const VkColorSpaceKHR SwapchainColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    static const uint32_t NumQueryValues = 2;

    void create(GLFWwindow* hWnd);
    void cleanUp();
    void update();
    void beginRender();
    void resizeRenderTarget(uint32_t Width, uint32_t Height);
    void clearRenderTargetView(const float color[4]) {}
    void present(uint32_t SyncInterval, uint32_t Flags);

    VkInstance instance() { return m_instance; }
    VkPhysicalDevice physicalDevice() { return m_physicalDevice; }
    VkDevice logicalDevice() { return m_device; }
    VkQueue queue() { return m_queue; }
    VkDescriptorPool descriptorPool() { return m_descriptorPool; }
    VkSampler* sampler() { return &m_sampler; }
    VkRenderPass UIrenderPass() { return m_renderPass; }
    VkFramebuffer UIframeBuffer() { return currentFrame()->framebuffer; }
    uint32_t numSwapchainImages() { return static_cast<uint32_t>(m_frames.size()); }
    uint32_t swapchainIndex() { return m_frameIndex; }
    VkCommandBuffer commandBuffer() { return currentFrame()->commandBuffer; }
    VkCommandPool commandPool() { return currentFrame()->commandPool; }
    VkImage backBuffer() { return currentFrame()->backbuffer; }
    VkImageView backBufferView() { return currentFrame()->backbufferView; }
    VkSwapchainKHR swapchain() { return m_swapchain; }
    VkQueryPool queryPool() { return m_queryPool; }

    bool initialized() const { return m_initialized; }

    void requestSwapChainRebuild() { m_swapChainRebuild = true; }
    VkCommandBuffer beginOneTimeSubmitCmd();
    void endOneTimeSubmitCmd();

    void createTexture2D(int w, int h, VkFormat format, const void* data, uint32_t rowPitch, uint32_t imageSize, VkImage* outImage, VkDeviceMemory* outDeviceMemory);
    void createTexture2D(int w, int h, VkFormat format, VkImage* outImage, VkDeviceMemory* outDeviceMemory);
    void createSRV(VkImage inputImage, VkFormat format, VkImageView* outSrv);
    void createConstBuffer(void* initialData, uint32_t size, VkBuffer* outBuffer, VkDeviceMemory* outBuffMem, VkDeviceSize* outOffset);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags buffUsage, VkMemoryPropertyFlags memProps, VkBuffer* outBuffer, VkDeviceMemory* outBuffMem);

    uint32_t minImageCount() const { return m_minImageCount; }
    float timestampPeriod() const { return m_physicalDeviceProperties.limits.timestampPeriod; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
private:
    void createSurfaceResources();
    void destroySurfaceResources();
    void selectPhysicalDeviceAndQueueFamily();
    uint32_t findMemoryTypeIndex(uint32_t memoryTypeBits, VkMemoryPropertyFlags memPropFlags);
    PerFrameResources* currentFrame() { return &m_frames[m_frameIndex]; }
    VkSemaphore imageAcquired() { return m_semaphores[m_semaphoreIndex].imageAcquired; }
    VkSemaphore renderComplete() { return m_semaphores[m_semaphoreIndex].renderComplete; }

    GLFWwindow*                         m_window = nullptr;
    const VkAllocationCallbacks*        m_allocator = nullptr;
    VkInstance                          m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice                    m_physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties          m_physicalDeviceProperties;
    VkDevice                            m_device = VK_NULL_HANDLE;
    VkQueue                             m_queue = VK_NULL_HANDLE;
    VkSurfaceKHR                        m_surface = VK_NULL_HANDLE;
    int                                 m_queueFamilyIndex = -1;
    VkSurfaceFormatKHR                  m_surfaceFormat;
    VkDescriptorPool                    m_descriptorPool = VK_NULL_HANDLE;
    VkSampler                           m_sampler = VK_NULL_HANDLE;
    VkQueryPool                         m_queryPool = VK_NULL_HANDLE;

    VkSwapchainKHR                      m_swapchain = VK_NULL_HANDLE;
    VkRenderPass                        m_renderPass = VK_NULL_HANDLE;

    std::vector<PerFrameResources>      m_frames;
    std::vector<FrameSync>              m_semaphores;
    uint32_t                            m_frameIndex = 0; // Index into m_frames
    uint32_t                            m_semaphoreIndex = 0; // Index into m_semaphores

    uint32_t                            m_minImageCount = 2;
    uint32_t                            m_width;
    uint32_t                            m_height;
    bool                                m_initialized = false;
    bool                                m_swapChainRebuild = false;
};
