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
#include <algorithm>
#include <array>
#include <vector>
#include <imgui_impl_vulkan.h>

static void glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
    auto devRes = reinterpret_cast<DeviceResources*>(glfwGetWindowUserPointer(window));
    devRes->resizeRenderTarget(width, height);
}

const VkFormat DeviceResources::SwapchainFormat = VK_FORMAT_R8G8B8A8_UNORM;

void DeviceResources::create(GLFWwindow* hWnd)
{
    m_window = hWnd;
    assert(glfwGetWindowUserPointer(hWnd) == nullptr);

    uint32_t extensionCount;
    auto extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (extensions == nullptr)
    {
        VK_DIE("glfwGetRequiredInstanceExtensions");
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.enabledExtensionCount = extensionCount;
    createInfo.ppEnabledExtensionNames = extensions;
    VK_OK(vkCreateInstance(&createInfo, nullptr, &m_instance));

    selectPhysicalDeviceAndQueueFamily();
    VkDeviceQueueCreateInfo qCreateInfo{};
    qCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qCreateInfo.queueFamilyIndex = m_queueFamilyIndex;
    qCreateInfo.queueCount = 1;
    const float qPriority = 1.f;
    qCreateInfo.pQueuePriorities = &qPriority;

    VkPhysicalDeviceFeatures physDevFeatures;
    vkGetPhysicalDeviceFeatures(m_physicalDevice, &physDevFeatures);

    const char* logicalDevExt[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_MAINTENANCE2_EXTENSION_NAME, // For VkImageViewUsageCreateInfo
    };
    VkDeviceCreateInfo devCreateInfo{};
    devCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCreateInfo.pQueueCreateInfos = &qCreateInfo;
    devCreateInfo.queueCreateInfoCount = 1;
    devCreateInfo.pEnabledFeatures = &physDevFeatures;
    devCreateInfo.enabledExtensionCount = static_cast<uint32_t>(std::size(logicalDevExt));
    devCreateInfo.ppEnabledExtensionNames = logicalDevExt;

    VK_OK(vkCreateDevice(m_physicalDevice, &devCreateInfo, nullptr, &m_device));
    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_physicalDeviceProperties);

    vkGetDeviceQueue(m_device, m_queueFamilyIndex, 0, &m_queue);

    //See: vkCreateWin32SurfaceKHR
    VK_OK(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));

    VkBool32 isSurfaceSupported;
    VK_OK(vkGetPhysicalDeviceSurfaceSupportKHR(m_physicalDevice, m_queueFamilyIndex, m_surface, &isSurfaceSupported));
    if (isSurfaceSupported != VK_TRUE)
    {
        VK_DIE("vkGetPhysicalDeviceSurfaceSupportKHR");
    }

    m_surfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(m_physicalDevice, m_surface, &SwapchainFormat, 1, SwapchainColorSpace);

    // Imgui needs a descriptor pool.
    // This is probably excessive, but it's what's in the sample and should cover all our other needs.
    // https://github.com/ocornut/imgui/wiki/Integrating-with-Vulkan
    // https://github.com/ocornut/imgui/blob/master/examples/example_glfw_vulkan/main.cpp
    {
        const uint32_t POOL_COUNT = 1000;

        VkDescriptorPoolSize pool_sizes[] =
        {
            { VK_DESCRIPTOR_TYPE_SAMPLER, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, POOL_COUNT },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, POOL_COUNT }
        };
        constexpr uint32_t NUM_POOLS = static_cast<uint32_t>(std::size(pool_sizes));
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = POOL_COUNT * NUM_POOLS;
        pool_info.poolSizeCount = NUM_POOLS;
        pool_info.pPoolSizes = pool_sizes;
        VK_OK(vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_descriptorPool));
    }

    // Texture sampler
    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.minLod = -1000;
        info.maxLod = 1000;
        info.maxAnisotropy = 1.0f;
        VK_OK(vkCreateSampler(m_device, &info, nullptr, &m_sampler));
    }

    if (m_physicalDeviceProperties.limits.timestampComputeAndGraphics)
    {
        VkQueryPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = DeviceResources::NumQueryValues;
        VK_OK(vkCreateQueryPool(m_device, &info, m_allocator, &m_queryPool));
    }

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    resizeRenderTarget(width, height);

    m_initialized = true;
    glfwSetWindowUserPointer(m_window, this);
    glfwSetWindowSizeCallback(m_window, glfw_window_size_callback);
}

void DeviceResources::createSurfaceResources()
{
    destroySurfaceResources();

    // Swapchain
    {
        const auto oldSwapchain = m_swapchain;

        VkSurfaceCapabilitiesKHR surfCaps{};
        VK_OK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &surfCaps));

        VkSwapchainCreateInfoKHR swapCreateInfo{};
        swapCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapCreateInfo.surface = m_surface;
        swapCreateInfo.minImageCount = surfCaps.minImageCount;
        swapCreateInfo.imageFormat = m_surfaceFormat.format;
        swapCreateInfo.imageColorSpace = m_surfaceFormat.colorSpace;
        swapCreateInfo.imageArrayLayers = 1;
        // `TRANSFER_DST` allows us to blit into the swapchain images
        // `STORAGE` allows us to bind to compute shader output
        swapCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        swapCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapCreateInfo.preTransform = surfCaps.currentTransform;
        swapCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapCreateInfo.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        swapCreateInfo.clipped = VK_TRUE;
        swapCreateInfo.oldSwapchain = oldSwapchain;

        if (surfCaps.currentExtent.width != UINT32_MAX)
        {
            swapCreateInfo.imageExtent = surfCaps.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(m_window, &width, &height);
            swapCreateInfo.imageExtent = {
                std::clamp<uint32_t>(width, surfCaps.minImageExtent.width, surfCaps.maxImageExtent.width),
                std::clamp<uint32_t>(height, surfCaps.minImageExtent.height, surfCaps.maxImageExtent.height)
            };
        }
        m_width = swapCreateInfo.imageExtent.width;
        m_height = swapCreateInfo.imageExtent.height;


        VK_OK(vkCreateSwapchainKHR(m_device, &swapCreateInfo, m_allocator, &m_swapchain));

        if (oldSwapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(m_device, oldSwapchain, m_allocator);
        }
    }

    // Render pass
    {
        VkAttachmentDescription colorAttachDesc {};
        colorAttachDesc.format = m_surfaceFormat.format;
        colorAttachDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        // Don't VK_ATTACHMENT_LOAD_OP_CLEAR because this pass is only used for imgui
        // and compute shader has already populated buffer
        colorAttachDesc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachDesc.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorAttachDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpassDesc{};
        subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.pColorAttachments = &colorRef;
        VkSubpassDependency subpassDep{};
        subpassDep.srcSubpass = VK_SUBPASS_EXTERNAL;
        subpassDep.dstSubpass = 0;
        subpassDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDep.srcAccessMask = 0;
        subpassDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = 1;
        info.pAttachments = &colorAttachDesc;
        info.subpassCount = 1;
        info.pSubpasses = &subpassDesc;
        info.dependencyCount = 1;
        info.pDependencies = &subpassDep;

        VK_OK(vkCreateRenderPass(m_device, &info, m_allocator, &m_renderPass));
    }

    // Swapchain entourage
    {
        uint32_t imageCount;
        VK_OK(vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr));
        std::vector<VkImage> swapchainImages(imageCount);
        VK_OK(vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, swapchainImages.data()));
        m_frames.resize(imageCount);
        m_semaphores.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i)
        {
            m_frames[i].backbuffer = swapchainImages[i];

            // Image views
            {
                const VkImageViewUsageCreateInfo usageInfo{
                    VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                    nullptr,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                };
                VkImageViewCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                info.pNext = &usageInfo;
                info.image = m_frames[i].backbuffer;
                info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                info.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
                info.format = m_surfaceFormat.format;
                info.subresourceRange.layerCount = 1;
                info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                info.subresourceRange.levelCount = 1;
                VK_OK(vkCreateImageView(m_device, &info, m_allocator, &m_frames[i].backbufferView));
            }

            // Framebuffers
            {
                VkFramebufferCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                info.renderPass = m_renderPass;
                info.attachmentCount = 1;
                info.pAttachments = &m_frames[i].backbufferView;
                info.width = m_width;
                info.height = m_height;
                info.layers = 1;
                VK_OK(vkCreateFramebuffer(m_device, &info, m_allocator, &m_frames[i].framebuffer));
            }

            // Command pool
            {
                VkCommandPoolCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                info.queueFamilyIndex = m_queueFamilyIndex;
                VK_OK(vkCreateCommandPool(m_device, &info, m_allocator, &m_frames[i].commandPool));
            }
            {
                VkCommandBufferAllocateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                info.commandPool = m_frames[i].commandPool;
                info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                info.commandBufferCount = 1;
                VK_OK(vkAllocateCommandBuffers(m_device, &info, &m_frames[i].commandBuffer));
            }
            {
                VkFenceCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
                VK_OK(vkCreateFence(m_device, &info, m_allocator, &m_frames[i].fence));
            }

            // Per-frame semaphores
            {
                VkSemaphoreCreateInfo info = {};
                info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                VK_OK(vkCreateSemaphore(m_device, &info, m_allocator, &m_semaphores[i].imageAcquired));
                VK_OK(vkCreateSemaphore(m_device, &info, m_allocator, &m_semaphores[i].renderComplete));
            }
        }
    }
}

void DeviceResources::destroySurfaceResources()
{
    VK_OK(vkDeviceWaitIdle(m_device));

    for (auto& frame : m_frames)
    {
        vkDestroyFence(m_device, frame.fence, m_allocator);
        vkFreeCommandBuffers(m_device, frame.commandPool, 1, &frame.commandBuffer);
        vkDestroyCommandPool(m_device, frame.commandPool, m_allocator);
        vkDestroyFramebuffer(m_device, frame.framebuffer, m_allocator);
        vkDestroyImageView(m_device, frame.backbufferView, m_allocator);
    }
    m_frames.clear();
    for (auto& frame : m_semaphores)
    {
        vkDestroySemaphore(m_device, frame.imageAcquired, m_allocator);
        vkDestroySemaphore(m_device, frame.renderComplete, m_allocator);
    }
    m_semaphores.clear();

    if (m_renderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(m_device, m_renderPass, m_allocator);
        m_renderPass = VK_NULL_HANDLE;
    }

    // NB: swapchain isn't destroyed because this is called from
    // createSurfaceResources() where we need old swapchain when recreating it
}

void DeviceResources::cleanUp()
{
    m_initialized = false;
    glfwSetWindowUserPointer(m_window, nullptr);

    destroySurfaceResources();

    vkDestroyQueryPool(m_device, m_queryPool, m_allocator);
    vkDestroySampler(m_device, m_sampler, m_allocator);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, m_allocator);

    vkDestroySwapchainKHR(m_device, m_swapchain, m_allocator);
    vkDestroySurfaceKHR(m_instance, m_surface, m_allocator);
    vkDestroyDevice(m_device, m_allocator);
    vkDestroyInstance(m_instance, m_allocator);
}

void DeviceResources::selectPhysicalDeviceAndQueueFamily()
{
    uint32_t deviceCount = 0;
    VK_OK(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr));
    if (deviceCount == 0) {
        VK_DIE("vkEnumeratePhysicalDevices");
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    VK_OK(vkEnumeratePhysicalDevices(m_instance, &deviceCount, physicalDevices.data()));
    for (const auto& physDev : physicalDevices)
    {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, queueFamilies.data());
        for (auto i = 0; i < queueFamilies.size(); ++i)
        {
            const auto& qFamily = queueFamilies[i];
            // To make things a bit simpler, we want a single queue that does everything we need
            if ((qFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
                && (qFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)
                && glfwGetPhysicalDevicePresentationSupport(m_instance, physDev, i))
            {
                // Additionally can:
                // - vkGetPhysicalDeviceSurfaceFormatsKHR
                // - vkGetPhysicalDeviceSurfacePresentModesKHR
                m_physicalDevice = physDev;
                m_queueFamilyIndex = i;
                return;
            }
        }
    }
    VK_DIE("selectPhysicalDeviceAndQueueFamily");
}

uint32_t DeviceResources::findMemoryTypeIndex(uint32_t memoryTypeBits, VkMemoryPropertyFlags memPropFlags)
{
    VkPhysicalDeviceMemoryProperties physDevMemProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &physDevMemProps);
    for (uint32_t i = 0; i < physDevMemProps.memoryTypeCount; ++i)
    {
        if ((memoryTypeBits & (1 << i))
            && (physDevMemProps.memoryTypes[i].propertyFlags & memPropFlags) == memPropFlags)
        {
            return i;
        }
    }
    VK_DIE("findMemoryTypeIndex");
    return -1;
}

void DeviceResources::createConstBuffer(void* initialData, uint32_t size, VkBuffer* outBuffer, VkDeviceMemory* outBuffMem, VkDeviceSize* outOffset)
{
    const auto align = m_physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
    *outOffset = ((size + align - 1) / align) * align;
    createBuffer(*outOffset * numSwapchainImages(),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        outBuffer, outBuffMem);
}

void DeviceResources::createBuffer(VkDeviceSize size, VkBufferUsageFlags buffUsage, VkMemoryPropertyFlags memProps, VkBuffer* outBuffer, VkDeviceMemory* outBuffMem)
{
    {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = buffUsage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_OK(vkCreateBuffer(m_device, &info, nullptr, outBuffer));
    }
    {
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(m_device, *outBuffer, &memReqs);

        VkMemoryAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize = memReqs.size;
        info.memoryTypeIndex = findMemoryTypeIndex(memReqs.memoryTypeBits, memProps);
        VK_OK(vkAllocateMemory(m_device, &info, nullptr, outBuffMem));
    }

    VK_OK(vkBindBufferMemory(m_device, *outBuffer, *outBuffMem, 0));
}

void DeviceResources::resizeRenderTarget(uint32_t Width, uint32_t Height)
{
    createSurfaceResources();
}

void DeviceResources::update()
{
    if (m_swapChainRebuild)
    {
        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);
        if (width > 0 && height > 0)
        {
            ImGui_ImplVulkan_SetMinImageCount(m_minImageCount);
            resizeRenderTarget(width, height);
            m_frameIndex = 0;
            m_swapChainRebuild = false;
        }
    }
}

void DeviceResources::beginRender()
{
    auto image_acquired_semaphore = imageAcquired();
    auto render_complete_semaphore = renderComplete();
    VkResult err = vkAcquireNextImageKHR(m_device, swapchain(), UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &m_frameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    {
        requestSwapChainRebuild();
        return;
    }

    auto frame = currentFrame();
    auto cmdBuff = commandBuffer();
    {
        VK_OK(vkWaitForFences(m_device, 1, &frame->fence, VK_TRUE, UINT64_MAX)); // wait indefinitely
        VK_OK(vkResetFences(m_device, 1, &frame->fence));
    }
    {
        VK_OK(vkResetCommandPool(m_device, frame->commandPool, 0));
        VkCommandBufferBeginInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_OK(vkBeginCommandBuffer(cmdBuff, &info));
    }

    // Layout transition backbuffer
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.image = backBuffer();
        vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
}

void DeviceResources::present(uint32_t SyncInterval, uint32_t Flags)
{
    if (m_swapChainRebuild)
        return;

    auto cmdBuff = commandBuffer();

    // Layout transition backbuffer for present
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.image = backBuffer();
        vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    auto image_acquired_semaphore = imageAcquired();
    auto render_complete_semaphore = renderComplete();
    auto frame = currentFrame();

    // Submit command buffer
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &image_acquired_semaphore;
        info.pWaitDstStageMask = &wait_stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &cmdBuff;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &render_complete_semaphore;

        VK_OK(vkEndCommandBuffer(cmdBuff));
        VK_OK(vkQueueSubmit(m_queue, 1, &info, frame->fence));
    }

    // Frame present
    VkPresentInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &m_swapchain;
    info.pImageIndices = &m_frameIndex;
    VkResult err = vkQueuePresentKHR(m_queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    {
        requestSwapChainRebuild();
        return;
    }
    assert(err == VK_SUCCESS);
    m_semaphoreIndex = (m_semaphoreIndex + 1) % numSwapchainImages(); // Now we can use the next set of semaphores

    // SwapBuffers not needed with Vulkan
    //glfwSwapBuffers(hwnd);
}

VkCommandBuffer DeviceResources::beginOneTimeSubmitCmd()
{
    VkCommandBuffer cmdBuff = commandBuffer();

    VK_OK(vkResetCommandPool(m_device, commandPool(), 0));
    VkCommandBufferBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_OK(vkBeginCommandBuffer(cmdBuff, &info));

    return cmdBuff;
}

void DeviceResources::endOneTimeSubmitCmd()
{
    VkCommandBuffer cmdBuff = commandBuffer();
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmdBuff;
    VK_OK(vkEndCommandBuffer(cmdBuff));
    VK_OK(vkQueueSubmit(m_queue, 1, &info, VK_NULL_HANDLE));
    VK_OK(vkDeviceWaitIdle(m_device));
}

void DeviceResources::createTexture2D(int w, int h, VkFormat format, const void* data, uint32_t rowPitch, uint32_t imageSize, VkImage* outImage, VkDeviceMemory* outDeviceMemory)
{
    auto width = static_cast<uint32_t>(w);
    auto height = static_cast<uint32_t>(h);

    VkBuffer stagingBuff;
    VkDeviceMemory stagingBuffMem;

    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &stagingBuff, &stagingBuffMem);

    void* mappedMem;
    VK_OK(vkMapMemory(m_device, stagingBuffMem, 0, imageSize, 0, &mappedMem));
    memcpy(mappedMem, data, imageSize);
    vkUnmapMemory(m_device, stagingBuffMem);

    // Create texture image object and backing memory
    {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.extent.width = width;
        info.extent.height = height;
        info.extent.depth = 1;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.format = format;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.samples = VK_SAMPLE_COUNT_1_BIT;

        VK_OK(vkCreateImage(m_device, &info, nullptr, outImage));
    }
    {
        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(m_device, *outImage, &memReq);

        VkMemoryAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize = memReq.size;
        info.memoryTypeIndex = findMemoryTypeIndex(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_OK(vkAllocateMemory(m_device, &info, nullptr, outDeviceMemory));

        VK_OK(vkBindImageMemory(m_device, *outImage, *outDeviceMemory, 0));
    }
    VkCommandBuffer cmdBuff = beginOneTimeSubmitCmd();
    {
        VkImageMemoryBarrier transferBarrier{};
        transferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        transferBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        transferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        transferBarrier.srcAccessMask = 0;
        transferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        transferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        transferBarrier.image = *outImage;
        transferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        transferBarrier.subresourceRange.levelCount = 1;
        transferBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transferBarrier);

        VkBufferImageCopy buffImageCopyRegion{};
        buffImageCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        buffImageCopyRegion.imageSubresource.layerCount = 1;
        buffImageCopyRegion.imageExtent = { width, height, 1 };
        vkCmdCopyBufferToImage(cmdBuff, stagingBuff, *outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &buffImageCopyRegion);

        VkImageMemoryBarrier useBarrier{};
        useBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        useBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        useBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        useBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        useBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        useBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        useBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        useBarrier.image = *outImage;
        useBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        useBarrier.subresourceRange.levelCount = 1;
        useBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &useBarrier);
    }
    endOneTimeSubmitCmd();

    vkFreeMemory(m_device, stagingBuffMem, nullptr);
    vkDestroyBuffer(m_device, stagingBuff, nullptr);
}

void DeviceResources::createTexture2D(int w, int h, VkFormat format, VkImage* outImage, VkDeviceMemory* outDeviceMemory)
{
    auto width = static_cast<uint32_t>(w);
    auto height = static_cast<uint32_t>(h);

    // Create texture image object and backing memory
    {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.extent.width = width;
        info.extent.height = height;
        info.extent.depth = 1;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.format = format;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.samples = VK_SAMPLE_COUNT_1_BIT;

        VK_OK(vkCreateImage(m_device, &info, nullptr, outImage));
    }
    {
        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(m_device, *outImage, &memReq);

        VkMemoryAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize = memReq.size;
        info.memoryTypeIndex = findMemoryTypeIndex(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_OK(vkAllocateMemory(m_device, &info, nullptr, outDeviceMemory));

        VK_OK(vkBindImageMemory(m_device, *outImage, *outDeviceMemory, 0));
    }
}

void DeviceResources::createSRV(VkImage inputImage, VkFormat format, VkImageView* outSrv)
{
    VkImageViewCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = inputImage;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.layerCount = 1;
    info.subresourceRange.levelCount = 1;
    VK_OK(vkCreateImageView(m_device, &info, nullptr, outSrv));
}
