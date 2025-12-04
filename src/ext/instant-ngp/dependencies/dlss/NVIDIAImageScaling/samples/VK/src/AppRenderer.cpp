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

#include "AppRenderer.h"

#include <array>

AppRenderer::AppRenderer(DeviceResources& deviceResources, UIData& ui, const std::vector<std::string>& shaderPaths, bool glsl)
    : m_ui(ui)
    , m_deviceResources(deviceResources)
    , m_NVSharpen(deviceResources, shaderPaths, glsl)
    , m_NVScaler(deviceResources, shaderPaths, glsl)
{}

bool AppRenderer::update()
{
    bool updateWindowSize = m_currentFilePath != m_ui.FilePath || m_currentScale != m_ui.Scale;
    bool updateSharpeness = m_ui.Sharpness != m_currentSharpness;
    if (updateWindowSize)
    {
        if (m_currentFilePath != m_ui.FilePath)
        {
            img::load(m_ui.FilePath.string(), m_image, m_inputWidth, m_inputHeight, m_rowPitch, img::Fmt::R8G8B8A8);
            if (m_input) {
                vkDestroyImage(m_deviceResources.logicalDevice(), m_input, nullptr);
                vkFreeMemory(m_deviceResources.logicalDevice(), m_inputDeviceMemory, nullptr);
                vkDestroyImageView(m_deviceResources.logicalDevice(), m_inputSRV, nullptr);
            }
            m_deviceResources.createTexture2D(m_inputWidth, m_inputHeight, DeviceResources::SwapchainFormat, m_image.data(), m_rowPitch, m_rowPitch * m_inputHeight, &m_input, &m_inputDeviceMemory);
            m_deviceResources.createSRV(m_input, DeviceResources::SwapchainFormat, &m_inputSRV);
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

        if (m_temp) {
            vkDestroyImage(m_deviceResources.logicalDevice(), m_temp, nullptr);
            vkDestroyImageView(m_deviceResources.logicalDevice(), m_tempSRV, nullptr);
            vkFreeMemory(m_deviceResources.logicalDevice(), m_tempDeviceMemory, nullptr);
        }
        m_deviceResources.createTexture2D(m_outputWidth, m_outputHeight, DeviceResources::SwapchainFormat, &m_temp, &m_tempDeviceMemory);
        m_deviceResources.createSRV(m_temp, DeviceResources::SwapchainFormat, &m_tempSRV);

        m_currentScale = m_ui.Scale;
        m_ui.InputWidth = m_inputWidth;
        m_ui.InputHeight = m_inputHeight;
        m_ui.OutputWidth = m_outputWidth;
        m_ui.OutputHeight = m_outputHeight;
        m_updateWindowSize = true;
    }
    if (updateSharpeness) {
        m_currentSharpness = m_ui.Sharpness;
    }
    if (updateSharpeness || updateWindowSize) {
        m_NVScaler.update(m_currentSharpness / 100.f, m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
        m_NVSharpen.update(m_currentSharpness / 100.f, m_inputWidth, m_inputHeight);
    }
    return updateWindowSize;
}

void AppRenderer::render()
{
    if (m_deviceResources.queryPool() != VK_NULL_HANDLE)
    {
        auto cmdBuff = m_deviceResources.commandBuffer();
        vkCmdResetQueryPool(cmdBuff, m_deviceResources.queryPool(), 0, DeviceResources::NumQueryValues);
        vkCmdWriteTimestamp(cmdBuff, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_deviceResources.queryPool(), 0);
    }

    if (!m_ui.EnableNVScaler)
    {
        blitInputToTemp();
    }
    else
    {
        if (m_ui.Scale == 100)
        {
            m_NVSharpen.dispatch(m_inputSRV, m_tempSRV);
        }
        else
        {
            m_NVScaler.dispatch(m_inputSRV, m_tempSRV);
        }
    }

    if (m_deviceResources.queryPool() != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(m_deviceResources.commandBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_deviceResources.queryPool(), 1);
    }
    blitCopyToRenderTarget();
}

void AppRenderer::present()
{
    if (m_deviceResources.queryPool() != VK_NULL_HANDLE)
    {
        VK_OK(vkDeviceWaitIdle(m_deviceResources.logicalDevice()));
        typedef uint64_t QueryType; // because of VK_QUERY_RESULT_64_BIT
        const uint32_t numQueryResultInts = 2; // because of VK_QUERY_RESULT_WITH_AVAILABILITY_BIT (an additional word)
        const auto stride = numQueryResultInts * sizeof(QueryType); // Bytes for single query result and its availability "bit"
        constexpr auto allQueriesSize = DeviceResources::NumQueryValues * stride;
        std::array<QueryType, numQueryResultInts * DeviceResources::NumQueryValues> query;
        VK_OK(vkGetQueryPoolResults(m_deviceResources.logicalDevice(), m_deviceResources.queryPool(), 0, DeviceResources::NumQueryValues, allQueriesSize, query.data(), stride, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
        const bool checkAvailabilityBits = query[1] != 0 && query[3] != 0;
        if (checkAvailabilityBits)
        {
            const auto topOfPipeTimestamp = query[0]; // "start"
            const auto bottomOfPipeTimestamp = query[2]; // "end"
            m_ui.FilterTime = double(bottomOfPipeTimestamp - topOfPipeTimestamp) * m_deviceResources.timestampPeriod() / 1E3; // ns => us
        }
    }
}

void AppRenderer::cleanUp()
{
    vkDestroyImageView(m_deviceResources.logicalDevice(), m_inputSRV, nullptr);
    vkFreeMemory(m_deviceResources.logicalDevice(), m_inputDeviceMemory, nullptr);
    vkDestroyImage(m_deviceResources.logicalDevice(), m_input, nullptr);
    m_NVScaler.cleanUp();
    m_NVSharpen.cleanUp();
}

void AppRenderer::blitCopyToRenderTarget()
{
    // Layout transition source texture for copy
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_temp;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(m_deviceResources.commandBuffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    // Copy source texture to backbuffer
    VkImageBlit region{};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[0] = { 0, 0, 0 };
    region.srcOffsets[1] = { (int32_t)m_outputWidth, (int32_t)m_outputHeight, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstOffsets[0] = { 0, 0, 0 };
    region.dstOffsets[1] = { (int32_t)m_deviceResources.width(), (int32_t)m_deviceResources.height(), 1 };

    // Must use vkCmdBlitImage instead of vkCmdCopyImage to handle RGBA to BGRA conversion for backbuffer.  Plus, it can also scale.
    vkCmdBlitImage(m_deviceResources.commandBuffer(), m_temp,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_deviceResources.backBuffer(),
        VK_IMAGE_LAYOUT_GENERAL,
        1, &region, VK_FILTER_LINEAR);

    // Layout transition source texture for shader access
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_input;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(m_deviceResources.commandBuffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
}


void AppRenderer::blitInputToTemp()
{
    // Layout transition source texture for copy
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_input;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(m_deviceResources.commandBuffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // Copy source texture to backbuffer
    VkImageBlit region{};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[0] = { 0, 0, 0 };
    region.srcOffsets[1] = { (int32_t)m_inputWidth, (int32_t)m_inputHeight, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstOffsets[0] = { 0, 0, 0 };
    region.dstOffsets[1] = { (int32_t)m_outputWidth, (int32_t)m_outputHeight, 1 };
    
    // Must use vkCmdBlitImage instead of vkCmdCopyImage to handle RGBA to BGRA conversion for backbuffer.  Plus, it can also scale.
    vkCmdBlitImage(m_deviceResources.commandBuffer(), m_input,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_temp,
        VK_IMAGE_LAYOUT_GENERAL,
        1, &region, VK_FILTER_LINEAR);

    // Layout transition source texture for shader access
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_input;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(m_deviceResources.commandBuffer(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
}