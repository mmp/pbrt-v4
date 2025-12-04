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

#include "NVSharpen.h"

#include <iostream>
#include <array>

#include "VKUtilities.h"
#include "DeviceResources.h"
#include "Utilities.h"


NVSharpen::NVSharpen(DeviceResources& deviceResources, const std::vector<std::string>& shaderPaths, bool glsl)
    : m_deviceResources(deviceResources)
    , m_outputWidth(1)
    , m_outputHeight(1)
{
    NISOptimizer opt(false, NISGPUArchitecture::NVIDIA_Generic);
    m_blockWidth = opt.GetOptimalBlockWidth();
    m_blockHeight = opt.GetOptimalBlockHeight();
    uint32_t threadGroupSize = opt.GetOptimalThreadGroupSize();

    // Shader
    {
        std::string shaderName = glsl ? "/nis_sharpen_glsl.spv" : "/nis_sharpen.spv";
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

        auto shaderBytes = readBytes(shaderPath);
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = shaderBytes.size();
        info.pCode = reinterpret_cast<uint32_t*>(shaderBytes.data());
        VK_OK(vkCreateShaderModule(m_deviceResources.logicalDevice(), &info, nullptr, &m_shaderModule));
    }

    // Descriptor set
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bindLayout{ {
            VK_COMMON_DESC_LAYOUT(m_deviceResources.sampler()),
        } };

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = (uint32_t)bindLayout.size();
        info.pBindings = bindLayout.data();
        VK_OK(vkCreateDescriptorSetLayout(m_deviceResources.logicalDevice(), &info, nullptr, &m_descriptorSetLayout));
    }
    {
        VkDescriptorSetAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = deviceResources.descriptorPool();
        info.descriptorSetCount = 1;
        info.pSetLayouts = &m_descriptorSetLayout;
        VK_OK(vkAllocateDescriptorSets(m_deviceResources.logicalDevice(), &info, &m_descriptorSet));
    }

    // Constant buffer
    {
        m_deviceResources.createConstBuffer(&m_config, sizeof(NISConfig), &m_buffer, &m_constantBufferDeviceMemory, &m_constantBufferStride);
        VK_OK(vkMapMemory(m_deviceResources.logicalDevice(), m_constantBufferDeviceMemory, 0, m_constantBufferStride, 0, (void**)&m_constantMemory));

        VkDescriptorBufferInfo descBuffInfo{};
        descBuffInfo.buffer = m_buffer;
        descBuffInfo.offset = 0;
        descBuffInfo.range = sizeof(NISConfig);
        VkWriteDescriptorSet writeDescSet{};
        writeDescSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescSet.dstSet = m_descriptorSet;
        writeDescSet.dstBinding = CB_BINDING;
        writeDescSet.descriptorCount = 1;
        writeDescSet.descriptorType = CB_DESC_TYPE;
        writeDescSet.dstArrayElement = 0;
        writeDescSet.pBufferInfo = &descBuffInfo;
        vkUpdateDescriptorSets(m_deviceResources.logicalDevice(), 1, &writeDescSet, 0, nullptr);
    }

    // Pipeline layout
    {
        VkPushConstantRange pushConstRange{};
        pushConstRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstRange.size = sizeof(m_config);
        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = 1;
        info.pSetLayouts = &m_descriptorSetLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &pushConstRange;

        VK_OK(vkCreatePipelineLayout(m_deviceResources.logicalDevice(), &info, nullptr, &m_pipelineLayout));
    }

    // Compute pipeline
    {
        VkPipelineShaderStageCreateInfo pipeShaderStageCreateInfo{};
        pipeShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeShaderStageCreateInfo.module = m_shaderModule;
        pipeShaderStageCreateInfo.pName = "main";

        VkComputePipelineCreateInfo csPipeCreateInfo{};
        csPipeCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        csPipeCreateInfo.stage = pipeShaderStageCreateInfo;
        csPipeCreateInfo.layout = m_pipelineLayout;
        VK_OK(vkCreateComputePipelines(m_deviceResources.logicalDevice(), VK_NULL_HANDLE, 1, &csPipeCreateInfo, nullptr, &m_pipeline));
    }
}

void NVSharpen::cleanUp()
{
    vkDestroyPipeline(m_deviceResources.logicalDevice(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_deviceResources.logicalDevice(), m_pipelineLayout, nullptr);
    vkFreeMemory(m_deviceResources.logicalDevice(), m_constantBufferDeviceMemory, nullptr);
    vkDestroyBuffer(m_deviceResources.logicalDevice(), m_buffer, nullptr);
    vkFreeDescriptorSets(m_deviceResources.logicalDevice(), m_deviceResources.descriptorPool(), 1, &m_descriptorSet);
    vkDestroyDescriptorSetLayout(m_deviceResources.logicalDevice(), m_descriptorSetLayout, nullptr);
    vkDestroyShaderModule(m_deviceResources.logicalDevice(), m_shaderModule, nullptr);
}

void NVSharpen::update(float sharpness, uint32_t inputWidth, uint32_t inputHeight)
{
    NVSharpenUpdateConfig(m_config, sharpness,
        0, 0, inputWidth, inputHeight, inputWidth, inputHeight,
        0, 0, NISHDRMode::None);
    m_outputWidth = inputWidth;
    m_outputHeight = inputHeight;
}

void NVSharpen::dispatch(VkImageView inputSrv, VkImageView outputUav)
{
    const auto offset = m_constantBufferStride * m_deviceResources.swapchainIndex();
    memcpy(m_constantMemory + offset, &m_config, sizeof(m_config));
    VkDescriptorBufferInfo descBuffInfo{};
    descBuffInfo.buffer = m_buffer;
    descBuffInfo.offset = offset;
    descBuffInfo.range = sizeof(NISConfig);

    VkWriteDescriptorSet inWriteDescSet{};
    VkWriteDescriptorSet outWriteDescSet{};
    VkDescriptorImageInfo inDescInfo{};
    inDescInfo.imageView = inputSrv;
    inDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    inWriteDescSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    inWriteDescSet.dstSet = m_descriptorSet;
    inWriteDescSet.dstBinding = IN_TEX_BINDING;
    inWriteDescSet.descriptorCount = 1;
    inWriteDescSet.descriptorType = IN_TEX_DESC_TYPE;
    inWriteDescSet.pImageInfo = &inDescInfo;
    VkDescriptorImageInfo outDescInfo{};
    outDescInfo.imageView = outputUav;
    outDescInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outWriteDescSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    outWriteDescSet.dstSet = m_descriptorSet;
    outWriteDescSet.dstBinding = OUT_TEX_BINDING;
    outWriteDescSet.descriptorCount = 1;
    outWriteDescSet.descriptorType = OUT_TEX_DESC_TYPE;
    outWriteDescSet.pImageInfo = &outDescInfo;
    const VkWriteDescriptorSet writeDescSets[] = {
        inWriteDescSet,
        outWriteDescSet
    };
    constexpr auto sizeWriteDescSets = static_cast<uint32_t>(std::size(writeDescSets));
    vkUpdateDescriptorSets(m_deviceResources.logicalDevice(), sizeWriteDescSets, writeDescSets, 0, nullptr);

    auto cmdBuffer = m_deviceResources.commandBuffer();

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 1, (uint32_t*)&descBuffInfo.offset);

    uint32_t gridX = uint32_t(std::ceil(m_outputWidth / float(m_blockWidth)));
    uint32_t gridY = uint32_t(std::ceil(m_outputHeight / float(m_blockHeight)));
    vkCmdDispatch(cmdBuffer, gridX, gridY, 1);
}