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
#include <d3d11_3.h>
#include <d3dcompiler.h>
#include <iostream>

#include "DXUtilities.h"
#include "DeviceResources.h"
#include "Utilities.h"
#include "NIS_Config.h"

class NVScaler {
public:
    NVScaler(DeviceResources& deviceResources, const std::vector<std::string>& shaderPaths);

    void update(float sharpness, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight);

    void dispatch(ID3D11ShaderResourceView* const* input, ID3D11UnorderedAccessView* const* output);
private:
    DeviceResources& m_deviceResources;
    NISConfig                           m_config;
    ComPtr<ID3D11ComputeShader>         m_computeShader;
    ComPtr<ID3D11Buffer>                m_csBuffer;
    ComPtr<ID3D11Texture2D>				m_coef_scale;
    ComPtr<ID3D11Texture2D>				m_coef_usm;
    ComPtr<ID3D11ShaderResourceView>    m_coef_scaleSRV;
    ComPtr<ID3D11ShaderResourceView>    m_coef_usmSRV;
    ComPtr<ID3D11SamplerState>			m_LinearClampSampler;
    uint32_t                            m_outputWidth;
    uint32_t                            m_outputHeight;
    uint32_t                            m_blockWidth;
    uint32_t                            m_blockHeight;
};