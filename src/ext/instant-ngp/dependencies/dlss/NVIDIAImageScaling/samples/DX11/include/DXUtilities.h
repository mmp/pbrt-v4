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

#include <stdio.h>
#include <tchar.h>
#include <dxgi1_4.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace DX
{
    using namespace Microsoft::WRL;

    inline LPTSTR GetErrorDescription(HRESULT hr, WCHAR* buffer, size_t size)
    {
        if (FACILITY_WINDOWS == HRESULT_FACILITY(hr))
            hr = HRESULT_CODE(hr);

        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,
            nullptr, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buffer, DWORD(size), nullptr);
        return buffer;
    }

    inline void ThrowIfFailed(HRESULT hr)
    {
        if (FAILED(hr))
        {
            // Set a breakpoint on this line to catch Win32 API errors.
            const size_t size = 1024;
            WCHAR buffer[size];
            GetErrorDescription(hr, buffer, size);
            char str[size];
            size_t i = 0;
            wcstombs_s(&i, str, 1024, buffer, 1024);
            throw std::runtime_error(str);
        }
    }

    inline void CompileComputeShader(ID3D11Device* device,
        LPCWSTR pFileName,
        LPCSTR pEntryPoint,
        ID3D11ComputeShader** csShader,
        const D3D_SHADER_MACRO* pDefines = nullptr,
        ID3DInclude* pInclude = nullptr,
        LPCSTR pTarget = "cs_5_0")
    {
        ComPtr<ID3DBlob> csBlob;
        ComPtr<ID3DBlob> cdErrorBlob = nullptr;
        HRESULT hr = D3DCompileFromFile(pFileName, pDefines, pInclude, pEntryPoint, pTarget, 0, 0, &csBlob, &cdErrorBlob);
        if (FAILED(hr)) {
            if (cdErrorBlob) {
                OutputDebugStringA((char*)cdErrorBlob->GetBufferPointer());
            }
            DX::ThrowIfFailed(hr);
        }
        DX::ThrowIfFailed(device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, csShader));
    }

    inline void CompileComputeShader(ID3D11Device* device,
        LPCVOID pSrcData,
        size_t SrcDataSize,
        LPCSTR pEntrypoint,
        ID3D11ComputeShader** csShader,
        const D3D_SHADER_MACRO* pDefines = nullptr,
        LPCSTR pTarget = "cs_5_0")
    {
        ComPtr<ID3DBlob> csBlob;
        ComPtr<ID3DBlob> cdErrorBlob = nullptr;

        HRESULT hr = D3DCompile(pSrcData, SrcDataSize, nullptr, pDefines, nullptr, pEntrypoint, pTarget, 0, 0, &csBlob, &cdErrorBlob);
        if (FAILED(hr)) {
            if (cdErrorBlob) {
                OutputDebugStringA((char*)cdErrorBlob->GetBufferPointer());
            }
            DX::ThrowIfFailed(hr);
        }
        DX::ThrowIfFailed(device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, csShader));
    }

    struct IncludeHeader : ID3DInclude {
        IncludeHeader(const std::vector<std::string>& includePath)
            : m_includePath(includePath)
            , m_idx(0) {}

        HRESULT Open(
            D3D_INCLUDE_TYPE IncludeType,
            LPCSTR           pFileName,
            LPCVOID          pParentData,
            LPCVOID* ppData,
            UINT* pBytes
        ) {
            m_data.push_back("");
            std::ifstream t;
            size_t i = 0;
            while (!t.is_open() && i < m_includePath.size()) {
                t.open(m_includePath[i] + "/" + pFileName);
                i++;
            }
            if (!t.is_open())
                throw std::runtime_error("Error opening D3DCompileFromFile include header");
            t.seekg(0, std::ios::end);
            size_t size = t.tellg();
            m_data[m_idx].resize(size);
            t.seekg(0, std::ios::beg);
            t.read(m_data[m_idx].data(), size);
            m_data[m_idx].erase(std::remove(m_data[m_idx].begin(), m_data[m_idx].end(), '\0'), m_data[m_idx].end());
            *ppData = m_data[m_idx].data();
            *pBytes = UINT(m_data[m_idx].size());
            m_idx++;
            return S_OK;
        }

        HRESULT Close(LPCVOID pData) {
            return S_OK;
        }

        std::vector<std::string> m_data;
        std::vector<std::string> m_includePath;
        size_t m_idx;
    };

    class Defines {
    public:
        template<typename T>
        void add(const std::string& define, const T& val) {
            m_definesVector.push_back({ define, toStr(val) });
        }
        D3D_SHADER_MACRO* get() {
            m_defines = std::make_unique<D3D_SHADER_MACRO[]>(m_definesVector.size() + 1);
            for (size_t i = 0; i < m_definesVector.size(); ++i)
                m_defines[i] = { m_definesVector[i].first.c_str(), m_definesVector[i].second.c_str() };
            m_defines[m_definesVector.size()] = { nullptr, nullptr };
            return m_defines.get();
        }
    private:
        std::vector<std::pair<std::string, std::string>> m_definesVector;
        std::unique_ptr<D3D_SHADER_MACRO[]> m_defines;
    };
}