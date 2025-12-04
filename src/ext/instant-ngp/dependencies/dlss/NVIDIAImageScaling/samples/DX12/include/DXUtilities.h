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
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl.h>
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
    inline std::vector<uint8_t> ReadBinaryFile(const std::string& filename)
    {
        std::ifstream ifs(filename, std::ios::in | std::ios::binary | std::ios::ate);
        if (!ifs)
            throw std::runtime_error("ReadBinaryFile");
        std::streampos len = ifs.tellg();
        std::vector<uint8_t> data;
        data.resize(size_t(len));
        ifs.seekg(0, std::ios::beg);
        ifs.read(reinterpret_cast<char*>(data.data()), len);
        ifs.close();
        return data;
    }
}