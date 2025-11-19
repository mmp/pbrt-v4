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
#pragma comment(lib, "user32")
#pragma comment(lib, "d3d11")
#pragma comment(lib, "dxgi")
#pragma comment(lib, "d3dcompiler")
#pragma comment(lib, "windowscodecs")


#include <d3d11.h>
#include <filesystem>
#include <iostream>
#include <tchar.h>
#include <wincodec.h>

#include "AppRenderer.h"
#include "DeviceResources.h"
#include "UIRenderer.h"
#include "Utilities.h"

DeviceResources deviceResources;
UIData uiData;

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int main(int argc, char* argv[])
{
    // Resources
    std::string mediaFolder = "media/images/";

    if (!std::filesystem::exists(mediaFolder))
        mediaFolder = "media/images/";
    if (!std::filesystem::exists(mediaFolder))
        mediaFolder = "../../media/images/";
    if (!std::filesystem::exists(mediaFolder))
        return -1;

    // Create Window
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr),
                    nullptr, nullptr, nullptr, nullptr, L"NVIDIA Image Scaling Demo", nullptr };
    ::RegisterClassEx(&wc);
    RECT wr = { 0, 0, 1280, 1080 };
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = ::CreateWindow(wc.lpszClassName, L"NVIDIA Image Scaling DX11 Demo", WS_OVERLAPPEDWINDOW,
                    0, 0, wr.right - wr.left, wr.bottom - wr.top, nullptr, nullptr, wc.hInstance, nullptr);
    ::SetWindowLong(hwnd, GWL_STYLE, GetWindowLong(hwnd, GWL_STYLE) & ~WS_SIZEBOX);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    // Initialize DX11
    deviceResources.create(hwnd, 0);

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // UI settings
    uiData.Files = getFiles(mediaFolder);
    if (uiData.Files.size() == 0)
        throw std::runtime_error("No media files");
    uiData.FileName = uiData.Files[0].filename().string();
    uiData.FilePath = uiData.Files[0];

    // Renderers
    std::vector<std::string> shaderPaths{ "NIS/", "../../../NIS/", "../../DX11/src/" };
    AppRenderer appRenderer(deviceResources, uiData, shaderPaths);
    UIRenderer uiRenderer(hwnd, deviceResources, uiData);
    FPS m_fps;

    MSG msg{};
    while (msg.message != WM_QUIT)
    {
        if (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            // press 's' to dump png file
            if (msg.message == WM_KEYDOWN && msg.wParam == 'S')
            {
                appRenderer.saveOutput("dump.png");
            }
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
        }

        // Update
        m_fps.update();
        if (appRenderer.update())
        {
            RECT wr = { 0, 0, LONG(appRenderer.width()), LONG(appRenderer.height()) };
            AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
            SetWindowPos(hwnd, nullptr, 0, 0, wr.right - wr.left, wr.bottom - wr.top, SWP_NOMOVE);
        }
        uiRenderer.update(m_fps.fps());

        // Render
        appRenderer.render();
        uiRenderer.render(); // render UI at the end

        deviceResources.present(uiData.EnableVsync, 0);
    }

    uiRenderer.cleanUp();

    ::DestroyWindow(hwnd);
    ::UnregisterClass(wc.lpszClassName, wc.hInstance);

    return 0;
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (deviceResources.isInitialized() && wParam != SIZE_MINIMIZED)
        {
            deviceResources.resizeRenderTarget((UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_R8G8B8A8_UNORM);
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_KEYUP:
    case WM_SYSKEYUP:
        if (wParam == VK_F1)
            uiData.ShowSettings = !uiData.ShowSettings;
        break;
    case WM_CLOSE:
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}