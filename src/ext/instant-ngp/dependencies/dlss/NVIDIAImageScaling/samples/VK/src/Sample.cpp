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

#include <filesystem>
#include <iostream>

#include "AppRenderer.h"
#include "DeviceResources.h"
#include "UIRenderer.h"

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "glfw error: %d %s\n", error, description);
    assert(0);
}

int main(int argc, char* argv[])
{
    // Resources
    std::string mediaFolder = "media/images/";

    if (!std::filesystem::exists(mediaFolder))
        mediaFolder = "media/images/";
    if (!std::filesystem::exists(mediaFolder))
        mediaFolder = "../../media/images/";

    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() == GLFW_FALSE)
    {
        fprintf(stderr, "glfwInit() failed\n");
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    // Don't need OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto app_name = "NVIDIA Image Scaling Vulkan Demo";
    GLFWwindow* hwnd = glfwCreateWindow(256, 256, app_name, nullptr, nullptr);
    if (!hwnd)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    DeviceResources deviceResources;
    UIData uiData;

    deviceResources.create(hwnd);

    // Show window and force resize similiar to ::UpdateWindow() in DX11 sample
    glfwShowWindow(hwnd);
    glfwFocusWindow(hwnd);
    glfwSetWindowSize(hwnd, 1280, 1080);

    // UI settings
    uiData.Files = getFiles(mediaFolder);
    if (uiData.Files.size() == 0)
        throw std::runtime_error("No media files");
    uiData.FileName = uiData.Files[0].filename().string();
    uiData.FilePath = uiData.Files[0];

    std::vector<std::string> shaderPaths{ "NIS/", "../../../NIS/", "." };
    bool useGlsl = false;
    AppRenderer appRenderer(deviceResources, uiData, shaderPaths, useGlsl);
    UIRenderer uiRenderer(hwnd, deviceResources, uiData);
    FPS m_fps;

    while (!glfwWindowShouldClose(hwnd))
    {
        glfwPollEvents();

        m_fps.update();
        deviceResources.update();
        uiRenderer.update(m_fps.fps());
        if (appRenderer.update())
        {
            glfwSetWindowSize(hwnd, appRenderer.width(), appRenderer.height());
        }
        deviceResources.beginRender();
        appRenderer.render();
        uiRenderer.render();
        deviceResources.present(uiData.EnableVsync /*ignored*/, 0);
        appRenderer.present();
    }

    uiRenderer.cleanUp();
    appRenderer.cleanUp();
    deviceResources.cleanUp();
    glfwDestroyWindow(hwnd);
    glfwTerminate();

    return EXIT_SUCCESS;
}
