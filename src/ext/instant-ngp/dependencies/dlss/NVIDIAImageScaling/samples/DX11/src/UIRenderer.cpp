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

#include <UIRenderer.h>

UIRenderer::UIRenderer(HWND hwnd, DeviceResources& deviceResources, UIData& ui)
: m_ui(ui)
, m_deviceResources(deviceResources)
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 6;
    ImGui::GetStyle().FrameBorderSize = 1;

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(m_deviceResources.device(), m_deviceResources.context());
}

void UIRenderer::cleanUp()
{
    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

void UIRenderer::update(double fps)
{
    m_elapsedTimer.start();
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    if (m_ui.ShowSettings)
    {
        ImGui::Begin("Settings", 0, ImGuiWindowFlags_AlwaysAutoResize);
        if (ImGui::CollapsingHeader("GPU Info", ImGuiTreeNodeFlags_DefaultOpen))
        {
            Adapter adapter = m_deviceResources.getAdapter();
            ImGui::Text("Description: %s", adapter.Description.c_str());
            ImGui::Text("VendorID: 0x%x DeviceID: 0x%x", adapter.VendorId, adapter.DeviceId);
            ImGui::Text("DedicatedSystemMemory: %d [MB]", adapter.DedicatedVideoMemory / 1024 / 1024);
            ImGui::Text("SharedSystemMemory   : %d [MB]", adapter.SharedSystemMemory / 1024 / 1024);
        }
        if (ImGui::CollapsingHeader("Images", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::BeginCombo("Filename", m_ui.FileName.c_str()))
            {
                for (auto& e : m_ui.Files)
                {
                    bool is_selected = (m_ui.FileName == e.filename().string());
                    if (ImGui::Selectable(e.filename().string().c_str(), is_selected))
                    {
                        m_ui.FileName = e.filename().string();
                        m_ui.FilePath = e;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }
        if (ImGui::CollapsingHeader("Filter", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::RadioButton("NVScaler", &m_ui.FilterMode, 0); ImGui::SameLine();
            ImGui::RadioButton("Bilinear", &m_ui.FilterMode, 2); ImGui::SameLine();
            ImGui::RadioButton("NVSharpen", &m_ui.FilterMode, 1);
            ;
            ImGui::Separator();
            if (m_ui.FilterMode == 0 || m_ui.FilterMode == 1)
            {
                m_ui.EnableNVScaler = true;
                ImGui::SliderFloat("Sharpness (0% - 100%)", &m_ui.Sharpness, 0, 100, "%2.1f%%");
            }
            else
            {
                m_ui.EnableNVScaler = false;
            }
            if (m_ui.FilterMode == 0 || m_ui.FilterMode == 2)
            {
                ImGui::Separator();
                std::vector<const char*> outputSizes = { "Variable", "1920x1080", "2560x1440", "3840x2160" };
                ImGui::Combo("Height Size", (int*)&m_ui.OutputMode, outputSizes.data(), int(outputSizes.size()));
                float fixScaleSize = 0;
                switch (m_ui.OutputMode) {
                case OutputSizeMode::VARIABLE:
                    ImGui::SliderFloat("Scale (50% - 100%)", &m_ui.Scale, 50, 100, "%2.1f%%");
                    break;
                case OutputSizeMode::P1080:
                    fixScaleSize = 1080.f;
                    break;
                case OutputSizeMode::P1440:
                    fixScaleSize = 1440.f;
                    break;
                case OutputSizeMode::P2160:
                    fixScaleSize = 2160.f;
                    break;
                }
                if (fixScaleSize > 0)
                {
                    m_ui.Scale = std::min<float>(100.f, std::max<float>(50.f, m_ui.InputHeight / fixScaleSize * 100.f));
                    ImGui::Text("Fix Scale : %2.1f%%", m_ui.Scale);
                }
            }
            else
            {
                m_ui.Scale = 100;
            }
            ImGui::Separator();
            if (m_ui.Scale == 100)
            {
                if (m_ui.EnableNVScaler)
                {
                    ImGui::Text("Using NVSharpen shader:");
                    ImGui::Text("Scale 100 %% performs only sharpening");
                }
                else
                {
                    ImGui::Text("Using CopyResource");
                }
            }
            else
            {
                if (m_ui.EnableNVScaler)
                {
                    ImGui::Text("Using NVScaler shader:");
                    ImGui::Text("Performs scaling and sharpening");
                }
                else
                {
                    ImGui::Text("Using bilinear upscale shader");
                }
            }
            ImGui::Separator();
            ImGui::Text("Input Size  : %d x %d", m_ui.InputWidth, m_ui.InputHeight);
            ImGui::Text("Output Size : %d x %d", m_ui.OutputWidth, m_ui.OutputHeight);
        }
        if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("VSync", &m_ui.EnableVsync);
        }
        if (ImGui::CollapsingHeader("Profiling", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::RadioButton("microseconds", &m_ui.UnitMicroseconds, 1); ImGui::SameLine();
            ImGui::RadioButton("milliseconds", &m_ui.UnitMicroseconds, 0);
            ImGui::Separator();
            double unitConst = 1E6;
            std::string unitStr = "us";
            if (!m_ui.UnitMicroseconds)
            {
                unitConst = 1E3;
                unitStr = "ms";
            }
            double filterTime = m_ui.FilterTime / 1E6 * unitConst;
            double totalTime = 1. / fps * unitConst;
            double uiTime = m_elapsedTimer.averageTime_us() / 1E6 * unitConst;
            ImGui::Text("FPS         : %9.2f", fps);
            ImGui::Text("Filter Time : %9.2f %s", filterTime, unitStr.c_str());
            ImGui::Text("UI Time     : %9.2f %s", uiTime, unitStr.c_str());
            ImGui::Text("Presnt Time : %9.2f %s", totalTime - filterTime - uiTime, unitStr.c_str());
            ImGui::Text("Total Time  : %9.2f %s", totalTime, unitStr.c_str());
        }

        ImGui::End();
    }
    ImGui::Render();
    m_elapsedTimer.end();
}

void UIRenderer::render()
{
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}