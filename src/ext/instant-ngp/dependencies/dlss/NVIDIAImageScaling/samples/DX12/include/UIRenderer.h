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

#include <iomanip>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>
#include "DeviceResources.h"
#include "Utilities.h"

enum class OutputSizeMode : uint32_t
{
	VARIABLE,
	P1080,
	P1440,
	P2160
};

struct UIData
{
	std::vector<std::filesystem::path> Files;
	std::filesystem::path FilePath;
	std::string FileName;
	float Scale = 75.f;
	bool EnableNVScaler = true;
	int FilterMode = 0;
	float Sharpness = 50.f;
	bool EnableVsync = false;
	OutputSizeMode OutputMode;
	uint32_t InputWidth;
	uint32_t InputHeight;
	uint32_t OutputWidth;
	uint32_t OutputHeight;
	double FilterTime;
	bool ShowSettings = true;
	int32_t UnitMicroseconds = true;
};

class UIRenderer
{
public:
	UIRenderer(HWND hwnd, DeviceResources& deviceResources, UIData& ui);
	void cleanUp();
	void update(double fps);
	void render();
private:
	UIData& m_ui;
	DeviceResources& m_deviceResources;
	ElapsedTimer m_elapsedTimer;
};