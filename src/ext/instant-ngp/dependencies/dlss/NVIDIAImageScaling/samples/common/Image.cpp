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

#include "Image.h"
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <unordered_map>
#include "Utilities.h"

namespace img
{

    using fp16_t = uint16_t;

    inline fp16_t floatToHalf(float v)
    {
        tinyexr::FP32 fp32;
        fp32.f = v;
        return tinyexr::float_to_half_full(fp32).u;
    }

    inline float halfToFloat(fp16_t v)
    {
        tinyexr::FP16 fp16;
        fp16.u = v;
        tinyexr::FP32 fp32 = tinyexr::half_to_float(fp16);
        return fp32.f;
    }

    template<typename T, typename K>
    inline K convertTo(T v) { return K(v); }

    template<>
    inline float convertTo(uint8_t v) { return v / 255.f; }

    template<>
    inline fp16_t convertTo(uint8_t v) { return floatToHalf(v / 255.f); }

    template<>
    inline uint8_t convertTo(float v) { return uint8_t(v * 255.f); }

    template<>
    inline fp16_t convertTo(float v) { return floatToHalf(v); }

    template<>
    inline uint8_t convertTo(fp16_t v) { return uint8_t(halfToFloat(v) * 255.f); }

    template<>
    inline float convertTo(fp16_t v) { return halfToFloat(v); }

    template<typename T>
    inline T alphaMax() { return 255; }

    template<>
    inline float alphaMax() { return 1.f; }

    template<>
    inline fp16_t alphaMax() { return floatToHalf(1.f); }


    template<typename T, typename K>
    inline void convertToFmt(uint8_t* input, uint8_t* output, uint32_t width, uint32_t height, uint32_t inputChannels, uint32_t inputRowPitch, uint32_t outputChannels, uint32_t outputRowPitch)
    {
        constexpr size_t BOut = sizeof(K); // bytes per pixel
        constexpr size_t BIn = sizeof(T);
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                size_t output_offset = y * outputRowPitch + x * outputChannels * BOut;
                size_t input_offset = y * inputRowPitch + x * inputChannels * BIn;
                ((K*)&output[output_offset])[0] = convertTo<T, K>(((T*)&input[input_offset])[0]);
                ((K*)&output[output_offset])[1] = convertTo<T, K>(((T*)&input[input_offset])[1]);
                ((K*)&output[output_offset])[2] = convertTo<T, K>(((T*)&input[input_offset])[2]);
                if (outputChannels > 3)
                    ((K*)&output[output_offset])[3] = inputChannels > 3 ? convertTo<T, K>(((T*)&input[input_offset])[3]) : alphaMax<K>();
            }
        }
    }

    template<typename T, typename K>
    inline void convertToFmtPlanesABGR(uint8_t* input, uint8_t* output, uint32_t width, uint32_t height, uint32_t inputChannels, uint32_t inputRowPitch, uint32_t outputChannels, uint32_t outputRowPitch)
    {
        constexpr size_t BOut = sizeof(K); // bytes per pixel
        constexpr size_t BIn = sizeof(T);
        size_t planeSize = size_t(outputRowPitch) * height;
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                size_t input_offset = y * inputRowPitch + x * inputChannels * BIn;
                size_t output_offset0 = (y * outputRowPitch) + x * BOut;
                size_t output_offset1 = output_offset0 + planeSize;
                size_t output_offset2 = output_offset1 + planeSize;
                if (outputChannels == 3)
                {
                    *((K*)&output[output_offset0]) = convertTo<T, K>(((T*)&input[input_offset])[2]);
                    *((K*)&output[output_offset1]) = convertTo<T, K>(((T*)&input[input_offset])[1]);
                    *((K*)&output[output_offset2]) = convertTo<T, K>(((T*)&input[input_offset])[0]);
                }
                else
                {
                    size_t output_offset3 = output_offset2 + planeSize;
                    *((K*)&output[output_offset0]) = inputChannels > 3 ? convertTo<T, K>(((T*)&input[input_offset])[3]) : alphaMax<K>();
                    *((K*)&output[output_offset1]) = convertTo<T, K>(((T*)&input[input_offset])[2]);
                    *((K*)&output[output_offset2]) = convertTo<T, K>(((T*)&input[input_offset])[1]);
                    *((K*)&output[output_offset3]) = convertTo<T, K>(((T*)&input[input_offset])[0]);
                }
            }
        }
    }

    uint32_t bytesPerPixel(Fmt fmt)
    {
        static std::unordered_map<Fmt, uint32_t> Bpp{ {Fmt::R8G8B8A8, 4}, {Fmt::R32G32B32A32, 16}, {Fmt::R16G16B16A16, 8} };
        return Bpp[fmt];
    }

    void load(const std::string& fileName, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& outRowPitch, Fmt outFormat, uint32_t outRowPitchAlignment)
    {
        std::string extension = std::filesystem::path(fileName).extension().string();
        for (auto& e : extension) e = std::tolower(e);
        if (extension == ".exr")
        {
            loadEXR(fileName, data, width, height, outRowPitch, outFormat, outRowPitchAlignment);
        }
        else if (extension == ".png")
        {
            loadPNG(fileName, data, width, height, outRowPitch, outFormat, outRowPitchAlignment);
        }
    }

    void loadPNG(const std::string& fileName, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& outRowPitch, Fmt outFormat, uint32_t outRowPitchAlignment)
    {
        uint32_t infileChannels;
        uint8_t* image = stbi_load(fileName.c_str(), (int*)&width, (int*)&height, (int*)&infileChannels, STBI_rgb_alpha);

        if (image == nullptr)
            throw std::runtime_error("Failed to load PNG Image : " + fileName);

        uint32_t inChannels = 4; // stb_image has already converted data to RGBA
        uint32_t inputRowPitch = width * inChannels;
        uint32_t outChannels = 4; // Hardcoded all output formats have 4 channel
        outRowPitch = Align(width * bytesPerPixel(outFormat), outRowPitchAlignment);
        uint32_t imageSize = outRowPitch * height;
        data.resize(imageSize);

        switch (outFormat)
        {
        case Fmt::R8G8B8A8:
            convertToFmt<uint8_t, uint8_t>(image, data.data(), width, height, inChannels, inputRowPitch, outChannels, outRowPitch);
            break;
        case Fmt::R32G32B32A32:
            convertToFmt<uint8_t, float>(image, data.data(), width, height, inChannels, inputRowPitch, outChannels, outRowPitch);
            break;
        case Fmt::R16G16B16A16:
            convertToFmt<uint8_t, fp16_t>(image, data.data(), width, height, inChannels, inputRowPitch, outChannels, outRowPitch);
            break;
        }

        stbi_image_free(image);
    }

    void loadEXR(const std::string& fileName, std::vector<uint8_t>& data, uint32_t& width, uint32_t& height, uint32_t& outRowPitch, Fmt outFormat, uint32_t outRowPitchAlignment)
    {
        uint32_t inChannels = 4; // fixed to only 4 channel EXR files
        float* image;
        const char* err = nullptr;
        int ret = LoadEXR(&image, (int*)&width, (int*)&height, fileName.c_str(), &err);
        uint32_t inputRowPitch = width * inChannels * sizeof(float);

        if (ret != TINYEXR_SUCCESS) {
            std::string serr = err;
            FreeEXRErrorMessage(err);
            throw std::runtime_error("Failed to load EXR Image : " + fileName + " Error: " + serr);
        }

        uint32_t outChannels = 4; // Hardcoded all output formats have 4 channel
        outRowPitch = Align(width * bytesPerPixel(outFormat), outRowPitchAlignment);
        uint32_t imageSize = outRowPitch * height;
        data.resize(imageSize);

        switch (outFormat)
        {
        case Fmt::R8G8B8A8:
            convertToFmt<float, uint8_t>((uint8_t*)image, data.data(), width, height, inChannels, inputRowPitch, outChannels, outRowPitch);
            break;
        case Fmt::R32G32B32A32:
            convertToFmt<float, float>((uint8_t*)image, data.data(), width, height, inChannels, inputRowPitch, outChannels, outRowPitch);
            break;
        case Fmt::R16G16B16A16:
            convertToFmt<float, fp16_t>((uint8_t*)image, data.data(), width, height, inChannels, inputRowPitch, outChannels, outRowPitch);
            break;
        }
        free(image);
    }

    void save(const std::string& fileName, uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t rowPitch, Fmt format)
    {
        std::string extension = std::filesystem::path(fileName).extension().string();
        for (auto& e : extension) e = std::tolower(e);
        if (extension == ".exr")
        {
            saveEXR(fileName, data, width, height, channels, rowPitch, format);
        }
        else if (extension == ".png")
        {
            savePNG(fileName, data, width, height, channels, rowPitch, format);
        }
    }

    void savePNG(const std::string& fileName, uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t rowPitch, Fmt format)
    {
        constexpr uint32_t outputChannels = 4;
        std::vector<uint8_t> image(size_t(width) * height * outputChannels);
        uint32_t outputRowPitch = width * outputChannels * sizeof(uint8_t);

        switch (format)
        {
        case Fmt::R8G8B8A8:
            convertToFmt<uint8_t, uint8_t>(data, image.data(), width, height, channels, rowPitch, outputChannels, outputRowPitch);
            break;
        case Fmt::R32G32B32A32:
            convertToFmt<float, uint8_t>(data, image.data(), width, height, channels, rowPitch, outputChannels, outputRowPitch);
            break;

        case Fmt::R16G16B16A16:
            convertToFmt<fp16_t, uint8_t>(data, image.data(), width, height, channels, rowPitch, outputChannels, outputRowPitch);
            break;
        }
        stbi_write_png(fileName.c_str(), width, height, outputChannels, image.data(), outputRowPitch);
    }


    void saveEXR(const std::string& fileName, uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t rowPitch, Fmt format)
    {
        EXRHeader header;
        InitEXRHeader(&header);
        EXRImage image;
        InitEXRImage(&image);

        constexpr uint32_t outputChannels = 4;
        image.num_channels = outputChannels;
        uint32_t plane_size = width * height;
        uint32_t outputRowPitch = width * sizeof(float);
        std::vector<float> images(size_t(outputChannels) * plane_size);
        switch (format)
        {
        case Fmt::R8G8B8A8:
            convertToFmtPlanesABGR<uint8_t, float>(data, (uint8_t*)images.data(), width, height, channels, rowPitch, outputChannels, outputRowPitch);
            break;
        case Fmt::R32G32B32A32:
            convertToFmtPlanesABGR<float, float>(data, (uint8_t*)images.data(), width, height, channels, rowPitch, outputChannels, outputRowPitch);
            break;
        case Fmt::R16G16B16A16:
            convertToFmtPlanesABGR<fp16_t, float>(data, (uint8_t*)images.data(), width, height, channels, rowPitch, outputChannels, outputRowPitch);
            break;
        }

        float* image_ptr[outputChannels];
        for (size_t i = 0; i < outputChannels; ++i)
            image_ptr[i] = &images[i * plane_size];

        image.images = (unsigned char**)image_ptr;
        image.width = width;
        image.height = height;

        header.num_channels = outputChannels;
        header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
        // Must be (A)BGR order, since most of EXR viewers expect this channel order.
        header.channels[0].name[0] = 'A'; header.channels[0].name[1] = '\0';
        header.channels[1].name[0] = 'B'; header.channels[1].name[1] = '\0';
        header.channels[2].name[0] = 'G'; header.channels[2].name[1] = '\0';
        header.channels[3].name[0] = 'R'; header.channels[3].name[1] = '\0';

        header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
        header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
        for (int i = 0; i < header.num_channels; i++) {
            header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        }

        const char* err = nullptr;
        int ret = SaveEXRImageToFile(&image, &header, fileName.c_str(), &err);
        if (ret != TINYEXR_SUCCESS) {
            std::string serr = err;
            FreeEXRErrorMessage(err);
            throw std::runtime_error("Failed to save EXR Image : " + fileName + " Error: " + serr);
        }
        free(header.channels);
        free(header.pixel_types);
        free(header.requested_pixel_types);
    }
}