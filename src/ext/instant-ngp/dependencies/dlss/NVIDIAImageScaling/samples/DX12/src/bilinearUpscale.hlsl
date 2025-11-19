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

#ifndef BLOCK_WIDTH
#define BLOCK_WIDTH 16
#endif

#ifndef BLOCK_HEIGHT
#define BLOCK_HEIGHT 16
#endif

SamplerState  samplerLinearClamp : register(s0);
cbuffer cb : register(b0) {
    uint     kInputViewportOriginX;
    uint     kInputViewportOriginY;
    uint     kInputViewportWidth;
    uint     kInputViewportHeight;
    uint     kOutputViewportOriginX;
    uint     kOutputViewportOriginY;
    uint     kOutputViewportWidth;
    uint     kOutputViewportHeight;
    float    kScaleX;
    float    kScaleY;
    float    kDstNormX;
    float    kDstNormY;
    float    kSrcNormX;
    float    kSrcNormY;
}
Texture2D                 in_texture   : register(t0); // image srv
RWTexture2D<unorm float4> out_texture  : register(u0); // working uav


[numthreads(BLOCK_WIDTH, BLOCK_HEIGHT, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float dX = (id.x + 0.5f) * kScaleX;
    float dY = (id.y + 0.5f) * kScaleY;
    if (id.x < kOutputViewportWidth && id.y < kOutputViewportHeight && dX < kInputViewportWidth && dY < kInputViewportHeight) {
        float uvX = (dX + kInputViewportOriginX) * kSrcNormX;
        float uvY = (dY + kInputViewportOriginY) * kSrcNormY;
        uint dstX = id.x + kOutputViewportOriginX;
        uint dstY = id.y + kOutputViewportOriginY;
        out_texture[uint2(dstX, dstY)] = in_texture.SampleLevel(samplerLinearClamp, float2(uvX, uvY), 0);
    }
}