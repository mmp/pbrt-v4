//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef PBRT_GPU_CUDAGL_H
#define PBRT_GPU_CUDAGL_H

#include <pbrt/gpu/util.h>
#include <pbrt/util/error.h>

#include <glad/glad.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#define GL_CHECK(call)                                                   \
    do {                                                                 \
        call;                                                            \
        if (GLenum err = glGetError(); err != GL_NO_ERROR)               \
            LOG_FATAL("GL error: %s for " #call, getGLErrorString(err)); \
    } while (0)

#define GL_CHECK_ERRORS()                                     \
    do {                                                      \
        if (GLenum err = glGetError(); err != GL_NO_ERROR)    \
            LOG_FATAL("GL error: %s", getGLErrorString(err)); \
    } while (0)

namespace pbrt {

// Specialized from the original to only support GL_INTEROP
template <typename PIXEL_FORMAT>
class CUDAOutputBuffer {
  public:
    CUDAOutputBuffer(int32_t width, int32_t height);
    ~CUDAOutputBuffer();

    void setDevice(int32_t device_idx) { m_device_idx = device_idx; }
    void setStream(CUstream stream) { m_stream = stream; }

    // Allocate or update device pointer as necessary for CUDA access
    PIXEL_FORMAT *map();
    void unmap();

    void StartAsynchronousReadback();
    const PIXEL_FORMAT *GetReadbackPixels();

    int32_t width() const { return m_width; }
    int32_t height() const { return m_height; }

    // Get output buffer
    GLuint getPBO();
    void deletePBO();

  private:
    void makeCurrent() { CUDA_CHECK(cudaSetDevice(m_device_idx)); }

    int32_t m_width = 0u;
    int32_t m_height = 0u;

    cudaGraphicsResource *m_cuda_gfx_resource = nullptr;
    GLuint m_pbo = 0u;
    PIXEL_FORMAT *m_device_pixels = nullptr;
    PIXEL_FORMAT *m_host_pixels = nullptr;

    bool readbackActive = false;
    cudaEvent_t readbackFinishedEvent;

    CUstream m_stream = 0u;
    int32_t m_device_idx = 0;
};

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer(int32_t width, int32_t height) {
    CHECK(width > 0 && height > 0);

    // If using GL Interop, expect that the active device is also the display device.
    int current_device, is_display_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout,
                                      current_device));
    if (!is_display_device)
        LOG_FATAL("GL interop is only available on display device.");

    m_width = width;
    m_height = height;

    makeCurrent();

    // GL buffer gets resized below
    GL_CHECK(glGenBuffers(1, &m_pbo));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height,
                          nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_gfx_resource, m_pbo,
                                            cudaGraphicsMapFlagsWriteDiscard));

    CUDA_CHECK(cudaEventCreate(&readbackFinishedEvent));
    CUDA_CHECK(cudaMallocHost(&m_host_pixels, m_width * m_height * sizeof(PIXEL_FORMAT)));
}

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer() {
    makeCurrent();

    if (m_pbo != 0u) {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CHECK(glDeleteBuffers(1, &m_pbo));
    }
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::map() {
    makeCurrent();

    size_t buffer_size = 0u;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&m_device_pixels), &buffer_size, m_cuda_gfx_resource));

    return m_device_pixels;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::unmap() {
    makeCurrent();

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource, m_stream));
}

template <typename PIXEL_FORMAT>
GLuint CUDAOutputBuffer<PIXEL_FORMAT>::getPBO() {
    if (m_pbo == 0u)
        GL_CHECK(glGenBuffers(1, &m_pbo));
    return m_pbo;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::deletePBO() {
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    GL_CHECK(glDeleteBuffers(1, &m_pbo));
    m_pbo = 0;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::StartAsynchronousReadback() {
    CHECK(!readbackActive);

    makeCurrent();

    CUDA_CHECK(cudaMemcpyAsync(m_host_pixels, m_device_pixels,
                               m_width * m_height * sizeof(PIXEL_FORMAT),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(readbackFinishedEvent));
    readbackActive = true;
}

template <typename PIXEL_FORMAT>
const PIXEL_FORMAT *CUDAOutputBuffer<PIXEL_FORMAT>::GetReadbackPixels() {
    if (!readbackActive)
        return nullptr;

    makeCurrent();

    CUDA_CHECK(cudaEventSynchronize(readbackFinishedEvent));
    readbackActive = false;
    return m_host_pixels;
}

}  // end namespace pbrt

#endif  // PBRT_GPU_CUDAGL_H
