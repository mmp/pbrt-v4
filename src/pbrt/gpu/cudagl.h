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

#if defined(__HIPCC__)
#include <pbrt/util/hip_aliases.h>
#else
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#endif

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

// BufferDisplay functionality
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// BSD 3-clause license

extern const char *getGLErrorString(GLenum error);

enum BufferImageFormat { UNSIGNED_BYTE4, FLOAT4, FLOAT3 };

class BufferDisplay {
  public:
    BufferDisplay(BufferImageFormat format = BufferImageFormat::UNSIGNED_BYTE4);

    void display(const int32_t screen_res_x, const int32_t screen_res_y,
                 const int32_t framebuf_res_x, const int32_t framebuf_res_y,
                 const uint32_t pbo) const;

  private:
    GLuint m_render_tex = 0u;
    GLuint m_program = 0u;
    GLint m_render_tex_uniform_loc = -1;
    GLuint m_quad_vertex_buffer = 0;

    BufferImageFormat m_image_format;
};

static GLuint createGLShader(const std::string& source, GLuint shader_type) {
    GLuint shader = glCreateShader(shader_type);

    const GLchar* source_data = reinterpret_cast<const GLchar*>(source.data());
    glShaderSource(shader, 1, &source_data, nullptr);
    glCompileShader(shader);

    GLint is_compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
    if (is_compiled == GL_FALSE) {
        GLint max_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

        std::string info_log(max_length, '\0');
        GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
        glGetShaderInfoLog(shader, max_length, nullptr, info_log_data);

        glDeleteShader(shader);
        LOG_FATAL("Shader compilation failed: %s", info_log);
    }

    GL_CHECK_ERRORS();

    return shader;
}

static GLuint createGLProgram(const std::string& vert_source,
                              const std::string& frag_source) {
    GLuint vert_shader = createGLShader(vert_source, GL_VERTEX_SHADER);
    if (vert_shader == 0)
        return 0;

    GLuint frag_shader = createGLShader(frag_source, GL_FRAGMENT_SHADER);
    if (frag_shader == 0) {
        glDeleteShader(vert_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE) {
        GLint max_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

        std::string info_log(max_length, '\0');
        GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
        glGetProgramInfoLog(program, max_length, nullptr, info_log_data);
        LOG_FATAL("Program linking failed: %s", info_log);

        glDeleteProgram(program);
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        return 0;
    }

    glDetachShader(program, vert_shader);
    glDetachShader(program, frag_shader);

    GL_CHECK_ERRORS();

    return program;
}

static GLint getGLUniformLocation(GLuint program, const std::string& name) {
    GLint loc = glGetUniformLocation(program, name.c_str());
    CHECK_NE(loc, -1);
    return loc;
}

static size_t pixelFormatSize(BufferImageFormat format) {
    switch (format) {
    case BufferImageFormat::UNSIGNED_BYTE4:
        return sizeof(char) * 4;
    case BufferImageFormat::FLOAT3:
        return sizeof(float) * 3;
    case BufferImageFormat::FLOAT4:
        return sizeof(float) * 4;
    default:
        LOG_FATAL("sutil::pixelFormatSize: Unrecognized buffer format");
    }
}

inline BufferDisplay::BufferDisplay(BufferImageFormat image_format)
    : m_image_format(image_format) {
    GLuint m_vertex_array;
    GL_CHECK(glGenVertexArrays(1, &m_vertex_array));
    GL_CHECK(glBindVertexArray(m_vertex_array));

    std::string vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
        gl_Position =  vec4(vertexPosition_modelspace,1);
        UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
        UV.y = 1 - UV.y;
}
)";

    std::string frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

    m_program = createGLProgram(vert_source, frag_source);
    m_render_tex_uniform_loc = getGLUniformLocation(m_program, "render_tex");

    GL_CHECK(glGenTextures(1, &m_render_tex));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_render_tex));

    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f,  0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
    };

    GL_CHECK(glGenBuffers(1, &m_quad_vertex_buffer));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data),
                          g_quad_vertex_buffer_data, GL_STATIC_DRAW));

    GL_CHECK_ERRORS();
}

inline void BufferDisplay::display(const int32_t screen_res_x, const int32_t screen_res_y,
                                   const int32_t framebuf_res_x,
                                   const int32_t framebuf_res_y,
                                   const uint32_t pbo) const {
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CHECK(glViewport(0, 0, framebuf_res_x, framebuf_res_y));

    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(m_program));

    // Bind our texture in Texture Unit 0
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_render_tex));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));

    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));  // TODO!!!!!!

    size_t elmt_size = pixelFormatSize(m_image_format);
    if (elmt_size % 8 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if (elmt_size % 4 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if (elmt_size % 2 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool convertToSrgb = true;

    if (m_image_format == BufferImageFormat::UNSIGNED_BYTE4) {
        // input is assumed to be in srgb since it is only 1 byte per channel in size
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_res_x, screen_res_y, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        convertToSrgb = false;
    } else if (m_image_format == BufferImageFormat::FLOAT3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, screen_res_x, screen_res_y, 0, GL_RGB,
                     GL_FLOAT, nullptr);

    else if (m_image_format == BufferImageFormat::FLOAT4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA,
                     GL_FLOAT, nullptr);

    else
        LOG_FATAL("Unknown buffer format");

    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CHECK(glUniform1i(m_render_tex_uniform_loc, 0));

    // 1st attribute buffer : vertices
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer));
    GL_CHECK(glVertexAttribPointer(0,  // attribute 0. No particular reason for 0, but
                                       // must match the layout in the shader.
                                   3,         // size
                                   GL_FLOAT,  // type
                                   GL_FALSE,  // normalized?
                                   0,         // stride
                                   (void*)0   // array buffer offset
                                   ));

    if (convertToSrgb)
        GL_CHECK(glEnable(GL_FRAMEBUFFER_SRGB));
    else
        GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    GL_CHECK(
        glDrawArrays(GL_TRIANGLES, 0, 6));  // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK(glDisableVertexAttribArray(0));

    GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    GL_CHECK_ERRORS();
}

// Specialized from the original to only support GL_INTEROP
template <typename PIXEL_FORMAT>
class CUDAOutputBuffer {
  public:
    CUDAOutputBuffer(int32_t width, int32_t height);
    ~CUDAOutputBuffer();

    void StartAsynchronousReadback();
    const PIXEL_FORMAT *GetReadbackPixels();

    void Draw(int windowWidth, int windowHeight) {
        display->display(m_width, m_height, windowWidth, windowHeight, getPBO());
    }

    PIXEL_FORMAT *Map();
    void Unmap();

  private:
    void setStream(CUstream stream) { m_stream = stream; }
    // Allocate or update device pointer as necessary for CUDA access
    void makeCurrent() { CUDA_CHECK(cudaSetDevice(m_device_idx)); }

    // Get output buffer
    GLuint getPBO();
    void deletePBO();

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

    BufferDisplay *display = nullptr;
};

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer(int32_t width, int32_t height) {
    CHECK(width > 0 && height > 0);

    // If using GL Interop, expect that the active device is also the display device.
    int current_device, is_display_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout,
                                      current_device));
    if (getenv("XDG_SESSION_TYPE") == nullptr || getenv("XDG_SESSION_TYPE") != std::string("wayland")) {
        if (!is_display_device)
            LOG_FATAL("GL interop is only available on display device.");
    }
    CUDA_CHECK(cudaGetDevice(&m_device_idx));

    m_width = width;
    m_height = height;

    makeCurrent();

    // GL buffer gets resized below
    GL_CHECK(glGenBuffers(1, &m_pbo));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height,
                          nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

#ifdef __HIPCC__
    uint32_t num_gl_devices = 0;

    int glDevice;
    cudaGLGetDevices(&num_gl_devices, &glDevice, 1, cudaGLDeviceListAll);

    if (glDevice != current_device)
        LOG_FATAL("Multi-GPU not supported with GL interop yet");
#endif
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_gfx_resource, m_pbo,
                                            cudaGraphicsMapFlagsWriteDiscard));

    CUDA_CHECK(cudaEventCreate(&readbackFinishedEvent));
    CUDA_CHECK(cudaMallocHost(&m_host_pixels, m_width * m_height * sizeof(PIXEL_FORMAT)));

    display = new BufferDisplay(BufferImageFormat::FLOAT3);
}

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer() {
    makeCurrent();

    delete display;

    if (m_pbo != 0u) {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CHECK(glDeleteBuffers(1, &m_pbo));
    }
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::Map() {
    makeCurrent();

    size_t buffer_size = 0u;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&m_device_pixels), &buffer_size, m_cuda_gfx_resource));

    return m_device_pixels;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::Unmap() {
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

#undef GL_CHECK
#undef GL_CHECK_ERRORS

#endif  // PBRT_GPU_CUDAGL_H
