// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <glad/glad.h>

#include <pbrt/options.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif PBRT_BUILD_GPU_RENDERER
#include <pbrt/util/error.h>
#include <pbrt/wavefront/interactive.h>

#define GL_CHECK(call)                                                  \
    do {                                                                \
        call;                                                           \
        if (GLenum err = glGetError(); err != GL_NO_ERROR)              \
            LOG_FATAL("GL error: %s for " #call, getGLErrorString(err)); \
    } while (0)


#define GL_CHECK_ERRORS()                                              \
    do {                                                               \
       if (GLenum err = glGetError(); err != GL_NO_ERROR)              \
           LOG_FATAL("GL error: %s", getGLErrorString(err));           \
    } while (0)

namespace pbrt {

// GLDisplay functionality
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// BSD 3-clause license

static const char *getGLErrorString(GLenum error)
{
    switch( error ) {
        case GL_NO_ERROR:            return "No error";
        case GL_INVALID_ENUM:        return "Invalid enum";
        case GL_INVALID_VALUE:       return "Invalid value";
        case GL_INVALID_OPERATION:   return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:       return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default:                     return "Unknown GL error";
    }
}

enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

class GLDisplay
{
public:
    GLDisplay(BufferImageFormat format = BufferImageFormat::UNSIGNED_BYTE4);

    void display(
            const int32_t  screen_res_x,
            const int32_t  screen_res_y,
            const int32_t  framebuf_res_x,
            const int32_t  framebuf_res_y,
            const uint32_t pbo) const;

private:
    GLuint   m_render_tex = 0u;
    GLuint   m_program = 0u;
    GLint    m_render_tex_uniform_loc = -1;
    GLuint   m_quad_vertex_buffer = 0;

    BufferImageFormat m_image_format;

    static const std::string s_vert_source;
    static const std::string s_frag_source;
};

static GLuint createGLShader( const std::string& source, GLuint shader_type )
{
    GLuint shader = glCreateShader( shader_type );
    {
        const GLchar* source_data= reinterpret_cast<const GLchar*>( source.data() );
        glShaderSource( shader, 1, &source_data, nullptr );
        glCompileShader( shader );

        GLint is_compiled = 0;
        glGetShaderiv( shader, GL_COMPILE_STATUS, &is_compiled );
        if( is_compiled == GL_FALSE )
            {
                GLint max_length = 0;
                glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &max_length );

                std::string info_log( max_length, '\0' );
                GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
                glGetShaderInfoLog( shader, max_length, nullptr, info_log_data );

                glDeleteShader(shader);
                LOG_FATAL("Shader compilation failed: %s", info_log);

                return 0;
            }
    }

    GL_CHECK_ERRORS();

    return shader;
}

static GLuint createGLProgram(const std::string& vert_source,
                              const std::string& frag_source)
{
    GLuint vert_shader = createGLShader( vert_source, GL_VERTEX_SHADER );
    if( vert_shader == 0 )
        return 0;

    GLuint frag_shader = createGLShader( frag_source, GL_FRAGMENT_SHADER );
    if( frag_shader == 0 )
        {
            glDeleteShader( vert_shader );
            return 0;
        }

    GLuint program = glCreateProgram();
    glAttachShader( program, vert_shader );
    glAttachShader( program, frag_shader );
    glLinkProgram( program );

    GLint is_linked = 0;
    glGetProgramiv( program, GL_LINK_STATUS, &is_linked );
    if (is_linked == GL_FALSE)
        {
            GLint max_length = 0;
            glGetProgramiv( program, GL_INFO_LOG_LENGTH, &max_length );

            std::string info_log( max_length, '\0' );
            GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
            glGetProgramInfoLog( program, max_length, nullptr, info_log_data );
            LOG_FATAL("Program linking failed: %s", info_log);

            glDeleteProgram( program );
            glDeleteShader( vert_shader );
            glDeleteShader( frag_shader );

            return 0;
        }

    glDetachShader( program, vert_shader );
    glDetachShader( program, frag_shader );

    GL_CHECK_ERRORS();

    return program;
}

static GLint getGLUniformLocation( GLuint program, const std::string& name )
{
    GLint loc = glGetUniformLocation( program, name.c_str() );
    CHECK_NE(loc, -1);
    return loc;
}

static size_t pixelFormatSize( BufferImageFormat format )
{
    switch( format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            return sizeof( char ) * 4;
        case BufferImageFormat::FLOAT3:
            return sizeof( float ) * 3;
        case BufferImageFormat::FLOAT4:
            return sizeof( float ) * 4;
        default:
            LOG_FATAL( "sutil::pixelFormatSize: Unrecognized buffer format" );
    }
}

//-----------------------------------------------------------------------------
//
// GLDisplay implementation
//
//-----------------------------------------------------------------------------

const std::string GLDisplay::s_vert_source = R"(
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

const std::string GLDisplay::s_frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

GLDisplay::GLDisplay( BufferImageFormat image_format )
    : m_image_format( image_format )
{
    GLuint m_vertex_array;
    GL_CHECK( glGenVertexArrays(1, &m_vertex_array ) );
    GL_CHECK( glBindVertexArray( m_vertex_array ) );

    m_program = createGLProgram( s_vert_source, s_frag_source );
    m_render_tex_uniform_loc = getGLUniformLocation( m_program, "render_tex");

    GL_CHECK( glGenTextures( 1, &m_render_tex ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );

    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) );

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f,
    };

    GL_CHECK( glGenBuffers( 1, &m_quad_vertex_buffer ) );
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
    GL_CHECK( glBufferData( GL_ARRAY_BUFFER,
                            sizeof( g_quad_vertex_buffer_data),
                            g_quad_vertex_buffer_data,
                            GL_STATIC_DRAW
                            )
              );

    GL_CHECK_ERRORS();
}

void GLDisplay::display(
        const int32_t  screen_res_x,
        const int32_t  screen_res_y,
        const int32_t  framebuf_res_x,
        const int32_t  framebuf_res_y,
        const uint32_t pbo
        ) const
{
    GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
    GL_CHECK( glViewport( 0, 0, framebuf_res_x, framebuf_res_y ) );

    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );

    GL_CHECK( glUseProgram( m_program ) );

    // Bind our texture in Texture Unit 0
    GL_CHECK( glActiveTexture( GL_TEXTURE0 ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );
    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo ) );

    GL_CHECK( glPixelStorei(GL_UNPACK_ALIGNMENT, 4) ); // TODO!!!!!!

    size_t elmt_size = pixelFormatSize( m_image_format );
    if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool convertToSrgb = true;

    if( m_image_format == BufferImageFormat::UNSIGNED_BYTE4 )
        {
            // input is assumed to be in srgb since it is only 1 byte per channel in size
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8,   screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );
            convertToSrgb = false;
        }
    else if( m_image_format == BufferImageFormat::FLOAT3 )
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F,  screen_res_x, screen_res_y, 0, GL_RGB,  GL_FLOAT,         nullptr );

    else if( m_image_format == BufferImageFormat::FLOAT4 )
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA, GL_FLOAT,         nullptr );

    else
        LOG_FATAL( "Unknown buffer format" );

    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) );
    GL_CHECK( glUniform1i( m_render_tex_uniform_loc , 0 ) );

    // 1st attribute buffer : vertices
    GL_CHECK( glEnableVertexAttribArray( 0 ) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
    GL_CHECK( glVertexAttribPointer(
                                    0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
                                    3,                  // size
                                    GL_FLOAT,           // type
                                    GL_FALSE,           // normalized?
                                    0,                  // stride
                                    (void*)0            // array buffer offset
                                    )
              );

    if( convertToSrgb )
        GL_CHECK( glEnable( GL_FRAMEBUFFER_SRGB ) );
    else
        GL_CHECK( glDisable( GL_FRAMEBUFFER_SRGB ) );

    // Draw the triangles !
    GL_CHECK( glDrawArrays(GL_TRIANGLES, 0, 6) ); // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK( glDisableVertexAttribArray(0) );

    GL_CHECK( glDisable( GL_FRAMEBUFFER_SRGB ) );

    GL_CHECK_ERRORS();
}


static void glfwErrorCallback(int error, const char *desc) {
    LOG_ERROR("GLFW [%d]: %s", error, desc);
}

void GUI::keyboardCallback(GLFWwindow *window, int key, int scan, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    auto doKey = [&](int k, char ch) {
        if (key == k) {
            if (action == GLFW_PRESS) keysDown.insert(ch);
            else if (action == GLFW_RELEASE) {
                if (auto iter = keysDown.find(ch); iter != keysDown.end())
                    keysDown.erase(iter);
            }
        }
    };

    doKey(GLFW_KEY_A, 'a');
    doKey(GLFW_KEY_D, 'd');
    doKey(GLFW_KEY_S, 's');
    doKey(GLFW_KEY_W, 'w');
    doKey(GLFW_KEY_D, 'd');
    if (mods & GLFW_MOD_SHIFT)
        doKey(GLFW_KEY_E, 'E');
    else
        doKey(GLFW_KEY_E, 'e');
    doKey(GLFW_KEY_R, 'r');
    doKey(GLFW_KEY_EQUAL, '=');
    doKey(GLFW_KEY_MINUS, '-');
    doKey(GLFW_KEY_LEFT, 'L');
    doKey(GLFW_KEY_RIGHT, 'R');
    doKey(GLFW_KEY_UP, 'U');
    doKey(GLFW_KEY_DOWN, 'D');
    doKey(GLFW_KEY_N, 'n');
}

bool GUI::processKeys() {
    bool needsReset = false;

    auto handleNeedsReset = [&](char key, std::function<Transform(Transform)> update) {
        if (keysDown.find(key) != keysDown.end()) {
            movingFromCamera = update(movingFromCamera);
            needsReset = true;
        }
    };

    handleNeedsReset('a', [&](Transform t) { return t * Translate(Vector3f(-moveScale, 0, 0)); });
    handleNeedsReset('d', [&](Transform t) { return t * Translate(Vector3f( moveScale, 0, 0)); });
    handleNeedsReset('s', [&](Transform t) { return t * Translate(Vector3f(0, 0, -moveScale)); });
    handleNeedsReset('w', [&](Transform t) { return t * Translate(Vector3f(0, 0,  moveScale)); });
    handleNeedsReset('L', [&](Transform t) { return t * Rotate(-.5f, Vector3f(0, 1, 0)); });
    handleNeedsReset('R', [&](Transform t) { return t * Rotate( .5f, Vector3f(0, 1, 0)); });
    handleNeedsReset('U', [&](Transform t) { return t * Rotate(-.5f, Vector3f(1, 0, 0)); });
    handleNeedsReset('D', [&](Transform t) { return t * Rotate( .5f, Vector3f(1, 0, 0)); });
    handleNeedsReset('r', [&](Transform t) { return Transform(); });

    // No reset needed for these.
    if (keysDown.find('e') != keysDown.end()) {
        keysDown.erase(keysDown.find('e'));
        exposure *= 1.125f;
    }
    if (keysDown.find('E') != keysDown.end()) {
        keysDown.erase(keysDown.find('E'));
        exposure /= 1.125f;
    }
    if (keysDown.find('=') != keysDown.end()) {
        keysDown.erase(keysDown.find('='));
        moveScale *= 2;
    }
    if (keysDown.find('-') != keysDown.end()) {
        keysDown.erase(keysDown.find('-'));
        moveScale *= 0.5;
    }
    if (keysDown.find('n') != keysDown.end()) {
        keysDown.erase(keysDown.find('n'));
        denoiserEnabled = !denoiserEnabled;
    }

    return needsReset;
}

static void glfwKeyCallback(GLFWwindow *window, int key, int scan, int action,
                            int mods) {
    GUI *gui = (GUI *)glfwGetWindowUserPointer(window);
    gui->keyboardCallback(window, key, scan, action, mods);
}

GUI::GUI(std::string title, Vector2i resolution)
    : resolution(resolution) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        LOG_FATAL("Unable to initialize GLFW");
    window = glfwCreateWindow(resolution.x, resolution.y, "pbrt", NULL, NULL);
    if (!window) {
        glfwTerminate();
        LOG_FATAL("Unable to create GLFW window");
    }
    glfwSetKeyCallback(window, glfwKeyCallback);
    glfwSetWindowUserPointer(window, this);
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        LOG_FATAL("gladLoadGLLoader failed");

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        cudaFramebuffer = new CUDAOutputBuffer<RGB>(resolution.x, resolution.y);
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaFramebuffer->setDevice(device);
    }
    else
#endif // PBRT_BUILD_GPU_RENDERER
        cpuFramebuffer = new RGB[resolution.x * resolution.y];

    glDisplay = new GLDisplay(BufferImageFormat::FLOAT3);
}

GUI::~GUI() {
#ifdef PBRT_BUILD_GPU_RENDERER
    delete cudaFramebuffer;
    cudaFramebuffer = nullptr;
#endif // PBRT_BUILD_GPU_RENDERER
    delete[] cpuFramebuffer;
    delete glDisplay;

    glfwDestroyWindow(window);
    glfwTerminate();
}

DisplayState GUI::RefreshDisplay() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    GL_CHECK(glViewport(0, 0, width, height));

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU)
        glDisplay->display(resolution.x, resolution.y, width, height,
                           cudaFramebuffer->getPBO());
    else
#endif // PBRT_BUILD_GPU_RENDERER
        {
            GL_CHECK(glEnable(GL_FRAMEBUFFER_SRGB));
            GL_CHECK(glDrawPixels(resolution.x, resolution.y, GL_RGB, GL_FLOAT, cpuFramebuffer));
        }

    glfwSwapBuffers(window);
    glfwPollEvents();

    if (glfwWindowShouldClose(window))
        return DisplayState::EXIT;
    else if (processKeys())
        return DisplayState::RESET;
    else
        return DisplayState::NONE;
}

} // namespace pbrt
