// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_GUI_H
#define PBRT_UTIL_GUI_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <pbrt/pbrt.h>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/cudagl.h>
#endif  //  PBRT_BUILD_GPU_RENDERER
#include <pbrt/util/color.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <set>
#include <string>

namespace pbrt {

enum DisplayState { EXIT, RESET, NONE };

class GLDisplay;

class GUI {
  public:
    GUI(std::string title, Vector2i resolution, Bounds3f sceneBounds);
    ~GUI();

    RGB *MapFramebuffer() {
#ifdef PBRT_BUILD_GPU_RENDERER
        if (cudaFramebuffer)
            return cudaFramebuffer->map();
        else
#endif  // PBRT_BUILD_GPU_RENDERER
            return cpuFramebuffer;
    }
    void UnmapFramebuffer() {
#ifdef PBRT_BUILD_GPU_RENDERER
        if (cudaFramebuffer)
            cudaFramebuffer->unmap();
#endif  // PBRT_BUILD_GPU_RENDERER
    }

    DisplayState RefreshDisplay();

    // It's a little messy that the state of values controlled via the UI
    // are just public variables here but it's probably not worth putting
    // an abstraction layer on top of all this at this point.
    Transform GetCameraTransform() const { return movingFromCamera; }
    Float exposure = 1.f;
    bool printCameraTransform = false;

    void keyboardCallback(GLFWwindow *window, int key, int scan, int action, int mods);

  private:
    bool processKeys();

    std::set<char> keysDown;
    Float moveScale = 1.f;
    Transform movingFromCamera;
    Vector2i resolution;
    bool recordFrames = false;
    int frameNumber = 0;

#ifdef PBRT_BUILD_GPU_RENDERER
    CUDAOutputBuffer<RGB> *cudaFramebuffer = nullptr;
#endif
    RGB *cpuFramebuffer = nullptr;
    GLDisplay *glDisplay = nullptr;
    GLFWwindow *window = nullptr;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_GUI_H
