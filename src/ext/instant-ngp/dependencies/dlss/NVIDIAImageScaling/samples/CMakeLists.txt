cmake_minimum_required(VERSION 3.12)
project(NIS_SDK)

option (NIS_VK_SAMPLE "Vulkan sample" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_SUPPRESS_REGENERATION ON)

set(NIS_PATH "${CMAKE_SOURCE_DIR}/../NIS")
set(THIRD_PARTY "${CMAKE_SOURCE_DIR}/third_party")
set(IMGUI_PATH    "${THIRD_PARTY}/imgui/")
set(IMGUI_INCLUDE "${IMGUI_PATH}" "${IMGUI_PATH}/backends/")
set(SAMPLES_PATH "${CMAKE_SOURCE_DIR}")
set(COMMON_PATH "${CMAKE_SOURCE_DIR}/common")

if (WIN32)
    add_subdirectory (DX11)
    add_subdirectory (DX12)
else(WIN32)
    set(NIS_VK_SAMPLE ON CACHE BOOL "Vulkan sample" FORCE)
endif(WIN32)

if(NIS_VK_SAMPLE)
    find_package(Vulkan REQUIRED)

    # GLFW
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${THIRD_PARTY}/glfw")

    add_subdirectory (VK)
endif()
