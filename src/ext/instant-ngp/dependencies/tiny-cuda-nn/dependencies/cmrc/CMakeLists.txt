cmake_minimum_required(VERSION 3.6)
project(CMakeRC VERSION 2.0.0)

list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")

# Include the main module
include(CMakeRC.cmake)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS AND (PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    enable_testing()
    if(NOT MSVC)
        add_compile_options(-Wall -Wextra)
    endif()
    add_subdirectory(tests)
endif()
