# This file was developed by Thomas Müller <thomas94@gmx.net>.
# It is published under the BSD 3-Clause License within the LICENSE.md file.

cmake_minimum_required(VERSION 2.8.12)

project(tinylogger)
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
add_executable(log-demo log-demo.cpp tinylogger/tinylogger.h)
