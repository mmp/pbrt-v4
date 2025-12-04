# Copyright (c) 2016-2017 Taylor C. Richberger <taywee@gmx.com> and Pavel
# Belikov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.2)

include(GNUInstallDirs)
string(REPLACE "/${CMAKE_LIBRARY_ARCHITECTURE}" "" CMAKE_INSTALL_LIBDIR_ARCHIND "${CMAKE_INSTALL_LIBDIR}")


set(ARGS_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(ARGS_MAIN_PROJECT ON)
endif()
project(args CXX)
set(PROJECT_DESCRIPTION "A simple, small, flexible, single-header C++11 argument parsing library. that is designed to appear somewhat similar to Python's argparse.")
set(PROJECT_HOMEPAGE_URL "https://github.com/Taywee/args")
set(PROJECT_VERSION 6.3.0)

option(ARGS_BUILD_EXAMPLE "Build example" ON)
option(ARGS_BUILD_UNITTESTS "Build unittests" ON)

set(CMAKE_CONFIG_DEST "${CMAKE_INSTALL_LIBDIR_ARCHIND}/cmake/${PROJECT_NAME}")
set(PackagingTemplatesDir "${CMAKE_CURRENT_SOURCE_DIR}/packaging")

add_library(args INTERFACE)
target_include_directories(args INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
	$<INSTALL_INTERFACE:include>)

if(ARGS_MAIN_PROJECT)
    install(FILES args.hxx DESTINATION include)
    install(TARGETS args EXPORT args-targets)
    install(EXPORT args-targets
        FILE args-config.cmake
        NAMESPACE taywee::
        DESTINATION lib/cmake/args)
endif()

if (ARGS_BUILD_EXAMPLE)
    add_executable(gitlike examples/gitlike.cxx)
    target_link_libraries(gitlike args)
    set_property(TARGET gitlike PROPERTY CXX_STANDARD 11)

    add_executable(completion examples/completion.cxx)
    target_link_libraries(completion args)
    set_property(TARGET completion PROPERTY CXX_STANDARD 11)
endif()

if (ARGS_BUILD_UNITTESTS)
    add_executable(argstest test.cxx)
    target_link_libraries(argstest args)
    set_property(TARGET argstest PROPERTY CXX_STANDARD 11)

    if (MSVC)
        target_compile_options(argstest PRIVATE /W4 /WX /bigobj)
    else ()
        target_compile_options(argstest PRIVATE -Wall -Wextra -Werror -pedantic -Wshadow -Wunused-parameter)
    endif ()

    add_executable(argstest-multiple-inclusion test/multiple_inclusion_1.cxx test/multiple_inclusion_2.cxx)

    target_link_libraries(argstest-multiple-inclusion args)
    set_property(TARGET argstest-multiple-inclusion PROPERTY CXX_STANDARD 11)

    add_executable(argstest-windows-h test/windows_h.cxx)
    target_link_libraries(argstest-windows-h args)
    set_property(TARGET argstest-windows-h PROPERTY CXX_STANDARD 11)

    enable_testing()
    add_test(NAME "test" COMMAND argstest)
    add_test(NAME "test-multiple-inclusion" COMMAND argstest-multiple-inclusion)
endif()

set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "${PROJECT_NAME} developers")
set(CPACK_PACKAGE_DESCRIPTION "${PROJECT_DESCRIPTION}")
set(CPACK_DEBIAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
set(CPACK_RPM_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
set(CPACK_PACKAGE_HOMEPAGE_URL "${PROJECT_HOMEPAGE_URL}")
set(CPACK_PACKAGE_MAINTAINER "${CPACK_PACKAGE_VENDOR}")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_MAINTAINER}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_DEBIAN_PACKAGE_NAME "lib${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6-dev")
set(CPACK_DEBIAN_PACKAGE_SUGGESTS "cmake, pkg-config, pkg-conf")

set(CPACK_RPM_PACKAGE_NAME "lib${PROJECT_NAME}-devel")
set(CPACK_RPM_PACKAGE_SUGGESTS "${CPACK_DEBIAN_PACKAGE_SUGGESTS}")

set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_NSIS_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_COMPRESSION_TYPE "xz")

set(PKG_CONFIG_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc")
configure_file("${PackagingTemplatesDir}/pkgconfig.pc.in" "${PKG_CONFIG_FILE_NAME}" @ONLY)
install(FILES "${PKG_CONFIG_FILE_NAME}"
	DESTINATION "${CMAKE_INSTALL_LIBDIR_ARCHIND}/pkgconfig"
)

include(CPack)
