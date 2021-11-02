
cmake_minimum_required (VERSION 3.12)

if (MSVC)
  add_definitions(/D _CRT_SECURE_NO_WARNINGS)
endif ()

###########################################################################
# stb

set (STB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/stb PARENT_SCOPE)

###########################################################################
# filesystem

set (FILESYSTEM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/filesystem PARENT_SCOPE)

###########################################################################
# libdeflate

add_subdirectory (libdeflate)

set (LIBDEFLATE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libdeflate PARENT_SCOPE)
set (LIBDEFLATE_LIBRARIES deflate::deflate PARENT_SCOPE)

set_property (TARGET deflate PROPERTY FOLDER "ext")

###########################################################################
# zlib

find_package (ZLIB)

if (NOT ZLIB_FOUND)
  # Build zlib
  set (ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
  set (ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
  add_subdirectory (zlib)

  set (ZLIB_LIBRARIES zlibstatic)
  set (ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)

  set (ZLIB_LIBRARY zlibstatic)
  set (ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)

  set_property (TARGET zlibstatic PROPERTY FOLDER "ext")

  add_library (ZLIB::ZLIB ALIAS zlibstatic)
  include_directories(${ZLIB_INCLUDE_DIRS})  # yuck, but so openexr/ptex can find zlib.h...
endif ()

set (ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS} PARENT_SCOPE)
set (ZLIB_LIBRARIES ${ZLIB_LIBARIES} PARENT_SCOPE)

###########################################################################
# Imath / OpenEXR

set (IMATH_IS_SUBPROJECT ON)
add_subdirectory (Imath)

set (OPENEXR_BUILD_TOOLS OFF CACHE BOOL " " FORCE)
if (WIN32)
  set (OPENEXR_FORCE_INTERNAL_ZLIB ON CACHE BOOL " " FORCE)
endif ()
set (OPENEXR_INSTALL_EXAMPLES OFF CACHE BOOL " " FORCE)

set (OPENEXR_IS_SUBPROJECT ON)
set (CMAKE_POLICY_DEFAULT_CMP0077 NEW)

get_target_property (imathinc Imath INTERFACE_INCLUDE_DIRECTORIES)
get_target_property (imathconfinc ImathConfig INTERFACE_INCLUDE_DIRECTORIES)
set (IMATH_HEADER_ONLY_INCLUDE_DIRS "${imathinc};${imathconfinc}")

add_subdirectory (openexr)

set_property (TARGET Iex IlmThread Imath OpenEXR OpenEXRCore OpenEXRUtil
              PROPERTY FOLDER "ext/OpenEXR")

get_target_property (iexinc Iex INTERFACE_INCLUDE_DIRECTORIES)
get_target_property (exrinc OpenEXR INTERFACE_INCLUDE_DIRECTORIES)
get_target_property (exrconfinc OpenEXRConfig INTERFACE_INCLUDE_DIRECTORIES)

set (OPENEXR_INCLUDE "${imathinc};${imathconfinc};${exrinc};${exrconfinc};${iexinc}" PARENT_SCOPE)
set (OPENEXR_LIBRARIES Imath::Imath OpenEXR::OpenEXR OpenEXR::Iex OpenEXR::IlmThread PARENT_SCOPE)

###########################################################################
# ptex

set (PTEX_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)

set (CMAKE_MACOSX_RPATH 1)
if (WIN32)
  add_definitions (/D PTEX_STATIC)
endif ()

add_subdirectory (ptex)

set_property (TARGET Ptex_static ptxinfo halftest ftest rtest wtest PROPERTY FOLDER "ext/ptex")

set (PTEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ptex/src/ptex PARENT_SCOPE)

###########################################################################
# double-conversion

add_subdirectory (double-conversion)

set (DOUBLE_CONVERSION_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/double-conversion PARENT_SCOPE)

set_property (TARGET double-conversion cctest PROPERTY FOLDER "ext")

###########################################################################
# nanovdb

set (NANOVDB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb PARENT_SCOPE)

###########################################################################
# FLIP

set (FLIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/flip PARENT_SCOPE)

add_library (flip_lib STATIC ${CMAKE_CURRENT_SOURCE_DIR}/flip/flip.cpp)

set_property (TARGET flip_lib PROPERTY FOLDER "ext")
