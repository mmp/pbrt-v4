cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(zstr LANGUAGES CXX)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.12)
  cmake_policy(SET CMP0074 NEW) # find_package uses <PackageName>_ROOT variables
endif()

if(${CMAKE_VERSION} VERSION_LESS 3.13)
  message(WARNING
    "Interface library targets are not well supported before cmake 3.13  .... "
    "You may need to add \${ZSTR_INCLUDE_DIRS} to your include directories\n"
    "target_include_directories(YourTarget PRIVATE \${ZSTR_INCLUDE_DIRS}) "
  )
endif()

# -- locate zlib

find_package(ZLIB 1.2.3 REQUIRED) # defines imported target ZLIB::ZLIB
message(STATUS "zstr - found ZLIB (version: ${ZLIB_VERSION_STRING})")

# -- add target

add_library(zstr INTERFACE)
add_library(zstr::zstr ALIAS zstr)

# -- set target properties

target_include_directories(zstr INTERFACE "${PROJECT_SOURCE_DIR}/src")
if (NOT MSVC)
    target_link_libraries(zstr INTERFACE ZLIB::ZLIB)
endif()
target_compile_features(zstr INTERFACE cxx_std_11) # require c++11 flag

# -- set cache variables

# NOTE: these vars are mostly useful to people using cmake < 3.13
set(ZSTR_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src;${ZLIB_INCLUDE_DIRS}" CACHE PATH "" FORCE)
set(ZSTR_LIBRARIES "${ZLIB_LIBRARIES}" CACHE PATH "" FORCE)

# -- print target summary

message(STATUS
  "zstr - added INTERFACE target 'zstr::zstr'
          includes : ${ZSTR_INCLUDE_DIRS}
          libraries: ZLIB::ZLIB
          features : cxx_std_11"
)
