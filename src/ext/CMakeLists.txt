
cmake_minimum_required (VERSION 3.12)

if (MSVC)
  add_definitions(/D _CRT_SECURE_NO_WARNINGS /Dstrdup=_strdup
    /wd4018 /wd4100 /wd4101 /wd4127 /wd4146 /wd4232 /wd4242 /wd4244 /wd4245 /wd4267 /wd4305 /wd4309
    /wd4310 /wd4334 /wd4456 /wd4464 /wd4668 /wd4701 /wd4703 /wd4711 /wd4756 /wd4820 /wd5045 /wd5250)
endif ()

set (FLIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/flip PARENT_SCOPE)

add_library (flip_lib STATIC ${CMAKE_CURRENT_SOURCE_DIR}/flip/flip.cpp)

set_property (TARGET flip_lib PROPERTY FOLDER "ext")
