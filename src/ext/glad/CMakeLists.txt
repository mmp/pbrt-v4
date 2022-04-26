
add_library (glad STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c)

target_include_directories(glad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_property (TARGET glad PROPERTY FOLDER "ext")
