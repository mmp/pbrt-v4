project(vk_sample)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin/VK/)

find_program(Vulkan_NIS_GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
      "$ENV{VULKAN_SDK}/bin"
)
find_program(Vulkan_NIS_DXC_EXECUTABLE
    NAMES dxc
    HINTS
      "$ENV{VULKAN_SDK}/bin"
)

set(IMGUI_SRC     "${IMGUI_PATH}/backends/imgui_impl_vulkan.cpp"
                  "${IMGUI_PATH}/backends/imgui_impl_glfw.cpp"
                  "${IMGUI_PATH}/imgui.cpp"
                  "${IMGUI_PATH}/imgui_demo.cpp"
                  "${IMGUI_PATH}/imgui_draw.cpp"
                  "${IMGUI_PATH}/imgui_tables.cpp"
                  "${IMGUI_PATH}/imgui_widgets.cpp"
)

set(TINYEXR_PATH    "${CMAKE_SOURCE_DIR}/third_party/tinyexr/")
set(TINYEXR_INCLUDE "${TINYEXR_PATH}" "${TINYEXR_PATH}/deps/miniz")
set(TINYEXR_SRC     "${TINYEXR_PATH}/deps/miniz/miniz.c")

set(STB_PATH     "${CMAKE_SOURCE_DIR}/third_party/stb/")
set(STB_INCLUDE  "${STB_PATH}")

set(SAMPLE_SRC      "src/AppRenderer.cpp"
                    "src/DeviceResources.cpp"
                    "src/NVScaler.cpp"
                    "src/NVSharpen.cpp"
                    "src/Sample.cpp"
                    "${COMMON_PATH}/Image.cpp"
                    "src/UIRenderer.cpp"

)

set(SAMPLE_HEADERS  "include/AppRenderer.h"
                    "include/DeviceResources.h"
                    "include/NVScaler.h"
                    "include/NVSharpen.h"
                    "include/UIRenderer.h"
                    "include/VKUtilities.h"
                    "${COMMON_PATH}/Utilities.h"
)

add_definitions(-DNIS_VK_SAMPLE -DNIS_DXC)

source_group("external" FILES ${IMGUI_SRC} ${TINYEXR_SRC})
source_group("shaders" FILES ${SAMPLE_SHADERS})

add_executable(${PROJECT_NAME} ${SAMPLE_SRC} ${IMGUI_SRC} ${TINYEXR_SRC} ${SAMPLE_HEADERS} ${SPIRV_BLOB_SCALER}
               ${SPIRV_BLOB_SHARPEN} ${SPIRV_BLOB_SCALER_GLSL} ${SPIRV_BLOB_SHARPEN_GLSL})
target_include_directories (${PROJECT_NAME} PUBLIC include ${IMGUI_INCLUDE} ${TINYEXR_INCLUDE} ${STB_INCLUDE} ${COMMON_PATH} ${NIS_PATH})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC glfw Vulkan::Vulkan)

set(SAMPLE_SHADERS  "${NIS_PATH}/NIS_Main.hlsl")
set(DXC_ARGS_HLSL -spirv -T cs_6_2 -D NIS_DXC=1 -DNIS_USE_HALF_PRECISION=1 -D NIS_BLOCK_WIDTH=32 -D NIS_THREAD_GROUP_SIZE=256)
set(SPIRV_BLOB_SCALER "nis_scaler.spv")
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    # OUTPUT ${SPIRV_BLOB_SCALER}
    COMMAND ${Vulkan_NIS_DXC_EXECUTABLE} -D NIS_SCALER=1 -D NIS_BLOCK_HEIGHT=24 ${DXC_ARGS_HLSL} -Fo ${SPIRV_BLOB_SCALER} ${SAMPLE_SHADERS}
    DEPENDS ${SAMPLE_SHADERS}
)
set(SPIRV_BLOB_SHARPEN "nis_sharpen.spv")
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    # OUTPUT ${SPIRV_BLOB_SHARPEN}
    COMMAND ${Vulkan_NIS_DXC_EXECUTABLE} -D NIS_SCALER=0 -D NIS_BLOCK_HEIGHT=32 ${DXC_ARGS_HLSL} -Fo ${SPIRV_BLOB_SHARPEN} ${SAMPLE_SHADERS}
    DEPENDS ${SAMPLE_SHADERS}
)

set(SAMPLE_SHADERS_GLSL  "${NIS_PATH}/NIS_Main.glsl")
set(SPIRV_BLOB_SCALER_GLSL "nis_scaler_glsl.spv")
set(GLSLC_ARGS -x glsl -DNIS_BLOCK_WIDTH=32 -DNIS_THREAD_GROUP_SIZE=256 -DNIS_USE_HALF_PRECISION=1 -DNIS_GLSL=1 -fshader-stage=comp)
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    # OUTPUT ${SPIRV_BLOB_SCALER_GLSL}
    COMMAND ${Vulkan_NIS_GLSLC_EXECUTABLE} -DNIS_SCALER=1 -DNIS_BLOCK_HEIGHT=24 ${GLSLC_ARGS} -o ${SPIRV_BLOB_SCALER_GLSL} ${SAMPLE_SHADERS_GLSL}
    DEPENDS ${SAMPLE_SHADERS_GLSL}
)
set(SPIRV_BLOB_SHARPEN_GLSL "nis_sharpen_glsl.spv")
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    # OUTPUT ${SPIRV_BLOB_SHARPEN_GLSL}
    COMMAND ${Vulkan_NIS_GLSLC_EXECUTABLE} -DNIS_SCALER=0 -DNIS_BLOCK_HEIGHT=32 ${GLSLC_ARGS} -o ${SPIRV_BLOB_SHARPEN_GLSL} ${SAMPLE_SHADERS_GLSL}
    DEPENDS ${SAMPLE_SHADERS_GLSL}
)

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/NIS
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${NIS_PATH} $<TARGET_FILE_DIR:${PROJECT_NAME}>/NIS
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "nis_scaler.spv" $<TARGET_FILE_DIR:${PROJECT_NAME}>/NIS
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "nis_sharpen.spv" $<TARGET_FILE_DIR:${PROJECT_NAME}>/NIS
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "nis_scaler_glsl.spv" $<TARGET_FILE_DIR:${PROJECT_NAME}>/NIS
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "nis_sharpen_glsl.spv" $<TARGET_FILE_DIR:${PROJECT_NAME}>/NIS
)

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/media/images
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${SAMPLES_PATH}/media/images $<TARGET_FILE_DIR:${PROJECT_NAME}>/media/images
)
