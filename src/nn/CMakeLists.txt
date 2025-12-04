cmake_minimum_required(VERSION 3.18)

project(pbrt-nn LANGUAGES C CXX CUDA)

# Ensure we can find ngp
if(NOT TARGET ngp)
    message(WARNING "Target 'ngp' not found. Ensure src/ext/instant-ngp is configured.")
endif()

add_executable(pbrt_nn main_nn.cu)

# Define the macro to enable the code in main_nn.cu
target_compile_definitions(pbrt_nn PRIVATE PBRT_GPU_BUILD_RENDERER)

# Link against ngp to get all dependencies and include paths
target_link_libraries(pbrt_nn PRIVATE ngp)

# Enable separable compilation as instant-ngp uses it
set_target_properties(pbrt_nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Try to inherit CUDA architectures from ngp target
get_target_property(NGP_CUDA_ARCHITECTURES ngp CUDA_ARCHITECTURES)

if(NGP_CUDA_ARCHITECTURES)
    set_target_properties(pbrt_nn PROPERTIES CUDA_ARCHITECTURES "${NGP_CUDA_ARCHITECTURES}")
else()
    # Fallback to native if not found
    set_target_properties(pbrt_nn PROPERTIES CUDA_ARCHITECTURES "native")
endif()
