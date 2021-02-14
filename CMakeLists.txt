# pbrt-v4 top-level CMakeLists.txt

cmake_minimum_required (VERSION 3.12)

project (PBRT-V4 LANGUAGES CXX C)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# For sanitizers
list (INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

# Configuration options

option (PBRT_FLOAT_AS_DOUBLE "Use 64-bit floats" OFF)
option (PBRT_BUILD_NATIVE_EXECUTABLE "Build executable optimized for CPU architecture of system pbrt was built on" ON)
option (PBRT_NVTX "Insert NVTX annotations for NVIDIA Profiling and Debugging Tools" OFF)
option (PBRT_USE_PREGENERATED_RGB_TO_SPECTRUM_TABLES "Use pregenerated rgbspectrum_*.cpp files rather than running rgb2spec_opt to generate them at build time" OFF)
set (PBRT_OPTIX7_PATH "" CACHE PATH "Path to OptiX 7 SDK")
set (PBRT_GPU_SHADER_MODEL "" CACHE STRING "")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to 'Release' as none was specified.")
  set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif ()

function (CHECK_EXT NAME DIR HASH)
  if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/${DIR}")
    message (FATAL_ERROR "The ${NAME} submodule directory is missing! "
      "You probably did not clone the project with --recursive. It is possible to recover by running:\n"
      "  \"git submodule update --init --recursive\"")
  endif ()

  find_package(Git)
  if (GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/${DIR}"
      RESULT_VARIABLE "git_return"
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE "git_hash")
    if (NOT ${git_hash} MATCHES "^${HASH}")
      message (FATAL_ERROR "The ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/${DIR} "
        "submodule isn't up to date (${git_hash} vs ${HASH}). Please run:\n"
        "  \"git submodule update --recursive\"")
    else()
      message(STATUS "${NAME} at commit: ${git_hash}")
    endif()
  else(GIT_FOUND)
    message(STATUS "git not found: unable to verify revisions in submodules")
  endif(GIT_FOUND)
endfunction()

check_ext ("OpenEXR" "openexr/OpenEXR" 5cfb5dab6dfada731586b0281bdb15ee75e26782)
check_ext ("Ptex" "ptex/src" 4cd8e9a6db2b06e478dfbbd8c26eb6df97f84483)
check_ext ("double-conversion" "double-conversion/cmake" cc1f75a114aca8d2af69f73a5a959aecbab0e87a)
check_ext ("filesystem" "filesystem/filesystem" f45da753728cde9b1c380b343e41c8b1ca6498d7)
check_ext ("stb" "stb/tools" b42009b3b9d4ca35bc703f5310eedc74f584be58)
check_ext ("OpenVDB" "openvdb/nanovdb" 1c306bcaf7e9f7d52525bab716202bd4f7fff5d1)
check_ext ("zlib" "zlib/doc" 54d591eabf9fe0e84c725638f8d5d8d202a093fa)

add_compile_definitions ("$<$<CONFIG:DEBUG>:PBRT_DEBUG_BUILD>")

enable_testing ()

find_package ( Sanitizers )
find_package ( Threads )

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (MSVC)
  list (APPEND PBRT_DEFINITIONS "PBRT_IS_MSVC" "_CRT_SECURE_NO_WARNINGS")
endif ()

if (PBRT_FLOAT_AS_DOUBLE)
  list (APPEND PBRT_DEFINITIONS "PBRT_FLOAT_AS_DOUBLE")
endif ()

#######################################
## ext

set (BUILD_SHARED_LIBS OFF)

add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/ext)

########################################
# os/compiler-specific stuff

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
  list (APPEND PBRT_DEFINITIONS "PBRT_IS_WINDOWS" "NOMINMAX")
elseif (CMAKE_SYSTEM_NAME STREQUAL Darwin)
  list (APPEND PBRT_DEFINITIONS "PBRT_IS_OSX")
elseif (CMAKE_SYSTEM_NAME STREQUAL Linux)
  list (APPEND PBRT_DEFINITIONS "PBRT_IS_LINUX")
  # -rdynamic so we can get backtrace symbols...
  # --no-as-needed so libprofiler sticks around
  string (APPEND CMAKE_EXE_LINKER_FLAGS " -rdynamic -Wl,--no-as-needed")
else ()
  message (SEND_ERROR "Unknown system name: " + CMAKE_SYSTEM_NAME)
endif()

# libgoogle-perftools-dev
find_library (PROFILE_LIB profiler)
if (NOT PROFILE_LIB)
  message (STATUS "Unable to find -lprofiler")
else ()
  message (STATUS "Found -lprofiler: ${PROFILE_LIB}")
endif ()

add_library (pbrt_warnings INTERFACE)
target_compile_options (
    pbrt_warnings
    INTERFACE
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4244>" # int -> float conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4267>" # size_t -> int conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4305>" # double constant assigned to float
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4552>" # result of expression not used
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4838>" # double -> int conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4843>" # double -> float conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd26451>" # arithmetic on 4-byte value, then cast to 8-byte
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd26495>" # uninitialized member variable
)

add_library (pbrt_opt INTERFACE)

#########################################
## CUDA / OptiX

add_library (cuda_build_configuration INTERFACE)

include (CheckLanguage)

check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    find_package (CUDA REQUIRED)

    # This seems to be necessary starting with 3.17.1, but gives an error
    # about 17 being an unsupported version earlier...
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
        set (CMAKE_CUDA_STANDARD 17)
    endif ()

    message (STATUS "Found CUDA: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
    if (CUDA_VERSION_MAJOR LESS 11)
      message(SEND_ERROR "pbrt-v4 requires CUDA version 11.0 or later. If you have multiple versions installed, please update your PATH.")
    endif ()

    if ("${PBRT_OPTIX7_PATH}" STREQUAL "")
        message (WARNING "Found CUDA but PBRT_OPTIX7_PATH is not set. Disabling GPU compilation.")
    elseif (CMAKE_SYSTEM_NAME STREQUAL Windows)
        message (FATAL_ERROR "pbrt on the GPU is not currently supported on Windows--sorry! It has worked in the past, but is now buggy. It will be necessary for someone to debug the current issues and send a PR for it to be re-enabled (we don't currently have the bandwidth to do so).")
    else ()
        enable_language (CUDA)
        list (APPEND PBRT_DEFINITIONS "PBRT_BUILD_GPU_RENDERER")
        if (PBRT_NVTX)
            list (APPEND PBRT_DEFINITIONS "NVTX")
        endif ()
        set (PBRT_CUDA_ENABLED ON)

        # FIXME
        include_directories (${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  # for regular c++ compiles

        # http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
        target_compile_options (
            pbrt_warnings
            INTERFACE
                #"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas --warn-on-double-precision-use>"
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=partial_override>"
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=virtual_function_decl_hidden>"
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=integer_sign_change>"
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=declared_but_not_referenced>"
                # WAR invalid warnings about this with "if constexpr"
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=implicit_return_from_non_void_function>"
        )

        # Willie hears yeh..
        string (APPEND CMAKE_CUDA_FLAGS " -Xnvlink -suppress-stack-size-warning")

        target_compile_options (
            cuda_build_configuration
            INTERFACE
                "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler>"
                # The "$<NOT:$<BOOL:$<TARGET_PROPERTY:CUDA_PTX_COMPILATION>>>" part is to not add debugging symbols when generating PTX files for OptiX; see https://github.com/mmp/pbrt-v4/issues/69#issuecomment-715499748.
                "$<$<COMPILE_LANGUAGE:CUDA>:$<IF:$<AND:$<CONFIG:Debug>,$<NOT:$<BOOL:$<TARGET_PROPERTY:CUDA_PTX_COMPILATION>>>>,-G;-g,-lineinfo;-maxrregcount;128>>"
        )

        if (PBRT_GPU_SHADER_MODEL STREQUAL "")
            # https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
            # Get CUDA compute capability
            set (CHECK_CUDA_OUTPUT_EXE ${CMAKE_BINARY_DIR}/checkcuda)
            if (MSVC)
                execute_process (COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CMAKE_SOURCE_DIR}/cmake/checkcuda.cu -ccbin ${CMAKE_CXX_COMPILER} -o ${CHECK_CUDA_OUTPUT_EXE}
                                 RESULT_VARIABLE BUILD_CHECK_CUDA_RETURN_CODE)
            else  ()
                execute_process (COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CMAKE_SOURCE_DIR}/cmake/checkcuda.cu -o ${CHECK_CUDA_OUTPUT_EXE}
                                 RESULT_VARIABLE BUILD_CHECK_CUDA_RETURN_CODE)
            endif ()

            if (NOT ${BUILD_CHECK_CUDA_RETURN_CODE} EQUAL 0)
                message (SEND_ERROR "Was unable to build checkcuda, consider manually setting PBRT_GPU_SHADER_MODEL")
            endif ()

            execute_process (COMMAND ${CHECK_CUDA_OUTPUT_EXE}
                             RESULT_VARIABLE CUDA_RETURN_CODE
                             OUTPUT_VARIABLE CHECK_CUDA_OUTPUT)

            if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
                message (SEND_ERROR ${CHECK_CUDA_OUTPUT})
              else ()
                set(ARCH "${CHECK_CUDA_OUTPUT}")
                message (STATUS "Detected CUDA Architecture: ${ARCH}")
                string (APPEND CMAKE_CUDA_FLAGS " --gpu-architecture=${ARCH}")
            endif ()
        else ()
            set(ARCH "${PBRT_GPU_SHADER_MODEL}")
            message (STATUS "Specified CUDA Architecture: ${ARCH}")
            string (APPEND CMAKE_CUDA_FLAGS " --gpu-architecture=${ARCH}")
        endif ()

        set (PBRT_CUDA_LIB cuda)
        # optix
        # FIXME
        include_directories (${PBRT_OPTIX7_PATH}/include)

        # from Ingo's configure_optix.cmake (Apache licensed)
        find_program (BIN2C bin2c DOC "Path to the CUDA SDK bin2c executable.")

        # this macro defines cmake rules that execute the following four steps:
        # 1) compile the given cuda file ${cuda_file} to an intermediary PTX file
        # 2) use the 'bin2c' tool (that comes with CUDA) to
        #    create a second intermediary (.c-)file which defines a const string variable
        #    (named '${c_var_name}') whose (constant) value is the PTX output
        #    from the previous step.
        # 3) compile the given .c file to an intermediary object file (why thus has
        #    that PTX string 'embedded' as a global constant.
        # 4) assign the name of the intermediary .o file to the cmake variable
        #    'output_var', which can then be added to cmake targets.
        macro (cuda_compile_and_embed output_var cuda_file lib_name)
          add_library ("${lib_name}" OBJECT "${cuda_file}")
          set_property (TARGET "${lib_name}" PROPERTY CUDA_PTX_COMPILATION ON)
          target_compile_options ("${lib_name}"
            PRIVATE
                # disable "extern declaration... is treated as a static definition" warning
                -Xcudafe=--display_error_number -Xcudafe=--diag_suppress=3089
                )
          # CUDA integration in Visual Studio seems broken as even if "Use
          # Host Preprocessor Definitions" is checked, the host preprocessor
          # definitions are still not used when compiling device code.
          # To work around that, define the macros using --define-macro to
          # avoid CMake identifying those as macros and using the proper (but
          # broken) way of specifying them.
          if (${CMAKE_GENERATOR} MATCHES "^Visual Studio")
            # As PBRT_DEBUG_BUILD is specified globally as a definition, we need to
            # manually add it due to the bug mentioned earlier and due to it
            # not being found in PBRT_DEFINITIONS.
            if (CMAKE_BUILD_TYPE MATCHES Debug)
                set (cuda_definitions "--define-macro=PBRT_DEBUG_BUILD")
            endif ()
            foreach (arg ${PBRT_DEFINITIONS})
              list (APPEND cuda_definitions "--define-macro=${arg}")
            endforeach ()
            target_compile_options("${lib_name}" PRIVATE ${cuda_definitions})
          else ()
            target_compile_definitions("${lib_name}" PRIVATE ${PBRT_DEFINITIONS})
          endif()
          target_include_directories ("${lib_name}" PRIVATE src ${CMAKE_BINARY_DIR})
          target_include_directories ("${lib_name}" SYSTEM PRIVATE ${NANOVDB_INCLUDE})
          target_link_libraries ("${lib_name}" PRIVATE cuda_build_configuration pbrt_opt pbrt_warnings)
          add_dependencies ("${lib_name}" pbrt_soa_generated)
          set (c_var_name ${output_var})
          set (embedded_file ${cuda_file}.ptx_embedded.c)
          add_custom_command (
            OUTPUT "${embedded_file}"
            COMMAND ${CMAKE_COMMAND}
              "-DBIN_TO_C_COMMAND=${BIN2C}"
              "-DOBJECTS=$<TARGET_OBJECTS:${lib_name}>"
              "-DVAR_NAME=${c_var_name}"
              "-DOUTPUT=${embedded_file}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
            VERBATIM
            DEPENDS "${lib_name}" $<TARGET_OBJECTS:${lib_name}>
            COMMENT "Embedding PTX generated from ${cuda_file}"
          )
          set (${output_var} ${embedded_file})
        endmacro ()
    endif ()
else ()
    message (STATUS "CUDA not found")
endif ()

###########################################################################
# Annoying compiler-specific details

INCLUDE(CheckCXXCompilerFlag)

check_cxx_compiler_flag ("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE AND PBRT_BUILD_NATIVE_EXECUTABLE)
    target_compile_options (pbrt_opt INTERFACE
          "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >-march=native")
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  list (APPEND PBRT_CXX_FLAGS "-std=c++17")

  FIND_PROGRAM(XIAR xiar)
  IF(XIAR)
    SET(CMAKE_AR "${XIAR}")
  ENDIF(XIAR)
  MARK_AS_ADVANCED(XIAR)

  FIND_PROGRAM(XILD xild)
  IF(XILD)
    SET(CMAKE_LINKER "${XILD}")
  ENDIF(XILD)
  MARK_AS_ADVANCED(XILD)

  # ICC will default to -fp-model fast=1, which performs value-unsafe optimizations which will
  # cause pbrt_test to fail. For safety, -fp-model precise is explicitly set here by default.
  set(FP_MODEL "precise" CACHE STRING "The floating point model to compile with.")
  set_property(CACHE FP_MODEL PROPERTY STRINGS "precise" "fast=1" "fast=2")

  list (APPEND PBRT_CXX_FLAGS "-fp-model" "${FP_MODEL}")
endif ()

if (MSVC AND MSVC_VERSION LESS 1920)
  message (SEND_ERROR "pbrt-v4 currently requires MSVC 2019 to build on Windows. PRs that get MSVC 2017 working as well would be welcomed. :-)")
endif ()

###########################################################################
# Check for various C++ features and set preprocessor variables or
# define workarounds.

include (CheckCXXSourceCompiles)

check_cxx_source_compiles ("
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
int main() {
   int fd = open(\"foo\", O_RDONLY);
   struct stat s;
   fstat(fd, &s);
   size_t len = s.st_size;
   void *ptr = mmap(0, len, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
   munmap(ptr, len);   
}
" HAVE_MMAP)

if (HAVE_MMAP)
  list (APPEND PBRT_DEFINITIONS "PBRT_HAVE_MMAP")
ENDIF ()

include (CheckIncludeFiles)

check_cxx_source_compiles ("
#include <intrin.h>
int main() {
    unsigned long lz = 0, v = 1234;
    if (_BitScanReverse(&lz, v)) return lz;
    return 0;
} " HAS_INTRIN_H)

if (HAS_INTRIN_H)
  list (APPEND PBRT_DEFINITIONS "PBRT_HAS_INTRIN_H")
endif ()

########################################
# noinline

check_cxx_source_compiles (
"__declspec(noinline) void foo() { }
int main() { }"
HAVE_DECLSPEC_NOINLINE)

check_cxx_source_compiles (
"__attribute__((noinline)) void foo() { }
int main() { }"
HAVE_ATTRIBUTE_NOINLINE)

if (HAVE_ATTRIBUTE_NOINLINE)
  list (APPEND PBRT_DEFINITIONS "PBRT_NOINLINE=__attribute__((noinline))")
elseif (HAVE_DECLSPEC_NOINLINE)
  list (APPEND PBRT_DEFINITIONS "PBRT_NOINLINE=__declspec(noinline)")
else ()
  list (APPEND PBRT_DEFINITIONS "PBRT_NOINLINE")
endif ()

########################################
# Aligned memory allocation

check_cxx_source_compiles ( "
#include <malloc.h>
int main() { void * ptr = _aligned_malloc(1024, 32); }
" HAVE__ALIGNED_MALLOC )

check_cxx_source_compiles ( "
#include <stdlib.h>
int main() {
  void *ptr;
  posix_memalign(&ptr, 32, 1024);
} " HAVE_POSIX_MEMALIGN )

if (HAVE__ALIGNED_MALLOC)
  list (APPEND PBRT_DEFINITIONS "PBRT_HAVE__ALIGNED_MALLOC")
elseif (HAVE_POSIX_MEMALIGN)
  list (APPEND PBRT_DEFINITIONS "PBRT_HAVE_POSIX_MEMALIGN")
else ()
  message (SEND_ERROR "Unable to find a way to allocate aligned memory")
endif ()

########################################
# are long and int64_t the same

check_cxx_source_compiles ("
#include <cstdint>
#include <type_traits>
static_assert(!std::is_same<long, int64_t>::value && !std::is_same<long long, int64_t>::value);
int main() { }
" INT64_IS_OWN_TYPE)

if (INT64_IS_OWN_TYPE)
  list (APPEND PBRT_DEFINITIONS "PBRT_INT64_IS_OWN_TYPE")

endif ()
if (PBRT_NVTX)
  add_definitions( -D NVTX )
endif()

###########################################################################
# On to pbrt...

set (PBRT_SOURCE
  src/pbrt/bsdf.cpp
  src/pbrt/bssrdf.cpp
  src/pbrt/bxdfs.cpp
  src/pbrt/cameras.cpp
  src/pbrt/film.cpp
  src/pbrt/filters.cpp
  src/pbrt/interaction.cpp
  src/pbrt/lights.cpp
  src/pbrt/lightsamplers.cpp
  src/pbrt/materials.cpp
  src/pbrt/media.cpp
  src/pbrt/options.cpp
  src/pbrt/paramdict.cpp
  src/pbrt/parsedscene.cpp
  src/pbrt/parser.cpp
  src/pbrt/pbrt.cpp
  src/pbrt/ray.cpp
  src/pbrt/samplers.cpp
  src/pbrt/shapes.cpp
  src/pbrt/textures.cpp

  src/pbrt/cpu/aggregates.cpp
  src/pbrt/cpu/integrators.cpp
  src/pbrt/cpu/primitive.cpp
  src/pbrt/cpu/render.cpp

  src/pbrt/cmd/pspec_gpu.cpp
  )

set (PBRT_SOURCE_HEADERS
  src/pbrt/bsdf.h
  src/pbrt/bssrdf.h
  src/pbrt/bxdfs.h
  src/pbrt/cameras.h
  src/pbrt/film.h
  src/pbrt/filters.h
  src/pbrt/interaction.h
  src/pbrt/lightsamplers.h
  src/pbrt/lights.h
  src/pbrt/materials.h
  src/pbrt/media.h
  src/pbrt/options.h
  src/pbrt/paramdict.h
  src/pbrt/parsedscene.h
  src/pbrt/parser.h
  src/pbrt/pbrt.h
  src/pbrt/pbrt.soa
  src/pbrt/ray.h
  src/pbrt/samplers.h
  src/pbrt/shapes.h
  src/pbrt/textures.h
  )  

SET (PBRT_UTIL_SOURCE
  src/pbrt/util/bluenoise.cpp
  src/pbrt/util/buffercache.cpp
  src/pbrt/util/check.cpp
  src/pbrt/util/color.cpp
  src/pbrt/util/colorspace.cpp
  src/pbrt/util/display.cpp
  src/pbrt/util/error.cpp
  src/pbrt/util/file.cpp
  src/pbrt/util/float.cpp
  src/pbrt/util/image.cpp
  src/pbrt/util/log.cpp
  src/pbrt/util/loopsubdiv.cpp
  src/pbrt/util/lowdiscrepancy.cpp
  src/pbrt/util/math.cpp
  src/pbrt/util/memory.cpp
  src/pbrt/util/mesh.cpp
  src/pbrt/util/mipmap.cpp
  src/pbrt/util/noise.cpp
  src/pbrt/util/parallel.cpp
  src/pbrt/util/pmj02tables.cpp
  src/pbrt/util/primes.cpp
  src/pbrt/util/print.cpp
  src/pbrt/util/progressreporter.cpp
  src/pbrt/util/pstd.cpp
  src/pbrt/util/rng.cpp
  src/pbrt/util/sampling.cpp
  src/pbrt/util/scattering.cpp
  src/pbrt/util/sobolmatrices.cpp
  src/pbrt/util/spectrum.cpp
  src/pbrt/util/stats.cpp
  src/pbrt/util/stbimage.cpp
  src/pbrt/util/string.cpp
  src/pbrt/util/transform.cpp
  src/pbrt/util/vecmath.cpp
)

SET (PBRT_UTIL_SOURCE_HEADERS
  src/pbrt/util/args.h
  src/pbrt/util/bits.h
  src/pbrt/util/bluenoise.h
  src/pbrt/util/buffercache.h
  src/pbrt/util/check.h
  src/pbrt/util/color.h
  src/pbrt/util/colorspace.h
  src/pbrt/util/containers.h
  src/pbrt/util/display.h
  src/pbrt/util/error.h
  src/pbrt/util/file.h
  src/pbrt/util/float.h
  src/pbrt/util/hash.h
  src/pbrt/util/image.h
  src/pbrt/util/log.h
  src/pbrt/util/loopsubdiv.h
  src/pbrt/util/lowdiscrepancy.h
  src/pbrt/util/math.h
  src/pbrt/util/memory.h
  src/pbrt/util/mesh.h
  src/pbrt/util/mipmap.h
  src/pbrt/util/noise.h
  src/pbrt/util/parallel.h
  src/pbrt/util/pmj02tables.h
  src/pbrt/util/primes.h
  src/pbrt/util/print.h
  src/pbrt/util/progressreporter.h
  src/pbrt/util/pstd.h
  src/pbrt/util/rng.h
  src/pbrt/util/sampling.h
  src/pbrt/util/scattering.h
  src/pbrt/util/soa.h
  src/pbrt/util/sobolmatrices.h
  src/pbrt/util/spectrum.h
  src/pbrt/util/splines.h
  src/pbrt/util/stats.h
  src/pbrt/util/string.h
  src/pbrt/util/taggedptr.h
  src/pbrt/util/transform.h
  src/pbrt/util/vecmath.h
  )

if (PBRT_CUDA_ENABLED)
  set (PBRT_GPU_SOURCE
     src/pbrt/gpu/accel.cpp
     src/pbrt/gpu/camera.cpp
     src/pbrt/gpu/film.cpp
     src/pbrt/gpu/init.cpp
     src/pbrt/gpu/launch.cpp
     src/pbrt/gpu/media.cpp
     src/pbrt/gpu/pathintegrator.cpp
     src/pbrt/gpu/samples.cpp
     src/pbrt/gpu/subsurface.cpp
     src/pbrt/gpu/surfscatter.cpp
  )
  set (PBRT_GPU_SOURCE_HEADERS
     src/pbrt/gpu/accel.h
     src/pbrt/gpu/init.h
     src/pbrt/gpu/launch.h
     src/pbrt/gpu/optix.h
     src/pbrt/gpu/pathintegrator.h
     src/pbrt/gpu/workitems.h
     src/pbrt/gpu/workitems.soa
     src/pbrt/gpu/workqueue.h
  )

  set_source_files_properties (
   src/pbrt/bsdf.cpp
   src/pbrt/bssrdf.cpp
   src/pbrt/bxdfs.cpp
   src/pbrt/cameras.cpp
   src/pbrt/film.cpp
   src/pbrt/filters.cpp
#   src/pbrt/genscene.cpp
   src/pbrt/interaction.cpp
   src/pbrt/lights.cpp
   src/pbrt/lightsamplers.cpp
   src/pbrt/materials.cpp
#   src/pbrt/media.cpp
   src/pbrt/options.cpp
#   src/pbrt/paramdict.cpp
#   src/pbrt/parser.cpp
   src/pbrt/pbrt.cpp
   src/pbrt/samplers.cpp
   src/pbrt/shapes.cpp
   src/pbrt/textures.cpp

   src/pbrt/util/bluenoise.cpp
   src/pbrt/util/check.cpp
   src/pbrt/util/color.cpp
   src/pbrt/util/colorspace.cpp
   src/pbrt/util/error.cpp
#   src/pbrt/util/file.cpp
#   src/pbrt/util/float.cpp
#   src/pbrt/util/image.cpp
   src/pbrt/util/log.cpp
#   src/pbrt/util/loopsubdiv.cpp
   src/pbrt/util/lowdiscrepancy.cpp
   src/pbrt/util/math.cpp
#   src/pbrt/util/memory.cpp
   src/pbrt/util/mesh.cpp
#   src/pbrt/util/mipmap.cpp
   src/pbrt/util/noise.cpp
#   src/pbrt/util/parallel.cpp
   src/pbrt/util/pmj02tables.cpp
   src/pbrt/util/primes.cpp
#   src/pbrt/util/print.cpp
#   src/pbrt/util/progressreporter.cpp
   src/pbrt/util/pstd.cpp
   src/pbrt/util/rng.cpp
   src/pbrt/util/sampling.cpp
   src/pbrt/util/scattering.cpp
   src/pbrt/util/sobolmatrices.cpp
   src/pbrt/util/spectrum.cpp
   src/pbrt/util/stats.cpp
#   src/pbrt/util/stbimage.cpp
#   src/pbrt/util/string.cpp
   src/pbrt/util/transform.cpp
   src/pbrt/util/vecmath.cpp

    ${PBRT_GPU_SOURCE}

    src/pbrt/cmd/pspec_gpu.cpp

    PROPERTIES LANGUAGE CUDA
  )

  cuda_compile_and_embed (PBRT_EMBEDDED_PTX src/pbrt/gpu/optix.cu optix.cu)
endif ()

source_group("Source Files" FILES ${PBRT_SOURCE})
source_group("Header Files" FILES ${PBRT_SOURCE_HEADERS})
source_group("Source Files/util" FILES ${PBRT_UTIL_SOURCE})
source_group("Header Files/util" FILES ${PBRT_UTIL_SOURCE_HEADERS})
if (PBRT_CUDA_ENABLED)
  source_group("Source Files/gpu" FILES ${PBRT_GPU_SOURCE})
  source_group("Header Files/gpu" FILES ${PBRT_GPU_SOURCE_HEADERS})
endif ()

###########################################################################
# pbrt libraries and executables

list (APPEND PBRT_DEFINITIONS "PTEX_STATIC")

######################
# soac

add_executable (soac src/pbrt/cmd/soac.cpp)
add_executable (pbrt::soac ALIAS soac)

target_compile_definitions (soac PRIVATE ${PBRT_DEFINITIONS})
target_compile_options (soac PUBLIC ${PBRT_CXX_FLAGS})
target_link_libraries (soac PRIVATE pbrt_warnings pbrt_opt)

set_target_properties (soac PROPERTIES OUTPUT_NAME soac)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pbrt_soa.h
    COMMAND soac ${CMAKE_SOURCE_DIR}/src/pbrt/pbrt.soa > ${CMAKE_CURRENT_BINARY_DIR}/pbrt_soa.h
    DEPENDS soac ${CMAKE_SOURCE_DIR}/src/pbrt/pbrt.soa)
set (PBRT_SOA_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/pbrt_soa.h)

if (PBRT_CUDA_ENABLED)
  add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_workitems_soa.h
      COMMAND soac ${CMAKE_SOURCE_DIR}/src/pbrt/gpu/workitems.soa > ${CMAKE_CURRENT_BINARY_DIR}/gpu_workitems_soa.h
      DEPENDS soac ${CMAKE_SOURCE_DIR}/src/pbrt/gpu/workitems.soa)
  set (PBRT_SOA_GENERATED ${PBRT_SOA_GENERATED} ${CMAKE_CURRENT_BINARY_DIR}/gpu_workitems_soa.h)
endif ()

add_custom_target (pbrt_soa_generated DEPENDS ${PBRT_SOA_GENERATED})

######################
# pbrt_lib

add_library (pbrt_lib STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_srgb.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_dci_p3.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_rec2020.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_aces.cpp
  ${PBRT_SOA_GENERATED}
  ${PBRT_SOURCE}
  ${PBRT_SOURCE_HEADERS}
  ${PBRT_UTIL_SOURCE}
  ${PBRT_UTIL_SOURCE_HEADERS}
  ${PBRT_GPU_SOURCE}
  ${PBRT_GPU_SOURCE_HEADERS}

  src/ext/gtest/gtest-all.cc
  src/ext/lodepng/lodepng.cpp
  src/ext/rply/rply.cpp
  )
add_library (pbrt::pbrt_lib ALIAS pbrt_lib)

add_dependencies (pbrt_lib pbrt_soa_generated)

target_compile_definitions (pbrt_lib PRIVATE ${PBRT_DEFINITIONS})

# Attempt to work-around Windows/CUDA build issues.
# As per https://stackoverflow.com/a/51566919, this works around a cmake
# bug that leads to undefined symbols...
set_property(TARGET pbrt_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories (pbrt_lib PUBLIC
  src
  src/ext
  ${STB_INCLUDE}
  ${OPENEXR_INCLUDE}
  ${ZLIB_INCLUDE_DIRS}
  ${FILESYSTEM_INCLUDE}
  ${PTEX_INCLUDE}
  ${DOUBLE_CONVERSION_INCLUDE}
  ${NANOVDB_INCLUDE}
  ${CMAKE_CURRENT_BINARY_DIR}
)
if (PBRT_CUDA_ENABLED AND PBRT_OPTIX7_PATH)
    target_include_directories (pbrt_lib SYSTEM PUBLIC ${PBRT_OPTIX7_PATH}/include)
endif ()

target_compile_options (pbrt_lib PUBLIC ${PBRT_CXX_FLAGS})

target_link_libraries (pbrt_lib PRIVATE pbrt_warnings pbrt_opt $<$<BOOL:PBRT_CUDA_ENABLED>:cuda_build_configuration>)

add_sanitizers (pbrt_lib)

if (WIN32)
  # Avoid a name clash when building on Visual Studio
  set_target_properties (pbrt_lib PROPERTIES OUTPUT_NAME libpbrt)
endif()

set (ALL_PBRT_LIBS
  pbrt_lib
  ${CMAKE_THREAD_LIBS_INIT}
  ${OPENEXR_LIBS}
  Ptex_static
  ${ZLIB_LIBRARIES}
  double-conversion
  ${PBRT_CUDA_LIB}
)

if (PBRT_CUDA_ENABLED)
  set_property (TARGET pbrt_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  add_library (pbrt_embedded_ptx_lib STATIC
      ${PBRT_EMBEDDED_PTX}
      )
  add_dependencies (pbrt_embedded_ptx_lib pbrt_soa_generated)
  list (APPEND ALL_PBRT_LIBS pbrt_embedded_ptx_lib)
endif()

if (WIN32)
  list (APPEND ALL_PBRT_LIBS "dbghelp" "wsock32" "ws2_32")
endif ()

if (PROFILE_LIB)
  list (APPEND ALL_PBRT_LIBS "${PROFILE_LIB}")
endif ()

######################
## rgb2spec_opt

add_executable (rgb2spec_opt src/pbrt/cmd/rgb2spec_opt.cpp)
add_executable (pbrt::rgb2spec_opt ALIAS rgb2spec_opt)

target_compile_definitions (rgb2spec_opt PRIVATE ${PBRT_DEFINITIONS})
target_compile_options (rgb2spec_opt PUBLIC ${PBRT_CXX_FLAGS})
target_link_libraries (rgb2spec_opt PRIVATE ${CMAKE_THREAD_LIBS_INIT} pbrt_opt pbrt_warnings)

if (NOT PBRT_USE_PREGENERATED_RGB_TO_SPECTRUM_TABLES)
  add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_aces.cpp
      COMMAND rgb2spec_opt 64 ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_aces.cpp ACES2065_1 
      DEPENDS rgb2spec_opt)

  add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_dci_p3.cpp
      COMMAND rgb2spec_opt 64 ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_dci_p3.cpp DCI_P3
      DEPENDS rgb2spec_opt)

  add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_rec2020.cpp
      COMMAND rgb2spec_opt 64 ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_rec2020.cpp REC2020
      DEPENDS rgb2spec_opt)

  add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_srgb.cpp
      COMMAND rgb2spec_opt 64 ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_srgb.cpp sRGB
      DEPENDS rgb2spec_opt)
endif ()

######################
# Main renderer

add_executable (pbrt_exe src/pbrt/cmd/pbrt.cpp)
add_executable (pbrt::pbrt_exe ALIAS pbrt_exe)

target_compile_definitions (pbrt_exe PRIVATE ${PBRT_DEFINITIONS})
target_compile_options (pbrt_exe PRIVATE ${PBRT_CXX_FLAGS})
target_include_directories (pbrt_exe PRIVATE src src/ext)
target_link_libraries (pbrt_exe PRIVATE ${ALL_PBRT_LIBS} pbrt_opt pbrt_warnings)

set_target_properties (pbrt_exe PROPERTIES OUTPUT_NAME pbrt)

add_sanitizers (pbrt_exe)

######################
# imgtool

add_executable (imgtool src/pbrt/cmd/imgtool.cpp)
add_executable (pbrt::imgtool ALIAS imgtool)

add_library (sky_lib STATIC src/ext/skymodel/ArHosekSkyModel.c)
set_property (TARGET sky_lib PROPERTY FOLDER "ext")

target_compile_definitions (imgtool PRIVATE ${PBRT_DEFINITIONS})
target_compile_options (imgtool PRIVATE ${PBRT_CXX_FLAGS})
target_include_directories (imgtool PRIVATE src src/ext ${FLIP_INCLUDE} )
target_link_libraries (imgtool PRIVATE ${ALL_PBRT_LIBS} pbrt_opt pbrt_warnings sky_lib flip_lib)

add_sanitizers (imgtool)

######################
# pspec

add_executable (pspec src/pbrt/cmd/pspec.cpp)
add_executable (pbrt::pspec ALIAS pspec)

target_compile_definitions (pspec PRIVATE ${PBRT_DEFINITIONS})
target_compile_options (pspec PRIVATE ${PBRT_CXX_FLAGS})
target_include_directories (pspec PRIVATE src src/ext)
target_link_libraries (pspec PRIVATE ${ALL_PBRT_LIBS})

add_sanitizers (pspec)

######################
# cyhair2pbrt

add_executable (cyhair2pbrt src/pbrt/cmd/cyhair2pbrt.cpp)

target_compile_definitions (cyhair2pbrt PRIVATE ${PBRT_DEFINITIONS})
target_compile_options (cyhair2pbrt PRIVATE ${PBRT_CXX_FLAGS})
target_link_libraries (cyhair2pbrt PRIVATE pbrt_opt pbrt_warnings)

add_sanitizers (cyhair2pbrt)

##################
# Unit tests

set (PBRT_TEST_SOURCE
  src/pbrt/bsdfs_test.cpp
  src/pbrt/filters_test.cpp
  src/pbrt/lights_test.cpp
  src/pbrt/lightsamplers_test.cpp
  src/pbrt/media_test.cpp
  src/pbrt/parser_test.cpp
  src/pbrt/samplers_test.cpp
  src/pbrt/shapes_test.cpp

  src/pbrt/cpu/integrators_test.cpp

  src/pbrt/util/args_test.cpp
  src/pbrt/util/bits_test.cpp
  src/pbrt/util/buffercache_test.cpp
  src/pbrt/util/color_test.cpp
  src/pbrt/util/containers_test.cpp
  src/pbrt/util/file_test.cpp
  src/pbrt/util/float_test.cpp
  src/pbrt/util/hash_test.cpp
  src/pbrt/util/image_test.cpp
  src/pbrt/util/math_test.cpp
  src/pbrt/util/parallel_test.cpp
  src/pbrt/util/print_test.cpp
  src/pbrt/util/pstd_test.cpp
  src/pbrt/util/rng_test.cpp
  src/pbrt/util/sampling_test.cpp
  src/pbrt/util/spectrum_test.cpp
  src/pbrt/util/splines_test.cpp
  src/pbrt/util/taggedptr_test.cpp
  src/pbrt/util/transform_test.cpp
  src/pbrt/util/vecmath_test.cpp
  )

add_executable (pbrt_test src/pbrt/cmd/pbrt_test.cpp ${PBRT_TEST_SOURCE})

target_link_libraries (pbrt_test PRIVATE ${ALL_PBRT_LIBS} pbrt_opt pbrt_warnings)
target_compile_definitions (pbrt_test PRIVATE ${PBRT_DEFINITIONS})
target_include_directories (pbrt_test PRIVATE src src/ext ${DOUBLE_CONVERSION_INCLUDE})
target_compile_options(pbrt_test PUBLIC ${PBRT_CXX_FLAGS})

add_sanitizers (pbrt_test)

add_test (pbrt_unit_test pbrt_test)

###############################
# Installation

install (TARGETS
  pbrt_exe
  imgtool
  pspec
  cyhair2pbrt
  DESTINATION
  bin
  )

install (TARGETS
  pbrt_lib
  DESTINATION
  lib
  )
