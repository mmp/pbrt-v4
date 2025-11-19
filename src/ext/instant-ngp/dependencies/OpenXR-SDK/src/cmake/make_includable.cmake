# Copyright (c) 2019-2025 The Khronos Group Inc.
#
# SPDX-License-Identifier: Apache-2.0

# inspired by: https://stackoverflow.com/a/47801116
function(make_includable input_file output_file)
    cmake_parse_arguments(
        PARSE_ARGV 2 MAKE_INCLUDABLE "" "" "REPLACE")

    file(READ "${input_file}" content)

    # regex replace
    list(LENGTH MAKE_INCLUDABLE_REPLACE length)
    while(length GREATER_EQUAL 2)
        list(GET MAKE_INCLUDABLE_REPLACE 0 find)
        list(GET MAKE_INCLUDABLE_REPLACE 1 replace)
        list(REMOVE_AT MAKE_INCLUDABLE_REPLACE 0 1)
        string(REGEX REPLACE "${find}" "${replace}" content "${content}")
        list(LENGTH MAKE_INCLUDABLE_REPLACE length)
    endwhile()

    file(WRITE "${output_file}" "R\"raw_text(\n${content})raw_text\"")
    # using https://stackoverflow.com/a/65945763
    # see https://stackoverflow.com/a/56828572 for a different approach
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input_file}")
endfunction(make_includable)
