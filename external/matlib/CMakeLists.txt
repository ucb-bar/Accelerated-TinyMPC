cmake_minimum_required(VERSION 3.15.0)

message("Processing ${CMAKE_CURRENT_LIST_DIR}")
if (NOT DEFINED MATLIB_CMAKE_PATH)
    include(${CMAKE_CURRENT_LIST_DIR}/matlib.cmake)
endif()

project(matlib VERSION 0.2.0 LANGUAGES CXX)

if(USE_RVV OR USE_RVVU OR USE_CPU)

    add_library (matlib STATIC src/matlib.cpp)
    target_include_directories (matlib PUBLIC include)

    add_executable(test_matlib_rvv test/test_matlib_rvv.cpp)
    target_link_libraries(test_matlib_rvv LINK_PUBLIC matlib)
    target_include_directories (test_matlib_rvv PUBLIC matlib)

endif(USE_RVV OR USE_RVVU OR USE_CPU)

function(add_matlib_executable target_name)

    set(unrolled_files "")
    set(flags WIN32_EXECUTABLE MACOSX_BUNDLE EXCLUDE_FROM_ALL)
    add_executable(${target_name})
    cmake_parse_arguments(arg "${flags}" "" "" ${ARGN})
    get_target_property(directory ${target_name} SOURCE_DIR)
    if (USE_RVVU)
        foreach(original_file ${arg_UNPARSED_ARGUMENTS})
            get_filename_component(basename ${original_file} NAME_WE)
            get_filename_component(extension ${original_file} EXT)
            set(unrolled_file "${directory}/${basename}_rvvu${extension}")
            set(preprocessed_file "{directory}/${basename}.ii")
            set(unrolled_file "${CMAKE_CURRENT_BINARY_DIR}/${basename}_rvvu${extension}")
            set(preprocessed_file "${CMAKE_CURRENT_BINARY_DIR}/${basename}.ii")
            add_custom_command(
                OUTPUT "${unrolled_file}"
                COMMAND "${CMAKE_CXX_COMPILER}" $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -P -o "${preprocessed_file}" -E "${directory}/${original_file}"
                COMMAND "python3" ${MATLIB_CMAKE_PATH}/include/matlib_rvvu.py ${preprocessed_file} ${USE_LMUL} > ${unrolled_file}
                DEPENDS "${original_file}" "${MATLIB_CMAKE_PATH}/include/matlib_rvvu.py"
                VERBATIM COMMAND_EXPAND_LISTS
            )
            list(APPEND unrolled_files ${unrolled_file})
        endforeach()
        target_sources(${target_name} PRIVATE ${unrolled_files})
    else(USE_RVVU)
        target_sources(${target_name} PRIVATE ${arg_UNPARSED_ARGUMENTS})
    endif(USE_RVVU)

endfunction()
