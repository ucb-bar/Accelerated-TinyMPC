cmake_minimum_required(VERSION 3.15.0)

message("Processing ${CMAKE_CURRENT_LIST_DIR}")
if (NOT DEFINED MATLIB_CMAKE_PATH)
    include(${CMAKE_CURRENT_LIST_DIR}/external/matlib/matlib.cmake)
endif()

project(admm VERSION 0.2.0 LANGUAGES CXX)

# Print configuration values for diagnostic purposes.
message(STATUS "admm configuration: USE_RVV   = ${USE_RVV}")
message(STATUS "admm configuration: USE_RVVU  = ${USE_RVVU}")
message(STATUS "admm configuration: USE_RVVF  = ${USE_RVVF}")
message(STATUS "admm configuration: USE_CPU   = ${USE_CPU}")
message(STATUS "admm configuration: USE_TYPE  = ${USE_TYPE}")
message(STATUS "admm configuration: MATLIB_EXTERNAL_BUILD = ${MATLIB_EXTERNAL_BUILD}")

if((USE_RVV OR USE_RVVU OR USE_RVVF OR USE_CPU) AND (USE_TYPE STREQUAL "float32" OR USE_TYPE STREQUAL "float64"))

    add_matlib_library (admm src/admm_rvv.cpp)
    target_include_directories (admm PUBLIC include)
    target_include_directories (admm PUBLIC ${MATLIB_CMAKE_PATH}/include/${USE_TYPE})
    target_include_directories (admm PUBLIC ${MATLIB_CMAKE_PATH}/include/)

    # if(NOT EXISTS test/test_admm_rvv.hpp OR test/test_admm_rvv.py IS_NEWER_THAN test/test_admm_rvv.hpp)
    #     add_custom_command(
    #         OUTPUT test/test_admm_rvv.hpp
	#         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_admm_rvv.py
    #         DEPENDS test/test_admm_rvv.py
    #         COMMENT "Generating test_admm_rvv.hpp"
    #     )
    # endif()

    # add_matlib_executable (test_admm_rvv test/test_admm_rvv.cpp test/test_admm_rvv.hpp)
    # target_link_libraries (test_admm_rvv LINK_PUBLIC admm matlib)
    # target_include_directories (test_admm_rvv PUBLIC admm matlib)

endif((USE_RVV OR USE_RVVU OR USE_RVVF OR USE_CPU) AND (USE_TYPE STREQUAL "float32" OR USE_TYPE STREQUAL "float64"))

