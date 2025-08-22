cmake_minimum_required(VERSION 3.15.0)

message("Processing ${CMAKE_CURRENT_LIST_DIR}")
if (NOT DEFINED MATLIB_CMAKE_PATH)
    include(${CMAKE_CURRENT_LIST_DIR}/external/matlib/matlib.cmake)
endif()

project(admm VERSION 0.2.0 LANGUAGES CXX)

if((USE_RVV OR USE_RVVU OR USE_RVVF OR USE_CPU) AND (USE_TYPE STREQUAL "float32" OR USE_TYPE STREQUAL "float64"))


    if(USE_EIGEN)
        message(STATUS "Eigen include directory: ${CMAKE_CURRENT_LIST_DIR}/../../include/Eigen/")
        add_matlib_library (admm src/admm_eigen.cpp)
        if(USE_GEMMINI) 
            target_include_directories (admm PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../include/Gemmini/)
        endif()
        target_include_directories (admm PUBLIC include)
        target_include_directories (admm PUBLIC ${MATLIB_CMAKE_PATH}/include/${USE_TYPE})
        target_include_directories (admm PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../include/Eigen/)
    else()
        add_matlib_library (admm src/admm_matlib.cpp)
        if(USE_GEMMINI) 
            target_include_directories (admm PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../include/Gemmini/)
        endif()
        target_include_directories (admm PUBLIC include)
        target_include_directories (admm PUBLIC ${MATLIB_CMAKE_PATH}/include/${USE_TYPE})
        if(NOT EXISTS test/test_admm_rvv.hpp OR test/test_admm_rvv.py IS_NEWER_THAN test/test_admm_rvv.hpp)
            add_custom_command(
                OUTPUT test/test_admm_rvv.hpp
                COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_admm_rvv.py
                DEPENDS test/test_admm_rvv.py
                COMMENT "Generating test_admm_rvv.hpp"
            )
        endif()

        add_matlib_executable (test_admm_rvv test/test_admm_rvv.cpp test/test_admm_rvv.hpp)
        target_link_libraries (test_admm_rvv LINK_PUBLIC admm matlib)
        target_include_directories (test_admm_rvv PUBLIC admm matlib)
    endif()

endif((USE_RVV OR USE_RVVU OR USE_RVVF OR USE_CPU) AND (USE_TYPE STREQUAL "float32" OR USE_TYPE STREQUAL "float64"))

