cmake_minimum_required(VERSION 3.15.0)

message("Processing ${CMAKE_CURRENT_LIST_DIR}")
include(${CMAKE_CURRENT_LIST_DIR}/external/matlib/matlib.cmake)
add_subdirectory(external/matlib)
add_subdirectory(external/admm)
# add_subdirectory(external/rtrbench)

project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)

include_directories(include/Eigen)
include_directories(include/Gemmini)
include_directories(include/RoSE)

if((USE_RVV OR USE_RVVU OR USE_RVVF) AND (USE_TYPE STREQUAL "float32" OR USE_TYPE STREQUAL "float64"))

    add_matlib_executable(example_quadrotor_hovering_rvv examples/example_quadrotor_hovering_rvv.cpp)
    target_link_libraries(example_quadrotor_hovering_rvv LINK_PUBLIC admm matlib)
    target_include_directories(example_quadrotor_hovering_rvv PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)

    add_matlib_executable(example_quadrotor_tracking_rvv examples/example_quadrotor_tracking_rvv.cpp)
    target_link_libraries(example_quadrotor_tracking_rvv LINK_PUBLIC admm matlib)
    target_include_directories(example_quadrotor_tracking_rvv PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)

endif((USE_RVV OR USE_RVVU OR USE_RVVF) AND (USE_TYPE STREQUAL "float32" OR USE_TYPE STREQUAL "float64"))

if(USE_CPU)

    if(USE_EIGEN)
        add_executable(example_quadrotor_tracking_eigen examples/example_quadrotor_tracking.cpp)
        target_link_libraries(example_quadrotor_tracking_eigen LINK_PUBLIC admm)
        target_include_directories(example_quadrotor_tracking_eigen PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)
    else()
        add_executable(example_quadrotor_hovering_cpu examples/example_quadrotor_hovering_rvv.cpp)
        target_link_libraries(example_quadrotor_hovering_cpu LINK_PUBLIC admm matlib)
        target_include_directories(example_quadrotor_hovering_cpu PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)

        add_executable(example_quadrotor_tracking_cpu examples/example_quadrotor_tracking_rvv.cpp)
        target_link_libraries(example_quadrotor_tracking_cpu LINK_PUBLIC admm matlib)
        target_include_directories(example_quadrotor_tracking_cpu PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

endif(USE_CPU)