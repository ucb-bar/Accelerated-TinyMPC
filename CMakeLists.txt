cmake_minimum_required(VERSION 3.15.0)

message("Processing ${CMAKE_CURRENT_LIST_DIR}")
include(${CMAKE_CURRENT_LIST_DIR}/external/matlib/matlib.cmake)
add_subdirectory(external/matlib)
add_subdirectory(external/admm)

project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)

include_directories(include/Eigen)
include_directories(include/Gemmini)
include_directories(include/RoSE)

if(USE_GEMMINI)

    add_library (admm STATIC src/admm_eigen.cpp)

    add_executable(test_gemmini test_gemmini.cpp)
    target_link_libraries(test_gemmini LINK_PUBLIC tinympc)

    add_executable(test_admm_eigen test_admm_eigen.cpp)
    target_link_libraries(test_admm_eigen LINK_PUBLIC tinympc)
    target_include_directories (test_admm_eigen PUBLIC tinympc)

    add_executable(example_quadrotor_hovering example_quadrotor_hovering.cpp)
    target_link_libraries(example_quadrotor_hovering LINK_PUBLIC admm)

    add_executable(example_quadrotor_tracking example_quadrotor_tracking.cpp)
    target_link_libraries(example_quadrotor_tracking LINK_PUBLIC admm)

endif(USE_GEMMINI)

if(USE_RVV OR USE_RVVU)

    add_matlib_executable(example_quadrotor_hovering_rvv examples/example_quadrotor_hovering_rvv.cpp)
    target_link_libraries(example_quadrotor_hovering_rvv LINK_PUBLIC admm matlib)
    target_include_directories(example_quadrotor_hovering_rvv PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)

    add_matlib_executable(example_quadrotor_tracking_rvv examples/example_quadrotor_tracking_rvv.cpp)
    target_link_libraries(example_quadrotor_tracking_rvv LINK_PUBLIC admm matlib)
    target_include_directories(example_quadrotor_tracking_rvv PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)

endif(USE_RVV OR USE_RVVU)

if(USE_CPU)

    add_executable(example_quadrotor_hovering_cpu examples/example_quadrotor_hovering_rvv.cpp)
    target_link_libraries(example_quadrotor_hovering_cpu LINK_PUBLIC admm matlib)
    target_include_directories(example_quadrotor_hovering_cpu PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)

    add_executable(example_quadrotor_tracking_cpu examples/example_quadrotor_tracking_rvv.cpp)
    target_link_libraries(example_quadrotor_tracking_cpu LINK_PUBLIC admm matlib)
    target_include_directories(example_quadrotor_tracking_cpu PUBLIC $<TARGET_PROPERTY:admm,INTERFACE_INCLUDE_DIRECTORIES>)

endif(USE_CPU)

# add_executable(example_quadrotor_hovering_rose example_quadrotor_hovering_rose.cpp)
# target_link_libraries(example_quadrotor_hovering_rose LINK_PUBLIC admm matlib)
