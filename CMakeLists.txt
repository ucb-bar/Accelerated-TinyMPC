cmake_minimum_required(VERSION 3.15.0)
project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)

option(USE_MATVEC "Use matvec instead of matmul for matrix multiplication" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_MATVEC)
    add_compile_definitions(USE_MATVEC=1)
endif(USE_MATVEC)

# Add optimization flags
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -Og")

set(STATIC_LINKING TRUE)

include_directories(include/matlib)
include_directories(include/tinympc)
add_subdirectory(src)
add_subdirectory(examples)
