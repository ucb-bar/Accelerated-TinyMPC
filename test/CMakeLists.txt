if(USE_GEMMINI)
    add_executable(test_gemmini test_gemmini.cpp)
    target_link_libraries(test_gemmini LINK_PUBLIC tinympc)
endif(USE_GEMMINI)

if(USE_RVV)
    add_executable(test_matlib_rvv matlib/test_matlib_rvv.cpp)
    target_link_libraries(test_matlib_rvv LINK_PUBLIC matlib)
    target_include_directories (test_matlib_rvv PUBLIC matlib)
    if(NOT EXISTS tinympc/test_admm_rvv.hpp OR tinympc/test_admm_rvv.py IS_NEWER_THAN tinympc/test_admm_rvv.hpp)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/test_admm_rvv.hpp
            COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/test_admm_rvv.py
            DEPENDS tinympc/test_admm_rvv.py
            COMMENT "Generating tinympc/test_admm_rvv.hpp"
        )
    endif()
    add_executable(test_admm_rvv tinympc/test_admm_rvv.cpp tinympc/test_admm_rvv.hpp)
    target_link_libraries(test_admm_rvv LINK_PUBLIC tinympc matlib)
    target_include_directories (test_admm_rvv PUBLIC tinympc matlib)
endif(USE_RVV)

if(USE_CPU)
    add_executable(test_admm_cpu tinympc/test_admm_cpu.cpp)
    target_link_libraries(test_admm_cpu LINK_PUBLIC tinympc)
    target_include_directories (test_admm_cpu PUBLIC tinympc)
endif(USE_CPU)



