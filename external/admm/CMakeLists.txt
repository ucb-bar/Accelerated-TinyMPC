if(USE_GEMMINI)
    add_library (tinympc STATIC src/admm_eigen.cpp)
endif(USE_GEMMINI)

if(USE_RVV OR USE_CPU)
    add_library (tinympc STATIC src/admm_rvv.cpp)
    target_include_directories (tinympc PUBLIC include)
    target_include_directories (tinympc PUBLIC $<TARGET_PROPERTY:matlib,INTERFACE_INCLUDE_DIRECTORIES>)
endif(USE_RVV OR USE_CPU)

if(USE_RVV OR USE_CPU)
    if(NOT EXISTS test/test_admm_rvv.hpp OR test/test_admm_rvv.py IS_NEWER_THAN test/test_admm_rvv.hpp)
        add_custom_command(
            OUTPUT test/test_admm_rvv.hpp
	        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_admm_rvv.py
            DEPENDS test/test_admm_rvv.py
            COMMENT "Generating test_admm_rvv.hpp"
        )
    endif()
    add_executable (test_admm_rvv test/test_admm_rvv.cpp test/test_admm_rvv.hpp)
    target_link_libraries (test_admm_rvv LINK_PUBLIC tinympc matlib)
    target_include_directories (test_admm_rvv PUBLIC tinympc matlib)
endif(USE_RVV OR USE_CPU)

