if(USE_GEMMINI)
    add_library (tinympc STATIC tinympc/admm_gemmini.cpp)
endif(USE_GEMMINI)

if(USE_CPU)
    add_library (tinympc STATIC tinympc/admm_cpu.cpp)
endif(USE_CPU)

if(USE_RVV)
    add_library (tinympc STATIC tinympc/admm_rvv.cpp)
    add_library (rvv_matlib STATIC rvv_matlib/rvv_matlib.cpp rvv_matlib/common.cpp)
    target_include_directories (rvv_matlib PUBLIC ../include)
endif(USE_RVV)

target_include_directories (tinympc PUBLIC ../include)
