if(USE_GEMMINI)
    add_library (tinympc STATIC tinympc/admm_gemmini.cpp)
endif(USE_GEMMINI)

if(USE_CPU)
    add_library (tinympc STATIC tinympc/admm_rvv.cpp)
    add_library (matlib STATIC matlib/matlib_rvv.cpp matlib/common.cpp)
    target_include_directories (matlib PUBLIC ../include)
endif(USE_CPU)

if(USE_RVV)
    add_library (tinympc STATIC tinympc/admm_rvv.cpp)
    add_library (matlib STATIC matlib/matlib_rvv.cpp matlib/common.cpp)
    target_include_directories (matlib PUBLIC ../include)
endif(USE_RVV)

target_include_directories (tinympc PUBLIC ../include)
