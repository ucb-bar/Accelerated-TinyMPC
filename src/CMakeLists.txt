# if(USE_GEMMINI)
#     add_library (tinympc STATIC tinympc/admm_eigen.cpp)
# endif(USE_GEMMINI)

if(USE_RVV OR USE_CPU OR USE_GEMMINI)
    add_library (tinympc STATIC tinympc/admm_rvv.cpp)
    add_library (matlib STATIC matlib/matlib.cpp)
    target_include_directories (matlib PUBLIC ../include)
endif(USE_RVV OR USE_CPU OR USE_GEMMINI)

target_include_directories (tinympc PUBLIC ../include)
