add_library (tinympc STATIC tinympc/admm.cpp)
add_library (matlib STATIC matlib/matlib.cpp)
target_include_directories (matlib PUBLIC ../include)

target_include_directories (tinympc PUBLIC ../include)
