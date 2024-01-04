
add_executable(test1 test1.cpp)
add_executable(test_gemmini test_gemmini.cpp)

target_link_libraries(test_gemmini LINK_PUBLIC tinympc)
