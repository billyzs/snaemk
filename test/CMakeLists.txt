find_package(GTest REQUIRED FATAL)
enable_testing()
add_executable(test-kmeans test-kmeans.cpp)
target_link_libraries(test-kmeans snaemk gtest gtest_main pthread tbb)
target_compile_features(test-kmeans PRIVATE cxx_return_type_deduction)
gtest_discover_tests(
        test-kmeans
)
