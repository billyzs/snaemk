find_package(GTest REQUIRED FATAL)
enable_testing()
add_executable(test-k-means test-kmeans.cpp)
target_link_libraries(test-k-means snaemk gtest gtest_main pthread)

gtest_discover_tests(
        test-k-means
)
