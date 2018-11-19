cmake_minimum_required(VERSION 3.12)
project(SNAEMK)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_subdirectory(test)
endif()

# add kmeans as a header-only library
add_library(snaemk INTERFACE)
target_sources(snaemk INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/snaemk/kmeans.hpp)
target_include_directories(snaemk INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        /opt/pstl/include)
target_compile_features(snaemk INTERFACE cxx_std_17) # parallel algorithms
