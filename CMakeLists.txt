cmake_minimum_required(VERSION 3.18)
project(clustering-core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# CPU library
add_library(clustering_cpu STATIC
    src/cpu/kmeans_cpu.cpp
    src/cpu/pfcm_cpu.cpp
    src/cpu/utils.cpp
)

# CPU benchmark
add_executable(benchmark_cpu benchmarks/main_benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu PRIVATE clustering_cpu)

# Tests
add_executable(test_clustering tests/main_test.cpp)
target_link_libraries(test_clustering PRIVATE clustering_cpu)

# CUDA library
find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_library(clustering_cuda STATIC
        src/cuda/kmeans_cuda.cu
        src/cuda/pfcm_cuda.cu
        src/cuda/device_utils.cu
    )
    target_link_libraries(clustering_cuda PUBLIC clustering_cpu)

    add_executable(benchmark_cuda benchmarks/main_benchmark_cuda.cu)
    target_link_libraries(benchmark_cuda PRIVATE clustering_cuda)
endif()
