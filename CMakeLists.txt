cmake_minimum_required(VERSION 3.18)
project(clustering-core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# ── Catch2 v3 via FetchContent ───────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
)
FetchContent_MakeAvailable(Catch2)

# ── GDAL ─────────────────────────────────────────────────────────────────────
find_package(GDAL REQUIRED)

# ── Core: shared types (Dataset, Clusterer interfaces) ───────────────────────
add_library(clustering_core STATIC
    src/dataset.cpp
)

# ── CPU algorithms ───────────────────────────────────────────────────────────
add_library(clustering_cpu STATIC
    src/cpu/kmeans_cpu.cpp
    src/cpu/pfcm_cpu.cpp
)
target_link_libraries(clustering_cpu PUBLIC clustering_core)

# ── I/O (GDAL TIF loader) ────────────────────────────────────────────────────
add_library(clustering_io STATIC
    src/io/tif_loader.cpp
)
target_link_libraries(clustering_io PUBLIC clustering_core GDAL::GDAL)

# ── CPU benchmark ────────────────────────────────────────────────────────────
add_executable(benchmark_cpu benchmarks/main_benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu PRIVATE clustering_cpu)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_clustering
    tests/main_test.cpp
    tests/test_tif_loader.cpp
    tests/test_pfcm.cpp
    tests/test_dataset.cpp
)
target_link_libraries(test_clustering PRIVATE
    clustering_cpu
    clustering_io
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(test_clustering)

# ── CUDA library + targets (optional) ────────────────────────────────────────
find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_library(clustering_gpu STATIC
        src/cuda/kmeans_cuda.cu
        src/cuda/pfcm_cuda.cu
        src/cuda/device_utils.cu
    )
    target_link_libraries(clustering_gpu PUBLIC clustering_core)

    add_executable(benchmark_cuda benchmarks/main_benchmark_cuda.cu)
    target_link_libraries(benchmark_cuda PRIVATE clustering_gpu clustering_io)

    add_executable(test_clustering_gpu tests/test_kmeans_gpu.cpp)
    target_link_libraries(test_clustering_gpu PRIVATE
        clustering_gpu
        Catch2::Catch2WithMain
    )
    catch_discover_tests(test_clustering_gpu)
endif()
