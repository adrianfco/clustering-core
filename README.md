# clustering-core

Implementation of **KMeans** and **Possibilistic Fuzzy C-Means (PFCM)**
for clustering multispectral images.

This project is developed as a learning, with a focus on
efficient **C++** design, modular structure, and optional **CUDA** acceleration. It is intended
to serve as a reusable core library that can later be integrated into higher-level applications (e.g. QGIS plugin).

## Features

- CPU implementations in **C++20** 
- Optional **CUDA** GPU acceleration
- Benchmark executables for performance evaluation
- Unit tests for basic correctness checks

## Project Structure
.\
├── include/ # Public headers\
├── src/\
│ ├── cpu/ # CPU implementations\
│ └── cuda/ # CUDA implementations\
├── benchmarks/ # Benchmark programs\
├── tests/ # Unit tests\
└── CMakeLists.txt


## Build

This project uses **CMake**.

### Requirements
- CMake ≥ 3.18
- C++ compiler with C++20 support
- CUDA Toolkit (optional, for GPU acceleration)

### Configure and build

From the project root:

```
cmake -S . -B build
cmake --build build
```


After building, executables are located in the build/ directory.

### CPU benchmark
```
./build/benchmark_cpu
```
### Unit tests
```
./build/test_clustering
```

### CUDA benchmark (if CUDA is available)
```
./build/benchmark_cuda
```

## Algorithms

* KMeans clustering

* Possibilistic Fuzzy C-Means (PFCM) clustering

The implementations are designed to operate on multi-dimensional feature vectors,
such as spectral bands in multispectral or hyperspectral imagery.

## Status

Work in progress