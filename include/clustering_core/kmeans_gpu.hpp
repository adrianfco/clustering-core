#pragma once
#include "clustering_core/tif_loader.hpp"
#include <vector>

// GPU-accelerated KMeans (Pascal sm_61+).
//
// Data must be in SoA float layout — use TifLoader::load_gpu().
// Internally uses float arithmetic; tol is compared against max centroid shift.
//
// Parameters match the CPU KMeans interface:
//   k          — number of clusters
//   max_iters  — hard iteration cap
//   tol        — convergence threshold (max centroid shift), default 1e-4
//   seed       — RNG seed for centroid initialisation
std::vector<int> kmeans_gpu(const GpuData& data, int k, int max_iters,
                             float tol = 1e-4f, int seed = 42);
