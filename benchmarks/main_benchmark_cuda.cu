#include "clustering_core/kmeans.hpp"
#include "clustering_core/dataset.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using clustering::Dataset;
using clustering::KMeansGpu;
using clustering::Layout;

// Synthetic SoA Dataset: N random float points in B dimensions.
static Dataset make_synthetic(int N, int B, int seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.f, 255.f);

    Dataset d;
    d.n = N;
    d.d = B;
    d.layout = Layout::SoA;
    d.features.resize(static_cast<std::size_t>(B) * N);
    for (auto& v : d.features) v = dist(rng);
    return d;
}

int main() {
    const int N      = 1 << 20;   // ~1 M pixels
    const int B      = 7;         // Landsat-like band count
    const int k      = 10;
    const int iters  = 100;
    const float tol  = 1e-4f;

    std::printf("GPU KMeans benchmark\n");
    std::printf("  N=%d  B=%d  k=%d  max_iters=%d\n\n", N, B, k, iters);

    Dataset data = make_synthetic(N, B);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    KMeansGpu km({.k = k, .max_iters = iters, .tol = tol, .seed = 42});

    cudaEventRecord(t0);
    km.fit(data);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    std::printf("  elapsed: %.1f ms\n", ms);
    std::printf("  labels[0]=%d  labels[N/2]=%d\n",
                km.labels()[0], km.labels()[N / 2]);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
