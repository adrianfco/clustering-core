#include "clustering_core/kmeans_gpu.hpp"
#include "clustering_core/tif_loader.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Error checking 
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(                                           \
                std::string("CUDA error at " __FILE__ ":") +                   \
                std::to_string(__LINE__) + " — " +                             \
                cudaGetErrorString(_e));                                         \
    } while (0)

static constexpr int BLOCK = 256;


// Assign each point to its nearest centroid.
//
// data      — SoA [B × N]: band b of point p → data[b*N + p]
// centroids —     [k × B]: band b of centroid c → centroids[c*B + b]
//
// Centroids are cooperatively loaded into shared memory so every distance
// computation reads from L1 cache rather than global memory.
__global__ void assign_kernel(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    int*         __restrict__ labels,
    int N, int B, int k)
{
    extern __shared__ float s_centroids[];   // k × B floats

    for (int i = threadIdx.x; i < k * B; i += blockDim.x)
        s_centroids[i] = centroids[i];
    __syncthreads();

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;

    float best_d2 = FLT_MAX;
    int   best_c  = 0;

    for (int c = 0; c < k; ++c) {
        float d2 = 0.f;
        const float* cent = s_centroids + c * B;
        for (int b = 0; b < B; ++b) {
            float diff = data[b * N + p] - cent[b];
            d2 += diff * diff;
        }
        if (d2 < best_d2) { best_d2 = d2; best_c = c; }
    }

    labels[p] = best_c;
}

// Accumulate per-cluster sums and counts into shared-memory partial buffers,
// then flush to global with one atomicAdd per element instead of one per point.
//
// smem layout: [k*B floats (sums) | k ints (counts)]
__global__ void accumulate_kernel(
    const float* __restrict__ data,
    const int*   __restrict__ labels,
    float*       sums,     // [k × B] pre-zeroed global accumulators
    int*         counts,   // [k]     pre-zeroed global accumulators
    int N, int B, int k)
{
    extern __shared__ float smem[];
    float* s_sums   = smem;
    int*   s_counts = reinterpret_cast<int*>(smem + k * B);

    for (int i = threadIdx.x; i < k * B; i += blockDim.x) s_sums[i]   = 0.f;
    for (int i = threadIdx.x; i < k;     i += blockDim.x) s_counts[i] = 0;
    __syncthreads();

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < N) {
        const int c = labels[p];
        atomicAdd(&s_counts[c], 1);
        for (int b = 0; b < B; ++b)
            atomicAdd(&s_sums[c * B + b], data[b * N + p]);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < k * B; i += blockDim.x)
        atomicAdd(&sums[i],   s_sums[i]);
    for (int i = threadIdx.x; i < k;     i += blockDim.x)
        atomicAdd(&counts[i], s_counts[i]);
}

// Divide accumulated sums by counts to produce new centroids.
// Empty clusters (count == 0) are left unchanged; the host re-seeds them.
__global__ void normalize_kernel(
    float*       centroids,
    const float* sums,
    const int*   counts,
    int k, int B)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * B) return;
    const int c = idx / B;
    if (counts[c] > 0)
        centroids[idx] = sums[idx] / static_cast<float>(counts[c]);
}

// Host entry point 
std::vector<int> kmeans_gpu(const GpuData& hdata, int k, int max_iters,
                             float tol, int seed)
{
    const int N     = hdata.n_pixels;
    const int B     = hdata.n_bands;
    const int eff_k = std::min(k, N);

    if (N == 0) return {};

    // Initialise centroids on CPU (random point selection, same as CPU path) 
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // [eff_k × B]: band b of centroid c = h_centroids[c*B + b]
    std::vector<float> h_centroids(eff_k * B);
    for (int c = 0; c < eff_k; ++c)
        for (int b = 0; b < B; ++b)
            h_centroids[c * B + b] =
                hdata.pixels[static_cast<size_t>(b) * N + perm[c]];

    // Device allocations 
    float* d_data;
    float* d_centroids;
    float* d_sums;
    int*   d_counts;
    int*   d_labels;

    CUDA_CHECK(cudaMalloc(&d_data,      static_cast<size_t>(N)     * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, static_cast<size_t>(eff_k) * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums,      static_cast<size_t>(eff_k) * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts,    static_cast<size_t>(eff_k)     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels,    static_cast<size_t>(N)         * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data,      hdata.pixels.data(),
                          static_cast<size_t>(N) * B * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                          static_cast<size_t>(eff_k) * B * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch parameters 
    const int    grid_N       = (N          + BLOCK - 1) / BLOCK;
    const int    grid_kB      = (eff_k * B  + BLOCK - 1) / BLOCK;
    const size_t assign_smem  = static_cast<size_t>(eff_k) * B * sizeof(float);
    const size_t accum_smem   = static_cast<size_t>(eff_k) * B * sizeof(float)
                              + static_cast<size_t>(eff_k)     * sizeof(int);

    std::vector<float> h_new_centroids(eff_k * B);
    std::vector<int>   h_counts(eff_k);

    // Main loop 
    for (int iter = 0; iter < max_iters; ++iter) {

        assign_kernel<<<grid_N, BLOCK, assign_smem>>>(
            d_data, d_centroids, d_labels, N, B, eff_k);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_sums,   0, static_cast<size_t>(eff_k) * B * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, static_cast<size_t>(eff_k)     * sizeof(int)));

        accumulate_kernel<<<grid_N, BLOCK, accum_smem>>>(
            d_data, d_labels, d_sums, d_counts, N, B, eff_k);
        CUDA_CHECK(cudaGetLastError());

        normalize_kernel<<<grid_kB, BLOCK>>>(
            d_centroids, d_sums, d_counts, eff_k, B);
        CUDA_CHECK(cudaGetLastError());

        // Pull centroids + counts to CPU — tiny transfer (k*B floats)
        CUDA_CHECK(cudaMemcpy(h_new_centroids.data(), d_centroids,
                              static_cast<size_t>(eff_k) * B * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                              static_cast<size_t>(eff_k) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Re-seed empty clusters
        bool reseeded = false;
        for (int c = 0; c < eff_k; ++c) {
            if (h_counts[c] == 0) {
                const int rp = static_cast<int>(
                    rng() % static_cast<unsigned>(N));
                for (int b = 0; b < B; ++b)
                    h_new_centroids[c * B + b] =
                        hdata.pixels[static_cast<size_t>(b) * N + rp];
                reseeded = true;
            }
        }
        if (reseeded)
            CUDA_CHECK(cudaMemcpy(d_centroids, h_new_centroids.data(),
                                  static_cast<size_t>(eff_k) * B * sizeof(float),
                                  cudaMemcpyHostToDevice));

        // Convergence: max centroid shift (Euclidean)
        float max_shift = 0.f;
        for (int c = 0; c < eff_k; ++c) {
            float d2 = 0.f;
            for (int b = 0; b < B; ++b) {
                float diff = h_new_centroids[c * B + b] - h_centroids[c * B + b];
                d2 += diff * diff;
            }
            max_shift = std::max(max_shift, sqrtf(d2));
        }

        h_centroids = h_new_centroids;

        if (max_shift < tol) break;
    }

    // Retrieve labels 
    std::vector<int> h_labels(N);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                          static_cast<size_t>(N) * sizeof(int),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_labels);

    return h_labels;
}
