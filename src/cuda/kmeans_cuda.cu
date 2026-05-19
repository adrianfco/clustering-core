#include "clustering_core/kmeans.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

//  CUDA error helper 
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(                                           \
                std::string("CUDA error at " __FILE__ ":") +                   \
                std::to_string(__LINE__) + " — " +                             \
                cudaGetErrorString(_e));                                         \
    } while (0)

namespace {

constexpr int BLOCK = 256;

// Assign each point to its nearest centroid.
//   data      — SoA [B × N]: band b of point p → data[b*N + p]
//   centroids —     [k × B]: band b of centroid c → centroids[c*B + b]
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

// Per-cluster sums and counts via shared-memory partials.
__global__ void accumulate_kernel(
    const float* __restrict__ data,
    const int*   __restrict__ labels,
    float*       sums,
    int*         counts,
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

} // namespace

namespace clustering {

KMeansGpu::KMeansGpu(KMeansParams params) : params_(params) {}

void KMeansGpu::fit(const Dataset& data) {
    labels_.clear();
    centroids_.clear();
    n_iters_   = 0;
    converged_ = false;

    const int N = data.n;
    const int B = data.d;
    if (N == 0 || B == 0) return;

    const int k = std::min(params_.k, N);

    // Backend prefers SoA; if data already is, this is a no-op.
    const Dataset soa = data.as(Layout::SoA);

    // Initialise centroids on host (random point selection)
    std::mt19937 rng(static_cast<unsigned>(params_.seed));
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    centroids_.assign(static_cast<std::size_t>(k) * B, 0.f);
    for (int c = 0; c < k; ++c)
        for (int b = 0; b < B; ++b)
            centroids_[c * B + b] =
                soa.features[static_cast<std::size_t>(b) * N + perm[c]];

    // Device allocations
    float* d_data       = nullptr;
    float* d_centroids  = nullptr;
    float* d_sums       = nullptr;
    int*   d_counts     = nullptr;
    int*   d_labels     = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data,      static_cast<std::size_t>(N) * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, static_cast<std::size_t>(k) * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums,      static_cast<std::size_t>(k) * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts,    static_cast<std::size_t>(k)     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels,    static_cast<std::size_t>(N)     * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, soa.features.data(),
                          static_cast<std::size_t>(N) * B * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids_.data(),
                          static_cast<std::size_t>(k) * B * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int    grid_N      = (N      + BLOCK - 1) / BLOCK;
    const int    grid_kB     = (k * B  + BLOCK - 1) / BLOCK;
    const std::size_t assign_smem = static_cast<std::size_t>(k) * B * sizeof(float);
    const std::size_t accum_smem  = static_cast<std::size_t>(k) * B * sizeof(float)
                                  + static_cast<std::size_t>(k)     * sizeof(int);

    std::vector<float> h_new_centroids(static_cast<std::size_t>(k) * B);
    std::vector<int>   h_counts(k);

    for (int iter = 0; iter < params_.max_iters; ++iter) {
        ++n_iters_;

        assign_kernel<<<grid_N, BLOCK, assign_smem>>>(
            d_data, d_centroids, d_labels, N, B, k);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_sums,   0, static_cast<std::size_t>(k) * B * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, static_cast<std::size_t>(k)     * sizeof(int)));

        accumulate_kernel<<<grid_N, BLOCK, accum_smem>>>(
            d_data, d_labels, d_sums, d_counts, N, B, k);
        CUDA_CHECK(cudaGetLastError());

        normalize_kernel<<<grid_kB, BLOCK>>>(
            d_centroids, d_sums, d_counts, k, B);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_new_centroids.data(), d_centroids,
                              static_cast<std::size_t>(k) * B * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                              static_cast<std::size_t>(k) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Re-seed empty clusters
        bool reseeded = false;
        for (int c = 0; c < k; ++c) {
            if (h_counts[c] == 0) {
                const int rp = static_cast<int>(rng() % static_cast<unsigned>(N));
                for (int b = 0; b < B; ++b)
                    h_new_centroids[c * B + b] =
                        soa.features[static_cast<std::size_t>(b) * N + rp];
                reseeded = true;
            }
        }
        if (reseeded)
            CUDA_CHECK(cudaMemcpy(d_centroids, h_new_centroids.data(),
                                  static_cast<std::size_t>(k) * B * sizeof(float),
                                  cudaMemcpyHostToDevice));

        // Convergence: max centroid shift (Euclidean)
        float max_shift = 0.f;
        for (int c = 0; c < k; ++c) {
            float d2 = 0.f;
            for (int b = 0; b < B; ++b) {
                float diff = h_new_centroids[c * B + b] - centroids_[c * B + b];
                d2 += diff * diff;
            }
            max_shift = std::max(max_shift, std::sqrt(d2));
        }

        centroids_ = h_new_centroids;

        if (max_shift < params_.tol) {
            converged_ = true;
            break;
        }
    }

    labels_.assign(N, 0);
    CUDA_CHECK(cudaMemcpy(labels_.data(), d_labels,
                          static_cast<std::size_t>(N) * sizeof(int),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_labels);
}

} // namespace clustering
