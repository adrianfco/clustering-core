#include "clustering_core/kmeans.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>

namespace clustering {

namespace {

inline float sq_dist(const float* a, const float* b, int d) {
    float s = 0.f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

} // namespace

KMeansCpu::KMeansCpu(KMeansParams params) : params_(params) {}

void KMeansCpu::fit(const Dataset& data) {
    labels_.clear();
    centroids_.clear();
    n_iters_   = 0;
    converged_ = false;

    const int n = data.n;
    const int d = data.d;
    if (n == 0 || d == 0) return;

    const int k = std::min(params_.k, n);
    const Dataset aos = data.as(Layout::AoS);
    const float* feat = aos.features.data();

    std::mt19937 rng(static_cast<unsigned>(params_.seed));

    // Random init: shuffle indices, pick first k as initial centroids.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    centroids_.resize(static_cast<std::size_t>(k) * d);
    for (int c = 0; c < k; ++c) {
        const float* src = feat + static_cast<std::size_t>(idx[c]) * d;
        float*       dst = centroids_.data() + static_cast<std::size_t>(c) * d;
        for (int b = 0; b < d; ++b) dst[b] = src[b];
    }

    labels_.assign(n, 0);
    std::vector<float> new_centroids(static_cast<std::size_t>(k) * d);
    std::vector<int>   counts(k);

    for (int iter = 0; iter < params_.max_iters; ++iter) {
        ++n_iters_;

        // Assign each point to the nearest centroid.
        for (int p = 0; p < n; ++p) {
            const float* x = feat + static_cast<std::size_t>(p) * d;
            int   best_c    = 0;
            float best_dist = sq_dist(x, centroids_.data(), d);
            for (int c = 1; c < k; ++c) {
                float dist = sq_dist(x, centroids_.data() + static_cast<std::size_t>(c) * d, d);
                if (dist < best_dist) { best_dist = dist; best_c = c; }
            }
            labels_[p] = best_c;
        }

        // Recompute centroids as the mean of assigned points.
        std::fill(new_centroids.begin(), new_centroids.end(), 0.f);
        std::fill(counts.begin(), counts.end(), 0);
        for (int p = 0; p < n; ++p) {
            int c = labels_[p];
            ++counts[c];
            float*       dst = new_centroids.data() + static_cast<std::size_t>(c) * d;
            const float* x   = feat + static_cast<std::size_t>(p) * d;
            for (int b = 0; b < d; ++b) dst[b] += x[b];
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                // Re-seed empty cluster at a random point.
                const float* src = feat + static_cast<std::size_t>(
                    rng() % static_cast<unsigned>(n)) * d;
                float* dst = new_centroids.data() + static_cast<std::size_t>(c) * d;
                for (int b = 0; b < d; ++b) dst[b] = src[b];
            } else {
                float inv = 1.f / static_cast<float>(counts[c]);
                float* dst = new_centroids.data() + static_cast<std::size_t>(c) * d;
                for (int b = 0; b < d; ++b) dst[b] *= inv;
            }
        }

        // Convergence: max centroid shift (squared).
        float max_shift_sq = 0.f;
        for (int c = 0; c < k; ++c) {
            float s = sq_dist(centroids_.data() + static_cast<std::size_t>(c) * d,
                              new_centroids.data() + static_cast<std::size_t>(c) * d, d);
            if (s > max_shift_sq) max_shift_sq = s;
        }

        centroids_.swap(new_centroids);

        if (max_shift_sq < params_.tol * params_.tol) {
            converged_ = true;
            break;
        }
    }
}

} // namespace clustering
