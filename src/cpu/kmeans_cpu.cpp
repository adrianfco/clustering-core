#include "kmeans.hpp"
#include "utils.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

KMeans::KMeans(int k, int max_iters, double tol, int seed)
    : k_(k), max_iters_(max_iters), tol_(tol), seed_(seed) {}

void KMeans::fit(const std::vector<std::vector<double>>& data) {
    const int n = static_cast<int>(data.size());
    if (n == 0) return;

    const int dims = static_cast<int>(data[0].size());
    const int effective_k = std::min(k_, n);

    std::mt19937 rng(static_cast<unsigned>(seed_));

    // Random init: shuffle indices, take first effective_k as initial centroids
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<std::vector<double>> centroids(effective_k, std::vector<double>(dims));
    for (int c = 0; c < effective_k; ++c)
        centroids[c] = data[idx[c]];

    labels_.assign(n, 0);

    for (int iter = 0; iter < max_iters_; ++iter) {
        // Assign each point to the nearest centroid
        std::vector<int> new_labels(n);
        for (int p = 0; p < n; ++p) {
            int best_c = 0;
            double best_dist = euclidean_distance(data[p], centroids[0]);
            for (int c = 1; c < effective_k; ++c) {
                double d = euclidean_distance(data[p], centroids[c]);
                if (d < best_dist) {
                    best_dist = d;
                    best_c = c;
                }
            }
            new_labels[p] = best_c;
        }

        // Update centroids as mean of assigned points
        std::vector<std::vector<double>> new_centroids(effective_k, std::vector<double>(dims, 0.0));
        std::vector<int> counts(effective_k, 0);
        for (int p = 0; p < n; ++p) {
            int c = new_labels[p];
            ++counts[c];
            for (int d = 0; d < dims; ++d)
                new_centroids[c][d] += data[p][d];
        }
        for (int c = 0; c < effective_k; ++c) {
            if (counts[c] == 0) {
                // Empty cluster: re-seed at a random point
                new_centroids[c] = data[rng() % static_cast<unsigned>(n)];
            } else {
                for (int d = 0; d < dims; ++d)
                    new_centroids[c][d] /= counts[c];
            }
        }

        // Check convergence: max centroid shift
        double max_shift = 0.0;
        for (int c = 0; c < effective_k; ++c) {
            double shift = euclidean_distance(centroids[c], new_centroids[c]);
            if (shift > max_shift) max_shift = shift;
        }

        centroids = std::move(new_centroids);
        labels_ = new_labels;

        if (max_shift < tol_) break;
    }
}

const std::vector<int>& KMeans::labels() const {
    return labels_;
}
