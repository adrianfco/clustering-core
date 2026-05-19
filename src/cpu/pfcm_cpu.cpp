#include "clustering_core/pfcm.hpp"
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

inline void copy_row(float* dst, const float* src, int d) {
    for (int i = 0; i < d; ++i) dst[i] = src[i];
}

} // namespace

PfcmCpu::PfcmCpu(PfcmParams params) : params_(params) {}

void PfcmCpu::fit(const Dataset& data) {
    labels_.clear();
    centroids_.clear();
    U_.clear();
    T_.clear();
    n_iters_   = 0;
    converged_ = false;

    const int n = data.n;
    const int d = data.d;
    if (n == 0 || d == 0) return;

    const int   c          = std::min(params_.c, n);
    const float m          = params_.m;
    const float alpha      = params_.alpha;
    const float exp_um     = 1.f / (m - 1.f);

    const Dataset aos = data.as(Layout::AoS);
    const float*  feat = aos.features.data();

    std::mt19937 rng(static_cast<unsigned>(params_.seed));

    // Max-min init: first center random, then farthest-from-existing.
    centroids_.assign(static_cast<std::size_t>(c) * d, 0.f);
    {
        std::uniform_int_distribution<int> dist(0, n - 1);
        copy_row(centroids_.data(),
                 feat + static_cast<std::size_t>(dist(rng)) * d, d);
    }
    for (int ci = 1; ci < c; ++ci) {
        int   best_j     = 0;
        float best_score = -1.f;
        for (int j = 0; j < n; ++j) {
            const float* x = feat + static_cast<std::size_t>(j) * d;
            float min_d = std::numeric_limits<float>::max();
            for (int k = 0; k < ci; ++k) {
                float s = sq_dist(x, centroids_.data() + static_cast<std::size_t>(k) * d, d);
                if (s < min_d) min_d = s;
            }
            if (min_d > best_score) { best_score = min_d; best_j = j; }
        }
        copy_row(centroids_.data() + static_cast<std::size_t>(ci) * d,
                 feat + static_cast<std::size_t>(best_j) * d, d);
    }

    const std::size_t cn = static_cast<std::size_t>(c) * n;
    U_.assign(cn, 0.f);
    T_.assign(cn, 0.f);
    std::vector<float> gamma(c, 1.f);
    std::vector<float> dsq(cn);
    std::vector<float> new_centroids(static_cast<std::size_t>(c) * d);

    for (int iter = 0; iter < params_.max_iters; ++iter) {
        ++n_iters_;

        //  1. Squared distances dsq[i*n + j] 
        for (int i = 0; i < c; ++i) {
            const float* ci_row = centroids_.data() + static_cast<std::size_t>(i) * d;
            float*       drow   = dsq.data() + static_cast<std::size_t>(i) * n;
            for (int j = 0; j < n; ++j)
                drow[j] = sq_dist(ci_row, feat + static_cast<std::size_t>(j) * d, d);
        }

        //  2. Update U (fuzzy memberships) 
        for (int j = 0; j < n; ++j) {
            int zero_count = 0;
            for (int i = 0; i < c; ++i)
                if (dsq[static_cast<std::size_t>(i) * n + j] == 0.f) ++zero_count;

            if (zero_count > 0) {
                float share = 1.f / static_cast<float>(zero_count);
                for (int i = 0; i < c; ++i) {
                    float v = dsq[static_cast<std::size_t>(i) * n + j];
                    U_[static_cast<std::size_t>(i) * n + j] = (v == 0.f) ? share : 0.f;
                }
            } else {
                for (int i = 0; i < c; ++i) {
                    float denom = 0.f;
                    float dij = dsq[static_cast<std::size_t>(i) * n + j];
                    for (int k = 0; k < c; ++k)
                        denom += std::pow(dij / dsq[static_cast<std::size_t>(k) * n + j], exp_um);
                    U_[static_cast<std::size_t>(i) * n + j] = 1.f / denom;
                }
            }
        }

        //  3. Compute γ_i once (after first U update) 
        if (iter == 0) {
            for (int i = 0; i < c; ++i) {
                float num = 0.f, den = 0.f;
                for (int j = 0; j < n; ++j) {
                    float um = std::pow(U_[static_cast<std::size_t>(i) * n + j], m);
                    num += um * dsq[static_cast<std::size_t>(i) * n + j];
                    den += um;
                }
                gamma[i] = (den > 0.f && num > 0.f) ? (num / den) : 1.f;
            }
        }

        //  4. Update T (possibilistic typicalities) 
        for (int i = 0; i < c; ++i) {
            float g = gamma[i];
            for (int j = 0; j < n; ++j)
                T_[static_cast<std::size_t>(i) * n + j] =
                    1.f / (1.f + std::pow(dsq[static_cast<std::size_t>(i) * n + j] / g, exp_um));
        }

        //  5. Update centers 
        std::fill(new_centroids.begin(), new_centroids.end(), 0.f);
        for (int i = 0; i < c; ++i) {
            float* ci_row = new_centroids.data() + static_cast<std::size_t>(i) * d;
            float weight_sum = 0.f;
            for (int j = 0; j < n; ++j) {
                float um = std::pow(U_[static_cast<std::size_t>(i) * n + j], m);
                float tm = std::pow(T_[static_cast<std::size_t>(i) * n + j], m);
                float w  = um + alpha * tm;
                weight_sum += w;
                const float* x = feat + static_cast<std::size_t>(j) * d;
                for (int b = 0; b < d; ++b) ci_row[b] += w * x[b];
            }
            if (weight_sum > 0.f) {
                float inv = 1.f / weight_sum;
                for (int b = 0; b < d; ++b) ci_row[b] *= inv;
            } else {
                copy_row(ci_row, centroids_.data() + static_cast<std::size_t>(i) * d, d);
            }
        }

        //  6. Convergence: max centroid shift 
        float max_shift_sq = 0.f;
        for (int i = 0; i < c; ++i) {
            float s = sq_dist(centroids_.data() + static_cast<std::size_t>(i) * d,
                              new_centroids.data() + static_cast<std::size_t>(i) * d, d);
            if (s > max_shift_sq) max_shift_sq = s;
        }
        centroids_.swap(new_centroids);
        if (max_shift_sq < params_.tol * params_.tol) {
            converged_ = true;
            break;
        }
    }

    //  7. Hard labels: argmax_i (u_ij + α·t_ij) 
    labels_.assign(n, 0);
    for (int j = 0; j < n; ++j) {
        int   best_i     = 0;
        float best_score = U_[j] + alpha * T_[j];
        for (int i = 1; i < c; ++i) {
            float score = U_[static_cast<std::size_t>(i) * n + j]
                        + alpha * T_[static_cast<std::size_t>(i) * n + j];
            if (score > best_score) { best_score = score; best_i = i; }
        }
        labels_[j] = best_i;
    }
}

} // namespace clustering
