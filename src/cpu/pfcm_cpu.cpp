#include "clustering_core/pfcm.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

PFCM::PFCM(int c, int max_iters, float m, float alpha, float tol, int seed)
    : c_(c), max_iters_(max_iters), m_(m), alpha_(alpha), tol_(tol), seed_(seed) {}

void PFCM::fit(const std::vector<std::vector<float>>& data) {
    const int n = static_cast<int>(data.size());
    if (n == 0) return;

    const int dims        = static_cast<int>(data[0].size());
    const int effective_c = std::min(c_, n);

    std::mt19937 rng(static_cast<unsigned>(seed_));

    // Max-min init: pick first center randomly, then each subsequent center as
    // the point farthest from all already-selected centers. This guarantees
    // spread and avoids the degenerate case where soft-membership FCM/PFCM
    // collapses when two centers start at the same location.
    centers_.assign(effective_c, std::vector<float>(dims));
    std::uniform_int_distribution<int> dist(0, n - 1);
    centers_[0] = data[dist(rng)];
    for (int ci = 1; ci < effective_c; ++ci) {
        int best_j = 0;
        float best_dist = -1.0;
        for (int j = 0; j < n; ++j) {
            float min_d = std::numeric_limits<float>::max();
            for (int k = 0; k < ci; ++k) {
                float d = euclidean_distance(centers_[k], data[j]);
                if (d < min_d) min_d = d;
            }
            if (min_d > best_dist) { best_dist = min_d; best_j = j; }
        }
        centers_[ci] = data[best_j];
    }

    U_.assign(effective_c, std::vector<float>(n, 0.0));
    T_.assign(effective_c, std::vector<float>(n, 0.0));
    gamma_.assign(effective_c, 1.0);

    const float exp_um = 1.0 / (m_ - 1.0); // exponent for U and T updates

    for (int iter = 0; iter < max_iters_; ++iter) {

        // ── 1. Compute squared distances d²[i][j] ────────────────────────────
        std::vector<std::vector<float>> dsq(effective_c, std::vector<float>(n));
        for (int i = 0; i < effective_c; ++i)
            for (int j = 0; j < n; ++j) {
                float d = euclidean_distance(centers_[i], data[j]);
                dsq[i][j] = d * d;
            }

        // ── 2. Update U (fuzzy memberships) ──────────────────────────────────
        // u_ij = [Σ_k (d²_ij / d²_kj)^(1/(m-1))]^(-1)
        for (int j = 0; j < n; ++j) {
            // Zero-distance guard: point coincides with one or more centers
            int zero_count = 0;
            for (int i = 0; i < effective_c; ++i)
                if (dsq[i][j] == 0.0) ++zero_count;

            if (zero_count > 0) {
                float share = 1.0 / zero_count;
                for (int i = 0; i < effective_c; ++i)
                    U_[i][j] = (dsq[i][j] == 0.0) ? share : 0.0;
            } else {
                for (int i = 0; i < effective_c; ++i) {
                    float denom = 0.0;
                    for (int k = 0; k < effective_c; ++k)
                        denom += std::pow(dsq[i][j] / dsq[k][j], exp_um);
                    U_[i][j] = 1.0 / denom;
                }
            }
        }

        // ── 3. Compute γ_i once (after first U update) ───────────────────────
        // γ_i = Σ_j u_ij^m · d²_ij / Σ_j u_ij^m
        if (iter == 0) {
            for (int i = 0; i < effective_c; ++i) {
                float num = 0.0, den = 0.0;
                for (int j = 0; j < n; ++j) {
                    float um = std::pow(U_[i][j], m_);
                    num += um * dsq[i][j];
                    den += um;
                }
                gamma_[i] = (den > 0.0 && num > 0.0) ? (num / den) : 1.0;
            }
        }

        // ── 4. Update T (possibilistic typicalities) ──────────────────────────
        // t_ij = [1 + (d²_ij / γ_i)^(1/(m-1))]^(-1)
        for (int i = 0; i < effective_c; ++i)
            for (int j = 0; j < n; ++j)
                T_[i][j] = 1.0 / (1.0 + std::pow(dsq[i][j] / gamma_[i], exp_um));

        // ── 5. Update centers ─────────────────────────────────────────────────
        // v_i = Σ_j (u_ij^m + α·t_ij^m)·x_j / Σ_j (u_ij^m + α·t_ij^m)
        std::vector<std::vector<float>> new_centers(effective_c,
                                                      std::vector<float>(dims, 0.0));
        for (int i = 0; i < effective_c; ++i) {
            float weight_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                float w = std::pow(U_[i][j], m_) + alpha_ * std::pow(T_[i][j], m_);
                weight_sum += w;
                for (int d = 0; d < dims; ++d)
                    new_centers[i][d] += w * data[j][d];
            }
            if (weight_sum > 0.0)
                for (int d = 0; d < dims; ++d)
                    new_centers[i][d] /= weight_sum;
            else
                new_centers[i] = centers_[i]; // degenerate: keep old center
        }

        // ── 6. Convergence check: max centroid shift ──────────────────────────
        float max_shift = 0.0;
        for (int i = 0; i < effective_c; ++i) {
            float shift = euclidean_distance(centers_[i], new_centers[i]);
            if (shift > max_shift) max_shift = shift;
        }
        centers_ = std::move(new_centers);
        if (max_shift < tol_) break;
    }

    // ── 7. Assign hard labels: argmax_i (u_ij + α·t_ij) ─────────────────────
    labels_.resize(n);
    for (int j = 0; j < n; ++j) {
        int    best_i     = 0;
        float best_score = U_[0][j] + alpha_ * T_[0][j];
        for (int i = 1; i < effective_c; ++i) {
            float score = U_[i][j] + alpha_ * T_[i][j];
            if (score > best_score) { best_score = score; best_i = i; }
        }
        labels_[j] = best_i;
    }
}

const std::vector<int>&                 PFCM::labels()       const { return labels_; }
const std::vector<std::vector<float>>& PFCM::memberships()  const { return U_; }
const std::vector<std::vector<float>>& PFCM::typicalities() const { return T_; }
const std::vector<std::vector<float>>& PFCM::centers()      const { return centers_; }
