#pragma once
#include "clustering_core/clusterer.hpp"
#include "clustering_core/dataset.hpp"
#include <vector>

namespace clustering {

struct PfcmParams {
    int   c         = 0;
    int   max_iters = 100;
    float m         = 2.0f;       // fuzziness exponent (> 1)
    float alpha     = 1.0f;       // possibilistic weight relative to fuzzy
    float tol       = 1e-4f;
    int   seed      = 42;
};

// CPU Possibilistic Fuzzy C-Means.
class PfcmCpu final : public FuzzyClusterer {
public:
    explicit PfcmCpu(PfcmParams params);

    void fit(const Dataset& data) override;

    const std::vector<int>&   labels()       const override { return labels_; }
    const std::vector<float>& centroids()    const override { return centroids_; }
    const std::vector<float>& memberships()  const override { return U_; }
    const std::vector<float>& typicalities() const override { return T_; }
    int  k()         const override { return params_.c; }
    int  n_iters()   const override { return n_iters_; }
    bool converged() const override { return converged_; }

private:
    PfcmParams         params_;
    std::vector<int>   labels_;
    std::vector<float> centroids_;   // row-major [c*d]
    std::vector<float> U_;           // row-major [c*n] — fuzzy memberships
    std::vector<float> T_;           // row-major [c*n] — possibilistic typicalities
    int                n_iters_   = 0;
    bool               converged_ = false;
};

// GPU PFCM (placeholder until kernels land; symbol exists so callers link).
class PfcmGpu final : public FuzzyClusterer {
public:
    explicit PfcmGpu(PfcmParams params);

    void fit(const Dataset& data) override;

    const std::vector<int>&   labels()       const override { return labels_; }
    const std::vector<float>& centroids()    const override { return centroids_; }
    const std::vector<float>& memberships()  const override { return U_; }
    const std::vector<float>& typicalities() const override { return T_; }
    int  k()         const override { return params_.c; }
    int  n_iters()   const override { return n_iters_; }
    bool converged() const override { return converged_; }

private:
    PfcmParams         params_;
    std::vector<int>   labels_;
    std::vector<float> centroids_;
    std::vector<float> U_;
    std::vector<float> T_;
    int                n_iters_   = 0;
    bool               converged_ = false;
};

} // namespace clustering
