#pragma once
#include "clustering_core/clusterer.hpp"
#include "clustering_core/dataset.hpp"
#include <vector>

namespace clustering {

struct KMeansParams {
    int   k         = 0;
    int   max_iters = 100;
    float tol       = 1e-4f;
    int   seed      = 42;
};

// CPU KMeans (consumes data.as(Layout::AoS) internally).
class KMeansCpu final : public Clusterer {
public:
    explicit KMeansCpu(KMeansParams params);

    void fit(const Dataset& data) override;

    const std::vector<int>&   labels()    const override { return labels_; }
    const std::vector<float>& centroids() const override { return centroids_; }
    int  k()         const override { return params_.k; }
    int  n_iters()   const override { return n_iters_; }
    bool converged() const override { return converged_; }

private:
    KMeansParams       params_;
    std::vector<int>   labels_;
    std::vector<float> centroids_;   // row-major [k*d]
    int                n_iters_   = 0;
    bool               converged_ = false;
};

// GPU KMeans (Pascal sm_61+, consumes data.as(Layout::SoA) internally).
class KMeansGpu final : public Clusterer {
public:
    explicit KMeansGpu(KMeansParams params);

    void fit(const Dataset& data) override;

    const std::vector<int>&   labels()    const override { return labels_; }
    const std::vector<float>& centroids() const override { return centroids_; }
    int  k()         const override { return params_.k; }
    int  n_iters()   const override { return n_iters_; }
    bool converged() const override { return converged_; }

private:
    KMeansParams       params_;
    std::vector<int>   labels_;
    std::vector<float> centroids_;   // row-major [k*d]
    int                n_iters_   = 0;
    bool               converged_ = false;
};

} // namespace clustering
