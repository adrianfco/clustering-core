#pragma once
#include "clustering_core/dataset.hpp"
#include <vector>

namespace clustering {

// Common interface for all clustering algorithms (CPU or GPU).
// Implementations are stateful: fit() then call accessors.
class Clusterer {
public:
    virtual ~Clusterer() = default;

    virtual void fit(const Dataset& data) = 0;

    virtual const std::vector<int>&   labels()    const = 0;  // length n
    virtual const std::vector<float>& centroids() const = 0;  // row-major [k*d]

    virtual int  k()         const = 0;
    virtual int  n_iters()   const = 0;   // iterations actually run
    virtual bool converged() const = 0;
};

// Fuzzy clusterers expose membership/typicality matrices in addition.
class FuzzyClusterer : public Clusterer {
public:
    virtual const std::vector<float>& memberships()  const = 0;  // row-major [k*n]
    virtual const std::vector<float>& typicalities() const = 0;  // row-major [k*n]
};

} // namespace clustering
