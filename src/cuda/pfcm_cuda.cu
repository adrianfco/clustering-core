#include "clustering_core/pfcm.hpp"
#include <cstddef>

namespace clustering {

// Placeholder GPU PFCM — real CUDA kernels land in a follow-up.
// Class exists so callers can link against the symbol and program against
// the FuzzyClusterer interface.

PfcmGpu::PfcmGpu(PfcmParams params) : params_(params) {}

void PfcmGpu::fit(const Dataset& data) {
    labels_.clear();
    centroids_.clear();
    U_.clear();
    T_.clear();
    n_iters_   = 0;
    converged_ = false;

    if (data.n == 0 || data.d == 0) return;

    labels_.assign(data.n, 0);
    centroids_.assign(static_cast<std::size_t>(params_.c) * data.d, 0.f);
    U_.assign(static_cast<std::size_t>(params_.c) * data.n, 0.f);
    T_.assign(static_cast<std::size_t>(params_.c) * data.n, 0.f);
}

} // namespace clustering
