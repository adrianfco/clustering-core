#include "clustering_core/dataset.hpp"
#include <cstddef>

namespace clustering {

Dataset Dataset::as(Layout target) const {
    if (target == layout) return *this;

    Dataset out;
    out.n      = n;
    out.d      = d;
    out.layout = target;
    out.features.resize(features.size());

    const std::size_t N = static_cast<std::size_t>(n);
    const std::size_t D = static_cast<std::size_t>(d);

    if (layout == Layout::AoS && target == Layout::SoA) {
        // src[p*d + b]  →  dst[b*n + p]
        for (std::size_t p = 0; p < N; ++p)
            for (std::size_t b = 0; b < D; ++b)
                out.features[b * N + p] = features[p * D + b];
    } else {
        // src[b*n + p]  →  dst[p*d + b]
        for (std::size_t b = 0; b < D; ++b)
            for (std::size_t p = 0; p < N; ++p)
                out.features[p * D + b] = features[b * N + p];
    }
    return out;
}

} // namespace clustering
