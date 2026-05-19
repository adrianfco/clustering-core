#pragma once
#include <cstddef>
#include <vector>

namespace clustering {

enum class Layout { AoS, SoA };

// Flat feature buffer with explicit layout.
//   AoS: features[p * d + b]   — pixel-major, all bands of pixel p contiguous
//   SoA: features[b * n + p]   — band-major,  all pixels of band b contiguous
//
// Backends pick their preferred layout via data.as(layout); conversion is a
// no-op when the dataset is already in the requested layout.
struct Dataset {
    std::vector<float> features;
    int n = 0;                       // pixel/point count
    int d = 0;                       // band/feature count
    Layout layout = Layout::SoA;

    float at(int point, int band) const {
        return layout == Layout::AoS
            ? features[static_cast<std::size_t>(point) * d + band]
            : features[static_cast<std::size_t>(band)  * n + point];
    }

    // Returns *this if already in target layout, otherwise a transposed copy.
    Dataset as(Layout target) const;
};

} // namespace clustering
