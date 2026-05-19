#pragma once
#include "clustering_core/dataset.hpp"
#include <string>

namespace clustering {

class TifLoader {
public:
    // Reads each band as a contiguous block (SoA is the zero-copy default).
    // Transposes to AoS on request.
    static Dataset load(const std::string& path, Layout layout = Layout::SoA);
};

} // namespace clustering
