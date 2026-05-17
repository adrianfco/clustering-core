#pragma once
#include <string>
#include <vector>

class TifLoader {
public:
    // Loads a GeoTIFF file. Each pixel becomes one data point;
    // each spectral band becomes one feature (dimension).
    // Returns: vector of n_pixels vectors of length n_bands.
    // Throws: std::runtime_error on open or read failure.
    static std::vector<std::vector<float>> load(const std::string& path);
};
