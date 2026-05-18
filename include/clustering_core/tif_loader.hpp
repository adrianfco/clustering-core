#pragma once
#include <string>
#include <vector>

// Flat float array in band-major (SoA) layout for GPU consumption.
// Pixel p in band b lives at pixels[b * n_pixels + p].
struct GpuData {
    std::vector<float> pixels;
    int n_pixels = 0;
    int n_bands  = 0;
};

class TifLoader {
public:
    // Returns AoS float layout (CPU path): result[pixel][band].
    static std::vector<std::vector<float>> load(const std::string& path);

    // Returns SoA float layout (GPU path): pixels[band * n_pixels + pixel].
    // Reads each band as a contiguous block — no scatter, no conversion loop.
    static GpuData load_gpu(const std::string& path);
};
