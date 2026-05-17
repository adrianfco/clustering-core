#include "utils.hpp"
#include <iostream>
#include <cmath>

void print_vector_size(size_t n) {
    std::cout << "Vector size: " << n << std::endl;
}

float euclidean_distance(const std::vector<float>& a,
                         const std::vector<float>& b) {
    float sum = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
