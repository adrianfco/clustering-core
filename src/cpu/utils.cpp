#include "utils.hpp"
#include <iostream>
#include <cmath>

void print_vector_size(size_t n) {
    std::cout << "Vector size: " << n << std::endl;
}

double euclidean_distance(const std::vector<double>& a,
                          const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
