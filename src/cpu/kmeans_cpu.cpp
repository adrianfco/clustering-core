#include "kmeans.hpp"
#include <iostream>

KMeans::KMeans(int k, int max_iters) : k_(k), max_iters_(max_iters) {}

void KMeans::fit(const std::vector<std::vector<double>>& data) {
    std::cout << "KMeans CPU fit() called on " << data.size() << " points\n";
    labels_.resize(data.size(), 0);
}

const std::vector<int>& KMeans::labels() const {
    return labels_;
}
