#include "pfcm.hpp"
#include <iostream>

PFCM::PFCM(int c, int max_iters, double m, double alpha)
    : c_(c), max_iters_(max_iters), m_(m), alpha_(alpha) {}

void PFCM::fit(const std::vector<std::vector<double>>& data) {
    std::cout << "PFCM CPU fit() called on " << data.size() << " points\n";
    labels_.resize(data.size(), 0);
}

const std::vector<int>& PFCM::labels() const {
    return labels_;
}
