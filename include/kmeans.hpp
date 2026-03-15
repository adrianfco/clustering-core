#pragma once
#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iters, double tol = 1e-4, int seed = 42);
    void fit(const std::vector<std::vector<double>>& data);
    const std::vector<int>& labels() const;

private:
    int k_;
    int max_iters_;
    double tol_;
    int seed_;
    std::vector<int> labels_;
};
