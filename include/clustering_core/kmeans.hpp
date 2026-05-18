#pragma once
#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iters, float tol = 1e-4f, int seed = 42);
    void fit(const std::vector<std::vector<float>>& data);
    const std::vector<int>& labels() const;

private:
    int k_;
    int max_iters_;
    float tol_;
    int seed_;
    std::vector<int> labels_;
};
