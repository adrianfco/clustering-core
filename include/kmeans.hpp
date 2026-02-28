#pragma once
#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iters);
    void fit(const std::vector<std::vector<double>>& data);
    const std::vector<int>& labels() const;

private:
    int k_;
    int max_iters_;
    std::vector<int> labels_;
};
