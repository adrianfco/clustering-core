#pragma once
#include <vector>

class PFCM {
public:
    PFCM(int c, int max_iters, double m, double alpha);
    void fit(const std::vector<std::vector<double>>& data);
    const std::vector<int>& labels() const;

private:
    int c_;
    int max_iters_;
    double m_;
    double alpha_;
    std::vector<int> labels_;
};
