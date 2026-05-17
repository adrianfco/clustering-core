#pragma once
#include <vector>

class PFCM {
public:
    PFCM(int c, int max_iters, float m, float alpha);
    void fit(const std::vector<std::vector<float>>& data);
    const std::vector<int>& labels() const;

private:
    int c_;
    int max_iters_;
    float m_;
    float alpha_;
    std::vector<int> labels_;
};
