#pragma once
#include <vector>

class PFCM {
public:
    // m: fuzziness exponent (must be > 1, typically 2.0)
    // alpha: possibilistic weight relative to fuzzy (b in Pal et al. 2005)
    PFCM(int c, int max_iters, float m, float alpha,
         float tol = 1e-4, int seed = 42);

    void fit(const std::vector<std::vector<float>>& data);

    const std::vector<int>&                 labels()       const;
    const std::vector<std::vector<float>>& memberships()  const; // U [c x n]
    const std::vector<std::vector<float>>& typicalities() const; // T [c x n]
    const std::vector<std::vector<float>>& centers()      const; // V [c x dims]

private:
    int    c_;
    int    max_iters_;
    float m_;
    float alpha_;
    float tol_;
    int    seed_;

    std::vector<int>                 labels_;
    std::vector<std::vector<float>> U_;       // fuzzy memberships        [c x n]
    std::vector<std::vector<float>> T_;       // possibilistic typicalities [c x n]
    std::vector<std::vector<float>> centers_; // cluster centers          [c x dims]
    std::vector<float>              gamma_;   // typicality scales        [c]
};
