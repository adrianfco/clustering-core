#pragma once
#include <vector>

class PFCM {
public:
    // m: fuzziness exponent (must be > 1, typically 2.0)
    // alpha: possibilistic weight relative to fuzzy (b in Pal et al. 2005)
    PFCM(int c, int max_iters, double m, double alpha,
         double tol = 1e-4, int seed = 42);

    void fit(const std::vector<std::vector<double>>& data);

    const std::vector<int>&                 labels()       const;
    const std::vector<std::vector<double>>& memberships()  const; // U [c x n]
    const std::vector<std::vector<double>>& typicalities() const; // T [c x n]
    const std::vector<std::vector<double>>& centers()      const; // V [c x dims]

private:
    int    c_;
    int    max_iters_;
    double m_;
    double alpha_;
    double tol_;
    int    seed_;

    std::vector<int>                 labels_;
    std::vector<std::vector<double>> U_;       // fuzzy memberships        [c x n]
    std::vector<std::vector<double>> T_;       // possibilistic typicalities [c x n]
    std::vector<std::vector<double>> centers_; // cluster centers          [c x dims]
    std::vector<double>              gamma_;   // typicality scales        [c]
};
