#include "clustering_core/kmeans.hpp"
#include "clustering_core/pfcm.hpp"
#include "clustering_core/dataset.hpp"
#include <iostream>

int main() {
    clustering::Dataset data;
    data.n = 10;
    data.d = 2;
    data.layout = clustering::Layout::AoS;
    data.features.assign(static_cast<std::size_t>(data.n) * data.d, 1.f);

    clustering::KMeansCpu km({.k = 3, .max_iters = 100});
    km.fit(data);

    clustering::PfcmCpu pf({.c = 3, .max_iters = 100, .m = 2.f, .alpha = 0.5f});
    pf.fit(data);

    std::cout << "CPU Benchmark done.\n";
    return 0;
}
