#include "clustering_core/kmeans.hpp"
#include "clustering_core/pfcm.hpp"
#include <vector>
#include <iostream>

int main() {
    std::vector<std::vector<float>> data(10, std::vector<float>(2, 1.f));

    KMeans km(3, 100);
    km.fit(data);

    PFCM pfcm(3, 100, 2.f, 0.5f);
    pfcm.fit(data);

    std::cout << "CPU Benchmark done.\n";
    return 0;
}
