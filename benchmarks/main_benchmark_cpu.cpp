#include "clustering_core/kmeans.hpp"
#include "clustering_core/pfcm.hpp"
#include <vector>
#include <iostream>

int main() {
    std::vector<std::vector<double>> data(10, std::vector<double>(2, 1.0));

    KMeans km(3, 100);
    km.fit(data);

    PFCM pfcm(3, 100, 2.0, 0.5);
    pfcm.fit(data);

    std::cout << "CPU Benchmark done.\n";
    return 0;
}
