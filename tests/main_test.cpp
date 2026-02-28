#include "../include/kmeans.hpp"
#include "../include/pfcm.hpp"
#include <cassert>
#include <vector>

int main() {
    std::vector<std::vector<double>> data(5, std::vector<double>(2, 1.0));

    KMeans km(2, 10);
    km.fit(data);
    assert(km.labels().size() == data.size());

    PFCM pfcm(2, 10, 2.0, 0.5);
    pfcm.fit(data);
    assert(pfcm.labels().size() == data.size());

    return 0;
}
