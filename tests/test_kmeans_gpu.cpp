#include <catch2/catch_test_macros.hpp>
#include "clustering_core/kmeans.hpp"
#include "clustering_core/dataset.hpp"
#include <cstddef>
#include <vector>

using clustering::Dataset;
using clustering::KMeansGpu;
using clustering::KMeansParams;
using clustering::Layout;

// SoA-flat Dataset: features[b * n + p]

static Dataset make_uniform_gpu(int n, int B, float val = 1.f) {
    Dataset d;
    d.n = n;
    d.d = B;
    d.layout = Layout::SoA;
    d.features.assign(static_cast<std::size_t>(B) * n, val);
    return d;
}

static Dataset make_separable_gpu(int n, int B, float offset = 100.f) {
    const int N = 2 * n;
    Dataset d;
    d.n = N;
    d.d = B;
    d.layout = Layout::SoA;
    d.features.resize(static_cast<std::size_t>(B) * N);
    for (int b = 0; b < B; ++b) {
        for (int p = 0; p < n; ++p)
            d.features[static_cast<std::size_t>(b) * N + p] =
                static_cast<float>(p % 5) * 0.01f;
        for (int p = 0; p < n; ++p)
            d.features[static_cast<std::size_t>(b) * N + n + p] =
                offset + static_cast<float>(p % 5) * 0.01f;
    }
    return d;
}

TEST_CASE("KMeansGpu label count matches input size", "[gpu]") {
    auto data = make_uniform_gpu(10, 2);
    KMeansGpu km({.k = 3, .max_iters = 100});
    km.fit(data);
    REQUIRE(km.labels().size() == 10);
}

TEST_CASE("KMeansGpu empty data produces empty labels", "[gpu]") {
    Dataset empty;
    KMeansGpu km({.k = 3, .max_iters = 100});
    km.fit(empty);
    REQUIRE(km.labels().empty());
    REQUIRE(km.centroids().empty());
}

TEST_CASE("KMeansGpu reproducibility with same seed", "[gpu]") {
    auto data = make_separable_gpu(20, 2);
    KMeansGpu km1({.k = 3, .max_iters = 200, .tol = 1e-6f, .seed = 42});
    KMeansGpu km2({.k = 3, .max_iters = 200, .tol = 1e-6f, .seed = 42});
    km1.fit(data);
    km2.fit(data);
    REQUIRE(km1.labels() == km2.labels());
}

TEST_CASE("KMeansGpu convergence on clearly separable data", "[gpu]") {
    auto data = make_separable_gpu(50, 2, 100.f);
    KMeansGpu km({.k = 2, .max_iters = 500, .tol = 1e-6f, .seed = 42});
    km.fit(data);

    REQUIRE(km.labels().size() == 100);

    int label_a = km.labels()[0];
    int label_b = km.labels()[50];
    REQUIRE(label_a != label_b);

    for (int i = 0; i < 50; ++i)
        REQUIRE(km.labels()[i] == label_a);
    for (int i = 50; i < 100; ++i)
        REQUIRE(km.labels()[i] == label_b);
}

TEST_CASE("KMeansGpu k=1 assigns all points to cluster 0", "[gpu]") {
    auto data = make_separable_gpu(10, 3);
    KMeansGpu km({.k = 1, .max_iters = 100});
    km.fit(data);

    REQUIRE(km.labels().size() == 20);
    for (int l : km.labels())
        REQUIRE(l == 0);
}

TEST_CASE("KMeansGpu edge case: all identical points", "[gpu]") {
    auto data = make_uniform_gpu(10, 3, 5.f);
    KMeansGpu km({.k = 3, .max_iters = 50, .tol = 1e-4f, .seed = 42});
    km.fit(data);

    REQUIRE(km.labels().size() == 10);
    int first = km.labels()[0];
    for (int l : km.labels())
        REQUIRE(l == first);
}

TEST_CASE("KMeansGpu labels stay within requested k clusters", "[gpu]") {
    auto data = make_separable_gpu(30, 4);
    const int k = 3;
    KMeansGpu km({.k = k, .max_iters = 100, .tol = 1e-4f, .seed = 42});
    km.fit(data);

    REQUIRE(km.labels().size() == 60);
    for (int l : km.labels()) {
        REQUIRE(l >= 0);
        REQUIRE(l < k);
    }
}

TEST_CASE("KMeansGpu accepts AoS input via internal conversion", "[gpu]") {
    auto soa = make_separable_gpu(20, 2);
    Dataset aos = soa.as(Layout::AoS);

    KMeansGpu km_soa({.k = 2, .max_iters = 200, .tol = 1e-6f, .seed = 42});
    KMeansGpu km_aos({.k = 2, .max_iters = 200, .tol = 1e-6f, .seed = 42});
    km_soa.fit(soa);
    km_aos.fit(aos);

    REQUIRE(km_soa.labels() == km_aos.labels());
}
