#include <catch2/catch_test_macros.hpp>
#include "clustering_core/kmeans.hpp"
#include "clustering_core/dataset.hpp"

using clustering::Dataset;
using clustering::KMeansCpu;
using clustering::KMeansParams;
using clustering::Layout;

// ── Helpers ──────────────────────────────────────────────────────────────────

// AoS-flat Dataset of n points × d dims, all equal to val.
static Dataset make_uniform_dataset(int n, int d, float val = 1.f) {
    Dataset ds;
    ds.n = n;
    ds.d = d;
    ds.layout = Layout::AoS;
    ds.features.assign(static_cast<std::size_t>(n) * d, val);
    return ds;
}

// Two tight clusters: n points near 0, n points near offset.
static Dataset make_separable_dataset(int n, int d, float offset = 100.f) {
    Dataset ds;
    ds.n = 2 * n;
    ds.d = d;
    ds.layout = Layout::AoS;
    ds.features.resize(static_cast<std::size_t>(ds.n) * d);
    for (int p = 0; p < n; ++p) {
        float v = static_cast<float>(p % 5) * 0.01f;
        for (int b = 0; b < d; ++b)
            ds.features[static_cast<std::size_t>(p) * d + b] = v;
    }
    for (int p = 0; p < n; ++p) {
        float v = offset + static_cast<float>(p % 5) * 0.01f;
        for (int b = 0; b < d; ++b)
            ds.features[static_cast<std::size_t>(n + p) * d + b] = v;
    }
    return ds;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("KMeans label count matches input size") {
    SECTION("small dataset") {
        auto data = make_uniform_dataset(10, 2);
        KMeansCpu km({.k = 3, .max_iters = 100});
        km.fit(data);
        REQUIRE(km.labels().size() == 10);
    }

    SECTION("single point") {
        auto data = make_uniform_dataset(1, 2);
        KMeansCpu km({.k = 1, .max_iters = 10});
        km.fit(data);
        REQUIRE(km.labels().size() == 1);
        REQUIRE(km.labels()[0] == 0);
    }
}

TEST_CASE("KMeans default params still work") {
    auto data = make_uniform_dataset(5, 2);
    KMeansCpu km({.k = 2, .max_iters = 10});
    km.fit(data);
    REQUIRE(km.labels().size() == 5);
}

TEST_CASE("KMeans reproducibility with same seed") {
    auto data = make_separable_dataset(20, 2);
    KMeansCpu km1({.k = 3, .max_iters = 200, .tol = 1e-6f, .seed = 42});
    KMeansCpu km2({.k = 3, .max_iters = 200, .tol = 1e-6f, .seed = 42});
    km1.fit(data);
    km2.fit(data);
    REQUIRE(km1.labels() == km2.labels());
}

TEST_CASE("KMeans convergence on clearly separable data") {
    auto data = make_separable_dataset(50, 2, 100.f);
    KMeansCpu km({.k = 2, .max_iters = 500, .tol = 1e-6f, .seed = 42});
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

TEST_CASE("KMeans edge case: all identical points") {
    auto data = make_uniform_dataset(10, 3, 5.f);
    KMeansCpu km({.k = 3, .max_iters = 50, .tol = 1e-4f, .seed = 42});
    km.fit(data);

    REQUIRE(km.labels().size() == 10);
    int first = km.labels()[0];
    for (int label : km.labels())
        REQUIRE(label == first);
}

TEST_CASE("KMeans k=1 assigns all points to cluster 0") {
    auto data = make_separable_dataset(10, 3);
    KMeansCpu km({.k = 1, .max_iters = 100});
    km.fit(data);

    REQUIRE(km.labels().size() == 20);
    for (int label : km.labels())
        REQUIRE(label == 0);
}

TEST_CASE("KMeans empty data produces empty labels") {
    Dataset empty;
    KMeansCpu km({.k = 3, .max_iters = 100});
    km.fit(empty);
    REQUIRE(km.labels().empty());
    REQUIRE(km.centroids().empty());
}

TEST_CASE("KMeans exposes centroids after fit") {
    auto data = make_separable_dataset(50, 2, 100.f);
    KMeansCpu km({.k = 2, .max_iters = 500, .tol = 1e-6f, .seed = 42});
    km.fit(data);

    const auto& c = km.centroids();
    REQUIRE(c.size() == static_cast<std::size_t>(2 * 2));   // k*d
    REQUIRE(km.converged());
}
