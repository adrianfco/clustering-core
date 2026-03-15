#include <catch2/catch_test_macros.hpp>
#include "clustering_core/kmeans.hpp"

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<std::vector<double>> make_uniform_data(int n, int dims, double val = 1.0) {
    return std::vector<std::vector<double>>(n, std::vector<double>(dims, val));
}

// Two tight clusters: n points near origin, n points near (offset, offset, ...)
static std::vector<std::vector<double>> make_separable_data(int n, int dims, double offset = 100.0) {
    std::vector<std::vector<double>> data;
    data.reserve(2 * n);
    for (int i = 0; i < n; ++i)
        data.push_back(std::vector<double>(dims, static_cast<double>(i % 5) * 0.01));
    for (int i = 0; i < n; ++i)
        data.push_back(std::vector<double>(dims, offset + static_cast<double>(i % 5) * 0.01));
    return data;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("KMeans label count matches input size") {
    SECTION("small dataset") {
        auto data = make_uniform_data(10, 2);
        KMeans km(3, 100);
        km.fit(data);
        REQUIRE(km.labels().size() == 10);
    }

    SECTION("single point") {
        auto data = make_uniform_data(1, 2);
        KMeans km(1, 10);
        km.fit(data);
        REQUIRE(km.labels().size() == 1);
        REQUIRE(km.labels()[0] == 0);
    }
}

TEST_CASE("KMeans backward-compatible constructor") {
    auto data = make_uniform_data(5, 2);
    KMeans km(2, 10);   // old call site — no tol/seed
    km.fit(data);
    REQUIRE(km.labels().size() == 5);
}

TEST_CASE("KMeans reproducibility with same seed") {
    auto data = make_separable_data(20, 2);
    KMeans km1(3, 200, 1e-6, 42);
    KMeans km2(3, 200, 1e-6, 42);
    km1.fit(data);
    km2.fit(data);
    REQUIRE(km1.labels() == km2.labels());
}

TEST_CASE("KMeans convergence on clearly separable data") {
    // 50 points near (0,0), 50 points near (100,100)
    auto data = make_separable_data(50, 2, 100.0);
    KMeans km(2, 500, 1e-6, 42);
    km.fit(data);

    REQUIRE(km.labels().size() == 100);

    // All first 50 must share the same label; all last 50 must share a different label
    int label_a = km.labels()[0];
    int label_b = km.labels()[50];
    REQUIRE(label_a != label_b);

    for (int i = 0; i < 50; ++i)
        REQUIRE(km.labels()[i] == label_a);
    for (int i = 50; i < 100; ++i)
        REQUIRE(km.labels()[i] == label_b);
}

TEST_CASE("KMeans edge case: all identical points") {
    auto data = make_uniform_data(10, 3, 5.0);
    KMeans km(3, 50, 1e-4, 42);
    km.fit(data);

    REQUIRE(km.labels().size() == 10);
    // All identical points must end up in the same cluster
    int first = km.labels()[0];
    for (int label : km.labels())
        REQUIRE(label == first);
}

TEST_CASE("KMeans k=1 assigns all points to cluster 0") {
    auto data = make_separable_data(10, 3);
    KMeans km(1, 100);
    km.fit(data);

    REQUIRE(km.labels().size() == 20);
    for (int label : km.labels())
        REQUIRE(label == 0);
}

TEST_CASE("KMeans empty data produces empty labels") {
    std::vector<std::vector<double>> data;
    KMeans km(3, 100);
    km.fit(data);
    REQUIRE(km.labels().empty());
}
