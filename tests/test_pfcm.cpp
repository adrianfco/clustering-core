#include <catch2/catch_test_macros.hpp>
#include "clustering_core/pfcm.hpp"
#include "clustering_core/dataset.hpp"
#include <cmath>
#include <cstddef>

using clustering::Dataset;
using clustering::Layout;
using clustering::PfcmCpu;
using clustering::PfcmParams;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Dataset make_uniform_dataset(int n, int d, float val = 1.f) {
    Dataset ds;
    ds.n = n;
    ds.d = d;
    ds.layout = Layout::AoS;
    ds.features.assign(static_cast<std::size_t>(n) * d, val);
    return ds;
}

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

TEST_CASE("PFCM label count matches input size") {
    SECTION("small dataset") {
        auto data = make_uniform_dataset(10, 2);
        PfcmCpu pf({.c = 3, .max_iters = 50, .m = 2.0f, .alpha = 1.0f});
        pf.fit(data);
        REQUIRE(pf.labels().size() == 10);
    }

    SECTION("single point") {
        auto data = make_uniform_dataset(1, 2);
        PfcmCpu pf({.c = 1, .max_iters = 10, .m = 2.0f, .alpha = 1.0f});
        pf.fit(data);
        REQUIRE(pf.labels().size() == 1);
        REQUIRE(pf.labels()[0] == 0);
    }
}

TEST_CASE("PFCM default params still work") {
    auto data = make_uniform_dataset(5, 2);
    PfcmCpu pf({.c = 2, .max_iters = 10, .m = 2.0f, .alpha = 1.0f});
    pf.fit(data);
    REQUIRE(pf.labels().size() == 5);
}

TEST_CASE("PFCM reproducibility with same seed") {
    auto data = make_separable_dataset(20, 2);
    PfcmCpu pf1({.c = 3, .max_iters = 200, .m = 2.0f, .alpha = 1.0f, .tol = 1e-6f, .seed = 42});
    PfcmCpu pf2({.c = 3, .max_iters = 200, .m = 2.0f, .alpha = 1.0f, .tol = 1e-6f, .seed = 42});
    pf1.fit(data);
    pf2.fit(data);
    REQUIRE(pf1.labels() == pf2.labels());
}

TEST_CASE("PFCM convergence on clearly separable data") {
    auto data = make_separable_dataset(50, 2, 100.f);
    PfcmCpu pf({.c = 2, .max_iters = 500, .m = 2.0f, .alpha = 1.0f, .tol = 1e-6f, .seed = 42});
    pf.fit(data);

    REQUIRE(pf.labels().size() == 100);

    int label_a = pf.labels()[0];
    int label_b = pf.labels()[50];
    REQUIRE(label_a != label_b);

    for (int i = 0; i < 50; ++i)
        REQUIRE(pf.labels()[i] == label_a);
    for (int i = 50; i < 100; ++i)
        REQUIRE(pf.labels()[i] == label_b);
}

TEST_CASE("PFCM edge case: all identical points") {
    auto data = make_uniform_dataset(10, 3, 5.f);
    PfcmCpu pf({.c = 3, .max_iters = 50, .m = 2.0f, .alpha = 1.0f, .tol = 1e-4f, .seed = 42});
    pf.fit(data);

    REQUIRE(pf.labels().size() == 10);
    int first = pf.labels()[0];
    for (int label : pf.labels())
        REQUIRE(label == first);
}

TEST_CASE("PFCM c=1 assigns all points to cluster 0") {
    auto data = make_separable_dataset(10, 3);
    PfcmCpu pf({.c = 1, .max_iters = 100, .m = 2.0f, .alpha = 1.0f});
    pf.fit(data);

    REQUIRE(pf.labels().size() == 20);
    for (int label : pf.labels())
        REQUIRE(label == 0);
}

TEST_CASE("PFCM empty data produces empty labels") {
    Dataset empty;
    PfcmCpu pf({.c = 3, .max_iters = 100, .m = 2.0f, .alpha = 1.0f});
    pf.fit(empty);
    REQUIRE(pf.labels().empty());
}

TEST_CASE("PFCM membership matrix shape and FCM row-sum constraint") {
    auto data = make_separable_dataset(10, 2);   // 20 points
    PfcmCpu pf({.c = 2, .max_iters = 100, .m = 2.0f, .alpha = 1.0f, .tol = 1e-6f, .seed = 42});
    pf.fit(data);

    const auto& U = pf.memberships();
    const int n = data.n;
    const int c = 2;
    REQUIRE(U.size() == static_cast<std::size_t>(c * n));

    // FCM constraint: for each point j, Σ_i u_ij == 1
    for (int j = 0; j < n; ++j) {
        float sum = 0.f;
        for (int i = 0; i < c; ++i)
            sum += U[static_cast<std::size_t>(i) * n + j];
        REQUIRE(std::abs(sum - 1.0f) < 1e-6f);
    }
}

TEST_CASE("PFCM typicality matrix shape and values in [0, 1]") {
    auto data = make_separable_dataset(10, 2);   // 20 points
    PfcmCpu pf({.c = 2, .max_iters = 100, .m = 2.0f, .alpha = 1.0f, .tol = 1e-6f, .seed = 42});
    pf.fit(data);

    const auto& T = pf.typicalities();
    const int n = data.n;
    const int c = 2;
    REQUIRE(T.size() == static_cast<std::size_t>(c * n));

    for (std::size_t i = 0; i < T.size(); ++i) {
        REQUIRE(T[i] >= 0.0f);
        REQUIRE(T[i] <= 1.0f);
    }
}

TEST_CASE("PFCM centroids matrix shape") {
    auto data = make_separable_dataset(10, 2);   // 20 points, 2 dims
    PfcmCpu pf({.c = 3, .max_iters = 50, .m = 2.0f, .alpha = 1.0f, .tol = 1e-4f, .seed = 42});
    pf.fit(data);

    const auto& V = pf.centroids();
    REQUIRE(V.size() == static_cast<std::size_t>(3 * 2));   // c*d
}
