#include <catch2/catch_test_macros.hpp>
#include "clustering_core/pfcm.hpp"

// ── Helpers ───────────────────────────────────────────────────────────────────

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

TEST_CASE("PFCM label count matches input size") {
    SECTION("small dataset") {
        auto data = make_uniform_data(10, 2);
        PFCM pf(3, 50, 2.0, 1.0);
        pf.fit(data);
        REQUIRE(pf.labels().size() == 10);
    }

    SECTION("single point") {
        auto data = make_uniform_data(1, 2);
        PFCM pf(1, 10, 2.0, 1.0);
        pf.fit(data);
        REQUIRE(pf.labels().size() == 1);
        REQUIRE(pf.labels()[0] == 0);
    }
}

TEST_CASE("PFCM backward-compatible constructor") {
    auto data = make_uniform_data(5, 2);
    PFCM pf(2, 10, 2.0, 1.0);  // four-arg form, tol/seed default
    pf.fit(data);
    REQUIRE(pf.labels().size() == 5);
}

TEST_CASE("PFCM reproducibility with same seed") {
    auto data = make_separable_data(20, 2);
    PFCM pf1(3, 200, 2.0, 1.0, 1e-6, 42);
    PFCM pf2(3, 200, 2.0, 1.0, 1e-6, 42);
    pf1.fit(data);
    pf2.fit(data);
    REQUIRE(pf1.labels() == pf2.labels());
}

TEST_CASE("PFCM convergence on clearly separable data") {
    // 50 points near (0,0), 50 points near (100,100)
    auto data = make_separable_data(50, 2, 100.0);
    PFCM pf(2, 500, 2.0, 1.0, 1e-6, 42);
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
    auto data = make_uniform_data(10, 3, 5.0);
    PFCM pf(3, 50, 2.0, 1.0, 1e-4, 42);
    pf.fit(data);

    REQUIRE(pf.labels().size() == 10);
    int first = pf.labels()[0];
    for (int label : pf.labels())
        REQUIRE(label == first);
}

TEST_CASE("PFCM c=1 assigns all points to cluster 0") {
    auto data = make_separable_data(10, 3);
    PFCM pf(1, 100, 2.0, 1.0);
    pf.fit(data);

    REQUIRE(pf.labels().size() == 20);
    for (int label : pf.labels())
        REQUIRE(label == 0);
}

TEST_CASE("PFCM empty data produces empty labels") {
    std::vector<std::vector<double>> data;
    PFCM pf(3, 100, 2.0, 1.0);
    pf.fit(data);
    REQUIRE(pf.labels().empty());
}

TEST_CASE("PFCM membership matrix shape and FCM row-sum constraint") {
    auto data = make_separable_data(10, 2);  // 20 points
    PFCM pf(2, 100, 2.0, 1.0, 1e-6, 42);
    pf.fit(data);

    const auto& U = pf.memberships();
    REQUIRE(U.size() == 2);
    REQUIRE(U[0].size() == 20);

    // FCM constraint: for each point j, Σ_i u_ij == 1
    for (int j = 0; j < 20; ++j) {
        double sum = 0.0;
        for (int i = 0; i < 2; ++i)
            sum += U[i][j];
        REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-9));
    }
}

TEST_CASE("PFCM typicality matrix shape and values in [0, 1]") {
    auto data = make_separable_data(10, 2);  // 20 points
    PFCM pf(2, 100, 2.0, 1.0, 1e-6, 42);
    pf.fit(data);

    const auto& T = pf.typicalities();
    REQUIRE(T.size() == 2);
    REQUIRE(T[0].size() == 20);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 20; ++j) {
            REQUIRE(T[i][j] >= 0.0);
            REQUIRE(T[i][j] <= 1.0);
        }
}

TEST_CASE("PFCM centers matrix shape") {
    auto data = make_separable_data(10, 2);  // 20 points, 2 dims
    PFCM pf(3, 50, 2.0, 1.0, 1e-4, 42);
    pf.fit(data);

    const auto& V = pf.centers();
    REQUIRE(V.size() == 3);
    REQUIRE(V[0].size() == 2);
}
