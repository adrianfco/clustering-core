#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "clustering_core/dataset.hpp"
#include <cstddef>

using Catch::Matchers::WithinAbs;
using clustering::Dataset;
using clustering::Layout;

TEST_CASE("Dataset::at returns identical values across layouts") {
    // 3 points × 2 bands, distinct value per (p,b)
    //   point 0: (1, 10)
    //   point 1: (2, 20)
    //   point 2: (3, 30)
    Dataset aos;
    aos.n = 3;
    aos.d = 2;
    aos.layout = Layout::AoS;
    aos.features = {1.f, 10.f, 2.f, 20.f, 3.f, 30.f};

    Dataset soa;
    soa.n = 3;
    soa.d = 2;
    soa.layout = Layout::SoA;
    soa.features = {1.f, 2.f, 3.f, 10.f, 20.f, 30.f};

    for (int p = 0; p < 3; ++p)
        for (int b = 0; b < 2; ++b)
            REQUIRE_THAT(aos.at(p, b), WithinAbs(soa.at(p, b), 1e-9));
}

TEST_CASE("Dataset::as is no-op when layout matches") {
    Dataset aos;
    aos.n = 2;
    aos.d = 3;
    aos.layout = Layout::AoS;
    aos.features = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

    Dataset out = aos.as(Layout::AoS);
    REQUIRE(out.layout == Layout::AoS);
    REQUIRE(out.features == aos.features);
}

TEST_CASE("Dataset::as round-trip AoS → SoA → AoS preserves values") {
    Dataset aos;
    aos.n = 4;
    aos.d = 3;
    aos.layout = Layout::AoS;
    aos.features = {
        1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,
        7.f,  8.f,  9.f,
       10.f, 11.f, 12.f,
    };

    Dataset soa = aos.as(Layout::SoA);
    REQUIRE(soa.layout == Layout::SoA);
    REQUIRE(soa.n == aos.n);
    REQUIRE(soa.d == aos.d);

    // SoA layout means band-major: b0 = [1,4,7,10]; b1 = [2,5,8,11]; b2 = [3,6,9,12]
    REQUIRE(soa.features[0] == 1.f);
    REQUIRE(soa.features[4] == 2.f);
    REQUIRE(soa.features[8] == 3.f);

    Dataset back = soa.as(Layout::AoS);
    REQUIRE(back.features == aos.features);
}

TEST_CASE("Dataset::as SoA → AoS produces correct transpose") {
    Dataset soa;
    soa.n = 3;
    soa.d = 2;
    soa.layout = Layout::SoA;
    soa.features = {1.f, 2.f, 3.f, 10.f, 20.f, 30.f};

    Dataset aos = soa.as(Layout::AoS);
    REQUIRE(aos.layout == Layout::AoS);
    REQUIRE(aos.features == std::vector<float>{1.f, 10.f, 2.f, 20.f, 3.f, 30.f});
}
