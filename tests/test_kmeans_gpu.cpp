#include <catch2/catch_test_macros.hpp>
#include "clustering_core/kmeans_gpu.hpp"
#include "clustering_core/tif_loader.hpp"
#include <vector>

// ── Helpers ──────────────────────────────────────────────────────────────────
//
// GpuData is SoA: pixels[b * n_pixels + p].

static GpuData make_uniform_gpu(int n, int B, float val = 1.f) {
    GpuData d;
    d.n_pixels = n;
    d.n_bands  = B;
    d.pixels.assign(static_cast<size_t>(B) * n, val);
    return d;
}

// Two tight clusters: n points near 0, n points near `offset` in every band.
static GpuData make_separable_gpu(int n, int B, float offset = 100.f) {
    const int N = 2 * n;
    GpuData d;
    d.n_pixels = N;
    d.n_bands  = B;
    d.pixels.resize(static_cast<size_t>(B) * N);
    for (int b = 0; b < B; ++b) {
        for (int p = 0; p < n; ++p)
            d.pixels[static_cast<size_t>(b) * N + p] =
                static_cast<float>(p % 5) * 0.01f;
        for (int p = 0; p < n; ++p)
            d.pixels[static_cast<size_t>(b) * N + n + p] =
                offset + static_cast<float>(p % 5) * 0.01f;
    }
    return d;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("kmeans_gpu label count matches input size", "[gpu]") {
    auto data = make_uniform_gpu(10, 2);
    auto labels = kmeans_gpu(data, 3, 100);
    REQUIRE(labels.size() == 10);
}

TEST_CASE("kmeans_gpu empty data produces empty labels", "[gpu]") {
    GpuData empty;
    auto labels = kmeans_gpu(empty, 3, 100);
    REQUIRE(labels.empty());
}

TEST_CASE("kmeans_gpu reproducibility with same seed", "[gpu]") {
    auto data = make_separable_gpu(20, 2);
    auto labels1 = kmeans_gpu(data, 3, 200, 1e-6f, 42);
    auto labels2 = kmeans_gpu(data, 3, 200, 1e-6f, 42);
    REQUIRE(labels1 == labels2);
}

TEST_CASE("kmeans_gpu convergence on clearly separable data", "[gpu]") {
    auto data = make_separable_gpu(50, 2, 100.f);
    auto labels = kmeans_gpu(data, 2, 500, 1e-6f, 42);

    REQUIRE(labels.size() == 100);

    // All first 50 must share one label; all last 50 must share the other
    int label_a = labels[0];
    int label_b = labels[50];
    REQUIRE(label_a != label_b);

    for (int i = 0; i < 50; ++i)
        REQUIRE(labels[i] == label_a);
    for (int i = 50; i < 100; ++i)
        REQUIRE(labels[i] == label_b);
}

TEST_CASE("kmeans_gpu k=1 assigns all points to cluster 0", "[gpu]") {
    auto data = make_separable_gpu(10, 3);
    auto labels = kmeans_gpu(data, 1, 100);

    REQUIRE(labels.size() == 20);
    for (int l : labels)
        REQUIRE(l == 0);
}

TEST_CASE("kmeans_gpu edge case: all identical points", "[gpu]") {
    auto data = make_uniform_gpu(10, 3, 5.f);
    auto labels = kmeans_gpu(data, 3, 50, 1e-4f, 42);

    REQUIRE(labels.size() == 10);
    int first = labels[0];
    for (int l : labels)
        REQUIRE(l == first);
}

TEST_CASE("kmeans_gpu labels stay within requested k clusters", "[gpu]") {
    auto data = make_separable_gpu(30, 4);
    const int k = 3;
    auto labels = kmeans_gpu(data, k, 100, 1e-4f, 42);

    REQUIRE(labels.size() == 60);
    for (int l : labels) {
        REQUIRE(l >= 0);
        REQUIRE(l < k);
    }
}
