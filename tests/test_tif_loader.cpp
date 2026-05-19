#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "clustering_core/tif_loader.hpp"
#include "clustering_core/kmeans.hpp"
#include "clustering_core/dataset.hpp"
#include <gdal_priv.h>
#include <cpl_vsi.h>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using clustering::Dataset;
using clustering::KMeansCpu;
using clustering::Layout;
using clustering::TifLoader;

static std::string create_vsimem_tif(const std::string& name,
                                      int width, int height, int n_bands,
                                      const std::vector<std::vector<float>>& band_data) {
    GDALAllRegister();
    const std::string path = "/vsimem/" + name;
    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    REQUIRE(drv != nullptr);

    GDALDataset* ds = drv->Create(path.c_str(), width, height, n_bands, GDT_Float32, nullptr);
    REQUIRE(ds != nullptr);

    for (int b = 0; b < n_bands; ++b) {
        GDALRasterBand* band = ds->GetRasterBand(b + 1);
        REQUIRE(band != nullptr);
        std::vector<float> buf = band_data[b];
        CPLErr err = band->RasterIO(GF_Write, 0, 0, width, height,
                                    buf.data(), width, height, GDT_Float32, 0, 0);
        REQUIRE(err == CE_None);
    }
    GDALClose(ds);
    return path;
}

TEST_CASE("TifLoader loads correct pixel count and band count") {
    const int w = 4, h = 3, n_pixels = 12;
    std::vector<std::vector<float>> bands = {
        std::vector<float>(n_pixels, 1.0f),
        std::vector<float>(n_pixels, 2.0f)
    };
    auto path = create_vsimem_tif("test_count.tif", w, h, 2, bands);

    auto data = TifLoader::load(path);

    REQUIRE(data.n == n_pixels);
    REQUIRE(data.d == 2);

    for (int p = 0; p < n_pixels; ++p) {
        REQUIRE_THAT(data.at(p, 0), WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(data.at(p, 1), WithinAbs(2.0, 1e-6));
    }

    VSIUnlink(path.c_str());
}

TEST_CASE("TifLoader preserves per-pixel band values") {
    // 2x2 raster, 3 bands with known distinct values per pixel
    std::vector<std::vector<float>> bands = {
        {10.f, 11.f, 12.f, 13.f},
        {20.f, 21.f, 22.f, 23.f},
        {30.f, 31.f, 32.f, 33.f}
    };
    auto path = create_vsimem_tif("test_values.tif", 2, 2, 3, bands);

    auto data = TifLoader::load(path);

    REQUIRE(data.n == 4);
    REQUIRE(data.d == 3);

    for (int p = 0; p < 4; ++p) {
        REQUIRE_THAT(data.at(p, 0), WithinAbs(10.0 + p, 1e-6));
        REQUIRE_THAT(data.at(p, 1), WithinAbs(20.0 + p, 1e-6));
        REQUIRE_THAT(data.at(p, 2), WithinAbs(30.0 + p, 1e-6));
    }

    VSIUnlink(path.c_str());
}

TEST_CASE("TifLoader returns requested layout") {
    std::vector<std::vector<float>> bands = {
        {1.f, 2.f, 3.f, 4.f},
        {10.f, 20.f, 30.f, 40.f}
    };
    auto path = create_vsimem_tif("test_layout.tif", 2, 2, 2, bands);

    auto soa = TifLoader::load(path, Layout::SoA);
    auto aos = TifLoader::load(path, Layout::AoS);

    REQUIRE(soa.layout == Layout::SoA);
    REQUIRE(aos.layout == Layout::AoS);
    REQUIRE(soa.n == aos.n);
    REQUIRE(soa.d == aos.d);

    // Both layouts must yield identical values via at()
    for (int p = 0; p < soa.n; ++p)
        for (int b = 0; b < soa.d; ++b)
            REQUIRE_THAT(soa.at(p, b), WithinAbs(aos.at(p, b), 1e-6));

    VSIUnlink(path.c_str());
}

TEST_CASE("TifLoader throws on nonexistent file") {
    REQUIRE_THROWS_AS(
        TifLoader::load("/nonexistent/path/fake.tif"),
        std::runtime_error
    );
}

TEST_CASE("TifLoader output is compatible with KMeansCpu") {
    const int n_pixels = 6;
    std::vector<std::vector<float>> bands = {
        {1.f, 2.f, 3.f, 100.f, 101.f, 102.f},
        {1.f, 2.f, 3.f, 100.f, 101.f, 102.f}
    };
    auto path = create_vsimem_tif("test_compat.tif", 3, 2, 2, bands);

    auto data = TifLoader::load(path);
    REQUIRE(data.n == n_pixels);

    KMeansCpu km({.k = 2, .max_iters = 200, .tol = 1e-6f, .seed = 42});
    km.fit(data);

    REQUIRE(km.labels().size() == static_cast<std::size_t>(n_pixels));

    VSIUnlink(path.c_str());
}
