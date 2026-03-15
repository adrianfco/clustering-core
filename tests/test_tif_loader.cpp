#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tif_loader.hpp"
#include "kmeans.hpp"
#include <gdal_priv.h>
#include <cpl_vsi.h>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;

// Creates a GeoTIFF in GDAL's in-memory filesystem (/vsimem/).
// Returns the vsimem path to pass to TifLoader::load().
// Caller must VSIUnlink(path) after use.
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

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("TifLoader loads correct pixel count and band count") {
    // 4x3 raster, 2 bands: band1 = all 1.0, band2 = all 2.0
    const int w = 4, h = 3, n_pixels = 12;
    std::vector<std::vector<float>> bands = {
        std::vector<float>(n_pixels, 1.0f),
        std::vector<float>(n_pixels, 2.0f)
    };
    auto path = create_vsimem_tif("test_count.tif", w, h, 2, bands);

    auto result = TifLoader::load(path);

    REQUIRE(result.size() == static_cast<std::size_t>(n_pixels));
    REQUIRE(result[0].size() == 2);

    for (const auto& point : result) {
        REQUIRE_THAT(point[0], WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(point[1], WithinAbs(2.0, 1e-6));
    }

    VSIUnlink(path.c_str());
}

TEST_CASE("TifLoader preserves per-pixel band values") {
    // 2x2 raster, 3 bands with known distinct values per pixel
    // Pixels layout (row-major): p0=(0,0), p1=(1,0), p2=(0,1), p3=(1,1)
    // band1: 10, 11, 12, 13
    // band2: 20, 21, 22, 23
    // band3: 30, 31, 32, 33
    std::vector<std::vector<float>> bands = {
        {10.f, 11.f, 12.f, 13.f},
        {20.f, 21.f, 22.f, 23.f},
        {30.f, 31.f, 32.f, 33.f}
    };
    auto path = create_vsimem_tif("test_values.tif", 2, 2, 3, bands);

    auto result = TifLoader::load(path);

    REQUIRE(result.size() == 4);
    REQUIRE(result[0].size() == 3);

    for (int p = 0; p < 4; ++p) {
        REQUIRE_THAT(result[p][0], WithinAbs(10.0 + p, 1e-6));
        REQUIRE_THAT(result[p][1], WithinAbs(20.0 + p, 1e-6));
        REQUIRE_THAT(result[p][2], WithinAbs(30.0 + p, 1e-6));
    }

    VSIUnlink(path.c_str());
}

TEST_CASE("TifLoader throws on nonexistent file") {
    REQUIRE_THROWS_AS(
        TifLoader::load("/nonexistent/path/fake.tif"),
        std::runtime_error
    );
}

TEST_CASE("TifLoader output is compatible with KMeans") {
    // Load a small multi-band raster and pass directly to KMeans::fit()
    const int n_pixels = 6;
    std::vector<std::vector<float>> bands = {
        {1.f, 2.f, 3.f, 100.f, 101.f, 102.f},
        {1.f, 2.f, 3.f, 100.f, 101.f, 102.f}
    };
    auto path = create_vsimem_tif("test_compat.tif", 3, 2, 2, bands);

    auto data = TifLoader::load(path);
    REQUIRE(data.size() == static_cast<std::size_t>(n_pixels));

    KMeans km(2, 200, 1e-6, 42);
    km.fit(data);

    REQUIRE(km.labels().size() == static_cast<std::size_t>(n_pixels));

    VSIUnlink(path.c_str());
}
