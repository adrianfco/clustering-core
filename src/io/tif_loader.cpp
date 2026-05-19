#include "clustering_core/tif_loader.hpp"
#include <gdal_priv.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace clustering {

Dataset TifLoader::load(const std::string& path, Layout layout) {
    GDALAllRegister();

    GDALDataset* ds = static_cast<GDALDataset*>(
        GDALOpen(path.c_str(), GA_ReadOnly));
    if (!ds)
        throw std::runtime_error("TifLoader: cannot open file: " + path);

    const int width    = ds->GetRasterXSize();
    const int height   = ds->GetRasterYSize();
    const int n_bands  = ds->GetRasterCount();
    const int n_pixels = width * height;

    if (n_bands == 0) {
        GDALClose(ds);
        throw std::runtime_error("TifLoader: file has no raster bands: " + path);
    }

    // Read directly into SoA — each band fills its [b*N .. (b+1)*N) slot.
    Dataset soa;
    soa.n      = n_pixels;
    soa.d      = n_bands;
    soa.layout = Layout::SoA;
    soa.features.resize(static_cast<std::size_t>(n_bands) * n_pixels);

    for (int b = 1; b <= n_bands; ++b) {
        GDALRasterBand* band = ds->GetRasterBand(b);
        if (!band) {
            GDALClose(ds);
            throw std::runtime_error("TifLoader: cannot get band " + std::to_string(b));
        }

        CPLErr err = band->RasterIO(
            GF_Read, 0, 0, width, height,
            soa.features.data() + static_cast<std::size_t>(b - 1) * n_pixels,
            width, height, GDT_Float32, 0, 0);
        if (err != CE_None) {
            GDALClose(ds);
            throw std::runtime_error("TifLoader: RasterIO failed on band " + std::to_string(b));
        }
    }

    GDALClose(ds);
    return (layout == Layout::SoA) ? soa : soa.as(Layout::AoS);
}

} // namespace clustering
