#include "clustering_core/tif_loader.hpp"
#include <gdal_priv.h>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<std::vector<float>> TifLoader::load(const std::string& path) {
    GDALAllRegister();

    GDALDataset* dataset = static_cast<GDALDataset*>(
        GDALOpen(path.c_str(), GA_ReadOnly)
    );
    if (!dataset)
        throw std::runtime_error("TifLoader: cannot open file: " + path);

    const int width   = dataset->GetRasterXSize();
    const int height  = dataset->GetRasterYSize();
    const int n_bands = dataset->GetRasterCount();

    if (n_bands == 0) {
        GDALClose(dataset);
        throw std::runtime_error("TifLoader: file has no raster bands: " + path);
    }

    const int n_pixels = width * height;
    std::vector<std::vector<float>> result(n_pixels, std::vector<float>(n_bands));
    std::vector<float> band_buf(n_pixels);

    for (int b = 1; b <= n_bands; ++b) {
        GDALRasterBand* band = dataset->GetRasterBand(b);
        if (!band) {
            GDALClose(dataset);
            throw std::runtime_error("TifLoader: cannot read band " + std::to_string(b));
        }

        CPLErr err = band->RasterIO(
            GF_Read,
            0, 0, width, height,
            band_buf.data(),
            width, height,
            GDT_Float32,
            0, 0
        );
        if (err != CE_None) {
            GDALClose(dataset);
            throw std::runtime_error("TifLoader: RasterIO failed on band " + std::to_string(b));
        }

        for (int p = 0; p < n_pixels; ++p)
            result[p][b - 1] = band_buf[p];
    }

    GDALClose(dataset);
    return result;
}
