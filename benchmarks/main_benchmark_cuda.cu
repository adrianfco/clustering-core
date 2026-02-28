#include "../src/cuda/kmeans_cuda.cu"
#include "../src/cuda/pfcm_cuda.cu"
#include <iostream>

int main() {
    kmeans_gpu_stub();
    pfcm_gpu_stub();
    std::cout << "CUDA Benchmark stub done.\n";
    return 0;
}
