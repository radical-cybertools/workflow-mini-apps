// matmul_cublas.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"" << std::endl;                                  \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t stat = (call);                                        \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << stat << std::endl;                      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <device> <matrix_size>\n";
        return EXIT_FAILURE;
    }

    int dev = std::atoi(argv[1]);
    int N   = std::atoi(argv[2]);

    // Select GPU
    CHECK_CUDA(cudaSetDevice(dev));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device matrices A, B, C (uninitialized, like cupy.empty)
    size_t bytes = size_t(N) * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Warm-up: one matmul to initialize kernels / cache
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_A, N,
                    d_B, N,
                    &beta,
                    d_C, N)
    );
    // Ensure warm-up completed
    CHECK_CUDA(cudaDeviceSynchronize());

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start, run 20 matmuls, record stop
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 20; ++i) {
        CHECK_CUBLAS(
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N,
                        &alpha,
                        d_A, N,
                        d_B, N,
                        &beta,
                        d_C, N)
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Matrix size: " << N << "×" << N << "\n"
              << "Total time for 20 calls: " << ms << " ms\n"
              << "Average per call: " << (ms / 20.0f) << " ms\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
