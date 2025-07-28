#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t st = call;                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS Error %s:%d: %d\n", __FILE__, __LINE__, st); \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

int main() {
    const int N = 1024;
    const size_t bytes = size_t(N) * N * sizeof(float);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int n_warmup = 3;
    const int n_repeat = 50;

    // Host allocations & initialize
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N*N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm‑up SGEMM calls (not timed)
    for (int i = 0; i < n_warmup; ++i) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed repeats
    float total_ms = 0.0f;
    for (int i = 0; i < n_repeat; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N
        ));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    float avg_ms = total_ms / n_repeat;
    double flops = 2.0 * double(N) * N * N;  // 2·N³ operations for SGEMM
    double gflops = (flops / (avg_ms / 1e3)) / 1e9;

    printf("cuBLAS SGEMM (N=%d) average over %d runs: %f ms → %f GFLOPS\n",
           N, n_repeat, avg_ms, gflops);

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

