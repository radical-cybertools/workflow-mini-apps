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
    const int N = 1024 * 1024;
    const float alpha = 2.5f;
    const int n_warmup = 3;
    const int n_repeat = 50;

    // Host allocations
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;      // or whatever initialization
        h_y[i] = 0.0f;
    }

    // Device allocations
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm‐up
    for (int i = 0; i < n_warmup; ++i) {
        CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed repeats
    float total_ms = 0.0f;
    for (int i = 0; i < n_repeat; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    float avg_ms = total_ms / n_repeat;
    double bytes = double(N) * sizeof(float) * 2;  // read x and read/write y
    double bandwidth = bytes / (avg_ms / 1e3) / 1e9;  // GB/s

    printf("AXPY (N=%d) average over %d runs: %f ms, bandwidth = %f GB/s\n",
           N, n_repeat, avg_ms, bandwidth);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}

