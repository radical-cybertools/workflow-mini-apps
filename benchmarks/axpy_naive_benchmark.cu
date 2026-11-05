#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

__global__ void axpy_kernel(int N, float alpha, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] += alpha * x[idx];
    }
}

int main() {
    const int N = 1024 * 1024 * 32;
    const float alpha = 1.1f;
    const int n_warmup = 3;
    const int n_repeat = 50;

    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 0.0f;
    }

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    const int TPB = 256;
    int blocks = (N + TPB - 1) / TPB;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < n_warmup; ++i) {
        axpy_kernel<<<blocks, TPB>>>(N, alpha, d_x, d_y);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < n_repeat; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        axpy_kernel<<<blocks, TPB>>>(N, alpha, d_x, d_y);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    float avg_ms = total_ms / n_repeat;
    printf("AXPY kernel (N=%d) average over %d runs: %f ms\n",
           N, n_repeat, avg_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x);
    free(h_y);

    return 0;
}

