#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CHECK_CUFFT(call)                                                     \
    do {                                                                      \
        cufftResult err = call;                                               \
        if (err != CUFFT_SUCCESS) {                                           \
            fprintf(stderr, "cuFFT Error %s:%d: %d\n", __FILE__, __LINE__, err); \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

int main() {
    const int N         = 1024;            // length of each FFT
    const int batch     = 1024;            // number of transforms (one per column)
    const int n_warmup  = 3;
    const int n_repeat  = 50;

    // host-side real input (simulate data_in); we'll pack into complex
    size_t real_elems = size_t(N) * batch;
    float *h_real = (float*)malloc(real_elems * sizeof(float));
    for (size_t i = 0; i < real_elems; ++i) {
        h_real[i] = 1.0f;  // or load your actual data here
    }

    // Device buffers: complex in/out
    cufftComplex *d_in = nullptr, *d_out = nullptr;
    size_t complex_bytes = real_elems * sizeof(cufftComplex);
    CHECK_CUDA(cudaMalloc(&d_in,  complex_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, complex_bytes));

    // Pack real→complex on host and copy once
    cufftComplex *h_pack = (cufftComplex*)malloc(complex_bytes);
    for (size_t i = 0; i < real_elems; ++i) {
        h_pack[i].x = h_real[i];
        h_pack[i].y = 0.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_pack, complex_bytes, cudaMemcpyHostToDevice));
    free(h_pack);
    free(h_real);

    // Create a batched 1D C2C plan
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm‑up (not timed)
    for (int i = 0; i < n_warmup; ++i) {
        CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed repeats
    float total_ms = 0.0f;
    for (int i = 0; i < n_repeat; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
        CHECK_CUDA(cudaEventRecord(stop,  0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    float avg_ms = total_ms / n_repeat;

    // Approximate FLOPS: ~5·N·log2(N) per transform
    double flops_per_transform = 5.0 * N * std::log2(double(N));
    double total_flops         = flops_per_transform * batch;
    double gflops              = total_flops / (avg_ms/1e3) / 1e9;

    printf("cuFFT C2C 1D FFT (N=%d, batch=%d)\n", N, batch);
    printf("  Average over %d runs: %f ms\n", n_repeat, avg_ms);
    printf("  Approx. throughput: %f GFLOPS\n", gflops);

    // Cleanup
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}

