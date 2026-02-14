#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolve2d_gpu(const unsigned int *image,
                               const float *kernel,
                               float *output,
                               int M,
                               int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= M) return;

    int half = N / 2;
    float acc = 0.0f;
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            if (ix >= 0 && ix < M && iy >= 0 && iy < M) {
                int k_row = ky + half;
                int k_col = kx + half;
                acc += (float)image[iy * M + ix] * kernel[k_row * N + k_col];
            }
        }
    }
    output[y * M + x] = acc;
}

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 3;
    int repeats = (argc > 3) ? atoi(argv[3]) : 10;
    if (M <= 0 || N <= 0 || (N % 2 == 0) || repeats <= 0) {
        fprintf(stderr, "Usage: %s [M] [N_odd] [repeats]\n", argv[0]);
        return 1;
    }

    size_t image_size = (size_t)M * M * sizeof(unsigned int);
    size_t kernel_size = (size_t)N * N * sizeof(float);
    size_t output_size = (size_t)M * M * sizeof(float);

    unsigned int *h_image = (unsigned int *)malloc(image_size);
    float *h_kernel = (float *)malloc(kernel_size);
    float *h_output = (float *)malloc(output_size);
    if (!h_image || !h_kernel || !h_output) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * M; i++) h_image[i] = rand() % 256;
    for (int i = 0; i < N * N; i++) h_kernel[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;

    unsigned int *d_image = NULL;
    float *d_kernel = NULL;
    float *d_output = NULL;
    checkCuda(cudaMalloc((void **)&d_image, image_size), "cudaMalloc d_image failed");
    checkCuda(cudaMalloc((void **)&d_kernel, kernel_size), "cudaMalloc d_kernel failed");
    checkCuda(cudaMalloc((void **)&d_output, output_size), "cudaMalloc d_output failed");
    checkCuda(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice), "H2D image failed");
    checkCuda(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice), "H2D kernel failed");

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    convolve2d_gpu<<<grid, block>>>(d_image, d_kernel, d_output, M, N);
    checkCuda(cudaGetLastError(), "warmup kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "warmup sync failed");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event start create failed");
    checkCuda(cudaEventCreate(&stop), "event stop create failed");
    checkCuda(cudaEventRecord(start), "event record start failed");
    for (int r = 0; r < repeats; r++) {
        convolve2d_gpu<<<grid, block>>>(d_image, d_kernel, d_output, M, N);
    }
    checkCuda(cudaEventRecord(stop), "event record stop failed");
    checkCuda(cudaEventSynchronize(stop), "event sync stop failed");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time failed");
    checkCuda(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "D2H output failed");

    printf("GPU convolution time (M=%d, N=%d, repeats=%d): %f ms\n", M, N, repeats, ms / repeats);
    printf("CSV,CUDA_CONV,%d,%d,%d,%f\n", M, N, repeats, ms / repeats);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_image);
    free(h_kernel);
    free(h_output);
    return 0;
}
