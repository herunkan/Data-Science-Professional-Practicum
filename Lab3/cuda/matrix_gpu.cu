#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMultiplyGPU(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int repeats = (argc > 2) ? atoi(argv[2]) : 10;
    if (N <= 0 || repeats <= 0) {
        fprintf(stderr, "Usage: %s [N] [repeats]\n", argv[0]);
        return 1;
    }

    size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCuda(cudaMalloc((void **)&d_A, size), "cudaMalloc d_A failed");
    checkCuda(cudaMalloc((void **)&d_B, size), "cudaMalloc d_B failed");
    checkCuda(cudaMalloc((void **)&d_C, size), "cudaMalloc d_C failed");

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "H2D copy A failed");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "H2D copy B failed");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start failed");
    checkCuda(cudaEventCreate(&stop), "event create stop failed");

    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError(), "warmup kernel failed");
    checkCuda(cudaDeviceSynchronize(), "warmup sync failed");

    checkCuda(cudaEventRecord(start), "event record start failed");
    for (int r = 0; r < repeats; r++) {
        matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    checkCuda(cudaEventRecord(stop), "event record stop failed");
    checkCuda(cudaEventSynchronize(stop), "event sync stop failed");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time failed");
    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "D2H copy C failed");

    printf("Naive CUDA execution time (N=%d, repeats=%d): %f ms\n", N, repeats, ms / repeats);
    printf("CSV,NAIVE_CUDA,%d,%d,%f\n", N, repeats, ms / repeats);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
