#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void checkCublas(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s\n", msg);
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

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate failed");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N),
        "cublasSgemm warmup failed"
    );
    checkCuda(cudaDeviceSynchronize(), "warmup sync failed");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start failed");
    checkCuda(cudaEventCreate(&stop), "event create stop failed");

    checkCuda(cudaEventRecord(start), "event start failed");
    for (int r = 0; r < repeats; r++) {
        checkCublas(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N),
            "cublasSgemm failed"
        );
    }
    checkCuda(cudaEventRecord(stop), "event stop failed");
    checkCuda(cudaEventSynchronize(stop), "event sync stop failed");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time failed");
    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "D2H copy C failed");

    printf("cuBLAS execution time (N=%d, repeats=%d): %f ms\n", N, repeats, ms / repeats);
    printf("CSV,CUBLAS,%d,%d,%f\n", N, repeats, ms / repeats);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
