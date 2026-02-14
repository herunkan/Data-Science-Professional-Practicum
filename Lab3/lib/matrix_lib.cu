#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float acc = 0.0f;

    int tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int m = 0; m < tiles; ++m) {
        int a_col = m * TILE_WIDTH + tx;
        int b_row = m * TILE_WIDTH + ty;
        ds_A[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        ds_B[ty][tx] = (col < N && b_row < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

extern "C" int gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = (size_t)N * N * sizeof(float);
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    if (cudaMalloc((void **)&d_A, size) != cudaSuccess) return 1;
    if (cudaMalloc((void **)&d_B, size) != cudaSuccess) return 2;
    if (cudaMalloc((void **)&d_C, size) != cudaSuccess) return 3;

    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess) return 4;
    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess) return 5;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    if (cudaGetLastError() != cudaSuccess) return 6;
    if (cudaDeviceSynchronize() != cudaSuccess) return 7;

    if (cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess) return 8;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
