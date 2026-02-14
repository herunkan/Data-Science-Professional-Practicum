#include <cuda_runtime.h>

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

extern "C" int gpu_convolve2d(unsigned int *h_image, float *h_kernel, float *h_output, int M, int N) {
    size_t image_size = (size_t)M * M * sizeof(unsigned int);
    size_t kernel_size = (size_t)N * N * sizeof(float);
    size_t output_size = (size_t)M * M * sizeof(float);

    unsigned int *d_image = NULL;
    float *d_kernel = NULL;
    float *d_output = NULL;

    if (N % 2 == 0 || M <= 0 || N <= 0) return 1;
    if (cudaMalloc((void **)&d_image, image_size) != cudaSuccess) return 2;
    if (cudaMalloc((void **)&d_kernel, kernel_size) != cudaSuccess) return 3;
    if (cudaMalloc((void **)&d_output, output_size) != cudaSuccess) return 4;
    if (cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice) != cudaSuccess) return 5;
    if (cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice) != cudaSuccess) return 6;

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    convolve2d_gpu<<<grid, block>>>(d_image, d_kernel, d_output, M, N);
    if (cudaGetLastError() != cudaSuccess) return 7;
    if (cudaDeviceSynchronize() != cudaSuccess) return 8;
    if (cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost) != cudaSuccess) return 9;

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}
