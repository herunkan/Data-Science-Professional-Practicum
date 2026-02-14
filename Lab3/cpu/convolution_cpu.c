#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void convolve2d_cpu(const unsigned int *image,
                           const float *kernel,
                           float *output,
                           int M,
                           int N) {
    int half = N / 2;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float acc = 0.0f;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int iy = y + ky;
                    int ix = x + kx;
                    if (iy >= 0 && iy < M && ix >= 0 && ix < M) {
                        int k_row = ky + half;
                        int k_col = kx + half;
                        acc += (float)image[iy * M + ix] * kernel[k_row * N + k_col];
                    }
                }
            }
            output[y * M + x] = acc;
        }
    }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 3;
    int repeats = (argc > 3) ? atoi(argv[3]) : 3;
    if (M <= 0 || N <= 0 || (N % 2 == 0) || repeats <= 0) {
        fprintf(stderr, "Usage: %s [M] [N_odd] [repeats]\n", argv[0]);
        return 1;
    }

    size_t image_count = (size_t)M * M;
    size_t kernel_count = (size_t)N * N;
    unsigned int *image = (unsigned int *)malloc(image_count * sizeof(unsigned int));
    float *kernel = (float *)malloc(kernel_count * sizeof(float));
    float *output = (float *)malloc(image_count * sizeof(float));
    if (!image || !kernel || !output) {
        fprintf(stderr, "Allocation failed.\n");
        free(image);
        free(kernel);
        free(output);
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < image_count; i++) image[i] = rand() % 256;
    for (size_t i = 0; i < kernel_count; i++) kernel[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;

    double total_elapsed = 0.0;
    for (int r = 0; r < repeats; r++) {
        clock_t start = clock();
        convolve2d_cpu(image, kernel, output, M, N);
        clock_t end = clock();
        total_elapsed += (double)(end - start) / CLOCKS_PER_SEC;
    }

    printf("CPU convolution time (M=%d, N=%d, repeats=%d): %f seconds\n",
           M, N, repeats, total_elapsed / repeats);
    printf("CSV,CPU_CONV,%d,%d,%d,%f\n", M, N, repeats, total_elapsed / repeats);

    free(image);
    free(kernel);
    free(output);
    return 0;
}
