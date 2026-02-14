#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int repeats = (argc > 2) ? atoi(argv[2]) : 1;
    size_t size = N * N * sizeof(float);

    if (N <= 0 || repeats <= 0) {
        fprintf(stderr, "Usage: %s [N] [repeats]\n", argv[0]);
        return 1;
    }

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed for N=%d\n", N);
        free(A);
        free(B);
        free(C);
        return 1;
    }

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    double total_elapsed = 0.0;
    for (int r = 0; r < repeats; r++) {
        clock_t start = clock();
        matrixMultiplyCPU(A, B, C, N);
        clock_t end = clock();
        total_elapsed += (double)(end - start) / CLOCKS_PER_SEC;
    }

    printf("CPU execution time (N=%d, repeats=%d): %f seconds\n",
           N, repeats, total_elapsed / repeats);
    printf("CSV,CPU,%d,%d,%f\n", N, repeats, total_elapsed / repeats);

    free(A);
    free(B);
    free(C);
    return 0;
}