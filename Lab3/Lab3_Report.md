# DSCI 560 - Lab 3 Report

**Student:** Herun Kan  
**Course:** DSCI 560 - Data Science Professional Practicum  
**Lab:** CUDA Program and Custom Python Library

## 1. Environment and Setup

- Local machine: Apple Silicon MacBook (M-chip, no native CUDA support).
- CUDA execution platform: Google Cloud GPU VM (`NVIDIA T4`).
- OS: Ubuntu Linux.
- Toolchain: `gcc`, `nvcc`, `cuBLAS`, Python `ctypes`, `numpy`, `pandas`, `matplotlib`.

Because CUDA is NVIDIA-specific, all GPU experiments were run on cloud infrastructure.

## 2. Implementations Completed

1. CPU matrix multiplication in C (`cpu/matrix_cpu.c`).
2. Naive CUDA matrix multiplication (`cuda/matrix_gpu.cu`).
3. Tiled shared-memory CUDA matrix multiplication (`cuda/matrix_tiled.cu`).
4. cuBLAS matrix multiplication (`cuda/matrix_cublas.cu`).
5. CUDA shared libraries for Python:
   - Matrix multiply (`lib/matrix_lib.cu`, `python/use_matrix_lib.py`)
   - Convolution (`lib/convolution_lib.cu`, `python/use_convolution_lib.py`)
6. Convolution programs:
   - CPU (`cpu/convolution_cpu.c`)
   - CUDA (`cuda/convolution_gpu.cu`)

## 3. Matrix Multiplication Results

The benchmark ran at matrix sizes `N = 512, 1024, 2048`.

### 3.1 Raw Runtime Table

| Implementation | N=512 | N=1024 | N=2048 | Unit |
|---|---:|---:|---:|---|
| CPU (C) | 0.317362 | 3.149439 | 75.530921 | sec |
| Naive CUDA | 0.714336 | 5.880467 | 44.022087 | ms |
| Tiled CUDA | 0.423936 | 3.496265 | 24.064592 | ms |
| cuBLAS | 0.134195 | 0.504979 | 2.417926 | ms |

For fair speedup comparison, CPU values were converted from seconds to milliseconds.

### 3.2 Speedup vs CPU (`CPU_time / GPU_time`)

| Implementation | N=512 | N=1024 | N=2048 |
|---|---:|---:|---:|
| Naive CUDA | 444.27x | 535.58x | 1715.79x |
| Tiled CUDA | 748.61x | 900.81x | 3138.70x |
| cuBLAS | 2364.92x | 6236.76x | 31238.19x |

### 3.3 Kernel-to-Kernel Comparison

| Comparison | N=512 | N=1024 | N=2048 |
|---|---:|---:|---:|
| Tiled speedup over Naive (`Naive/Tiled`) | 1.69x | 1.68x | 1.83x |
| cuBLAS speedup over Tiled (`Tiled/cuBLAS`) | 3.16x | 6.92x | 9.95x |

## 4. Analysis Questions

### Q1) How does performance change as matrix size increases?

All implementations become slower as `N` increases, but not equally. CPU time grows much faster than GPU methods due to `O(N^3)` arithmetic and limited parallelism. GPU implementations also increase with size, but with much better scaling because work is distributed across thousands of CUDA cores.

### Q2) At what point does the GPU significantly outperform the CPU?

In this experiment, GPU already strongly outperforms CPU at `N=512` and the gap grows at `N=1024` and `N=2048`. This indicates the fixed overhead of GPU launch/memory transfer is amortized quickly for these matrix sizes.

### Q3) How much speedup is gained by tiling optimization vs naive CUDA?

Tiling gives a consistent improvement of about `1.68x` to `1.83x` over naive CUDA. The gain comes from shared-memory reuse and fewer expensive global-memory reads.

### Q4) How close is optimized tiled kernel to cuBLAS?

The tiled kernel is still behind cuBLAS by roughly `3.16x` (`N=512`) to `9.95x` (`N=2048`). cuBLAS advantage grows with size.

### Q5) Why might cuBLAS still outperform hand-written kernels?

cuBLAS uses highly tuned architecture-specific implementations, including advanced tiling strategies, better memory scheduling, vectorization, and low-level optimizations developed and maintained by NVIDIA. Hand-written kernels are usually less optimized and harder to tune across all problem sizes.

## 5. Shared Library + Python Integration

The lab requirement to expose CUDA functions as shared libraries and call from Python was completed:

- Matrix shared library call:
  - `lib/libmatrix.so`
  - `python/use_matrix_lib.py`
- Convolution shared library call:
  - `lib/libconv.so`
  - `python/use_convolution_lib.py`

These scripts execute CUDA functions via `ctypes` and print runtime output for reporting.

## 6. Convolution Results (To Fill with Your Run Outputs)

The required CPU vs CUDA convolution implementations are completed in code.  
Add your measured results from VM runs in the table below.

| Image Size (M) | Kernel Size (N) | CPU Time | CUDA Time | Speedup |
|---:|---:|---:|---:|---:|
| 512 | 3 |  |  |  |
| 1024 | 5 |  |  |  |
| 2048 | 7 |  |  |  |

You can collect these quickly with:

```bash
./cpu/convolution_cpu 512 3 3
./cpu/convolution_cpu 1024 5 3
./cpu/convolution_cpu 2048 7 3

./cuda/convolution_gpu 512 3 10
./cuda/convolution_gpu 1024 5 10
./cuda/convolution_gpu 2048 7 10
```

## 7. Graphs

Generate runtime graph from `results.csv`:

```bash
python3 scripts/plot_results.py --input results.csv --output matrix_runtime_plot.png
```

Include the produced graph image in the submitted report.  
If needed, create a second graph for speedup vs matrix size.

## 8. Overhead and Optimization Reflection

- GPU acceleration includes overhead from host-device memory transfers and kernel launch.
- For sufficiently large matrices, parallel throughput dominates this overhead.
- Shared-memory tiling reduces memory bottlenecks and improves arithmetic intensity.
- cuBLAS demonstrates that production-grade kernels can significantly exceed custom implementations.

## 9. Deliverables Checklist

- [x] CPU, naive CUDA, tiled CUDA, cuBLAS source code.
- [x] Shared library source code and Python callers.
- [x] Matrix runtime results with scaling analysis.
- [ ] Convolution timing table fully populated.
- [ ] Graph images embedded in final submission document.
