# Lab 3 - CUDA Program and Custom Python Library

This folder includes complete starter/solution code for all required Lab 3 parts:
- CPU matrix multiplication
- Naive CUDA matrix multiplication
- Tiled (shared-memory) CUDA matrix multiplication
- cuBLAS matrix multiplication
- CUDA shared libraries callable from Python (`ctypes`)
- CPU and CUDA 2D convolution

## Folder Layout

- `cpu/matrix_cpu.c`
- `cpu/convolution_cpu.c`
- `cuda/matrix_gpu.cu`
- `cuda/matrix_tiled.cu`
- `cuda/matrix_cublas.cu`
- `cuda/convolution_gpu.cu`
- `lib/matrix_lib.cu`
- `lib/convolution_lib.cu`
- `python/use_matrix_lib.py`
- `python/use_convolution_lib.py`
- `scripts/run_benchmarks.sh`
- `scripts/plot_results.py`

## Apple Silicon (M-chip) Note

CUDA requires NVIDIA GPUs and will not run natively on an M-chip Mac.  
Use a cloud NVIDIA environment for all CUDA steps.

Recommended options:
1. **Google Cloud Compute Engine** (best for this lab report)
2. **Paperspace/Lambda Cloud/RunPod** (good alternatives)
3. **Google Colab** (easy start, but session/runtime limits)

## Quick Google Cloud Setup (Recommended)

1. Create VM:
   - Machine: `n1-standard-8` (or better)
   - GPU: `NVIDIA T4`
   - OS: Ubuntu 22.04 (or 20.04)
2. SSH into VM and run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-pip git
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
nvidia-smi
nvcc --version
```

3. Clone your repo and build:

```bash
git clone https://github.com/herunkan/Data-Science-Professional-Practicum.git
cd Data-Science-Professional-Practicum/Lab3

gcc cpu/matrix_cpu.c -O2 -o cpu/matrix_cpu
gcc cpu/convolution_cpu.c -O2 -o cpu/convolution_cpu

nvcc cuda/matrix_gpu.cu -O3 -o cuda/matrix_gpu
nvcc cuda/matrix_tiled.cu -O3 -o cuda/matrix_tiled
nvcc cuda/matrix_cublas.cu -O3 -lcublas -o cuda/matrix_cublas
nvcc cuda/convolution_gpu.cu -O3 -o cuda/convolution_gpu

nvcc -Xcompiler -fPIC -shared lib/matrix_lib.cu -O3 -o lib/libmatrix.so
nvcc -Xcompiler -fPIC -shared lib/convolution_lib.cu -O3 -o lib/libconv.so
```

4. Install Python deps:

```bash
python3 -m pip install numpy pandas matplotlib
```

## Run Matrix Benchmarks

```bash
./cpu/matrix_cpu 512 1
./cpu/matrix_cpu 1024 1
./cpu/matrix_cpu 2048 1

./cuda/matrix_gpu 512 10
./cuda/matrix_gpu 1024 10
./cuda/matrix_gpu 2048 10

./cuda/matrix_tiled 512 10
./cuda/matrix_tiled 1024 10
./cuda/matrix_tiled 2048 10

./cuda/matrix_cublas 512 10
./cuda/matrix_cublas 1024 10
./cuda/matrix_cublas 2048 10
```

Or automate:

```bash
chmod +x scripts/run_benchmarks.sh
./scripts/run_benchmarks.sh | tee results.csv
```

## Run Shared Library from Python

```bash
python3 python/use_matrix_lib.py --n 1024 --lib ./lib/libmatrix.so
python3 python/use_convolution_lib.py --m 1024 --n 3 --lib ./lib/libconv.so
```

## Run Convolution CPU/CUDA

```bash
./cpu/convolution_cpu 512 3 3
./cpu/convolution_cpu 1024 5 3
./cpu/convolution_cpu 2048 7 3

./cuda/convolution_gpu 512 3 10
./cuda/convolution_gpu 1024 5 10
./cuda/convolution_gpu 2048 7 10
```

## Report Checklist

1. Runtime table for `N=512,1024,2048`:
   - CPU
   - Naive CUDA
   - Tiled CUDA
   - cuBLAS
2. Speedup columns (`CPU / GPU`).
3. Graph runtime vs matrix size.
4. Analysis answers:
   - performance scaling
   - crossover point where GPU wins
   - tiled vs naive speedup
   - tiled vs cuBLAS gap
   - why cuBLAS is faster
5. Convolution comparison:
   - CPU vs CUDA for at least 3 image sizes and 3 kernel sizes.
6. Shared library list and Python integration proof (screenshots/output logs).
