import argparse
import ctypes
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--lib", type=str, default="../lib/libconv.so")
    args = parser.parse_args()

    m = args.m
    n = args.n
    if n % 2 == 0:
        raise ValueError("Kernel size n must be odd.")

    lib_path = Path(args.lib)
    lib = ctypes.cdll.LoadLibrary(str(lib_path))

    lib.gpu_convolve2d.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gpu_convolve2d.restype = ctypes.c_int

    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (m, m), dtype=np.uint32)
    kernel = rng.uniform(-1.0, 1.0, (n, n)).astype(np.float32)
    output = np.zeros((m, m), dtype=np.float32)

    start = time.perf_counter()
    rc = lib.gpu_convolve2d(image.ravel(), kernel.ravel(), output.ravel(), m, n)
    elapsed = time.perf_counter() - start

    if rc != 0:
        raise RuntimeError(f"gpu_convolve2d failed with code {rc}")

    print(f"Python->CUDA convolution time (M={m}, N={n}): {elapsed:.6f} seconds")
    print(f"CSV,PY_CONV_LIB,{m},{n},1,{elapsed:.6f}")


if __name__ == "__main__":
    main()
