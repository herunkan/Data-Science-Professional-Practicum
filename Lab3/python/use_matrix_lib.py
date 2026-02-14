import argparse
import ctypes
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--lib", type=str, default="../lib/libmatrix.so")
    args = parser.parse_args()

    n = args.n
    lib_path = Path(args.lib)
    lib = ctypes.cdll.LoadLibrary(str(lib_path))

    lib.gpu_matrix_multiply.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]
    lib.gpu_matrix_multiply.restype = ctypes.c_int

    rng = np.random.default_rng(42)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    c = np.zeros((n, n), dtype=np.float32)

    start = time.perf_counter()
    rc = lib.gpu_matrix_multiply(a.ravel(), b.ravel(), c.ravel(), n)
    elapsed = time.perf_counter() - start

    if rc != 0:
        raise RuntimeError(f"gpu_matrix_multiply failed with code {rc}")

    print(f"Python->CUDA matrix multiply time (N={n}): {elapsed:.6f} seconds")
    print(f"CSV,PY_MATRIX_LIB,{n},1,{elapsed:.6f}")


if __name__ == "__main__":
    main()
