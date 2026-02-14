import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.csv")
    parser.add_argument("--output", default="matrix_runtime_plot.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["N"] = df["N"].astype(int)
    df["Time"] = df["Time"].astype(float)

    plt.figure(figsize=(8, 5))
    for impl, grp in df.groupby("Implementation"):
        plt.plot(grp["N"], grp["Time"], marker="o", label=impl)

    plt.xlabel("Matrix size N")
    plt.ylabel("Time (sec for CPU, ms for GPU)")
    plt.title("Matrix Multiplication Runtime vs Matrix Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
