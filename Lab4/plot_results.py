from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strategy and portfolio performance")
    parser.add_argument("--history", type=str, required=True, help="Path to history_*.csv")
    parser.add_argument("--output", type=str, default="outputs/performance_plot.png")
    args = parser.parse_args()

    df = pd.read_csv(args.history)
    df["Date"] = pd.to_datetime(df["Date"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(df["Date"], df["Close"], label="Close", linewidth=1.2)
    buy_mask = df["Action"] == "BUY"
    sell_mask = df["Action"] == "SELL"
    axes[0].scatter(df.loc[buy_mask, "Date"], df.loc[buy_mask, "Close"], marker="^", s=70, label="Buy")
    axes[0].scatter(df.loc[sell_mask, "Date"], df.loc[sell_mask, "Close"], marker="v", s=70, label="Sell")
    axes[0].set_ylabel("Price")
    axes[0].set_title("Price and Trading Signals")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["Date"], df["PortfolioValue"], label="Portfolio Value", color="tab:green", linewidth=1.2)
    axes[1].set_ylabel("Portfolio Value")
    axes[1].set_title("Portfolio Value Over Time")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
