from __future__ import annotations

import argparse
from pathlib import Path

from src.backtest import BacktestConfig, run_backtest
from src.data_loader import load_stock_data
from src.metrics import compute_metrics, metrics_to_frame
from src.strategy import generate_signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 4 stock algorithm + mock trading environment")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--csv", type=str, default=None, help="Optional local CSV with Date/Close columns")
    parser.add_argument("--short-window", type=int, default=20)
    parser.add_argument("--long-window", type=int, default=50)
    parser.add_argument("--use-rsi-filter", action="store_true")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-buy-max", type=float, default=70.0)
    parser.add_argument("--rsi-sell-min", type=float, default=30.0)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    price_df = load_stock_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        csv_path=args.csv,
    )

    signals = generate_signals(
        price_df,
        short_window=args.short_window,
        long_window=args.long_window,
        use_rsi_filter=args.use_rsi_filter,
        rsi_period=args.rsi_period,
        rsi_buy_max=args.rsi_buy_max,
        rsi_sell_min=args.rsi_sell_min,
    )

    config = BacktestConfig(
        initial_cash=args.initial_cash,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    history, trades = run_backtest(signals, config)
    metrics = compute_metrics(history, initial_cash=args.initial_cash)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker_tag = args.ticker if not args.csv else Path(args.csv).stem
    history_file = out_dir / f"history_{ticker_tag}.csv"
    trades_file = out_dir / f"trades_{ticker_tag}.csv"
    metrics_file = out_dir / f"metrics_{ticker_tag}.csv"

    history.to_csv(history_file, index=False)
    trades.to_csv(trades_file, index=False)
    metrics_to_frame(metrics).to_csv(metrics_file, index=False)

    print("Run complete.")
    print(f"Saved history: {history_file}")
    print(f"Saved trades:  {trades_file}")
    print(f"Saved metrics: {metrics_file}")
    print("")
    print("Performance Metrics")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.6f}")
        else:
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
