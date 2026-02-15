from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestConfig:
    initial_cash: float = 10000.0
    transaction_cost_bps: float = 5.0


def run_backtest(signals_df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    All-in / all-out long-only backtest on one symbol.
    """
    cash = float(config.initial_cash)
    shares = 0
    fee_rate = config.transaction_cost_bps / 10000.0

    rows = []
    trades = []

    for row in signals_df.itertuples(index=False):
        date = row.Date
        price = float(row.Close)
        signal = int(row.Signal)

        action = "HOLD"
        if signal == 1 and shares == 0:
            buyable = int(cash // price)
            if buyable > 0:
                notional = buyable * price
                fee = notional * fee_rate
                cash -= notional + fee
                shares += buyable
                action = "BUY"
                trades.append(
                    {
                        "Date": date,
                        "Action": action,
                        "Price": price,
                        "Shares": buyable,
                        "Notional": notional,
                        "Fee": fee,
                    }
                )
        elif signal == 0 and shares > 0:
            notional = shares * price
            fee = notional * fee_rate
            cash += notional - fee
            action = "SELL"
            trades.append(
                {
                    "Date": date,
                    "Action": action,
                    "Price": price,
                    "Shares": shares,
                    "Notional": notional,
                    "Fee": fee,
                }
            )
            shares = 0

        holdings_value = shares * price
        portfolio_value = cash + holdings_value
        rows.append(
            {
                "Date": date,
                "Close": price,
                "Signal": signal,
                "Action": action,
                "Cash": cash,
                "Shares": shares,
                "HoldingsValue": holdings_value,
                "PortfolioValue": portfolio_value,
            }
        )

    history = pd.DataFrame(rows)
    history["DailyReturn"] = history["PortfolioValue"].pct_change().fillna(0.0)
    trades_df = pd.DataFrame(trades)
    return history, trades_df
