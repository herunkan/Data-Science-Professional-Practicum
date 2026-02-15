from __future__ import annotations

import math

import pandas as pd


def compute_metrics(history: pd.DataFrame, initial_cash: float, trading_days_per_year: int = 252) -> dict:
    values = history["PortfolioValue"]
    daily_ret = history["DailyReturn"]
    total_return = (values.iloc[-1] / initial_cash) - 1.0

    n_days = max(len(history), 1)
    annualized_return = (1.0 + total_return) ** (trading_days_per_year / n_days) - 1.0

    ret_std = daily_ret.std(ddof=1)
    if ret_std == 0 or math.isnan(ret_std):
        sharpe = 0.0
    else:
        sharpe = (daily_ret.mean() / ret_std) * math.sqrt(trading_days_per_year)

    running_max = values.cummax()
    drawdown = (values / running_max) - 1.0
    max_drawdown = drawdown.min()

    return {
        "InitialCash": initial_cash,
        "FinalPortfolioValue": float(values.iloc[-1]),
        "TotalReturn": float(total_return),
        "AnnualizedReturn": float(annualized_return),
        "SharpeRatio": float(sharpe),
        "MaxDrawdown": float(max_drawdown),
        "NumTradingDays": int(n_days),
        "NumTrades": int((history["Action"] != "HOLD").sum()),
    }


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([metrics])
