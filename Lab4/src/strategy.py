from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def generate_signals(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    use_rsi_filter: bool = True,
    rsi_period: int = 14,
    rsi_buy_max: float = 70.0,
    rsi_sell_min: float = 30.0,
) -> pd.DataFrame:
    """
    Generate buy/sell state using moving-average crossover.

    signal:
      - 1 means hold long position
      - 0 means hold cash
    position_change:
      - +1 means buy
      - -1 means sell
    """
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    out = df.copy()
    out["SMA_short"] = out["Close"].rolling(window=short_window, min_periods=short_window).mean()
    out["SMA_long"] = out["Close"].rolling(window=long_window, min_periods=long_window).mean()
    out["RSI"] = compute_rsi(out["Close"], period=rsi_period)

    base_signal = (out["SMA_short"] > out["SMA_long"]).astype(int)

    if use_rsi_filter:
        buy_cond = (out["SMA_short"] > out["SMA_long"]) & (out["RSI"] <= rsi_buy_max)
        sell_cond = (out["SMA_short"] < out["SMA_long"]) & (out["RSI"] >= rsi_sell_min)

        signal_vals = np.zeros(len(out), dtype=int)
        for i in range(1, len(out)):
            signal_vals[i] = signal_vals[i - 1]
            if buy_cond.iat[i]:
                signal_vals[i] = 1
            elif sell_cond.iat[i]:
                signal_vals[i] = 0
        out["Signal"] = signal_vals
    else:
        out["Signal"] = base_signal

    out["PositionChange"] = out["Signal"].diff().fillna(0).astype(int)
    return out
