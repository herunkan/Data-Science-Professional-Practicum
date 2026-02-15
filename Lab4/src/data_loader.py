from __future__ import annotations

import pandas as pd


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    if "Date" not in df.columns:
        raise ValueError("Input data must contain a Date column.")

    close_col = None
    for candidate in ("Close", "Adj Close", "close", "adj_close"):
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        raise ValueError("Input data must contain Close or Adj Close column.")

    out = df[["Date", close_col]].copy()
    out = out.rename(columns={close_col: "Close"})
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").dropna()
    out = out.drop_duplicates(subset=["Date"])
    out = out.reset_index(drop=True)
    return out


def load_stock_data(
    ticker: str,
    start: str,
    end: str,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Load stock data from CSV (if provided) or Yahoo Finance.
    Returns a DataFrame with columns: Date, Close.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        return _normalize_price_frame(df)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for online download. "
            "Install with: pip install yfinance"
        ) from exc

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(
            "No data returned from Yahoo Finance. "
            "Check ticker/symbol, date range, or use --csv path/to/data.csv."
        )

    raw = raw.reset_index()
    return _normalize_price_frame(raw)
