# Lab 4 - Real-Time Stock Analysis and Algorithmic Trading

This folder implements the required Lab 4 workflow:

1. Develop a stock trading algorithm (SMA crossover with optional RSI filter).
2. Build a mock trading environment.
3. Evaluate portfolio performance (total return, annualized return, Sharpe ratio).
4. Save outputs for report and demo video.

## Project Structure

- `main.py` - end-to-end runner (load data -> generate signals -> backtest -> metrics)
- `src/data_loader.py` - load stock data from CSV or Yahoo Finance
- `src/strategy.py` - trading signal generation (SMA + optional RSI)
- `src/backtest.py` - mock trading simulator
- `src/metrics.py` - performance metrics
- `plot_results.py` - generate charts for report
- `meeting_minutes_template.md` - team meeting log template
- `report_outline.md` - suggested report sections

## Setup

```bash
cd Lab4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run with Yahoo Finance Data

```bash
python3 main.py \
  --ticker AAPL \
  --start 2023-01-01 \
  --end 2025-01-01 \
  --short-window 20 \
  --long-window 50 \
  --use-rsi-filter \
  --initial-cash 10000 \
  --transaction-cost-bps 5
```

## Run with Local CSV Data

Your CSV should include at least `Date` and `Close` (or `Adj Close`) columns.

```bash
python3 main.py \
  --csv path/to/stock_prices.csv \
  --short-window 20 \
  --long-window 50 \
  --initial-cash 10000
```

## Output Files

Generated in `outputs/`:

- `history_<ticker>.csv` - daily price, signal, action, holdings, portfolio value
- `trades_<ticker>.csv` - buy/sell records
- `metrics_<ticker>.csv` - summary metrics

## Create Plot for Report

```bash
python3 plot_results.py \
  --history outputs/history_AAPL.csv \
  --output outputs/performance_plot.png
```

## Required Lab Deliverables Mapping

- **Algorithm Development**
  - SMA crossover implemented; optional RSI filter included.
  - Extensible structure to add ARIMA/LSTM later.
- **Mock Trading Environment**
  - Initial investment, buy/sell execution, holdings tracking implemented.
  - Performance metrics computed: total return, annualized return, Sharpe ratio.
- **Team Discussions**
  - Fill `meeting_minutes_template.md` daily.
- **Submission**
  - Code files: present in this folder.
  - README: this file (export to PDF if required).
  - Meeting minutes: complete template and export to PDF.
  - Demo video: explain algorithm rationale and performance metrics.

## Notes

- The default strategy is long-only and all-in/all-out for simplicity.
- For extra credit quality, compare multiple parameter sets and tickers.
