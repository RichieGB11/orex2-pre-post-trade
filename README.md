# Pre/Post Trade Metric Study

Small CLI program that analyzes sampled backtest trades against ORATS historical data to compare **winners vs losers**.

It uses the local `orats-client` library wrappers (no raw HTTP) and is designed to be lightweight but cleanly structured (input handling, data fetch, analysis, output).

## What this study answers

1. Did winners have higher IV-RV spread at entry than losers?
2. Did winners enter when spread was rising, falling, or flat?
3. Was slope percentile meaningfully different between winners and losers?
4. Did contango at entry predict anything about outcome?
5. Is there a combination of metrics that strongly favors winners?
6. How did IV-RV spread and contango change post-entry for winners vs. losers?

## Requirements

- Windows (or any environment with Python 3.8+)
- Python 3.8+
- ORATS API key in environment:

```powershell
setx ORATS_API_KEY "your-token"
```

- Local editable install of client library:

```powershell
pip install -e ../orats-client
```

## Project layout

- `pre_post_trade_study.py` — main CLI program
- `backtests/` — place backtest CSV files here

## Input CSV format

Required columns:

- `date` (trade open date, `YYYY-MM-DD`)
- `ticker`
- `exitDate` (`YYYY-MM-DD`)
- `profit`
- `leg`

The program groups legs into one trade per `(ticker, date)` and sums `profit` across legs.

## ORATS data usage

Program uses **one `get_hist_cores` call per sampled ticker** with a comma-delimited fields list:

`tradeDate,iv30d,orHv20d,contango,slope,slopepctile,ivPctile1y`

## Install & run

### 1) Install dependency

```powershell
pip install -e ../orats-client
```

### 2) Put CSV(s) into `backtests/`

### 3) Run (interactive picker)

```powershell
python pre_post_trade_study.py
```

### 4) Run with explicit file

```powershell
python pre_post_trade_study.py --backtest backtests\sample_backtest.csv --sample-size 3
```

### 5) Deterministic vs randomized sampling

Fixed seed (default 42):

```powershell
python pre_post_trade_study.py --backtest backtests\sample_backtest.csv --sample-size 100 --seed 42
```

Randomized (non-deterministic):

```powershell
python pre_post_trade_study.py --backtest backtests\sample_backtest.csv --sample-size 100 --randomize
```

### 6) Optional export/debug

```powershell
python pre_post_trade_study.py --backtest backtests\sample_backtest.csv --sample-size 100 --export-csv outputs\trade_metrics.csv --debug
```

## Expected output format

Program prints:

1. Run metadata (file, sample size, seed mode, ticker count, API call count)
2. Edge-case/data-quality summary
3. **Summary Comparison Table**
   - IV-RV Spread at Entry
   - IV-RV Spread Trend Coefficient
   - Contango at Entry
   - Slope Percentile at Entry
   - IV Percentile at Entry
   - IV-RV Spread Change (10d post)
   - Contango Change (10d post)
4. **Trend Distribution Table**
   - Rising / Flat / Falling counts for winners and losers

If `--export-csv` is provided, a per-trade metrics audit file is written.

## Assumptions / limitations

- Winner/loser segmentation is top/bottom 33 trades when sample size is 100.
  - For other sample sizes, it uses floor thirds (`n//3`, `n//3`, remainder as middle).
- If fewer than 10 pre-entry trading days exist, available days are used and flagged.
- If trade exits before +10 trading days, exit date is used for post-entry metrics.
- Missing ORATS fields produce null derived metrics, excluded from averages.
- Entry/exit dates are aligned to nearest available ORATS trading dates.
- Network/API failures and validation issues are surfaced with user-friendly errors.
