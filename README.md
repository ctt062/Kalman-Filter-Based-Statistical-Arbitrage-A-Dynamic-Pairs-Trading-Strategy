# 💹 Kalman Filter-Based Statistical Arbitrage Strategy

This project implements a dynamic pairs trading strategy using **Kalman Filters** to estimate time-varying hedge ratios in cointegrated stock pairs from the **Consumer Staples** sector. Developed for the **Data-Driven Portfolio Optimization (IEDA3180)** course at **HKUST**.

## 🧠 Project Objectives
- Detect mean-reverting opportunities in stock pairs
- Estimate dynamic hedge ratios using Kalman Filters
- Backtest and compare strategy performance against benchmarks

## ⚙️ Techniques & Tools
- **Languages & Libraries:** Python, pykalman, statsmodels, matplotlib, seaborn, Yahoo Finance API
- **Pair Selection:**
  - Engle-Granger test
  - Augmented Dickey-Fuller (ADF)
  - Half-life analysis
- **Signal Generation:**
  - Rolling Z-score of Kalman-estimated spread
  - Entry: ±2.0 | Exit: ±0.5 thresholds

## 📊 Strategy Performance (Out-of-Sample: 2022–2024)
- **CAGR:** 6.7%
- **Sharpe Ratio:** 1.26
- **Max Drawdown:** -3.5%
- Outperformed SPY and static OLS strategies in both returns and risk-adjusted metrics

## 📂 Files
- `notebooks/`: Backtesting and signal generation scripts
- `report.pdf`: Final research report
- `slides.pdf`: Presentation slides

## 🧰 Skills Demonstrated
Quantitative Research · Kalman Filtering · Portfolio Optimization · Time Series Analysis · Backtesting

## 🔗 Connect
[LinkedIn]([https://www.linkedin.com/in/your-profile](http://www.linkedin.com/in/tin-tak-chong)) • [Email](mailto:chongtt062@gmail.com)
