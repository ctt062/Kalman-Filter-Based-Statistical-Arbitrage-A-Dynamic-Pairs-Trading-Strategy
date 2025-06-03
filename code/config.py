import pandas as pd
import numpy as np

# --- Project Configuration ---

# Ticker selection and date ranges
TICKER_LIST = ['PG', 'CL', 'KMB', 'PEP', 'KO', 'COST', 'WMT', 'GIS', 'CHD'] # Consumer Staples
BENCHMARK_TICKER = 'SPY'
START_DATE = "2017-01-01"
SPLIT_DATE_STR = "2022-01-01"
END_DATE = "2023-12-31"

SPLIT_DATE = pd.to_datetime(SPLIT_DATE_STR) # Derived

# Kalman Filter and Trading Strategy Parameters
# These reflect potentially optimal parameters from a previous sensitivity run or defaults
KF_DELTA = 1e-6
Z_SCORE_WINDOW = 60
ENTRY_THRESHOLD_Z = 2.0
EXIT_THRESHOLD_Z = 0.5
STOP_LOSS_Z_THRESHOLD = 3.0 # Set to None to disable stop-loss
USE_DYNAMIC_THRESHOLDS = False # True to use dynamic Z-score thresholds, False for fixed

# Half-Life Filter Configuration for Pair Selection
ENABLE_HALF_LIFE_FILTER = False # Set to True to activate filter during pair selection
MIN_HALF_LIFE_DAYS = 5
MAX_HALF_LIFE_DAYS = 100

# Dynamic Threshold Configuration (if USE_DYNAMIC_THRESHOLDS = True)
DYNAMIC_Z_WINDOW_FACTOR = 1.5 # Multiplier for Z-score window to calculate rolling std of Z
MIN_Z_STD_DEV = 0.1           # Minimum standard deviation for dynamic Z-score thresholds

# Backtesting Parameters
INITIAL_CAPITAL = 100000.0
FIXED_COST_PER_TRADE = 1.00  # Cost per leg, so a round trip (open+close) for two assets involves 4x this if applied per asset trade
VARIABLE_COST_PCT = 0.0005 # Percentage of trade value
SLIPPAGE_PCT = 0.0005      # Percentage of price

# Sensitivity Analysis Parameter Ranges (can be overridden in main.py if needed)
SENS_ENTRY_Z_THRESHOLDS = [1.5, 2.0, 2.5]
SENS_EXIT_Z_THRESHOLDS = [0.0, 0.5, 1.0] # Ensure exit < entry
SENS_KF_DELTAS = [1e-4, 1e-5, 1e-6]
SENS_Z_SCORE_WINDOWS = [30, 45, 60]
SENS_USE_DYNAMIC_THRESHOLDS_OPTIONS = [False, True]

# Risk-free rate for Sharpe Ratio (annualized)
RISK_FREE_RATE = 0.0

# Plotting output folder (relative to script execution directory)
# This will be created by main.py
GRAPHS_OUTPUT_FOLDER = "graphs"