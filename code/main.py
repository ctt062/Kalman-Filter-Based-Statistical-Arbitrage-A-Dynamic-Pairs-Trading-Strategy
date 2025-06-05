import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# --- Start: Add project root to sys.path ---
# This allows the script to be run directly (e.g., python code/main.py from project root)
# and resolves imports correctly.
# Assumes main.py is in your_project_root/code/
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# --- End: Add project root to sys.path ---

# Import from local modules within the 'code' package using absolute paths from 'code'
# This requires 'code' to be findable, which the sys.path modification above ensures.
from code import config  # Import the whole config module
from code.data_acquisition import fetch_data
from code.pair_selection import find_cointegrated_pair
from code.parameter_estimation import (
    calculate_static_ols,
    initialize_kalman_filter,
    run_kalman_filter
)
from code.signal_generation import generate_signals
from code.backtesting import backtest_strategy
from code.performance_metrics import calculate_metrics
from code.plotting_utils import (
    plot_pair_selection_diagnostics,
    # plot_in_sample_static_results, # Optional: uncomment if you want this plot
    plot_in_sample_kf_results,
    plot_out_of_sample_kf_results,
    plot_in_sample_cumulative_returns,
    plot_out_of_sample_cumulative_returns
)
from code.sensitivity_analysis import run_sensitivity_analysis


def main():
    """
    Main function to execute the Kalman Filter pairs trading strategy pipeline.

    This function orchestrates the entire process, including:
    1.  Setting up global configurations (warnings, plotting style) and output directories.
    2.  Loading and printing a summary of key strategy parameters from `config.py`.
    3.  Fetching and preparing historical price data for specified tickers and benchmark.
        This includes handling NaNs and splitting data into in-sample (IS) and
        out-of-sample (OOS) periods.
    4.  Selecting a cointegrated pair of assets from the IS data using statistical tests
        and optional half-life filtering. Diagnostic plots for pair selection are generated.
    5.  Estimating model parameters:
        - Static Ordinary Least Squares (OLS) regression on IS data.
        - Kalman Filter initialization and application to derive dynamic hedge ratios
          (beta) and intercepts (alpha) for both IS and OOS periods.
    6.  Generating trading signals based on the calculated spread and z-scores for:
        - Static OLS (IS and OOS).
        - Kalman Filter (IS and OOS), with support for dynamic z-score thresholds.
    7.  Backtesting the trading strategies using the generated signals. This step
        considers initial capital, transaction costs (fixed and variable), slippage,
        and optional stop-loss mechanisms.
    8.  Calculating and reporting performance metrics (e.g., CAGR, Sharpe Ratio, Max Drawdown,
        Calmar Ratio, Number of Trades). A summary table compares the OOS performance
        of the Kalman Filter strategy against benchmarks (Static OLS, Buy & Hold individual legs,
        and a market benchmark if specified).
    9.  Generating and saving various plots to visualize:
        - In-sample Kalman Filter results (prices, spread, z-score, signals, equity curve).
        - Out-of-sample Kalman Filter results (similar to IS, plus benchmark comparison).
        - Cumulative returns for IS and OOS periods for all strategies and benchmarks.
    10. Optionally, performing a sensitivity analysis on key OOS strategy parameters
        (e.g., entry/exit z-score thresholds, Kalman Filter delta, z-score window size,
        dynamic threshold usage) to assess robustness. Results are printed and plots saved.

    The execution flow is sequential, with early exits and informative messages
    if critical steps (like data fetching, pair selection, or Kalman Filter
    initialization) fail or result in insufficient data. All configurations are
    drawn from the `config` module. Output, including logs and plots, is directed
    accordingly.
    """
    # --- Global Settings & Configuration ---
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create graphs folder if it doesn't exist
    # Assumes main.py is run from the project root (e.g., `python code/main.py`)
    # or that the 'graphs' folder should be relative to the current working directory.
    # If _project_root is used, graphs folder will be in project root.
    graphs_dir_path = os.path.join(_project_root, config.GRAPHS_OUTPUT_FOLDER)
    os.makedirs(graphs_dir_path, exist_ok=True)
    print(f"Graphs will be saved to: {os.path.abspath(graphs_dir_path)}")


    # Print Configuration Summary
    print("--- Configuration ---")
    print(f"Tickers for pair search: {config.TICKER_LIST}")
    print(f"Benchmark Ticker: {config.BENCHMARK_TICKER}")
    print(f"Dates: {config.START_DATE} to {config.END_DATE} (Split: {config.SPLIT_DATE_STR})")
    print(f"KF Params: Delta={config.KF_DELTA:.1e}, Z-Win={config.Z_SCORE_WINDOW}")
    print(f"Thresholds: Entry Z={config.ENTRY_THRESHOLD_Z}, Exit Z={config.EXIT_THRESHOLD_Z}, Dynamic={config.USE_DYNAMIC_THRESHOLDS}")
    if config.USE_DYNAMIC_THRESHOLDS:
        print(f"  Dynamic Threshold Config: WindowFactor={config.DYNAMIC_Z_WINDOW_FACTOR}, MinStdDev={config.MIN_Z_STD_DEV}")
    print(f"StopLoss Z-Delta: {config.STOP_LOSS_Z_THRESHOLD if config.STOP_LOSS_Z_THRESHOLD is not None else 'Disabled'}")
    print(f"HL Filter: Enabled={config.ENABLE_HALF_LIFE_FILTER} ({config.MIN_HALF_LIFE_DAYS}-{config.MAX_HALF_LIFE_DAYS} days for valid pair)")
    print(f"Backtest: Initial Cap=${config.INITIAL_CAPITAL:,.0f}, Fixed Cost=${config.FIXED_COST_PER_TRADE:.2f}/trade/leg, Var Cost={config.VARIABLE_COST_PCT:.4%}, Slippage={config.SLIPPAGE_PCT:.4%}")
    print(f"Risk-Free Rate for Sharpe: {config.RISK_FREE_RATE:.2%}")
    print("-" * 20)

    # --- Step 1: Fetch and Prepare Data ---
    all_tickers_to_fetch = list(set(config.TICKER_LIST + ([config.BENCHMARK_TICKER] if config.BENCHMARK_TICKER else [])))
    full_data_raw = fetch_data(all_tickers_to_fetch, config.START_DATE, config.END_DATE)

    if full_data_raw is None or full_data_raw.empty:
        print("Exiting: Data fetching failed or returned no data.")
        return

    full_data = full_data_raw.copy()
    nan_cols_initial = full_data.columns[full_data.isnull().any()].tolist()
    if nan_cols_initial:
        print(f"Warning: Dropping columns with NaNs after initial fetch & fill: {nan_cols_initial}")
        full_data.dropna(axis=1, how='any', inplace=True)

    if full_data.empty:
        print("Exiting: No columns left after NaN drop from fetched data.")
        return

    print(f"\nUsing tickers with complete data: {full_data.columns.tolist()}")

    # Ensure benchmark and pair candidates are present
    bench_data = pd.DataFrame()
    if config.BENCHMARK_TICKER and config.BENCHMARK_TICKER in full_data.columns:
        bench_data = full_data[[config.BENCHMARK_TICKER]]
    elif config.BENCHMARK_TICKER:
        print(f"Warning: Benchmark ticker {config.BENCHMARK_TICKER} not available in fetched data.")

    pair_candidate_tickers = [t for t in config.TICKER_LIST if t in full_data.columns]
    if len(pair_candidate_tickers) < 2:
        print("Exiting: Need at least 2 pair candidate tickers with complete data.")
        return

    pair_data_full_candidates = full_data[pair_candidate_tickers]

    # Split Data
    is_data_all_candidates = pair_data_full_candidates.loc[:config.SPLIT_DATE-pd.Timedelta(days=1)].copy() # Ensure split is exclusive for IS
    oos_data_all_candidates = pair_data_full_candidates.loc[config.SPLIT_DATE:].copy()


    if is_data_all_candidates.isnull().any().any():
        print("Critical Warning: NaNs found in In-Sample candidate data post-split. This should not happen.")
        is_data_all_candidates.dropna(axis=1, inplace=True)
        pair_candidate_tickers = list(is_data_all_candidates.columns)
        if len(pair_candidate_tickers) < 2:
             print("Exiting: Not enough candidates after In-Sample NaN drop.")
             return

    if is_data_all_candidates.empty or oos_data_all_candidates.empty:
        print("Exiting: Data split resulted in an empty period for In-Sample or Out-of-Sample candidates.")
        return

    # --- Step 2: Find Cointegrated Pair & Plot Diagnostics ---
    t1, t2, pair_info = find_cointegrated_pair(
        is_data_all_candidates,
        significance_level=0.05,
        min_half_life=config.MIN_HALF_LIFE_DAYS,
        max_half_life=config.MAX_HALF_LIFE_DAYS,
        enable_half_life_filter=config.ENABLE_HALF_LIFE_FILTER
    )
    plot_pair_selection_diagnostics(is_data_all_candidates, pair_info, graphs_dir_path)

    if t1 is None or t2 is None:
        print("Exiting: No suitable cointegrated pair found.")
        return

    # --- Step 3: Prepare Final Pair Data & Run KF ---
    if not (t1 in is_data_all_candidates.columns and t2 in is_data_all_candidates.columns and \
            t1 in oos_data_all_candidates.columns and t2 in oos_data_all_candidates.columns):
        print(f"Exiting: Selected pair {t1}/{t2} missing from processed IS or OOS candidate data pools.")
        return

    is_pair_data = is_data_all_candidates[[t1, t2]].dropna()
    oos_pair_data = oos_data_all_candidates[[t1, t2]].dropna()

    if is_pair_data.empty or oos_pair_data.empty:
        print(f"Exiting: Data missing for selected pair {t1}/{t2} after final processing and date splitting.")
        return

    y_is, x_is = is_pair_data[t1], is_pair_data[t2]
    y_oos, x_oos = oos_pair_data[t1], oos_pair_data[t2]

    static_beta, static_alpha, _ = calculate_static_ols(y_is, x_is)
    static_params_valid = not (pd.isna(static_beta) or pd.isna(static_alpha))

    kf_model = initialize_kalman_filter(y_is, x_is, config.KF_DELTA)
    if kf_model is None:
        print("Exiting: Kalman Filter initialization failed.")
        return

    beta_kf_in, alpha_kf_in = run_kalman_filter(kf_model, y_is, x_is)
    beta_kf_oos, alpha_kf_oos = run_kalman_filter(kf_model, y_oos, x_oos)

    if beta_kf_in.empty or alpha_kf_in.empty or beta_kf_oos.empty or alpha_kf_oos.empty:
        print("Exiting: Kalman Filter filtering step failed for In-Sample or Out-of-Sample period.")
        return

    # --- Step 4: Generate Signals ---
    print("\n--- Generating Trading Signals ---")
    sigs_static_is, sigs_static_oos = pd.DataFrame(), pd.DataFrame()
    if static_params_valid:
        sigs_static_is = generate_signals(y_is, x_is, static_beta, static_alpha,
                                          config.Z_SCORE_WINDOW, config.ENTRY_THRESHOLD_Z, config.EXIT_THRESHOLD_Z,
                                          use_dynamic_thresholds=False, strategy_name="Static_IS")
        sigs_static_oos = generate_signals(y_oos, x_oos, static_beta, static_alpha,
                                           config.Z_SCORE_WINDOW, config.ENTRY_THRESHOLD_Z, config.EXIT_THRESHOLD_Z,
                                           use_dynamic_thresholds=False, strategy_name="Static_OOS")

    sigs_kf_is = generate_signals(y_is, x_is, beta_kf_in, alpha_kf_in,
                                  config.Z_SCORE_WINDOW, config.ENTRY_THRESHOLD_Z, config.EXIT_THRESHOLD_Z,
                                  config.USE_DYNAMIC_THRESHOLDS, config.DYNAMIC_Z_WINDOW_FACTOR,
                                  config.MIN_Z_STD_DEV, strategy_name="KF_IS")
    sigs_kf_oos = generate_signals(y_oos, x_oos, beta_kf_oos, alpha_kf_oos,
                                   config.Z_SCORE_WINDOW, config.ENTRY_THRESHOLD_Z, config.EXIT_THRESHOLD_Z,
                                   config.USE_DYNAMIC_THRESHOLDS, config.DYNAMIC_Z_WINDOW_FACTOR,
                                   config.MIN_Z_STD_DEV, strategy_name="KF_OOS")

    # --- Step 5: Run Backtests ---
    print("\n--- Running Backtests ---")
    empty_metrics_init = calculate_metrics(pd.DataFrame(),0, config.RISK_FREE_RATE)
    port_static_is, met_static_is = pd.DataFrame(), empty_metrics_init.copy()
    port_static_oos, met_static_oos = pd.DataFrame(), empty_metrics_init.copy()

    if static_params_valid:
        if not sigs_static_is.empty:
            port_static_is, met_static_is = backtest_strategy(sigs_static_is, config.INITIAL_CAPITAL,
                                                              config.FIXED_COST_PER_TRADE, config.VARIABLE_COST_PCT,
                                                              config.SLIPPAGE_PCT, config.STOP_LOSS_Z_THRESHOLD, config.RISK_FREE_RATE)
        if not sigs_static_oos.empty:
            port_static_oos, met_static_oos = backtest_strategy(sigs_static_oos, config.INITIAL_CAPITAL,
                                                                config.FIXED_COST_PER_TRADE, config.VARIABLE_COST_PCT,
                                                                config.SLIPPAGE_PCT, config.STOP_LOSS_Z_THRESHOLD, config.RISK_FREE_RATE)

    port_kf_is, met_kf_is = pd.DataFrame(), empty_metrics_init.copy()
    if not sigs_kf_is.empty:
        port_kf_is, met_kf_is = backtest_strategy(sigs_kf_is, config.INITIAL_CAPITAL,
                                                  config.FIXED_COST_PER_TRADE, config.VARIABLE_COST_PCT,
                                                  config.SLIPPAGE_PCT, config.STOP_LOSS_Z_THRESHOLD, config.RISK_FREE_RATE)

    port_kf_oos, met_kf_oos = pd.DataFrame(), empty_metrics_init.copy()
    if not sigs_kf_oos.empty:
        port_kf_oos, met_kf_oos = backtest_strategy(sigs_kf_oos, config.INITIAL_CAPITAL,
                                                    config.FIXED_COST_PER_TRADE, config.VARIABLE_COST_PCT,
                                                    config.SLIPPAGE_PCT, config.STOP_LOSS_Z_THRESHOLD, config.RISK_FREE_RATE)

    # --- Step 6: Summarize OOS Performance ---
    print("\n--- OOS Performance Analysis ---")
    benchmark_results_oos = {}
    benchmark_portfolios_oos = pd.DataFrame(index=oos_pair_data.index if not oos_pair_data.empty else None)

    if not bench_data.empty and config.BENCHMARK_TICKER:
        bench_data_oos_aligned = bench_data.reindex(benchmark_portfolios_oos.index).dropna()
        if not bench_data_oos_aligned.empty and config.BENCHMARK_TICKER in bench_data_oos_aligned.columns:
            temp_port_df = pd.DataFrame(index=bench_data_oos_aligned.index)
            if not bench_data_oos_aligned[config.BENCHMARK_TICKER].empty:
                temp_port_df['total'] = config.INITIAL_CAPITAL * (bench_data_oos_aligned[config.BENCHMARK_TICKER] / bench_data_oos_aligned[config.BENCHMARK_TICKER].iloc[0])
                temp_port_df['trades'] = 0
                benchmark_results_oos[config.BENCHMARK_TICKER] = calculate_metrics(temp_port_df, config.INITIAL_CAPITAL, config.RISK_FREE_RATE)
                benchmark_portfolios_oos[config.BENCHMARK_TICKER] = temp_port_df['total']

    for ticker_leg, data_leg_oos in [(t1, y_oos), (t2, x_oos)]:
        if not data_leg_oos.empty:
            temp_port_df = pd.DataFrame(index=data_leg_oos.index)
            temp_port_df['total'] = config.INITIAL_CAPITAL * (data_leg_oos / data_leg_oos.iloc[0])
            temp_port_df['trades'] = 0
            benchmark_results_oos[f"B&H {ticker_leg}"] = calculate_metrics(temp_port_df, config.INITIAL_CAPITAL, config.RISK_FREE_RATE)
            benchmark_portfolios_oos[f"B&H {ticker_leg}"] = temp_port_df['total']

    if static_params_valid and not port_static_oos.empty:
        benchmark_results_oos['Static OLS'] = met_static_oos
        benchmark_portfolios_oos['Static OLS'] = port_static_oos['total']

    oos_summary_dict = {"Kalman Filter": met_kf_oos}
    oos_summary_dict.update({f"Bench: {k}": v for k, v in benchmark_results_oos.items()})

    summary_df = pd.DataFrame(oos_summary_dict).T
    display_columns = ['CAGR', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Number of Trades']
    for col in display_columns:
        if col not in summary_df.columns:
            summary_df[col] = 0.0

    print("\n--- FINAL OOS PERFORMANCE SUMMARY ---")
    print(summary_df[display_columns].round(3))


    # --- Step 7: Generate Strategy Performance Plots ---
    print("\n--- Generating Strategy Performance Plots ---")
    if pair_info:
        is_returns_dict = {}
        oos_returns_dict = {}

        if not port_kf_is.empty: is_returns_dict['Kalman Filter'] = port_kf_is['returns']
        if static_params_valid and not port_static_is.empty: is_returns_dict['Static OLS'] = port_static_is['returns']
        if not y_is.empty: is_returns_dict[f'B&H {t1}'] = y_is.pct_change().fillna(0)
        if not x_is.empty: is_returns_dict[f'B&H {t2}'] = x_is.pct_change().fillna(0)
        if not bench_data.empty and config.BENCHMARK_TICKER:
            bench_is_aligned = bench_data.loc[:config.SPLIT_DATE-pd.Timedelta(days=1)].reindex(is_pair_data.index).ffill().bfill()
            if not bench_is_aligned.empty and config.BENCHMARK_TICKER in bench_is_aligned.columns and not bench_is_aligned[config.BENCHMARK_TICKER].empty:
                is_returns_dict[f'B&H {config.BENCHMARK_TICKER}'] = bench_is_aligned[config.BENCHMARK_TICKER].pct_change().fillna(0)

        if not port_kf_oos.empty: oos_returns_dict['Kalman Filter'] = port_kf_oos['returns']
        if static_params_valid and not port_static_oos.empty: oos_returns_dict['Static OLS'] = port_static_oos['returns']
        if not y_oos.empty: oos_returns_dict[f'B&H {t1}'] = y_oos.pct_change().fillna(0)
        if not x_oos.empty: oos_returns_dict[f'B&H {t2}'] = x_oos.pct_change().fillna(0)
        if not bench_data.empty and config.BENCHMARK_TICKER:
            bench_oos_aligned = bench_data.loc[config.SPLIT_DATE:].reindex(oos_pair_data.index).ffill().bfill()
            if not bench_oos_aligned.empty and config.BENCHMARK_TICKER in bench_oos_aligned.columns and not bench_oos_aligned[config.BENCHMARK_TICKER].empty:
                oos_returns_dict[f'B&H {config.BENCHMARK_TICKER}'] = bench_oos_aligned[config.BENCHMARK_TICKER].pct_change().fillna(0)

        if not port_kf_is.empty:
            plot_in_sample_kf_results(is_pair_data, sigs_kf_is, port_kf_is, t1, t2, config.SPLIT_DATE,
                                      config.ENTRY_THRESHOLD_Z, config.EXIT_THRESHOLD_Z, config.INITIAL_CAPITAL,
                                      pair_info, config.USE_DYNAMIC_THRESHOLDS, graphs_dir_path)
        if not port_kf_oos.empty:
            plot_out_of_sample_kf_results(oos_pair_data, sigs_kf_oos, port_kf_oos, benchmark_portfolios_oos,
                                          t1, t2, config.SPLIT_DATE, config.ENTRY_THRESHOLD_Z, config.EXIT_THRESHOLD_Z,
                                          config.STOP_LOSS_Z_THRESHOLD, pair_info, config.USE_DYNAMIC_THRESHOLDS,
                                          config.BENCHMARK_TICKER, graphs_dir_path)

        plot_in_sample_cumulative_returns(is_returns_dict, config.INITIAL_CAPITAL, config.SPLIT_DATE, pair_info, graphs_dir_path)
        plot_out_of_sample_cumulative_returns(oos_returns_dict, config.INITIAL_CAPITAL, config.SPLIT_DATE, pair_info, graphs_dir_path)
    else:
        print("Skipping strategy performance plots as no valid pair was selected or processed.")

    # --- Step 8: Run Sensitivity Analysis (Optional) ---
    run_sens_analysis = True
    if run_sens_analysis and t1 and t2:
        print("\n--- Sensitivity Analysis (on OOS Performance of selected pair) ---")
        sensitivity_params_config = {
            'ticker1': t1,
            'ticker2': t2,
            'sens_entry_z': config.SENS_ENTRY_Z_THRESHOLDS,
            'sens_exit_z': config.SENS_EXIT_Z_THRESHOLDS,
            'sens_kf_delta': config.SENS_KF_DELTAS,
            'sens_z_window': config.SENS_Z_SCORE_WINDOWS,
            'sens_use_dynamic': config.SENS_USE_DYNAMIC_THRESHOLDS_OPTIONS,
            'output_folder': os.path.join(graphs_dir_path, "sensitivity_plots")
        }
        sens_df = run_sensitivity_analysis(sensitivity_params_config, config) # Pass base config module

        if not sens_df.empty:
            print("\n--- Sensitivity Analysis Results (Top 15 by Sharpe Ratio) ---")
            original_float_format = pd.options.display.float_format
            pd.options.display.float_format = '{:.3f}'.format
            sens_sorted = sens_df.sort_values(by='Sharpe Ratio', ascending=False).reset_index(drop=True)
            formatters = {col: (lambda x: f"{x:.3f}") for col in sens_sorted.columns if sens_sorted[col].dtype == 'float64'}
            if 'KF Delta' in sens_sorted.columns: formatters['KF Delta'] = lambda x: f"{x:.1e}"
            if 'Number of Trades' in sens_sorted.columns: formatters['Number of Trades'] = lambda x: f"{int(x):d}"
            sens_display_cols = ['Dynamic Thresholds', 'Entry Z', 'Exit Z', 'KF Delta', 'Z Window',
                                 'Sharpe Ratio', 'CAGR', 'Max Drawdown', 'Number of Trades']
            valid_sens_cols = [col for col in sens_display_cols if col in sens_sorted.columns]
            print(sens_sorted[valid_sens_cols].head(15).to_string(formatters=formatters if formatters else None))
            pd.options.display.float_format = original_float_format
        else:
            print("\nSensitivity analysis completed but produced no results.")
    elif run_sens_analysis:
        print("\nSkipping Sensitivity Analysis as no valid pair was selected for the main run.")

    print("\n--- Project Execution Finished ---")

if __name__ == "__main__":
    main()