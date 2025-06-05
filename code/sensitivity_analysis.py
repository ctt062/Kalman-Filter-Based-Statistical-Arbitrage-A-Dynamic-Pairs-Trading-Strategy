import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming these functions are in the same 'code' package
from .data_acquisition import fetch_data
from .parameter_estimation import initialize_kalman_filter, run_kalman_filter
from .signal_generation import generate_signals
from .backtesting import backtest_strategy
from .plotting_utils import _sanitize_filename # For consistent filename sanitization

def run_sensitivity_analysis(sens_params_dict, base_config):
    """
    Performs a sensitivity analysis by running the Kalman Filter (KF) pairs
    trading strategy across various combinations of key parameters on
    out-of-sample (OOS) data.

    This function systematically varies parameters such as Z-score entry/exit
    thresholds, Kalman Filter delta, Z-score calculation window, and the use
    of dynamic thresholds. For each parameter combination, it:
    1.  Re-initializes the Kalman Filter on in-sample (IS) data.
    2.  Runs the KF to get beta/alpha estimates on OOS data.
    3.  Generates trading signals on OOS data.
    4.  Backtests the strategy on OOS data.
    5.  Collects performance metrics.

    Finally, it compiles all results into a DataFrame and generates plots
    visualizing the impact of parameter changes on a key metric (e.g., Sharpe Ratio).

    Parameters
    ----------
    sens_params_dict : dict
        A dictionary defining the parameters for the sensitivity analysis.
        Expected keys:
        -   'ticker1', 'ticker2': Names of the assets in the pair.
        -   'sens_entry_z': List of entry Z-score thresholds to test.
        -   'sens_exit_z': List of exit Z-score thresholds to test.
        -   'sens_kf_delta': List of Kalman Filter delta values to test.
        -   'sens_z_window': List of Z-score window sizes to test.
        -   'sens_use_dynamic': List of boolean values (True/False) for using
            dynamic thresholds.
        -   'output_folder': (Optional) Path to save sensitivity plots.
            Defaults to 'graphs_sensitivity'.
    base_config : module
        The main configuration module (`config.py`) of the project. This provides
        base values for parameters not being varied, such as data date ranges,
        split date, transaction costs, initial capital, risk-free rate, and
        settings for dynamic thresholds if used.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents one sensitivity run, and columns
        include the tested parameter values and the resulting performance metrics
        (e.g., 'Entry Z', 'Exit Z', 'KF Delta', 'Sharpe Ratio', 'CAGR', etc.).
        Returns an empty DataFrame if critical errors occur (e.g., data fetching
        fails, or no successful runs).

    Notes
    -----
    -   Data for the specified pair (`ticker1`, `ticker2`) is fetched once and then
        split into IS and OOS periods for all runs.
    -   The Kalman Filter is re-initialized on IS data for each unique `kf_delta`
        to reflect how delta impacts the EM algorithm's estimation of Q.
    -   Only combinations where exit Z-score < entry Z-score are tested.
    -   Error handling is included for individual runs; a run that fails will be
        skipped, and the analysis will continue.
    -   Plots are generated to show the relationship between key parameters and
        the Sharpe Ratio, saved to the specified `output_folder`.
    """

    # --- Parameter Setup ---
    # Get parameter ranges from the input dictionary, or use defaults from base_config
    entry_thr_range = sens_params_dict.get('sens_entry_z', [base_config.ENTRY_THRESHOLD_Z])
    exit_thr_range = sens_params_dict.get('sens_exit_z', [base_config.EXIT_THRESHOLD_Z])
    kf_delta_range = sens_params_dict.get('sens_kf_delta', [base_config.KF_DELTA])
    z_window_range = sens_params_dict.get('sens_z_window', [base_config.Z_SCORE_WINDOW])
    dyn_opts_range = sens_params_dict.get('sens_use_dynamic', [base_config.USE_DYNAMIC_THRESHOLDS])

    # Setup output folder
    output_folder = sens_params_dict.get('output_folder', 'graphs_sensitivity')
    os.makedirs(output_folder, exist_ok=True)

    # --- Prepare Data for Sensitivity Runs ---
    # Check for required ticker information
    if 'ticker1' not in sens_params_dict or 'ticker2' not in sens_params_dict:
        print("Sens Error: ticker1 and ticker2 must be in sens_params_dict.")
        return pd.DataFrame()

    # Fetch data for the specified pair
    pair_data = fetch_data([sens_params_dict['ticker1'], sens_params_dict['ticker2']],
                           base_config.START_DATE, base_config.END_DATE)

    # Validate fetched data
    if pair_data is None or pair_data.isnull().any().any() or pair_data.shape[1] < 2:
        print("Sens Error: Data fetch failed for selected pair or data is incomplete.")
        return pd.DataFrame()

    # Split data into In-Sample (IS) and Out-of-Sample (OOS)
    is_data = pair_data.loc[:base_config.SPLIT_DATE-pd.Timedelta(days=1)].dropna() # Ensure IS ends before split date
    oos_data = pair_data.loc[base_config.SPLIT_DATE:].dropna()

    # Validate data splits
    if is_data.empty or oos_data.empty:
        print("Sens Error: Empty data after IS/OOS split for sensitivity analysis.")
        return pd.DataFrame()

    # Extract Y and X series for IS and OOS periods
    y_is = is_data[sens_params_dict['ticker1']]
    x_is = is_data[sens_params_dict['ticker2']]
    y_oos = oos_data[sens_params_dict['ticker1']]
    x_oos = oos_data[sens_params_dict['ticker2']]

    # Validate price series
    if y_is.empty or x_is.empty or y_oos.empty or x_oos.empty:
        print("Sens Error: Empty price series for one or more assets in IS or OOS periods.")
        return pd.DataFrame()

    # --- Generate Parameter Combinations ---
    results = [] # To store results of each run
    # Create all combinations of parameters, ensuring exit threshold is less than entry threshold
    combos = [(ez, xz, d, w, dyn)
              for ez in entry_thr_range
              for xz in exit_thr_range if xz < ez # Logical constraint for thresholds
              for d in kf_delta_range
              for w in z_window_range
              for dyn in dyn_opts_range]
    n_runs = len(combos)
    print(f"\n--- Starting Sensitivity Analysis ({n_runs} OOS runs planned) ---")

    # --- Iterate Through Parameter Combinations ---
    for i, params in enumerate(combos, 1):
        ez, xz, delta, win, dyn = params # Unpack current parameters
        # Progress indicator
        print(f" Sens Run {i}/{n_runs}: EntryZ={ez:.1f}, ExitZ={xz:.1f}, KFDelta={delta:.1e}, ZWin={win}, DynThr={dyn}", end='\r')

        try:
            # 1. Initialize Kalman Filter (on IS data with current delta)
            kf = initialize_kalman_filter(y_is, x_is, kf_delta=delta)
            if kf is None: # KF initialization failed
                # print(f"\n  Skipping run {i}: KF initialization failed for delta {delta:.1e}")
                continue

            # 2. Run Kalman Filter (on OOS data)
            beta_oos, alpha_oos = run_kalman_filter(kf, y_oos, x_oos)
            if beta_oos.empty or alpha_oos.empty: # KF run failed
                # print(f"\n  Skipping run {i}: KF run failed on OOS data.")
                continue

            # 3. Generate Signals (on OOS data with current parameters)
            signals = generate_signals(y_oos, x_oos, beta_oos, alpha_oos,
                                       z_score_window=win,
                                       entry_threshold_z=ez, exit_threshold_z=xz,
                                       use_dynamic_thresholds=dyn,
                                       dynamic_z_window_factor=base_config.DYNAMIC_Z_WINDOW_FACTOR,
                                       min_z_std_dev=base_config.MIN_Z_STD_DEV,
                                       strategy_name="KF_Sens_Run") # Unique name for signal gen logs
            if signals.empty: # Signal generation failed
                # print(f"\n  Skipping run {i}: Signal generation failed.")
                continue

            # 4. Backtest Strategy (on OOS data)
            _, metrics = backtest_strategy(signals, base_config.INITIAL_CAPITAL,
                                           base_config.FIXED_COST_PER_TRADE,
                                           base_config.VARIABLE_COST_PCT,
                                           base_config.SLIPPAGE_PCT,
                                           base_config.STOP_LOSS_Z_THRESHOLD, # Using base stop-loss for sensitivity
                                           base_config.RISK_FREE_RATE)

            # 5. Store Results
            run_result = {'Dynamic Thresholds': dyn, 'Entry Z': ez, 'Exit Z': xz,
                          'KF Delta': delta, 'Z Window': win, **metrics}
            results.append(run_result)

        except Exception as e_sens:
            # Catch any other unexpected errors during a specific run
            # print(f"\nError in sensitivity run {i} for params {params}: {e_sens}") # Optional: for detailed debugging
            continue # Move to the next parameter combination

    print(f"\n--- Sensitivity Analysis Complete ({len(results)} successful runs out of {n_runs}) ---")

    # --- Process and Plot Results ---
    sens_df = pd.DataFrame(results)

    if not sens_df.empty:
        try:
            print("\nGenerating sensitivity plots...")
            # Sort by Sharpe Ratio for potentially better visual grouping in plots if needed
            sens_sorted_plot = sens_df.sort_values(by='Sharpe Ratio', ascending=False)

            # Plot 1: Sharpe Ratio vs. Entry/Exit Z & Threshold Type
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=sens_sorted_plot, x='Entry Z', y='Sharpe Ratio', hue='Exit Z',
                         style='Dynamic Thresholds', marker='o', errorbar=None, ax=ax1)
            ax1.set_title('Sensitivity: Sharpe Ratio vs. Entry/Exit Z & Threshold Type')
            ax1.grid(True, alpha=0.5)
            ax1.legend(title='Exit Z / Dyn Thr') # Combine legends for hue and style
            filepath1 = os.path.join(output_folder, _sanitize_filename("sensitivity_sharpe_vs_entry_exit_z.png"))
            plt.savefig(filepath1, bbox_inches='tight'); plt.close(fig1)
            print(f"Saved: {filepath1}")

            # Plot 2: Sharpe Ratio vs. Z-Window / KF Delta & Threshold Type
            if 'KF Delta' in sens_sorted_plot.columns:
                # Create a categorical label for KF Delta for better distinction in plot legend
                sens_sorted_plot['KF Delta Label'] = sens_sorted_plot['KF Delta'].apply(lambda x: f"{x:.0e}")

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=sens_sorted_plot, x='Z Window', y='Sharpe Ratio', hue='KF Delta Label',
                             style='Dynamic Thresholds', marker='o', palette='viridis', errorbar=None, ax=ax2)
                ax2.set_title('Sensitivity: Sharpe Ratio vs. Z-Window / KF Delta & Threshold Type')
                ax2.grid(True, alpha=0.5)
                ax2.legend(title='KF Delta / Dyn Thr')
                filepath2 = os.path.join(output_folder, _sanitize_filename("sensitivity_sharpe_vs_z_window_kf_delta.png"))
                plt.savefig(filepath2, bbox_inches='tight'); plt.close(fig2)
                print(f"Saved: {filepath2}")
            else:
                print("Skipping KF Delta sensitivity plot as 'KF Delta' column is missing in results.")

        except Exception as e_plot:
            print(f"Warning: Sensitivity plot generation failed: {e_plot}")
    else:
        print("Sensitivity analysis produced no results to plot.")

    return sens_df