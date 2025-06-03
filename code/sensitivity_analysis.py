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
    """ Runs the KF strategy backtest across ranges of parameters on OOS data. """
    
    entry_thr_range = sens_params_dict.get('sens_entry_z', [base_config.ENTRY_THRESHOLD_Z])
    exit_thr_range = sens_params_dict.get('sens_exit_z', [base_config.EXIT_THRESHOLD_Z])
    kf_delta_range = sens_params_dict.get('sens_kf_delta', [base_config.KF_DELTA])
    z_window_range = sens_params_dict.get('sens_z_window', [base_config.Z_SCORE_WINDOW])
    dyn_opts_range = sens_params_dict.get('sens_use_dynamic', [base_config.USE_DYNAMIC_THRESHOLDS])
    
    output_folder = sens_params_dict.get('output_folder', 'graphs_sensitivity') # Default output folder
    os.makedirs(output_folder, exist_ok=True)

    results = []
    combos = [(ez, xz, d, w, dyn) for ez in entry_thr_range for xz in exit_thr_range if xz < ez
              for d in kf_delta_range for w in z_window_range for dyn in dyn_opts_range]
    n_runs = len(combos)
    print(f"\n--- Starting Sensitivity Analysis ({n_runs} OOS runs) ---")

    if 'ticker1' not in sens_params_dict or 'ticker2' not in sens_params_dict:
        print("Sens Error: ticker1 and ticker2 must be in sens_params_dict."); return pd.DataFrame()
    
    pair_data = fetch_data([sens_params_dict['ticker1'], sens_params_dict['ticker2']], 
                           base_config.START_DATE, base_config.END_DATE)
    if pair_data is None or pair_data.isnull().any().any() or pair_data.shape[1]<2:
        print("Sens Error: Data fetch failed for selected pair."); return pd.DataFrame()
    
    is_data = pair_data[:base_config.SPLIT_DATE].dropna()
    oos_data = pair_data[base_config.SPLIT_DATE:].dropna()

    if is_data.empty or oos_data.empty:
        print("Sens Error: Empty data split for sensitivity analysis."); return pd.DataFrame()
    
    y_is = is_data[sens_params_dict['ticker1']]; x_is = is_data[sens_params_dict['ticker2']]
    y_oos = oos_data[sens_params_dict['ticker1']]; x_oos = oos_data[sens_params_dict['ticker2']]

    if y_is.empty or x_is.empty or y_oos.empty or x_oos.empty:
        print("Sens Error: Empty price series for sensitivity."); return pd.DataFrame()

    for i, params in enumerate(combos, 1):
        ez, xz, delta, win, dyn = params
        print(f" Sens {i}/{n_runs}: E={ez:.1f},X={xz:.1f},D={delta:.1e},W={win},Dyn={dyn}", end='\r')
        try:
            kf = initialize_kalman_filter(y_is, x_is, kf_delta=delta)
            if kf is None: continue
            beta_oos, alpha_oos = run_kalman_filter(kf, y_oos, x_oos)
            if beta_oos.empty: continue

            signals = generate_signals(y_oos, x_oos, beta_oos, alpha_oos, z_score_window=win,
                                       entry_threshold_z=ez, exit_threshold_z=xz, use_dynamic_thresholds=dyn,
                                       dynamic_z_window_factor=base_config.DYNAMIC_Z_WINDOW_FACTOR,
                                       min_z_std_dev=base_config.MIN_Z_STD_DEV, strategy_name="KF_Sens")
            if signals.empty: continue

            _, metrics = backtest_strategy(signals, base_config.INITIAL_CAPITAL, 
                                           base_config.FIXED_COST_PER_TRADE,
                                           base_config.VARIABLE_COST_PCT, base_config.SLIPPAGE_PCT,
                                           base_config.STOP_LOSS_Z_THRESHOLD, base_config.RISK_FREE_RATE)

            results.append({'Dynamic Thresholds': dyn, 'Entry Z': ez, 'Exit Z': xz, 'KF Delta': delta,
                            'Z Window': win, **metrics})
        except Exception as e_sens:
            # print(f"\nError in sens run {i} for params {params}: {e_sens}") # Optional detailed debug
            continue

    print(f"\n--- Sensitivity Analysis Complete ({len(results)} successful runs out of {n_runs}) ---")
    
    sens_df = pd.DataFrame(results)
    if not sens_df.empty:
        # Sensitivity Plots
        try:
            print("\nGenerating sensitivity plots...")
            sens_sorted_plot = sens_df.sort_values(by='Sharpe Ratio', ascending=False)

            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=sens_sorted_plot, x='Entry Z', y='Sharpe Ratio', hue='Exit Z', 
                         style='Dynamic Thresholds', marker='o', errorbar=None, ax=ax1)
            ax1.set_title('Sensitivity: Sharpe Ratio vs. Entry/Exit Z & Threshold Type')
            ax1.grid(True, alpha=0.5); ax1.legend(title='Exit Z / Dyn Thr')
            filepath1 = os.path.join(output_folder, _sanitize_filename("sensitivity_sharpe_vs_entry_exit_z.png"))
            plt.savefig(filepath1, bbox_inches='tight'); plt.close(fig1)
            print(f"Saved: {filepath1}")

            # Create a label for KF Delta for better plotting
            if 'KF Delta' in sens_sorted_plot.columns:
                sens_sorted_plot['KF Delta Label'] = sens_sorted_plot['KF Delta'].apply(lambda x: f"{x:.0e}")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=sens_sorted_plot, x='Z Window', y='Sharpe Ratio', hue='KF Delta Label', 
                             style='Dynamic Thresholds', marker='o', palette='viridis', errorbar=None, ax=ax2)
                ax2.set_title('Sensitivity: Sharpe Ratio vs. Z-Window / KF Delta & Threshold Type')
                ax2.grid(True, alpha=0.5); ax2.legend(title='KF Delta / Dyn Thr')
                filepath2 = os.path.join(output_folder, _sanitize_filename("sensitivity_sharpe_vs_z_window_kf_delta.png"))
                plt.savefig(filepath2, bbox_inches='tight'); plt.close(fig2)
                print(f"Saved: {filepath2}")
            else:
                print("Skipping KF Delta sensitivity plot as 'KF Delta' column is missing.")

        except Exception as e_plot:
            print(f"Warning: Sensitivity plot generation failed: {e_plot}")
    else:
        print("Sensitivity analysis produced no results to plot.")
        
    return sens_df