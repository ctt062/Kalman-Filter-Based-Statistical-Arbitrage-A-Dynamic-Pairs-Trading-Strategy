import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Helper to generate safe filenames
def _sanitize_filename(name):
    """
    Sanitizes a string to be used as a filename by replacing non-alphanumeric
    characters with underscores.

    Parameters
    ----------
    name : str
        The input string to sanitize.

    Returns
    -------
    str
        The sanitized string suitable for use as a filename.
    """
    # Join characters if alphanumeric, otherwise replace with underscore
    return "".join([c if c.isalnum() else "_" for c in str(name)])


def plot_pair_selection_diagnostics(in_sample_data_all_pairs, pair_info, output_folder):
    """
    Generates and saves diagnostic plots related to the pair selection process.

    This includes:
    1.  A heatmap of the correlation matrix for all candidate assets in the in-sample period.
    2.  If a pair is selected (`pair_info` is provided), a plot of its in-sample spread,
        calculated using the static OLS parameters derived during pair selection.
        This plot also includes the mean of the spread, ADF p-value, and half-life.

    Parameters
    ----------
    in_sample_data_all_pairs : pd.DataFrame
        DataFrame containing in-sample price data for all potential pair candidates.
        Columns are ticker names, index is DatetimeIndex.
    pair_info : dict or None
        A dictionary containing information about the selected pair, including
        'ticker1', 'ticker2', 'ols_beta', 'ols_alpha', 'adf_p_value', 'half_life_days'.
        If None, only the correlation matrix is plotted (if data allows).
    output_folder : str
        The directory path where the generated plots will be saved.
        The folder will be created if it doesn't exist.
    """
    print("\n--- Generating Pair Selection Diagnostic Plots ---")
    os.makedirs(output_folder, exist_ok=True)

    # --- 1. Correlation Matrix Heatmap ---
    # Check if data is available and has enough columns for correlation
    if not in_sample_data_all_pairs.empty and in_sample_data_all_pairs.shape[1] > 1:
        correlation_matrix = in_sample_data_all_pairs.corr()

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('In-Sample Price Correlation Matrix (Pair Candidates)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        filepath = os.path.join(output_folder, "pair_selection_correlation_matrix.png")
        plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
        print(f"Saved: {filepath}")
    else:
        print("Skipping correlation heatmap (not enough data or candidates).")

    # Add a blank line for separation
    print()

    # --- 2. In-Sample Spread Plot for Selected Pair ---
    if pair_info:
        # Extract pair information
        t1, t2 = pair_info['ticker1'], pair_info['ticker2']
        ols_beta, ols_alpha = pair_info['ols_beta'], pair_info['ols_alpha']
        adf_p, hl = pair_info['adf_p_value'], pair_info['half_life_days']

        # Check if selected tickers are present in the data
        if t1 in in_sample_data_all_pairs.columns and t2 in in_sample_data_all_pairs.columns:
            # Calculate spread using OLS parameters
            spread = in_sample_data_all_pairs[t1] - (ols_beta * in_sample_data_all_pairs[t2] + ols_alpha)

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            spread.plot(ax=ax, title=f'In-Sample Spread: {t1} - ({ols_beta:.2f}*{t2} + {ols_alpha:.2f})', lw=1.5)
            ax.axhline(spread.mean(), color='red', linestyle='--', lw=1, label=f'Mean ({spread.mean():.3f})')
            ax.set_ylabel("Spread Value"); ax.set_xlabel("Date")
            ax.grid(True, alpha=0.4); ax.legend()

            # Add statistics text to the plot
            stats_text = f'ADF p: {adf_p:.3f}\nHalf-Life: {hl:.1f}d'
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                     verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            plt.tight_layout()

            # Save plot
            filename = _sanitize_filename(f"pair_selection_IS_spread_{t1}_vs_{t2}.png")
            filepath = os.path.join(output_folder, filename)
            plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {filepath}")
        else:
            print(f"Could not plot spread for selected pair {t1}/{t2} - data missing.")
    else:
        print("Skipping selected pair spread plot (no pair selected).")


def plot_in_sample_static_results(in_sample_data, static_beta, static_alpha,
                                  signals_static_in, portfolio_static_in,
                                  ticker1, ticker2, split_date, entry_z, exit_z,
                                  initial_capital, pair_info, output_folder):
    """
    Generates and saves a multi-panel plot summarizing the in-sample performance
    of the pairs trading strategy using static OLS parameters.

    The plot includes:
    1.  Prices of the two assets (`ticker1`, `ticker2`) and the static OLS regression line.
    2.  The spread calculated using static parameters and its corresponding z-score,
        along with entry/exit threshold lines and trade entry markers.
    3.  The equity curve of the static OLS strategy, normalized to the initial capital,
        compared with Buy & Hold of the individual assets.
    4.  The drawdown curve of the static OLS strategy.

    Parameters
    ----------
    in_sample_data : pd.DataFrame
        In-sample price data for the selected pair (columns: `ticker1`, `ticker2`).
    static_beta : float
        The static OLS beta (hedge ratio) calculated from in-sample data.
    static_alpha : float
        The static OLS alpha (intercept) calculated from in-sample data.
    signals_static_in : pd.DataFrame
        DataFrame containing signals generated using static parameters for the in-sample period.
        Expected columns: 'spread', 'z_score', 'signal'.
    portfolio_static_in : pd.DataFrame
        DataFrame from `backtest_strategy` for the static OLS strategy in-sample.
        Expected column: 'total' (portfolio value).
    ticker1 : str
        Name of the first asset in the pair.
    ticker2 : str
        Name of the second asset in the pair.
    split_date : pd.Timestamp
        The date marking the end of the in-sample period.
    entry_z : float
        The z-score threshold for entering a trade.
    exit_z : float
        The z-score threshold for exiting a trade.
    initial_capital : float
        The initial capital used for backtesting.
    pair_info : dict
        Dictionary containing information about the selected pair, used here for
        displaying the in-sample half-life.
    output_folder : str
        Directory to save the plot.
    """
    # Check for empty inputs
    if signals_static_in.empty or portfolio_static_in.empty or pd.isna(static_beta) or pd.isna(static_alpha):
        print("Skipping In-Sample Static Plots (no data/params)."); return

    # Prepare output directory and plot details
    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan)

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'In-Sample Static OLS: {ticker1} vs {ticker2} (up to {split_date.date()}, HL: {half_life:.1f}d)', fontsize=14)

    # --- Panel 1: Prices and Static OLS Fit ---
    axs[0].plot(in_sample_data.index, in_sample_data[ticker1], label=ticker1, alpha=0.8, lw=1)
    axs[0].plot(in_sample_data.index, in_sample_data[ticker2], label=ticker2, alpha=0.8, lw=1)
    axs[0].plot(in_sample_data.index, static_beta * in_sample_data[ticker2] + static_alpha,
                label=f'Static Fit', color='red', linestyle=':', lw=1.5)
    axs[0].set_ylabel("Price")
    axs[0].set_title("Prices & Static OLS Regression")
    axs[0].legend(loc='best')
    axs[0].grid(True, alpha=0.3)

    # --- Panel 2: Spread and Z-Score (Static) ---
    ax1b = axs[1].twinx() # Create a second y-axis for Z-score
    p1, = axs[1].plot(signals_static_in.index, signals_static_in['spread'], label='Spread (Static)', color='purple', lw=1.5)
    axs[1].set_ylabel("Spread", color='purple')
    axs[1].tick_params(axis='y', labelcolor='purple')
    axs[1].grid(True, axis='y', linestyle=':', alpha=0.5)

    p2, = ax1b.plot(signals_static_in.index, signals_static_in['z_score'], label='Z-Score (Static)', color='green', lw=1.5)
    ax1b.set_ylabel("Z-Score", color='green')
    ax1b.tick_params(axis='y', labelcolor='green')

    # Plot Z-score thresholds
    ax1b.axhline(entry_z, color='red', linestyle='--', lw=1, label=f'Entry ({entry_z:.1f})')
    ax1b.axhline(-entry_z, color='red', linestyle='--', lw=1)
    ax1b.axhline(exit_z, color='orange', linestyle='--', lw=1, label=f'Exit ({exit_z:.1f})')
    ax1b.axhline(-exit_z, color='orange', linestyle='--', lw=1)

    # Plot entry signals
    entry_long = signals_static_in[signals_static_in['signal'] == 1].index
    entry_short = signals_static_in[signals_static_in['signal'] == -1].index
    ax1b.plot(entry_long, signals_static_in.loc[entry_long, 'z_score'], '^', markersize=5, color='lime', alpha=0.8, label='Long Entry')
    ax1b.plot(entry_short, signals_static_in.loc[entry_short, 'z_score'], 'v', markersize=5, color='red', alpha=0.8, label='Short Entry')
    ax1b.legend(loc='lower left')
    axs[1].set_title("Spread & Z-Score (Static OLS Params)")

    # --- Panel 3: Equity Curve ---
    portfolio_norm = portfolio_static_in['total'] / initial_capital
    axs[2].plot(portfolio_norm.index, portfolio_norm, label=f'Static OLS Strategy', lw=1.5)

    # Plot Buy & Hold for individual assets
    s1_norm = (in_sample_data[ticker1] / in_sample_data[ticker1].iloc[0]).reindex(portfolio_norm.index)
    s2_norm = (in_sample_data[ticker2] / in_sample_data[ticker2].iloc[0]).reindex(portfolio_norm.index)
    axs[2].plot(s1_norm.index, s1_norm, label=f'B&H {ticker1}', linestyle='--', alpha=0.7, lw=1)
    axs[2].plot(s2_norm.index, s2_norm, label=f'B&H {ticker2}', linestyle=':', alpha=0.7, lw=1)
    axs[2].set_ylabel("Normalized Value")
    axs[2].set_title("Equity Curve")
    axs[2].legend(loc='best')
    axs[2].grid(True, alpha=0.3)

    # --- Panel 4: Drawdown Curve ---
    rolling_max = portfolio_static_in['total'].cummax()
    daily_dd = portfolio_static_in['total'] / rolling_max - 1.0
    axs[3].plot(daily_dd.index, daily_dd * 100, label='Static OLS Drawdown', lw=1.5)
    axs[3].fill_between(daily_dd.index, daily_dd * 100, 0, alpha=0.3)
    axs[3].set_ylabel("Drawdown (%)")
    axs[3].set_title("Drawdown Curve")
    axs[3].yaxis.set_major_formatter(FuncFormatter('{:.0f}%'.format)) # Format y-axis as percentage
    axs[3].legend(loc='best')
    axs[3].grid(True, alpha=0.3)

    # Final plot adjustments and saving
    plt.xlabel("Date")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make space for suptitle
    filename = _sanitize_filename(f"IS_static_results_{ticker1}_vs_{ticker2}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_in_sample_kf_results(in_sample_data, signals_kf_in, portfolio_kf_in,
                              ticker1, ticker2, split_date,
                              entry_z, exit_z, initial_capital, pair_info,
                              use_dynamic_thresholds, output_folder):
    """
    Generates and saves a multi-panel plot summarizing the in-sample performance
    of the pairs trading strategy using Kalman Filter estimated parameters.

    The plot includes:
    1.  Prices of the two assets (`ticker1`, `ticker2`).
    2.  Time series of the Kalman Filter estimated beta and alpha.
    3.  The spread calculated using KF parameters and its corresponding z-score.
        Entry/exit threshold lines (fixed or dynamic) and trade entry markers are shown.
    4.  The equity curve of the KF strategy, normalized, compared with Buy & Hold.
    5.  The drawdown curve of the KF strategy.

    Parameters
    ----------
    in_sample_data : pd.DataFrame
        In-sample price data for the selected pair.
    signals_kf_in : pd.DataFrame
        Signals generated using KF parameters for the in-sample period.
        Expected columns: 'beta', 'alpha', 'spread', 'z_score', 'signal', and
        optionally 'entry_z_upper', 'entry_z_lower', etc., if `use_dynamic_thresholds` is True.
    portfolio_kf_in : pd.DataFrame
        Portfolio data from `backtest_strategy` for the KF strategy in-sample.
    ticker1 : str, ticker2 : str
        Names of the assets in the pair.
    split_date : pd.Timestamp
        End date of the in-sample period.
    entry_z : float, exit_z : float
        Base z-score thresholds for entry/exit (used if not dynamic, or as base for dynamic).
    initial_capital : float
        Initial backtest capital.
    pair_info : dict
        Information about the selected pair (for displaying in-sample half-life).
    use_dynamic_thresholds : bool
        If True, plots dynamic z-score thresholds from `signals_kf_in`.
    output_folder : str
        Directory to save the plot.
    """
    # Check for empty inputs
    if portfolio_kf_in.empty or signals_kf_in.empty:
        print("Skipping In-Sample KF Plots (no data)."); return

    # Prepare output directory and plot details
    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan)

    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'In-Sample KF: {ticker1} vs {ticker2} (up to {split_date.date()}, HL: {half_life:.1f}d)', fontsize=14)

    # --- Panel 1: In-Sample Prices ---
    axs[0].plot(in_sample_data.index, in_sample_data[ticker1], label=ticker1, alpha=0.8, lw=1)
    axs[0].plot(in_sample_data.index, in_sample_data[ticker2], label=ticker2, alpha=0.8, lw=1)
    axs[0].set_ylabel("Price")
    axs[0].set_title("In-Sample Prices")
    axs[0].legend(loc='best')
    axs[0].grid(True, alpha=0.3)

    # --- Panel 2: KF State Estimates (Beta and Alpha) ---
    # Reindex and fill to ensure states plot over the full in_sample_data range
    beta_plot = signals_kf_in['beta'].reindex(in_sample_data.index, method='ffill').bfill()
    alpha_plot = signals_kf_in['alpha'].reindex(in_sample_data.index, method='ffill').bfill()
    axs[1].plot(beta_plot.index, beta_plot, label='KF Beta', lw=1.5)
    axs[1].plot(alpha_plot.index, alpha_plot, label='KF Alpha', lw=1.5, alpha=0.8)
    axs[1].set_ylabel("State Value")
    axs[1].set_title("KF State Estimates")
    axs[1].legend(loc='best')
    axs[1].grid(True, alpha=0.3)

    # --- Panel 3: Spread and Z-Score (KF) ---
    ax2b = axs[2].twinx() # Second y-axis for Z-score
    p1, = axs[2].plot(signals_kf_in.index, signals_kf_in['spread'], label='Spread (KF)', color='purple', lw=1.5)
    axs[2].set_ylabel("Spread", color='purple')
    axs[2].tick_params(axis='y', labelcolor='purple')
    axs[2].grid(True, axis='y', linestyle=':', alpha=0.5)

    p2, = ax2b.plot(signals_kf_in.index, signals_kf_in['z_score'], label='Z-Score (KF)', color='green', lw=1.5)
    ax2b.set_ylabel("Z-Score", color='green')
    ax2b.tick_params(axis='y', labelcolor='green')

    # Plot Z-score thresholds (fixed or dynamic)
    if use_dynamic_thresholds and 'entry_z_upper' in signals_kf_in.columns:
        ax2b.plot(signals_kf_in.index, signals_kf_in['entry_z_upper'], color='red', linestyle='--', lw=1, label=f'Dyn Entry ({entry_z:.1f}x)')
        ax2b.plot(signals_kf_in.index, signals_kf_in['entry_z_lower'], color='red', linestyle='--', lw=1)
        ax2b.plot(signals_kf_in.index, signals_kf_in['exit_z_upper'], color='orange', linestyle='--', lw=1, label=f'Dyn Exit ({exit_z:.1f}x)')
        ax2b.plot(signals_kf_in.index, signals_kf_in['exit_z_lower'], color='orange', linestyle='--', lw=1)
    else: # Fixed thresholds
        ax2b.axhline(entry_z, color='red', linestyle='--', lw=1, label=f'Entry ({entry_z:.1f})')
        ax2b.axhline(-entry_z, color='red', linestyle='--', lw=1)
        ax2b.axhline(exit_z, color='orange', linestyle='--', lw=1, label=f'Exit ({exit_z:.1f})')
        ax2b.axhline(-exit_z, color='orange', linestyle='--', lw=1)
    ax2b.legend(loc='lower left')

    # Plot entry signals
    entry_long = signals_kf_in[signals_kf_in['signal'] == 1].index
    entry_short = signals_kf_in[signals_kf_in['signal'] == -1].index
    ax2b.plot(entry_long, signals_kf_in.loc[entry_long, 'z_score'], '^', markersize=5, color='lime', alpha=0.8, label='Long Entry')
    ax2b.plot(entry_short, signals_kf_in.loc[entry_short, 'z_score'], 'v', markersize=5, color='red', alpha=0.8, label='Short Entry')
    axs[2].set_title("In-Sample Spread and Z-Score (KF)")

    # --- Panel 4: Equity Curve Comparison ---
    portfolio_norm = portfolio_kf_in['total'] / initial_capital
    axs[3].plot(portfolio_norm.index, portfolio_norm, label='KF Strategy', linewidth=2)

    # Plot Buy & Hold for individual assets, ensuring alignment and non-empty data
    if not in_sample_data.empty:
        s1_aligned = in_sample_data[ticker1].reindex(portfolio_norm.index).dropna()
        s2_aligned = in_sample_data[ticker2].reindex(portfolio_norm.index).dropna()
        if not s1_aligned.empty:
            s1_norm = s1_aligned / s1_aligned.iloc[0]
            axs[3].plot(s1_norm.index, s1_norm, label=f'B&H {ticker1}', linestyle='--', alpha=0.7, lw=1)
        if not s2_aligned.empty:
            s2_norm = s2_aligned / s2_aligned.iloc[0]
            axs[3].plot(s2_norm.index, s2_norm, label=f'B&H {ticker2}', linestyle=':', alpha=0.7, lw=1)
    axs[3].set_ylabel("Normalized Value")
    axs[3].set_title("Equity Curve Comparison")
    axs[3].legend(loc='best')
    axs[3].grid(True, alpha=0.3)

    # --- Panel 5: Drawdown Curve ---
    rolling_max_kf = portfolio_kf_in['total'].cummax()
    daily_dd_kf = portfolio_kf_in['total'] / rolling_max_kf - 1.0
    axs[4].plot(daily_dd_kf.index, daily_dd_kf * 100, label='KF Drawdown', lw=1.5)
    axs[4].fill_between(daily_dd_kf.index, daily_dd_kf * 100, 0, alpha=0.3)
    axs[4].set_ylabel("Drawdown (%)")
    axs[4].set_title("Drawdown Curve")
    axs[4].yaxis.set_major_formatter(FuncFormatter('{:.0f}%'.format)) # Format y-axis as percentage
    axs[4].legend(loc='best')
    axs[4].grid(True, alpha=0.3)

    # Final plot adjustments and saving
    plt.xlabel("Date")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for suptitle
    filename = _sanitize_filename(f"IS_KF_results_{ticker1}_vs_{ticker2}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_out_of_sample_kf_results(oos_data, signals_kf_oos, portfolio_kf_oos,
                                  bench_portfolio_oos, ticker1, ticker2, split_date,
                                  entry_z, exit_z, stop_loss_z, pair_info,
                                  use_dynamic_thresholds, benchmark_ticker, output_folder):
    """
    Generates and saves a multi-panel plot summarizing the out-of-sample (OOS)
    performance of the Kalman Filter pairs trading strategy.

    The plot includes:
    1.  OOS prices of the two assets.
    2.  Time series of the OOS Kalman Filter estimated beta and alpha.
    3.  The OOS spread (KF parameters) and its z-score, with entry/exit/stop-loss
        thresholds and trade entry markers.
    4.  The OOS equity curve of the KF strategy, normalized, compared with
        benchmark portfolios (e.g., Static OLS, Buy & Hold individual assets, market benchmark).
    5.  The OOS drawdown curve of the KF strategy, optionally compared with a market benchmark's drawdown.

    Parameters
    ----------
    oos_data : pd.DataFrame
        Out-of-sample price data for the selected pair.
    signals_kf_oos : pd.DataFrame
        Signals generated using KF parameters for the OOS period.
    portfolio_kf_oos : pd.DataFrame
        Portfolio data from `backtest_strategy` for the KF strategy OOS.
    bench_portfolio_oos : pd.DataFrame
        DataFrame containing normalized portfolio values for various benchmarks OOS.
        Index should align with `portfolio_kf_oos`.
    ticker1 : str, ticker2 : str
        Names of the assets in the pair.
    split_date : pd.Timestamp
        Start date of the OOS period.
    entry_z : float, exit_z : float
        Base z-score thresholds.
    stop_loss_z : float or None
        The z-score delta for stop-loss, if enabled.
    pair_info : dict
        Information about the pair (for displaying in-sample half-life).
    use_dynamic_thresholds : bool
        If True, plots dynamic z-score thresholds.
    benchmark_ticker : str or None
        Ticker symbol of the market benchmark for drawdown comparison.
    output_folder : str
        Directory to save the plot.
    """
    # Check for empty inputs
    if portfolio_kf_oos.empty or signals_kf_oos.empty:
        print("Skipping Out-of-Sample KF Plots (no data)."); return

    # Prepare output directory and plot details
    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan) # In-sample half-life

    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'OOS KF: {ticker1} vs {ticker2} (from {split_date.date()}, IS HL: {half_life:.1f}d)', fontsize=14)

    # --- Panel 1: Out-of-Sample Prices ---
    axs[0].plot(oos_data.index, oos_data[ticker1], label=ticker1, alpha=0.8, lw=1)
    axs[0].plot(oos_data.index, oos_data[ticker2], label=ticker2, alpha=0.8, lw=1)
    axs[0].set_ylabel("Price")
    axs[0].set_title("Out-of-Sample Prices")
    axs[0].legend(loc='best')
    axs[0].grid(True, alpha=0.3)

    # --- Panel 2: KF State Estimates (OOS) ---
    axs[1].plot(signals_kf_oos.index, signals_kf_oos['beta'], label='KF Beta', lw=1.5)
    axs[1].plot(signals_kf_oos.index, signals_kf_oos['alpha'], label='KF Alpha', lw=1.5, alpha=0.8)
    axs[1].set_ylabel("State Value")
    axs[1].set_title("KF State Estimates (OOS)")
    axs[1].legend(loc='best')
    axs[1].grid(True, alpha=0.3)

    # --- Panel 3: Spread and Z-Score (OOS) ---
    ax2b = axs[2].twinx() # Second y-axis for Z-score
    p1, = axs[2].plot(signals_kf_oos.index, signals_kf_oos['spread'], label='Spread (KF)', color='purple', lw=1.5)
    axs[2].set_ylabel("Spread", color='purple')
    axs[2].tick_params(axis='y', labelcolor='purple')
    axs[2].grid(True, axis='y', linestyle=':', alpha=0.5)

    p2, = ax2b.plot(signals_kf_oos.index, signals_kf_oos['z_score'], label='Z-Score (KF)', color='green', lw=1.5)
    ax2b.set_ylabel("Z-Score", color='green')
    ax2b.tick_params(axis='y', labelcolor='green')

    # Plot Z-score thresholds (fixed or dynamic) and stop-loss info
    legend_title = f"Stop Loss Z Delta: {stop_loss_z}" if stop_loss_z is not None else None
    if use_dynamic_thresholds and 'entry_z_upper' in signals_kf_oos.columns:
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['entry_z_upper'], color='red', linestyle='--', lw=1, label=f'Dyn Entry ({entry_z:.1f}x)')
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['entry_z_lower'], color='red', linestyle='--', lw=1)
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['exit_z_upper'], color='orange', linestyle='--', lw=1, label=f'Dyn Exit ({exit_z:.1f}x)')
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['exit_z_lower'], color='orange', linestyle='--', lw=1)
    else: # Fixed thresholds
        ax2b.axhline(entry_z, color='red', linestyle='--', lw=1, label=f'Entry ({entry_z:.1f})')
        ax2b.axhline(-entry_z, color='red', linestyle='--', lw=1)
        ax2b.axhline(exit_z, color='orange', linestyle='--', lw=1, label=f'Exit ({exit_z:.1f})')
        ax2b.axhline(-exit_z, color='orange', linestyle='--', lw=1)
        if stop_loss_z is not None: # Indicate zero line if stop loss is active, for context
            ax2b.axhline(0, color='black', linestyle=':', lw=0.5)
    ax2b.legend(loc='lower left', title=legend_title)

    # Plot entry signals
    entry_long = signals_kf_oos[signals_kf_oos['signal'] == 1].index
    entry_short = signals_kf_oos[signals_kf_oos['signal'] == -1].index
    ax2b.plot(entry_long, signals_kf_oos.loc[entry_long, 'z_score'], '^', markersize=5, color='lime', alpha=0.8, label='Long Entry')
    ax2b.plot(entry_short, signals_kf_oos.loc[entry_short, 'z_score'], 'v', markersize=5, color='red', alpha=0.8, label='Short Entry')
    axs[2].set_title("Out-of-Sample Spread and Z-Score (KF)")

    # --- Panel 4: OOS Equity Curve Comparison ---
    # Normalize KF strategy portfolio
    portfolio_norm = portfolio_kf_oos['total'] / portfolio_kf_oos['total'].iloc[0]
    axs[3].plot(portfolio_norm.index, portfolio_norm, label='Kalman Filter Strategy', linewidth=2)

    # Plot benchmark portfolios
    for col in bench_portfolio_oos.columns:
        if not bench_portfolio_oos[col].empty:
             bench_aligned = bench_portfolio_oos[col].reindex(portfolio_norm.index).dropna()
             if not bench_aligned.empty:
                  bench_norm = bench_aligned / bench_aligned.iloc[0] # Normalize benchmark
                  style = '--' if col=='Static OLS' else ':' # Different style for static OLS
                  axs[3].plot(bench_norm.index, bench_norm, label=f'{col} (Bench)', linestyle=style, lw=1.5, alpha=0.9)
    axs[3].set_ylabel("Normalized Value")
    axs[3].set_title("OOS Equity Curve Comparison")
    axs[3].legend(loc='best')
    axs[3].grid(True, alpha=0.3)

    # --- Panel 5: OOS Drawdown Curve ---
    rolling_max_kf = portfolio_kf_oos['total'].cummax()
    daily_dd_kf = portfolio_kf_oos['total'] / rolling_max_kf - 1.0
    axs[4].plot(daily_dd_kf.index, daily_dd_kf * 100, label='KF Drawdown', lw=1.5)
    axs[4].fill_between(daily_dd_kf.index, daily_dd_kf * 100, 0, alpha=0.3)

    # Optionally plot benchmark drawdown
    if benchmark_ticker in bench_portfolio_oos.columns and not bench_portfolio_oos[benchmark_ticker].empty:
         bench_bench_aligned = bench_portfolio_oos[benchmark_ticker].reindex(daily_dd_kf.index).dropna()
         if not bench_bench_aligned.empty:
              rolling_max_bench = bench_bench_aligned.cummax()
              daily_dd_bench = bench_bench_aligned / rolling_max_bench - 1.0
              axs[4].plot(daily_dd_bench.index, daily_dd_bench * 100, label=f'{benchmark_ticker} Drawdown', linestyle=':', lw=1.5, alpha=0.8)
    axs[4].set_ylabel("Drawdown (%)")
    axs[4].set_title("OOS Drawdown Curve")
    axs[4].yaxis.set_major_formatter(FuncFormatter('{:.0f}%'.format)) # Format y-axis as percentage
    axs[4].legend(loc='best')
    axs[4].grid(True, alpha=0.3)

    # Final plot adjustments and saving
    plt.xlabel("Date")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for suptitle
    filename = _sanitize_filename(f"OOS_KF_results_{ticker1}_vs_{ticker2}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_in_sample_cumulative_returns(in_sample_returns_dict, initial_capital, split_date, pair_info, output_folder):
    """
    Plots the cumulative performance (equity curves on a log scale) of various
    strategies and benchmarks during the in-sample period.

    Parameters
    ----------
    in_sample_returns_dict : dict[str, pd.Series]
        A dictionary where keys are strategy/benchmark labels (e.g., "Kalman Filter",
        "B&H Asset1") and values are pandas Series of their daily returns
        for the in-sample period.
    initial_capital : float
        The initial capital to normalize portfolio values.
    split_date : pd.Timestamp
        The end date of the in-sample period, used in the plot title.
    pair_info : dict
        Information about the selected pair (e.g., tickers, in-sample half-life),
        used in plot titles and filenames.
    output_folder : str
        Directory to save the plot.
    """
    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan) # In-sample half-life

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(f'In-Sample Cumulative Performance (Log Scale) - up to {split_date.date()} (HL: {half_life:.1f}d)', fontsize=14)

    has_data_to_plot = False
    # Define styles for different strategies
    styles = {'Kalman Filter': ('-', 2.0), 'Static OLS': ('--', 1.5), 'default': (':', 1.5)}

    # Iterate through each returns series provided
    for label, returns_series in in_sample_returns_dict.items():
        if returns_series.empty or returns_series.isnull().all():
            continue # Skip if no valid returns data

        cumulative_product = (1 + returns_series).cumprod()

        # Handle cases where cumulative product might go to zero or negative (e.g., extreme losses)
        if (cumulative_product <= 1e-8).any(): # Using a small epsilon
            print(f"Warning IS: Cumulative value near/below zero for '{label}'. Plotting up to that point.")
            cumulative_product = cumulative_product[cumulative_product > 1e-8] # Filter out non-positive values
            if cumulative_product.empty:
                continue # Skip if no data remains after filtering

        portfolio_value = cumulative_product * initial_capital

        # Determine plot style
        style_key = 'default'
        if 'Kalman Filter' in label: style_key = 'Kalman Filter'
        elif 'Static OLS' in label: style_key = 'Static OLS'
        linestyle, linewidth = styles.get(style_key, styles['default'])

        ax.plot(portfolio_value.index, portfolio_value, label=label, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
        has_data_to_plot = True

    if not has_data_to_plot:
        print("No valid data series to plot IS cumulative returns.")
        plt.close(fig)
        return

    # Set plot properties
    ax.set_ylabel("Portfolio Value (Log Scale)")
    ax.set_xlabel("Date")
    ax.set_yscale('log') # Use log scale for y-axis
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5) # Grid for both major and minor ticks on log scale
    plt.tight_layout()

    # Save plot
    filename_suffix = f"IS_cumulative_returns_pair_{pair_info.get('ticker1','')}_{pair_info.get('ticker2','')}.png"
    filename = _sanitize_filename(filename_suffix)
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_out_of_sample_cumulative_returns(out_of_sample_returns_dict, initial_capital, split_date, pair_info, output_folder):
    """
    Plots the cumulative performance (equity curves on a log scale) of various
    strategies and benchmarks during the out-of-sample (OOS) period.

    Parameters
    ----------
    out_of_sample_returns_dict : dict[str, pd.Series]
        A dictionary where keys are strategy/benchmark labels and values are
        pandas Series of their daily returns for the OOS period.
    initial_capital : float
        The initial capital to normalize portfolio values.
    split_date : pd.Timestamp
        The start date of the OOS period, used in the plot title.
    pair_info : dict
        Information about the selected pair, used in plot titles and filenames.
    output_folder : str
        Directory to save the plot.
    """
    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan) # In-sample half-life

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(f'OOS Cumulative Performance (Log Scale) - from {split_date.date()} (IS HL: {half_life:.1f}d)', fontsize=14)

    has_data_to_plot = False
    # Define styles for different strategies
    styles = {'Kalman Filter': ('-', 2.0), 'Static OLS': ('--', 1.5), 'default': (':', 1.5)}

    # Iterate through each returns series provided
    for label, returns_series in out_of_sample_returns_dict.items():
        if returns_series.empty or returns_series.isnull().all():
            continue # Skip if no valid returns data

        cumulative_product = (1 + returns_series).cumprod()

        # Handle cases where cumulative product might go to zero or negative
        if (cumulative_product <= 1e-8).any(): # Using a small epsilon
            print(f"Warning OOS: Cumulative value near/below zero for '{label}'. Plotting up to that point.")
            cumulative_product = cumulative_product[cumulative_product > 1e-8] # Filter out non-positive values
            if cumulative_product.empty:
                continue # Skip if no data remains after filtering

        portfolio_value = cumulative_product * initial_capital

        # Determine plot style
        style_key = 'default'
        if 'Kalman Filter' in label: style_key = 'Kalman Filter'
        elif 'Static OLS' in label: style_key = 'Static OLS'
        linestyle, linewidth = styles.get(style_key, styles['default'])

        ax.plot(portfolio_value.index, portfolio_value, label=label, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
        has_data_to_plot = True

    if not has_data_to_plot:
        print("No valid data series to plot OOS cumulative returns.")
        plt.close(fig)
        return

    # Set plot properties
    ax.set_ylabel("Portfolio Value (Log Scale)")
    ax.set_xlabel("Date")
    ax.set_yscale('log') # Use log scale for y-axis
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5) # Grid for both major and minor ticks on log scale
    plt.tight_layout()

    # Save plot
    filename_suffix = f"OOS_cumulative_returns_pair_{pair_info.get('ticker1','')}_{pair_info.get('ticker2','')}.png"
    filename = _sanitize_filename(filename_suffix)
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")