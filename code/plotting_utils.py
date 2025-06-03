import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Helper to generate safe filenames
def _sanitize_filename(name):
    return "".join([c if c.isalnum() else "_" for c in str(name)])

def plot_pair_selection_diagnostics(in_sample_data_all_pairs, pair_info, output_folder):
    """ Plots diagnostics: Correlation matrix and selected pair's IS spread. """
    print("\n--- Generating Pair Selection Diagnostic Plots ---")
    os.makedirs(output_folder, exist_ok=True)

    # 1. Correlation Matrix Heatmap
    if not in_sample_data_all_pairs.empty and in_sample_data_all_pairs.shape[1] > 1:
        correlation_matrix = in_sample_data_all_pairs.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('In-Sample Price Correlation Matrix (Pair Candidates)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        filepath = os.path.join(output_folder, "pair_selection_correlation_matrix.png")
        plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
        print(f"Saved: {filepath}")
    else:
        print("Skipping correlation heatmap (not enough data or candidates).")

    # 2. In-Sample Spread Plot for Selected Pair
    if pair_info:
        t1, t2 = pair_info['ticker1'], pair_info['ticker2']
        ols_beta, ols_alpha = pair_info['ols_beta'], pair_info['ols_alpha']
        adf_p, hl = pair_info['adf_p_value'], pair_info['half_life_days']

        if t1 in in_sample_data_all_pairs.columns and t2 in in_sample_data_all_pairs.columns:
            spread = in_sample_data_all_pairs[t1] - (ols_beta * in_sample_data_all_pairs[t2] + ols_alpha)
            fig, ax = plt.subplots(figsize=(12, 6))
            spread.plot(ax=ax, title=f'In-Sample Spread: {t1} - ({ols_beta:.2f}*{t2} + {ols_alpha:.2f})', lw=1.5)
            ax.axhline(spread.mean(), color='red', linestyle='--', lw=1, label=f'Mean ({spread.mean():.3f})')
            ax.set_ylabel("Spread Value"); ax.set_xlabel("Date")
            ax.grid(True, alpha=0.4); ax.legend()
            stats_text = f'ADF p: {adf_p:.3f}\nHalf-Life: {hl:.1f}d'
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                     verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            plt.tight_layout()
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
    """ Plots IN-SAMPLE results using static OLS parameters. """
    if signals_static_in.empty or portfolio_static_in.empty or pd.isna(static_beta) or pd.isna(static_alpha):
        print("Skipping In-Sample Static Plots (no data/params)."); return

    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan)
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'In-Sample Static OLS: {ticker1} vs {ticker2} (up to {split_date.date()}, HL: {half_life:.1f}d)', fontsize=14)

    axs[0].plot(in_sample_data.index, in_sample_data[ticker1], label=ticker1, alpha=0.8, lw=1)
    axs[0].plot(in_sample_data.index, in_sample_data[ticker2], label=ticker2, alpha=0.8, lw=1)
    axs[0].plot(in_sample_data.index, static_beta * in_sample_data[ticker2] + static_alpha,
                label=f'Static Fit', color='red', linestyle=':', lw=1.5)
    axs[0].set_ylabel("Price"); axs[0].set_title("Prices & Static OLS Regression"); axs[0].legend(loc='best'); axs[0].grid(True, alpha=0.3)

    ax1b = axs[1].twinx()
    p1, = axs[1].plot(signals_static_in.index, signals_static_in['spread'], label='Spread (Static)', color='purple', lw=1.5)
    axs[1].set_ylabel("Spread", color='purple'); axs[1].tick_params(axis='y', labelcolor='purple'); axs[1].grid(True, axis='y', linestyle=':', alpha=0.5)
    p2, = ax1b.plot(signals_static_in.index, signals_static_in['z_score'], label='Z-Score (Static)', color='green', lw=1.5)
    ax1b.set_ylabel("Z-Score", color='green'); ax1b.tick_params(axis='y', labelcolor='green')
    ax1b.axhline(entry_z, color='red', linestyle='--', lw=1, label=f'Entry ({entry_z:.1f})')
    ax1b.axhline(-entry_z, color='red', linestyle='--', lw=1)
    ax1b.axhline(exit_z, color='orange', linestyle='--', lw=1, label=f'Exit ({exit_z:.1f})')
    ax1b.axhline(-exit_z, color='orange', linestyle='--', lw=1)
    entry_long = signals_static_in[signals_static_in['signal'] == 1].index
    entry_short = signals_static_in[signals_static_in['signal'] == -1].index
    ax1b.plot(entry_long, signals_static_in.loc[entry_long, 'z_score'], '^', markersize=5, color='lime', alpha=0.8, label='Long Entry')
    ax1b.plot(entry_short, signals_static_in.loc[entry_short, 'z_score'], 'v', markersize=5, color='red', alpha=0.8, label='Short Entry')
    ax1b.legend(loc='lower left'); axs[1].set_title("Spread & Z-Score (Static OLS Params)")

    portfolio_norm = portfolio_static_in['total'] / initial_capital
    axs[2].plot(portfolio_norm.index, portfolio_norm, label=f'Static OLS Strategy', lw=1.5)
    s1_norm = (in_sample_data[ticker1] / in_sample_data[ticker1].iloc[0]).reindex(portfolio_norm.index)
    s2_norm = (in_sample_data[ticker2] / in_sample_data[ticker2].iloc[0]).reindex(portfolio_norm.index)
    axs[2].plot(s1_norm.index, s1_norm, label=f'B&H {ticker1}', linestyle='--', alpha=0.7, lw=1)
    axs[2].plot(s2_norm.index, s2_norm, label=f'B&H {ticker2}', linestyle=':', alpha=0.7, lw=1)
    axs[2].set_ylabel("Normalized Value"); axs[2].set_title("Equity Curve"); axs[2].legend(loc='best'); axs[2].grid(True, alpha=0.3)

    rolling_max = portfolio_static_in['total'].cummax()
    daily_dd = portfolio_static_in['total'] / rolling_max - 1.0
    axs[3].plot(daily_dd.index, daily_dd * 100, label='Static OLS Drawdown', lw=1.5)
    axs[3].fill_between(daily_dd.index, daily_dd * 100, 0, alpha=0.3)
    axs[3].set_ylabel("Drawdown (%)"); axs[3].set_title("Drawdown Curve"); axs[3].yaxis.set_major_formatter(FuncFormatter('{:.0f}%'.format)); axs[3].legend(loc='best'); axs[3].grid(True, alpha=0.3)

    plt.xlabel("Date"); plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = _sanitize_filename(f"IS_static_results_{ticker1}_vs_{ticker2}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_in_sample_kf_results(in_sample_data, signals_kf_in, portfolio_kf_in,
                              ticker1, ticker2, split_date,
                              entry_z, exit_z, initial_capital, pair_info,
                              use_dynamic_thresholds, output_folder):
    if portfolio_kf_in.empty or signals_kf_in.empty:
        print("Skipping In-Sample KF Plots (no data)."); return

    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan)
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'In-Sample KF: {ticker1} vs {ticker2} (up to {split_date.date()}, HL: {half_life:.1f}d)', fontsize=14)

    axs[0].plot(in_sample_data.index, in_sample_data[ticker1], label=ticker1, alpha=0.8, lw=1)
    axs[0].plot(in_sample_data.index, in_sample_data[ticker2], label=ticker2, alpha=0.8, lw=1)
    axs[0].set_ylabel("Price"); axs[0].set_title("In-Sample Prices"); axs[0].legend(loc='best'); axs[0].grid(True, alpha=0.3)

    beta_plot = signals_kf_in['beta'].reindex(in_sample_data.index, method='ffill').bfill()
    alpha_plot = signals_kf_in['alpha'].reindex(in_sample_data.index, method='ffill').bfill()
    axs[1].plot(beta_plot.index, beta_plot, label='KF Beta', lw=1.5)
    axs[1].plot(alpha_plot.index, alpha_plot, label='KF Alpha', lw=1.5, alpha=0.8)
    axs[1].set_ylabel("State Value"); axs[1].set_title("KF State Estimates"); axs[1].legend(loc='best'); axs[1].grid(True, alpha=0.3)

    ax2b = axs[2].twinx()
    p1, = axs[2].plot(signals_kf_in.index, signals_kf_in['spread'], label='Spread (KF)', color='purple', lw=1.5)
    axs[2].set_ylabel("Spread", color='purple'); axs[2].tick_params(axis='y', labelcolor='purple'); axs[2].grid(True, axis='y', linestyle=':', alpha=0.5)
    p2, = ax2b.plot(signals_kf_in.index, signals_kf_in['z_score'], label='Z-Score (KF)', color='green', lw=1.5)
    ax2b.set_ylabel("Z-Score", color='green'); ax2b.tick_params(axis='y', labelcolor='green')

    if use_dynamic_thresholds and 'entry_z_upper' in signals_kf_in.columns:
        ax2b.plot(signals_kf_in.index, signals_kf_in['entry_z_upper'], color='red', linestyle='--', lw=1, label=f'Dyn Entry ({entry_z:.1f}x)')
        ax2b.plot(signals_kf_in.index, signals_kf_in['entry_z_lower'], color='red', linestyle='--', lw=1)
        ax2b.plot(signals_kf_in.index, signals_kf_in['exit_z_upper'], color='orange', linestyle='--', lw=1, label=f'Dyn Exit ({exit_z:.1f}x)')
        ax2b.plot(signals_kf_in.index, signals_kf_in['exit_z_lower'], color='orange', linestyle='--', lw=1)
    else:
        ax2b.axhline(entry_z, color='red', linestyle='--', lw=1, label=f'Entry ({entry_z:.1f})')
        ax2b.axhline(-entry_z, color='red', linestyle='--', lw=1)
        ax2b.axhline(exit_z, color='orange', linestyle='--', lw=1, label=f'Exit ({exit_z:.1f})')
        ax2b.axhline(-exit_z, color='orange', linestyle='--', lw=1)
    ax2b.legend(loc='lower left')

    entry_long = signals_kf_in[signals_kf_in['signal'] == 1].index
    entry_short = signals_kf_in[signals_kf_in['signal'] == -1].index
    ax2b.plot(entry_long, signals_kf_in.loc[entry_long, 'z_score'], '^', markersize=5, color='lime', alpha=0.8, label='Long Entry')
    ax2b.plot(entry_short, signals_kf_in.loc[entry_short, 'z_score'], 'v', markersize=5, color='red', alpha=0.8, label='Short Entry')
    axs[2].set_title("In-Sample Spread and Z-Score (KF)")

    portfolio_norm = portfolio_kf_in['total'] / initial_capital
    axs[3].plot(portfolio_norm.index, portfolio_norm, label='KF Strategy', linewidth=2)
    if not in_sample_data.empty:
        s1_aligned = in_sample_data[ticker1].reindex(portfolio_norm.index).dropna()
        s2_aligned = in_sample_data[ticker2].reindex(portfolio_norm.index).dropna()
        if not s1_aligned.empty: s1_norm = s1_aligned / s1_aligned.iloc[0]; axs[3].plot(s1_norm.index, s1_norm, label=f'B&H {ticker1}', linestyle='--', alpha=0.7, lw=1)
        if not s2_aligned.empty: s2_norm = s2_aligned / s2_aligned.iloc[0]; axs[3].plot(s2_norm.index, s2_norm, label=f'B&H {ticker2}', linestyle=':', alpha=0.7, lw=1)
    axs[3].set_ylabel("Normalized Value"); axs[3].set_title("Equity Curve Comparison"); axs[3].legend(loc='best'); axs[3].grid(True, alpha=0.3)

    rolling_max_kf = portfolio_kf_in['total'].cummax()
    daily_dd_kf = portfolio_kf_in['total'] / rolling_max_kf - 1.0
    axs[4].plot(daily_dd_kf.index, daily_dd_kf * 100, label='KF Drawdown', lw=1.5)
    axs[4].fill_between(daily_dd_kf.index, daily_dd_kf * 100, 0, alpha=0.3)
    axs[4].set_ylabel("Drawdown (%)"); axs[4].set_title("Drawdown Curve"); axs[4].yaxis.set_major_formatter(FuncFormatter('{:.0f}%'.format)); axs[4].legend(loc='best'); axs[4].grid(True, alpha=0.3)

    plt.xlabel("Date"); plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = _sanitize_filename(f"IS_KF_results_{ticker1}_vs_{ticker2}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_out_of_sample_kf_results(oos_data, signals_kf_oos, portfolio_kf_oos,
                                  bench_portfolio_oos, ticker1, ticker2, split_date,
                                  entry_z, exit_z, stop_loss_z, pair_info,
                                  use_dynamic_thresholds, benchmark_ticker, output_folder):
    if portfolio_kf_oos.empty or signals_kf_oos.empty:
        print("Skipping Out-of-Sample KF Plots (no data)."); return

    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan)
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'OOS KF: {ticker1} vs {ticker2} (from {split_date.date()}, IS HL: {half_life:.1f}d)', fontsize=14)

    axs[0].plot(oos_data.index, oos_data[ticker1], label=ticker1, alpha=0.8, lw=1)
    axs[0].plot(oos_data.index, oos_data[ticker2], label=ticker2, alpha=0.8, lw=1)
    axs[0].set_ylabel("Price"); axs[0].set_title("Out-of-Sample Prices"); axs[0].legend(loc='best'); axs[0].grid(True, alpha=0.3)

    axs[1].plot(signals_kf_oos.index, signals_kf_oos['beta'], label='KF Beta', lw=1.5)
    axs[1].plot(signals_kf_oos.index, signals_kf_oos['alpha'], label='KF Alpha', lw=1.5, alpha=0.8)
    axs[1].set_ylabel("State Value"); axs[1].set_title("KF State Estimates (OOS)"); axs[1].legend(loc='best'); axs[1].grid(True, alpha=0.3)

    ax2b = axs[2].twinx()
    p1, = axs[2].plot(signals_kf_oos.index, signals_kf_oos['spread'], label='Spread (KF)', color='purple', lw=1.5)
    axs[2].set_ylabel("Spread", color='purple'); axs[2].tick_params(axis='y', labelcolor='purple'); axs[2].grid(True, axis='y', linestyle=':', alpha=0.5)
    p2, = ax2b.plot(signals_kf_oos.index, signals_kf_oos['z_score'], label='Z-Score (KF)', color='green', lw=1.5)
    ax2b.set_ylabel("Z-Score", color='green'); ax2b.tick_params(axis='y', labelcolor='green')

    legend_title = f"Stop Loss Z Delta: {stop_loss_z}" if stop_loss_z is not None else None
    if use_dynamic_thresholds and 'entry_z_upper' in signals_kf_oos.columns:
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['entry_z_upper'], color='red', linestyle='--', lw=1, label=f'Dyn Entry ({entry_z:.1f}x)')
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['entry_z_lower'], color='red', linestyle='--', lw=1)
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['exit_z_upper'], color='orange', linestyle='--', lw=1, label=f'Dyn Exit ({exit_z:.1f}x)')
        ax2b.plot(signals_kf_oos.index, signals_kf_oos['exit_z_lower'], color='orange', linestyle='--', lw=1)
    else:
        ax2b.axhline(entry_z, color='red', linestyle='--', lw=1, label=f'Entry ({entry_z:.1f})')
        ax2b.axhline(-entry_z, color='red', linestyle='--', lw=1)
        ax2b.axhline(exit_z, color='orange', linestyle='--', lw=1, label=f'Exit ({exit_z:.1f})')
        ax2b.axhline(-exit_z, color='orange', linestyle='--', lw=1)
        if stop_loss_z is not None: ax2b.axhline(0, color='black', linestyle=':', lw=0.5)
    ax2b.legend(loc='lower left', title=legend_title)

    entry_long = signals_kf_oos[signals_kf_oos['signal'] == 1].index
    entry_short = signals_kf_oos[signals_kf_oos['signal'] == -1].index
    ax2b.plot(entry_long, signals_kf_oos.loc[entry_long, 'z_score'], '^', markersize=5, color='lime', alpha=0.8, label='Long Entry')
    ax2b.plot(entry_short, signals_kf_oos.loc[entry_short, 'z_score'], 'v', markersize=5, color='red', alpha=0.8, label='Short Entry')
    axs[2].set_title("Out-of-Sample Spread and Z-Score (KF)")

    portfolio_norm = portfolio_kf_oos['total'] / portfolio_kf_oos['total'].iloc[0]
    axs[3].plot(portfolio_norm.index, portfolio_norm, label='Kalman Filter Strategy', linewidth=2)
    for col in bench_portfolio_oos.columns:
        if not bench_portfolio_oos[col].empty:
             bench_aligned = bench_portfolio_oos[col].reindex(portfolio_norm.index).dropna()
             if not bench_aligned.empty:
                  bench_norm = bench_aligned / bench_aligned.iloc[0]
                  style = '--' if col=='Static OLS' else ':'
                  axs[3].plot(bench_norm.index, bench_norm, label=f'{col} (Bench)', linestyle=style, lw=1.5, alpha=0.9)
    axs[3].set_ylabel("Normalized Value"); axs[3].set_title("OOS Equity Curve Comparison"); axs[3].legend(loc='best'); axs[3].grid(True, alpha=0.3)

    rolling_max_kf = portfolio_kf_oos['total'].cummax()
    daily_dd_kf = portfolio_kf_oos['total'] / rolling_max_kf - 1.0
    axs[4].plot(daily_dd_kf.index, daily_dd_kf * 100, label='KF Drawdown', lw=1.5)
    axs[4].fill_between(daily_dd_kf.index, daily_dd_kf * 100, 0, alpha=0.3)
    if benchmark_ticker in bench_portfolio_oos.columns and not bench_portfolio_oos[benchmark_ticker].empty:
         bench_bench_aligned = bench_portfolio_oos[benchmark_ticker].reindex(daily_dd_kf.index).dropna()
         if not bench_bench_aligned.empty:
              rolling_max_bench = bench_bench_aligned.cummax()
              daily_dd_bench = bench_bench_aligned / rolling_max_bench - 1.0
              axs[4].plot(daily_dd_bench.index, daily_dd_bench * 100, label=f'{benchmark_ticker} Drawdown', linestyle=':', lw=1.5, alpha=0.8)
    axs[4].set_ylabel("Drawdown (%)"); axs[4].set_title("OOS Drawdown Curve"); axs[4].yaxis.set_major_formatter(FuncFormatter('{:.0f}%'.format)); axs[4].legend(loc='best'); axs[4].grid(True, alpha=0.3)

    plt.xlabel("Date"); plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = _sanitize_filename(f"OOS_KF_results_{ticker1}_vs_{ticker2}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_in_sample_cumulative_returns(in_sample_returns_dict, initial_capital, split_date, pair_info, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(f'In-Sample Cumulative Performance (Log Scale) - up to {split_date.date()} (HL: {half_life:.1f}d)', fontsize=14)

    has_data_to_plot = False
    styles = {'Kalman Filter': ('-', 2.0), 'Static OLS': ('--', 1.5), 'default': (':', 1.5)}

    for label, returns_series in in_sample_returns_dict.items():
        if returns_series.empty or returns_series.isnull().all(): continue
        cumulative_product = (1 + returns_series).cumprod()
        if (cumulative_product <= 1e-8).any():
            print(f"Warning IS: Cumulative value <= 0 for '{label}'. Plotting up to that point.")
            cumulative_product = cumulative_product[cumulative_product > 1e-8]
            if cumulative_product.empty: continue
        portfolio_value = cumulative_product * initial_capital
        style_key = 'default'
        if 'Kalman Filter' in label: style_key = 'Kalman Filter'
        elif 'Static OLS' in label: style_key = 'Static OLS'
        linestyle, linewidth = styles.get(style_key, styles['default'])
        ax.plot(portfolio_value.index, portfolio_value, label=label, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
        has_data_to_plot = True

    if not has_data_to_plot: print("No valid data series to plot IS cumulative returns."); plt.close(fig); return

    ax.set_ylabel("Portfolio Value (Log Scale)"); ax.set_xlabel("Date"); ax.set_yscale('log'); ax.legend(loc='best'); ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    filename = _sanitize_filename(f"IS_cumulative_returns_pair_{pair_info.get('ticker1','')}_{pair_info.get('ticker2','')}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")


def plot_out_of_sample_cumulative_returns(out_of_sample_returns_dict, initial_capital, split_date, pair_info, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    half_life = pair_info.get('half_life_days', np.nan)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(f'OOS Cumulative Performance (Log Scale) - from {split_date.date()} (IS HL: {half_life:.1f}d)', fontsize=14)

    has_data_to_plot = False
    styles = {'Kalman Filter': ('-', 2.0), 'Static OLS': ('--', 1.5), 'default': (':', 1.5)}

    for label, returns_series in out_of_sample_returns_dict.items():
        if returns_series.empty or returns_series.isnull().all(): continue
        cumulative_product = (1 + returns_series).cumprod()
        if (cumulative_product <= 1e-8).any():
            print(f"Warning OOS: Cumulative value <= 0 for '{label}'. Plotting up to that point.")
            cumulative_product = cumulative_product[cumulative_product > 1e-8]
            if cumulative_product.empty: continue
        portfolio_value = cumulative_product * initial_capital
        style_key = 'default'
        if 'Kalman Filter' in label: style_key = 'Kalman Filter'
        elif 'Static OLS' in label: style_key = 'Static OLS'
        linestyle, linewidth = styles.get(style_key, styles['default'])
        ax.plot(portfolio_value.index, portfolio_value, label=label, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
        has_data_to_plot = True

    if not has_data_to_plot: print("No valid data series to plot OOS cumulative returns."); plt.close(fig); return

    ax.set_ylabel("Portfolio Value (Log Scale)"); ax.set_xlabel("Date"); ax.set_yscale('log'); ax.legend(loc='best'); ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    filename = _sanitize_filename(f"OOS_cumulative_returns_pair_{pair_info.get('ticker1','')}_{pair_info.get('ticker2','')}.png")
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {filepath}")