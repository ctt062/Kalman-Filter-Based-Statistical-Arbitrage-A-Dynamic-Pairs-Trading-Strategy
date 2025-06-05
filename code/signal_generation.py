import numpy as np
import pandas as pd

def generate_signals(y, x, beta, alpha, z_score_window,
                     entry_threshold_z, exit_threshold_z,
                     use_dynamic_thresholds=False,
                     dynamic_z_window_factor=1.5,
                     min_z_std_dev = 0.1,
                     strategy_name="KF"):
    """
    Generates trading signals for a pairs trading strategy based on the
    rolling Z-score of the spread.

    The process involves:
    1.  Calculating the spread: `spread = y - (beta * x + alpha)`.
        `beta` and `alpha` can be static (float) or dynamic (pd.Series).
    2.  Calculating the rolling Z-score of the spread:
        `z_score = (spread - rolling_mean(spread)) / rolling_std(spread)`.
    3.  Determining entry/exit thresholds:
        -   Fixed: `entry_threshold_z` and `exit_threshold_z` are used directly.
        -   Dynamic: Thresholds are calculated as `multiplier * rolling_std(z_score)`.
            The `rolling_std(z_score)` is itself calculated over a window derived
            from `z_score_window * dynamic_z_window_factor`, and clipped at
            `min_z_std_dev`.
    4.  Generating positions (1 for long spread, -1 for short spread, 0 for flat)
        based on whether the Z-score crosses these thresholds.

    Parameters
    ----------
    y : pd.Series
        Time series of the dependent asset's prices (e.g., asset Y).
    x : pd.Series
        Time series of the independent asset's prices (e.g., asset X).
    beta : float or pd.Series
        The hedge ratio. Can be a single float (static) or a time series (dynamic).
    alpha : float or pd.Series
        The intercept. Can be a single float (static) or a time series (dynamic).
    z_score_window : int
        The rolling window size (number of periods) for calculating the mean
        and standard deviation of the spread to compute its Z-score.
    entry_threshold_z : float
        The Z-score value (or multiplier for dynamic thresholds) beyond which
        a position is initiated. E.g., if Z < -entry_threshold, go long spread.
    exit_threshold_z : float
        The Z-score value (or multiplier for dynamic thresholds) towards which
        the Z-score must revert for an existing position to be closed.
        E.g., if long spread and Z >= -exit_threshold, exit.
    use_dynamic_thresholds : bool, optional
        If True, entry/exit thresholds are dynamic based on the rolling standard
        deviation of the Z-score itself. Defaults to False (fixed thresholds).
    dynamic_z_window_factor : float, optional
        Factor to multiply `z_score_window` by to get the window for calculating
        the rolling standard deviation of the Z-score (used for dynamic thresholds).
        Defaults to 1.5.
    min_z_std_dev : float, optional
        Minimum standard deviation for the Z-score when calculating dynamic thresholds,
        to prevent overly sensitive thresholds during flat Z-score periods. Defaults to 0.1.
    strategy_name : str, optional
        A name for the strategy, used in print statements for clarity. Defaults to "KF".

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by date, containing:
        -   'price_y', 'price_x': Original (aligned) prices.
        -   'beta', 'alpha': The beta and alpha used (aligned).
        -   'spread': The calculated spread time series.
        -   'z_score': The Z-score of the spread.
        -   'entry_z_upper', 'entry_z_lower', 'exit_z_upper', 'exit_z_lower':
            The threshold values used for trading decisions.
        -   'positions': The target position for each day (1, -1, or 0).
        -   'signal': The change in position from the previous day (trade signal).
        Returns an empty DataFrame if critical errors occur or if insufficient
        data is available.

    Notes
    -----
    -   Handles alignment of y, x, beta, and alpha if beta/alpha are dynamic.
    -   Ensures minimum periods for rolling calculations to avoid NaNs from insufficient data.
    -   Handles zero standard deviation in Z-score calculation robustly.
    -   The trading logic is:
        -   From Flat: Enter long spread if Z < lower_entry_threshold.
                      Enter short spread if Z > upper_entry_threshold.
        -   From Long Spread: Exit to flat if Z >= lower_exit_threshold.
        -   From Short Spread: Exit to flat if Z <= upper_exit_threshold.
    """
    print(f"Generating signals ({strategy_name}, DynThr: {use_dynamic_thresholds}): {y.index.min().date()} to {y.index.max().date()}...")
    if y.empty or x.empty: return pd.DataFrame()

    # --- Calculate Spread ---
    if isinstance(beta, (int, float)) and isinstance(alpha, (int, float)):
        spread = y - (beta * x + alpha)
        aligned_y, aligned_x = y, x # Already aligned if static
    elif isinstance(beta, pd.Series) and isinstance(alpha, pd.Series):
        # Align all series for dynamic calculation
        aligned = pd.concat([y.rename('y'), x.rename('x'), beta, alpha], axis=1, join='inner')
        if aligned.empty: print(f"Error ({strategy_name}): Data alignment failed."); return pd.DataFrame()
        aligned_y, aligned_x = aligned['y'], aligned['x']
        beta_aligned, alpha_aligned = aligned['beta'], aligned['alpha'] # Use aligned versions
        spread = aligned_y - (beta_aligned * aligned_x + alpha_aligned)
    else:
        print(f"Error ({strategy_name}): Invalid type for beta/alpha."); return pd.DataFrame()

    # --- Calculate Rolling Z-score ---
    effective_window = min(z_score_window, len(spread))
    min_periods_z = max(5, effective_window // 2) # Require reasonable min periods
    if effective_window < min_periods_z:
        print(f"Warning ({strategy_name}): Not enough data ({len(spread)}) for Z-window {z_score_window}."); return pd.DataFrame()

    spread_mean = spread.rolling(window=effective_window, min_periods=min_periods_z).mean()
    spread_std = spread.rolling(window=effective_window, min_periods=min_periods_z).std()
    # Handle zero standard deviation robustly
    spread_std = spread_std.replace(0, np.nan).ffill().bfill().fillna(1e-6)
    z_score = (spread - spread_mean) / spread_std
    z_score.dropna(inplace=True) # Remove initial NaNs
    if z_score.empty:
        print(f"Warning ({strategy_name}): Z-score calculation resulted in empty series."); return pd.DataFrame()

    # --- Assemble Base Signal DataFrame ---
    # Reindex original data to match the valid z_score index
    signals_df = pd.DataFrame(index=z_score.index)
    signals_df['price_y'] = aligned_y.reindex(z_score.index)
    signals_df['price_x'] = aligned_x.reindex(z_score.index)
    signals_df['beta'] = beta.reindex(z_score.index) if isinstance(beta, pd.Series) else beta
    signals_df['alpha'] = alpha.reindex(z_score.index) if isinstance(alpha, pd.Series) else alpha
    signals_df['spread'] = spread.reindex(z_score.index)
    signals_df['z_score'] = z_score

    # --- Calculate Thresholds (Fixed or Dynamic) ---
    if use_dynamic_thresholds:
        dynamic_z_window = max(min_periods_z, int(z_score_window * dynamic_z_window_factor))
        rolling_z_std = z_score.rolling(window=dynamic_z_window, min_periods=min_periods_z).std()
        # Apply minimum std dev threshold and fill NaNs
        rolling_z_std = rolling_z_std.clip(lower=min_z_std_dev).ffill().bfill().fillna(min_z_std_dev)
        signals_df['entry_z_upper'] = rolling_z_std * entry_threshold_z
        signals_df['entry_z_lower'] = -rolling_z_std * entry_threshold_z
        signals_df['exit_z_upper'] = rolling_z_std * exit_threshold_z
        signals_df['exit_z_lower'] = -rolling_z_std * exit_threshold_z
        # print(f"  Using dynamic Z-thresholds (window={dynamic_z_window}, multiplier={entry_threshold_z:.1f}/{exit_threshold_z:.1f})") # Debug
    else:
        signals_df['entry_z_upper'] = entry_threshold_z
        signals_df['entry_z_lower'] = -entry_threshold_z
        signals_df['exit_z_upper'] = exit_threshold_z
        signals_df['exit_z_lower'] = -exit_threshold_z
        # print(f"  Using fixed Z-thresholds (Entry={entry_threshold_z:.1f}, Exit={exit_threshold_z:.1f})") # Debug

    # --- Generate Positions (Trading Logic) ---
    current_position = 0
    positions = np.zeros(len(signals_df))
    # Use .values for faster loop access
    z_values = signals_df['z_score'].values
    entry_upper = signals_df['entry_z_upper'].values; entry_lower = signals_df['entry_z_lower'].values
    exit_upper = signals_df['exit_z_upper'].values;   exit_lower = signals_df['exit_z_lower'].values

    for i in range(len(signals_df)):
        z = z_values[i]
        entry_u, entry_l = entry_upper[i], entry_lower[i]
        exit_u, exit_l = exit_upper[i], exit_lower[i]

        if pd.isna(z): # Skip if Z-score is invalid
            positions[i] = current_position; continue

        # State transitions based on Z-score and thresholds
        if current_position == 0:
            if z < entry_l: current_position = 1  # Enter Long Spread
            elif z > entry_u: current_position = -1 # Enter Short Spread
        elif current_position == 1: # Currently Long Spread
            if z >= exit_l: current_position = 0 # Exit Long towards mean
        elif current_position == -1: # Currently Short Spread
            if z <= exit_u: current_position = 0 # Exit Short towards mean

        positions[i] = current_position

    signals_df['positions'] = positions
    signals_df['signal'] = signals_df['positions'].diff().fillna(0) # Signal: change in position

    num_trades = len(signals_df[signals_df['signal'] != 0])
    print(f"  Generated {num_trades} trade signals for {strategy_name}.")
    return signals_df