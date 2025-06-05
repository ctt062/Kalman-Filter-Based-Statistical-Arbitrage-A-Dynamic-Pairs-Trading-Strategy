import numpy as np
import pandas as pd
from .performance_metrics import calculate_metrics # Use . for relative import

def backtest_strategy(signals, initial_capital, fixed_cost_per_trade,
                      variable_cost_pct, slippage_pct,
                      stop_loss_z_threshold=None, risk_free_rate=0.0): # Added risk_free_rate
    """
    Backtests a pairs trading strategy based on generated signals, incorporating
    transaction costs, slippage, and an optional stop-loss mechanism.

    The strategy involves taking positions in two assets (Y and X) based on
    the deviation of their spread from its mean (z-score).

    Parameters
    ----------
    signals : pd.DataFrame
        DataFrame containing trading signals and market data. Expected columns:
        - 'positions': Target position for the pair (1 for long spread, -1 for short spread, 0 for flat).
        - 'price_y': Price of asset Y.
        - 'price_x': Price of asset X.
        - 'beta': Hedge ratio (dynamically or statically estimated).
        - 'z_score': The z-score of the spread, used for entry/exit and stop-loss.
        The DataFrame should be indexed by date.
    initial_capital : float
        The starting capital for the backtest.
    fixed_cost_per_trade : float
        Fixed transaction cost applied to each leg of a trade (e.g., $1 per trade per asset).
    variable_cost_pct : float
        Variable transaction cost as a percentage of the trade value for each leg
        (e.g., 0.001 for 0.1%).
    slippage_pct : float
        Slippage cost as a percentage of the execution price, applied to each leg.
        Positive slippage makes buy prices higher and sell prices lower.
    stop_loss_z_threshold : float, optional
        If provided, a stop-loss is triggered if the z-score moves adversely
        by this amount from the z-score at entry. For example, if entry Z was -2.0
        and `stop_loss_z_threshold` is 1.0, a stop is triggered if Z rises above -1.0.
        If None (default), no stop-loss is applied.
    risk_free_rate : float, optional
        Annualized risk-free rate used for calculating performance metrics like
        the Sharpe ratio. Defaults to 0.0.

    Returns
    -------
    portfolio : pd.DataFrame
        DataFrame tracking the portfolio's evolution over time. Columns include:
        - 'holdings': Value of current asset holdings.
        - 'cash': Cash balance.
        - 'total': Total portfolio value (holdings + cash).
        - 'positions': Actual position held in the pair (1, -1, or 0).
        - 'trades': Indicator (1.0) if a trade occurred on that day, else 0.0.
        - 'returns': Daily percentage returns of the total portfolio value.
    metrics : pd.Series
        A Series containing various performance metrics calculated from the
        backtest (e.g., CAGR, Sharpe Ratio, Max Drawdown).

    Methodology
    -----------
    1.  Initialization: Sets up the portfolio DataFrame with initial capital.
    2.  Iteration: Loops through each day in the `signals` DataFrame.
        a.  Carry Forward: Portfolio values (cash, holdings, total) are carried
            forward from the previous day.
        b.  Stop-Loss Check: If a stop-loss threshold is set and a position is open,
            it checks if the current z-score has moved adversely beyond the
            `stop_loss_z_threshold` from the z-score at the time of entry.
            If triggered, the target position for the day is set to 0 (flat).
        c.  Trade Execution: If the target position (from signals or stop-loss)
            differs from the previous day's position, a trade is executed:
            i.  Close Existing Position: If currently holding a position, it's
                closed. Cash is updated by the proceeds, and transaction costs
                (fixed + variable + slippage) for both legs are deducted.
            ii. Open New Position: If the target position is not flat, a new
                position is opened.
                - Capital Allocation: Half of the previous day's total portfolio
                  value is targeted for the Y-leg of the pair.
                - Shares Calculation: Shares for asset Y are calculated based on
                  its execution price (including slippage). Shares for asset X
                  are calculated to hedge asset Y using the provided `beta`
                  (shares_x = -beta * shares_y).
                - Cash Update: Cash is reduced by the net cost of opening the
                  position.
                - Transaction Costs: Costs (fixed + variable + slippage) for
                  both legs are deducted from cash.
                - Entry Z-score: The z-score at entry is recorded if not a
                  stop-loss induced trade.
        d.  Mark-to-Market: The 'holdings' value is updated using the current day's
            market prices for Y and X (without slippage). The 'total' portfolio
            value is updated.
        e.  Bankruptcy Check: If total portfolio value drops to or below zero,
            the backtest is terminated.
    3.  Final Metrics: After the loop, daily returns are calculated, and overall
        performance metrics are computed using `calculate_metrics`.
    """
    print(f"Backtesting: {signals.index.min().date()} to {signals.index.max().date()}...")
    if signals.empty or 'positions' not in signals.columns:
        print("Warning: Invalid signals DataFrame for backtesting.")
        empty_metrics = calculate_metrics(pd.DataFrame(), initial_capital, risk_free_rate)
        return pd.DataFrame(), empty_metrics

    # Initialize portfolio DataFrame
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['holdings'] = 0.0
    portfolio['cash'] = float(initial_capital)
    portfolio['total'] = float(initial_capital)
    portfolio['positions'] = signals['positions'] # Desired EOD position
    portfolio['trades'] = 0.0 # Mark days with trades
    # portfolio['z_score'] = signals['z_score'] # Keep z_score if needed for analysis

    # State variables for tracking position
    current_shares_y, current_shares_x = 0.0, 0.0
    z_at_entry = None # Store z-score when position is opened

    # Iterate through signals DataFrame (use index for clarity)
    for i in range(1, len(signals)):
        current_idx = signals.index[i]
        prev_idx = signals.index[i-1]

        # --- Carry forward values from previous day ---
        portfolio.loc[current_idx, ['holdings', 'cash', 'total']] = portfolio.loc[prev_idx, ['holdings', 'cash', 'total']]
        # Position state is taken directly from the signals row for the current day: portfolio.loc[current_idx, 'positions'] = signals.loc[current_idx, 'positions']

        # --- Get data for current day & previous state ---
        current_row = signals.iloc[i]
        prev_state = portfolio.loc[prev_idx, 'positions']
        current_state_from_signal = current_row['positions'] # Desired state for *end* of day idx
        
        # This will be the actual state after considering stop loss
        current_target_state = current_state_from_signal

        price_y = current_row.get('price_y'); price_x = current_row.get('price_x')
        beta = current_row.get('beta'); z_score = current_row.get('z_score')

        # Skip day if essential data is missing
        if pd.isna(price_y) or pd.isna(price_x) or pd.isna(beta):
            # Update holdings value based on previous shares and current prices if possible
            if not (pd.isna(price_y) or pd.isna(price_x)):
                 portfolio.loc[current_idx, 'holdings'] = (current_shares_y * price_y) + (current_shares_x * price_x)
                 portfolio.loc[current_idx, 'total'] = portfolio.loc[current_idx, 'cash'] + portfolio.loc[current_idx, 'holdings']
            continue # Skip trading logic

        # --- Stop Loss Check ---
        stop_loss_triggered = False
        if stop_loss_z_threshold is not None and prev_state != 0 and z_at_entry is not None and not pd.isna(z_score):
            # Stop loss is based on Z moving X points *further away* from zero (mean)
            # If long spread (z_at_entry < 0), stop if z_score > z_at_entry + SL_threshold
            # If short spread (z_at_entry > 0), stop if z_score < z_at_entry - SL_threshold
            if (prev_state == 1 and z_at_entry < 0 and z_score > z_at_entry + stop_loss_z_threshold) or \
               (prev_state == -1 and z_at_entry > 0 and z_score < z_at_entry - stop_loss_z_threshold):
                stop_loss_triggered = True
                current_target_state = 0 # Force exit
                # print(f"  {current_idx.date()}: Stop Loss triggered (Z:{z_score:.2f}, EntryZ:{z_at_entry:.2f}, State:{prev_state})") # Debug

        trade_signal = current_target_state - prev_state
        portfolio.loc[current_idx, 'positions'] = current_target_state # Update portfolio state for this day based on signal or SL


        # --- Execute Trades if signal or stop loss ---
        if trade_signal != 0:
            portfolio.loc[current_idx, 'trades'] = 1.0 # Mark a trade event

            # 1. Close Existing Position (if any)
            if prev_state != 0:
                # Apply slippage to closing prices (opposite direction of holding)
                close_price_y = price_y * (1 - np.sign(current_shares_y) * slippage_pct) if current_shares_y != 0 else price_y
                close_price_x = price_x * (1 - np.sign(current_shares_x) * slippage_pct) if current_shares_x != 0 else price_x
                
                # Calculate cash change from closing legs
                cash_change_close = (current_shares_y * close_price_y) + (current_shares_x * close_price_x)
                portfolio.loc[current_idx, 'cash'] += cash_change_close
                
                # Calculate transaction costs for closing (applied for each leg traded)
                cost_close = 0
                if current_shares_y != 0: cost_close += fixed_cost_per_trade + (abs(current_shares_y * close_price_y) * variable_cost_pct)
                if current_shares_x != 0: cost_close += fixed_cost_per_trade + (abs(current_shares_x * close_price_x) * variable_cost_pct)
                portfolio.loc[current_idx, 'cash'] -= cost_close
                
                # Reset position state
                current_shares_y, current_shares_x = 0.0, 0.0
                if not stop_loss_triggered: # Only reset z_at_entry if not SL, SL means entry was bad
                     z_at_entry = None # Reset Z entry point on normal exit

            # 2. Open New Position (if not exiting to flat)
            if current_target_state != 0:
                # Use capital *before* any closing trades on this day for sizing
                # Or, use total portfolio value from previous day to size new trade
                capital_available_for_trade = portfolio.loc[prev_idx, 'total'] 
                if capital_available_for_trade <= 0: continue # Skip if busted

                # For simplicity, let's assume we allocate total capital to the pair strategy
                # A more robust sizing would consider cash available after costs, etc.
                # Allocate 50% of capital to each leg (dollar neutral before costs/slippage if beta=1)
                target_dollar_value_y_leg = capital_available_for_trade / 2.0
                
                # Check for valid prices/beta before calculation
                if price_y <= 1e-6 or price_x <= 1e-6 or pd.isna(beta) or abs(beta) < 1e-6: continue

                if current_target_state == 1: # Enter Long Spread (Buy Y, Sell X)
                    exec_price_y = price_y * (1 + slippage_pct) # Buy Y higher
                    exec_price_x = price_x * (1 - slippage_pct) # Sell X lower
                    if exec_price_y <= 1e-6 or exec_price_x <= 1e-6: continue

                    current_shares_y = target_dollar_value_y_leg / exec_price_y
                    # X shares are such that value of X leg is beta * value of Y leg
                    # (shares_x * price_x) = -beta * (shares_y * price_y) -- but using exec_price_x for shares_x
                    # shares_x * exec_price_x = -beta * shares_y * exec_price_y (approximately, if beta changes slowly)
                    # For simplicity: shares_x = -beta * shares_y (hedge ratio based on shares)
                    current_shares_x = - (current_shares_y * beta) 

                elif current_target_state == -1: # Enter Short Spread (Sell Y, Buy X)
                    exec_price_y = price_y * (1 - slippage_pct) # Sell Y lower
                    exec_price_x = price_x * (1 + slippage_pct) # Buy X higher
                    if exec_price_y <= 1e-6 or exec_price_x <= 1e-6: continue
                    
                    current_shares_y = -target_dollar_value_y_leg / exec_price_y # Negative for short Y
                    current_shares_x = - (current_shares_y * beta) # Positive for long X (as current_shares_y is neg)

                # Calculate cash change from opening new legs
                cash_change_open = - (current_shares_y * exec_price_y) - (current_shares_x * exec_price_x)
                portfolio.loc[current_idx, 'cash'] += cash_change_open
                
                # Calculate transaction costs for opening
                cost_open = 0
                if current_shares_y != 0: cost_open += fixed_cost_per_trade + (abs(current_shares_y * exec_price_y) * variable_cost_pct)
                if current_shares_x != 0: cost_open += fixed_cost_per_trade + (abs(current_shares_x * exec_price_x) * variable_cost_pct)
                portfolio.loc[current_idx, 'cash'] -= cost_open
                
                if not stop_loss_triggered: # Only record entry Z if not immediately stopped out by the same day's data
                     z_at_entry = z_score # Record Z at entry point

        # --- Update Portfolio Value (Mark-to-Market at end of day) ---
        # Use current market prices (no slippage) for daily valuation
        holdings_value = (current_shares_y * price_y) + (current_shares_x * price_x)
        portfolio.loc[current_idx, 'holdings'] = holdings_value
        portfolio.loc[current_idx, 'total'] = portfolio.loc[current_idx, 'cash'] + holdings_value

        # --- Check for Bankruptcy ---
        if portfolio.loc[current_idx, 'total'] <= 0:
            print(f"Strategy Busted! Portfolio value <= 0 at {current_idx.date()}. Truncating.")
            portfolio = portfolio.loc[:current_idx] # Cut off history here
            break # Exit backtest loop

    # --- Final Calculations ---
    portfolio['returns'] = portfolio['total'].pct_change().fillna(0)
    metrics = calculate_metrics(portfolio.copy(), initial_capital, risk_free_rate) # Use final portfolio for metrics

    print(f"  Backtest complete. Final Value: ${portfolio['total'].iloc[-1]:,.2f}")
    return portfolio, metrics