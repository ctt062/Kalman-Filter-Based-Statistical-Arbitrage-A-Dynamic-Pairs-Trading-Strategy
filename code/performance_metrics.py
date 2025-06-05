import numpy as np
import pandas as pd

def calculate_metrics(portfolio_df, initial_capital, risk_free_rate=0.0):
    """
    Calculates standard performance metrics for a backtested trading strategy.

    The function takes a portfolio DataFrame, initial capital, and risk-free rate
    to compute metrics such as Total Return, Compound Annual Growth Rate (CAGR),
    Annualized Volatility, Sharpe Ratio, Maximum Drawdown, Calmar Ratio, and
    the Number of Trades.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        A pandas DataFrame representing the portfolio's performance over time.
        It must contain:
        -   A 'total' column: Time series of the total portfolio value.
        -   A DatetimeIndex.
        It should ideally also contain:
        -   A 'returns' column: Daily percentage returns of the portfolio. If not
            present, it will be calculated from the 'total' column.
        -   A 'trades' column: Indicator (e.g., 1.0) on days when a trade occurred.
            Used to count the total number of trades. If absent, number of trades
            will be 0.
    initial_capital : float
        The starting capital of the backtest.
    risk_free_rate : float, optional
        The annualized risk-free rate used for calculating the Sharpe Ratio.
        Defaults to 0.0.

    Returns
    -------
    dict[str, float | int]
        A dictionary where keys are metric names (strings) and values are the
        calculated metric values (float or int). The metrics included are:
        -   "Total Return": The total percentage return over the entire period.
        -   "CAGR": Compound Annual Growth Rate.
        -   "Annualized Volatility": Standard deviation of daily returns, annualized (sqrt(252)).
        -   "Sharpe Ratio": (CAGR - risk_free_rate) / Annualized Volatility.
        -   "Max Drawdown": The largest peak-to-trough decline during a specific period.
        -   "Calmar Ratio": CAGR / abs(Max Drawdown).
        -   "Number of Trades": Total count of trades executed.

    Notes
    -----
    - Assumes 252 trading days in a year for annualization.
    - CAGR calculation uses the actual number of days and converts to years.
    - Handles edge cases such as empty DataFrames, very short periods, zero volatility,
      or zero drawdown to prevent division by zero and return sensible values (e.g., 0.0 or np.inf).
    - If `initial_capital` or `final_value` is non-positive, CAGR and Total Return
      are set to 0.0.
    """
    # Ensure 'returns' column exists
    if 'returns' not in portfolio_df.columns:
        portfolio = portfolio_df.copy()
        if 'total' in portfolio.columns and not portfolio.empty and len(portfolio) > 1:
            portfolio['returns'] = portfolio['total'].pct_change().fillna(0)
        else: # Handle cases where calculation isn't possible
             portfolio['returns'] = pd.Series(dtype=float) # Empty returns series
    else:
        portfolio = portfolio_df

    # Handle empty or invalid portfolio DataFrame
    if portfolio.empty or 'total' not in portfolio.columns or portfolio['total'].isnull().all() or len(portfolio) < 2:
        return {"Total Return": 0.0, "CAGR": 0.0, "Annualized Volatility": 0.0,
                "Sharpe Ratio": 0.0, "Max Drawdown": 0.0, "Calmar Ratio": 0.0,
                "Number of Trades": 0}

    returns = portfolio['returns']
    final_value = portfolio['total'].iloc[-1]

    # Total Return
    total_return = (final_value / initial_capital) - 1 if initial_capital > 0 else 0.0

    # Annualized Volatility
    ann_volatility = returns.std() * np.sqrt(252) if returns.std() > 1e-9 else 0.0

    # Annualized Return (Arithmetic)
    # ann_return = returns.mean() * 252 # Original based on arithmetic mean

    # CAGR - Geometric Mean is more appropriate for multi-period returns
    start_date, end_date = portfolio.index[0], portfolio.index[-1]
    days = (end_date - start_date).days
    if days <= 0 or initial_capital <= 0 or final_value <= 0:
        cagr = 0.0
    else:
        years = days / 365.25
        if years < 1/252: # if less than one day, CAGR is not meaningful
             cagr = total_return # Use total return for very short periods
        else:
             cagr = (final_value / initial_capital) ** (1 / years) - 1

    # Sharpe Ratio (using CAGR as the expected return)
    sharpe_ratio = (cagr - risk_free_rate) / ann_volatility if ann_volatility > 1e-9 else 0.0
    if ann_volatility <= 1e-9 and (cagr - risk_free_rate) > 0: # Handle zero volatility with positive excess return
        sharpe_ratio = np.inf
    elif ann_volatility <= 1e-9 and (cagr - risk_free_rate) <= 0:
        sharpe_ratio = 0.0


    # Max Drawdown
    rolling_max = portfolio['total'].cummax()
    daily_drawdown = portfolio['total'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min() if not daily_drawdown.empty else 0.0

    # Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0
    if abs(max_drawdown) < 1e-9 and cagr > 0: # Handle zero drawdown with positive CAGR
        calmar_ratio = np.inf
    elif abs(max_drawdown) < 1e-9 and cagr <=0:
        calmar_ratio = 0.0


    # Number of Trades
    num_trades = int(portfolio['trades'].sum()) if 'trades' in portfolio.columns else 0

    metrics = {"Total Return": total_return, "CAGR": cagr, "Annualized Volatility": ann_volatility,
               "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown, "Calmar Ratio": calmar_ratio,
               "Number of Trades": num_trades}
    return metrics