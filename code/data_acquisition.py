import pandas as pd
import yfinance as yf

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical 'Close' prices for a list of stock tickers from Yahoo Finance
    for a specified date range.

    The function attempts to download data for all provided tickers. It then
    handles potential issues such as:
    -   Single ticker downloads (which yfinance might return as a Series).
    -   MultiIndex columns (common when yfinance groups by 'Adj Close', 'Close', etc.).
    -   Missing data points by performing a forward fill (ffill) followed by a
        backward fill (bfill).
    -   Tickers for which no data could be retrieved or that still contain NaNs
        after filling (these tickers/columns are dropped).

    It prints progress, warnings, and error messages to the console.

    Parameters
    ----------
    tickers : list[str]
        A list of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOG']).
    start_date : str
        The start date for data retrieval, formatted as 'YYYY-MM-DD'.
    end_date : str
        The end date for data retrieval, formatted as 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame or None
        A pandas DataFrame where:
        -   The index is a DatetimeIndex representing trading days.
        -   Columns are the ticker symbols for which data was successfully
            retrieved and cleaned.
        -   Values are the closing prices.
        Returns None if a critical error occurs during fetching or if no data
        can be retrieved for any of the requested tickers.
    """
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']

        # Handle single ticker download (returns Series)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        # Handle yfinance returning extra columns sometimes for single ticker
        elif isinstance(data, pd.DataFrame) and len(tickers) == 1 and tickers[0] in data.columns:
             data = data[[tickers[0]]]

        # Clean column names if MultiIndex resulted
        if isinstance(data.columns, pd.MultiIndex):
             data.columns = data.columns.get_level_values(-1)

        # Fill missing values: forward first, then backward
        data = data.ffill().bfill()

        # Drop any columns that *still* have NaNs after filling
        if data.isnull().any().any():
             nan_tickers = data.columns[data.isnull().any()].tolist()
             print(f"Warning: NaNs remain after ffill/bfill for {nan_tickers}. Dropping.")
             data.dropna(axis=1, inplace=True)
             if data.empty:
                  print("Error: DataFrame empty after dropping NaN columns.")
                  return None

        # Return data only for requested tickers that are still present
        final_tickers = [t for t in tickers if t in data.columns]
        if len(final_tickers) < len(tickers):
             dropped_tickers = list(set(tickers)-set(final_tickers))
             print(f"Warning: Could not retrieve full data for: {dropped_tickers}")
        
        if not final_tickers:
            print("Error: No data retrieved for any ticker.")
            return None

        print(f"Data fetching complete. Using {len(final_tickers)} tickers.")
        return data[final_tickers]

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None