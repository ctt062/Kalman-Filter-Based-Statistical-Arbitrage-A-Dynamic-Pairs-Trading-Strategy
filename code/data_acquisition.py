import pandas as pd
import yfinance as yf

def fetch_data(tickers, start_date, end_date):
    """
    Fetches closing prices for tickers from Yahoo Finance, fills missing values.
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