import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

def calculate_half_life(spread_series):
    """
    Calculates the Half-Life of mean reversion for a spread series using
    the Ornstein-Uhlenbeck formula derived from an AR(1) process.
    """
    if spread_series.isnull().any() or len(spread_series) < 10:
        return np.inf # Not enough data or contains NaNs

    try:
        # Create lagged spread and difference
        spread_lagged = spread_series.shift(1)
        spread_diff = spread_series.diff()

        # Drop NaNs introduced by shifting/differencing
        spread_diff = spread_diff.dropna()
        spread_lagged = spread_lagged.reindex(spread_diff.index).dropna() # Align and drop potential NaNs

        # Ensure enough data after alignment
        if len(spread_lagged) < 10:
            return np.inf

        # Regress spread_diff on spread_lagged: dS = lambda * S(t-1) + error
        X = sm.add_constant(spread_lagged, prepend=True) # Ensure const is first column
        model = sm.OLS(spread_diff, X).fit()

        # Extract lambda (coefficient of the lagged spread)
        # Use .get() for robustness in case column name isn't exactly the series name
        lambda_coeff = model.params.get(spread_lagged.name, model.params.iloc[1])

        # Calculate half-life
        if lambda_coeff < 0:
            half_life = -np.log(2) / lambda_coeff
            return half_life
        else:
            return np.inf # Not mean-reverting or random walk

    except Exception:
        # print(f"Half-life calculation failed: {e}") # Optional debug
        return np.inf


def find_cointegrated_pair(data, significance_level=0.05,
                           min_half_life=5, max_half_life=100,
                           enable_half_life_filter=False):
    """
    Finds the best cointegrated pair using Engle-Granger & ADF tests,
    calculates Half-Life, and optionally filters by Half-Life.
    """
    n = data.shape[1]
    if n < 2:
        print("Need at least two series for cointegration testing.")
        return None, None, None

    keys = data.columns
    pairs_found = []
    num_pairs_to_check = n * (n - 1) // 2
    print(f"Checking {num_pairs_to_check} pairs for cointegration (EG + ADF)...")
    if enable_half_life_filter:
        print(f"  Applying Half-Life Filter ({min_half_life}-{max_half_life} days)")

    # Iterate through unique pairs
    for i in range(n):
        for j in range(i + 1, n):
            ticker1_name, ticker2_name = keys[i], keys[j]
            series1, series2 = data[ticker1_name], data[ticker2_name]

            # Ensure sufficient overlapping data
            combined = pd.concat([series1, series2], axis=1).dropna()
            if combined.shape[0] < 50:  # Need minimum data points
                continue

            try:
                # 1. Engle-Granger Test
                coint_score, coint_pvalue, coint_crit_values = coint(combined.iloc[:, 0], combined.iloc[:, 1], trend='c')

                # Check EG significance (p-value and critical value)
                if coint_pvalue < significance_level and coint_score < coint_crit_values[1]:  # Compare t-stat to 5% critical value

                    # 2. Calculate Static OLS Spread (for ADF and Half-Life)
                    y_ols = combined.iloc[:, 0]
                    x_ols_with_const = sm.add_constant(combined.iloc[:, 1])
                    ols_model = sm.OLS(y_ols, x_ols_with_const).fit()
                    ols_spread = ols_model.resid

                    # 3. ADF Test on the OLS Spread
                    adf_result = adfuller(ols_spread)
                    adf_pvalue = adf_result[1]

                    # Check ADF significance
                    if adf_pvalue < significance_level:
                        # 4. Calculate Half-Life
                        half_life = calculate_half_life(ols_spread)

                        # 5. Optional Half-Life Filter
                        if enable_half_life_filter and not (min_half_life <= half_life <= max_half_life):
                            continue  # Skip pair if half-life is outside the desired range

                        # Store results if all checks passed
                        pairs_found.append({
                            'ticker1': ticker1_name, 'ticker2': ticker2_name,
                            'coint_p_value': coint_pvalue, 'coint_t_stat': coint_score, 'coint_crit_values': coint_crit_values,
                            'ols_beta': ols_model.params.iloc[1], 'ols_alpha': ols_model.params.iloc[0], # Assuming const is first
                            'adf_p_value': adf_pvalue, 'adf_stat': adf_result[0],
                            'half_life_days': half_life
                        })
            except Exception as e:
                # print(f"\nTest failed for {ticker1_name}/{ticker2_name}: {e}") # Optional
                continue

    print("\nPair checking complete.")

    if not pairs_found:
        filter_msg = f"and Half-Life filter ({min_half_life}-{max_half_life} days)" if enable_half_life_filter else ""
        print(f"No pairs found satisfying EG (p<{significance_level}, t<5%CV), ADF (p<{significance_level}) {filter_msg}.")
        return None, None, None

    # Select the best pair (lowest coint p-value among valid ones)
    best_pair_info = min(pairs_found, key=lambda x: x['coint_p_value'])
    t1, t2 = best_pair_info['ticker1'], best_pair_info['ticker2']

    print(f"\nSelected Best Pair: {t1} / {t2}")
    print(f"  EG P-value: {best_pair_info['coint_p_value']:.4f} (T-stat: {best_pair_info['coint_t_stat']:.4f}, 5% Crit: {best_pair_info['coint_crit_values'][1]:.4f})")
    print(f"  Spread ADF P-value: {best_pair_info['adf_p_value']:.4f}")
    print(f"  Spread Half-Life: {best_pair_info['half_life_days']:.2f} days")
    print(f"  In-Sample OLS Beta: {best_pair_info['ols_beta']:.4f}, Alpha: {best_pair_info['ols_alpha']:.4f}")
    print("\n!! Economic Rationale required: Justify this pair's potential cointegration. !!")

    return t1, t2, best_pair_info