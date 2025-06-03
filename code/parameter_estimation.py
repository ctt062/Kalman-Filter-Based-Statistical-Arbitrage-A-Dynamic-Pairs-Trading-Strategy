import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter

def calculate_static_ols(y_in_sample, x_in_sample):
    """ Calculates static OLS regression coefficients (beta, alpha) on in-sample data. """
    print("Calculating Static OLS parameters (In-Sample)...")
    if y_in_sample.empty or x_in_sample.empty:
        print("Error: Empty data provided to calculate_static_ols.")
        return np.nan, np.nan, None
    try:
        x_with_const = sm.add_constant(x_in_sample, prepend=True)
        ols_model = sm.OLS(y_in_sample, x_with_const).fit()

        # Robustly get parameters by name (handle potential column order issues)
        x_col_name = x_in_sample.name if x_in_sample.name in x_with_const.columns else x_with_const.columns[1] # Fallback if name mismatch
        static_beta = ols_model.params.get(x_col_name, np.nan)
        static_alpha = ols_model.params.get('const', np.nan)

        if pd.isna(static_beta) or pd.isna(static_alpha):
            print("Error: Could not extract OLS parameters.")
            return np.nan, np.nan, None

        print(f"  Static OLS Beta: {static_beta:.4f}, Alpha: {static_alpha:.4f}")
        return static_beta, static_alpha, ols_model.resid

    except Exception as e:
        print(f"Static OLS calculation failed: {e}")
        return np.nan, np.nan, None


def initialize_kalman_filter(y_in_sample, x_in_sample, kf_delta):
    """ Initializes or trains Kalman Filter parameters using the EM algorithm on in-sample data. """
    print("Initializing Kalman Filter parameters (In-Sample)...")
    if y_in_sample.empty or x_in_sample.empty:
        print("Error: Empty data provided to initialize_kalman_filter."); return None

    try:
        # Ensure x_in_sample has a name for observation matrix creation
        if x_in_sample.name is None: x_in_sample.name = 'X' # Assign default name if needed
        x_with_const = sm.add_constant(x_in_sample, prepend=True)
        # Observation matrix H = [X, 1] -> columns: [X_col, const]
        obs_mat_in_sample = np.expand_dims(x_with_const[[x_in_sample.name, 'const']].values, axis=1)

    except Exception as e:
        print(f"Error shaping observation matrix for KF init: {e}. Check input data."); return None

    # State transition covariance matrix Q = delta/(1-delta) * I
    trans_cov = kf_delta / (1 - kf_delta) * np.eye(2) # State dimension = 2 (beta, alpha)

    # Define initial KF structure
    kf = KalmanFilter(
        n_dim_obs=1,                # Observation dimension (y)
        n_dim_state=2,              # State dimension (beta, alpha)
        transition_matrices=np.eye(2), # Assume state follows random walk (F = Identity)
        initial_state_mean=np.zeros(2),     # Initial guess for state [beta, alpha]
        initial_state_covariance=np.ones((2, 2)), # Initial uncertainty
        observation_covariance=1.0,            # Initial guess for R (obs noise variance)
        transition_covariance=trans_cov        # Initial guess for Q (state noise variance)
    )

    print("Running EM algorithm to refine KF parameters...")
    try:
         # EM algorithm estimates parameters (covariances, initial state) from data
         # Observation matrix maps state [beta, alpha] to observation y: y = beta*X + alpha*1 = H * state
         kf_trained = kf.em(y_in_sample.values, observation_matrices=obs_mat_in_sample, n_iter=10)
         print("  EM estimation complete.")
         print(f"  Est. Obs Cov (R): {kf_trained.observation_covariance[0,0]:.4f}")
         print(f"  Est. Trans Cov (Q diag): [{kf_trained.transition_covariance[0,0]:.4e}, {kf_trained.transition_covariance[1,1]:.4e}]")
         print(f"  Est. Initial State [beta, alpha]: [{kf_trained.initial_state_mean[0]:.4f}, {kf_trained.initial_state_mean[1]:.4f}]")
         return kf_trained

    except Exception as e:
         print(f"EM algorithm failed: {e}. Using initial parameter guesses.")
         return kf # Return the non-trained KF object


def run_kalman_filter(kf_object, y_data, x_data):
    """ Runs the Kalman Filter *filtering* step on provided data. """
    print(f"Running KF filtering: {y_data.index.min().date()} to {y_data.index.max().date()}...")
    if y_data.empty or x_data.empty or kf_object is None:
        print("Error: Invalid input to run_kalman_filter."); return pd.Series(dtype=float), pd.Series(dtype=float)

    try:
        # Ensure x_data has a name
        if x_data.name is None: x_data.name = 'X'
        x_with_const = sm.add_constant(x_data, prepend=True)

        # Create time-varying observation matrices H_t = [x_t, 1]
        # Shape: (n_timesteps, n_dim_obs=1, n_dim_state=2)
        obs_mat = np.expand_dims(x_with_const[[x_data.name, 'const']].values, axis=1)

    except Exception as e:
        print(f"Error creating KF observation matrix: {e}"); return pd.Series(dtype=float), pd.Series(dtype=float)

    # Assign time-varying observation matrices to the KF object attribute
    kf_object.observation_matrices = obs_mat

    try:
        # Run the filtering step (uses kf_object.observation_matrices implicitly)
        state_means, _ = kf_object.filter(y_data.values) # We only need the filtered means

        # Extract filtered state estimates: state is [beta, alpha]
        beta_filtered = pd.Series(state_means[:, 0], index=y_data.index, name='beta')
        alpha_filtered = pd.Series(state_means[:, 1], index=y_data.index, name='alpha')

        print("  Kalman filtering complete.")
        return beta_filtered, alpha_filtered

    except Exception as e:
        print(f"Kalman filtering failed: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)