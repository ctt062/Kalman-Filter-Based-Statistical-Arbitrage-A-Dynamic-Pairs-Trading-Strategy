import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter

def calculate_static_ols(y_in_sample, x_in_sample):
    """
    Calculates static Ordinary Least Squares (OLS) regression coefficients (beta and alpha)
    for a pair of time series using in-sample data.

    The regression model is: y = alpha + beta * x + epsilon.

    Parameters
    ----------
    y_in_sample : pd.Series
        The dependent variable time series (e.g., prices of asset Y) for the in-sample period.
    x_in_sample : pd.Series
        The independent variable time series (e.g., prices of asset X) for the in-sample period.

    Returns
    -------
    static_beta : float
        The estimated slope coefficient (beta) from the OLS regression. Returns `np.nan` on failure.
    static_alpha : float
        The estimated intercept coefficient (alpha) from the OLS regression. Returns `np.nan` on failure.
    ols_residuals : pd.Series or None
        The residuals (y_observed - y_predicted) from the OLS regression. Returns None on failure.
    """
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
    """
    Initializes and "trains" a Kalman Filter for estimating dynamic hedge ratios (beta)
    and intercepts (alpha) for a pairs trading strategy.

    The state vector is [beta, alpha]. The observation equation is y_t = beta_t * x_t + alpha_t + v_t,
    where v_t is observation noise. The state transition is assumed to be a random walk.

    The function sets up the initial structure of the Kalman Filter (transition matrices,
    initial state guesses, covariances) and then uses the Expectation-Maximization (EM)
    algorithm on the in-sample data (y_in_sample, x_in_sample) to refine the
    observation covariance (R), transition covariance (Q), and initial state mean/covariance.

    Parameters
    ----------
    y_in_sample : pd.Series
        The dependent variable time series (asset Y prices) for the in-sample period.
    x_in_sample : pd.Series
        The independent variable time series (asset X prices) for the in-sample period.
    kf_delta : float
        A parameter used to set the initial state transition covariance matrix Q.
        Specifically, Q is initialized as `(kf_delta / (1 - kf_delta)) * np.eye(2)`.
        A smaller delta implies more confidence in the previous state (slower adaptation).

    Returns
    -------
    pykalman.KalmanFilter or None
        A `KalmanFilter` object from the `pykalman` library, with parameters
        (observation_covariance, transition_covariance, initial_state_mean,
        initial_state_covariance) estimated by the EM algorithm.
        Returns the initially structured (but un-trained) KF object if EM fails,
        or None if critical errors occur during setup (e.g., empty data).

    Notes
    -----
    - The observation matrix H_t for the EM algorithm is constructed as `[x_t, 1]`.
    - The state transition matrix F is assumed to be the identity matrix (random walk for states).
    - The EM algorithm iterates up to 10 times to estimate parameters.
    """
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
    """
    Runs the Kalman Filter *filtering* step on new data (in-sample or out-of-sample)
    using a pre-initialized/trained Kalman Filter object.

    This function takes an existing `KalmanFilter` object (presumably initialized
    by `initialize_kalman_filter`) and applies it to the provided `y_data` and
    `x_data` to obtain filtered estimates of the state vector [beta, alpha]
    at each time step.

    Parameters
    ----------
    kf_object : pykalman.KalmanFilter
        The initialized/trained Kalman Filter object. Its internal parameters
        (like Q, R, initial state for the *start* of this data segment) are used.
    y_data : pd.Series
        The dependent variable time series (observations) for the period to be filtered.
    x_data : pd.Series
        The independent variable time series, used to construct the time-varying
        observation matrix H_t = `[x_data_t, 1]`.

    Returns
    -------
    beta_filtered : pd.Series
        A time series of the filtered estimates for the hedge ratio (beta).
        Indexed same as `y_data`. Returns an empty Series on failure.
    alpha_filtered : pd.Series
        A time series of the filtered estimates for the intercept (alpha).
        Indexed same as `y_data`. Returns an empty Series on failure.

    Notes
    -----
    - The `kf_object.observation_matrices` attribute is dynamically updated within
      this function based on `x_data` before running the `filter` method.
    - The function uses `kf_object.filter()`, which performs the forward pass
      of the Kalman Filter equations.
    """
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