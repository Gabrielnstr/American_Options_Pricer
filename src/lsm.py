"""Longstaff-Schwartz Monte Carlo pricer for American options"""

import numpy as np
from src.paths import simulate_gbm

def payoff(S, K, option="call"):
    """Intrinsic value of a call or a put."""
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option must be 'call' or 'put'")

def basis_matrix(S, K, degree):
    """Polynomial basis [1, x, x^2, ..., x^degree] with x = S / K."""
    return np.vander(S / K, degree + 1, increasing=True)

def continuation_value(S, cashflows, K, degree, mask):
    """Estimate E[cashflow | S] by least-squares regression.
    
    Parameters
    ----------
    S : spot at the current date, shape (n_paths,)
    cashflows : future discounted cashflows, shape (n_paths,)
    K : strike (used to normalise the regressor)
    degree : degree of the regression polynomial
    mask : boolean array, paths used in the regression

    Returns
    -------
    cont : array of shape (n_paths,), fitted continuation value on the paths selected by 'mask', NaN elsewhere (and everywhere if there are not enough points to fit the polynomial).
    """

    cont = np.full(S.shape, np.nan)
    if mask.sum() <= degree + 1:
        return cont

    X = basis_matrix(S[mask], K, degree)
    beta, *_ = np.linalg.lstsq(X, cashflows[mask], rcond=None)
    cont[mask] = X @ beta
    return cont

def lsm_price(S0, K, r, q, sigma, T, n_steps, n_paths, rng, option="put", degree=2, itm_only=True):
    """Price an American option (Bermudan with n_steps dates) by Longstaff-Schwartz.
    
    Parameters
    ----------
    S0, K, r, q, sigma, T : usual Black-Scholes-Merton parameters
    n_steps : number of exercise dates (time steps)
    n_paths : number of Monte Carlo paths
    rng : random generator exposing .normal(n), e.g. src.rng.LCG
    option : "call" or "put"
    degree : degree of the regression polynomial
    itm_only : if True, regress only on in-the-money paths; if False, regress on all paths

    Returns
    -------
    dict with price, std_error, ci_low, ci_high
    """

    dt = T / n_steps
    disc = np.exp(-r * dt)

    paths = simulate_gbm(S0, r, q, sigma, T, n_steps, n_paths, rng)

    # Cash flows at maturity
    cashflow = payoff(paths[:, -1], K, option)

    # Backward induction over the exercise dates t_{n-1}; ..., t_1
    for j in range(n_steps -1, 0, -1):
        cashflow = cashflow * disc # value at date t_j
        S = paths[:, j]
        exercise = payoff(S, K, option)

        mask = exercise > 0 if itm_only else np.ones(n_paths, dtype=bool)
        cont = continuation_value(S, cashflow, K, degree, mask)

        exercise_now = (exercise > 0) & (exercise > cont) # NaN -> False
        cashflow[exercise_now] = exercise[exercise_now]

    cashflow = cashflow * disc # value at date t_0

    price = max(cashflow.mean(), payoff(S0, K, option))
    se = cashflow.std(ddof=1) / np.sqrt(n_paths)

    return {
        "price": price,
        "std_error": se,
        "ci_low": price - 1.96 * se,
        "ci_high": price + 1.96 * se,
    }
