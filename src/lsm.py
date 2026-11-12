"""Longstaff-Schwartz Monte Carlo pricer for American options"""

import numpy as np

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